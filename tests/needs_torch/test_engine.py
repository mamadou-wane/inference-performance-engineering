"""The cached engine on the CPU: bitwise parity with the uncached path and with Transformers'
cached decoding, the accuracy rule with seeded bugs and legitimate reorderings, the cache
contract, sampling, and what a step does on the device.

Not covered: GPU kernels and SDPA backend selection on CUDA, batches above 2.
"""

import dataclasses

import accuracy
import mpmath
import pytest
import torch
import torch.nn.functional as F
from test_forward import CONFIGS, load_reference, same_bits, tokens
from test_weights import WEIGHTS
from torch.utils._python_dispatch import TorchDispatchMode

import ipe.engine
from ipe import ops
from ipe.engine import Engine, sample
from ipe.forward import final_hidden, forward_full, lm_logits
from ipe.weights import load_checkpoint

F64 = torch.float64
RUN_DTYPES = (torch.bfloat16, torch.float32)
BACKENDS = {"math": "eager", "sdpa": "sdpa"}


def teacher_forced(engine, token_ids, prompt):
    """fp32 logits [B, T - prompt + 1, V] for positions prompt-1 to T-1, fed the given tokens."""
    logits = [engine.prefill(token_ids[:, :prompt])]
    logits += [engine.decode(token_ids[:, p]) for p in range(prompt, token_ids.shape[1])]
    return torch.stack(logits, dim=1)


@torch.no_grad()
def reference_teacher_forced(model, token_ids, prompt):
    out = model(input_ids=token_ids[:, :prompt], use_cache=True, logits_to_keep=1)
    logits, cache = [out.logits[:, -1]], out.past_key_values
    for p in range(prompt, token_ids.shape[1]):
        out = model(input_ids=token_ids[:, p:p + 1], past_key_values=cache, use_cache=True)
        logits.append(out.logits[:, -1])
        cache = out.past_key_values
    return torch.stack(logits, dim=1)


def load(checkpoints, config="gqa"):
    return load_checkpoint(checkpoints[config], dtype=torch.bfloat16, device="cpu")


# ---------------------------------------------------------------- parity

@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dtype", RUN_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_prefill_equals_the_uncached_last_position_bitwise(checkpoints, config, dtype, backend):
    loaded = load(checkpoints, config)
    token_ids = tokens(config, 2, 37)
    engine = Engine(loaded, batch=2, capacity=64, dtype=dtype, backend=backend)
    expected = forward_full(loaded, token_ids, dtype=dtype, backend=backend, logit_positions=[36])[:, 0]
    assert same_bits(engine.prefill(token_ids), expected)


# Without KV grouping repeat_kv copies nothing, and the fp32 CPU BLAS result then depends on
# layout: the reference's cache is contiguous, our cache views are strided. Pinned models are grouped.
UNGROUPED_FP32_MATH = pytest.mark.xfail(reason="layout-dependent fp32 BLAS without KV grouping; machine-specific")
CACHE_PARITY = [pytest.param(config, dtype, backend, marks=UNGROUPED_FP32_MATH)
                if (config, dtype, backend) == ("deep", torch.float32, "math") else (config, dtype, backend)
                for config in CONFIGS for dtype in RUN_DTYPES for backend in BACKENDS]


@pytest.mark.parametrize(("config", "dtype", "backend"), CACHE_PARITY)
def test_cached_decoding_equals_the_reference_cache_bitwise(checkpoints, config, dtype, backend):
    token_ids = tokens(config, 2, 60, seed=1)
    expected = reference_teacher_forced(load_reference(checkpoints[config], dtype, BACKENDS[backend]), token_ids, 9)
    engine = Engine(load(checkpoints, config), batch=2, capacity=60, dtype=dtype, backend=backend)
    assert same_bits(teacher_forced(engine, token_ids, 9), expected.float())


# ---------------------------------------------------------------- the accuracy rule

PROMPT, LENGTH = 16, 80


@pytest.fixture(scope="module")
def rule_inputs(checkpoints):
    """Oracle logits and the reference variants' errors at positions PROMPT-1 to LENGTH-1."""
    loaded = load(checkpoints, "mqa")
    token_ids = tokens("mqa", 2, LENGTH, seed=2)
    positions = list(range(PROMPT - 1, LENGTH))
    oracle = lm_logits(loaded, final_hidden(loaded, token_ids, dtype=F64)[:, positions])
    references = {dtype: [accuracy.errors(oracle, forward_full(loaded, token_ids, dtype=dtype, backend=backend,
                                                                logit_positions=positions)) for backend in BACKENDS]
                  for dtype in RUN_DTYPES}
    return loaded, token_ids, oracle, references


def rule_failures(rule_inputs, dtype, backend, loaded=None):
    default, token_ids, oracle, references = rule_inputs
    engine = Engine(loaded or default, batch=2, capacity=LENGTH + 1, dtype=dtype, backend=backend)  # one spare position
    return accuracy.failures(accuracy.errors(oracle, teacher_forced(engine, token_ids, PROMPT)), references[dtype])


@pytest.mark.parametrize("dtype", RUN_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_cached_decoding_meets_the_accuracy_rule(rule_inputs, dtype, backend):
    assert rule_failures(rule_inputs, dtype, backend) == []


def interleaved_rotary(x, cos, sin):
    half = x.shape[-1] // 2
    c, s = cos[..., :half].repeat_interleave(2, -1), sin[..., :half].repeat_interleave(2, -1)
    rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return x * c + rotated * s


def bf16_rms_norm(x, weight, eps):
    h = x.bfloat16()
    return (weight.bfloat16() * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))).to(x.dtype)


def divided_rms_norm(x, weight, eps):
    """x / sqrt(mean + eps) for x * rsqrt(mean + eps): the same value, rounded differently."""
    h = x.float() if x.dtype != F64 else x
    return weight * (h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def seed(monkeypatch, bug):
    attention, decoder_layer, rotary_tables = ops.attention, ipe.engine.decoder_layer, ops.rotary_tables
    if bug == "interleaved rotary":
        monkeypatch.setattr(ops, "apply_rotary", interleaved_rotary)
    elif bug == "decode rotated one position late":
        def late(x, layer, rope, config, *, backend, cache=None, start=0):
            if start > 0:
                rope = tuple(t[1:] for t in rotary_tables(torch.arange(start, start + 2), config.head_dim,
                                                          config.rope_theta, x.dtype))
            return decoder_layer(x, layer, rope, config, backend=backend, cache=cache, start=start)
        monkeypatch.setattr(ipe.engine, "decoder_layer", late)
    elif bug == "no QK-norm":
        monkeypatch.setattr(ops, "qk_norm", lambda x, weight, eps: x)
    elif bug == "RMSNorm in bf16":
        monkeypatch.setattr(ops, "rms_norm", bf16_rms_norm)
    elif bug == "first decode step rotated one position late":
        def first_late(x, layer, rope, config, *, backend, cache=None, start=0):
            if start == PROMPT:
                rope = tuple(t[1:] for t in rotary_tables(torch.arange(start, start + 2), config.head_dim,
                                                          config.rope_theta, x.dtype))
            return decoder_layer(x, layer, rope, config, backend=backend, cache=cache, start=start)
        monkeypatch.setattr(ipe.engine, "decoder_layer", first_late)
    elif bug == "decode reads one position past its end":
        def past_end(t):
            batch, heads, length, dim = t.shape
            return t.as_strided((batch, heads, length + 1, dim), t.stride(), t.storage_offset())
        monkeypatch.setattr(ops, "attention", lambda q, k, v, *, causal, backend: attention(
            q, k if causal else past_end(k), v if causal else past_end(v), causal=causal, backend=backend))
    elif bug == "decode misses its own key":
        monkeypatch.setattr(ops, "attention", lambda q, k, v, *, causal, backend: attention(
            q, k if causal else k[:, :, :-1], v if causal else v[:, :, :-1], causal=causal, backend=backend))
    elif bug == "uniform rotary offset":
        monkeypatch.setattr(ops, "rotary_tables", lambda positions, *args: rotary_tables(positions + 1, *args))
    elif bug == "RMSNorm divides by the root":
        monkeypatch.setattr(ops, "rms_norm", divided_rms_norm)


BUGS = ("interleaved rotary", "decode rotated one position late", "first decode step rotated one position late",
        "no QK-norm", "RMSNorm in bf16", "decode misses its own key", "decode reads one position past its end",
        "epsilon 1e-5")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("bug", BUGS)
def test_the_rule_fails_seeded_bugs_at_fp32(rule_inputs, monkeypatch, backend, bug):
    loaded = rule_inputs[0]
    if bug == "epsilon 1e-5":
        loaded = dataclasses.replace(loaded, config=dataclasses.replace(loaded.config, rms_norm_eps=1e-5))
    seed(monkeypatch, bug)
    assert rule_failures(rule_inputs, torch.float32, backend, loaded) != []


@pytest.mark.parametrize("dtype", RUN_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", ("uniform rotary offset", "RMSNorm divides by the root"))
def test_the_rule_accepts_legitimate_reorderings(rule_inputs, monkeypatch, dtype, backend, variant):
    seed(monkeypatch, variant)
    assert rule_failures(rule_inputs, dtype, backend) == []


# ---------------------------------------------------------------- the cache contract

def test_decode_validates_state_and_inputs(checkpoints):
    engine = Engine(load(checkpoints), batch=2, capacity=6, dtype=torch.float32)
    with pytest.raises(ValueError, match="prefill first"):
        engine.decode(torch.tensor([1, 2]))
    engine.prefill(tokens("gqa", 2, 5))
    for bad in (torch.tensor([1]), torch.tensor([1.0, 2.0]), torch.tensor([1, 96]), [1, 2]):
        with pytest.raises(ValueError):
            engine.decode(bad)
    assert engine.length == 5
    engine.decode(torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="all 6 positions"):
        engine.decode(torch.tensor([1, 2]))
    assert engine.length == 6


def test_an_invalid_prefill_leaves_the_current_batch(checkpoints):
    engine = Engine(load(checkpoints), batch=2, capacity=6, dtype=torch.float32)
    engine.prefill(tokens("gqa", 2, 4))
    for bad in (tokens("gqa", 3, 4), tokens("gqa", 1, 7), torch.zeros(1, 0, dtype=torch.int64), torch.tensor([[1, 96]])):
        with pytest.raises(ValueError):
            engine.prefill(bad)
    assert (engine.batch, engine.length) == (2, 4)
    engine.decode(torch.tensor([1, 2]))


@pytest.mark.parametrize("arguments", [{"batch": 0, "capacity": 4}, {"batch": 1, "capacity": 0},
                                       {"batch": 1, "capacity": 8193}, {"batch": 1, "capacity": 4, "backend": "flash"},
                                       {"batch": 1, "capacity": 4, "dtype": F64}])
def test_construction_validates_its_arguments(checkpoints, arguments):
    with pytest.raises(ValueError):
        Engine(load(checkpoints), **arguments)


def test_a_new_batch_never_reads_the_previous_one(checkpoints):
    """A short batch after a long one decodes as on a fresh engine, with unwritten positions NaN."""
    loaded = load(checkpoints)
    long_ids, short_ids = tokens("gqa", 2, 30, seed=3), tokens("gqa", 1, 12, seed=4)
    fresh = teacher_forced(Engine(loaded, batch=2, capacity=30, dtype=torch.float32), short_ids, 4)
    engine = Engine(loaded, batch=2, capacity=30, dtype=torch.float32)
    teacher_forced(engine, long_ids, 3)
    engine.prefill(short_ids[:, :4])
    engine._k[:, :, :, 4:] = float("nan")  # positions the short batch has not written yet
    engine._v[:, :, :, 4:] = float("nan")
    engine._k[:, 1:] = engine._v[:, 1:] = float("nan")  # the row the short batch does not use
    logits = [engine.decode(short_ids[:, p]) for p in range(4, 12)]
    assert same_bits(torch.stack(logits, dim=1), fresh[:, 1:])


def test_a_failed_step_keeps_the_committed_prefix(checkpoints, monkeypatch):
    loaded = load(checkpoints)
    token_ids = tokens("gqa", 2, 12, seed=5)
    expected = teacher_forced(Engine(loaded, batch=2, capacity=12, dtype=torch.float32), token_ids, 6)
    engine = Engine(loaded, batch=2, capacity=12, dtype=torch.float32)
    logits = [engine.prefill(token_ids[:, :6])]
    real, calls = ipe.engine.decoder_layer, []

    def fail_in_layer_1(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("injected")
        return real(*args, **kwargs)

    monkeypatch.setattr(ipe.engine, "decoder_layer", fail_in_layer_1)
    with pytest.raises(RuntimeError, match="injected"):
        engine.decode(token_ids[:, 6])  # layer 0 has already written position 6
    monkeypatch.undo()
    assert engine.length == 6
    logits += [engine.decode(token_ids[:, p]) for p in range(6, 12)]
    assert same_bits(torch.stack(logits, dim=1), expected)


def test_capacity_holds_prompt_plus_new_tokens_minus_one(checkpoints):
    engine = Engine(load(checkpoints), batch=1, capacity=8, dtype=torch.float32)
    assert engine.generate(tokens("gqa", 1, 5), 4).tokens.shape == (1, 4)
    assert engine.length == 8
    with pytest.raises(ValueError, match="exceeds capacity"):
        engine.generate(tokens("gqa", 1, 5), 5)
    assert engine.generate(tokens("gqa", 1, 1), 1).tokens.shape == (1, 1)  # P = 1, N = 1
    assert engine.length == 1


# ---------------------------------------------------------------- sampling and generation

def test_greedy_breaks_ties_to_the_lowest_id_and_reports_log_probabilities():
    logits = torch.tensor([[0.0, 3.0, 3.0, 1.0], [5.0, 5.0, 5.0, 5.0]])
    tokens_, logprobs = sample(logits, 0.0)
    assert tokens_.tolist() == [1, 0]
    assert torch.equal(logprobs, F.log_softmax(logits, dim=-1)[[0, 1], [1, 0]])


@pytest.mark.parametrize("temperature", (0.6, 1.0))
def test_temperature_sampling_follows_the_tempered_distribution(temperature):
    """Chi-square, 100,000 draws, alpha 0.01, one fixed seed; expected counts below 5 pooled."""
    logits = torch.linspace(-6.0, 4.0, 50)
    draws, _ = sample(logits.expand(100_000, 50), temperature, torch.Generator().manual_seed(20261016))
    observed = torch.bincount(draws, minlength=50).double()
    expected = torch.softmax(logits.double() / temperature, dim=0) * 100_000
    small = expected < 5
    observed = torch.cat((observed[~small], observed[small].sum()[None]))
    expected = torch.cat((expected[~small], expected[small].sum()[None]))
    statistic = ((observed - expected) ** 2 / expected).sum().item()
    degrees = len(expected) - 1
    critical = mpmath.findroot(lambda x: mpmath.gammainc(degrees / 2, 0, x / 2, regularized=True) - 0.99, degrees)
    assert statistic < critical, (statistic, degrees, critical)


def test_the_same_seed_gives_the_same_draws():
    logits = torch.randn((4, 100), generator=torch.Generator().manual_seed(11))
    draws = [sample(logits, 1.0, torch.Generator().manual_seed(seed))[0] for seed in (7, 7, 8)]
    assert torch.equal(draws[0], draws[1]) and not torch.equal(draws[0], draws[2])


def test_generation_is_reproducible_and_logprobs_match_the_logits(checkpoints):
    # The synthetic models put nearly all mass on one token; a high temperature makes draws vary.
    loaded = load(checkpoints)
    engine = Engine(loaded, batch=2, capacity=24, dtype=torch.float32)
    prompt = tokens("gqa", 2, 8, seed=6)
    first = engine.generate(prompt, 16, temperature=30.0, seed=7)
    assert torch.equal(engine.generate(prompt, 16, temperature=30.0, seed=7).tokens, first.tokens)
    assert len(first.tokens.unique()) > 4
    # Replaying the sequence through the cache reproduces the logits each token was drawn from.
    logits = teacher_forced(engine, torch.cat((prompt, first.tokens[:, :-1]), dim=1), 8)
    expected = F.log_softmax(logits, dim=-1).gather(-1, first.tokens[..., None]).squeeze(-1)
    assert same_bits(first.logprobs, expected)


def test_greedy_generation_equals_the_reference(checkpoints):
    prompt = tokens("gqa", 2, 8, seed=9)
    model = load_reference(checkpoints["gqa"], torch.float32, "eager")
    with torch.no_grad():
        expected = model.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=False, max_new_tokens=20,
                                  eos_token_id=None, pad_token_id=0)[:, 8:]
    engine = Engine(load(checkpoints), batch=2, capacity=27, dtype=torch.float32, backend="math")
    assert torch.equal(engine.generate(prompt, 20).tokens, expected)


def test_stop_ids_set_lengths_and_end_generation_when_every_sequence_has_stopped(checkpoints):
    engine = Engine(load(checkpoints), batch=2, capacity=40, dtype=torch.float32)
    prompt = tokens("gqa", 2, 8, seed=10)
    free = engine.generate(prompt, 30).tokens
    stops = (free[0, 3].item(), free[1, 5].item())
    lengths = [int(torch.isin(row, torch.tensor(stops)).nonzero()[0]) + 1 for row in free]
    stopped = engine.generate(prompt, 30, stop_ids=stops)
    assert stopped.lengths.tolist() == lengths
    assert max(lengths) < 30 and torch.equal(stopped.tokens, free[:, :max(lengths)])


# ---------------------------------------------------------------- what a step does on the device

# Operators that read device values to the host: scalars, data-dependent shapes.
SYNCHRONIZING = {"_local_scalar_dense", "is_nonzero", "equal", "nonzero", "masked_select", "_unique2",
                 "repeat_interleave", "index"}


class Trace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_step_never_reads_device_values(checkpoints, backend):
    """The device work of a prefill and a decode step; decode checks only host IDs."""
    engine = Engine(load(checkpoints), batch=2, capacity=16, dtype=torch.float32, backend=backend)
    engine.prefill(tokens("gqa", 2, 8))
    cos, sin = ops.rotary_tables(torch.arange(8), 16, CONFIGS["gqa"]["rope_theta"], torch.float32)
    with Trace() as trace:
        engine._step(tokens("gqa", 2, 8), (cos, sin), 0)
        engine._step(torch.tensor([[1], [2]]), (engine._cos[8:9], engine._sin[8:9]), 8)
    assert SYNCHRONIZING.isdisjoint(trace.ops), sorted(SYNCHRONIZING & set(trace.ops))


def test_sdpa_is_called_without_a_mask(checkpoints, monkeypatch):
    calls, original = [], F.scaled_dot_product_attention

    def spy(*args, **kwargs):
        calls.append({key: kwargs.get(key) for key in ("attn_mask", "is_causal")})
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    engine = Engine(load(checkpoints), batch=2, capacity=16, dtype=torch.float32, backend="sdpa")
    engine.prefill(tokens("gqa", 2, 8))
    engine.decode(torch.tensor([1, 2]))
    layers = CONFIGS["gqa"]["num_hidden_layers"]
    assert calls == [{"attn_mask": None, "is_causal": True}] * layers + [{"attn_mask": None, "is_causal": False}] * layers


# ---------------------------------------------------------------- the development checkpoint

@pytest.mark.checkpoint
@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_cached_decoding_meets_the_accuracy_rule_on_the_development_checkpoint(stored, dtype):
    """Qwen3-0.6B, one 256-token README window, prompt 64, both backends."""
    from forward_child import token_rows

    stored("qwen3-0.6b")  # verified against its pin
    directory = WEIGHTS / "qwen3-0.6b"
    loaded = load_checkpoint(directory, dtype=torch.bfloat16, device="cpu")
    token_ids = token_rows(directory, 256, 1)
    positions = list(range(63, 256))
    oracle = lm_logits(loaded, final_hidden(loaded, token_ids, dtype=F64)[:, positions])
    references = [accuracy.errors(oracle, forward_full(loaded, token_ids, dtype=dtype, backend=backend,
                                                       logit_positions=positions)) for backend in BACKENDS]
    for backend in BACKENDS:
        engine = Engine(loaded, batch=1, capacity=256, dtype=dtype, backend=backend)
        assert accuracy.failures(accuracy.errors(oracle, teacher_forced(engine, token_ids, 64)), references) == []


@pytest.mark.checkpoint
@pytest.mark.parametrize("dtype", RUN_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_cached_and_greedy_decoding_equal_the_reference_on_the_development_checkpoint(stored, dtype, backend):
    """Qwen3-0.6B: teacher-forced over a 256-token window from position 64, and 64 greedy tokens."""
    from forward_child import token_rows

    stored("qwen3-0.6b")
    directory = WEIGHTS / "qwen3-0.6b"
    token_ids = token_rows(directory, 256, 1)
    model = load_reference(directory, dtype, BACKENDS[backend])
    expected = reference_teacher_forced(model, token_ids, 64).float()
    with torch.no_grad():
        greedy = model.generate(token_ids[:, :64], attention_mask=torch.ones_like(token_ids[:, :64]), do_sample=False,
                                max_new_tokens=64, eos_token_id=None, pad_token_id=0)[:, 64:]
    del model
    engine = Engine(load_checkpoint(directory, dtype=torch.bfloat16, device="cpu"), batch=1, capacity=256, dtype=dtype,
                    backend=backend)
    assert same_bits(teacher_forced(engine, token_ids, 64), expected)
    assert torch.equal(engine.generate(token_ids[:, :64], 64).tokens, greedy)
