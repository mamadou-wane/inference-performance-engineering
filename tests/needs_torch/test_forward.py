"""The forward pass on synthetic checkpoints: bitwise parity with Transformers 5.17.0 at bf16 and
fp32, the fp64 oracle against a Transformers model patched to fp64, error bounds at fp64, output
contracts and input validation.

Synthetic checkpoints hold seeded random bf16 weights scaled to keep activations near unit size,
in shapes that separate Q from H and K from Q, with an unusual epsilon and rotary base so a
hard-coded constant fails. The reference loads the same files with from_pretrained. Layer outputs
are captured with forward hooks on model.model.layers[i]; output_hidden_states would replace the
last one with the final-normed state.

What these tests leave unproven: the pinned checkpoints and lengths up to 4,224 (checkpoint-marked
tests), GPU kernels, and the cached path (test_engine.py).
"""

import ast
import dataclasses
import os
import subprocess
import sys
import warnings
from pathlib import Path

import bounded as bd
import pytest
import torch
from safetensors import safe_open
from test_weights import config_document, reference_shapes
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as reference

import ipe.forward
from ipe import ops
from ipe.forward import decoder_layer, final_hidden, forward_full, lm_logits
from ipe.weights import load_checkpoint

F64 = torch.float64
RUN_DTYPES = (torch.bfloat16, torch.float32)
BACKENDS = {"math": "eager", "sdpa": "sdpa"}


def document(*, eps, theta, **dimensions):
    result = config_document(**dimensions)
    result.update(max_position_embeddings=8192, rms_norm_eps=eps, rope_theta=theta)
    return result


CONFIGS = {
    # Q = 4 x 16 = 64 differs from H = 48; K = 32 differs from Q; two query heads per KV head.
    "gqa": document(vocab=96, hidden=48, intermediate=80, layers=2, heads=4, kv_heads=2, head_dim=16, eps=1e-5, theta=1e4),
    # Four query heads share one KV head.
    "mqa": document(vocab=64, hidden=32, intermediate=64, layers=3, heads=4, kv_heads=1, head_dim=8, eps=1e-6, theta=1e6),
    # 11 layers, so layer order is not lexical; no grouping.
    "deep": document(vocab=40, hidden=16, intermediate=32, layers=11, heads=2, kv_heads=2, head_dim=8, eps=1e-6, theta=1e6),
}


def random_tensors(doc, seed):
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, shape in sorted(reference_shapes(doc).items()):
        if len(shape) == 1:  # norms near 1
            value = 1 + 0.1 * torch.randn(shape, generator=generator)
        elif name.endswith("embed_tokens.weight"):
            value = torch.randn(shape, generator=generator)
        else:
            value = torch.randn(shape, generator=generator) / shape[1] ** 0.5
        tensors[name] = value.to(torch.bfloat16)
    return tensors


def tokens(name, batch, length, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, CONFIGS[name]["vocab_size"], (batch, length), generator=generator)


def same_bits(actual, expected):
    return actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def load_reference(directory, dtype, attention):
    model = Qwen3ForCausalLM.from_pretrained(directory, dtype=dtype, attn_implementation=attention).eval()
    # model.to(dtype) after loading would round inv_freq; from_pretrained keeps it fp32.
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    return model


@torch.no_grad()
def reference_run(model, token_ids):
    """Logits, each layer's output and the final-normed state, from one uncached forward."""
    outputs = []
    hooks = [layer.register_forward_hook(lambda _m, _i, out: outputs.append(out)) for layer in model.model.layers]
    hooks.append(model.model.norm.register_forward_hook(lambda _m, _i, out: outputs.append(out)))
    try:
        logits = model(input_ids=token_ids, use_cache=False).logits
    finally:
        for hook in hooks:
            hook.remove()
    return logits, outputs[:-1], outputs[-1]


def engine_run(loaded, token_ids, dtype, backend):
    layers = []
    hidden = final_hidden(loaded, token_ids, dtype=dtype, backend=backend,
                          capture=lambda index, output: layers.append((index, output)))
    assert [index for index, _ in layers] == list(range(len(loaded.weights.layers)))
    return lm_logits(loaded, hidden), [output for _, output in layers], hidden


def first_divergence(engine_layers, reference_layers):
    """The earliest layer whose output differs, for the failure message."""
    for index, (ours, theirs) in enumerate(zip(engine_layers, reference_layers, strict=True)):
        if not same_bits(ours, theirs):
            return f"layer {index}: {int((ours != theirs).sum())} elements differ"
    return "no layer differs"


# ---------------------------------------------------------------- parity with Transformers

@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dtype", RUN_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("length", (1, 13, 70, 300))
def test_logits_and_every_layer_match_the_reference_bitwise(checkpoints, config, dtype, backend, length):
    token_ids = tokens(config, 2, length)
    expected_logits, expected_layers, expected_hidden = reference_run(
        load_reference(checkpoints[config], dtype, BACKENDS[backend]), token_ids)
    loaded = load_checkpoint(checkpoints[config], dtype=torch.bfloat16, device="cpu")
    logits, layers, hidden = engine_run(loaded, token_ids, dtype, backend)
    assert all(same_bits(o, e) for o, e in zip(layers, expected_layers, strict=True)), first_divergence(layers, expected_layers)
    assert same_bits(hidden, expected_hidden)
    assert same_bits(logits, expected_logits)
    assert same_bits(forward_full(loaded, token_ids, dtype=dtype, backend=backend), expected_logits.float())


@pytest.mark.parametrize("config", CONFIGS)
def test_fp32_storage_gives_the_same_fp32_run_as_bf16_storage(checkpoints, config):
    token_ids = tokens(config, 2, 40)
    from_views = load_checkpoint(checkpoints[config], dtype=torch.bfloat16, device="cpu")
    converted = load_checkpoint(checkpoints[config], dtype=torch.float32, device="cpu")
    assert same_bits(forward_full(from_views, token_ids, dtype=torch.float32),
                     forward_full(converted, token_ids, dtype=torch.float32))


# ---------------------------------------------------------------- the fp64 oracle

def _rms_norm_fp64(self, x):
    h = x.to(F64)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return self.weight.to(F64) * h


def _rotary_fp64(self, x, position_ids):
    dim = self.config.head_dim
    inv_freq = 1.0 / (float(self.config.rope_parameters["rope_theta"]) ** (torch.arange(0, dim, 2, dtype=F64) / dim))
    angles = position_ids[:, :, None].to(F64) * inv_freq[None, None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


def _eager_fp64(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = reference.repeat_kv(key, module.num_key_value_groups)
    value = reference.repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = torch.nn.functional.softmax(weights, dim=-1, dtype=F64).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


@pytest.fixture
def patched_reference(monkeypatch):
    """Transformers with RMSNorm, rotary and eager softmax in fp64: an independent fp64 forward."""
    monkeypatch.setattr(reference.Qwen3RMSNorm, "forward", _rms_norm_fp64)
    monkeypatch.setattr(reference.Qwen3RotaryEmbedding, "forward", _rotary_fp64)
    monkeypatch.setattr(reference, "eager_attention_forward", _eager_fp64)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_the_oracle_matches_the_patched_reference_bitwise(checkpoints, patched_reference, config, backend):
    token_ids = tokens(config, 2, 70)
    expected_logits, expected_layers, _ = reference_run(load_reference(checkpoints[config], F64, BACKENDS[backend]), token_ids)
    loaded = load_checkpoint(checkpoints[config], dtype=torch.bfloat16, device="cpu")
    logits, layers, _ = engine_run(loaded, token_ids, F64, backend)
    assert all(same_bits(o, e) for o, e in zip(layers, expected_layers, strict=True)), first_divergence(layers, expected_layers)
    assert same_bits(logits, expected_logits)


# Data movement and type conversion; every other op must compute in fp64.
MOVEMENT = {"embedding", "_to_copy", "view", "_unsafe_view", "t", "transpose", "expand", "clone", "reshape", "slice",
            "select", "unsqueeze", "index", "detach", "alias", "permute", "lift_fresh", "_local_scalar_dense"}


class DtypeTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.violations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        floats = [t.dtype for t in torch.utils._pytree.tree_leaves((args, kwargs, out))
                  if isinstance(t, torch.Tensor) and t.dtype.is_floating_point]
        if name not in MOVEMENT and any(d != F64 for d in floats):
            self.violations.append((name, floats))
        # Conversions may only widen to fp64; a round trip through fp32 would pass the rule above.
        if name == "_to_copy" and out.dtype.is_floating_point and out.dtype != F64:
            self.violations.append((name, floats))
        return out


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_fp64_path_computes_every_step_in_fp64(checkpoints, backend):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    trace = DtypeTrace()
    with trace:
        lm_logits(loaded, final_hidden(loaded, tokens("gqa", 1, 20), dtype=F64, backend=backend))
    assert trace.violations == []


def stored_layer(directory, index):
    """Layer weights read by checkpoint name, outside ipe.weights, as fp64 arrays."""
    prefix = f"model.layers.{index}."
    with safe_open(str(directory / "model.safetensors"), framework="pt") as handle:
        return {name.removeprefix(prefix): handle.get_tensor(name).double().numpy()
                for name in handle.keys() if name.startswith(prefix)}  # noqa: SIM118 - safe_open handles are not iterable


@pytest.mark.parametrize("config", CONFIGS)
def test_one_fp64_layer_is_within_its_error_bound(checkpoints, config):
    loaded = load_checkpoint(checkpoints[config], dtype=torch.bfloat16, device="cpu")
    x = torch.randn((2, 60, CONFIGS[config]["hidden_size"]), generator=torch.Generator().manual_seed(3), dtype=F64)
    positions = torch.arange(60)
    rope = ops.rotary_tables(positions, loaded.config.head_dim, loaded.config.rope_theta, F64)
    actual = decoder_layer(x, loaded.weights.layers[-1], rope, loaded.config, backend="math")
    expected = bd.decoder_layer(bd.exact(x.numpy()), stored_layer(checkpoints[config], CONFIGS[config]["num_hidden_layers"] - 1),
                                positions.numpy(), loaded.config, bd.FP64)
    assert bd.ratio(actual.numpy(), expected, 2.0) <= 1.0


def test_fp64_lm_logits_are_within_their_error_bound(checkpoints):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    hidden = torch.randn((2, 30, 48), generator=torch.Generator().manual_seed(4), dtype=F64)
    with safe_open(str(checkpoints["gqa"] / "model.safetensors"), framework="pt") as handle:
        head = handle.get_tensor("model.embed_tokens.weight").double().numpy()
    actual = lm_logits(loaded, hidden)
    assert actual.dtype == F64
    assert bd.ratio(actual.numpy(), bd.linear(bd.exact(hidden.numpy()), head, bd.FP64), 2.0) <= 1.0


# ---------------------------------------------------------------- outputs

@pytest.mark.parametrize("dtype", RUN_DTYPES)
def test_output_dtypes(checkpoints, dtype):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    token_ids = tokens("gqa", 2, 9)
    hidden = final_hidden(loaded, token_ids, dtype=dtype)
    assert hidden.dtype == dtype and hidden.shape == (2, 9, 48)
    assert lm_logits(loaded, hidden).dtype == dtype
    logits = forward_full(loaded, token_ids, dtype=dtype)
    assert logits.dtype == torch.float32 and logits.shape == (2, 9, 96)


ROW_REQUESTS = ([0], [299], [0, 1], [5, 6, 7], [3, 100, 299], list(range(7)), list(range(1, 300, 2)))
# At fp32 a selected-row LM-head product may differ in the last bits from the same rows of the
# all-positions product: the CPU BLAS picks its kernel by shape, and a one-row product is bitwise a
# matrix-vector product, and which shapes differ varies between machines. The difference must stay
# within the fp32 forward-error bound; bf16 must be bitwise. Parity with the reference for the same
# request is a separate, bitwise requirement (test above; logit_position_check).


def check_selected_rows(label, loaded, token_ids, dtype, backend, rows, selected, every, record):
    """Selected rows equal the all-positions rows bitwise, or, at fp32 only, both lie within the
    fp32 forward-error bound of the product. Each non-bitwise case is recorded and reported as a
    warning, so CI logs keep the measurement."""
    if same_bits(selected, every[:, rows]):
        return
    assert dtype == torch.float32, "bf16 selected rows must equal the all-positions rows bitwise"
    hidden = final_hidden(loaded, token_ids, dtype=dtype, backend=backend)[:, rows].double().numpy()
    bound = bd.linear(bd.exact(hidden), loaded.weights.lm_head.double().numpy(), bd.FP32)
    ratios = (bd.ratio(selected.double().numpy(), bound, 1.0), bd.ratio(every[:, rows].double().numpy(), bound, 1.0))
    difference = (selected - every[:, rows]).abs().max().item()
    measurement = (f"{label} {backend} {str(dtype).removeprefix('torch.')} batch {token_ids.shape[0]} rows {len(rows)}: "
                   f"max |difference| {difference:.3e}, largest bound {float(bound.bound.max()):.3e}, "
                   f"error-to-bound ratio selected {ratios[0]:.3e} all-positions {ratios[1]:.3e}")
    record(f"fp32 selected rows {label} {backend} batch {token_ids.shape[0]} rows {rows[:4]}", measurement)
    warnings.warn(f"fp32 selected-row difference within bound: {measurement}", stacklevel=2)
    assert max(ratios) <= 1.0, measurement


@pytest.mark.parametrize("dtype", RUN_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rows", ROW_REQUESTS)
def test_logit_positions_match_the_reference_for_the_same_request(checkpoints, dtype, backend, rows):
    token_ids = tokens("gqa", 2, 300)
    model = load_reference(checkpoints["gqa"], dtype, BACKENDS[backend])
    with torch.no_grad():
        expected = model(input_ids=token_ids, use_cache=False, logits_to_keep=torch.tensor(rows)).logits
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    for request in (rows, torch.tensor(rows), torch.tensor(rows, dtype=torch.int32)):
        assert same_bits(forward_full(loaded, token_ids, dtype=dtype, backend=backend, logit_positions=request),
                         expected.float())


@pytest.mark.parametrize("dtype", RUN_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rows", ROW_REQUESTS)
@pytest.mark.parametrize("batch", (1, 2))
def test_selected_rows_equal_the_all_positions_rows(checkpoints, record_property, dtype, backend, rows, batch):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    token_ids = tokens("gqa", batch, 300)
    every = forward_full(loaded, token_ids, dtype=dtype, backend=backend)
    selected = forward_full(loaded, token_ids, dtype=dtype, backend=backend, logit_positions=rows)
    check_selected_rows("synthetic gqa", loaded, token_ids, dtype, backend, rows, selected, every, record_property)


@pytest.mark.parametrize("dtype", RUN_DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_tokens_after_a_position_leave_its_logits_unchanged(checkpoints, dtype, backend):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    token_ids = tokens("gqa", 2, 80)
    changed = token_ids.clone()
    changed[:, 50:] = (changed[:, 50:] + 1) % CONFIGS["gqa"]["vocab_size"]
    before, after = (forward_full(loaded, t, dtype=dtype, backend=backend) for t in (token_ids, changed))
    assert same_bits(after[:, :50], before[:, :50])
    assert not torch.equal(after[:, 50], before[:, 50])


@pytest.mark.parametrize("dtype", (*RUN_DTYPES, F64))
def test_a_forward_pass_leaves_the_weights_unchanged(checkpoints, dtype):
    loaded = load_checkpoint(checkpoints["mqa"], dtype=torch.bfloat16, device="cpu")
    weights = loaded.weights
    tensors = [weights.embed_tokens, weights.norm, *(t for layer in weights.layers for t in
                                                     (getattr(layer, f) for f in layer.__dataclass_fields__))]
    copies = [t.clone() for t in tensors]
    final_hidden(loaded, tokens("mqa", 2, 30), dtype=dtype)
    assert all(same_bits(t, c) for t, c in zip(tensors, copies, strict=True))


# ---------------------------------------------------------------- validation

@pytest.fixture
def no_compute(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("computation started before validation finished")
    monkeypatch.setattr(ipe.forward, "_final_hidden", refuse)


INVALID_CALLS = {
    "fp64": {"dtype": F64},
    "fp16": {"dtype": torch.float16},
    "int dtype": {"dtype": torch.int32},
    "float ids": {"token_ids": torch.zeros(1, 4)},
    "bool ids": {"token_ids": torch.zeros(1, 4, dtype=torch.bool)},
    "1-D ids": {"token_ids": torch.zeros(4, dtype=torch.int64)},
    "list ids": {"token_ids": [[1, 2, 3]]},
    "empty batch": {"token_ids": torch.zeros(0, 4, dtype=torch.int64)},
    "empty sequence": {"token_ids": torch.zeros(1, 0, dtype=torch.int64)},
    "negative id": {"token_ids": torch.tensor([[1, -1]])},
    "id at vocab size": {"token_ids": torch.tensor([[1, 96]])},
    "longer than the position limit": {"token_ids": torch.zeros(1, 8193, dtype=torch.int64)},
    "backend": {"backend": "flash"},
    "positions empty": {"logit_positions": []},
    "positions float": {"logit_positions": [0.0, 1.0]},
    "positions bool": {"logit_positions": [True]},
    "positions 2-D": {"logit_positions": [[0, 1]]},
    "positions negative": {"logit_positions": [-1]},
    "positions at length": {"logit_positions": [4]},
    "positions empty integers": {"logit_positions": torch.zeros(0, dtype=torch.int64)},
    "positions string": {"logit_positions": "ab"},
    "positions with None": {"logit_positions": [0, None]},
    "positions set": {"logit_positions": {0, 1}},
    "uint16 ids": {"token_ids": torch.zeros(1, 4, dtype=torch.uint16)},
}


@pytest.mark.parametrize("case", INVALID_CALLS)
def test_invalid_forward_full_arguments_raise_before_any_compute(checkpoints, no_compute, case):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    arguments = {"token_ids": torch.zeros(1, 4, dtype=torch.int64), "dtype": torch.float32} | INVALID_CALLS[case]
    with pytest.raises(ValueError):
        forward_full(loaded, arguments.pop("token_ids"), **arguments)


@pytest.mark.parametrize(("dtype", "ids"), [(torch.int8, [5, 100]), (torch.uint8, [5, 200]), (torch.int16, [5, 30000])])
def test_narrow_integer_ids_are_bounds_checked_without_wrapping(checkpoints, no_compute, dtype, ids):
    # 151936 wraps to -128, 128 and 20864 in these dtypes, so a comparison in the input dtype
    # rejects valid IDs.
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    wide = dataclasses.replace(loaded, config=dataclasses.replace(loaded.config, vocab_size=151936))
    with pytest.raises(AssertionError, match="computation started"):
        forward_full(wide, torch.tensor([ids], dtype=dtype), dtype=torch.float32)


def test_narrow_integer_logit_positions_are_bounds_checked_without_wrapping(checkpoints, no_compute):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    with pytest.raises(AssertionError, match="computation started"):
        forward_full(loaded, tokens("gqa", 1, 256), dtype=torch.float32,
                     logit_positions=torch.tensor([255], dtype=torch.uint8))


def test_the_position_limit_bounds_the_length(checkpoints):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    short = dataclasses.replace(loaded, config=dataclasses.replace(loaded.config, max_position_embeddings=64))
    assert forward_full(short, tokens("gqa", 1, 64), dtype=torch.float32).shape == (1, 64, 96)
    with pytest.raises(ValueError, match="length 65 is outside 1 to 64"):
        forward_full(short, tokens("gqa", 1, 65), dtype=torch.float32)


def test_logit_positions_are_returned_in_the_order_given(checkpoints):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    token_ids = tokens("gqa", 2, 20)
    every = forward_full(loaded, token_ids, dtype=torch.bfloat16)
    assert same_bits(forward_full(loaded, token_ids, dtype=torch.bfloat16, logit_positions=[7, 2, 7]), every[:, [7, 2, 7]])


def test_lm_logits_rejects_invalid_hidden_states(checkpoints):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    for hidden in ([[0.0] * 48], torch.zeros(()), torch.zeros(1, 4, 47), torch.zeros(1, 4, 48, dtype=torch.float16),
                   torch.zeros(1, 4, 48, device="meta")):
        with pytest.raises(ValueError):
            lm_logits(loaded, hidden)


@pytest.mark.skipif(os.environ.get("IPE_TEST_MPS") != "1", reason="set IPE_TEST_MPS=1 to run on MPS")
def test_fp64_is_rejected_on_mps(checkpoints):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="mps")
    with pytest.raises(ValueError, match="mps"):
        final_hidden(loaded, torch.zeros(1, 4, dtype=torch.int64), dtype=F64)


@pytest.mark.parametrize(("stored", "run"), [(torch.float32, torch.bfloat16), (F64, torch.float32), (F64, torch.bfloat16)])
def test_weights_wider_than_the_run_are_rejected(checkpoints, no_compute, stored, run):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=stored, device="cpu")
    with pytest.raises(ValueError, match="wider"):
        forward_full(loaded, torch.zeros(1, 4, dtype=torch.int64), dtype=run)
    with pytest.raises(ValueError, match="wider"):
        final_hidden(loaded, torch.zeros(1, 4, dtype=torch.int64), dtype=run)
    with pytest.raises(ValueError, match="wider"):
        lm_logits(loaded, torch.zeros(1, 4, 48, dtype=run))


def test_final_hidden_accepts_fp64_and_validates_like_forward_full(checkpoints, no_compute):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu")
    for case in ("float ids", "id at vocab size", "longer than the position limit", "backend", "fp16"):
        arguments = {"token_ids": torch.zeros(1, 4, dtype=torch.int64), "dtype": F64} | INVALID_CALLS[case]
        with pytest.raises(ValueError):
            final_hidden(loaded, arguments.pop("token_ids"), **arguments)


def test_ipe_forward_never_imports_transformers():
    code = "import sys, ipe.forward; assert not any(m.startswith('transformers') for m in sys.modules)"
    env = {**os.environ, "PYTHONPATH": str(Path(ipe.forward.__file__).parents[1])}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_the_independent_formulas_do_not_import_ipe():
    tree = ast.parse((Path(__file__).parent / "bounded.py").read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported == {"dataclasses", "numpy"}
