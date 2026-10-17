"""GPU correctness on Qwen3-1.7B in bf16 and synthetic checkpoints: the accuracy rule against an fp64 oracle on the same GPU,
decode without host synchronization or math-kernel fallback, reproducible sampling, the TF32 guard.
Bitwise parity with Transformers is recorded, not asserted. 1,024 tokens at session entry; 4,224
before any W3 result (-k 4224).
"""

import gc

import accuracy
import pytest
import torch
from test_engine import BACKENDS, reference_teacher_forced, same_bits, teacher_forced
from test_weights import WEIGHTS
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3ForCausalLM

from ipe import ops
from ipe.engine import Engine, sample
from ipe.forward import final_hidden, forward_full, lm_logits
from ipe.weights import load_checkpoint

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")
PROMPT = 64
FAST_KERNELS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]


def free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def rule_on_device(directory, token_ids, device, record):
    """Accuracy-rule failures per engine backend for bf16 cached decoding of token_ids [1, T]."""
    device, length = torch.device(device), token_ids.shape[1]
    positions = list(range(PROMPT - 1, length))
    loaded = load_checkpoint(directory, dtype=torch.bfloat16, device=device)
    candidates = {}
    for backend in BACKENDS:
        engine = Engine(loaded, batch=1, capacity=length, backend=backend)
        candidates[backend] = teacher_forced(engine, token_ids, PROMPT)[0].cpu()
        del engine
    if length <= 1024:
        ours = {backend: forward_full(loaded, token_ids, dtype=torch.bfloat16, backend=backend)[0, positions].cpu()
                for backend in BACKENDS}
    free(device)

    references, cached_reference = {}, None
    record("bf16 reduced-precision reduction", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    for backend, attention in BACKENDS.items():
        model = Qwen3ForCausalLM.from_pretrained(directory, dtype=torch.bfloat16, attn_implementation=attention)
        model = model.to(device).eval()
        references[backend] = model(input_ids=token_ids.to(device), use_cache=False).logits[0, positions].float().cpu()
        if length <= 1024:
            equal = torch.equal(ours[backend], references[backend])
            record(f"bitwise uncached logits {backend} {length}", equal)
            if not equal:
                largest = (ours[backend] - references[backend]).abs().max().item()
                record(f"largest uncached difference {backend} {length}", largest)
            cos, sin = model.model.rotary_emb(torch.zeros(1, dtype=torch.bfloat16, device=device),
                                              torch.arange(length, device=device)[None])
            ours_cos, ours_sin = ops.rotary_tables(torch.arange(length), loaded.config.head_dim, loaded.config.rope_theta,
                                                   torch.bfloat16)
            record(f"rotary tables equal {length}", same_bits(cos[0].cpu(), ours_cos) and same_bits(sin[0].cpu(), ours_sin))
            if backend == "sdpa":
                cached_reference = reference_teacher_forced(model, token_ids.to(device), PROMPT)[0].float().cpu()
        del model
        free(device)

    hidden = final_hidden(loaded, token_ids.to(device), dtype=torch.float64)[0, positions]
    if device.type != "cpu":  # anchor: the device oracle equals the CPU oracle on a prefix
        prefix = min(length, PROMPT + 64)
        cpu = load_checkpoint(directory, dtype=torch.bfloat16, device="cpu")
        expected = lm_logits(cpu, final_hidden(cpu, token_ids[:, :prefix], dtype=torch.float64)[0, PROMPT - 1:prefix])
        actual = lm_logits(loaded, hidden[:prefix - PROMPT + 1]).cpu()
        record(f"oracle device vs cpu max difference {length}", (actual - expected).abs().max().item())
        assert torch.allclose(actual, expected, rtol=0, atol=1e-8)
    names = (*BACKENDS, *(f"engine {backend}" for backend in BACKENDS)) + (("transformers cached",) if cached_reference is not None else ())
    errors = {name: ([], []) for name in names}
    for start in range(0, len(positions), 512):
        rows = slice(start, start + 512)
        oracle = lm_logits(loaded, hidden[rows])
        pairs = [(backend, references[backend]) for backend in BACKENDS]
        pairs += [(f"engine {backend}", candidates[backend]) for backend in BACKENDS]
        pairs += [("transformers cached", cached_reference)] if cached_reference is not None else []
        for name, logits in pairs:
            part = accuracy.errors(oracle, logits[rows])
            errors[name][0].append(part.max_abs.cpu())
            errors[name][1].append(part.kl.cpu())
        del oracle
    errors = {name: accuracy.Errors(torch.cat(e), torch.cat(kl)) for name, (e, kl) in errors.items()}
    for name, value in errors.items():
        record(f"summary {name} {length}", value.summary())
    # Acceptance envelope: Transformers eager and SDPA, uncached; its cached decoding is reported only.
    reference_errors = [errors[backend] for backend in BACKENDS]
    return {backend: accuracy.failures(errors[f"engine {backend}"], reference_errors) for backend in BACKENDS}


def target_tokens(length):
    from forward_child import token_rows

    return token_rows(WEIGHTS / "qwen3-1.7b", length, 1)


@CUDA
@pytest.mark.gpu
@pytest.mark.checkpoint
@pytest.mark.parametrize("length", (1024, 4224))
def test_cached_decoding_meets_the_accuracy_rule_on_the_gpu(stored, record_property, length):
    stored("qwen3-1.7b")
    failures = rule_on_device(WEIGHTS / "qwen3-1.7b", target_tokens(length), "cuda", record_property)
    assert failures == {backend: [] for backend in BACKENDS}


@CUDA
@pytest.mark.gpu
@pytest.mark.checkpoint
def test_decode_steps_never_synchronize_or_grow_the_allocator(stored):
    stored("qwen3-1.7b")
    engine = Engine(load_checkpoint(WEIGHTS / "qwen3-1.7b", dtype=torch.bfloat16, device="cuda"), batch=1, capacity=600)
    tokens = sample(engine.prefill(target_tokens(512)), 0.0)[0]
    tokens = sample(engine.decode(tokens), 0.0)[0]  # first decode allocates its working memory
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(32):
            tokens = sample(engine.decode(tokens), 0.0)[0]
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    assert torch.cuda.memory_reserved() == reserved


def run_without_math_attention(loaded, token_ids):
    """Prefill and four decode steps with SDPA limited to flash, memory-efficient and cuDNN."""
    engine = Engine(loaded, batch=1, capacity=token_ids.shape[1] + 4, backend="sdpa")
    with sdpa_kernel(FAST_KERNELS):
        tokens = sample(engine.prefill(token_ids), 0.0)[0]
        for _ in range(4):
            tokens = sample(engine.decode(tokens), 0.0)[0]


@CUDA
@pytest.mark.gpu
@pytest.mark.checkpoint
def test_engine_attention_runs_without_the_math_kernel(stored):
    stored("qwen3-1.7b")
    run_without_math_attention(load_checkpoint(WEIGHTS / "qwen3-1.7b", dtype=torch.bfloat16, device="cuda"),
                               target_tokens(512).cuda())


@CUDA
@pytest.mark.gpu
def test_sampling_and_generation_are_reproducible_on_the_gpu(checkpoints):
    """Batch 2 at temperature 30: the same seed gives the same tokens, and the log-probabilities are
    those of the logits each token was drawn from (replayed through the cache)."""
    engine = Engine(load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cuda"), batch=2, capacity=40)
    prompt = torch.randint(0, 96, (2, 8), generator=torch.Generator().manual_seed(6)).cuda()
    first = engine.generate(prompt, 24, temperature=30.0, seed=7)  # synthetic models put nearly all mass on one token
    assert torch.equal(engine.generate(prompt, 24, temperature=30.0, seed=7).tokens, first.tokens)
    assert len(first.tokens.unique()) > 4
    logits = teacher_forced(engine, torch.cat((prompt.cpu(), first.tokens[:, :-1]), dim=1), 8)  # on the GPU
    expected = torch.log_softmax(logits, dim=-1).gather(-1, first.tokens[..., None].cuda()).squeeze(-1)
    assert same_bits(first.logprobs, expected.cpu())


@CUDA
@pytest.mark.gpu
def test_fp32_on_cuda_refuses_tf32_matmul(checkpoints):
    loaded = load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cuda")
    previous = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    try:
        with pytest.raises(ValueError, match="TF32"):
            Engine(loaded, batch=1, capacity=8, dtype=torch.float32)
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous
    Engine(loaded, batch=1, capacity=8, dtype=torch.float32)


def test_the_gpu_checks_run_on_the_cpu(checkpoints, record_property):
    """The rule code and the SDPA kernel restriction on a synthetic checkpoint on the CPU; this says
    nothing about CUDA kernels."""
    token_ids = torch.randint(0, 96, (1, PROMPT + 40), generator=torch.Generator().manual_seed(0))
    assert rule_on_device(checkpoints["gqa"], token_ids, "cpu", record_property) == {backend: [] for backend in BACKENDS}
    run_without_math_attention(load_checkpoint(checkpoints["gqa"], dtype=torch.bfloat16, device="cpu"), token_ids)
