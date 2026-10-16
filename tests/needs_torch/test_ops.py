"""Operators: bitwise parity with Transformers 5.17.0 at bf16 and fp32, and fp64 and fp32 accuracy
against independent formulas with running error bounds (bounded.py).

Parity tests feed identical inputs to ipe.ops and to the reference's own functions and modules.
The eager-attention parity test builds the additive mask the way the reference's masking code
does; the model-level tests in test_forward.py exercise the reference's own mask construction.

What these tests leave unproven: GPU kernels, inputs outside the sampled ranges, and model-level
composition (test_forward.py).
"""

from types import SimpleNamespace

import bounded as bd
import numpy as np
import pytest
import torch
import transformers
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3 import modeling_qwen3 as reference

from ipe import ops

BITWISE_DTYPES = (torch.bfloat16, torch.float32)
BOUNDED = {torch.float64: (bd.FP64,), torch.float32: (bd.FP32, bd.FP64)}


def same_bits(actual, expected):
    return actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def activations(shape, seed, *, scale=1.0):
    """Normal values with one large outlier and one near-constant row, as in real hidden states."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=generator, dtype=torch.float64) * scale
    flat = x.view(-1, shape[-1])
    flat[0, 1] = 80 * scale
    flat[-1] = flat[-1] * 1e-4
    return x


def norm_weight(size, seed):
    """Norm weights near 1 that are not exactly 1 in bf16, so the weight multiply's rounding shows."""
    generator = torch.Generator().manual_seed(seed)
    return 1 + 0.2 * torch.randn(size, generator=generator, dtype=torch.float64)


def hf_config(*, hidden=64, heads=4, kv_heads=2, head_dim=16, intermediate=96, theta=1e6, eps=1e-6):
    return transformers.Qwen3Config(
        vocab_size=32, hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=1,
        num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim,
        max_position_embeddings=65536, rms_norm_eps=eps, rope_theta=theta, tie_word_embeddings=True)


def check_bounded(actual: torch.Tensor, independent, dtype):
    """actual within the bound at its own precision plus the independent value's fp64 bound."""
    precisions = BOUNDED[dtype]
    if len(precisions) == 1:
        expected = independent(precisions[0])
        assert bd.ratio(actual.numpy(), expected, 2.0) <= 1.0
    else:
        low, high = (independent(p) for p in precisions)
        combined = bd.Bounded(high.value, low.bound + high.bound)
        assert bd.ratio(actual.double().numpy(), combined, 1.0) <= 1.0


def representable(x: torch.Tensor, dtype) -> np.ndarray:
    """x rounded to dtype, as fp64 numpy: the exact inputs both sides receive."""
    return x.to(dtype).double().numpy()


# ---------------------------------------------------------------- bitwise parity with Transformers

@pytest.mark.parametrize("dtype", BITWISE_DTYPES)
@pytest.mark.parametrize(("shape", "eps"), [((2, 37, 64), 1e-6), ((2, 37, 4, 16), 1e-6), ((1, 9, 64), 1e-3)])
def test_rms_norm_matches_the_reference_bitwise(dtype, shape, eps):
    x = activations(shape, 1).to(dtype)
    weight = norm_weight(shape[-1], 2).to(dtype)
    assert (weight != 1).float().mean() > 0.9
    module = reference.Qwen3RMSNorm(shape[-1], eps=eps).to(dtype)
    with torch.no_grad():
        module.weight.copy_(weight)
        expected = module(x)
    assert same_bits(ops.rms_norm(x, weight, eps), expected)
    assert same_bits(ops.qk_norm(x, weight, eps), expected)


@pytest.mark.parametrize("dtype", BITWISE_DTYPES)
@pytest.mark.parametrize(("head_dim", "theta", "positions"), [(128, 1e6, 40960), (16, 1e4, 300), (8, 1e6, 1)])
def test_rotary_tables_match_the_reference_bitwise(dtype, head_dim, theta, positions):
    rotary = reference.Qwen3RotaryEmbedding(hf_config(head_dim=head_dim, theta=theta))
    assert rotary.inv_freq.dtype == torch.float32
    position_ids = torch.arange(positions)[None]
    expected_cos, expected_sin = rotary(torch.empty(0, dtype=dtype), position_ids)
    cos, sin = ops.rotary_tables(torch.arange(positions), head_dim, theta, dtype)
    assert same_bits(cos, expected_cos[0]) and same_bits(sin, expected_sin[0])


@pytest.mark.parametrize("dtype", BITWISE_DTYPES)
def test_apply_rotary_matches_the_reference_bitwise(dtype):
    q, k = activations((2, 4, 300, 16), 3).to(dtype), activations((2, 2, 300, 16), 4).to(dtype)
    cos, sin = ops.rotary_tables(torch.arange(300), 16, 1e4, dtype)
    expected_q, expected_k = reference.apply_rotary_pos_emb(q, k, cos[None], sin[None])
    assert same_bits(ops.apply_rotary(q, cos, sin), expected_q)
    assert same_bits(ops.apply_rotary(k, cos, sin), expected_k)


def attention_inputs(dtype, *, heads, kv_heads, length, seed=5):
    q = activations((2, heads, length, 16), seed)
    k = activations((2, kv_heads, length, 16), seed + 1)
    v = activations((2, kv_heads, length, 16), seed + 2)
    return q.to(dtype), k.to(dtype), v.to(dtype)


GQA = [(4, 4, 1), (4, 2, 37), (8, 2, 300), (4, 1, 257)]


@pytest.mark.parametrize("dtype", BITWISE_DTYPES)
@pytest.mark.parametrize(("heads", "kv_heads", "length"), GQA)
def test_math_attention_matches_eager_bitwise(dtype, heads, kv_heads, length):
    q, k, v = attention_inputs(dtype, heads=heads, kv_heads=kv_heads, length=length)
    positions = torch.arange(length)
    visible = positions[None, :] <= positions[:, None]
    mask = torch.where(visible, torch.tensor(0.0, dtype=dtype), torch.finfo(dtype).min)[None, None]
    module = SimpleNamespace(num_key_value_groups=heads // kv_heads, training=False)
    expected, _ = reference.eager_attention_forward(module, q, k, v, mask, scaling=16 ** -0.5)
    actual = ops.attention(q, k, v, causal=True, backend="math")
    assert same_bits(actual.transpose(1, 2).contiguous(), expected)
    # A decode step: one query after every key; the reference adds an all-zero mask.
    expected, _ = reference.eager_attention_forward(module, q[:, :, -1:], k, v, torch.zeros_like(mask[..., -1:, :]),
                                                    scaling=16 ** -0.5)
    actual = ops.attention(q[:, :, -1:], k, v, causal=False, backend="math")
    assert same_bits(actual.transpose(1, 2).contiguous(), expected)


@pytest.mark.parametrize("dtype", (*BITWISE_DTYPES, torch.float64))
@pytest.mark.parametrize(("heads", "kv_heads", "length"), GQA)
def test_sdpa_attention_matches_the_reference_bitwise(dtype, heads, kv_heads, length):
    q, k, v = attention_inputs(dtype, heads=heads, kv_heads=kv_heads, length=length)
    module = SimpleNamespace(num_key_value_groups=heads // kv_heads, is_causal=True, training=False)
    expected, _ = sdpa_attention_forward(module, q, k, v, None, scaling=16 ** -0.5)
    actual = ops.attention(q, k, v, causal=True, backend="sdpa")
    assert same_bits(actual.transpose(1, 2).contiguous(), expected)
    # A decode step: the reference passes no mask for one query.
    expected, _ = sdpa_attention_forward(module, q[:, :, -1:], k, v, None, scaling=16 ** -0.5)
    actual = ops.attention(q[:, :, -1:], k, v, causal=False, backend="sdpa")
    assert same_bits(actual.transpose(1, 2).contiguous(), expected)


@pytest.mark.parametrize("dtype", BITWISE_DTYPES)
def test_swiglu_matches_the_reference_mlp_bitwise(dtype):
    config = hf_config(hidden=64, intermediate=96)
    mlp = reference.Qwen3MLP(config).to(dtype)
    weights = [(0.1 * activations(tuple(p.shape), 10 + i)).to(dtype) for i, p in enumerate(
        (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight))]
    x = activations((2, 33, 64), 9).to(dtype)
    with torch.no_grad():
        for parameter, value in zip((mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight), weights, strict=True):
            parameter.copy_(value)
        expected = mlp(x)
    assert same_bits(ops.swiglu(x, *weights), expected)


# ---------------------------------------------------------------- fp64 and fp32 against independent formulas

@pytest.mark.parametrize("dtype", BOUNDED)
@pytest.mark.parametrize(("shape", "eps"), [((2, 64, 2048), 1e-6), ((1, 33, 8, 128), 1e-6), ((1, 5, 12), 1e-3)])
def test_rms_norm_is_within_its_error_bound(dtype, shape, eps):
    x, weight = activations(shape, 21), norm_weight(shape[-1], 22)
    x_exact, w_exact = representable(x, dtype), representable(weight, dtype)
    actual = ops.rms_norm(torch.from_numpy(x_exact).to(dtype), torch.from_numpy(w_exact).to(dtype), eps)
    check_bounded(actual, lambda p: bd.rms_norm(bd.exact(x_exact), w_exact, eps, p), dtype)


@pytest.mark.parametrize("dtype", BOUNDED)
@pytest.mark.parametrize(("head_dim", "theta", "positions"), [(128, 1e6, 40960), (8, 1e4, 300)])
def test_rotary_tables_are_within_their_error_bound_at_every_position(dtype, head_dim, theta, positions):
    cos, sin = ops.rotary_tables(torch.arange(positions), head_dim, theta, dtype)
    half = head_dim // 2
    assert torch.equal(cos[:, :half], cos[:, half:]) and torch.equal(sin[:, :half], sin[:, half:])
    tables = {p: bd.rotary_tables(np.arange(positions), head_dim, theta, p) for p in BOUNDED[dtype]}
    check_bounded(cos[:, :half], lambda p: tables[p][0], dtype)
    check_bounded(sin[:, :half], lambda p: tables[p][1], dtype)


@pytest.mark.parametrize("dtype", BOUNDED)
def test_apply_rotary_is_within_its_error_bound(dtype):
    x = representable(activations((1, 4, 300, 128), 23), dtype)
    positions = np.arange(40660, 40960)  # long positions, where the angle error is largest
    cos, sin = ops.rotary_tables(torch.from_numpy(positions), 128, 1e6, dtype)
    actual = ops.apply_rotary(torch.from_numpy(x).to(dtype), cos, sin)
    def independent(p):
        cos_half, sin_half = bd.rotary_tables(positions, 128, 1e6, p)
        return bd.apply_rotary(bd.exact(x), cos_half, sin_half, p)
    check_bounded(actual, independent, dtype)


POSITIONS = {
    "prefill": (np.arange(257), np.arange(257)),
    "single": (np.arange(1), np.arange(1)),
    "decode": (np.array([299]), np.arange(300)),  # one query after every key
}


@pytest.mark.parametrize("dtype", BOUNDED)
@pytest.mark.parametrize("backend", ops.BACKENDS)
@pytest.mark.parametrize(("heads", "kv_heads"), [(4, 4), (4, 2), (8, 2)])
@pytest.mark.parametrize("case", POSITIONS)
def test_attention_is_within_its_error_bound(dtype, backend, heads, kv_heads, case):
    q_positions, k_positions = POSITIONS[case]
    q = representable(activations((1, heads, len(q_positions), 128), 31), dtype)
    k = representable(activations((1, kv_heads, len(k_positions), 128), 32), dtype)
    v = representable(activations((1, kv_heads, len(k_positions), 128), 33), dtype)
    actual = ops.attention(*(torch.from_numpy(t).to(dtype) for t in (q, k, v)), causal=case != "decode", backend=backend)
    check_bounded(actual, lambda p: bd.attention(bd.exact(q), bd.exact(k), bd.exact(v), q_positions, k_positions,
                                                 p, fused=backend == "sdpa"), dtype)


@pytest.mark.parametrize("dtype", BOUNDED)
def test_swiglu_is_within_its_error_bound(dtype):
    x = representable(activations((1, 32, 2048), 41), dtype)
    gate, up, down = (representable(0.02 * activations(shape, seed), dtype) for shape, seed in
                      (((6144, 2048), 42), ((6144, 2048), 43), ((2048, 6144), 44)))
    actual = ops.swiglu(*(torch.from_numpy(t).to(dtype) for t in (x, gate, up, down)))
    check_bounded(actual, lambda p: bd.swiglu(bd.exact(x), gate, up, down, p), dtype)


# ---------------------------------------------------------------- targeted

@pytest.mark.parametrize("backend", ops.BACKENDS)
@pytest.mark.parametrize(("heads", "kv_heads"), [(8, 2), (4, 4), (6, 3)])
def test_query_head_j_reads_only_kv_head_j_over_groups(backend, heads, kv_heads):
    q, k, v = attention_inputs(torch.float32, heads=heads, kv_heads=kv_heads, length=20)
    base = ops.attention(q, k, v, causal=True, backend=backend)
    groups = heads // kv_heads
    for kv in range(kv_heads):
        k2, v2 = k.clone(), v.clone()
        k2[:, kv] += 1.0
        v2[:, kv] += 1.0
        changed = ops.attention(q, k2, v2, causal=True, backend=backend)
        moved = [j for j in range(heads) if not torch.equal(changed[:, j], base[:, j])]
        assert moved == [j for j in range(heads) if j // groups == kv]


@pytest.mark.parametrize("backend", ops.BACKENDS)
def test_a_query_never_reads_a_later_key(backend):
    q, k, v = attention_inputs(torch.float32, heads=4, kv_heads=2, length=40)
    base = ops.attention(q, k, v, causal=True, backend=backend)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 25:] = 100.0
    v2[:, :, 25:] = -100.0
    changed = ops.attention(q, k2, v2, causal=True, backend=backend)
    assert torch.equal(changed[:, :, :25], base[:, :, :25])
    assert not torch.equal(changed[:, :, 25], base[:, :, 25])  # position 25 reads itself
