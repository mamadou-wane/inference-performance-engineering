"""Independent numpy formulas for the Qwen3 operators, each carrying a running error bound.

A Bounded holds a value computed in numpy fp64 and an elementwise bound on how far an evaluation
of the same formula at unit roundoff u can be from the formula's exact value on the same inputs
(Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., ch. 3). Each rule bounds every
algebraically equivalent form the tests compare: where ipe.ops and these formulas round a
different number of times (x*rsqrt(m) against x/sqrt(m), division against multiplication by a
reciprocal), the rule allows the larger count. The production result and the independent value
then each lie within the bound of the exact value, so they agree within twice the bound.

The rules are rigorous except that local roundings are charged at the computed value rather than
the exact one, a second-order difference (relative size about u). Elementary functions are
charged their declared error: DECLARED_ULPS ulps of the result plus the smallest normal number,
which covers results flushed from the subnormal range. Those declarations are not guarantees:
test_elementary_function_accuracy measures them against mpmath over the sampled domains only.

The formulas are written from the model's mathematical definition without ipe code: the GQA
mapping as a loop over query heads, rotary as a rotation of index pairs (j, j + D/2), the causal
mask as a comparison of positions, RMSNorm as x / sqrt(mean(x^2) + eps).
"""

from dataclasses import dataclass

import numpy as np

# Declared before any measurement; the accuracy test fails rather than raising them.
DECLARED_ULPS = {"exp": 2.0, "cos": 2.0, "sin": 2.0, "pow": 2.0, "rsqrt": 2.0, "sqrt": 0.5, "silu": 4.0}


@dataclass(frozen=True)
class Precision:
    name: str
    u: float  # unit roundoff
    tiny: float  # smallest positive subnormal: the absolute floor of one rounding
    normal: float  # smallest positive normal number
    largest: float  # largest finite number


FP64 = Precision("fp64", 2.0**-53, 2.0**-1074, 2.0**-1022, float(np.finfo(np.float64).max))
FP32 = Precision("fp32", 2.0**-24, 2.0**-149, 2.0**-126, float(np.finfo(np.float32).max))


def silu_flush(p: Precision) -> float:
    """Largest |silu(x)| where exp(-x) overflows: x / (1 + exp(-x)) then returns zero.

    Measured for torch's fp32 silu (test_bounds.py); a characterized exception to DECLARED_ULPS.
    """
    return float(np.log(p.largest) / p.largest)


@dataclass(frozen=True)
class Bounded:
    value: np.ndarray
    bound: np.ndarray


def gamma(k: int, p: Precision) -> float:
    return k * p.u / (1 - k * p.u)


def _rounding(value, p: Precision, count: float = 1.0):
    """count roundings of a result of this size."""
    return count * (p.u / (1 - p.u) * np.abs(value) + p.tiny)


def _ulps(value, p: Precision, name: str):
    """The declared error of a library function; one ulp of a normal result is at most 2u|v|."""
    return DECLARED_ULPS[name] * 2 * p.u * np.abs(value) + p.normal


def exact(x) -> Bounded:
    value = np.asarray(x, dtype=np.float64)
    return Bounded(value, np.zeros_like(value))


def constant(c: float, p: Precision) -> Bounded:
    """A Python float constant; at fp32 the bound is its exact fp32 representation error."""
    error = 0.0 if p is FP64 else abs(float(np.float32(c)) - c)
    return Bounded(np.float64(c), np.float64(error))


def add(a: Bounded, b: Bounded, p: Precision) -> Bounded:
    value = a.value + b.value
    return Bounded(value, a.bound + b.bound + _rounding(value, p))


def sub(a: Bounded, b: Bounded, p: Precision) -> Bounded:
    value = a.value - b.value
    return Bounded(value, a.bound + b.bound + _rounding(value, p))


def mul(a: Bounded, b: Bounded, p: Precision, roundings: float = 1.0) -> Bounded:
    value = a.value * b.value
    bound = np.abs(a.value) * b.bound + np.abs(b.value) * a.bound + a.bound * b.bound
    return Bounded(value, bound + _rounding(value, p, roundings))


def div(a: Bounded, b: Bounded, p: Precision, roundings: float = 1.0) -> Bounded:
    assert (b.bound < np.abs(b.value)).all()
    value = a.value / b.value
    bound = (np.abs(a.value) * b.bound + np.abs(b.value) * a.bound) / (np.abs(b.value) * (np.abs(b.value) - b.bound))
    return Bounded(value, bound + _rounding(value, p, roundings))


def matmul(a: Bounded, b: Bounded, p: Precision) -> Bounded:
    """a [..., M, K] @ b [..., K, N] in any summation order (gamma_K covers blocking and FMA)."""
    k = a.value.shape[-1]
    abs_a, abs_b = np.abs(a.value), np.abs(b.value)
    bound = abs_a @ b.bound + a.bound @ abs_b + a.bound @ b.bound
    bound = bound + gamma(k, p) * (abs_a @ abs_b) + k * p.tiny
    return Bounded(a.value @ b.value, bound)


def transpose(a: Bounded, *axes) -> Bounded:
    return Bounded(np.transpose(a.value, axes), np.transpose(a.bound, axes))


def linear(x: Bounded, weight: np.ndarray, p: Precision) -> Bounded:
    """x [..., K] times weight [N, K] transposed; the weight is exact."""
    return matmul(x, exact(weight.T), p)


def mean_last(a: Bounded, p: Precision) -> Bounded:
    n = a.value.shape[-1]
    total = Bounded(a.value.sum(-1, keepdims=True),
                    a.bound.sum(-1, keepdims=True) + gamma(n, p) * np.abs(a.value).sum(-1, keepdims=True) + n * p.tiny)
    return div(total, exact(np.float64(n)), p, roundings=2)  # covers sum * (1 / n)


def exp(a: Bounded, p: Precision) -> Bounded:
    value = np.exp(a.value)
    # Mean value theorem: exp grows by at most exp(x + bound) - exp(x) over the interval.
    return Bounded(value, value * np.expm1(a.bound) + _ulps(value, p, "exp"))


def cos(a: Bounded, p: Precision) -> Bounded:
    value = np.cos(a.value)
    return Bounded(value, a.bound + _ulps(value, p, "cos"))


def sin(a: Bounded, p: Precision) -> Bounded:
    value = np.sin(a.value)
    return Bounded(value, a.bound + _ulps(value, p, "sin"))


def normalize(x: Bounded, m: Bounded, p: Precision) -> Bounded:
    """x / sqrt(m), with m > 0. Covers x*rsqrt(m) (rsqrt, then a product) and x/sqrt(m)."""
    assert (m.bound < m.value).all()
    root = np.sqrt(m.value)
    value = x.value / root
    low = m.value - m.bound
    propagated = x.bound / np.sqrt(low) + np.abs(x.value) * 0.5 * low**-1.5 * m.bound
    local = max(DECLARED_ULPS["rsqrt"] * 2 * p.u + p.u, DECLARED_ULPS["sqrt"] * 2 * p.u + p.u)
    return Bounded(value, propagated + local / (1 - p.u) * np.abs(value) + 2 * p.tiny)


def silu(a: Bounded, p: Precision) -> Bounded:
    """x / (1 + exp(-x)). Covers a library silu (declared ulps, or zero where exp(-x) overflows)
    and this composite form."""
    value = a.value / (1 + np.exp(-a.value))
    composite = (2 * DECLARED_ULPS["exp"] + 2) * p.u * np.abs(value) + 3 * p.normal
    # |silu'(x)| <= 1.0999 for every real x.
    return Bounded(value, 1.1 * a.bound + np.maximum(composite, _ulps(value, p, "silu")) + silu_flush(p))


def softmax_rows(scores: Bounded, visible: np.ndarray, p: Precision, rescales: int = 0) -> Bounded:
    """Softmax over the last axis where visible; invisible entries are exactly 0 in every form.

    Covers exp(s - max) / sum and multiplication by 1 / sum. rescales > 0 adds an online softmax's
    running rescaling (an assumption about fused kernels; see attention).
    """
    masked = np.where(visible, scores.value, -np.inf)
    top = masked.max(-1, keepdims=True)
    top_bound = np.where(visible, scores.bound, 0.0).max(-1, keepdims=True)
    shifted = sub(Bounded(np.where(visible, scores.value, top), scores.bound), Bounded(top, top_bound), p)
    e = exp(shifted, p)
    e = Bounded(np.where(visible, e.value, 0.0), np.where(visible, e.bound, 0.0))
    n = visible.shape[-1]
    total = Bounded(e.value.sum(-1, keepdims=True),
                    e.bound.sum(-1, keepdims=True) + gamma(n, p) * e.value.sum(-1, keepdims=True) + n * p.tiny)
    probabilities = div(e, total, p, roundings=2)
    if rescales:
        extra = rescales * (2 * DECLARED_ULPS["exp"] + 2) * p.u * probabilities.value
        probabilities = Bounded(probabilities.value, probabilities.bound + extra)
    return Bounded(np.where(visible, probabilities.value, 0.0), np.where(visible, probabilities.bound, 0.0))


# ---------------------------------------------------------------- Qwen3 formulas

def rms_norm(x: Bounded, weight: np.ndarray, eps: float, p: Precision) -> Bounded:
    """weight * x / sqrt(mean(x^2) + eps) over the last axis."""
    m = add(mean_last(mul(x, x, p), p), constant(eps, p), p)
    return mul(exact(weight), normalize(x, m, p), p)


def rotary_tables(positions: np.ndarray, head_dim: int, theta: float, p: Precision) -> tuple[Bounded, Bounded]:
    """cos and sin [T, D/2] for angle p * theta^(-2j/D). Covers 1 / theta^(2j/D) (one more rounding)."""
    exponents = np.arange(0, head_dim, 2, dtype=np.float64) / head_dim
    assert (exponents * head_dim == np.arange(0, head_dim, 2)).all()  # exact exponents
    frequencies = np.float64(theta) ** -exponents
    frequencies = Bounded(frequencies, _ulps(frequencies, p, "pow") + _rounding(frequencies, p)
                          + constant(theta, p).bound * exponents * frequencies / theta)
    angles = mul(exact(positions.astype(np.float64)[:, None]), Bounded(frequencies.value[None], frequencies.bound[None]), p)
    return cos(angles, p), sin(angles, p)


def apply_rotary(x: Bounded, cos_half: Bounded, sin_half: Bounded, p: Precision) -> Bounded:
    """x [..., T, D]: each pair (j, j + D/2) rotated by the angle of frequency j."""
    half = x.value.shape[-1] // 2
    first = Bounded(x.value[..., :half], x.bound[..., :half])
    second = Bounded(x.value[..., half:], x.bound[..., half:])
    rotated_first = sub(mul(first, cos_half, p), mul(second, sin_half, p), p)
    rotated_second = add(mul(second, cos_half, p), mul(first, sin_half, p), p)
    return Bounded(np.concatenate((rotated_first.value, rotated_second.value), -1),
                   np.concatenate((rotated_first.bound, rotated_second.bound), -1))


def attention(q: Bounded, k: Bounded, v: Bounded, q_positions: np.ndarray, k_positions: np.ndarray,
              p: Precision, *, fused: bool = False) -> Bounded:
    """q [B, Hq, T, D]; k, v [B, Hkv, S, D]. Query head j attends with KV head j // (Hq / Hkv).

    fused=True also covers a fused kernel such as PyTorch's CPU flash attention, under two
    assumptions not read from its source: the scale may be applied to q before the product, and
    the softmax may be computed online with at most S rescalings of each running sum.
    """
    heads, kv_heads, dim = q.value.shape[1], k.value.shape[1], q.value.shape[-1]
    groups = heads // kv_heads
    # D ** -0.5 as a Python float is within one fp64 ulp of 1/sqrt(D); at fp32 it is rounded again.
    scale = constant(1.0 / np.sqrt(dim), p)
    scale = Bounded(scale.value, scale.bound + 2 * FP64.u * scale.value)
    visible = k_positions[None, :] <= q_positions[:, None]
    outputs = []
    for head in range(heads):
        kv = head // groups
        qh = Bounded(q.value[:, head], q.bound[:, head])
        kh = Bounded(k.value[:, kv], k.bound[:, kv])
        vh = Bounded(v.value[:, kv], v.bound[:, kv])
        scores = mul(matmul(qh, transpose(kh, 0, 2, 1), p), scale, p)
        if fused:
            prescaled = p.u * scale.value * (np.abs(qh.value) @ np.abs(np.transpose(kh.value, (0, 2, 1))))
            scores = Bounded(scores.value, scores.bound + prescaled)
        probabilities = softmax_rows(scores, np.broadcast_to(visible, scores.value.shape), p,
                                     rescales=visible.shape[-1] if fused else 0)
        outputs.append(matmul(probabilities, vh, p))
    return Bounded(np.stack([o.value for o in outputs], 1), np.stack([o.bound for o in outputs], 1))


def swiglu(x: Bounded, gate: np.ndarray, up: np.ndarray, down: np.ndarray, p: Precision) -> Bounded:
    return linear(mul(silu(linear(x, gate, p), p), linear(x, up, p), p), down, p)


def decoder_layer(x: Bounded, weights: dict, positions: np.ndarray, config, p: Precision) -> Bounded:
    """One layer; weights maps the checkpoint suffixes (input_layernorm.weight, ...) to arrays."""
    batch, length, _ = x.value.shape
    heads, kv_heads, dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    eps = config.rms_norm_eps

    def split(t: Bounded, count: int) -> Bounded:  # [B, T, count * D] -> [B, count, T, D]
        return transpose(Bounded(t.value.reshape(batch, length, count, dim), t.bound.reshape(batch, length, count, dim)), 0, 2, 1, 3)

    h = rms_norm(x, weights["input_layernorm.weight"], eps, p)
    q = split(linear(h, weights["self_attn.q_proj.weight"], p), heads)
    k = split(linear(h, weights["self_attn.k_proj.weight"], p), kv_heads)
    v = split(linear(h, weights["self_attn.v_proj.weight"], p), kv_heads)
    q = rms_norm(q, weights["self_attn.q_norm.weight"], eps, p)
    k = rms_norm(k, weights["self_attn.k_norm.weight"], eps, p)
    cos_half, sin_half = rotary_tables(positions, dim, config.rope_theta, p)
    q, k = apply_rotary(q, cos_half, sin_half, p), apply_rotary(k, cos_half, sin_half, p)
    attended = transpose(attention(q, k, v, positions, positions, p), 0, 2, 1, 3)
    attended = Bounded(attended.value.reshape(batch, length, -1), attended.bound.reshape(batch, length, -1))
    x = add(x, linear(attended, weights["self_attn.o_proj.weight"], p), p)
    h = rms_norm(x, weights["post_attention_layernorm.weight"], eps, p)
    mlp = swiglu(h, weights["mlp.gate_proj.weight"], weights["mlp.up_proj.weight"], weights["mlp.down_proj.weight"], p)
    return add(x, mlp, p)


def ratio(actual, expected: Bounded, factor: float) -> float:
    """max |actual - value| / (factor * bound); a pass is <= 1. Zero bounds demand equality."""
    actual = np.asarray(actual, dtype=np.float64)
    difference = np.abs(actual - expected.value)
    limit = factor * expected.bound
    if not (np.isfinite(difference).all() and np.isfinite(limit).all()) or ((limit == 0) & (difference != 0)).any():
        return float("inf")
    return float(np.max(np.where(limit > 0, difference / np.where(limit > 0, limit, 1.0), 0.0)))
