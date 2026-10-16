"""Qwen3 decoder operators as pure functions over tensors.

At bf16 and fp32 each function performs Transformers 5.17.0's operations and casts in its order,
so results are bitwise comparable with the reference; equivalent rewrites (one rounding instead
of two, fused rotary) change bf16 results. At fp64 every step is fp64, including the norm
statistics, rotary angles and softmax that the reference keeps in fp32.

Tensors are batch first; attention tensors are [B, heads, T, D]; weights are [out, in] in the
activations' dtype. Arguments are not checked here; ipe.forward validates at its boundary.
"""

import torch
import torch.nn.functional as F

BACKENDS = ("math", "sdpa")


def _statistics_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp32 for bf16 and fp32 activations, as the reference; fp64 for the oracle."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last axis: weight * (x / sqrt(mean(x^2) + eps)).

    The normalized value is cast to x.dtype before the weight multiply, so a bf16 run rounds
    twice, as the reference does. One rounding would be slightly more accurate but changes about
    a quarter of bf16 outputs by one ulp.
    """
    normalized = x.to(_statistics_dtype(x.dtype))
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    return weight * normalized.to(x.dtype)


def qk_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm of queries or keys, [B, T, heads, D] over D, applied before rotary."""
    return rms_norm(x, weight, eps)


def rotary_tables(positions: torch.Tensor, head_dim: int, theta: float,
                  dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin [T, head_dim] for int64 positions [T], split-half layout, cast to dtype.

    Computed in fp32 by the reference formula for bf16 and fp32 runs, and in fp64 for fp64. The
    fp32 angles drift from exact linearly with position (about 1.8e-4 rad near 4,096).
    """
    compute = _statistics_dtype(dtype)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=compute) / head_dim))
    # One rounded product per angle: the reference's K = 1 matmul rounds the same way.
    angles = positions.to(compute)[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x [B, H, T, D] by tables [T, D] in x.dtype: three roundings, no fused multiply-add."""
    return (x * cos) + (_rotate_half(x) * sin)


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    """[B, Hkv, S, D] -> [B, Hkv * groups, S, D]; query head j reads KV head j // groups.

    Materialized as the reference does: a grouped reshape without the copy is not bitwise equal
    at bf16 for longer sequences on the CPU.
    """
    if groups == 1:
        return x
    batch, heads, length, dim = x.shape
    return x[:, :, None].expand(batch, heads, groups, length, dim).reshape(batch, heads * groups, length, dim)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool,
              backend: str) -> torch.Tensor:
    """Scaled dot-product attention; q [B, Hq, T, D], k and v [B, Hkv, S, D] -> [B, Hq, T, D].

    causal: T == S and query i sees keys 0 to i. Otherwise every query sees every key (a decode
    step). The mask comes from this host flag, so nothing reads device values.

    "math" is the reference's eager attention: additive finfo.min mask, softmax in fp32 (fp64 for
    fp64), probabilities cast back. "sdpa" passes is_causal and enable_gqa without a mask, as the
    reference does, which keeps flash attention eligible on CUDA; it matches "math" to rounding.
    """
    groups = q.shape[1] // k.shape[1]
    scale = q.shape[-1] ** -0.5
    if backend == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal and q.shape[2] > 1, scale=scale,
                                              enable_gqa=groups > 1)
    k, v = _repeat_kv(k, groups), _repeat_kv(v, groups)
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    if causal:
        length = q.shape[2]
        scores = scores + torch.full((length, length), torch.finfo(q.dtype).min, dtype=q.dtype,
                                     device=q.device).triu(1)
    probabilities = F.softmax(scores, dim=-1, dtype=_statistics_dtype(q.dtype)).to(q.dtype)
    return torch.matmul(probabilities, v)


def swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """down(silu(gate(x)) * up(x)) for x [B, T, H]; silu and the product each round once."""
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
