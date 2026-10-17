"""RMSNorm operator contract, for every implementation: ops.rms_norm and any fused kernel.

Each output element is compared with the exact value w * x / sqrt(mean(x^2) + eps) of its
bf16- or fp32-representable inputs:
- bf16: within 1.5 bf16 ulps, the most that two roundings of a correctly computed value allow.
- fp32: within (n/2 + 5) * 2^-24 relative, which holds for statistics summed in any order.
An all-zero row must give exact zeros. The inputs include rows where epsilon dominates, rows with
massive outliers, and widths from 8 to 2048. The end-to-end accuracy rule cannot see bf16 defects
of rounding size; this contract can.
"""

import numpy as np
import pytest
import torch

from ipe import ops

EPS = 1e-6
WIDTHS = (8, 128, 1000, 2048)

# A native kernel adds itself here, wrapped to take and return CPU tensors.
IMPLEMENTATIONS = {
    "ops.rms_norm": ops.rms_norm,
    "ops.rms_norm on cuda": lambda x, weight, eps: ops.rms_norm(x.cuda(), weight.cuda(), eps).cpu(),
}
ON_GPU = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")]
NAMES = [pytest.param(name, marks=ON_GPU) if "cuda" in name else name for name in IMPLEMENTATIONS]


def inputs(width, dtype, seed=0):
    """Rows of N(0, s^2) for s from 1e-4 (epsilon dominates) to 100, outlier rows, a zero row."""
    generator = torch.Generator().manual_seed(seed)
    scales = torch.tensor([1e-4, 1e-3, 3e-3, 1e-2, 0.1, 1.0, 1.0, 1.0, 10.0, 100.0], dtype=torch.float64)
    x = torch.randn((len(scales) * 4, width), generator=generator, dtype=torch.float64) * scales.repeat(4)[:, None]
    x[-4:-1, 0] *= 300  # massive activations, as in real residual streams
    x[-2, width // 2] *= 50
    x[-1] = 0
    weight = 1 + 0.2 * torch.randn(width, generator=generator, dtype=torch.float64)
    weight[:: max(width // 8, 1)] *= 20
    return x.to(dtype), weight.to(dtype)


def exact(x, weight, eps):
    x, weight = x.double(), weight.double()
    return weight * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def violations(actual, expected, dtype, width):
    """Elements outside the contract's bound."""
    actual = actual.double()
    if dtype == torch.bfloat16:
        _, exponent = np.frexp(expected.abs().numpy())
        bound = torch.from_numpy(1.5 * np.ldexp(1.0, exponent - 8))  # 1.5 ulp of the exact value
    else:
        bound = (width / 2 + 5) * 2.0 ** -24 * expected.abs()
    return int(((actual - expected).abs() > bound).sum()) + int((actual[expected == 0] != 0).sum())


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("width", WIDTHS)
def test_rms_norm_meets_the_contract(name, dtype, width):
    x, weight = inputs(width, dtype)
    assert violations(IMPLEMENTATIONS[name](x, weight, EPS), exact(x, weight, EPS), dtype, width) == 0


def one_rounding(x, weight, eps):
    """Weight multiply in fp32, one cast: more accurate than the reference's order."""
    h = x.float()
    return (weight.float() * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))).to(x.dtype)


def test_a_legitimate_reordering_meets_the_contract():
    for dtype in (torch.bfloat16, torch.float32):
        for width in WIDTHS:
            x, weight = inputs(width, dtype)
            assert violations(one_rounding(x, weight, EPS), exact(x, weight, EPS), dtype, width) == 0


def no_fp32_step(x, weight, eps):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def bf16_statistics(x, weight, eps):
    h = x.bfloat16()
    return (weight * (x.float() * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps).float()).to(x.dtype))


def bf16_accumulation(x, weight, eps):
    """Sum of squares accumulated left to right in bf16, as a naive fused kernel might."""
    squares, total = x.bfloat16().pow(2), torch.zeros(x.shape[:-1] + (1,), dtype=torch.bfloat16)
    for column in range(x.shape[-1]):
        total = total + squares[..., column:column + 1]
    rsqrt = torch.rsqrt(total.float() / x.shape[-1] + eps)
    return weight * (x.float() * rsqrt).to(x.dtype)


DEFECTS = {
    "epsilon 1e-5": (lambda x, weight, eps: ops.rms_norm(x, weight, 1e-5), (torch.bfloat16, torch.float32)),
    "no fp32 step": (no_fp32_step, (torch.bfloat16,)),
    "statistics in bf16": (bf16_statistics, (torch.float32,)),
    "accumulation in bf16": (bf16_accumulation, (torch.bfloat16, torch.float32)),
}


@pytest.mark.parametrize("defect", DEFECTS)
@pytest.mark.parametrize("width", (128, 2048))  # per-head QK-norm and the hidden size
def test_seeded_defects_violate_the_contract(defect, width):
    implementation, dtypes = DEFECTS[defect]
    for dtype in dtypes:
        x, weight = inputs(width, dtype)
        assert violations(implementation(x, weight, EPS), exact(x, weight, EPS), dtype, width) > 0, dtype
