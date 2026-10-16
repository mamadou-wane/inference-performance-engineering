"""Validation of the error-bound rules in bounded.py against exact values from mpmath.

Two kinds of evidence, both empirical over the sampled inputs:
- Library elementary functions (torch fp64 and fp32, numpy fp64) stay within DECLARED_ULPS ulps
  of the exact result plus the smallest normal number (results in the subnormal range may be
  flushed to zero). This is a measurement over the sampled domains, not a guarantee
  for other arguments, other builds or other CPUs.
- On small shapes, the independent formulas and ipe.ops each lie within the running bound of the
  exact value of the formula, computed in mpmath at 50 significant digits. This checks the bound
  rules themselves before they are used to judge the full-size tests.
"""

import bounded as bd
import mpmath
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mpmath import mp, mpf

from ipe import ops
from ipe.config import Qwen3Config

mp.dps = 50
SAMPLES = 2000


def sample(kind, count, seed):
    rng = np.random.default_rng(seed)
    if kind == "exp":
        return np.concatenate((rng.uniform(-745, 0, count // 2), rng.uniform(-60, 60, count - count // 2)))
    if kind == "angle":
        return np.concatenate((rng.uniform(0, 40960, count // 2), rng.uniform(0, 8, count - count // 2)))
    if kind == "rsqrt":
        return 10 ** rng.uniform(-7, 8, count)
    if kind == "silu":
        return rng.uniform(-100, 100, count)
    raise ValueError(kind)


FORMATS = {np.float64: (53, -1022), np.float32: (24, -126)}  # precision, smallest normal exponent


def ulp(exact_value, dtype):
    """The spacing of dtype in the binade of the exact value (subnormal spacing below it)."""
    precision, minimum = FORMATS[dtype]
    magnitude = abs(exact_value)
    if magnitude == 0:
        return mpmath.ldexp(1, minimum - precision + 1)
    exponent = int(mpmath.floor(mpmath.log(magnitude, 2)))
    exponent += 1 if mpmath.ldexp(1, exponent + 1) <= magnitude else -1 if mpmath.ldexp(1, exponent) > magnitude else 0
    return mpmath.ldexp(1, max(exponent, minimum) - precision + 1)


def ulp_errors(computed: np.ndarray, exact_values, dtype, floor=None) -> float:
    """Largest error in ulps of the exact value, after an absolute allowance (default: the smallest
    normal number)."""
    floor = mpmath.ldexp(1, FORMATS[dtype][1]) if floor is None else mpf(floor)
    worst = 0.0
    for value, exact_value in zip(computed.tolist(), exact_values, strict=True):
        excess = max(abs(mpf(value) - exact_value) - floor, 0)
        worst = max(worst, float(excess / ulp(exact_value, dtype)))
    return worst


EXACT = {
    "exp": mpmath.exp, "cos": mpmath.cos, "sin": mpmath.sin,
    "rsqrt": lambda x: 1 / mpmath.sqrt(x), "sqrt": mpmath.sqrt, "silu": lambda x: x / (1 + mpmath.exp(-x)),
}
TORCH = {"exp": torch.exp, "cos": torch.cos, "sin": torch.sin, "rsqrt": torch.rsqrt, "silu": F.silu}
NUMPY = {"exp": np.exp, "cos": np.cos, "sin": np.sin, "rsqrt": lambda x: 1 / np.sqrt(x), "sqrt": np.sqrt}
DOMAIN = {"exp": "exp", "cos": "angle", "sin": "angle", "rsqrt": "rsqrt", "sqrt": "rsqrt", "silu": "silu"}
CASES = [("torch", n, d) for n in TORCH for d in (torch.float64, torch.float32)] + [("numpy", n, torch.float64) for n in NUMPY]


@pytest.mark.parametrize(("library", "name", "dtype"), CASES)
def test_elementary_function_accuracy(library, name, dtype):
    x = torch.from_numpy(sample(DOMAIN[name], SAMPLES, 1)).to(dtype)
    computed = (TORCH[name](x) if library == "torch" else torch.from_numpy(NUMPY[name](x.numpy()))).double().numpy()
    exact_values = [EXACT[name](mpf(value)) for value in x.double().tolist()]
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
    if name == "silu":  # zero where exp(-x) overflows; characterized in the next test
        overflow = x.double().numpy() < -np.log(np.finfo(numpy_dtype).max)
        computed, exact_values = computed[~overflow], [e for e, o in zip(exact_values, overflow, strict=True) if not o]
    assert ulp_errors(computed, exact_values, numpy_dtype) <= bd.DECLARED_ULPS[name]


@pytest.mark.parametrize(("dtype", "precision"), [(torch.float32, bd.FP32), (torch.float64, bd.FP64)])
def test_silu_returns_zero_where_exp_overflows(dtype, precision):
    """Characterized exception: below -log(largest), exp(-x) overflows and silu returns -0.

    The error there is the whole result, at most log(largest) / largest (2.6e-37 at fp32), which
    bounded.silu adds as an absolute term."""
    limit = float(np.log(np.finfo(np.float32 if dtype == torch.float32 else np.float64).max))
    x = torch.linspace(-limit - 20, -limit - 1e-3 * limit, 4000, dtype=torch.float64).to(dtype)
    x = x[x.double() < -limit]
    computed = F.silu(x).double().numpy()
    exact_values = [EXACT["silu"](mpf(value)) for value in x.double().tolist()]
    assert ulp_errors(computed, exact_values, np.float64, floor=bd.silu_flush(precision)) == 0


@pytest.mark.parametrize("dtype", (torch.float64, torch.float32))
@pytest.mark.parametrize("theta", (1e6, 1e4))
def test_pow_accuracy_for_rotary_exponents(dtype, theta):
    exponents = torch.cat((torch.arange(0, 128, 2, dtype=torch.float64) / 128,
                           torch.from_numpy(np.random.default_rng(2).uniform(0, 1, 500)))).to(dtype)
    computed = (theta ** exponents).double().numpy()
    exact_values = [mpmath.power(mpf(theta), mpf(e)) for e in exponents.double().tolist()]
    assert ulp_errors(computed, exact_values, np.float64 if dtype == torch.float64 else np.float32) <= bd.DECLARED_ULPS["pow"]
    exact_numpy = [mpmath.power(mpf(theta), -mpf(e)) for e in exponents.double().tolist()]
    if dtype == torch.float64:
        assert ulp_errors(np.float64(theta) ** -exponents.numpy(), exact_numpy, np.float64) <= bd.DECLARED_ULPS["pow"]


# ---------------------------------------------------------------- exact formulas in mpmath

def mp_rms_norm(x, weight, eps):
    """x: nested lists over the last axis."""
    if isinstance(x[0], list):
        return [mp_rms_norm(row, weight, eps) for row in x]
    r = 1 / mpmath.sqrt(mpmath.fsum(v * v for v in x) / len(x) + mpf(eps))
    return [w * v * r for w, v in zip(weight, x, strict=True)]


def mp_tables(positions, head_dim, theta):
    half = head_dim // 2
    frequencies = [mpmath.power(mpf(theta), -mpf(2 * j) / head_dim) for j in range(half)]
    return ([[mpmath.cos(p * f) for f in frequencies] for p in positions],
            [[mpmath.sin(p * f) for f in frequencies] for p in positions])


def mp_rotate(x, cos, sin):
    """x [T][D] for one head."""
    half = len(x[0]) // 2
    out = []
    for row, c, s in zip(x, cos, sin, strict=True):
        a, b = row[:half], row[half:]
        out.append([a[j] * c[j] - b[j] * s[j] for j in range(half)] + [b[j] * c[j] + a[j] * s[j] for j in range(half)])
    return out


def mp_attention(q, k, v, q_positions, k_positions):
    """q [Hq][T][D]; k, v [Hkv][S][D]."""
    groups = len(q) // len(k)
    scale = 1 / mpmath.sqrt(len(q[0][0]))
    out = []
    for head, qh in enumerate(q):
        kh, vh = k[head // groups], v[head // groups]
        rows = []
        for qi, query in zip(q_positions, qh, strict=True):
            keys = [j for j, kj in enumerate(k_positions) if kj <= qi]
            scores = [mpmath.fsum(a * b for a, b in zip(query, kh[j], strict=True)) * scale for j in keys]
            top = max(scores)
            weights = [mpmath.exp(s - top) for s in scores]
            total = mpmath.fsum(weights)
            rows.append([mpmath.fsum(w * vh[j][d] for w, j in zip(weights, keys, strict=True)) / total for d in range(len(vh[0]))])
        out.append(rows)
    return out


def mp_linear(x, weight):
    return [mp_linear(row, weight) for row in x] if isinstance(x[0], list) else [
        mpmath.fsum(a * b for a, b in zip(x, w, strict=True)) for w in weight]


def mp_swiglu(x, gate, up, down):
    g, u = mp_linear(x, gate), mp_linear(x, up)
    hidden = [[gi / (1 + mpmath.exp(-gi)) * ui for gi, ui in zip(gr, ur, strict=True)] for gr, ur in zip(g, u, strict=True)]
    return mp_linear(hidden, down)


def to_mp(a: np.ndarray):
    return [to_mp(row) for row in a] if a.ndim > 1 else [mpf(float(v)) for v in a]


def within(actual: np.ndarray, exact_values, bound: np.ndarray) -> float:
    """Largest |actual - exact| / bound, in mpmath; a pass is <= 1."""
    worst = 0.0
    flat_exact = np.array(exact_values, dtype=object).reshape(-1)
    for value, exact_value, limit in zip(actual.reshape(-1).tolist(), flat_exact, bound.reshape(-1).tolist(), strict=True):
        difference = abs(mpf(value) - exact_value)
        if difference:
            worst = max(worst, float(difference / mpf(limit)) if limit > 0 else float("inf"))
    return worst


def representable(shape, seed, scale=1.0, *, offset=0.0, small_row=False):
    """Values exact at fp32 and fp64, so both production runs see the inputs the rules assume."""
    values = offset + np.random.default_rng(seed).normal(0, scale, shape)
    values.reshape(-1, shape[-1])[0, 0] = offset + 20 * scale
    if small_row:
        values.reshape(-1, shape[-1])[-1] *= 1e-4
    return values.astype(np.float32).astype(np.float64)


def check_all(production, independent, exact_values):
    """ipe.ops at fp64 and fp32, and the independent value, each within their bound of exact."""
    for p, dtype in ((bd.FP64, torch.float64), (bd.FP32, torch.float32)):
        result = independent(p)
        assert within(production(dtype), exact_values, result.bound) <= 1.0, p.name
        if p is bd.FP64:
            assert within(result.value, exact_values, result.bound) <= 1.0


def t(a, dtype):
    return torch.from_numpy(a).to(dtype)


def test_rms_norm_bound_holds_against_exact_values():
    x, weight = representable((2, 3, 16), 1, small_row=True), representable((16,), 2, 0.2, offset=1.0)
    exact_values = mp_rms_norm(to_mp(x), to_mp(weight), 1e-6)
    check_all(lambda d: ops.rms_norm(t(x, d), t(weight, d), 1e-6).double().numpy(),
              lambda p: bd.rms_norm(bd.exact(x), weight, 1e-6, p), exact_values)


def test_rotary_bound_holds_against_exact_values():
    positions = np.array([0, 1, 2, 3, 4095, 40958, 40959])
    x = representable((1, 2, len(positions), 8), 3)
    cos, sin = mp_tables(positions.tolist(), 8, 1e6)
    exact_values = [[mp_rotate(head, cos, sin) for head in batch] for batch in to_mp(x)]
    def production(dtype):
        c, s = ops.rotary_tables(torch.from_numpy(positions), 8, 1e6, dtype)
        return ops.apply_rotary(t(x, dtype), c, s).double().numpy()
    def independent(p):
        c, s = bd.rotary_tables(positions, 8, 1e6, p)
        return bd.apply_rotary(bd.exact(x), c, s, p)
    check_all(production, independent, exact_values)


@pytest.mark.parametrize("backend", ops.BACKENDS)
@pytest.mark.parametrize(("q_positions", "k_positions"), [(list(range(6)), list(range(6))), ([5], list(range(6)))])
def test_attention_bound_holds_against_exact_values(backend, q_positions, k_positions):
    q = representable((1, 4, len(q_positions), 8), 4, 2.0)
    k, v = representable((1, 2, len(k_positions), 8), 5, 2.0), representable((1, 2, len(k_positions), 8), 6)
    exact_values = [mp_attention(to_mp(q[0]), to_mp(k[0]), to_mp(v[0]), q_positions, k_positions)]
    qp, kp = np.array(q_positions), np.array(k_positions)
    check_all(lambda d: ops.attention(t(q, d), t(k, d), t(v, d), causal=q_positions == k_positions,
                                      backend=backend).double().numpy(),
              lambda p: bd.attention(bd.exact(q), bd.exact(k), bd.exact(v), qp, kp, p, fused=backend == "sdpa"),
              exact_values)


def test_swiglu_bound_holds_against_exact_values():
    x = representable((1, 4, 16), 7)
    gate, up, down = representable((24, 16), 8, 0.3), representable((24, 16), 9, 0.3), representable((16, 24), 10, 0.3)
    exact_values = [mp_swiglu(to_mp(x[0]), to_mp(gate), to_mp(up), to_mp(down))]
    check_all(lambda d: ops.swiglu(t(x, d), t(gate, d), t(up, d), t(down, d)).double().numpy(),
              lambda p: bd.swiglu(bd.exact(x), gate, up, down, p), exact_values)


def test_decoder_layer_bound_holds_against_exact_values():
    from ipe.forward import decoder_layer
    from ipe.weights import LayerWeights

    config = Qwen3Config(vocab_size=8, hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=1e4)
    shapes = {"input_layernorm.weight": (16,), "self_attn.q_proj.weight": (16, 16), "self_attn.k_proj.weight": (8, 16),
              "self_attn.v_proj.weight": (8, 16), "self_attn.q_norm.weight": (8,), "self_attn.k_norm.weight": (8,),
              "self_attn.o_proj.weight": (16, 16), "post_attention_layernorm.weight": (16,),
              "mlp.gate_proj.weight": (24, 16), "mlp.up_proj.weight": (24, 16), "mlp.down_proj.weight": (16, 24)}
    weights = {name: (representable(shape, 20 + i, 0.2, offset=1.0) if len(shape) == 1 else representable(shape, 20 + i, 0.3))
               for i, (name, shape) in enumerate(shapes.items())}
    x = representable((1, 5, 16), 11)
    positions = np.arange(5)

    # Exact layer, composed from the exact operators above.
    xs = to_mp(x[0])
    h = mp_rms_norm(xs, to_mp(weights["input_layernorm.weight"]), 1e-6)
    def heads(t_, count):
        return [[row[n * 8:(n + 1) * 8] for row in t_] for n in range(count)]
    cos, sin = mp_tables(positions.tolist(), 8, 1e4)
    q = [mp_rotate(mp_rms_norm(head, to_mp(weights["self_attn.q_norm.weight"]), 1e-6), cos, sin)
         for head in heads(mp_linear(h, to_mp(weights["self_attn.q_proj.weight"])), 2)]
    k = [mp_rotate(mp_rms_norm(head, to_mp(weights["self_attn.k_norm.weight"]), 1e-6), cos, sin)
         for head in heads(mp_linear(h, to_mp(weights["self_attn.k_proj.weight"])), 1)]
    v = heads(mp_linear(h, to_mp(weights["self_attn.v_proj.weight"])), 1)
    attended = mp_attention(q, k, v, positions.tolist(), positions.tolist())
    merged = [attended[0][i] + attended[1][i] for i in range(5)]
    x1 = [[a + b for a, b in zip(r1, r2, strict=True)] for r1, r2 in zip(xs, mp_linear(merged, to_mp(weights["self_attn.o_proj.weight"])), strict=True)]
    h = mp_rms_norm(x1, to_mp(weights["post_attention_layernorm.weight"]), 1e-6)
    mlp = mp_swiglu(h, to_mp(weights["mlp.gate_proj.weight"]), to_mp(weights["mlp.up_proj.weight"]), to_mp(weights["mlp.down_proj.weight"]))
    exact_values = [[[a + b for a, b in zip(r1, r2, strict=True)] for r1, r2 in zip(x1, mlp, strict=True)]]

    fields = {"input_layernorm.weight": "input_layernorm", "self_attn.q_proj.weight": "q_proj",
              "self_attn.k_proj.weight": "k_proj", "self_attn.v_proj.weight": "v_proj",
              "self_attn.q_norm.weight": "q_norm", "self_attn.k_norm.weight": "k_norm",
              "self_attn.o_proj.weight": "o_proj", "post_attention_layernorm.weight": "post_attention_layernorm",
              "mlp.gate_proj.weight": "gate_proj", "mlp.up_proj.weight": "up_proj", "mlp.down_proj.weight": "down_proj"}
    def production(dtype):
        layer = LayerWeights(**{fields[name]: t(value, dtype) for name, value in weights.items()})
        rope = ops.rotary_tables(torch.arange(5), 8, 1e4, dtype)
        return decoder_layer(t(x, dtype), layer, rope, config, backend="math").double().numpy()
    check_all(production, lambda p: bd.decoder_layer(bd.exact(x), weights, positions, config, p), exact_values)
