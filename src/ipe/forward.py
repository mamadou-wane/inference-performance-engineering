"""The Qwen3 forward pass over loaded weights: the uncached path and the fp64 oracle.

A run has a dtype: bfloat16, float32 or float64. Weights stored in a narrower dtype are upcast at
each use, which is exact, so an fp32 or fp64 run can use the bf16 file views ipe.weights returns
and only one layer's converted copy is alive at a time. The fp64 oracle is this code at
dtype=float64; ipe.ops gives the precision of each step.

Three entry points, three outputs:
- forward_full: the engine's uncached call. fp32 logits [B, n, V] for the requested positions,
  materialized in one tensor; bf16 logits are widened, which is exact. fp64 is rejected, so the
  oracle's logits are never rounded to fp32.
- final_hidden: final-normed hidden states [B, T, H] in the run dtype; fp64 allowed.
- lm_logits: logits in the dtype of the hidden states it is given, for any subset of rows. Long
  comparisons call it per chunk of positions.

Positions are 0 to T-1 for every row. Arguments are validated before any compute and raise
ValueError. Stateless; ipe.engine runs the same decoder layer over a KV cache.
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from ipe import ops
from ipe.config import Qwen3Config
from ipe.weights import DTYPES, LayerWeights, LoadedCheckpoint

INTEGER_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)


def check_dtype(loaded: LoadedCheckpoint, dtype: torch.dtype, allowed: Sequence[torch.dtype]) -> None:
    if dtype not in allowed:
        raise ValueError(f"dtype {dtype!r}: expected one of {list(allowed)}")
    stored = loaded.weights.embed_tokens
    if stored.dtype.itemsize > dtype.itemsize:
        raise ValueError(f"dtype {dtype!r}: weights are stored in the wider {stored.dtype!r}")
    if stored.device.type == "mps" and dtype == torch.float64:
        raise ValueError("device mps: float64 is not supported")
    # TF32 rounds matmul inputs to 10 mantissa bits. This getter reports TF32 from either API;
    # the legacy one raises once the new API has been used.
    if stored.device.type == "cuda" and dtype == torch.float32 and torch.backends.cuda.matmul.fp32_precision == "tf32":
        raise ValueError("dtype float32 on cuda: TF32 matmul is enabled; set torch.backends.cuda.matmul.fp32_precision = 'ieee'")


def check_token_ids(config: Qwen3Config, token_ids: object, capacity: int) -> tuple[int, int]:
    """Require integer IDs [B, T] in the vocabulary, 1 <= T <= capacity; return (B, T).

    Syncs when the IDs are on a GPU.
    """
    if not isinstance(token_ids, torch.Tensor):
        raise ValueError(f"token_ids: expected a tensor, got {type(token_ids).__name__}")  # noqa: TRY004 - one error type for every invalid argument
    if token_ids.dtype not in INTEGER_DTYPES:
        raise ValueError(f"token_ids: expected one of {list(INTEGER_DTYPES)}, got {token_ids.dtype}")
    if token_ids.dim() != 2 or token_ids.shape[0] < 1:
        raise ValueError(f"token_ids: expected shape [B, T] with B >= 1, got {list(token_ids.shape)}")
    batch, length = token_ids.shape
    if not 1 <= length <= capacity:
        raise ValueError(f"token_ids: length {length} is outside 1 to {capacity}")
    ids = token_ids.to(torch.int64)  # a bound compared in a narrow dtype wraps
    if ids.min() < 0 or ids.max() >= config.vocab_size:
        raise ValueError(f"token_ids: IDs must lie in [0, {config.vocab_size})")
    return batch, length


def _check_backend(backend: object) -> None:
    if backend not in ops.BACKENDS:
        raise ValueError(f"backend {backend!r}: expected one of {list(ops.BACKENDS)}")


def _check_logit_positions(logit_positions: object, length: int) -> torch.Tensor:
    try:
        rows = torch.as_tensor(logit_positions)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"logit_positions: {error}") from error
    if rows.dtype not in INTEGER_DTYPES:
        raise ValueError(f"logit_positions: expected integers, got {rows.dtype}")
    if rows.dim() != 1 or rows.numel() == 0:
        raise ValueError("logit_positions: expected a non-empty 1-D sequence")
    rows = rows.to(torch.int64)
    if rows.min() < 0 or rows.max() >= length:
        raise ValueError(f"logit_positions: positions must lie in [0, {length})")
    return rows


def decoder_layer(x: torch.Tensor, layer: LayerWeights, rope: tuple[torch.Tensor, torch.Tensor],
                  config: Qwen3Config, *, backend: str,
                  cache: tuple[torch.Tensor, torch.Tensor] | None = None, start: int = 0) -> torch.Tensor:
    """One pre-norm decoder layer on x [B, T, H] at positions start to start+T-1; rope is their
    (cos, sin) [T, D].

    cache, (K, V) buffers [B, KV heads, capacity, D], receives this step's keys and values. A
    prompt (start 0) attends causally to its own keys; a decode step (T 1) to positions 0 to start.
    """
    if start > 0 and x.shape[1] > 1:
        raise ValueError("decoder_layer: a step after position 0 takes one token; chunked prefill is not supported")
    dtype, eps = x.dtype, config.rms_norm_eps
    batch, length, _ = x.shape
    heads = (batch, length, -1, config.head_dim)
    h = ops.rms_norm(x, layer.input_layernorm.to(dtype), eps)
    q = ops.qk_norm(F.linear(h, layer.q_proj.to(dtype)).view(heads), layer.q_norm.to(dtype), eps)
    k = ops.qk_norm(F.linear(h, layer.k_proj.to(dtype)).view(heads), layer.k_norm.to(dtype), eps)
    v = F.linear(h, layer.v_proj.to(dtype)).view(heads).transpose(1, 2)
    cos, sin = rope
    q, k = ops.apply_rotary(q.transpose(1, 2), cos, sin), ops.apply_rotary(k.transpose(1, 2), cos, sin)
    if cache is not None:
        end = start + length
        cache[0][:, :, start:end] = k
        cache[1][:, :, start:end] = v
        if start > 0:
            k, v = cache[0][:, :, :end], cache[1][:, :, :end]
    attended = ops.attention(q, k, v, causal=start == 0, backend=backend)
    x = x + F.linear(attended.transpose(1, 2).reshape(batch, length, -1), layer.o_proj.to(dtype))
    h = ops.rms_norm(x, layer.post_attention_layernorm.to(dtype), eps)
    return x + ops.swiglu(h, layer.gate_proj.to(dtype), layer.up_proj.to(dtype), layer.down_proj.to(dtype))


def _final_hidden(loaded: LoadedCheckpoint, token_ids: torch.Tensor, dtype: torch.dtype, backend: str,
                  capture: Callable[[int, torch.Tensor], None] | None) -> torch.Tensor:
    config, weights = loaded.config, loaded.weights
    device = weights.embed_tokens.device
    # The reference's formula, evaluated on the host for the whole sequence at once, as the reference does.
    cos, sin = ops.rotary_tables(torch.arange(token_ids.shape[1]), config.head_dim, config.rope_theta, dtype)
    rope = (cos.to(device), sin.to(device))
    x = F.embedding(token_ids.to(device=device, dtype=torch.int64), weights.embed_tokens).to(dtype)
    for index, layer in enumerate(weights.layers):
        x = decoder_layer(x, layer, rope, config, backend=backend)
        if capture is not None:
            capture(index, x)
    return ops.rms_norm(x, weights.norm.to(dtype), config.rms_norm_eps)


def final_hidden(loaded: LoadedCheckpoint, token_ids: torch.Tensor, *, dtype: torch.dtype,
                 backend: str = "math", capture: Callable[[int, torch.Tensor], None] | None = None) -> torch.Tensor:
    """Final-normed hidden states [B, T, H] in dtype for token_ids [B, T].

    capture(layer_index, output), if given, receives each layer's output; it must not modify it.
    """
    check_dtype(loaded, dtype, DTYPES)
    _check_backend(backend)
    check_token_ids(loaded.config, token_ids, loaded.config.max_position_embeddings)
    return _final_hidden(loaded, token_ids, dtype, backend, capture)


def lm_logits(loaded: LoadedCheckpoint, hidden: torch.Tensor) -> torch.Tensor:
    """Logits [..., V] in hidden.dtype for final-normed hidden states [..., H]; no cast."""
    if not isinstance(hidden, torch.Tensor) or hidden.dim() < 1 or hidden.shape[-1] != loaded.config.hidden_size:
        raise ValueError(f"hidden: expected a tensor [..., {loaded.config.hidden_size}]")
    if hidden.device != loaded.weights.lm_head.device:
        raise ValueError(f"hidden: on {hidden.device}, weights on {loaded.weights.lm_head.device}")
    check_dtype(loaded, hidden.dtype, DTYPES)
    return F.linear(hidden, loaded.weights.lm_head.to(hidden.dtype))


def forward_full(loaded: LoadedCheckpoint, token_ids: torch.Tensor, *, dtype: torch.dtype,
                 logit_positions: Sequence[int] | torch.Tensor | None = None, backend: str = "math") -> torch.Tensor:
    """fp32 logits [B, n, V] at logit_positions (all T positions when None), without a cache.

    dtype is bfloat16 or float32; logit_positions index [0, T), shared by the batch, in the order
    given. At fp32, selected rows can differ in the last bits from the same rows of an
    all-positions call, since the CPU BLAS picks its kernel by shape; the reference does the same.
    """
    check_dtype(loaded, dtype, (torch.bfloat16, torch.float32))
    _check_backend(backend)
    _, length = check_token_ids(loaded.config, token_ids, loaded.config.max_position_embeddings)
    rows = None if logit_positions is None else _check_logit_positions(logit_positions, length)
    hidden = _final_hidden(loaded, token_ids, dtype, backend, None)
    if rows is not None:
        hidden = hidden[:, rows.to(hidden.device)]
    return lm_logits(loaded, hidden).float()
