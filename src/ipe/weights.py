"""Load a Qwen3 checkpoint into the engine's weight structure.

No tensor data is read until config.json, the layout, every header and the tensor mapping pass
(ipe.checkpoint); hashes are not checked here (run ipe.checkpoint verify once per download). Each
bf16 tensor is cast exactly to the requested dtype. A stored LM head equal to the embedding is
dropped and lm_head is the embedding itself, as in Transformers 5.17.0.

Weights are read-only: at bf16 on the CPU they are copy-on-write views of the files. If loading
fails after the checks, every held tensor is dropped and the callee frames are cleared, so a
traceback kept by the caller does not pin gigabytes of weights. Errors: ValueError for dtype or
device, ConfigError, CheckpointError, and OSError for file access.
"""

import traceback
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import torch
from safetensors import safe_open

from ipe.checkpoint import (
    EMBED_TOKENS,
    FINAL_NORM,
    LAYER_TENSORS,
    LM_HEAD,
    check_mapping,
    inspect_checkpoint,
)
from ipe.config import Qwen3Config, load_config

DTYPES = (torch.bfloat16, torch.float32, torch.float64)
# Rows per chunk when comparing the embedding with the stored head; bounds the fp32 temporaries.
_DIFF_ROWS = 4096


@dataclass(frozen=True, slots=True)
class LayerWeights:
    """One decoder layer; one field per ipe.checkpoint.LAYER_TENSORS key, in [out, in] layout."""

    input_layernorm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o_proj: torch.Tensor
    post_attention_layernorm: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor


@dataclass(frozen=True, slots=True)
class Qwen3Weights:
    embed_tokens: torch.Tensor
    layers: tuple[LayerWeights, ...]
    norm: torch.Tensor
    lm_head: torch.Tensor  # the same object as embed_tokens whenever tied


@dataclass(frozen=True, slots=True)
class TieRecord:
    """How the LM head was chosen. The comparisons use the stored bf16 tensors."""

    stored: bool  # lm_head.weight is present in the files
    tied: bool  # not stored, or torch.equal to the embedding: the head is the embedding
    max_abs_diff: float | None  # fp32, over unequal positions; NaN if either side holds a NaN


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    config: Qwen3Config
    weights: Qwen3Weights
    tie: TieRecord


def _check_arguments(dtype: torch.dtype, device: str | torch.device) -> torch.device:
    if dtype not in DTYPES:
        raise ValueError(f"dtype {dtype!r}: expected one of {list(DTYPES)}")
    if not isinstance(device, (str, torch.device)):
        raise ValueError(f"device {device!r}: expected a string or torch.device")  # noqa: TRY004 - one error type for every unsupported argument
    try:
        device = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"device {device!r}: {error}") from error
    if device.type in ("cpu", "mps") and device.index not in (None, 0):
        raise ValueError(f"device {device}: only index 0 exists")
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("device mps: not available")
        if dtype == torch.float64:
            raise ValueError("device mps: float64 is not supported")
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device cuda: not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"device {device}: only {torch.cuda.device_count()} CUDA device(s)")
    elif device.type != "cpu":
        raise ValueError(f"device {device}: expected cpu, mps or cuda")
    return device


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Equal positions count as 0, so matching infinities do not produce inf - inf = NaN."""
    result = torch.zeros((), dtype=torch.float32)
    for start in range(0, a.shape[0], _DIFF_ROWS):
        x, y = a[start:start + _DIFF_ROWS].float(), b[start:start + _DIFF_ROWS].float()
        result = torch.maximum(result, torch.where(x == y, 0.0, (x - y).abs()).amax())
    return result.item()


def _tie_record(stored: dict[str, torch.Tensor]) -> TieRecord:
    """Tie when the stored head equals the embedding in value (torch.equal, as in Transformers).

    +0 equals -0, so a signed-zero tie drops the head's -0 bytes; NaN equals nothing.
    """
    if LM_HEAD not in stored:
        return TieRecord(stored=False, tied=True, max_abs_diff=None)
    embed, head = stored[EMBED_TOKENS], stored[LM_HEAD]
    tied = torch.equal(embed, head)
    return TieRecord(stored=True, tied=tied, max_abs_diff=0.0 if tied else _max_abs_diff(embed, head))


def _assemble(config: Qwen3Config, loaded: dict[str, torch.Tensor], tied: bool) -> Qwen3Weights:
    embed_tokens = loaded[EMBED_TOKENS]
    layers = tuple(
        LayerWeights(**{field: loaded[f"model.layers.{layer}.{suffix}"] for field, suffix in LAYER_TENSORS.items()})
        for layer in range(config.num_hidden_layers)
    )
    lm_head = embed_tokens if tied else loaded[LM_HEAD]
    return Qwen3Weights(embed_tokens, layers, loaded[FINAL_NORM], lm_head)


def load_checkpoint(directory: str | PathLike[str], *, dtype: torch.dtype,
                    device: str | torch.device) -> LoadedCheckpoint:
    """Validate a checkpoint directory, then load its tensors at dtype on device.

    dtype is bfloat16, float32 or float64; device is cpu, mps (bfloat16 and float32) or an
    available cuda device.
    """
    device = _check_arguments(dtype, device)
    root = Path(directory)
    config = load_config(root / "config.json")
    inventory = inspect_checkpoint(root)
    check_mapping(inventory, config)

    stored: dict[str, torch.Tensor] = {}
    loaded: dict[str, torch.Tensor] = {}
    try:
        # One mapping per file; at bf16 on the CPU the tensors are views of it.
        for file in sorted({info.file for info in inventory.values()}):
            with safe_open(str(root / file), framework="pt", device="cpu") as handle:
                for name, info in inventory.items():
                    if info.file == file:
                        stored[name] = handle.get_tensor(name)
        tie = _tie_record(stored)
        for name in inventory:
            if not (tie.tied and name == LM_HEAD):
                loaded[name] = stored[name].to(dtype).to(device)
        weights = _assemble(config, loaded, tie.tied)
    except BaseException as error:
        # Keep a traceback held by the caller from pinning partial results.
        stored.clear()
        loaded.clear()
        traceback.clear_frames(error.__traceback__.tb_next)
        raise
    return LoadedCheckpoint(config, weights, tie)
