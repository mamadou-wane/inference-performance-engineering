"""Validate dense Qwen3 structure before any checkpoint tensors are read.

Dimensions and numerical parameters must be explicit; Transformers' model-class
defaults describe a different checkpoint. A successful validation says nothing
about the presence, shapes, dtypes, or correctness of checkpoint tensors.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


class ConfigError(ValueError):
    """A configuration is malformed or requests an unsupported feature."""


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Immutable structural values returned by :func:`validate_config`."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float


_DIMENSIONS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
)
_FEATURES = {
    "model_type", "architectures", "hidden_act", "tie_word_embeddings",
    "attention_bias", "attention_dropout", "use_sliding_window", "sliding_window",
    "max_window_layers", "layer_types", "rms_norm_eps", "rope_theta",
    "rope_scaling", "rope_parameters", "partial_rotary_factor",
}
# These annotations do not select execution dtype, cache behavior, or sampling policy.
_ANNOTATIONS = {
    "bos_token_id", "eos_token_id", "pad_token_id", "torch_dtype", "dtype",
    "transformers_version", "initializer_range", "use_cache", "_name_or_path", "_commit_hash",
}


def _required(document: Mapping, field: str):
    if field not in document:
        raise ConfigError(f"{field}: an explicit value is required")
    return document[field]


def _integer(value: object, field: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{field}: expected an integer >= {minimum}, excluding booleans")
    return value


def _number(value: object, field: str, *, positive: bool = True) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{field}: expected a finite number, excluding booleans")
    try:
        result = float(value)
    except OverflowError as error:
        raise ConfigError(f"{field}: expected a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ConfigError(f"{field}: expected a finite {'positive ' if positive else ''}number")
    return result


def _full_rotary(value: object, field: str) -> None:
    if _number(value, field) != 1.0:
        raise ConfigError(f"{field}: only full-head rotary (1.0) is supported")


def _rope_theta(document: Mapping) -> float:
    # Transformers 5.17.0 takes a non-empty rope_scaling in place of rope_parameters, ignoring
    # the latter; accepting both would let a theta the reference never reads set ours.
    if document.get("rope_scaling") and document.get("rope_parameters") is not None:
        raise ConfigError("rope_scaling, rope_parameters: set at most one")
    candidates = []
    if "rope_theta" in document:
        candidates.append(_number(document["rope_theta"], "rope_theta"))
    if "partial_rotary_factor" in document:
        _full_rotary(document["partial_rotary_factor"], "partial_rotary_factor")

    for field in ("rope_scaling", "rope_parameters"):
        params = document.get(field)
        if params is None:
            continue
        if not isinstance(params, Mapping):
            raise ConfigError(f"{field}: expected a global RoPE object or null")
        allowed = {"type", "rope_type", "rope_theta", "partial_rotary_factor"}
        unknown = set(params) - allowed
        if unknown:
            raise ConfigError(f"{field}: unsupported RoPE keys {sorted(map(str, unknown))}")
        # Legacy and modern aliases may coexist only when they describe the same operation.
        for key in ("type", "rope_type"):
            if key in params and params[key] != "default":
                raise ConfigError(f"{field}.{key}: only unscaled default RoPE is supported")
        if "rope_theta" in params:
            candidates.append(_number(params["rope_theta"], f"{field}.rope_theta"))
        if "partial_rotary_factor" in params:
            _full_rotary(params["partial_rotary_factor"], f"{field}.partial_rotary_factor")

    if not candidates:
        raise ConfigError("rope_theta: an explicit value is required at top level or in RoPE parameters")
    if any(theta != candidates[0] for theta in candidates[1:]):
        raise ConfigError("rope_theta: conflicting top-level or RoPE alias values")
    return candidates[0]


def validate_config(document: Mapping[str, object]) -> Qwen3Config:
    """Validate explicit dimensions and the supported feature subset without mutation.

    The input is a checkpoint-style JSON object, with legacy or modern unscaled
    RoPE fields. Unknown fields are rejected. Known training/tokenizer annotations
    are ignored; they never supply runtime or generation defaults.
    """
    if not isinstance(document, Mapping):
        raise ConfigError("config: expected a JSON object")
    if any(not isinstance(key, str) for key in document):
        raise ConfigError("config: field names must be strings")
    unknown = set(document) - set(_DIMENSIONS) - _FEATURES - _ANNOTATIONS
    if unknown:
        raise ConfigError(f"config: unsupported fields {sorted(unknown)}")
    for field, expected in (
        ("model_type", "qwen3"),
        ("architectures", ["Qwen3ForCausalLM"]),
        ("hidden_act", "silu"),
    ):
        if _required(document, field) != expected:
            raise ConfigError(f"{field}: only {expected!r} is supported")
    if _required(document, "tie_word_embeddings") is not True:
        raise ConfigError("tie_word_embeddings: tied embeddings (true) are required")
    for field in ("attention_bias", "use_sliding_window"):
        if document.get(field, False) is not False:
            raise ConfigError(f"{field}: only false is supported")
    if _number(document.get("attention_dropout", 0), "attention_dropout", positive=False) != 0:
        raise ConfigError("attention_dropout: only zero dropout is supported")

    dimensions = {field: _integer(_required(document, field), field) for field in _DIMENSIONS}
    if dimensions["num_attention_heads"] % dimensions["num_key_value_heads"]:
        raise ConfigError("num_attention_heads: must be divisible by num_key_value_heads")
    if dimensions["head_dim"] % 2:
        raise ConfigError("head_dim: full split-half rotary requires an even dimension")
    # Qwen3's projection width is independent of hidden_size, including in Qwen3-0.6B.
    layers = document.get("layer_types")
    if layers is not None and (
        not isinstance(layers, list)
        or len(layers) != dimensions["num_hidden_layers"]
        or any(layer != "full_attention" for layer in layers)
    ):
        raise ConfigError("layer_types: expected one full_attention entry per num_hidden_layers")
    # The pinned reference discards window size when use_sliding_window is false.
    if document.get("sliding_window") is not None:
        _integer(document["sliding_window"], "sliding_window")
    if "max_window_layers" in document:
        _integer(document["max_window_layers"], "max_window_layers", minimum=0)

    return Qwen3Config(
        **dimensions,
        rms_norm_eps=_number(_required(document, "rms_norm_eps"), "rms_norm_eps"),
        rope_theta=_rope_theta(document),
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"{key}: duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ConfigError(f"JSON: non-finite constant {value} is not supported")


def load_config(path: str | PathLike[str]) -> Qwen3Config:
    """Read one local UTF-8 JSON file; reject duplicate keys and nonstandard constants.

    File access errors retain their normal OSError type. Parsing and validation
    errors include the path and never trigger model or checkpoint loading.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        document = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        return validate_config(document)
    except (ValueError, RecursionError) as error:
        raise ConfigError(f"{path}: {error}") from error
