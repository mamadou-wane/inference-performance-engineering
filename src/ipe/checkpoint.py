"""Verify Qwen3 checkpoint files against a pin, and inspect their tensors from headers alone.

Verification hashes every byte of every pinned file; run it once after a download. Inspection
reads only the safetensors headers and never requests tensor payloads. A malformed checkpoint
raises CheckpointError naming the file or tensor; file-access errors stay OSError. Pins are
repository files and are parsed without validation beyond their fields.
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType

from safetensors import SafetensorError, safe_open

from ipe.config import ConfigError, Qwen3Config, load_config


class CheckpointError(ValueError):
    """A checkpoint file, index, pin or tensor inventory is invalid or incompatible."""


@dataclass(frozen=True, slots=True)
class PinnedFile:
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointPin:
    """One Hub revision: every file it contains, with its size and SHA-256."""

    repo_id: str
    revision: str
    license: str
    files: Mapping[str, PinnedFile]


@dataclass(frozen=True, slots=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    file: str


SINGLE_FILE = "model.safetensors"
INDEX_FILE = "model.safetensors.index.json"
EMBED_TOKENS = "model.embed_tokens.weight"
FINAL_NORM = "model.norm.weight"
LM_HEAD = "lm_head.weight"
STORED_DTYPE = "BF16"

# Engine field -> checkpoint name suffix after "model.layers.{i}.", in decoder order.
LAYER_TENSORS = MappingProxyType({
    "input_layernorm": "input_layernorm.weight",
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "o_proj": "self_attn.o_proj.weight",
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
})


def load_pin(path: str | PathLike[str]) -> CheckpointPin:
    """Read a checkpoint pin: the repository, revision, license and every file's size and hash.

    File names are plain names in the checkpoint directory.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(document) != {"repo_id", "revision", "license", "files"}:
        raise CheckpointError(f"{path}: expected the fields files, license, repo_id and revision")
    files = {}
    for name, entry in document["files"].items():
        if "/" in name or name in ("", ".", ".."):
            raise CheckpointError(f"{path}: {name!r} is not a plain file name")
        if set(entry) != {"size", "sha256"}:
            raise CheckpointError(f"{path}: {name}: expected the fields sha256 and size")
        files[name] = PinnedFile(entry["size"], entry["sha256"])
    return CheckpointPin(document["repo_id"], document["revision"], document["license"],
                         MappingProxyType(files))


def _sha256(path: Path) -> str:
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_files(directory: str | PathLike[str], pin: CheckpointPin) -> None:
    """Require every pinned file, with its pinned size and SHA-256; other files are ignored.

    Reads every byte of every pinned file. Reports every problem in one CheckpointError.
    """
    root = Path(directory)
    problems = []
    for name, expected in pin.files.items():
        path = root / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif (size := path.stat().st_size) != expected.size:
            problems.append(f"{name}: size {size}, expected {expected.size}")
        elif (digest := _sha256(path)) != expected.sha256:
            problems.append(f"{name}: sha256 {digest}, expected {expected.sha256}")
    if problems:
        raise CheckpointError(f"{root}: {len(problems)} problem(s): " + "; ".join(problems))


def _read_file(path: Path, file: str) -> dict[str, TensorInfo]:
    """Return a file's tensors, reading only its header."""
    try:
        with safe_open(str(path), framework="numpy") as handle:
            tensors = {}
            for name in handle.keys():  # noqa: SIM118 - safe_open handles are not iterable
                tensor = handle.get_slice(name)
                tensors[name] = TensorInfo(tensor.get_dtype(), tuple(tensor.get_shape()), file)
    except SafetensorError as error:
        raise CheckpointError(f"{file}: {error}") from error
    return tensors


def inspect_checkpoint(directory: str | PathLike[str]) -> dict[str, TensorInfo]:
    """Return every stored tensor's dtype, shape and file, from headers alone.

    The directory holds either model.safetensors or an index with exactly the shards it names,
    and no other .safetensors file. The index must map exactly the stored names, and no tensor
    may be stored twice.
    """
    root = Path(directory)
    top_level = set(os.listdir(root))
    has_single, has_index = SINGLE_FILE in top_level, INDEX_FILE in top_level
    if has_single == has_index:
        raise CheckpointError(f"{root}: expected exactly one of {SINGLE_FILE} or {INDEX_FILE}")
    if has_index:
        try:
            weight_map = json.loads((root / INDEX_FILE).read_text(encoding="utf-8"))["weight_map"]
            files = sorted(set(weight_map.values()))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise CheckpointError(f"{INDEX_FILE}: malformed ({error!r})") from error
    else:
        files = [SINGLE_FILE]
    found = sorted(name for name in top_level if name.endswith(".safetensors"))
    if found != files:
        raise CheckpointError(f"{root}: safetensors files {found}, expected {files}")

    inventory: dict[str, TensorInfo] = {}
    for file in files:
        for name, info in _read_file(root / file, file).items():
            if name in inventory:
                raise CheckpointError(f"{name}: stored in both {inventory[name].file} and {file}")
            inventory[name] = info
    if has_index and set(weight_map) != set(inventory):
        unmatched = sorted(set(weight_map) ^ set(inventory))
        raise CheckpointError(f"{INDEX_FILE}: names mapped or stored, but not both: {unmatched}")
    return dict(sorted(inventory.items()))


def _layer_shapes(config: Qwen3Config) -> dict[str, tuple[int, ...]]:
    hidden, intermediate = config.hidden_size, config.intermediate_size
    query = config.num_attention_heads * config.head_dim
    key_value = config.num_key_value_heads * config.head_dim
    return {
        "input_layernorm": (hidden,),
        "q_proj": (query, hidden),
        "k_proj": (key_value, hidden),
        "v_proj": (key_value, hidden),
        "q_norm": (config.head_dim,),
        "k_norm": (config.head_dim,),
        "o_proj": (hidden, query),
        "post_attention_layernorm": (hidden,),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }


def expected_tensors(config: Qwen3Config) -> dict[str, tuple[int, ...]]:
    """Every tensor the engine consumes, with its shape; the optional stored LM head is excluded."""
    expected = {EMBED_TOKENS: (config.vocab_size, config.hidden_size)}
    shapes = _layer_shapes(config)
    for layer in range(config.num_hidden_layers):
        for field, suffix in LAYER_TENSORS.items():
            expected[f"model.layers.{layer}.{suffix}"] = shapes[field]
    expected[FINAL_NORM] = (config.hidden_size,)
    return expected


def check_mapping(inventory: Mapping[str, TensorInfo], config: Qwen3Config) -> None:
    """Require every expected tensor, stored as BF16 with its derived shape, and nothing else.

    A stored LM head is allowed when it has the embedding's shape.
    """
    allowed = expected_tensors(config)
    missing = sorted(set(allowed) - set(inventory))
    allowed[LM_HEAD] = (config.vocab_size, config.hidden_size)
    stored = sorted(inventory.items())
    unexpected = [f"{name} in {info.file}" for name, info in stored if name not in allowed]
    mismatched = [
        f"{name} in {info.file}: {info.dtype} {list(info.shape)}, "
        f"expected {STORED_DTYPE} {list(allowed[name])}"
        for name, info in stored
        if name in allowed and (info.dtype != STORED_DTYPE or info.shape != allowed[name])
    ]
    if missing or unexpected or mismatched:
        raise CheckpointError(
            "checkpoint does not match the configuration: "
            f"{len(missing)} missing {missing}, {len(unexpected)} unexpected {unexpected}, "
            f"{len(mismatched)} mismatched [{'; '.join(mismatched)}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ipe.checkpoint")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="check a directory against a checkpoint pin")
    verify.add_argument("pin")
    verify.add_argument("directory")
    inspect = commands.add_parser("inspect", help="print the tensor inventory and check the mapping")
    inspect.add_argument("directory")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "verify":
            pin = load_pin(arguments.pin)
            verify_files(arguments.directory, pin)
            print(f"verified {len(pin.files)} files: {pin.repo_id}@{pin.revision}")
        else:
            inventory = inspect_checkpoint(arguments.directory)
            config = load_config(Path(arguments.directory) / "config.json")
            for name, info in inventory.items():
                print(f"{name}\t{info.dtype}\t{list(info.shape)}\t{info.file}")
            check_mapping(inventory, config)
            print(f"{len(inventory)} tensors; mapping matches config.json", file=sys.stderr)
    except (CheckpointError, ConfigError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
