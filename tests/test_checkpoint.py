"""Checkpoint pins, verification, header inspection and the configuration mapping."""

import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import ipe.checkpoint
from ipe.checkpoint import (
    EMBED_TOKENS,
    FINAL_NORM,
    INDEX_FILE,
    LAYER_TENSORS,
    LM_HEAD,
    SINGLE_FILE,
    CheckpointError,
    TensorInfo,
    check_mapping,
    expected_tensors,
    inspect_checkpoint,
    load_pin,
    main,
    verify_files,
)
from ipe.config import load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PINS = ROOT / "checkpoints"
WEIGHTS = ROOT / "weights"
ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1, "I32": 4, "I64": 8}
REVISIONS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
}


# ---------------------------------------------------------------- synthetic files

def small_config():
    # Projection width 4 x 8 = 32 differs from hidden 12, and K = 16 differs from Q = 32.
    return validate_config({
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "vocab_size": 32,
        "hidden_size": 12, "intermediate_size": 24, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "max_position_embeddings": 64, "hidden_act": "silu", "tie_word_embeddings": True,
        "rms_norm_eps": 1e-6, "rope_theta": 10000,
    })


def raw_file(path, header: bytes, data_size: int):
    """Write the 8-byte length, the header bytes, then a sparse zero-filled data region."""
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", len(header)) + header)
        file.truncate(8 + len(header) + data_size)


def write_safetensors(path, tensors):
    """Write a valid safetensors file whose tensors (name -> (dtype, shape)) hold zeros."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = ITEMSIZE[dtype]
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    text = json.dumps(header, separators=(",", ":")).encode()
    raw_file(path, text + b" " * (-len(text) % 8), offset)


def bf16(config, *, lm_head=False):
    tensors = {name: ("BF16", shape) for name, shape in expected_tensors(config).items()}
    if lm_head:
        tensors[LM_HEAD] = ("BF16", (config.vocab_size, config.hidden_size))
    return tensors


def write_single(directory, tensors):
    directory.mkdir(exist_ok=True)
    write_safetensors(directory / SINGLE_FILE, tensors)
    return directory


def write_sharded(directory, tensors, split):
    """Write two shards and an index: names in `split` go to the second shard, the rest to the first."""
    directory.mkdir(exist_ok=True)
    files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    groups = [{n: t for n, t in tensors.items() if n not in split}, {n: tensors[n] for n in split}]
    weight_map = {}
    for file, group in zip(files, groups):
        write_safetensors(directory / file, group)
        weight_map.update({name: file for name in group})
    (directory / INDEX_FILE).write_text(json.dumps({"weight_map": weight_map}))
    return directory


# ---------------------------------------------------------------- expected tensors

def test_layer_fields_follow_the_decoder_order():
    assert list(LAYER_TENSORS) == [
        "input_layernorm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj",
        "post_attention_layernorm", "gate_proj", "up_proj", "down_proj",
    ]


def test_expected_shapes_use_projection_widths_not_hidden_size():
    expected = expected_tensors(small_config())
    assert len(expected) == 2 * 11 + 2
    assert expected[EMBED_TOKENS] == (32, 12)
    assert expected[FINAL_NORM] == (12,)
    assert expected["model.layers.1.self_attn.q_proj.weight"] == (32, 12)
    assert expected["model.layers.1.self_attn.k_proj.weight"] == (16, 12)
    assert expected["model.layers.1.self_attn.v_proj.weight"] == (16, 12)
    assert expected["model.layers.1.self_attn.o_proj.weight"] == (12, 32)
    assert expected["model.layers.1.self_attn.q_norm.weight"] == (8,)
    assert expected["model.layers.1.mlp.down_proj.weight"] == (12, 24)
    assert LM_HEAD not in expected


@pytest.mark.parametrize(("fixture", "tied", "stored"), [
    ("qwen3-1.7b-config.json", 1_720_574_976, 2_031_739_904),
    ("qwen3-0.6b-config.json", 596_049_920, 751_632_384),
])
def test_expected_parameter_counts_match_the_published_totals(fixture, tied, stored):
    config = load_config(FIXTURES / fixture)
    counts = []
    for shape in expected_tensors(config).values():
        count = 1
        for dim in shape:
            count *= dim
        counts.append(count)
    assert len(counts) == 310
    assert sum(counts) == tied
    assert sum(counts) + config.vocab_size * config.hidden_size == stored


# ---------------------------------------------------------------- mapping check

def inventory_of(tensors, file=SINGLE_FILE):
    return {name: TensorInfo(dtype, tuple(shape), file) for name, (dtype, shape) in tensors.items()}


@pytest.mark.parametrize("lm_head", [False, True])
def test_mapping_accepts_the_expected_tensors_with_or_without_a_stored_head(lm_head):
    config = small_config()
    check_mapping(inventory_of(bf16(config, lm_head=lm_head)), config)


@pytest.mark.parametrize(("change", "name", "kind"), [
    (lambda t: t.pop("model.layers.1.mlp.up_proj.weight"), "model.layers.1.mlp.up_proj.weight", "missing"),
    (lambda t: t.pop(EMBED_TOKENS), EMBED_TOKENS, "missing"),
    (lambda t: t.__setitem__("model.layers.0.self_attn.q_proj.bias", ("BF16", (32,))),
     "model.layers.0.self_attn.q_proj.bias", "unexpected"),
    (lambda t: t.__setitem__("model.layers.2.input_layernorm.weight", ("BF16", (12,))),
     "model.layers.2.input_layernorm.weight", "unexpected"),
    (lambda t: t.__setitem__("model.layers.01.input_layernorm.weight", ("BF16", (12,))),
     "model.layers.01.input_layernorm.weight", "unexpected"),
    (lambda t: t.__setitem__("model.rotary_emb.inv_freq", ("F32", (4,))), "model.rotary_emb.inv_freq", "unexpected"),
    (lambda t: t.__setitem__("model.layers.0.self_attn.q_proj.weight", ("BF16", (12, 32))),
     "model.layers.0.self_attn.q_proj.weight", "mismatched"),
    (lambda t: t.__setitem__("model.layers.0.self_attn.k_proj.weight", ("BF16", (32, 12))),
     "model.layers.0.self_attn.k_proj.weight", "mismatched"),
    (lambda t: t.__setitem__("model.layers.1.post_attention_layernorm.weight", ("F32", (12,))),
     "model.layers.1.post_attention_layernorm.weight", "mismatched"),
    (lambda t: t.__setitem__(LM_HEAD, ("BF16", (12, 32))), LM_HEAD, "mismatched"),
    (lambda t: t.__setitem__(LM_HEAD, ("F16", (32, 12))), LM_HEAD, "mismatched"),
])
def test_mapping_rejects_each_problem_by_name(change, name, kind):
    config = small_config()
    tensors = bf16(config)
    change(tensors)
    with pytest.raises(CheckpointError, match=re.escape(name)) as error:
        check_mapping(inventory_of(tensors), config)
    for category in ("missing", "unexpected", "mismatched"):
        assert f"{int(category == kind)} {category} " in str(error.value)
    if kind != "missing":
        assert f"{name} in {SINGLE_FILE}" in str(error.value)


def test_mapping_reports_every_problem_at_once():
    config = small_config()
    tensors = bf16(config)
    tensors.pop(FINAL_NORM)
    tensors["extra.weight"] = ("BF16", (1,))
    tensors[EMBED_TOKENS] = ("F32", (32, 12))
    with pytest.raises(CheckpointError) as error:
        check_mapping(inventory_of(tensors), config)
    message = str(error.value)
    assert f"1 missing ['{FINAL_NORM}']" in message
    assert f"1 unexpected ['extra.weight in {SINGLE_FILE}']" in message
    assert f"1 mismatched [{EMBED_TOKENS} in {SINGLE_FILE}: F32" in message


# ---------------------------------------------------------------- inspection

def test_inspection_reads_dtypes_shapes_and_files_from_a_single_file(tmp_path):
    config = small_config()
    tensors = bf16(config, lm_head=True)
    inventory = inspect_checkpoint(write_single(tmp_path / "ckpt", tensors))
    assert inventory == inventory_of(tensors)
    check_mapping(inventory, config)


def test_inspection_reads_a_sharded_checkpoint_through_its_index(tmp_path):
    config = small_config()
    tensors = bf16(config, lm_head=True)
    inventory = inspect_checkpoint(write_sharded(tmp_path / "ckpt", tensors, [LM_HEAD]))
    assert inventory[LM_HEAD].file == "model-00002-of-00002.safetensors"
    assert {info.file for name, info in inventory.items() if name != LM_HEAD} == {
        "model-00001-of-00002.safetensors"}
    check_mapping(inventory, config)


def sharded_case(tmp_path, edit):
    tensors = bf16(small_config(), lm_head=True)
    directory = write_sharded(tmp_path / "ckpt", tensors, [LM_HEAD])
    index = json.loads((directory / INDEX_FILE).read_text())
    edit(index["weight_map"])
    (directory / INDEX_FILE).write_text(json.dumps(index))
    return directory


@pytest.mark.parametrize(("edit", "message"), [
    (lambda m: m.__setitem__(FINAL_NORM, "model-00003-of-00003.safetensors"),
     "model-00003-of-00003.safetensors"),
    (lambda m: m.__setitem__("model.extra.weight", "model-00001-of-00002.safetensors"),
     f"{INDEX_FILE}: names mapped or stored, but not both: ['model.extra.weight']"),
    (lambda m: m.pop(FINAL_NORM), f"{INDEX_FILE}: names mapped or stored, but not both: ['{FINAL_NORM}']"),
])
def test_inspection_rejects_an_index_that_disagrees_with_the_shards(tmp_path, edit, message):
    directory = sharded_case(tmp_path, edit)
    with pytest.raises(CheckpointError, match=re.escape(message)):
        inspect_checkpoint(directory)


def test_inspection_rejects_a_tensor_stored_in_two_shards(tmp_path):
    tensors = bf16(small_config(), lm_head=True)
    directory = write_sharded(tmp_path / "ckpt", tensors, [LM_HEAD])
    write_safetensors(directory / "model-00002-of-00002.safetensors",
                      {LM_HEAD: tensors[LM_HEAD], FINAL_NORM: tensors[FINAL_NORM]})
    with pytest.raises(CheckpointError, match=re.escape(f"{FINAL_NORM}: stored in both")):
        inspect_checkpoint(directory)


def test_inspection_requires_exactly_one_layout(tmp_path):
    tensors = bf16(small_config())
    both = write_sharded(tmp_path / "both", tensors, [FINAL_NORM])
    write_safetensors(both / SINGLE_FILE, tensors)
    (tmp_path / "neither").mkdir()
    for directory in (both, tmp_path / "neither"):
        with pytest.raises(CheckpointError, match="exactly one of"):
            inspect_checkpoint(directory)


def test_inspection_rejects_a_safetensors_file_outside_the_layout(tmp_path):
    directory = write_single(tmp_path / "ckpt", bf16(small_config()))
    write_safetensors(directory / "extra.safetensors", {"x": ("BF16", (1,))})
    with pytest.raises(CheckpointError, match="extra.safetensors"):
        inspect_checkpoint(directory)


def test_inspection_rejects_a_stray_shard_next_to_an_index(tmp_path):
    directory = write_sharded(tmp_path / "ckpt", bf16(small_config()), [FINAL_NORM])
    write_safetensors(directory / "model-00003-of-00003.safetensors", {"x": ("BF16", (1,))})
    with pytest.raises(CheckpointError, match="model-00003-of-00003.safetensors"):
        inspect_checkpoint(directory)


ENTRY = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}


@pytest.mark.parametrize(("header", "data_size"), [
    (b"\x01\x02", None),                                               # shorter than 8 bytes
    (struct.pack("<Q", 200_000_000) + b"{}", None),                    # oversized length prefix
    (struct.pack("<Q", 100) + b"{}", None),                            # header past the end of file
    (b'{"a": ', 0),                                                    # invalid JSON
    (json.dumps({"a": {**ENTRY, "dtype": "bf16"}}).encode(), 4),       # unknown dtype spelling
    (json.dumps({"a": ENTRY, "b": {**ENTRY, "data_offsets": [8, 12]}}).encode(), 12),  # gap
    (json.dumps({"a": ENTRY}).encode(), 2),                            # buffer shorter than the offsets
    (json.dumps({"a": ENTRY}).encode(), 8),                            # bytes after the last tensor
])
def test_inspection_reports_library_rejections_with_the_file_name(tmp_path, header, data_size):
    directory = tmp_path / "ckpt"
    directory.mkdir()
    if data_size is None:
        (directory / SINGLE_FILE).write_bytes(header)
    else:
        raw_file(directory / SINGLE_FILE, header, data_size)
    with pytest.raises(CheckpointError, match=re.escape(SINGLE_FILE)):
        inspect_checkpoint(directory)


# ---------------------------------------------------------------- payload access during inspection

class RecordingSlice:
    def __init__(self, inner, calls):
        self._inner, self._calls = inner, calls

    def get_dtype(self):
        self._calls.append("get_dtype")
        return self._inner.get_dtype()

    def get_shape(self):
        self._calls.append("get_shape")
        return self._inner.get_shape()

    def __getitem__(self, index):
        self._calls.append("payload")
        return self._inner[index]


class RecordingHandle:
    """Wraps a safe_open handle and records every method used on it."""

    def __init__(self, inner, calls):
        self._inner, self._calls = inner, calls

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def keys(self):
        self._calls.append("keys")
        return self._inner.keys()

    def get_slice(self, name):
        self._calls.append("get_slice")
        return RecordingSlice(self._inner.get_slice(name), self._calls)

    def get_tensor(self, name):
        self._calls.append("payload")
        return self._inner.get_tensor(name)

    def __getattr__(self, name):
        self._calls.append(name)
        return getattr(self._inner, name)


@pytest.fixture(autouse=True)
def payload_recorder(monkeypatch):
    """Every test in this module runs with inspection recorded; none may request tensor payloads."""
    calls = []
    real = ipe.checkpoint.safe_open
    monkeypatch.setattr(ipe.checkpoint, "safe_open", lambda *a, **k: RecordingHandle(real(*a, **k), calls))
    yield calls
    assert "payload" not in calls


def test_recorder_records_payload_requests(tmp_path):
    # Positive control for the recorder itself.
    from safetensors import safe_open

    write_safetensors(tmp_path / "f.safetensors", {"x": ("F32", (2,))})
    calls = []
    with RecordingHandle(safe_open(str(tmp_path / "f.safetensors"), framework="numpy"), calls) as handle:
        handle.get_tensor("x")
        handle.get_slice("x")[0:1]
    assert calls == ["payload", "get_slice", "payload"]


def test_inspection_recorder_sees_header_reads_on_a_valid_checkpoint(tmp_path, payload_recorder):
    # Positive control: inspection goes through the recorded handle.
    config = small_config()
    inspect_checkpoint(write_sharded(tmp_path / "ckpt", bf16(config, lm_head=True), [LM_HEAD]))
    assert payload_recorder.count("keys") == 2
    assert payload_recorder.count("get_slice") == len(expected_tensors(config)) + 1
    assert "payload" not in payload_recorder


@pytest.mark.parametrize(("problem", "named"), [
    ("mapping", "model.layers.1.mlp.up_proj.weight"),
    ("index mismatch", FINAL_NORM),
])
def test_structural_failures_never_request_tensor_payloads(tmp_path, payload_recorder, problem, named):
    config = small_config()
    tensors = bf16(config, lm_head=True)
    if problem == "mapping":
        tensors["model.layers.1.mlp.up_proj.weight"] = ("F32", (24, 12))
        directory = write_sharded(tmp_path / "ckpt", tensors, [LM_HEAD])
        with pytest.raises(CheckpointError, match=re.escape(named)):
            check_mapping(inspect_checkpoint(directory), config)
    else:
        directory = sharded_case(tmp_path, lambda m: m.pop(FINAL_NORM))
        with pytest.raises(CheckpointError, match=re.escape(named)):
            inspect_checkpoint(directory)
    assert "payload" not in payload_recorder


# ---------------------------------------------------------------- pins

def pin_document(**files):
    return {"repo_id": "Qwen/Qwen3-0.6B", "revision": "c" * 40, "license": "apache-2.0",
            "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                      for name, data in files.items()}}


def write_pin(tmp_path, document):
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(document) if not isinstance(document, str) else document)
    return path


def test_pins_parse_with_every_field(tmp_path):
    pin = load_pin(write_pin(tmp_path, pin_document(**{"config.json": b"{}", "file.bin": b"x"})))
    assert (pin.repo_id, pin.revision, pin.license) == ("Qwen/Qwen3-0.6B", "c" * 40, "apache-2.0")
    assert list(pin.files) == ["config.json", "file.bin"]
    assert pin.files["file.bin"].size == 1


@pytest.mark.parametrize("name", ["", ".", "..", "../config.json", "/etc/passwd", "sub/file.bin"])
def test_pins_reject_file_names_that_are_not_plain_names(tmp_path, name):
    document = pin_document(**{"config.json": b"{}"})
    document["files"][name] = document["files"].pop("config.json")
    with pytest.raises(CheckpointError, match="not a plain file name"):
        load_pin(write_pin(tmp_path, document))


@pytest.mark.parametrize(("edit", "message"), [
    (lambda d: d.update(extra=1), "expected the fields files, license, repo_id and revision"),
    (lambda d: d.pop("license"), "expected the fields files, license, repo_id and revision"),
    (lambda d: d["files"]["config.json"].update(extra=1), "config.json: expected the fields sha256 and size"),
    (lambda d: d["files"]["config.json"].pop("sha256"), "config.json: expected the fields sha256 and size"),
])
def test_pins_reject_missing_or_extra_fields(tmp_path, edit, message):
    document = pin_document(**{"config.json": b"{}"})
    edit(document)
    with pytest.raises(CheckpointError, match=message):
        load_pin(write_pin(tmp_path, document))


# ---------------------------------------------------------------- verification

FILES = {"config.json": b'{"a": 1}', "model.safetensors": b"\x00" * 64, "tokenizer.json": b"tokens"}


@pytest.fixture
def pinned(tmp_path):
    directory = tmp_path / "ckpt"
    directory.mkdir()
    for name, data in FILES.items():
        (directory / name).write_bytes(data)
    return directory, load_pin(write_pin(tmp_path, pin_document(**FILES)))


def test_verification_accepts_the_pinned_files_and_ignores_other_files(pinned):
    directory, pin = pinned
    (directory / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (directory / ".cache" / "huggingface" / "download" / "config.json.metadata").write_text("x")
    (directory / "extra.txt").write_text("x")
    verify_files(directory, pin)


def flip_byte(directory):
    path = directory / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[10] ^= 1
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(("damage", "message"), [
    (flip_byte, "model.safetensors: sha256"),
    (lambda d: (d / "config.json").write_bytes(b'{"a": 12}'), "config.json: size 9, expected 8"),
    (lambda d: (d / "tokenizer.json").unlink(), "tokenizer.json: missing"),
])
def test_verification_names_each_damaged_file(pinned, damage, message):
    directory, pin = pinned
    damage(directory)
    with pytest.raises(CheckpointError, match=re.escape(message)):
        verify_files(directory, pin)


def test_verification_reports_every_problem_at_once(pinned):
    directory, pin = pinned
    flip_byte(directory)
    (directory / "config.json").write_bytes(b'{"a": 12}')
    (directory / "tokenizer.json").unlink()
    with pytest.raises(CheckpointError, match=re.escape("3 problem(s): config.json: size 9, expected 8; "
                                                        "model.safetensors: sha256 ")) as error:
        verify_files(directory, pin)
    assert "tokenizer.json: missing" in str(error.value)


# ---------------------------------------------------------------- tracked pins and real-header fixtures

def test_tracked_pins_name_the_adopted_revisions():
    for name, (repo_id, revision) in REVISIONS.items():
        pin = load_pin(PINS / f"{name}.json")
        assert (pin.repo_id, pin.revision, pin.license) == (repo_id, revision, "apache-2.0")
    large = {"qwen3-1.7b": {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", INDEX_FILE},
             "qwen3-0.6b": {SINGLE_FILE}}
    common = {".gitattributes", "LICENSE", "README.md", "config.json", "generation_config.json",
              "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    for name in REVISIONS:
        assert set(load_pin(PINS / f"{name}.json").files) == common | large[name]


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracked_fixtures_match_the_pinned_files():
    pin_17 = load_pin(PINS / "qwen3-1.7b.json")
    pin_06 = load_pin(PINS / "qwen3-0.6b.json")
    assert sha256_of(FIXTURES / "LICENSE-Qwen3-Apache-2.0") == pin_17.files["LICENSE"].sha256
    assert pin_06.files["LICENSE"] == pin_17.files["LICENSE"]
    assert sha256_of(FIXTURES / "qwen3-1.7b-config.json") == pin_17.files["config.json"].sha256
    assert sha256_of(FIXTURES / "qwen3-0.6b-config.json") == pin_06.files["config.json"].sha256
    assert sha256_of(FIXTURES / "qwen3-1.7b-model.safetensors.index.json") == pin_17.files[INDEX_FILE].sha256


# Extract hashes recorded when the extracts were cut from verified downloads.
HEADERS = {
    "qwen3-1.7b-model-00001-of-00002.header": "1ee0e4e7bf2b79af3ee159ee7894dc37663ca8118eeb88ddd9a3ff2c1c4d91e7",
    "qwen3-1.7b-model-00002-of-00002.header": "de3c1e531e8d90b962f43c40af894e503d94d171899631d258620241fff6c898",
    "qwen3-0.6b-model.header": "1119c4d33e17f4cd2d3fbebbc4176d40555e61361cb44076319167621f4034d4",
}


def reconstruct(directory, name, files):
    """Rebuild a checkpoint's structure from header extracts: real headers, zero-filled data."""
    pin = load_pin(PINS / f"{name}.json")
    directory.mkdir()
    for extract, target in files.items():
        data = (FIXTURES / extract).read_bytes()
        assert hashlib.sha256(data).hexdigest() == HEADERS[extract]
        with open(directory / target, "wb") as file:
            file.write(data)
            file.truncate(pin.files[target].size)
    return directory


@pytest.mark.parametrize(("name", "files", "shard_counts"), [
    ("qwen3-1.7b", {"qwen3-1.7b-model-00001-of-00002.header": "model-00001-of-00002.safetensors",
                    "qwen3-1.7b-model-00002-of-00002.header": "model-00002-of-00002.safetensors"},
     {"model-00001-of-00002.safetensors": 310, "model-00002-of-00002.safetensors": 1}),
    ("qwen3-0.6b", {"qwen3-0.6b-model.header": SINGLE_FILE}, {SINGLE_FILE: 311}),
])
def test_real_checkpoint_structure_matches_its_configuration(tmp_path, name, files, shard_counts):
    directory = reconstruct(tmp_path / name, name, files)
    if name == "qwen3-1.7b":
        (directory / INDEX_FILE).write_bytes((FIXTURES / "qwen3-1.7b-model.safetensors.index.json").read_bytes())
    inventory = inspect_checkpoint(directory)
    config = load_config(FIXTURES / f"{name}-config.json")
    check_mapping(inventory, config)
    assert len(inventory) == 311
    assert {info.dtype for info in inventory.values()} == {"BF16"}
    assert inventory[LM_HEAD].shape == (config.vocab_size, config.hidden_size)
    counts = {}
    for info in inventory.values():
        counts[info.file] = counts.get(info.file, 0) + 1
    assert counts == shard_counts


# ---------------------------------------------------------------- import isolation and CLI

def test_checkpoint_module_imports_and_inspects_without_torch(tmp_path):
    write_single(tmp_path / "ckpt", bf16(small_config()))
    script = r"""
import importlib.abc, sys
class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "transformers"}:
            raise AssertionError(f"unexpected import of {name}")
sys.meta_path.insert(0, RejectTorch())
sys.path.insert(0, sys.argv[1])
from ipe.checkpoint import inspect_checkpoint
print(len(inspect_checkpoint(sys.argv[2])), "torch" in sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT / "src"), str(tmp_path / "ckpt")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == [str(2 * 11 + 2), "False"]


def test_cli_inspect_prints_the_inventory_and_checks_the_mapping(tmp_path, capsys):
    directory = write_single(tmp_path / "ckpt", bf16(small_config(), lm_head=True))
    (directory / "config.json").write_bytes(json.dumps({
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "vocab_size": 32,
        "hidden_size": 12, "intermediate_size": 24, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "max_position_embeddings": 64, "hidden_act": "silu", "tie_word_embeddings": True,
        "rms_norm_eps": 1e-6, "rope_theta": 10000,
    }).encode())
    assert main(["inspect", str(directory)]) == 0
    output = capsys.readouterr()
    assert f"{LM_HEAD}\tBF16\t[32, 12]\t{SINGLE_FILE}" in output.out.splitlines()
    assert "25 tensors; mapping matches config.json" in output.err


def test_cli_verify_reports_failure_with_a_nonzero_status(pinned, tmp_path, capsys):
    directory, _ = pinned
    flip_byte(directory)
    assert main(["verify", str(tmp_path / "pin.json"), str(directory)]) == 1
    assert "model.safetensors: sha256" in capsys.readouterr().err


# ---------------------------------------------------------------- downloaded checkpoints

def downloaded(name):
    directory = WEIGHTS / name
    if not directory.is_dir():
        pytest.fail(f"{directory} is missing; download it as described in the README")
    return directory


@pytest.mark.checkpoint
@pytest.mark.parametrize("name", list(REVISIONS))
def test_downloaded_checkpoint_matches_its_pin_and_configuration(name):
    directory = downloaded(name)
    verify_files(directory, load_pin(PINS / f"{name}.json"))
    inventory = inspect_checkpoint(directory)
    check_mapping(inventory, load_config(directory / "config.json"))
    assert len(inventory) == 311


@pytest.mark.checkpoint
@pytest.mark.parametrize("name", list(REVISIONS))
def test_downloaded_license_is_apache_2(name):
    text = (downloaded(name) / "LICENSE").read_text(encoding="utf-8")
    assert text.split()[:4] == ["Apache", "License", "Version", "2.0,"]
    assert "Copyright 2024 Alibaba Cloud" in text
