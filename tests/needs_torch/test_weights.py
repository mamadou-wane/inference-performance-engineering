"""Weight loading: failure ordering, exactness, tying and identity, and parity with Transformers.

Synthetic checkpoints give every element of every tensor a distinct bf16 bit pattern, so any swap
of tensors, layers or axes changes a compared value. Their names and shapes come from a
Transformers model on the meta device. Tests pair fields by checkpoint names and attribute paths
written out in this file, independently of ipe.checkpoint, and compare bit patterns as integers.
"""

import dataclasses
import gc
import json
import math
import os
import struct
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
import safetensors
import safetensors.torch
import torch
import transformers
from safetensors import safe_open
from transformers import Qwen3ForCausalLM

import ipe.checkpoint
import ipe.weights
from ipe.checkpoint import INDEX_FILE, LAYER_TENSORS, CheckpointError, load_pin, verify_files
from ipe.config import ConfigError
from ipe.weights import DTYPES, LayerWeights, TieRecord, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PINS = ROOT / "checkpoints"
WEIGHTS = ROOT / "weights"
INTEGER = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}
STORED_NAMES = {torch.bfloat16: "BF16", torch.float32: "F32"}
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
EMBED, HEAD = "model.embed_tokens.weight", "lm_head.weight"

# MPS cases run only with IPE_TEST_MPS=1; CI runners have no usable GPU.
MPS_REQUESTED = os.environ.get("IPE_TEST_MPS") == "1"
needs_mps = pytest.mark.skipif(not MPS_REQUESTED, reason="set IPE_TEST_MPS=1 to run on MPS")
DEVICES = [("cpu", dtype) for dtype in DTYPES] + [
    pytest.param("mps", dtype, marks=needs_mps) for dtype in (torch.bfloat16, torch.float32)
]


# ---------------------------------------------------------------- synthetic checkpoints

def config_document(*, vocab, hidden, intermediate, layers, heads, kv_heads, head_dim):
    return {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "attention_bias": False,
        "attention_dropout": 0.0, "bos_token_id": 0, "eos_token_id": 1, "head_dim": head_dim,
        "hidden_act": "silu", "hidden_size": hidden, "intermediate_size": intermediate,
        "max_position_embeddings": 64, "max_window_layers": layers, "num_attention_heads": heads,
        "num_hidden_layers": layers, "num_key_value_heads": kv_heads, "rms_norm_eps": 1e-6,
        "rope_scaling": None, "rope_theta": 10000, "sliding_window": None,
        "tie_word_embeddings": True, "torch_dtype": "bfloat16", "use_cache": True,
        "use_sliding_window": False, "vocab_size": vocab,
    }


CONFIGS = {
    # Q = 4 x 8 = 32 differs from hidden 12, and K = 16 differs from Q.
    "narrow": config_document(vocab=32, hidden=12, intermediate=24, layers=2, heads=4, kv_heads=2, head_dim=8),
    # Q = H = 32: q_proj and o_proj have the same, square shape.
    "square": config_document(vocab=40, hidden=32, intermediate=48, layers=2, heads=4, kv_heads=2, head_dim=8),
    # 11 layers: names sort as 0, 1, 10, 2, ... and layer 10 is the last.
    "deep": config_document(vocab=16, hidden=8, intermediate=16, layers=11, heads=2, kv_heads=1, head_dim=4),
}


def reference_shapes(document):
    """Checkpoint names and shapes as Transformers defines them, the stored head excluded."""
    with torch.device("meta"):
        model = Qwen3ForCausalLM(transformers.Qwen3Config(**document))
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items() if name != HEAD}


def normal_bf16_patterns(seed):
    """Every normal bf16 value (no zeros, subnormals, infinities or NaN) in a seeded order."""
    bits = torch.arange(1 << 16, dtype=torch.int32)
    exponent = (bits >> 7) & 0xFF
    bits = bits[(exponent > 0) & (exponent < 0xFF)]
    bits = bits[torch.randperm(len(bits), generator=torch.Generator().manual_seed(seed))]
    return torch.where(bits >= 1 << 15, bits - (1 << 16), bits).to(torch.int16).view(torch.bfloat16)


def distinct_tensors(shapes, seed=0):
    values = normal_bf16_patterns(seed)
    tensors, used = {}, 0
    for name, shape in shapes.items():
        count = math.prod(shape)
        tensors[name] = values[used:used + count].reshape(shape).clone()
        used += count
    return tensors, values[used:]


def with_head(tensors, unused, head):
    """Apply one stored-head case: absent, equal, signed_zero, different, last_element, nan or head_only."""
    tensors = dict(tensors)
    embed = tensors[EMBED].clone()
    if head == "equal":
        tensors[HEAD] = embed.clone()
    elif head == "signed_zero":  # equal in value, different in bytes
        embed[0, 0] = 0.0
        tensors[HEAD] = embed.clone()
        tensors[HEAD][0, 0] = -0.0
    elif head == "different":
        tensors[HEAD] = unused[:embed.numel()].reshape(embed.shape).clone()
    elif head == "last_element":  # one mantissa bit differs, in the last row and column
        tensors[HEAD] = embed.clone()
        tensors[HEAD].view(torch.int16)[-1, -1] ^= 1
    elif head == "nan":  # bitwise equal, yet unequal in value
        embed[1, 1] = float("nan")
        tensors[HEAD] = embed.clone()
    elif head == "head_only":
        tensors[HEAD] = tensors.pop(EMBED)
        return tensors
    else:
        assert head == "absent"
    tensors[EMBED] = embed
    return tensors


def write_file(path, tensors):
    """Write a safetensors file byte by byte."""
    header, payload, offset = {}, [], 0
    for name, tensor in tensors.items():
        data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": STORED_NAMES[tensor.dtype], "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(data)]}
        payload.append(data)
        offset += len(data)
    text = json.dumps(header)
    text = text.encode() + b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(payload))
    return offset


def write_checkpoint(directory, document, tensors, *, second_shard=None, weight_map=None):
    """Write config.json and the tensors: one file, or two shards when `second_shard` names any."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(document))
    if second_shard is None:
        write_file(directory / "model.safetensors", tensors)
        return directory
    groups = [{n: t for n, t in tensors.items() if n not in second_shard},
              {n: t for n, t in tensors.items() if n in second_shard}]
    total = sum(write_file(directory / file, group) for file, group in zip(SHARDS, groups))
    computed = {name: file for file, group in zip(SHARDS, groups) for name in group}
    index = {"metadata": {"total_size": total}, "weight_map": computed if weight_map is None else weight_map}
    (directory / INDEX_FILE).write_text(json.dumps(index))
    return directory


def synthetic(directory, config="narrow", head="equal", *, sharded=False):
    document = CONFIGS[config]
    base, unused = distinct_tensors(reference_shapes(document))
    tensors = with_head(base, unused, head)
    last = document["num_hidden_layers"] - 1
    second = [n for n in tensors if n.startswith(f"model.layers.{last}.")] + ["model.norm.weight", HEAD]
    write_checkpoint(directory, document, tensors, second_shard=second if sharded else None)
    return tensors


# ---------------------------------------------------------------- pairing and comparison

def by_checkpoint_name(weights, *, include_head):
    """(checkpoint name, engine field), written out independently of ipe.checkpoint.LAYER_TENSORS."""
    pairs = [(EMBED, weights.embed_tokens)]
    for i, layer in enumerate(weights.layers):
        prefix = f"model.layers.{i}."
        pairs += [
            (prefix + "input_layernorm.weight", layer.input_layernorm),
            (prefix + "self_attn.q_proj.weight", layer.q_proj),
            (prefix + "self_attn.k_proj.weight", layer.k_proj),
            (prefix + "self_attn.v_proj.weight", layer.v_proj),
            (prefix + "self_attn.q_norm.weight", layer.q_norm),
            (prefix + "self_attn.k_norm.weight", layer.k_norm),
            (prefix + "self_attn.o_proj.weight", layer.o_proj),
            (prefix + "post_attention_layernorm.weight", layer.post_attention_layernorm),
            (prefix + "mlp.gate_proj.weight", layer.gate_proj),
            (prefix + "mlp.up_proj.weight", layer.up_proj),
            (prefix + "mlp.down_proj.weight", layer.down_proj),
        ]
    pairs.append(("model.norm.weight", weights.norm))
    if include_head:
        pairs.append((HEAD, weights.lm_head))
    return pairs


def by_attribute_path(reference, weights):
    """(label, Transformers parameter, engine field), paired by attribute paths written here."""
    model = reference.model
    pairs = [("embed_tokens", model.embed_tokens.weight, weights.embed_tokens)]
    assert len(weights.layers) == len(model.layers)
    for i, (ref, layer) in enumerate(zip(model.layers, weights.layers)):
        pairs += [
            (f"layers[{i}].input_layernorm", ref.input_layernorm.weight, layer.input_layernorm),
            (f"layers[{i}].q_proj", ref.self_attn.q_proj.weight, layer.q_proj),
            (f"layers[{i}].k_proj", ref.self_attn.k_proj.weight, layer.k_proj),
            (f"layers[{i}].v_proj", ref.self_attn.v_proj.weight, layer.v_proj),
            (f"layers[{i}].q_norm", ref.self_attn.q_norm.weight, layer.q_norm),
            (f"layers[{i}].k_norm", ref.self_attn.k_norm.weight, layer.k_norm),
            (f"layers[{i}].o_proj", ref.self_attn.o_proj.weight, layer.o_proj),
            (f"layers[{i}].post_attention_layernorm", ref.post_attention_layernorm.weight,
             layer.post_attention_layernorm),
            (f"layers[{i}].gate_proj", ref.mlp.gate_proj.weight, layer.gate_proj),
            (f"layers[{i}].up_proj", ref.mlp.up_proj.weight, layer.up_proj),
            (f"layers[{i}].down_proj", ref.mlp.down_proj.weight, layer.down_proj),
        ]
    pairs += [("norm", model.norm.weight, weights.norm), ("lm_head", reference.lm_head.weight, weights.lm_head)]
    return pairs


def same_bits(actual, expected):
    """Equal dtype, shape and bit pattern; expected is converted to actual's dtype first."""
    expected = expected.to(actual.dtype)
    integer = INTEGER[actual.dtype]
    return actual.shape == expected.shape and torch.equal(actual.cpu().view(integer), expected.view(integer))


def engine_tensors(weights):
    fields = [weights.embed_tokens, weights.norm, weights.lm_head]
    fields += [getattr(layer, f.name) for layer in weights.layers for f in dataclasses.fields(LayerWeights)]
    return fields


def load_reference(directory):
    reference, info = Qwen3ForCausalLM.from_pretrained(directory, dtype=torch.bfloat16, output_loading_info=True)
    assert {key: list(value) for key, value in info.items()} == {
        "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
    return reference


# ---------------------------------------------------------------- recorders

class Recorder:
    """Payload reads and torch.Tensor.to calls, recorded only inside `with recorder:`.

    Payload reads are seen only on handles opened through the safe_open names in ipe.checkpoint,
    ipe.weights, safetensors and safetensors.torch (used by load_file).
    """

    def __init__(self):
        self.active = False
        self.payload = []  # (file, tensor name)
        self.conversions = []  # results of .to calls that changed the dtype
        self.transfers = []  # results of .to calls that changed the device
        self.calls = 0

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *exc):
        self.active = False

    def payload_names(self):
        return {name for _, name in self.payload}


class RecordingSlice:
    def __init__(self, inner, recorder, file, name):
        self._inner, self._recorder, self._file, self._name = inner, recorder, file, name

    def get_dtype(self):
        return self._inner.get_dtype()

    def get_shape(self):
        return self._inner.get_shape()

    def __getitem__(self, index):
        if self._recorder.active:
            self._recorder.payload.append((self._file, self._name))
        return self._inner[index]


class RecordingHandle:
    def __init__(self, inner, recorder, file):
        self._inner, self._recorder, self._file = inner, recorder, file

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def keys(self):
        return self._inner.keys()

    def metadata(self):
        return self._inner.metadata()

    def get_slice(self, name):
        return RecordingSlice(self._inner.get_slice(name), self._recorder, self._file, name)

    def get_tensor(self, name):
        if self._recorder.active:
            self._recorder.payload.append((self._file, name))
        return self._inner.get_tensor(name)

    def get_tensors(self, *args, **kwargs):
        tensors = self._inner.get_tensors(*args, **kwargs)
        if self._recorder.active:
            self._recorder.payload.extend((self._file, name) for name in tensors)
        return tensors


@pytest.fixture
def recorder(monkeypatch):
    recorder = Recorder()
    for module in (ipe.checkpoint, ipe.weights, safetensors, safetensors.torch):
        def opener(path, *args, _open=module.safe_open, **kwargs):
            return RecordingHandle(_open(path, *args, **kwargs), recorder, Path(path).name)
        monkeypatch.setattr(module, "safe_open", opener)
    original = torch.Tensor.to

    def to(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not recorder.active:
            return result
        recorder.calls += 1
        if result.dtype != self.dtype:
            recorder.conversions.append(result)
        if result.device != self.device:
            recorder.transfers.append(result)
        return result

    monkeypatch.setattr(torch.Tensor, "to", to)
    return recorder


def test_recorder_sees_payload_reads_and_classifies_to_calls(tmp_path, recorder):
    write_file(tmp_path / "f.safetensors", {"x": torch.ones(2, dtype=torch.bfloat16)})
    with recorder, ipe.weights.safe_open(str(tmp_path / "f.safetensors"), framework="pt") as handle:
        x = handle.get_tensor("x")
        handle.get_slice("x")[0:1]
        x.to(torch.bfloat16)
        x.to(torch.float32)
        x.to("cpu")
        safetensors.torch.load_file(tmp_path / "f.safetensors")
    x.to(torch.float64)  # outside the recording window
    assert recorder.payload == [("f.safetensors", "x")] * 3
    assert (recorder.calls, len(recorder.conversions), len(recorder.transfers)) == (3, 1, 0)


# ---------------------------------------------------------------- failure ordering

def test_layer_fields_match_the_field_table():
    assert [field.name for field in dataclasses.fields(LayerWeights)] == list(LAYER_TENSORS)


DOWN = "model.layers.1.mlp.down_proj.weight"  # stored in the second shard


def later_shard_case(directory, kind):
    """A narrow sharded checkpoint whose first shard is valid and whose second has one problem."""
    document = CONFIGS["narrow"]
    base, unused = distinct_tensors(reference_shapes(document))
    tensors = with_head(base, unused, "equal")
    second = [n for n in tensors if n.startswith("model.layers.1.")] + ["model.norm.weight", HEAD]
    weight_map = {name: SHARDS[name in second] for name in tensors}
    if kind == "dtype":
        tensors[DOWN] = tensors[DOWN].float()
    elif kind == "shape":
        tensors[DOWN] = tensors[DOWN].t().contiguous()
    elif kind == "unexpected":
        tensors[DOWN.replace("weight", "bias")] = torch.ones(12, dtype=torch.bfloat16)
        second.append(DOWN.replace("weight", "bias"))
        weight_map[DOWN.replace("weight", "bias")] = SHARDS[1]
    elif kind == "missing":  # still mapped to the second shard by the index
        del tensors[DOWN]
    else:
        assert kind == "valid"
    write_checkpoint(directory, document, tensors, second_shard=second, weight_map=weight_map)
    return tensors


@pytest.mark.parametrize("kind", ["dtype", "shape", "unexpected", "missing"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_problem_in_a_later_shard_is_rejected_before_any_payload_read_or_conversion(
        tmp_path, recorder, kind, dtype):
    later_shard_case(tmp_path / "ckpt", kind)
    result = None
    with recorder, pytest.raises(CheckpointError) as error:
        result = load_checkpoint(tmp_path / "ckpt", dtype=dtype, device="cpu")
    named = DOWN.replace("weight", "bias") if kind == "unexpected" else DOWN
    assert named in str(error.value)
    if kind != "missing":  # a tensor that no shard stores has no file to name
        assert SHARDS[1] in str(error.value)
    assert result is None
    assert recorder.payload == []
    assert recorder.calls == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=needs_mps)])
def test_positive_control_reads_every_tensor_and_converts_each_field_once(tmp_path, recorder, device):
    tensors = later_shard_case(tmp_path / "ckpt", "valid")
    with recorder:
        weights = load_checkpoint(tmp_path / "ckpt", dtype=torch.float32, device=device).weights
    assert recorder.payload_names() == set(tensors)
    fields = {id(tensor) for tensor in engine_tensors(weights)}  # the tied head counts as the embedding
    assert len(fields) == len(tensors) - 1
    assert len(recorder.conversions) == len(fields)
    if device == "cpu":
        assert {id(t) for t in recorder.conversions} == fields
        assert recorder.transfers == []  # a dtype-only .to on the CPU is not a transfer
    else:
        assert {t.device.type for t in recorder.conversions} == {"cpu"}
        assert {id(t) for t in recorder.transfers} == fields
        assert len(recorder.transfers) == len(fields)


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8, "bfloat16", None])
def test_unsupported_dtypes_are_rejected_before_any_file_is_opened(tmp_path, dtype):
    with pytest.raises(ValueError, match="dtype"):
        load_checkpoint(tmp_path / "does-not-exist", dtype=dtype, device="cpu")


@pytest.mark.parametrize("device", [
    "meta", "tpu", "not a device", 3.5, 0, "cpu:1", "mps:1", torch.device("mps", 3),
    pytest.param("cuda", marks=pytest.mark.skipif(torch.cuda.is_available(), reason="has CUDA")),
])
def test_unsupported_devices_are_rejected_before_any_file_is_opened(tmp_path, device):
    with pytest.raises(ValueError, match="device"):
        load_checkpoint(tmp_path / "does-not-exist", dtype=torch.float32, device=device)


def test_float64_is_rejected_on_mps_even_when_mps_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(ValueError, match="float64"):
        load_checkpoint(tmp_path / "does-not-exist", dtype=torch.float64, device="mps")


def test_configuration_errors_come_before_checkpoint_errors(tmp_path, recorder):
    synthetic(tmp_path / "ckpt")
    (tmp_path / "ckpt" / "config.json").write_text(json.dumps({**CONFIGS["narrow"], "attention_bias": True}))
    (tmp_path / "ckpt" / "model.safetensors").write_bytes(b"not a checkpoint")
    with recorder, pytest.raises(ConfigError, match="attention_bias"):
        load_checkpoint(tmp_path / "ckpt", dtype=torch.float32, device="cpu")
    assert recorder.payload == [] and recorder.calls == 0


def test_a_missing_directory_raises_an_os_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing", dtype=torch.float32, device="cpu")


def test_an_invalid_generation_config_is_never_read(tmp_path):
    synthetic(tmp_path / "ckpt")
    (tmp_path / "ckpt" / "generation_config.json").write_text('{"do_sample": true, "temperature": ')
    assert load_checkpoint(tmp_path / "ckpt", dtype=torch.float32, device="cpu").tie.tied


@pytest.mark.parametrize(("head", "fail_in", "fail_on_call"), [
    ("equal", "to", 5), ("equal", "to", 6), ("different", "tie", 1),
])
def test_a_failure_while_loading_releases_every_tensor_built_so_far(tmp_path, monkeypatch, head, fail_in, fail_on_call):
    # Odd .to calls convert and even ones transfer, so failing on call 6 leaves call 5's result in
    # flight. "tie" fails inside the embedding-head comparison, whose frames hold both stored
    # views. pytest.raises keeps the traceback alive; it must keep no tensor alive.
    synthetic(tmp_path / "ckpt", head=head)
    built = []
    original_to, original_maximum, original_open = torch.Tensor.to, torch.maximum, ipe.weights.safe_open
    calls = 0

    def fail_here():
        nonlocal calls
        calls += 1
        return calls == fail_on_call

    def to(self, *args, **kwargs):
        if fail_in == "to" and fail_here():
            del self  # keep this test's own frame from holding the tensor
            raise RuntimeError("injected failure")
        result = original_to(self, *args, **kwargs)
        built.append(weakref.ref(result))
        return result

    def maximum(*args, **kwargs):
        if fail_in == "tie" and fail_here():
            del args, kwargs
            raise RuntimeError("injected failure")
        return original_maximum(*args, **kwargs)

    class ViewRecorder:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def get_tensor(self, name):
            tensor = self._inner.get_tensor(name)
            built.append(weakref.ref(tensor))
            return tensor

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch, "maximum", maximum)
    monkeypatch.setattr(ipe.weights, "safe_open", lambda *a, **k: ViewRecorder(original_open(*a, **k)))
    with pytest.raises(RuntimeError, match="injected failure") as error:
        load_checkpoint(tmp_path / "ckpt", dtype=torch.float32, device="cpu")
    gc.collect()
    assert error.value.__traceback__ is not None
    assert calls == fail_on_call
    assert len(built) >= 2 * 11 + 3  # a view of every stored tensor, plus any conversions
    assert [ref for ref in built if ref() is not None] == []


def test_ipe_weights_never_imports_transformers(tmp_path):
    synthetic(tmp_path / "ckpt")
    code = ("import sys, torch, ipe, ipe.weights\n"
            f"ipe.weights.load_checkpoint({str(tmp_path / 'ckpt')!r}, dtype=torch.float32, device='cpu')\n"
            "assert 'transformers' not in sys.modules, 'transformers imported'\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    package = subprocess.run([sys.executable, "-c", "import sys, ipe; print('ipe.weights' in sys.modules)"],
                             check=True, env=env, capture_output=True, text=True)
    assert package.stdout.strip() == "False"


# ---------------------------------------------------------------- exactness, tying and identity

@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_every_field_holds_its_own_stored_tensor_exactly(tmp_path, config, device, dtype):
    tensors = synthetic(tmp_path / "ckpt", config, "different", sharded=config == "narrow")
    weights = load_checkpoint(tmp_path / "ckpt", dtype=dtype, device=device).weights
    pairs = by_checkpoint_name(weights, include_head=True)
    assert len(pairs) == len(tensors)
    for name, field in pairs:
        assert field.dtype == dtype and field.device.type == device, name
        assert same_bits(field, tensors[name]), name


TIE_RECORDS = {
    "absent": TieRecord(stored=False, tied=True, max_abs_diff=None),
    "equal": TieRecord(stored=True, tied=True, max_abs_diff=0.0),
    "signed_zero": TieRecord(stored=True, tied=True, max_abs_diff=0.0),
}


@pytest.mark.parametrize("head", list(TIE_RECORDS))
@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_a_tied_head_is_the_embedding_object_at_every_dtype_and_device(tmp_path, head, device, dtype):
    tensors = synthetic(tmp_path / "ckpt", head=head)
    loaded = load_checkpoint(tmp_path / "ckpt", dtype=dtype, device=device)
    weights = loaded.weights
    assert loaded.tie == TIE_RECORDS[head]
    assert weights.lm_head is weights.embed_tokens
    assert weights.lm_head.data_ptr() == weights.embed_tokens.data_ptr()
    assert weights.lm_head.dtype == dtype and weights.lm_head.device.type == device
    assert same_bits(weights.lm_head, tensors[EMBED])  # a signed-zero head takes the embedding's +0


@pytest.mark.parametrize("head", ["different", "last_element", "nan"])
@pytest.mark.parametrize(("device", "dtype"), DEVICES)
def test_an_unequal_head_is_used_untied_and_exactly(tmp_path, monkeypatch, head, device, dtype):
    monkeypatch.setattr(ipe.weights, "_DIFF_ROWS", 5)  # 32 vocabulary rows span 7 chunks
    tensors = synthetic(tmp_path / "ckpt", head=head)
    loaded = load_checkpoint(tmp_path / "ckpt", dtype=dtype, device=device)
    weights = loaded.weights
    assert weights.lm_head is not weights.embed_tokens
    assert weights.lm_head.untyped_storage().data_ptr() != weights.embed_tokens.untyped_storage().data_ptr()
    assert same_bits(weights.lm_head, tensors[HEAD]) and same_bits(weights.embed_tokens, tensors[EMBED])
    tie = loaded.tie
    assert (tie.stored, tie.tied) == (True, False)
    if head == "nan":  # NaN is unequal to itself, though the bytes match
        assert math.isnan(tie.max_abs_diff)
    else:
        expected = (tensors[EMBED].float() - tensors[HEAD].float()).abs().max().item()
        assert tie.max_abs_diff == expected > 0


def test_the_difference_is_computed_across_row_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(ipe.weights, "_DIFF_ROWS", 5)
    document = CONFIGS["narrow"]
    base, unused = distinct_tensors(reference_shapes(document))
    tensors = with_head(base, unused, "equal")
    # Rows 0, 14 and 31: the first chunk, the last row of a middle chunk, the last chunk. The
    # largest difference, 256 - 1.0078125 = 254.9921875, is exact in fp32 and not in bf16.
    for (row, column), (embed, head) in {(0, 0): (1.0, 2.0), (14, 3): (256.0, 1.0078125),
                                         (31, 11): (1.0, 3.0)}.items():
        tensors[EMBED][row, column], tensors[HEAD][row, column] = embed, head
    write_checkpoint(tmp_path / "ckpt", document, tensors)
    tie = load_checkpoint(tmp_path / "ckpt", dtype=torch.bfloat16, device="cpu").tie
    assert (tie.tied, tie.max_abs_diff) == (False, 254.9921875)


def test_matching_infinities_do_not_make_the_difference_nan(tmp_path):
    document = CONFIGS["narrow"]
    base, unused = distinct_tensors(reference_shapes(document))
    tensors = with_head(base, unused, "equal")
    tensors[EMBED][2, 2] = tensors[HEAD][2, 2] = float("inf")
    tensors[EMBED][5, 5], tensors[HEAD][5, 5] = 1.0, 3.0
    write_checkpoint(tmp_path / "ckpt", document, tensors)
    tie = load_checkpoint(tmp_path / "ckpt", dtype=torch.bfloat16, device="cpu").tie
    assert (tie.tied, tie.max_abs_diff) == (False, 2.0)


def test_a_head_without_an_embedding_is_rejected(tmp_path):
    synthetic(tmp_path / "ckpt", head="head_only")
    with pytest.raises(CheckpointError, match=EMBED):
        load_checkpoint(tmp_path / "ckpt", dtype=torch.float32, device="cpu")


# ---------------------------------------------------------------- parity with Transformers

@pytest.mark.parametrize(("config", "head"), [
    ("narrow", "different"), ("square", "different"), ("deep", "different"),
    ("narrow", "absent"), ("narrow", "equal"), ("narrow", "signed_zero"), ("narrow", "last_element"),
    ("narrow", "nan"),
])
def test_every_field_matches_transformers_by_attribute_path(tmp_path, config, head):
    synthetic(tmp_path / "ckpt", config, head, sharded=config == "narrow")
    weights = load_checkpoint(tmp_path / "ckpt", dtype=torch.bfloat16, device="cpu").weights
    reference = load_reference(tmp_path / "ckpt")
    pairs = by_attribute_path(reference, weights)
    assert {id(p) for p in reference.parameters()} <= {id(ref) for _, ref, _ in pairs}
    assert len(pairs) == 11 * CONFIGS[config]["num_hidden_layers"] + 3
    for label, ref, field in pairs:
        assert same_bits(field, ref.detach()), label
    assert (reference.lm_head.weight is reference.model.embed_tokens.weight) == (weights.lm_head is weights.embed_tokens)


# ---------------------------------------------------------------- downloaded checkpoints

REVISIONS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
}
CHUNK_ROWS = 4096


class StoredCheckpoints:
    """Verified directories and their stored tensors as bf16 views, read without ipe.weights."""

    def __init__(self):
        self.views = {}

    def directory(self, name):
        return WEIGHTS / name

    def get(self, name, record):
        if name not in self.views:
            directory = WEIGHTS / name
            if not directory.is_dir():
                pytest.fail(f"{directory} is missing; download it as described in the README")
            pin = load_pin(PINS / f"{name}.json")
            assert (pin.repo_id, pin.revision) == REVISIONS[name]
            verify_files(directory, pin)
            record(name, f"{pin.repo_id}@{pin.revision} verified")
            self.views[name] = read_views(directory)
        return self.views[name]


def read_views(directory):
    views = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open handles are not iterable
                views[name] = handle.get_tensor(name)
    return views


def chunks_equal(actual, expected):
    """same_bits over row chunks, bounding the converted temporaries."""
    if actual.shape != expected.shape:
        return False
    for start in range(0, max(actual.shape[0], 1), CHUNK_ROWS):
        if not same_bits(actual[start:start + CHUNK_ROWS], expected[start:start + CHUNK_ROWS]):
            return False
    return True


@pytest.fixture(scope="session")
def stored(record_testsuite_property):
    checkpoints = StoredCheckpoints()
    yield lambda name: checkpoints.get(name, record_testsuite_property)
    # Session end: the files still match their pins and the held views were never written.
    for name, views in checkpoints.views.items():
        directory = checkpoints.directory(name)
        verify_files(directory, load_pin(PINS / f"{name}.json"))
        fresh = read_views(directory)
        assert fresh.keys() == views.keys()
        for tensor, view in views.items():
            assert torch.equal(view.view(torch.int16), fresh[tensor].view(torch.int16)), tensor


REAL = pytest.mark.parametrize("name", list(REVISIONS))


@pytest.mark.checkpoint
@REAL
def test_downloaded_checkpoint_tie_state(stored, name, record_testsuite_property):
    views = stored(name)
    embed, head = views[EMBED], views[HEAD]
    tie = load_checkpoint(WEIGHTS / name, dtype=torch.bfloat16, device="cpu").tie
    record_testsuite_property(f"{name} tie", repr(tie))
    assert tie.stored and tie.tied == torch.equal(embed, head)
    # The observed state of both pinned revisions; equal bytes mean the tie drops no stored value.
    assert tie == TieRecord(stored=True, tied=True, max_abs_diff=0.0)
    assert torch.equal(embed.view(torch.int16), head.view(torch.int16))


@pytest.mark.checkpoint
@REAL
def test_downloaded_values_are_finite(stored, name):
    # Chunked: an unchunked isfinite needs about 5 bytes per element (1.6 GB for the 1.7B embedding).
    for tensor, view in stored(name).items():
        flat = view.reshape(view.shape[0], -1) if view.dim() > 1 else view.reshape(-1, 1)
        for start in range(0, flat.shape[0], CHUNK_ROWS):
            assert torch.isfinite(flat[start:start + CHUNK_ROWS]).all(), tensor


# Large loads live only inside these helpers, which return a list of problems, so a failing
# assertion cannot keep a load alive through its traceback.

def exactness_problems(name, dtype, views):
    loaded = load_checkpoint(WEIGHTS / name, dtype=dtype, device="cpu")
    weights = loaded.weights
    pairs = by_checkpoint_name(weights, include_head=not loaded.tie.tied)
    problems = [] if len(pairs) == len(views) - loaded.tie.tied else [f"{len(pairs)} fields"]
    problems += [tensor for tensor, field in pairs if field.dtype != dtype or not chunks_equal(field, views[tensor])]
    if (weights.lm_head is weights.embed_tokens) != loaded.tie.tied:
        problems.append("lm_head identity")
    return problems


def parity_problems(name):
    weights = load_checkpoint(WEIGHTS / name, dtype=torch.bfloat16, device="cpu").weights
    reference = load_reference(WEIGHTS / name)
    pairs = by_attribute_path(reference, weights)
    problems = [] if {id(p) for p in reference.parameters()} <= {id(ref) for _, ref, _ in pairs} else ["coverage"]
    problems += [label for label, ref, field in pairs if not same_bits(field, ref.detach())]
    if (reference.lm_head.weight is reference.model.embed_tokens.weight) != (weights.lm_head is weights.embed_tokens):
        problems.append("tie state")
    return problems


EXACTNESS_CHILD = """
import json, sys
import torch
sys.path.insert(0, sys.argv[1])
import test_weights
name, dtype = sys.argv[2], getattr(torch, sys.argv[3])
print(json.dumps(test_weights.exactness_problems(name, dtype, test_weights.read_views(test_weights.WEIGHTS / name))))
"""


@pytest.mark.checkpoint
@REAL
@pytest.mark.parametrize("dtype", DTYPES)
def test_downloaded_fields_hold_their_stored_tensors_exactly(stored, name, dtype):
    # A fresh process per case: the fp64 Qwen3-1.7B load holds 13.8 GB and, beside this session's
    # views, swaps a 24 GiB machine. The child imports this ipe package and reads the files this
    # session verified.
    stored(name)  # verifies the directory first
    env = {**os.environ, "PYTHONPATH": str(Path(ipe.weights.__file__).parents[1])}
    arguments = [str(Path(__file__).parent), name, str(dtype).removeprefix("torch.")]
    child = subprocess.run([sys.executable, "-c", EXACTNESS_CHILD, *arguments],
                           check=False, env=env, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout.splitlines()[-1]) == []


@pytest.mark.checkpoint
@REAL
def test_downloaded_fields_match_transformers_by_attribute_path(stored, name):
    stored(name)  # verifies the directory first
    problems = parity_problems(name)
    gc.collect()
    assert problems == []
