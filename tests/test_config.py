"""Configuration admission, local JSON parsing, and dependency-isolation contracts."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from ipe.config import ConfigError, Qwen3Config, load_config, validate_config

FIXTURES = Path(__file__).parent / "fixtures"
DIMENSIONS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
)


@pytest.fixture
def small_config():
    # Attention projection width is 4 * 8 = 32, deliberately different from hidden_size.
    return {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 12,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
    }


def test_two_layer_structural_config_is_accepted(small_config):
    result = validate_config(small_config)
    assert isinstance(result, Qwen3Config)
    assert is_dataclass(result)
    assert result.vocab_size == 32
    assert result.hidden_size == 12
    assert result.intermediate_size == 24
    assert result.num_hidden_layers == 2
    assert result.num_attention_heads == 4
    assert result.num_key_value_heads == 2
    assert result.head_dim == 8
    assert result.max_position_embeddings == 64
    assert result.rms_norm_eps == 1e-6
    assert result.rope_theta == 10000
    with pytest.raises(FrozenInstanceError):
        result.hidden_size = 99


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_structural_attention_groupings_are_accepted(small_config, kv_heads):
    small_config["num_key_value_heads"] = kv_heads
    assert validate_config(small_config).num_key_value_heads == kv_heads


def test_valid_dimensions_and_scalar_parameters_are_not_checkpoint_whitelisted(small_config):
    small_config.update(
        hidden_size=13,
        intermediate_size=17,
        num_hidden_layers=3,
        vocab_size=41,
        max_position_embeddings=96,
        rms_norm_eps=1e-5,
        rope_theta=50000,
    )
    result = validate_config(small_config)
    assert (result.hidden_size, result.intermediate_size, result.num_hidden_layers) == (13, 17, 3)
    assert (result.vocab_size, result.max_position_embeddings) == (41, 96)
    assert (result.rms_norm_eps, result.rope_theta) == (1e-5, 50000)


@pytest.mark.parametrize(
    ("filename", "hidden_size", "intermediate_size"),
    [("qwen3-1.7b-config.json", 2048, 6144), ("qwen3-0.6b-config.json", 1024, 3072)],
)
def test_pinned_config_only_fixtures(filename, hidden_size, intermediate_size):
    result = load_config(FIXTURES / filename)
    assert result.hidden_size == hidden_size
    assert result.intermediate_size == intermediate_size
    assert result.num_hidden_layers == 28
    assert result.vocab_size == 151936
    assert result.num_attention_heads == 16
    assert result.num_key_value_heads == 8
    assert result.head_dim == 128
    assert result.rms_norm_eps == 1e-6
    assert result.rope_theta == 1000000


def test_validation_accepts_a_mapping_without_mutating_it(small_config):
    small_config["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000}
    small_config["layer_types"] = ["full_attention", "full_attention"]
    before = deepcopy(small_config)
    result = validate_config(MappingProxyType(small_config))
    assert small_config == before
    small_config["hidden_size"] = 100
    small_config["rope_parameters"]["rope_theta"] = 20000
    assert result.hidden_size == 12
    assert result.rope_theta == 10000


@pytest.mark.parametrize("value", [None, [], "{}", 3, True])
def test_non_mapping_input_is_rejected(value):
    with pytest.raises(ConfigError) as error:
        validate_config(value)
    assert str(error.value)
    assert isinstance(error.value, ValueError)


@pytest.mark.parametrize(
    "field",
    DIMENSIONS
    + ("model_type", "architectures", "hidden_act", "tie_word_embeddings", "rms_norm_eps"),
)
def test_required_fields_do_not_silently_take_reference_defaults(small_config, field):
    del small_config[field]
    with pytest.raises(ConfigError, match=field):
        validate_config(small_config)


@pytest.mark.parametrize("field", DIMENSIONS)
@pytest.mark.parametrize("value", [True, False, 0, -1, 2.0, "2", None, float("inf")])
def test_dimensions_require_positive_strict_integers(small_config, field, value):
    small_config[field] = value
    with pytest.raises(ConfigError, match=field):
        validate_config(small_config)


@pytest.mark.parametrize("kv_heads", [3, 5, 8])
def test_invalid_gqa_groups_are_rejected(small_config, kv_heads):
    small_config["num_key_value_heads"] = kv_heads
    with pytest.raises(ConfigError, match="num_attention_heads|num_key_value_heads"):
        validate_config(small_config)


@pytest.mark.parametrize("head_dim", [1, 3, 15])
def test_split_half_rotary_requires_an_even_head_dimension(small_config, head_dim):
    small_config["head_dim"] = head_dim
    with pytest.raises(ConfigError, match="head_dim"):
        validate_config(small_config)


@pytest.mark.parametrize("field", ["rms_norm_eps", "rope_theta"])
@pytest.mark.parametrize(
    "value", [True, False, 0, -1, "1e-6", None, float("nan"), float("inf"), -float("inf")]
)
def test_numeric_parameters_require_finite_positive_numbers(small_config, field, value):
    small_config[field] = value
    with pytest.raises(ConfigError, match=field):
        validate_config(small_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen3_moe"),
        ("model_type", "llama"),
        ("model_type", None),
        ("architectures", "Qwen3ForCausalLM"),
        ("architectures", []),
        ("architectures", ["Qwen3Model"]),
        ("architectures", ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"]),
        ("hidden_act", "gelu"),
        ("tie_word_embeddings", False),
        ("tie_word_embeddings", 1),
        ("attention_bias", True),
        ("attention_bias", 0),
        ("attention_dropout", 0.1),
        ("attention_dropout", -0.1),
        ("attention_dropout", False),
        ("attention_dropout", float("nan")),
        ("use_sliding_window", True),
        ("use_sliding_window", 0),
    ],
)
def test_unsupported_features_and_wrong_flag_types_are_rejected(small_config, field, value):
    small_config[field] = value
    with pytest.raises(ConfigError, match=field):
        validate_config(small_config)


@pytest.mark.parametrize("sliding_window", [None, 1, 4096])
def test_disabled_window_metadata_is_inert(small_config, sliding_window):
    baseline = validate_config(small_config)
    small_config.update(
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=sliding_window,
        max_window_layers=0,
        layer_types=["full_attention", "full_attention"],
    )
    assert validate_config(small_config) == baseline


@pytest.mark.parametrize("value", [False, 0, -1, 1.0, "4096"])
def test_disabled_window_still_requires_well_typed_metadata(small_config, value):
    small_config["sliding_window"] = value
    with pytest.raises(ConfigError, match="sliding_window"):
        validate_config(small_config)


@pytest.mark.parametrize("value", [False, -1, 1.0, "1", None])
def test_max_window_layers_requires_a_nonnegative_integer(small_config, value):
    small_config["max_window_layers"] = value
    with pytest.raises(ConfigError, match="max_window_layers"):
        validate_config(small_config)


@pytest.mark.parametrize(
    "value",
    [
        "full_attention",
        ["full_attention"],
        ["full_attention"] * 3,
        ["full_attention", "sliding_attention"],
        ["full_attention", "future_attention"],
        ["full_attention", None],
    ],
)
def test_layer_types_cannot_hide_an_unsupported_layer(small_config, value):
    small_config.update(use_sliding_window=False, sliding_window=None, layer_types=value)
    with pytest.raises(ConfigError, match="layer_types"):
        validate_config(small_config)


def test_null_optional_rope_and_layer_metadata_is_accepted(small_config):
    baseline = validate_config(small_config)
    small_config.update(rope_scaling=None, rope_parameters=None, layer_types=None)
    assert validate_config(small_config) == baseline


@pytest.mark.parametrize("container", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize("type_key", ["type", "rope_type"])
def test_unscaled_rope_aliases_normalize_without_mutation(small_config, container, type_key):
    baseline = validate_config(small_config)
    del small_config["rope_theta"]
    small_config[container] = {
        type_key: "default",
        "rope_theta": 10000,
        "partial_rotary_factor": 1.0,
    }
    before = deepcopy(small_config)
    assert validate_config(small_config) == baseline
    assert small_config == before


def test_rope_scaling_and_rope_parameters_cannot_both_be_set(small_config):
    # Transformers 5.17.0 reads rope_scaling and ignores rope_parameters when both are set, so a
    # theta given only in rope_parameters would run at its default of 10000 in the reference.
    del small_config["rope_theta"]
    small_config["rope_scaling"] = {"type": "default"}
    small_config["rope_parameters"] = {"rope_type": "default", "rope_theta": 1e6}
    with pytest.raises(ConfigError, match="rope_scaling"):
        validate_config(small_config)


def test_empty_rope_scaling_defers_to_rope_parameters(small_config):
    baseline = validate_config(small_config)
    del small_config["rope_theta"]
    small_config["rope_scaling"] = {}
    small_config["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000}
    assert validate_config(small_config) == baseline


def test_rope_theta_must_be_supplied_somewhere(small_config):
    del small_config["rope_theta"]
    small_config["rope_parameters"] = {"rope_type": "default"}
    with pytest.raises(ConfigError, match="rope_theta"):
        validate_config(small_config)


@pytest.mark.parametrize("container", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize(
    "value",
    [
        [],
        "default",
        {"rope_type": "linear", "factor": 1.0},
        {"rope_type": "dynamic"},
        {"rope_type": "yarn"},
        {"rope_type": "default", "factor": 1.0},
        {"rope_type": "default", "mystery_scaling": None},
        {"full_attention": {"rope_type": "default", "rope_theta": 10000}},
        {"type": "default", "rope_type": "linear"},
    ],
)
def test_unsupported_rope_forms_are_rejected(small_config, container, value):
    small_config[container] = value
    with pytest.raises(ConfigError, match=container):
        validate_config(small_config)


@pytest.mark.parametrize("value", [1, 1.0])
def test_top_level_full_rotary_factor_is_accepted(small_config, value):
    baseline = validate_config(small_config)
    small_config["partial_rotary_factor"] = value
    assert validate_config(small_config) == baseline


@pytest.mark.parametrize(
    "value", [0, 0.5, 2, -1, True, False, "1", None, float("nan"), float("inf"), -float("inf"), [], {}]
)
def test_top_level_partial_rotary_factor_is_rejected(small_config, value):
    small_config["partial_rotary_factor"] = value
    with pytest.raises(ConfigError, match="partial_rotary_factor"):
        validate_config(small_config)


@pytest.mark.parametrize("container", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize("value", [0, 0.5, 2, True, "1", float("nan")])
def test_partial_rotary_is_rejected(small_config, container, value):
    small_config[container] = {"rope_type": "default", "partial_rotary_factor": value}
    with pytest.raises(ConfigError, match="partial_rotary_factor"):
        validate_config(small_config)


@pytest.mark.parametrize("container", ["rope_scaling", "rope_parameters"])
def test_top_and_nested_theta_conflicts_are_rejected(small_config, container):
    small_config[container] = {"rope_type": "default", "rope_theta": 20000}
    with pytest.raises(ConfigError, match="rope_theta"):
        validate_config(small_config)


@pytest.mark.parametrize("container", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize("value", [False, 0, "10000", float("nan"), float("inf")])
def test_nested_theta_does_not_bypass_numeric_validation(small_config, container, value):
    del small_config["rope_theta"]
    small_config[container] = {"rope_type": "default", "rope_theta": value}
    with pytest.raises(ConfigError, match="rope_theta"):
        validate_config(small_config)


def test_documented_metadata_does_not_choose_execution_or_generation(small_config):
    baseline = validate_config(small_config)
    small_config.update(
        bos_token_id=1,
        eos_token_id=[2, 3],
        pad_token_id=None,
        dtype="float32",
        torch_dtype="bfloat16",
        transformers_version="5.17.0",
        initializer_range=0.02,
        use_cache=False,
        _name_or_path="fixture-only",
        _commit_hash="a" * 40,
    )
    assert validate_config(small_config) == baseline


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attention_bais", True),
        ("quantization_config", {}),
        ("quantization_config", None),
        ("mlp_bias", False),
        ("num_experts", 0),
        ("num_experts_per_tok", 2),
        ("auto_map", {"AutoModelForCausalLM": "remote.Model"}),
        ("do_sample", True),
        ("temperature", 0.6),
        ("future_feature", None),
    ],
)
def test_unknown_fields_are_not_silently_discarded(small_config, field, value):
    small_config[field] = value
    with pytest.raises(ConfigError, match=field):
        validate_config(small_config)


def test_load_reads_local_utf8_json_and_accepts_path_or_string(tmp_path, small_config):
    small_config["_name_or_path"] = "配置-é"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(small_config, ensure_ascii=False), encoding="utf-8")
    expected = validate_config(small_config)
    assert load_config(path) == expected
    assert load_config(str(path)) == expected


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ('{"hidden_size": 12, "hidden_size": 16}', "hidden_size"),
        ('{"rope_parameters": {"rope_theta": 10000, "rope_theta": 20000}}', "rope_theta"),
    ],
)
def test_load_rejects_duplicate_keys_at_any_depth(tmp_path, text, field):
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=field):
        load_config(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_nonfinite_constants_are_rejected_even_in_ignored_metadata(
    tmp_path, small_config, constant
):
    text = json.dumps(small_config)
    text = text[:-1] + ', "initializer_range": ' + constant + "}"
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert str(error.value)


@pytest.mark.parametrize("text", ["", "{", '{"model_type":}', "[]", "null", '"qwen3"', "1"])
def test_load_rejects_malformed_json_and_non_object_roots(tmp_path, text):
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert str(error.value)


def test_missing_config_preserves_the_normal_filesystem_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


def test_package_and_validator_import_without_third_party_runtime_dependencies(small_config):
    package = Path(__file__).resolve().parents[1] / "src" / "ipe"
    # -I ignores external Python path settings; -S excludes site-packages. Loading the real
    # package by its spec exercises __init__ and relative imports without sys.path injection.
    script = """
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

class RejectModelDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'numpy', 'safetensors', 'tokenizers'}:
            raise AssertionError('forbidden dependency import: ' + fullname)
        return None

sys.meta_path.insert(0, RejectModelDependencies())
package = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    'ipe', package / '__init__.py', submodule_search_locations=[str(package)]
)
module = importlib.util.module_from_spec(spec)
sys.modules['ipe'] = module
spec.loader.exec_module(module)
config_module = importlib.import_module('ipe.config')
result = config_module.validate_config(json.loads(sys.argv[2]))
assert result.hidden_size == 12 and result.head_dim == 8 and result.num_hidden_layers == 2
assert not any(name.split('.')[0] in {'torch', 'transformers'} for name in sys.modules)
print('stdlib-only import and validation passed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(package), json.dumps(small_config)],
        cwd=package.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "stdlib-only import and validation passed"


@pytest.mark.parametrize(
    ("text", "decoder_error_type"),
    [
        ('{"vocab_size": ' + "9" * 5000 + "}", ValueError),
        ('{"_name_or_path": ' + "[" * 10000 + "0" + "]" * 10000 + "}", RecursionError),
    ],
    ids=["oversized-integer", "excessive-nesting"],
)
def test_json_decoder_limits_produce_config_errors_with_path(tmp_path, text, decoder_error_type):
    # Establish that the payload actually hits the decoder limit in this interpreter.
    with pytest.raises(decoder_error_type) as decoder_error:
        json.loads(text)
    assert type(decoder_error.value) is decoder_error_type

    path = tmp_path / "decoder-limit.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert str(path) in str(error.value)
    cause = error.value.__cause__
    # ConfigError subclasses ValueError; an unrelated admission error is not this cause.
    assert cause is not None
    assert type(cause) is decoder_error_type
    assert cause.args == decoder_error.value.args
    assert error.value.__context__ is cause
    # Explicit chaining sets this flag too; from None is rejected by the cause assertions.
    assert error.value.__suppress_context__ is True
