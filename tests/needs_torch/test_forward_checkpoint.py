"""The forward pass on the pinned checkpoints: bitwise parity with Transformers 5.17.0, the fp64
oracle against the patched reference, an fp64 layer within its error bound, operators on
captured activations, and logit selection.

Parity cases run in child processes (forward_child.py) so the reference and the engine are never
resident together. The engine child loads bf16 file views and the reference child loads
Transformers at the run dtype, both from the directory this session verified.

What these tests leave unproven: other checkpoints and lengths, GPU kernels, the cached path.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import bounded as bd
import pytest
import torch
from forward_child import token_rows
from safetensors import safe_open
from test_forward import check_selected_rows, same_bits
from test_weights import WEIGHTS
from transformers import Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as reference

import ipe.forward
from ipe import ops
from ipe.forward import decoder_layer, final_hidden, forward_full
from ipe.weights import load_checkpoint

CHILD = Path(__file__).parent / "forward_child.py"
F64 = torch.float64

# (checkpoint, run dtype, engine backend, length, batch)
PARITY = [
    *(("qwen3-0.6b", dtype, backend, length, 2) for dtype in ("bfloat16", "float32")
      for backend in ("math", "sdpa") for length in (1, 1024)),
    *(("qwen3-1.7b", "float32", backend, length, 1) for backend in ("math", "sdpa") for length in (1024, 4224)),
    *(("qwen3-1.7b", "bfloat16", backend, 1024, 1) for backend in ("math", "sdpa")),
]
ORACLE = [("qwen3-0.6b", "float64", backend, 1024, 1) for backend in ("math", "sdpa")]
ATTENTION = {"math": "eager", "sdpa": "sdpa"}


def run_child(*arguments):
    env = {**os.environ, "PYTHONPATH": str(Path(ipe.forward.__file__).parents[1])}
    child = subprocess.run([sys.executable, str(CHILD), *map(str, arguments)], check=False, env=env,
                           capture_output=True, text=True)
    assert child.returncode == 0, child.stderr[-4000:]
    return child.stdout


def parity_problems(tmp_path, name, dtype, backend, length, batch, *, patched=False):
    directory = WEIGHTS / name
    saved = tmp_path / "reference.pt"
    run_child("reference", directory, dtype, ATTENTION[backend], length, batch, saved, *(["patched"] if patched else []))
    output = run_child("engine", directory, dtype, backend, length, batch, saved)
    saved.unlink()
    return json.loads(output.splitlines()[-1])


@pytest.mark.checkpoint
@pytest.mark.parametrize(("name", "dtype", "backend", "length", "batch"), PARITY)
def test_every_layer_and_the_logits_match_the_reference_bitwise(stored, tmp_path, name, dtype, backend, length, batch):
    stored(name)  # verifies the directory first
    assert parity_problems(tmp_path, name, dtype, backend, length, batch) == []


@pytest.mark.checkpoint
@pytest.mark.parametrize(("name", "dtype", "backend", "length", "batch"), ORACLE)
def test_the_oracle_matches_the_patched_reference_bitwise(stored, tmp_path, name, dtype, backend, length, batch):
    stored(name)
    assert parity_problems(tmp_path, name, dtype, backend, length, batch, patched=True) == []


def stored_layer(directory, index):
    prefix = f"model.layers.{index}."
    weights = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            weights |= {name.removeprefix(prefix): handle.get_tensor(name).double().numpy()
                        for name in handle.keys() if name.startswith(prefix)}  # noqa: SIM118 - not iterable
    return weights


@pytest.mark.checkpoint
@pytest.mark.parametrize("layer", (0, 27))
def test_an_fp64_layer_of_the_development_checkpoint_is_within_its_error_bound(stored, layer):
    stored("qwen3-0.6b")
    directory = WEIGHTS / "qwen3-0.6b"
    loaded = load_checkpoint(directory, dtype=torch.bfloat16, device="cpu")
    token_ids = token_rows(directory, 64, 1)
    inputs = {}
    final_hidden(loaded, token_ids, dtype=F64, capture=lambda index, output: inputs.setdefault(index + 1, output))
    x = loaded.weights.embed_tokens[token_ids].to(F64) if layer == 0 else inputs[layer]
    positions = torch.arange(64)
    rope = ops.rotary_tables(positions, loaded.config.head_dim, loaded.config.rope_theta, F64)
    actual = decoder_layer(x, loaded.weights.layers[layer], rope, loaded.config, backend="math")
    expected = bd.decoder_layer(bd.exact(x.numpy()), stored_layer(directory, layer), positions.numpy(), loaded.config, bd.FP64)
    assert bd.ratio(actual.numpy(), expected, 2.0) <= 1.0


# The two checks below load Transformers' Qwen3-0.6B. They run in a child process so the pytest
# process never holds a model: in-process runs preceded swap in the later 1.7B children.
CHECK_CHILD = """
import json, sys
import torch
sys.path.insert(0, sys.argv[1])
import ipe
import test_forward_checkpoint as checks
print(json.dumps({"ipe": ipe.__file__}), flush=True)
records = []
getattr(checks, sys.argv[2])(getattr(torch, sys.argv[3]), lambda name, value: records.append([name, value]))
print(json.dumps({"records": records}))
"""


def run_check(check, dtype, source=None):
    """Run a check in a child that imports ipe from source (default: the package under test)."""
    source = source or Path(ipe.forward.__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(source)}
    return subprocess.run([sys.executable, "-c", CHECK_CHILD, str(Path(__file__).parent), check,
                           str(dtype).removeprefix("torch.")], check=False, env=env, capture_output=True, text=True)


def imported_package(child):
    return Path(json.loads(child.stdout.splitlines()[0])["ipe"]).parent


def child_result(child):
    assert child.returncode == 0, child.stderr[-4000:]
    assert imported_package(child) == Path(ipe.forward.__file__).parent  # the intended sources
    return json.loads(child.stdout.splitlines()[-1])


@pytest.mark.checkpoint
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_operators_match_the_reference_on_captured_activations(stored, dtype):
    stored("qwen3-0.6b")
    child_result(run_check("captured_activation_check", dtype))


def captured_activation_check(dtype, record):
    directory = WEIGHTS / "qwen3-0.6b"
    model = Qwen3ForCausalLM.from_pretrained(directory, dtype=dtype, attn_implementation="eager").eval()
    captured = []
    def keep(module, args, kwargs, output):
        captured.append((module, args[0] if args else kwargs["hidden_states"], output))
    attention = []
    def keep_attention(module, args, kwargs, output):
        attention.append((module, kwargs["position_embeddings"], output))
    layers = (model.model.layers[0], model.model.layers[20])
    hooks = [m.register_forward_hook(keep, with_kwargs=True) for layer in layers
             for m in (layer.input_layernorm, layer.self_attn.q_norm, layer.self_attn.k_norm, layer.self_attn.v_proj,
                       layer.mlp, layer.post_attention_layernorm)]
    hooks += [layer.self_attn.register_forward_hook(keep_attention, with_kwargs=True) for layer in layers]
    with torch.no_grad():
        model(input_ids=token_rows(directory, 256, 1), use_cache=False)
        for hook in hooks:
            hook.remove()
        eps = model.config.rms_norm_eps
        assert len(captured) == 12 and len(attention) == 2
        for module, x, output in captured:
            if isinstance(module, reference.Qwen3MLP):
                assert same_bits(ops.swiglu(x, module.gate_proj.weight, module.up_proj.weight, module.down_proj.weight), output)
            elif isinstance(module, reference.Qwen3RMSNorm):
                assert same_bits(ops.rms_norm(x, module.weight, eps), output)
        # Rotary and attention on the captured normed q and k and projected v of each layer, against
        # the reference's rotary tables, apply_rotary_pos_emb and eager attention, then o_proj.
        normed = [output for module, _, output in captured if isinstance(module, reference.Qwen3RMSNorm)
                  and module.weight.shape[0] == model.config.head_dim]
        values = [output for module, _, output in captured if isinstance(module, torch.nn.Linear)]
        for index, (module, (cos, sin), expected) in enumerate(attention):
            q, k = normed[2 * index].transpose(1, 2), normed[2 * index + 1].transpose(1, 2)
            v = values[index].view(*values[index].shape[:2], -1, model.config.head_dim).transpose(1, 2)
            ours_cos, ours_sin = ops.rotary_tables(torch.arange(256), model.config.head_dim, loaded_theta(model), dtype)
            assert same_bits(ours_cos, cos[0]) and same_bits(ours_sin, sin[0])
            expected_q, expected_k = reference.apply_rotary_pos_emb(q, k, cos, sin)
            q, k = ops.apply_rotary(q, ours_cos, ours_sin), ops.apply_rotary(k, ours_cos, ours_sin)
            assert same_bits(q, expected_q) and same_bits(k, expected_k)
            attended = ops.attention(q, k, v, causal=True, backend="math")
            actual = module.o_proj(attended.transpose(1, 2).reshape(1, 256, -1))
            assert same_bits(actual, expected[0])


def loaded_theta(model):
    return float(model.config.rope_parameters["rope_theta"])


@pytest.mark.checkpoint
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_logit_positions_on_the_development_checkpoint(stored, record_property, dtype):
    stored("qwen3-0.6b")
    for name, value in child_result(run_check("logit_position_check", dtype))["records"]:
        record_property(name, value)


def logit_position_check(dtype, record):
    """Bitwise parity with the reference for each request, then, separately, the selected rows
    against the all-positions rows (check_selected_rows)."""
    directory = WEIGHTS / "qwen3-0.6b"
    loaded = load_checkpoint(directory, dtype=torch.bfloat16, device="cpu")
    model = Qwen3ForCausalLM.from_pretrained(directory, dtype=dtype, attn_implementation="eager").eval()
    for batch in (1, 2):
        token_ids = token_rows(directory, 1024, batch)
        every = forward_full(loaded, token_ids, dtype=dtype)
        for rows in ([1023], [0, 1023], list(range(7)), list(range(0, 1024, 64))):
            with torch.no_grad():
                expected = model(input_ids=token_ids, use_cache=False, logits_to_keep=torch.tensor(rows)).logits
            selected = forward_full(loaded, token_ids, dtype=dtype, logit_positions=rows)
            assert same_bits(selected, expected.float())
            check_selected_rows("qwen3-0.6b", loaded, token_ids, dtype, "math", rows, selected, every, record)


NEGATIVE_CONTROLS = {
    # The bf16 norm rounds once instead of twice: captured-activation parity must fail.
    "captured_activation_check": ("ops.py", "return weight * normalized.to(x.dtype)",
                                  "return (weight.to(normalized.dtype) * normalized).to(x.dtype)", torch.bfloat16),
    # Selected rows shifted by one: parity with the reference's same request must fail.
    "logit_position_check": ("forward.py", "hidden = hidden[:, rows.to(hidden.device)]",
                             "hidden = hidden[:, (rows.to(hidden.device) + 1) % length]", torch.float32),
}


@pytest.mark.checkpoint
@pytest.mark.parametrize("check", NEGATIVE_CONTROLS)
def test_the_child_checks_fail_on_a_wrong_result(stored, tmp_path, check):
    stored("qwen3-0.6b")
    file, old, new, dtype = NEGATIVE_CONTROLS[check]
    package = tmp_path / "src" / "ipe"
    shutil.copytree(Path(ipe.forward.__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
    text = (package / file).read_text()
    assert text.count(old) == 1
    (package / file).write_text(text.replace(old, new))
    child = run_check(check, dtype, source=tmp_path / "src")
    assert imported_package(child) == package
    assert child.returncode != 0 and "AssertionError" in child.stderr
