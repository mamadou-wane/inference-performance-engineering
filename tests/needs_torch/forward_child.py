"""Child processes for the checkpoint parity tests in test_forward_checkpoint.py.

One process runs the reference and saves its logits, layer outputs and final-normed state; a
second runs the engine and compares. Separate processes keep the 1.7B fp32 reference (13 GB at
4,224 tokens) and the engine from being resident together on a 24 GiB machine.

    forward_child.py reference DIRECTORY DTYPE ATTENTION LENGTH BATCH OUT [patched]
    forward_child.py engine DIRECTORY DTYPE BACKEND LENGTH BATCH REFERENCE

The engine prints a JSON list of problems on its last line; an empty list is a pass.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from tokenizers import Tokenizer

F64 = torch.float64
# Rows are deterministic windows of the checkpoint's own README followed by its LICENSE.
ROW_STRIDE = 97


def token_rows(directory: Path, length: int, batch: int) -> torch.Tensor:
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    stream = []
    for name in ("README.md", "LICENSE"):
        stream += tokenizer.encode((directory / name).read_text()).ids
    assert len(stream) >= length + ROW_STRIDE * (batch - 1), len(stream)
    return torch.tensor([stream[b * ROW_STRIDE:b * ROW_STRIDE + length] for b in range(batch)])


def patch_reference_to_fp64():
    """RMSNorm, rotary and eager softmax in fp64, as test_forward.patched_reference."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_forward
    from transformers.models.qwen3 import modeling_qwen3 as reference
    reference.Qwen3RMSNorm.forward = test_forward._rms_norm_fp64
    reference.Qwen3RotaryEmbedding.forward = test_forward._rotary_fp64
    reference.eager_attention_forward = test_forward._eager_fp64


@torch.no_grad()
def run_reference(directory, dtype, attention, length, batch, out, patched):
    if patched:
        patch_reference_to_fp64()
    from transformers import Qwen3ForCausalLM

    model = Qwen3ForCausalLM.from_pretrained(directory, dtype=dtype, attn_implementation=attention).eval()
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    outputs = []
    for module in (*model.model.layers, model.model.norm):
        module.register_forward_hook(lambda _m, _i, output: outputs.append(output))
    logits = model(input_ids=token_rows(directory, length, batch), use_cache=False).logits
    torch.save({"layers": outputs[:-1], "hidden": outputs[-1], "logits": logits}, out)


def differences(name, actual, expected):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return [f"{name}: {actual.dtype} {list(actual.shape)} against {expected.dtype} {list(expected.shape)}"]
    if torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
        return []
    unequal = actual != expected
    largest = (actual.double() - expected.double()).abs().max().item()
    return [f"{name}: {int(unequal.sum())} of {unequal.numel()} elements differ, max {largest:.3e}"]


@torch.no_grad()
def run_engine(directory, dtype, backend, length, batch, reference_file):
    from ipe.forward import final_hidden, forward_full, lm_logits
    from ipe.weights import load_checkpoint

    expected = torch.load(reference_file, mmap=True, weights_only=True)
    loaded = load_checkpoint(directory, dtype=torch.bfloat16, device="cpu")
    token_ids = token_rows(directory, length, batch)
    problems = []

    def compare(index, output):
        problems.extend(differences(f"layer {index}", output, expected["layers"][index]))

    hidden = final_hidden(loaded, token_ids, dtype=dtype, backend=backend, capture=compare)
    problems += differences("final norm", hidden, expected["hidden"])
    problems += differences("lm_logits", lm_logits(loaded, hidden), expected["logits"])
    del hidden
    if dtype != F64:
        problems += differences("forward_full", forward_full(loaded, token_ids, dtype=dtype, backend=backend),
                                expected["logits"].float())
    print(json.dumps(problems))


if __name__ == "__main__":
    role, directory, dtype_name, implementation, length, batch, path = sys.argv[1:8]
    arguments = (Path(directory), getattr(torch, dtype_name), implementation, int(length), int(batch), path)
    if role == "reference":
        run_reference(*arguments, patched=sys.argv[8:] == ["patched"])
    else:
        run_engine(*arguments)
