"""Child processes for the checkpoint parity tests in test_forward_checkpoint.py.

One process runs the reference and saves its logits, layer outputs and final-normed state; a
second runs the engine and compares. Separate processes keep the 1.7B fp32 reference (13 GB at
4,224 tokens) and the engine from being resident together on a 24 GiB machine.

    forward_child.py reference DIRECTORY DTYPE ATTENTION LENGTH BATCH OUT [patched]
    forward_child.py engine DIRECTORY DTYPE BACKEND LENGTH BATCH REFERENCE
    forward_child.py cached-reference DIRECTORY DTYPE ATTENTION LENGTH PROMPT OUT
    forward_child.py cached-engine DIRECTORY DTYPE BACKEND LENGTH PROMPT REFERENCE [SAVE]
    forward_child.py hidden DIRECTORY DTYPE BACKEND LENGTH OUT

The cached commands run one sequence teacher-forced from PROMPT and keep the fp32 logits at
positions PROMPT-1 to LENGTH-1. Engine commands print a JSON list of problems on their last line;
an empty list is a pass.
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


@torch.no_grad()
def run_cached_reference(directory, dtype, attention, length, prompt, out):
    from transformers import Qwen3ForCausalLM

    model = Qwen3ForCausalLM.from_pretrained(directory, dtype=dtype, attn_implementation=attention).eval()
    token_ids = token_rows(directory, length, 1)
    logits = torch.empty(length - prompt + 1, model.config.vocab_size)
    output = model(input_ids=token_ids[:, :prompt], use_cache=True, logits_to_keep=1)
    logits[0], cache = output.logits[0, -1], output.past_key_values
    for row, position in enumerate(range(prompt, length), start=1):
        output = model(input_ids=token_ids[:, position:position + 1], past_key_values=cache, use_cache=True)
        logits[row], cache = output.logits[0, -1], output.past_key_values
    torch.save(logits, out)


@torch.no_grad()
def run_cached_engine(directory, dtype, backend, length, prompt, reference_file, save):
    from ipe.engine import Engine
    from ipe.weights import load_checkpoint

    # fp32 weights cast once give the same bits as bf16 views cast at each use, twice as fast.
    engine = Engine(load_checkpoint(directory, dtype=dtype, device="cpu"), batch=1, capacity=length, backend=backend)
    token_ids = token_rows(directory, length, 1)
    logits = torch.empty(length - prompt + 1, engine.loaded.config.vocab_size)
    logits[0] = engine.prefill(token_ids[:, :prompt])[0]
    for row, position in enumerate(range(prompt, length), start=1):
        logits[row] = engine.decode(token_ids[:, position])[0]
    del engine  # the comparison below needs the memory
    expected = torch.load(reference_file, mmap=True, weights_only=True)
    problems = differences("cached logits", logits, expected)
    if problems:
        unequal = (logits != expected).any(-1).nonzero()
        problems.append(f"first differing position {prompt - 1 + int(unequal[0])}")
    if save:
        torch.save(logits, save)
    print(json.dumps(problems))


@torch.no_grad()
def run_hidden(directory, dtype, backend, length, out):
    from ipe.forward import final_hidden
    from ipe.weights import load_checkpoint

    loaded = load_checkpoint(directory, dtype=torch.bfloat16, device="cpu")
    torch.save(final_hidden(loaded, token_rows(directory, length, 1), dtype=dtype, backend=backend)[0], out)


if __name__ == "__main__":
    role, directory, dtype_name, implementation, length = sys.argv[1:6]
    common = (Path(directory), getattr(torch, dtype_name), implementation, int(length))
    if role == "reference":
        run_reference(*common, int(sys.argv[6]), sys.argv[7], patched=sys.argv[8:] == ["patched"])
    elif role == "engine":
        run_engine(*common, int(sys.argv[6]), sys.argv[7])
    elif role == "cached-reference":
        run_cached_reference(*common, int(sys.argv[6]), sys.argv[7])
    elif role == "cached-engine":
        run_cached_engine(*common, int(sys.argv[6]), sys.argv[7], sys.argv[8] if len(sys.argv) > 8 else None)
    else:
        run_hidden(*common, sys.argv[6])
