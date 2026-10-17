# Qwen3 inference engine

`ipe` is a compact inference engine for [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), written in Python on PyTorch, built to be checked logit by logit against Hugging Face Transformers and to have its GPU time accounted for kernel by kernel. vLLM and Transformers' static-cache generation are its performance baselines.

## Status

Everything below runs and is verified on the CPU. Nothing has run on an NVIDIA GPU yet, and there are no performance measurements.

- [src/ipe/config.py](src/ipe/config.py): admits only the supported dense Qwen3 structure.
- [src/ipe/checkpoint.py](src/ipe/checkpoint.py), [checkpoints/](checkpoints/): pinned checkpoint revisions, hash verification, and a tensor inventory read from the file headers alone.
- [src/ipe/weights.py](src/ipe/weights.py): maps every checkpoint tensor to the engine's weights at bf16, fp32 or fp64, with Transformers' LM-head tying rule.
- [src/ipe/ops.py](src/ipe/ops.py), [src/ipe/forward.py](src/ipe/forward.py): the decoder operators and the uncached forward pass, whose fp64 mode is the correctness oracle.
- [src/ipe/engine.py](src/ipe/engine.py): prefill and decode over a preallocated KV cache, greedy and temperature sampling, and generation for static batches of equal-length prompts.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 to 3.14.

```bash
uv sync --locked                    # PyTorch 2.13.0, Transformers 5.17.0 (tests only)
uv sync --locked --extra bench      # Linux: adds vLLM 0.31.0, which the lock satisfies
```

The checkpoints are pinned in [checkpoints/](checkpoints/), each file with its size and SHA-256. Download a revision into `weights/` and verify it; `HF_HOME` keeps the download client's files in the ignored `hf-cache/`.

```bash
HF_HOME="$PWD/hf-cache" uv run --locked hf download Qwen/Qwen3-1.7B \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir weights/qwen3-1.7b
uv run --locked python -m ipe.checkpoint verify checkpoints/qwen3-1.7b.json weights/qwen3-1.7b
uv run --locked python -m ipe.checkpoint inspect weights/qwen3-1.7b
```

Qwen3-0.6B (revision `c1899de289a04d12100db370d81485cdf75e47ca`, into `weights/qwen3-0.6b`) is the fast test fixture. Loading checks structure but not hashes, so verify once after each download.

## Usage

```python
import torch
from tokenizers import Tokenizer
from ipe.engine import Engine
from ipe.weights import load_checkpoint

tokenizer = Tokenizer.from_file("weights/qwen3-1.7b/tokenizer.json")
loaded = load_checkpoint("weights/qwen3-1.7b", dtype=torch.bfloat16, device="cuda")
engine = Engine(loaded, batch=1, capacity=1024)      # K/V and rotary tables allocated here
prompt = torch.tensor([tokenizer.encode("The capital of France is").ids])
result = engine.generate(prompt, 32)                 # greedy; temperature=, seed=, stop_ids= optional
print(tokenizer.decode(result.tokens[0].tolist()))
```

`engine.prefill(token_ids)` and `engine.decode(tokens)` expose single steps, each returning fp32 logits for the last position. `ipe.forward.forward_full(loaded, token_ids, dtype=...)` is the uncached path, and `final_hidden` with `lm_logits` at `dtype=torch.float64` is the oracle. Sampling settings are always explicit; the checkpoint's `generation_config.json` is never read.

A decode step indexes preallocated buffers with host integers, never reads a device value, and calls PyTorch's scaled-dot-product attention without a mask, so the flash kernel stays eligible on CUDA. When timing decode, pass token IDs already on the device: `decode` bounds-checks host IDs and copies them synchronously. `prefill` checks its IDs wherever they are. The KV cache holds one static batch: `prefill` starts a new one, positions past the committed length are never read, and a step that raises leaves the committed prefix intact.

## Tests

```bash
make lint                                                   # ruff
make test                                                   # tests that need no PyTorch, GPU or checkpoint
uv run --locked pytest -m "not gpu and not checkpoint"      # adds the PyTorch tests on synthetic checkpoints
uv run --locked pytest -m checkpoint                        # needs both checkpoints downloaded
uv run --locked pytest -m gpu                               # needs an NVIDIA GPU and Qwen3-1.7B
```

`IPE_TEST_MPS=1` enables device tests on Apple MPS. Some checkpoint tests require approximately 14 GB of RAM per process.

IPE uses two verification approaches:

- **Bitwise parity:** When operations follow the same execution order, outputs must match Transformers 5.17.0 exactly. On an Apple M4 Pro CPU, uncached forward passes match layer by layer for Qwen3-1.7B and Qwen3-0.6B at fp32 and bf16, including sequences up to 4,224 tokens at fp32. Teacher-forced cached decoding also matches Transformers' cached implementation for Qwen3-0.6B (256 tokens) and synthetic grouped-attention checkpoints.
- **Numerical accuracy:** When implementations legitimately differ, such as cached versus uncached decoding or SDPA versus math attention, `tests/needs_torch/accuracy.py` evaluates their errors against an fp64 oracle and reference variants. At fp32, the tests accept valid numerical reorderings while detecting eight seeded defects.

Because certain bf16 RMSNorm defects fall within the end-to-end tolerance, `tests/needs_torch/test_rms_norm.py` independently verifies operator accuracy against exact values. This contract also applies to future fused kernels.

## Benchmarks

No GPU measurements exist yet. [bench/run.py](bench/run.py) times fixed-work generation for the engine, Transformers (dynamic and static cache) and vLLM, one implementation per process, and writes raw per-iteration timings to `results/`, which is not tracked:

| Workload | Batch | Prompt | New tokens |
| --- | --- | --- | --- |
| W1 | 1 | 512 | 256 |
| W2-4, W2-16 | 4, 16 | 512 | 256 |
| W3 | 1 | 4,096 | 128 |

```bash
uv run --locked python bench/run.py run --impl engine --model weights/qwen3-1.7b --workloads W1 --out results/run1
uv run --locked python bench/run.py summarize results/run1
```

Each process makes one cold call, then 10 warm-up and 30 measured pairs of latency(N) and latency(1), timed from a synchronized start to the device finishing; TPOT is (latency(N) - latency(1)) / (N - 1) for every implementation. Per-step host latency comes only from separate runs that synchronize every step, labeled as such. Existing results are never overwritten. `vllm-calibration` repeats the vLLM run with the sampling and prompt distribution of `vllm bench latency`, which cross-checks the harness.

[bench/session.sh](bench/session.sh) holds the GPU session stages: requirements screen, probes, long correctness check, interleaved benchmark launches. A failed correctness check stops a stage before any profiling or timing. [bench/probes.py](bench/probes.py) holds the environment record, microbenchmarks and decode profile. [bench/brev_session.sh](bench/brev_session.sh) runs one stage on a new Brev instance within what is left of a session's budget and time, recorded in a ledger beside the results; it retrieves the results as one archive checked against its SHA-256 on the instance, then deletes the instance, confirmed by `brev ls`.

## Scope

IPE targets a pinned Qwen3-1.7B checkpoint using bf16 on a single NVIDIA GPU (compute capability ≥ 8.0, ≥ 24 GB).

The engine implements transformer execution, KV caching, static-batch prefill and incremental decoding, and greedy and temperature sampling.

PyTorch provides matrix multiplication and attention kernels, `safetensors` handles checkpoint loading, and Transformers serves as a test reference.

Out of scope: model serving, paged KV allocation, continuous batching, multi-GPU execution, quantization, and speculative decoding.

## License

MIT. See [LICENSE](LICENSE). The test fixtures' provenance and licensing are in [tests/fixtures/README.md](tests/fixtures/README.md).
