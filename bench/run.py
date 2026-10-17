"""Fixed-work generation benchmarks: one implementation per process, raw timings to JSON.

Each process makes one cold call, then warm-up and measured pairs of latency(N) and latency(1),
timed from a synchronized start until the device has finished; TPOT is derived from the pairs.
The engine also records per-step device times from CUDA events and, in a separate labeled pass,
per-step host latency with a sync after every step. vllm-calibration matches `vllm bench
latency`'s sampling and prompt distribution. Results are never overwritten or committed.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

WORKLOADS = {"W1": (1, 512, 256), "W2-4": (4, 512, 256), "W2-16": (16, 512, 256), "W3": (1, 4096, 128)}  # B, P, N
FIRST_SPECIAL_ID = 151643  # Qwen3's special tokens start here
BENCH_LATENCY_IDS = 10000  # vllm bench latency draws prompt IDs below this, without a seed
IMPLEMENTATIONS = ("engine", "transformers-dynamic", "transformers-static", "vllm", "vllm-calibration")


def workload(name: str) -> tuple[int, int, int]:
    """A named workload, or "B,P,N" for smoke runs."""
    return WORKLOADS[name] if name in WORKLOADS else tuple(int(v) for v in name.split(","))


def prompts(batch: int, length: int, seed: int = 0, high: int = FIRST_SPECIAL_ID):
    import torch

    return torch.randint(0, high, (batch, length), generator=torch.Generator().manual_seed(seed))


def vllm_settings(impl: str) -> tuple[int, dict]:
    """Prompt-ID bound and SamplingParams arguments; vllm-calibration copies vllm/benchmarks/latency.py."""
    if impl == "vllm-calibration":
        return BENCH_LATENCY_IDS, {"n": 1, "temperature": 1.0, "top_p": 1.0, "ignore_eos": True, "detokenize": False}
    return FIRST_SPECIAL_ID, {"temperature": 0, "ignore_eos": True, "detokenize": False}


def result_path(out: Path, impl: str, name: str, launch: int) -> Path:
    return out / f"{impl}-{name.replace(',', '_')}-launch{launch}.json"


def environment() -> dict:
    """The environment record: software, GPU, driver, clocks and precision settings at this moment."""
    import platform

    import torch

    def query(fields):
        try:
            return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()

    versions = {}
    for package in ("torch", "transformers", "vllm"):  # metadata only: importing vLLM patches torch
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor() or platform.machine()
    try:
        container = Path("/.dockerenv").exists() or any(
            kind in Path("/proc/1/cgroup").read_text() for kind in ("docker", "containerd", "kubepods"))
    except OSError:
        container = False
    cuda = torch.cuda.is_available()
    return {
        "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
        "versions": versions, "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
        "cpu_model": cpu, "logical_cpus": os.cpu_count(), "container": container,
        "gpu": torch.cuda.get_device_name() if cuda else None,
        "compute_capability": list(torch.cuda.get_device_capability()) if cuda else None,
        "nvidia_smi": query("name,memory.total,driver_version,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,"
                            "power.draw,persistence_mode"),
        "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


class Runner:
    """generate(prompt, n) returns new tokens [B, n] on the host, after the device has finished."""

    def __init__(self, impl: str, model: str, device: str, max_length: int, batch: int):
        import torch

        self.torch, self.device, self.impl = torch, torch.device(device), impl
        if impl == "engine":
            from ipe.engine import Engine
            from ipe.weights import load_checkpoint

            self.engine = Engine(load_checkpoint(model, dtype=torch.bfloat16, device=self.device), batch=batch,
                                 capacity=max_length, backend="sdpa")
        elif impl.startswith("transformers"):
            from transformers import GenerationConfig, Qwen3ForCausalLM

            self.model = Qwen3ForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa")
            self.model = self.model.to(self.device).eval()
            # generate() fills unset fields from generation_config.json, including its EOS IDs.
            self.model.generation_config.eos_token_id = None
            self.config = GenerationConfig(do_sample=False, eos_token_id=None, pad_token_id=0,
                                           cache_implementation="static" if impl.endswith("static") else None)
        elif impl.startswith("vllm"):
            from vllm import LLM

            self.llm = LLM(model=model, dtype="bfloat16", enable_prefix_caching=False, seed=0, max_model_len=max_length,
                           max_num_seqs=batch)
        else:
            raise ValueError(f"unknown implementation {impl!r}")

    def sync(self):
        # vLLM runs the model in its engine process and returns after it finishes; a sync here would
        # only create an idle CUDA context in this process.
        if self.device.type == "cuda" and not self.impl.startswith("vllm"):
            self.torch.cuda.synchronize()

    def generate(self, prompt, n: int):
        if self.impl == "engine":
            return self.engine.generate(prompt, n).tokens
        if self.impl.startswith("transformers"):
            with self.torch.no_grad():
                out = self.model.generate(prompt, attention_mask=self.torch.ones_like(prompt), generation_config=self.config,
                                          max_new_tokens=n)
            return out[:, prompt.shape[1]:].cpu()
        from vllm import SamplingParams, TokensPrompt

        params = SamplingParams(max_tokens=n, **vllm_settings(self.impl)[1])
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=row.tolist()) for row in prompt], params, use_tqdm=False)
        return self.torch.tensor([output.outputs[0].token_ids for output in outputs])

    def latency(self, prompt, n: int) -> float:
        self.sync()
        start = time.perf_counter()
        tokens = self.generate(prompt, n)
        self.sync()
        elapsed = time.perf_counter() - start
        if tuple(tokens.shape) != (prompt.shape[0], n):
            raise RuntimeError(f"{self.impl}: expected {n} new tokens per sequence, got shape {tuple(tokens.shape)}")
        return elapsed

    def engine_steps(self, prompt, n: int, synced: bool) -> dict:
        """Greedy loop as Engine.generate runs it: per-step device times from CUDA events, or per-step
        host latency with a sync after every step (labeled; it removes the host's run-ahead)."""
        from ipe.engine import sample

        torch, engine = self.torch, self.engine
        if synced:
            times, start = [], time.perf_counter()
            tokens = sample(engine.prefill(prompt), 0.0)[0]
            self.sync()
            times.append(time.perf_counter() - start)
            for _ in range(n - 1):
                start = time.perf_counter()
                tokens = sample(engine.decode(tokens), 0.0)[0]
                self.sync()
                times.append(time.perf_counter() - start)
            return {"host_step_synced_ms": [1e3 * t for t in times]}
        events = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
        events[0].record()
        tokens = sample(engine.prefill(prompt), 0.0)[0]
        events[1].record()
        for step in range(1, n):
            tokens = sample(engine.decode(tokens), 0.0)[0]
            events[step + 1].record()
        events[-1].synchronize()
        times = [events[i].elapsed_time(events[i + 1]) for i in range(n)]
        return {"prefill_device_ms": times[0], "decode_device_ms": times[1:]}


def run(args) -> None:
    shapes = {name: workload(name) for name in args.workloads}
    out = Path(args.out)
    existing = [str(path) for name in shapes if (path := result_path(out, args.impl, name, args.launch)).exists()]
    if existing:
        raise SystemExit(f"results are never overwritten; already present: {existing}")
    import torch

    out.mkdir(parents=True, exist_ok=True)
    env, runner, failed = environment(), None, False
    for name, (batch, length, n) in shapes.items():
        record = {"implementation": args.impl, "workload": name, "batch": batch, "prompt_length": length,
                  "new_tokens": n, "launch": args.launch, "environment": env, "status": "running"}
        try:
            # vLLM serves every workload from one engine; the others are built per workload, so the
            # engine's cache and Transformers' static cache are sized for it alone.
            vllm = args.impl.startswith("vllm")
            if runner is None or not vllm:
                runner = None
                start = time.perf_counter()
                if vllm:
                    runner = Runner(args.impl, args.model, args.device, max(p + k for _, p, k in shapes.values()),
                                    max(b for b, _, _ in shapes.values()))
                else:
                    runner = Runner(args.impl, args.model, args.device, length + n, batch)
                record["init_s"] = time.perf_counter() - start
            prompt = prompts(batch, length, high=vllm_settings(args.impl)[0])
            record["prompt_sha256"] = hashlib.sha256(prompt.numpy().tobytes()).hexdigest()
            prompt = prompt if vllm else prompt.to(runner.device)
            record["cold_s"] = runner.latency(prompt, n)
            for _ in range(args.warmup):
                runner.latency(prompt, n)
                runner.latency(prompt, 1)
            local = runner.device.type == "cuda" and not vllm  # vLLM's model lives in another process
            if local:
                torch.cuda.reset_peak_memory_stats()
                reserved = torch.cuda.memory_reserved()
            record["latency_n_s"], record["latency_1_s"] = [], []
            for _ in range(args.iterations):
                record["latency_n_s"].append(runner.latency(prompt, n))
                record["latency_1_s"].append(runner.latency(prompt, 1))
            if local:
                record["memory"] = {"reserved_before": reserved, "reserved_after": torch.cuda.memory_reserved(),
                                    "peak_allocated": torch.cuda.max_memory_allocated(),
                                    "peak_reserved": torch.cuda.max_memory_reserved()}
            if args.impl == "engine" and runner.device.type == "cuda":
                record |= runner.engine_steps(prompt, n, synced=False)
                record |= runner.engine_steps(prompt, n, synced=True)
            record["status"] = "completed"
        except Exception as error:  # noqa: BLE001 - any failure keeps what was measured; the run is excluded
            record["status"] = f"failed: {error!r}"
            failed = True
        path = result_path(out, args.impl, name, args.launch)
        with open(path, "x") as file:  # "x" fails if the file appeared meanwhile
            file.write(json.dumps(record, indent=1))
        print(f"{path}: {record['status']}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


def quartiles(values: list[float]) -> tuple[float, float, float]:
    q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, median, q3


def summary(records: list[dict]) -> dict:
    """Median and IQR per (implementation, workload), pooled over launches and per launch."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if record.get("status", "completed") != "completed":
            continue
        groups.setdefault((record["implementation"], record["workload"]), []).append(record)
    table = {}
    for (impl, name), runs in sorted(groups.items()):
        n, batch = runs[0]["new_tokens"], runs[0]["batch"]
        pooled_n = [t for r in runs for t in r["latency_n_s"]]
        pooled_1 = [t for r in runs for t in r["latency_1_s"]]
        tpot = [(tn - t1) / (n - 1) for r in runs for tn, t1 in zip(r["latency_n_s"], r["latency_1_s"], strict=True)] if n > 1 else []
        launch_medians = [statistics.median(r["latency_n_s"]) for r in sorted(runs, key=lambda r: r["launch"])]
        q1, median, q3 = quartiles(pooled_n)
        table[f"{impl} {name}"] = {
            "launches": len(runs), "latency_n_ms": [1e3 * q1, 1e3 * median, 1e3 * q3],
            "latency_1_ms": [1e3 * v for v in quartiles(pooled_1)],
            "tpot_ms": [1e3 * v for v in quartiles(tpot)] if len(tpot) > 1 else None,
            "tokens_per_s": batch * n / median, "launch_medians_ms": [1e3 * m for m in launch_medians],
            "across_launch_iqr_ms": 1e3 * (quartiles(launch_medians)[2] - quartiles(launch_medians)[0]) if len(runs) > 1 else None,
            "reserved_changed": any(r["memory"]["reserved_before"] != r["memory"]["reserved_after"] for r in runs
                                    if "memory" in r) if any("memory" in r for r in runs) else None,
        }
    return table


def agreement(harness: list[list[float]], bench: list[list[float]]) -> dict:
    """Agreement: the medians differ by less than the larger across-launch IQR; equal medians
    agree when both spreads are zero. Each argument holds one list of latencies (seconds) per
    launch, at least two launches each.
    """
    if len(harness) < 2 or len(bench) < 2:
        raise ValueError(f"needs at least 2 launches per tool, got {len(harness)} and {len(bench)}")

    def across(launches):
        medians = [statistics.median(latencies) for latencies in launches]
        q1, _, q3 = quartiles(medians)
        return statistics.median([t for latencies in launches for t in latencies]), q3 - q1

    (harness_median, harness_iqr), (bench_median, bench_iqr) = across(harness), across(bench)
    return {"harness_median_s": harness_median, "bench_median_s": bench_median,
            "difference_s": abs(harness_median - bench_median), "tolerance_s": max(harness_iqr, bench_iqr),
            "agree": abs(harness_median - bench_median) < max(harness_iqr, bench_iqr) or harness_median == bench_median}


def summarize(args) -> None:
    records = [json.loads(path.read_text()) for path in sorted(Path(args.results).glob("*-launch*.json"))]
    result = {"table": summary(records)}
    if args.vllm_bench:
        harness = [r["latency_n_s"] for r in records if r["implementation"] == "vllm-calibration"
                   and r["workload"] == args.bench_workload and r.get("status") == "completed"]
        bench = [json.loads(Path(path).read_text())["latencies"] for path in args.vllm_bench]
        try:
            result["vllm_bench_latency_agreement"] = agreement(harness, bench)
        except ValueError as error:  # the table stands without it
            result["vllm_bench_latency_agreement"] = f"not computed: {error}"
    print(json.dumps(result, indent=1))
    if isinstance(result.get("vllm_bench_latency_agreement"), str):
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runs = commands.add_parser("run")
    runs.add_argument("--impl", choices=IMPLEMENTATIONS, required=True)
    runs.add_argument("--model", required=True)
    runs.add_argument("--workloads", nargs="+", default=["W1"], help='names, or "B,P,N" for smoke runs')
    runs.add_argument("--launch", type=int, default=1)
    runs.add_argument("--out", required=True)
    runs.add_argument("--device", default="cuda")
    runs.add_argument("--warmup", type=int, default=10)
    runs.add_argument("--iterations", type=int, default=30)
    summaries = commands.add_parser("summarize")
    summaries.add_argument("results")
    summaries.add_argument("--vllm-bench", nargs="*", default=[], help="vllm bench latency --output-json files, one per launch")
    summaries.add_argument("--bench-workload", default="W1")
    args = parser.parse_args(argv)
    run(args) if args.command == "run" else summarize(args)


if __name__ == "__main__":
    main()
