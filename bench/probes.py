"""GPU session probes, run from the repository root on the instance.

    env                 environment record (JSON)
    extension           build and run a trivial CUDA extension (target for compute-sanitizer and ncu)
    ncu-check FILE      exit 0 only if ncu's CSV output holds a positive counter for that kernel
    vllm                vLLM's compiled CUDA extensions load and run
    microbench          copy bandwidth; device time and host enqueue cost of a minimal operator
    profile MODEL OUT   W1 decode trace: kernels and host time per step, idle share, attention kernels

Profiled numbers explain; they never supply timings for a claim.
"""

import csv
import io
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import environment, prompts


def ncu_counter_ok(text: str) -> bool:
    """True when ncu's CSV output has a positive sm__cycles_elapsed.avg row for add_one. An error such
    as ERR_NVGPUCTRPERM may name the metric but produces no such row."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('"ID"')]
    if not starts:
        return False
    for row in csv.DictReader(io.StringIO("\n".join(lines[starts[0]:]))):
        if row.get("Metric Name") == "sm__cycles_elapsed.avg" and "add_one" in row.get("Kernel Name", ""):
            try:
                if float(row.get("Metric Value", "").replace(",", "")) > 0:
                    return True
            except ValueError:
                continue
    return False


def idle_share(intervals: list[tuple[float, float]], window: tuple[float, float]) -> float:
    """Fraction of window [start, end) covered by no interval; intervals may overlap or extend past it."""
    start, end = window
    busy, covered = 0.0, start
    for first, last in sorted(intervals):
        first, last = max(first, covered), min(last, end)
        if last > first:
            busy += last - first
            covered = last
    return 1 - busy / (end - start)


def extension() -> None:
    import torch
    from torch.utils.cpp_extension import load_inline

    source = """
    __global__ void add_one(float* x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += 1.0f; }
    torch::Tensor add_one_(torch::Tensor x) {
        add_one<<<(x.numel() + 255) / 256, 256>>>(x.data_ptr<float>(), x.numel());
        return x;
    }
    """
    module = load_inline("ipe_probe", cpp_sources="torch::Tensor add_one_(torch::Tensor x);", cuda_sources=source,
                         functions=["add_one_"], verbose=False)
    x = module.add_one_(torch.zeros(1000, device="cuda"))
    torch.cuda.synchronize()
    assert torch.equal(x, torch.ones_like(x))
    print("extension built and ran")


def vllm_check() -> None:
    import torch
    import vllm._custom_ops  # registers torch.ops._C
    import vllm.platforms

    platform = vllm.platforms.current_platform
    assert platform.is_cuda(), type(platform).__name__
    import vllm._C_stable_libtorch
    import vllm._moe_C_stable_libtorch
    from vllm.vllm_flash_attn import FA2_AVAILABLE, FA3_AVAILABLE

    x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, torch.ones(4096, device="cuda", dtype=torch.bfloat16), 1e-6)
    reference = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    print(json.dumps({"platform": type(platform).__name__, "fa2": FA2_AVAILABLE, "fa3": FA3_AVAILABLE,
                      "rms_norm_max_abs_error": (out.float() - reference).abs().max().item()}))


def microbench() -> None:
    import torch

    def device_ms(fn, repeats, queued=False):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(repeats):
            if queued:  # fn's kernels queue behind the sleep and then run back to back
                torch.cuda._sleep(100_000_000)
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop))
        return statistics.median(times)

    def copy_ms():
        source = torch.empty(2**29, dtype=torch.bfloat16, device="cuda")  # 1 GiB
        target = torch.empty_like(source)
        for _ in range(3):
            target.copy_(source)
        return device_ms(lambda: target.copy_(source), 20)

    one = torch.zeros(1, device="cuda")
    for _ in range(100):
        one.add_(1)
    torch.cuda.synchronize()
    per_op_device_ms = device_ms(lambda: [one.add_(1) for _ in range(1000)], 10, queued=True) / 1000
    enqueue = []
    for _ in range(10):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(1000):
            one.add_(1)
        enqueue.append((time.perf_counter() - begin) / 1000)
        torch.cuda.synchronize()
    print(json.dumps({"copy_bandwidth_gb_s": 2 * 2**30 / (copy_ms() * 1e-3) / 1e9,
                      "minimal_op_device_us": 1e3 * per_op_device_ms,
                      "minimal_op_host_enqueue_us": 1e6 * statistics.median(enqueue)}))


def profile(model: str, out: str, steps: int = 16) -> None:
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, record_function

    from ipe.engine import Engine, sample
    from ipe.weights import load_checkpoint

    engine = Engine(load_checkpoint(model, dtype=torch.bfloat16, device="cuda"), batch=1, capacity=512 + 256)
    tokens = sample(engine.prefill(prompts(1, 512).cuda()), 0.0)[0]
    for _ in range(8):
        tokens = sample(engine.decode(tokens), 0.0)[0]
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            with record_function("decode_step"):
                tokens = sample(engine.decode(tokens), 0.0)[0]
        torch.cuda.synchronize()
    events = prof.events()
    # Device activity: kernels, copies and memsets; record_function's device-side span is not activity.
    device = [e for e in events if e.device_type == DeviceType.CUDA and not e.is_user_annotation]
    kernels = [e for e in device if not e.name.startswith(("Memcpy", "Memset"))]
    steps_cpu = [e for e in events if e.name == "decode_step" and e.device_type == DeviceType.CPU]
    intervals = [(e.time_range.start, e.time_range.end) for e in device]
    # From the host starting the first step to the device finishing the last, on the trace's clock.
    window = (min(e.time_range.start for e in steps_cpu), max(end for _, end in intervals))
    attention = sorted({e.name for e in kernels if any(k in e.name.lower() for k in ("flash", "fmha", "attention", "sdpa", "cudnn"))})
    Path(out).mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(Path(out) / "w1-decode-trace.json"))
    print(json.dumps({"steps": steps, "kernels_per_step": len(kernels) / steps,
                      "device_activities_per_step": len(device) / steps,
                      # The host's time per step; it equals enqueue cost only while the launch queue has room.
                      "traced_host_step_ms": statistics.median(e.time_range.elapsed_us() for e in steps_cpu) / 1e3,
                      "traced_window_ms_per_step": (window[1] - window[0]) / 1e3 / steps,
                      "traced_device_busy_ms_per_step": (1 - idle_share(intervals, window)) * (window[1] - window[0]) / 1e3 / steps,
                      "traced_gpu_idle_share": idle_share(intervals, window), "attention_kernels": attention,
                      "note": "from the profiler trace; profiling perturbs timing"}))


if __name__ == "__main__":
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "env":
        print(json.dumps(environment(), indent=1))
    elif command == "extension":
        extension()
    elif command == "ncu-check":
        raise SystemExit(0 if ncu_counter_ok(Path(rest[0]).read_text(errors="replace")) else 1)
    elif command == "vllm":
        vllm_check()
    elif command == "microbench":
        microbench()
    elif command == "profile":
        profile(*rest)
    else:
        raise SystemExit(__doc__)
