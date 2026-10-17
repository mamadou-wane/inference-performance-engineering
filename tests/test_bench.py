"""bench/run.py summary, agreement, overwrite refusal and timing boundary; bench/probes.py ncu and
idle-share parsing. Without PyTorch."""

import pytest

from bench import probes
from bench import run as bench


def record(impl, launch, latency_n, latency_1, workload="W1", new_tokens=5, batch=2):
    return {"implementation": impl, "workload": workload, "launch": launch, "new_tokens": new_tokens, "batch": batch,
            "latency_n_s": latency_n, "latency_1_s": latency_1}


def test_tpot_is_derived_per_iteration_pair():
    table = bench.summary([record("engine", 1, [1.0, 1.4, 1.2], [0.2, 0.2, 0.4])])["engine W1"]
    assert table["tpot_ms"][1] == pytest.approx(200.0)
    assert table["latency_n_ms"][1] == pytest.approx(1200.0)
    assert table["tokens_per_s"] == pytest.approx(2 * 5 / 1.2)


def test_launches_pool_and_report_their_own_medians():
    table = bench.summary([record("vllm", 2, [2.0, 2.0], [1.0, 1.0]), record("vllm", 1, [1.0, 3.0], [0.5, 0.5])])
    assert table["vllm W1"]["launches"] == 2
    assert table["vllm W1"]["launch_medians_ms"] == [2000.0, 2000.0]


def test_named_and_smoke_workloads():
    assert bench.workload("W2-16") == (16, 512, 256)
    assert bench.workload("1,16,4") == (1, 16, 4)


@pytest.mark.parametrize(("bench_latencies", "agree"), [([[1.00, 1.02], [1.01, 1.03], [0.99, 1.01]], True),
                                                         ([[1.50, 1.52], [1.51, 1.53], [1.49, 1.51]], False)])
def test_agreement_with_vllm_bench_latency(bench_latencies, agree):
    harness = [[1.00, 1.04], [1.02, 1.06], [0.98, 1.02]]
    assert bench.agreement(harness, bench_latencies)["agree"] is agree


def test_equal_medians_agree_when_both_spreads_are_zero():
    flat = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    assert bench.agreement(flat, flat)["agree"] is True
    assert bench.agreement(flat, [[1.1, 1.1], [1.1, 1.1], [1.1, 1.1]])["agree"] is False


def test_agreement_needs_two_launches_per_tool():
    with pytest.raises(ValueError, match="2 launches"):
        bench.agreement([[1.0, 1.1]], [[1.0, 1.1], [1.0, 1.2]])


def test_failed_runs_are_left_out_of_the_summary():
    failed = record("engine", 2, [9.0, 9.0], [1.0, 1.0]) | {"status": "failed: RuntimeError()"}
    table = bench.summary([record("engine", 1, [1.0, 1.0], [0.5, 0.5]), failed])
    assert table["engine W1"]["launches"] == 1


def test_an_existing_result_is_never_overwritten(tmp_path):
    bench.result_path(tmp_path, "engine", "W1", 1).write_text("{}")
    arguments = bench.argparse.Namespace(impl="engine", model="unused", workloads=["W1", "W3"], launch=1,
                                         out=str(tmp_path), device="cpu", warmup=0, iterations=1)
    with pytest.raises(SystemExit, match="never overwritten"):
        bench.run(arguments)  # refuses before loading anything
    assert bench.result_path(tmp_path, "engine", "W1", 1).read_text() == "{}"


class Tokens:
    def __init__(self, shape):
        self.shape = shape


def timed_runner(monkeypatch, shape):
    calls = []
    runner = object.__new__(bench.Runner)
    runner.impl = "engine"
    runner.sync = lambda: calls.append("sync")
    runner.generate = lambda prompt, n: calls.append("generate") or Tokens(shape)
    clock = iter(range(100))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: calls.append("clock") or next(clock))
    return runner, calls


def test_the_timed_boundary_ends_after_the_device_finishes(monkeypatch):
    runner, calls = timed_runner(monkeypatch, (2, 4))
    runner.latency(Tokens((2, 8)), 4)
    assert calls == ["sync", "clock", "generate", "sync", "clock"]


def test_a_short_generation_is_an_error(monkeypatch):
    runner, _ = timed_runner(monkeypatch, (2, 1))
    with pytest.raises(RuntimeError, match="expected 4 new tokens"):
        runner.latency(Tokens((2, 8)), 4)


NCU_OK = '''==PROF== Connected to process 4242
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","4242","python3","127.0.0.1","add_one(float *, int)","1","7","Command line profiler metrics","sm__cycles_elapsed.avg","cycle","4,812.50"
'''
NCU_DENIED = '''==PROF== Connected to process 4242
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters.
==ERROR== Failed to profile "add_one" in process 4242: sm__cycles_elapsed.avg unavailable
'''


@pytest.mark.parametrize(("text", "ok"), [
    (NCU_OK, True),
    (NCU_DENIED, False),  # names the metric, holds no value
    (NCU_OK.replace('"4,812.50"', '"0"'), False),
    (NCU_OK.replace('"4,812.50"', '"n/a"'), False),
    (NCU_OK.replace("add_one(float *, int)", "other_kernel"), False),
    ("", False),
])
def test_the_ncu_check_needs_a_counter_value(text, ok):
    assert probes.ncu_counter_ok(text) is ok


@pytest.mark.parametrize(("intervals", "idle"), [
    ([], 1.0),
    ([(0, 10)], 0.0),
    ([(0, 2), (4, 6)], 0.6),
    ([(0, 4), (2, 6)], 0.4),  # overlap counts once
    ([(-5, 2), (8, 20)], 0.6),  # clipped to the window
    ([(1, 2), (1, 2), (1.5, 1.8)], 0.9),
])
def test_idle_share_is_the_uncovered_part_of_the_window(intervals, idle):
    assert probes.idle_share(intervals, (0, 10)) == pytest.approx(idle)


def test_calibration_uses_vllm_bench_latency_settings_and_baselines_stay_greedy():
    # vllm/benchmarks/latency.py (0.31.0): SamplingParams(n, temperature=1.0, top_p=1.0, ignore_eos=True,
    # max_tokens, detokenize=not --disable-detokenize) and prompts np.random.randint(10000, ...).
    high, sampling = bench.vllm_settings("vllm-calibration")
    assert high == 10000
    assert sampling == {"n": 1, "temperature": 1.0, "top_p": 1.0, "ignore_eos": True, "detokenize": False}
    high, sampling = bench.vllm_settings("vllm")
    assert high == bench.FIRST_SPECIAL_ID and sampling["temperature"] == 0


def test_the_summary_survives_an_agreement_that_cannot_be_computed(tmp_path, capsys):
    path = bench.result_path(tmp_path, "vllm-calibration", "W1", 1)
    path.write_text(bench.json.dumps(record("vllm-calibration", 1, [1.0, 1.1], [0.1, 0.1]) | {"status": "completed"}))
    (tmp_path / "bench-latency-1.json").write_text('{"latencies": [1.0, 1.1]}')
    arguments = bench.argparse.Namespace(results=str(tmp_path), vllm_bench=[str(tmp_path / "bench-latency-1.json")],
                                         bench_workload="W1")
    with pytest.raises(SystemExit):
        bench.summarize(arguments)
    result = bench.json.loads(capsys.readouterr().out)
    assert "vllm-calibration W1" in result["table"]
    assert result["vllm_bench_latency_agreement"].startswith("not computed")
