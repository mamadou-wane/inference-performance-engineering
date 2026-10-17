"""bench/session.sh and bench/brev_session.sh against stub tools: no GPU, no Brev account.

session.sh: a failed correctness check stops the stage before any profiling or timing, and the
ncu check needs both a clean exit and a counter value. brev_session.sh: the instance is deleted,
and the deletion confirmed by a successful brev ls, after completion, failure, timeout and
interruption; a budget too small creates nothing; an unconfirmed deletion exits 3.
"""

import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def tools(tmp_path):
    """A PATH directory of stubs; each call is appended to calls.log."""
    directory = tmp_path / "bin"
    directory.mkdir()
    log = tmp_path / "calls.log"
    for name in ("compute-sanitizer", "nsys"):
        stub(directory, name, f'echo "{name} $*" >> {log}\n')
    stub(directory, "nvidia-smi", f'echo "nvidia-smi $*" >> {log}\necho "NVIDIA RTX A6000, 49140 MiB, 580.95.05, 8.6"\n')
    stub(directory, "nvcc", 'echo "Cuda compilation tools, release 13.0, V13.0.88"\n')
    stub(directory, "uv", f'echo "uv $*" >> {log}\n[[ "$*" == *pytest* && -n "${{SUITE_FAILS:-}}" ]] && exit 1\nexit 0\n')
    return directory, log


def session(tools, tmp_path, stage, **env):
    directory, _ = tools
    out = tmp_path / "out"
    return subprocess.run(["bash", str(ROOT / "bench/session.sh"), stage, str(out)], cwd=ROOT, text=True,
                          capture_output=True, env={**os.environ, "PATH": f"{directory}:{os.environ['PATH']}", **env},
                          timeout=120, check=False), out


def test_a_failed_gpu_suite_stops_the_probe_stage(tools, tmp_path):
    result, out = session(tools, tmp_path, "probe", SUITE_FAILS="1")
    log = (out / "probe.log").read_text()
    assert result.returncode == 1
    assert "FAIL GPU suite" in log and "stopping" in log
    calls = tools[1].read_text()
    assert "probes.py microbench" not in calls and "probes.py profile" not in calls


def test_a_passing_gpu_suite_continues_to_the_probes(tools, tmp_path):
    result, _ = session(tools, tmp_path, "probe")
    calls = tools[1].read_text()
    assert "probes.py microbench" in calls and "probes.py profile" in calls, result.stdout


def test_a_failed_suite_stops_the_bench_stage_before_any_timing(tools, tmp_path):
    result, _ = session(tools, tmp_path, "bench", SUITE_FAILS="1")
    assert result.returncode == 1
    assert "bench/run.py" not in tools[1].read_text()


NCU_ROW = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name",'
           '"Metric Name","Metric Unit","Metric Value"\n"0","1","python3","h","add_one(float *, int)","1","7","m",'
           '"sm__cycles_elapsed.avg","cycle","4,812.50"')
NCU_DENIED = "==ERROR== ERR_NVGPUCTRPERM - permission to access sm__cycles_elapsed.avg denied"


@pytest.mark.parametrize(("output", "code", "passes"), [(NCU_ROW, 0, True), (NCU_DENIED, 0, False), (NCU_ROW, 1, False)])
def test_the_ncu_screen_needs_a_clean_exit_and_a_counter(tools, tmp_path, output, code, passes):
    stub(tools[0], "ncu", f"cat <<'EOF'\n{output}\nEOF\nexit {code}\n")
    _, out = session(tools, tmp_path, "screen")
    log = (out / "screen.log").read_text()
    assert ("PASS ncu counters" in log) is passes and ("FAIL ncu counters" in log) is not passes


@pytest.fixture
def brev(tmp_path):
    """A stub brev with the output shapes of brev 0.6.337: search rows as in a live `brev search
    --json`; `ls --json` as ls.go's MarshalIndent of {"workspaces": [...]}, null when empty. exec and
    copy start child processes, as the real CLI does; exec packs a real archive for retrieval."""
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    state = tmp_path / "state"
    (state / "results" / "screen").mkdir(parents=True)
    (state / "results" / "screen" / "screen.log").write_text("PASS ncu counters\n")
    stub(directory, "brev", f'''
echo "brev $*" >> {state}/calls.log
case $1 in
  search) python3 -c 'import json, os; row = {{"type": "hyperstack_A6000", "provider": "hyperstack:shadeform", "cloud": "hyperstack",
    "gpu_name": "A6000", "gpu_count": 1, "vram_per_gpu_gb": 48, "capability": 8.6, "ram_gb": 58, "vcpus": 28,
    "target_disk_gb": 100, "stoppable": False, "price_per_hour": float(os.environ.get("PRICE", "0.6"))}}
row.update({{"disk_price_per_gb_mo": float(os.environ["DISK_PRICE"])}} if "DISK_PRICE" in os.environ else {{}})
print(json.dumps([row, dict(row, type="other_type", price_per_hour=9.0)], indent=2))' ;;
  create) [ -n "${{CREATE_FAILS:-}}" ] && exit 1; echo "$2" > {state}/name ;;
  exec) if [[ "$*" == *"tar czf"* ]]; then
          [ -n "${{PACK_FAILS:-}}" ] && exit 1
          tar czf {state}/screen.tgz -C {state}/results screen
          echo "Executing on $2"; echo "SHA256 $(shasum -a 256 {state}/screen.tgz | cut -d" " -f1)  ../screen.tgz"
        else sh -c "${{REMOTE:-true}}"; exit 1; fi ;;  # brev exec returns 1 for any remote failure
  copy) sleep "${{COPY_SLEEP:-0}}"; cp {state}/screen.tgz "$4"
        [ -n "${{CORRUPT:-}}" ] && echo x >> "$4"; echo "copied" ;;
  delete) touch {state}/deleted ;;
  ls) [ -n "${{LS_FAILS:-}}" ] && exit 1
      [ -n "${{LS_GARBLED:-}}" ] && {{ echo "You are currently logged out"; exit 0; }}
      names=(); [ -n "${{SIMILAR:-}}" ] && names+=("$(cat {state}/name)-clone")
      if [ -f {state}/name ] && {{ [ ! -f {state}/deleted ] || [ -n "${{NEVER_GONE:-}}" ]; }}; then names+=("$(cat {state}/name)"); fi
      python3 -c 'import json, sys; ws = [{{"name": n, "id": "ws" + str(i), "status": "RUNNING", "build_status": "COMPLETED",
        "shell_status": "READY", "health_status": "", "instance_type": "hyperstack_A6000", "instance_kind": "gpu", "gpu": "A6000"}}
        for i, n in enumerate(sys.argv[1:])]
print(json.dumps({{"workspaces": ws or None}}, indent=2))' "${{names[@]}}" ;;
esac
''')
    return directory, state


# The margin must cover one call, the copy bound and the polls: 5 + 2 + 20 * 0.
ENV = {"IPE_DELETE_MARGIN_S": "7", "IPE_POLL_S": "0", "IPE_COPY_S": "2", "IPE_CALL_S": "5"}


def guard(brev, tmp_path, budget="15", minutes="120", stage="screen", shell=None, **env):
    directory, _ = brev
    command = ["bash", str(ROOT / "bench/brev_session.sh"), "hyperstack_A6000", budget, minutes, "abc123", stage,
               str(tmp_path / "results")]
    if shell:  # run inside a shell pipeline, for example into tee
        command = ["bash", "-c", " ".join(f"'{part}'" for part in command) + shell]
    environment = {**os.environ, "PATH": f"{directory}:{os.environ['PATH']}", **ENV, **env}
    return subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
                            start_new_session=True)


def finish(process):
    output, _ = process.communicate(timeout=60)
    return process.returncode, output


def calls(brev):
    path = brev[1] / "calls.log"
    return path.read_text() if path.exists() else ""


def ledger(tmp_path):
    lines = (tmp_path / "results" / "brev-ledger.tsv").read_text().splitlines()
    return [line.split("\t") for line in lines[1:]]


def write_ledger(tmp_path, *rows):
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "brev-ledger.tsv").write_text(
        "instance\ttype\tseconds\tprice_per_hour\tworst_case_usd\tdeleted\n" + "".join(r + "\n" for r in rows))


def wait_for_create(brev, process):
    began = time.monotonic()
    while not (brev[1] / "name").exists():
        assert process.poll() is None and time.monotonic() - began < 30, "the guard ended before creating"
        time.sleep(0.1)


@pytest.mark.parametrize(("remote", "code"), [("true", 1), ("sleep 0", 1)])
def test_the_instance_is_deleted_after_the_stage(brev, tmp_path, remote, code):
    status, output = finish(guard(brev, tmp_path, REMOTE=remote))
    assert status == code and "deleted, confirmed by brev ls" in output
    name = (brev[1] / "name").read_text().strip()
    assert f"brev exec {name} --host" in calls(brev)
    assert "results retrieved, SHA-256 verified" in output
    assert (tmp_path / "results" / name / "screen.log").read_text() == "PASS ncu counters\n"  # one directory per run
    assert ledger(tmp_path)[-1][0] == name and ledger(tmp_path)[-1][-1] == "1"
    assert not (tmp_path / "results" / ".brev-session.lock").exists()


def test_the_time_limit_ends_the_stage_and_its_children(brev, tmp_path):
    began = time.monotonic()
    status, output = finish(guard(brev, tmp_path, minutes="0.2", REMOTE="sleep 30"))  # 12 s less the 7 s margin
    assert status == 124 and "time limit reached" in output and "deleted, confirmed" in output
    assert time.monotonic() - began < 20  # the remote child was killed with its process group


def test_the_session_ledger_limits_later_runs(brev, tmp_path):
    write_ledger(tmp_path, "earlier\thyperstack_A6000\t108\t0.6\t0.6\t1")
    status, output = finish(guard(brev, tmp_path, minutes="2", REMOTE="sleep 30"))  # 120 - 108 - 7 = 5 s
    assert status == 124 and "limit 5s" in output


def test_a_session_budget_below_one_hour_at_the_price_creates_nothing(brev, tmp_path):
    # Brev does not document its rounding, so every run counts as whole hours.
    status, output = finish(guard(brev, tmp_path, budget="0.5"))
    assert status == 2 and "does not cover" in output and "create" not in calls(brev)


@pytest.mark.parametrize(("row", "message"), [
    ("old\thyperstack_A6000\t600\t0.6\t14.5\t1", "does not cover"),
    ("old\thyperstack_A6000\t600\t0.6\t0.6\t0", "not confirmed deleted"),
    ("old\thyperstack_A6000\t-1792193652\t0.6\t-298698.6\t1", "malformed"),
    ("old\thyperstack_A6000\t600\t0.6", "malformed"),
])
def test_a_spent_unconfirmed_or_malformed_ledger_creates_nothing(brev, tmp_path, row, message):
    write_ledger(tmp_path, row)
    status, output = finish(guard(brev, tmp_path))
    assert status == 2 and message in output and "create" not in calls(brev)


def test_an_empty_ledger_gets_its_header(brev, tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "brev-ledger.tsv").write_text("")
    status, _ = finish(guard(brev, tmp_path))
    assert status == 1 and len(ledger(tmp_path)) == 1


def test_a_second_run_into_the_same_results_is_refused(brev, tmp_path):
    (tmp_path / "results" / ".brev-session.lock").mkdir(parents=True)
    status, output = finish(guard(brev, tmp_path))
    assert status == 2 and "another run is active" in output and "search" not in calls(brev)


def test_an_unknown_stage_or_a_separately_billed_disk_creates_nothing(brev, tmp_path):
    status, output = finish(guard(brev, tmp_path, stage="scren"))
    assert status == 2 and "unknown stage" in output
    status, output = finish(guard(brev, tmp_path, DISK_PRICE="0.1"))
    assert status == 2 and "bills its disk separately" in output and "create" not in calls(brev)


def test_the_recorded_cost_rounds_up(brev, tmp_path):
    finish(guard(brev, tmp_path, PRICE="0.482702"))
    assert ledger(tmp_path)[-1][4] == "0.49"


def test_an_interrupt_deletes_the_instance_even_if_repeated_during_cleanup(brev, tmp_path):
    process = guard(brev, tmp_path, REMOTE="sleep 30", COPY_SLEEP="20", IPE_COPY_S="5", IPE_DELETE_MARGIN_S="10")
    wait_for_create(brev, process)
    time.sleep(1)
    os.killpg(process.pid, signal.SIGINT)  # as a terminal's Ctrl-C does
    time.sleep(1)
    os.killpg(process.pid, signal.SIGINT)  # again, while results are being copied
    status, output = finish(process)
    assert status == 130 and "deleted, confirmed" in output


def test_an_interrupt_with_output_piped_into_a_reader_that_exits(brev, tmp_path):
    """Ctrl-C also ends tee; the cleanup's writes then fail, and it must still delete."""
    log = tmp_path / "tee.log"
    process = guard(brev, tmp_path, shell=f" 2>&1 | tee {log}", REMOTE="sleep 30", COPY_SLEEP="1")
    wait_for_create(brev, process)
    time.sleep(1)
    os.killpg(process.pid, signal.SIGINT)
    finish(process)
    session_log = (tmp_path / "results" / "brev-session.log").read_text()
    assert "deleted, confirmed" in session_log and ledger(tmp_path)[-1][-1] == "1"


def test_a_hung_copy_does_not_delay_deletion_past_its_bound(brev, tmp_path):
    began = time.monotonic()
    _, output = finish(guard(brev, tmp_path, COPY_SLEEP="60"))
    assert "results not retrieved or not verified" in output and "deleted, confirmed" in output
    assert time.monotonic() - began < 20


@pytest.mark.parametrize(("failure", "message"), [("CORRUPT", "archive hash mismatch"), ("PACK_FAILS", "not retrieved")])
def test_unverified_results_are_reported_and_the_instance_still_deleted(brev, tmp_path, failure, message):
    _, output = finish(guard(brev, tmp_path, **{failure: "1"}))
    name = (brev[1] / "name").read_text().strip()
    assert message in output and "SHA-256 verified" not in output and "deleted, confirmed" in output
    assert not (tmp_path / "results" / name).exists()


def test_deletion_is_confirmed_by_exact_name_only(brev, tmp_path):
    """An instance whose name contains this one's is someone else's; it does not keep this one listed."""
    _, output = finish(guard(brev, tmp_path, SIMILAR="1"))
    assert "deleted, confirmed" in output


@pytest.mark.parametrize("failure", ["NEVER_GONE", "LS_FAILS", "LS_GARBLED"])
def test_deletion_that_brev_ls_does_not_confirm_exits_3_and_blocks_the_next_run(brev, tmp_path, failure):
    status, output = finish(guard(brev, tmp_path, **{failure: "1"}))
    assert status == 3 and "STILL LISTED" in output and ledger(tmp_path)[-1][-1] == "0"
    if failure == "NEVER_GONE":
        assert calls(brev).count("brev delete") > 1  # deletion is requested again while listed
    status, output = finish(guard(brev, tmp_path))
    assert status == 2 and "not confirmed deleted" in output


def test_a_failed_create_still_checks_for_an_instance(brev, tmp_path):
    status, output = finish(guard(brev, tmp_path, CREATE_FAILS="1"))
    assert status == 1 and "deleted, confirmed" in output


def test_session_refuses_an_existing_results_directory(tools, tmp_path):
    (tmp_path / "out").mkdir()
    result, _ = session(tools, tmp_path, "probe")
    assert result.returncode == 2 and "never overwritten" in result.stdout


@pytest.mark.parametrize("stage", ["probe", "long", "bench"])
def test_every_gpu_stage_fetches_weights_and_checks_cuda_first(tools, tmp_path, stage):
    session(tools, tmp_path, stage)
    log = tools[1].read_text()
    assert "torch.cuda.is_available" in log and "hf download Qwen/Qwen3-1.7B" in log and "ipe.checkpoint verify" in log


def test_a_margin_that_cannot_cover_cleanup_is_refused(brev, tmp_path):
    status, output = finish(guard(brev, tmp_path, IPE_DELETE_MARGIN_S="6"))
    assert status == 2 and "too small" in output and "create" not in calls(brev)
