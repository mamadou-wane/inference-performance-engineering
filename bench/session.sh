#!/usr/bin/env bash
# GPU session steps, run on the instance from a clone at a fixed commit:
#   bench/session.sh screen OUT   requirements check of a new instance type; no weights; CUDA 13 tools if missing
#   bench/session.sh probe OUT    weights, GPU suite at 1,024 tokens, microbenchmarks, W1 profile
#   bench/session.sh long OUT     GPU suite at 4,224 tokens; required before any W3 result
#   bench/session.sh bench OUT    workloads: implementations interleaved over 3 launches, vllm bench latency
# Every stage writes to OUT, which is copied off the instance and never committed.
set -uo pipefail
stage=${1:-} out=${2:-}
[ -n "$stage" ] && [ -n "$out" ] || { sed -n 2,7p "$0"; exit 2; }
[ -e "$out" ] && { echo "$out exists; results are never overwritten"; exit 2; }
mkdir -p "$out"
log() { echo "$*" | tee -a "$out/$stage.log"; }
check() { local name=$1; shift; if "$@" >> "$out/$stage.log" 2>&1; then log "PASS $name"; else log "FAIL $name"; fi; }
run() { uv run --locked --extra bench "$@"; }
require() { grep -q "^FAIL" "$out/$stage.log" && { log "stopping: a required check failed"; exit 1; }; return 0; }
export PATH="$PWD/.venv/bin:$PATH"  # tools started directly (ncu, compute-sanitizer) still find the venv's ninja

MODEL=weights/qwen3-1.7b
REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export HF_HOME=$PWD/hf-cache

# torch's CUDA version, and the MD5 NVIDIA publishes for its runfile in docs/sidebar/md5sum.txt
CUDA=13.0.3 RUNFILE=cuda_13.0.3_580.126.20_linux.run RUNFILE_MD5=13acdcabcb9e939f716c29eff8c12181

cuda_toolkit() {  # prints the first toolkit whose nvcc is CUDA 13 and that holds ncu and compute-sanitizer
  local nvcc root
  nvcc=$(command -v nvcc)
  for root in ${nvcc:+"${nvcc%/bin/nvcc}"} /usr/local/cuda /usr/local/cuda-13*; do
    "$root/bin/nvcc" --version 2>/dev/null | grep -q "release 13\." && [ -x "$root/bin/ncu" ] &&
      [ -x "$root/bin/compute-sanitizer" ] && { echo "$root"; return 0; }
  done
  return 1
}

cuda_prerequisites() {  # prints what the install lacks; the runfile's compiler check runs `which gcc`
  local tool major
  for tool in curl md5sum setpriv; do command -v "$tool" > /dev/null || { echo "no $tool"; return 1; }; done
  which gcc > /dev/null 2>&1 || { echo "which gcc fails"; return 1; }
  major=$(gcc -dumpversion) major=${major%%.*}
  [ "$major" -ge 6 ] 2>/dev/null && [ "$major" -le 15 ] || { echo "gcc $major, CUDA 13.0 supports 6 to 15"; return 1; }
}

cuda_download() {
  curl -fsSL --retry 3 -o "$HOME/$RUNFILE" "https://developer.download.nvidia.com/compute/cuda/$CUDA/local_installers/$RUNFILE" &&
    echo "$RUNFILE_MD5  $HOME/$RUNFILE" | md5sum -c -
}

cuda_install() {  # ROOT: toolkit only, no driver; no_new_privs blocks sudo
  mkdir -p "$HOME/cuda-tmp" &&
    setpriv --no-new-privs sh "$HOME/$RUNFILE" --silent --toolkit --toolkitpath="$1" --defaultroot="$1" \
      --no-man-page --tmpdir="$HOME/cuda-tmp"
}

weights_and_gpu() {  # every GPU stage starts on a fresh instance
  check "CUDA visible to torch" run python -c "import torch; assert torch.cuda.is_available()"
  check "weights download" run hf download Qwen/Qwen3-1.7B --revision $REVISION --local-dir $MODEL
  check "weights verify" run python -m ipe.checkpoint verify checkpoints/qwen3-1.7b.json $MODEL
  require
}

sample_gpu() {  # SM clock, temperature and power every 500 ms for the rest of the stage
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -lms 500 \
    > "$out/$stage-gpu-samples.csv" &
  sampler=$!
  trap 'kill $sampler 2>/dev/null' EXIT
}

case $stage in
screen)
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | tee "$out/gpu.csv"
  IFS=, read -r _ memory driver capability < "$out/gpu.csv"
  memory=${memory//[^0-9]/}; driver=${driver%%.*}; driver=${driver// /}
  check "memory ${memory} MiB >= 22888" test "$memory" -ge 22888
  check "driver $driver >= 580" test "$driver" -ge 580
  check "compute capability $capability >= 8.0" python3 -c "import sys; sys.exit(float('$capability') < 8.0)"
  log "host: $(nproc) CPUs, $(free -g | awk '/Mem:/ {print $2}') GB RAM, $(df -h . | awk 'NR == 2 {print $4}') free disk"
  # The install supplies tools; the checks below, unchanged, decide whether they work.
  if root=$(cuda_toolkit); then
    log "CUDA toolkit: existing, $root"
  else
    root=$HOME/cuda-$CUDA start=$(date +%s)
    if ! problem=$(cuda_prerequisites); then
      result="not installed, $problem"
    elif ! cuda_download >> "$out/$stage.log" 2>&1; then
      result="download or checksum failed"
    else
      cuda_install "$root" >> "$out/$stage.log" 2>&1
      result="installer exit $?"  # can be nonzero as non-root with the toolkit complete
      cp /tmp/cuda-installer.log "$out/" 2>/dev/null  # where the non-root installer writes its log
    fi
    log "CUDA toolkit: $CUDA $result, $root, $(( $(date +%s) - start )) s"
  fi
  export PATH="$root/bin:$PATH" CUDA_HOME=$root  # the extension build, the sanitizer and ncu use this toolkit
  check "nvcc is CUDA 13" sh -c 'nvcc --version | grep -q "release 13\."'
  log "execution context: $([ -f /.dockerenv ] && echo container || echo host)"
  log "sudo without password: $(sudo -n true 2>/dev/null && echo yes || echo no)"
  log "profiling restricted to admins: $(grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null || echo unknown)"
  start=$(date +%s)
  check "locked install" uv sync --locked --extra bench
  log "install seconds: $(( $(date +%s) - start ))"
  log "CPU model: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  run python bench/probes.py env > "$out/environment.json"
  check "extension build" run python bench/probes.py extension
  check "compute-sanitizer memcheck" compute-sanitizer --tool memcheck --error-exitcode 1 .venv/bin/python bench/probes.py extension
  # ncu must exit 0 and its CSV must hold a positive counter value; an error message can name the metric.
  check "ncu counters" sh -c "ncu --csv --kernel-name add_one --metrics sm__cycles_elapsed.avg .venv/bin/python bench/probes.py extension > '$out/ncu.csv' 2>&1 && .venv/bin/python bench/probes.py ncu-check '$out/ncu.csv'"
  log "nsys: $(nsys --version 2>/dev/null || echo unavailable)"  # optional; the PyTorch profiler supplies the timeline
  check "vllm extensions" run python bench/probes.py vllm
  ;;
probe)
  sample_gpu
  weights_and_gpu
  check "GPU suite, 1,024 tokens" run pytest -q -m gpu -k "not 4224" --junitxml="$out/gpu-suite.xml" -o junit_logging=all
  require  # no profiling or timing after a failed correctness check
  check "microbenchmarks" sh -c "uv run --locked --extra bench python bench/probes.py microbench > '$out/microbench.json'"
  check "W1 profile" sh -c "uv run --locked --extra bench python bench/probes.py profile $MODEL '$out' > '$out/profile.json'"
  ;;
long)
  sample_gpu
  weights_and_gpu
  check "GPU suite, 4,224 tokens" run pytest -q -m gpu -k 4224 --junitxml="$out/gpu-suite-long.xml" -o junit_logging=all
  ;;
bench)
  sample_gpu
  workloads=${WORKLOADS:-W1}
  weights_and_gpu
  # No timing counts in a session whose GPU suite fails; W3 also needs the 4,224-token suite.
  check "GPU suite, 1,024 tokens" run pytest -q -m gpu -k "not 4224" --junitxml="$out/gpu-suite.xml"
  [[ $workloads == *W3* ]] && check "GPU suite, 4,224 tokens" run pytest -q -m gpu -k 4224 --junitxml="$out/gpu-suite-long.xml"
  require
  for launch in 1 2 3; do
    for impl in engine vllm transformers-dynamic transformers-static; do
      check "$impl launch $launch" run python bench/run.py run --impl $impl --model $MODEL --workloads $workloads \
        --launch $launch --out "$out"
    done
    check "vllm-calibration launch $launch" run python bench/run.py run --impl vllm-calibration --model $MODEL \
      --workloads W1 --launch $launch --out "$out"
    check "vllm bench latency launch $launch" run vllm bench latency --model $MODEL --dtype bfloat16 \
      --input-len 512 --output-len 256 --batch-size 1 --num-iters-warmup 10 --num-iters 30 \
      --max-model-len 768 --max-num-seqs 1 --seed 0 --disable-log-stats \
      --no-enable-prefix-caching --disable-detokenize --output-json "$out/bench-latency-$launch.json"
  done
  # Agreement compares vllm bench latency with vllm-calibration, which matches its sampling and prompts.
  check "summary" sh -c "uv run --locked --extra bench python bench/run.py summarize '$out' --vllm-bench '$out'/bench-latency-*.json > '$out/summary.json'"
  ;;
*)
  sed -n 2,7p "$0"; exit 2
  ;;
esac
grep -q '^FAIL' "$out/$stage.log" && exit 1 || exit 0
