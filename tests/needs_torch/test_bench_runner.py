"""The benchmark runner on a synthetic checkpoint on the CPU: every implementation produces exactly
N new tokens, even when the checkpoint's generation defaults would stop at EOS. vLLM is Linux and
GPU only and is not covered here."""

import json
import shutil

import pytest
import torch

from bench.run import BENCH_LATENCY_IDS, FIRST_SPECIAL_ID, Runner, prompts


@pytest.mark.parametrize("impl", ("engine", "transformers-dynamic", "transformers-static"))
def test_generation_ignores_the_checkpoints_eos_defaults(checkpoints, tmp_path, impl):
    directory = tmp_path / "with-generation-defaults"
    shutil.copytree(checkpoints["gqa"], directory)
    # Every ID is an EOS ID, so honoring the defaults would stop after the first token.
    (directory / "generation_config.json").write_text(json.dumps({"do_sample": True, "eos_token_id": list(range(96)),
                                                                  "temperature": 0.6, "top_k": 20, "top_p": 0.95}))
    runner = Runner(impl, str(directory), "cpu", 12, 2)
    prompt = torch.randint(0, 96, (2, 8), generator=torch.Generator().manual_seed(0))
    assert runner.latency(prompt, 4) > 0  # raises unless both rows got 4 new tokens


def test_calibration_prompts_follow_vllm_bench_latency_and_baselines_avoid_special_ids():
    calibration, baseline = prompts(1, 512, high=BENCH_LATENCY_IDS), prompts(16, 512)
    assert calibration.shape == (1, 512) and int(calibration.max()) < 10000
    assert baseline.shape == (16, 512) and int(baseline.max()) < FIRST_SPECIAL_ID
    assert torch.equal(prompts(16, 512), baseline)  # fixed seed
