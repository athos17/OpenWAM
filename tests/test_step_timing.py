"""StepTimer: disabled no-op, nested stage accounting, window summary (single process)."""

import json
import time

from openwam.train.utils.step_timing import StepTimer


def test_disabled_is_noop():
    t = StepTimer(enabled=False)
    t.begin_step()
    with t.stage("x"):
        pass
    assert t.end_step() == {} and t.maybe_summarize(50) is None


def test_stages_and_summary(tmp_path):
    t = StepTimer(enabled=True, summary_every=2, samples_per_step=8, output_path=str(tmp_path))
    t._cuda = False  # CPU-only accounting in this test

    class Engine:
        def step(self):
            time.sleep(0.002)

    eng = Engine()
    t.wrap(eng, "step", "ds_step")
    for _ in range(2):
        t.begin_step()
        with t.stage("compute_loss"):
            with t.stage("prepare_inputs"):
                time.sleep(0.003)
        eng.step()
        eng.step()
        rec = t.end_step()
        t.finish_iteration()
    assert rec["compute_loss_cpu_s"] >= rec["prepare_inputs_cpu_s"] >= 0.003
    assert rec["ds_step_cpu_s"] >= 0.004  # two calls summed
    assert "iteration_cpu_s" in rec and "data_wait_cpu_s" in rec
    summary = t.maybe_summarize(2)
    assert summary["window_steps"] == 2 and summary["ranks"] == 1 and summary["samples_per_sec"] > 0
    line = json.loads((tmp_path / "step_timing.jsonl").read_text())
    assert line["stages"]["ds_step_cpu_s"]["rank_max"] >= 0.004


def test_mfu_counts_first_step_only(tmp_path):
    import torch

    from openwam.train.utils.step_timing import detect_peak_tflops

    assert detect_peak_tflops("NVIDIA H20") == 148.0
    assert detect_peak_tflops("NVIDIA A800-SXM4-80GB") == 312.0
    assert detect_peak_tflops("Some GPU") is None

    t = StepTimer(enabled=True, summary_every=3, samples_per_step=4, output_path=str(tmp_path), mfu=True, peak_tflops=1.0)
    t._cuda = False
    lin = torch.nn.Linear(64, 32, bias=False)
    x = torch.randn(16, 64)
    for _ in range(3):
        t.begin_step()
        with t.count_flops("encoder"):
            x @ torch.randn(64, 8)  # 2 * 16 * 64 * 8
        with t.count_flops("model"):
            lin(x)  # 2 * 16 * 64 * 32 forward
        t.end_step()
        t.finish_iteration()
    assert t.forward_flops == {"encoder": 2 * 16 * 64 * 8, "model": 2 * 16 * 64 * 32}
    summary = t.maybe_summarize(3)
    stages = summary["stages"]
    model_step = 3 * 2 * 16 * 64 * 32
    assert abs(stages["model_tflop_per_step"]["median_of_ranks"] - model_step / 1e12) < 1e-18
    it = stages["iteration_cpu_s"]["median_of_ranks"]
    assert abs(summary["mfu"] - model_step / (it * 1e12)) / summary["mfu"] < 1e-6
    assert summary["mfu_incl_encoders"] > summary["mfu"]
