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
