"""Per-step stage timing for the training loop (``training.step_timing``).

Each stage records CPU wall time and, on CUDA, the current-stream elapsed time
between two events (includes GPU idle inside the stage; not kernel-only time).
Events are read in :meth:`StepTimer.end_step`, which the loop calls after the
step's metrics reduction has already synchronized the stream, so timing adds no
extra sync. Stages nest (``prepare_inputs`` inside ``compute_loss``) and are not
additive; ``step`` is the wall time from receiving the batch to the end of the
step and ``iteration`` the time between consecutive step starts. ``data_wait``
runs from the end of the previous loop iteration (:meth:`finish_iteration`) to the
batch arriving. Stages recorded after :meth:`end_step` (logging, checkpointing) are
reported with the following step.

Every ``summary_every`` steps all ranks exchange their window medians and rank 0
prints ``STEP_TIMING {...}`` and appends it to ``<output>/step_timing.jsonl``
(per-stage median of the rank medians and the rank max, plus samples/s).
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import time
from collections import defaultdict
from typing import Optional

import torch


class StepTimer:
    def __init__(
        self,
        *,
        enabled: bool,
        summary_every: int = 50,
        samples_per_step: int = 0,
        output_path: Optional[str] = None,
        is_main: bool = True,
    ):
        self.enabled = bool(enabled)
        self.summary_every = int(summary_every)
        self.samples_per_step = int(samples_per_step)
        self.output_path = output_path
        self.is_main = is_main
        self._cuda = torch.cuda.is_available()
        self._pending: list = []
        self._window: dict = defaultdict(list)
        self._step_t0 = None
        self._prev_step_t0 = None
        self._loop_end = None
        self._data_wait = 0.0
        self.last: dict = {}

    @classmethod
    def from_cfg(cls, cfg, **kw) -> "StepTimer":
        cfg = cfg or {}
        return cls(enabled=bool(cfg.get("enabled", False)), summary_every=int(cfg.get("summary_every", 50)), **kw)

    # ------------------------------------------------------------------ recording

    @contextlib.contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        start = end = None
        if self._cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            if end is not None:
                end.record()
            self._pending.append((name, t1 - t0, start, end))

    def wrap(self, obj, method: str, name: str) -> None:
        """Time every call of ``obj.<method>`` as stage ``name`` (instance-level patch)."""
        if not self.enabled:
            return
        fn = getattr(obj, method)

        def timed(*args, **kwargs):
            with self.stage(name):
                return fn(*args, **kwargs)

        setattr(obj, method, timed)

    def begin_step(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self._data_wait = 0.0 if self._loop_end is None else now - self._loop_end
        self._prev_step_t0, self._step_t0 = self._step_t0, now

    def end_step(self) -> dict:
        """Resolve this step's stages into ``{name_cpu_s, name_gpu_s}`` (summed per name)."""
        if not self.enabled:
            return {}
        out = {"data_wait_cpu_s": self._data_wait, "step_cpu_s": time.perf_counter() - self._step_t0}
        if self._prev_step_t0 is not None:
            out["iteration_cpu_s"] = self._step_t0 - self._prev_step_t0
        if self._pending and self._pending[-1][3] is not None:
            self._pending[-1][3].synchronize()
        for name, cpu_s, start, end in self._pending:
            out[f"{name}_cpu_s"] = out.get(f"{name}_cpu_s", 0.0) + cpu_s
            if start is not None:
                out[f"{name}_gpu_s"] = out.get(f"{name}_gpu_s", 0.0) + start.elapsed_time(end) / 1000.0
        self._pending.clear()
        if self._cuda:
            # Allocator high-water mark since start (GiB); a running max, not per step.
            out["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / (1 << 30)
            out["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / (1 << 30)
        for k, v in out.items():
            self._window[k].append(v)
        self.last = out
        return out

    def finish_iteration(self) -> None:
        if self.enabled:
            self._loop_end = time.perf_counter()

    # ------------------------------------------------------------------ summary

    def maybe_summarize(self, global_step: int) -> Optional[dict]:
        """Collective on all ranks every ``summary_every`` steps; rank 0 logs the result."""
        if not self.enabled or self.summary_every <= 0 or global_step % self.summary_every != 0:
            return None
        local = {k: statistics.median(v) for k, v in self._window.items() if v}
        n = len(self._window.get("step_cpu_s", []))
        self._window.clear()
        gathered = [local]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        keys = sorted(set().union(*gathered))
        stages = {
            k: {
                "median_of_ranks": statistics.median([g[k] for g in gathered if k in g]),
                "rank_max": max(g[k] for g in gathered if k in g),
            }
            for k in keys
        }
        summary = {"step": global_step, "window_steps": n, "ranks": len(gathered), "stages": stages}
        it = stages.get("iteration_cpu_s")
        if it and self.samples_per_step:
            summary["samples_per_sec"] = self.samples_per_step / it["median_of_ranks"]
        if self.is_main:
            line = json.dumps(summary)
            print("STEP_TIMING " + line, flush=True)
            if self.output_path:
                with open(os.path.join(self.output_path, "step_timing.jsonl"), "a") as f:
                    f.write(line + "\n")
        return summary
