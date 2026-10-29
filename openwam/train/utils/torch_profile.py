"""Opt-in ``torch.profiler`` window for the training loop (``training.torch_profile``).

Profiles ``active`` steps after skipping ``wait`` and warming up ``warmup`` steps, on
the ranks listed in ``ranks``, and writes ``<output>/torch_profile/rank<r>.json.gz``
(Chrome trace) plus ``rank<r>_ops.txt`` (top CUDA ops). While the window is open,
:class:`StepTimer` stages show up as ``stage/<name>`` ranges and each MoT layer body
as ``mot_layer`` (ranges inside ``stage/ds_backward`` are activation recompute), so
``scripts/perf/analyze_torch_profile.py`` can split GPU time by stage and kernel kind.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TorchProfile:
    def __init__(self, *, enabled: bool, wait: int = 5, warmup: int = 2, active: int = 3,
                 ranks=(0,), record_shapes: bool = False, with_stack: bool = False,
                 output_path: Optional[str] = None, rank: int = 0):
        self.enabled = bool(enabled) and rank in {int(r) for r in ranks}
        self.prof = None
        if not self.enabled:
            return
        out_dir = os.path.join(output_path or ".", "torch_profile")
        os.makedirs(out_dir, exist_ok=True)

        def on_trace_ready(prof):
            prof.export_chrome_trace(os.path.join(out_dir, f"rank{rank}.json.gz"))
            with open(os.path.join(out_dir, f"rank{rank}_ops.txt"), "w") as f:
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
            logger.info("torch_profile: wrote %s/rank%d.json.gz", out_dir, rank)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=int(wait), warmup=int(warmup), active=int(active), repeat=1),
            on_trace_ready=on_trace_ready,
            record_shapes=bool(record_shapes),
            with_stack=bool(with_stack),
        )
        self.prof.start()

    @classmethod
    def from_cfg(cls, cfg, **kw) -> "TorchProfile":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            wait=int(cfg.get("wait", 5)),
            warmup=int(cfg.get("warmup", 2)),
            active=int(cfg.get("active", 3)),
            ranks=list(cfg.get("ranks", [0])),
            record_shapes=bool(cfg.get("record_shapes", False)),
            with_stack=bool(cfg.get("with_stack", False)),
            **kw,
        )

    def step(self) -> None:
        if self.prof is not None:
            self.prof.step()

    def stop(self) -> None:
        if self.prof is not None:
            self.prof.stop()
            self.prof = None
