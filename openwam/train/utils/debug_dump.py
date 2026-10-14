"""Opt-in per-step fingerprints for A/B equivalence checks.

Set ``OPENWAM_DEBUG_BATCH_DUMP=<dir>`` (and optionally
``OPENWAM_DEBUG_BATCH_DUMP_STEPS``, default 20) to append one JSON line per
call to ``<dir>/rank-<RANK>.jsonl`` for the first N training steps. Records the
sample identities, prompts, sampled timesteps and sha256 of the noise tensors —
everything that must be identical between an online-encoding run and an
offline-cache run with the same seed. Disabled (a dict lookup) otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_DIR = os.environ.get("OPENWAM_DEBUG_BATCH_DUMP")
_LIMIT = int(os.environ.get("OPENWAM_DEBUG_BATCH_DUMP_STEPS", "20"))
_counts: dict = {}


def enabled(kind: str) -> bool:
    return bool(_DIR) and _counts.get(kind, 0) < _LIMIT


def tensor_sha256(tensor) -> str:
    t = tensor.detach().to("cpu").contiguous()
    import torch

    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def dump(kind: str, payload: dict) -> None:
    if not enabled(kind):
        return
    index = _counts.get(kind, 0)
    _counts[kind] = index + 1
    path = Path(_DIR) / f"rank-{os.environ.get('RANK', '0')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"kind": kind, "index": index, **payload}, ensure_ascii=False) + "\n")
