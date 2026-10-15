"""Subtask annotations for language-conditioned training of single-task LeRobot buckets.

The annotation is a sidecar parquet (``scripts/subtask/vocab.py`` -> ``subtasks.parquet``)
that tiles every episode with contiguous segments::

    episode_index, start, end   frame range [start, end)
    type                        PICK | STILL
    prompt, prompt_nohand       canonical subtask instructions (PICK)
    variants                    paraphrases (PICK)
    exclude                     drop windows that start inside this segment

Window semantics (one window = observation frame ``o`` + ``num_frames - 1`` actions):

* ``d`` is the RTC delay (0 without RTC): actions ``[o, o + d)`` are the committed prefix
  and the subtask that must be commanded is the one containing ``o + d``.
* *sub* mode: prompt of that segment; the window is truncated at the segment end and the
  action tail is held (repeat of the last real target) — "after finishing, stay".
* *main* mode: the original episode-level instruction, untruncated.

Nothing here changes the dataset's window enumeration, so encoder caches keyed by window
index stay valid; truncated clips are served from a separate boundary cache whose rows are
``(window_index, raw_len)`` pairs enumerated by :meth:`SubtaskIndex.boundary_rows`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("episode_index", "start", "end", "type", "prompt", "prompt_nohand", "variants", "exclude")


@dataclass(frozen=True)
class SubtaskConfig:
    annotation_path: Optional[str] = None
    main_prompt: str = "Pick all items into the bin"
    still_prompt: str = "Stay still."
    main_prompt_prob: float = 0.15
    prompt_mix: dict = field(default_factory=lambda: {"canonical": 0.25, "nohand": 0.25, "variants": 0.5})
    rtc_max_delay_steps: int = 0
    hold: bool = True
    supervise_padded_video: bool = False
    decision_weight: int = 2
    decision_pre_frames: int = 16
    decision_post_frames: int = 45
    holdout_episodes: tuple = ()

    @classmethod
    def from_any(cls, cfg) -> "SubtaskConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, SubtaskConfig):
            return cfg
        raw = dict(cfg.items()) if hasattr(cfg, "items") else dict(cfg)
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown subtask config keys: {sorted(unknown)}")
        if "prompt_mix" in raw and raw["prompt_mix"] is not None:
            raw["prompt_mix"] = dict(raw["prompt_mix"].items()) if hasattr(raw["prompt_mix"], "items") else dict(raw["prompt_mix"])
        if raw.get("holdout_episodes") is not None:
            raw["holdout_episodes"] = tuple(int(e) for e in raw["holdout_episodes"])
        cfg = cls(**{k: v for k, v in raw.items() if v is not None})
        mix = cfg.prompt_mix
        if set(mix) - {"canonical", "nohand", "variants"} or abs(sum(mix.values()) - 1.0) > 1e-6:
            raise ValueError(f"subtask.prompt_mix must weight canonical/nohand/variants and sum to 1, got {mix}")
        if not 0.0 <= cfg.main_prompt_prob <= 1.0:
            raise ValueError("subtask.main_prompt_prob must be in [0, 1]")
        if cfg.rtc_max_delay_steps < 0 or cfg.decision_weight < 1:
            raise ValueError("subtask.rtc_max_delay_steps must be >= 0 and decision_weight >= 1")
        return cfg

    @property
    def enabled(self) -> bool:
        return bool(self.annotation_path)

    def digest(self) -> str:
        """Fields that change which clips / prompts the reader can produce."""
        keep = {"annotation_sha256": sha256_file(self.annotation_path), "main_prompt": self.main_prompt,
                "still_prompt": self.still_prompt, "rtc_max_delay_steps": self.rtc_max_delay_steps}
        return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SubtaskIndex:
    """Frame -> segment lookup for one reader bucket (episode-local indices)."""

    def __init__(self, cfg: SubtaskConfig, table: pd.DataFrame, episode_index: np.ndarray, lengths: np.ndarray):
        missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"subtask annotation lacks columns {missing}")
        self.cfg = cfg
        table = table.sort_values(["episode_index", "start"]).reset_index(drop=True)
        self.start = table["start"].to_numpy(np.int64)
        self.end = table["end"].to_numpy(np.int64)
        self.is_still = (table["type"] == "STILL").to_numpy()
        self.exclude = table["exclude"].to_numpy(bool)
        self.prompts = []
        for r in table.itertuples():
            if r.type == "STILL":
                self.prompts.append({"canonical": [cfg.still_prompt]})
            else:
                self.prompts.append({"canonical": [r.prompt], "nohand": [r.prompt_nohand],
                                     "variants": [str(v) for v in (r.variants if r.variants is not None else [])]})
        # per local episode: global segment-row offsets and per-frame segment id
        self.frame_seg = []
        by_ep = {int(e): g.index.to_numpy() for e, g in table.groupby("episode_index")}
        for ep, n in zip(episode_index.tolist(), lengths.tolist()):
            rows = by_ep.get(int(ep))
            if rows is None:
                raise ValueError(f"subtask annotation has no segments for episode {ep}")
            s, e = self.start[rows], self.end[rows]
            if s[0] != 0 or e[-1] != n or not np.array_equal(s[1:], e[:-1]):
                raise ValueError(f"subtask segments of episode {ep} do not tile [0, {n})")
            self.frame_seg.append(np.repeat(rows, e - s))
        self.holdout = {int(e) for e in cfg.holdout_episodes}
        self.episode_index = episode_index

    def seg_at(self, ep_local: int, frame: int) -> int:
        return int(self.frame_seg[ep_local][frame])

    def resolve(self, ep_local: int, offset: int, delay: int, full_len: int) -> tuple[int, int, int]:
        """(segment, raw_len, delay) of a sub-mode window; ``delay`` is clamped so that the
        commanded segment is never an excluded one (it then stays in the first segment)."""
        s0 = self.seg_at(ep_local, offset)
        s = self.seg_at(ep_local, offset + delay)
        if s != s0 and self.exclude[s]:
            delay, s = int(self.end[s0] - offset - 1), s0
        return s, int(min(full_len, self.end[s] - offset)), int(delay)

    def window_ok(self, ep_local: int, offset: int) -> bool:
        return not self.exclude[self.seg_at(ep_local, offset)] and int(self.episode_index[ep_local]) not in self.holdout

    def prompt_choices(self, seg: int) -> dict:
        return self.prompts[seg]

    def all_prompts(self) -> list:
        out = {self.cfg.main_prompt, self.cfg.still_prompt}
        for p, ex in zip(self.prompts, self.exclude):
            if not ex:
                for v in p.values():
                    out.update(x for x in v if x)
        return sorted(out)

    def decision_window(self, ep_local: int, offset: int) -> bool:
        """``offset`` lies just before / at the start of a kept PICK (where the command matters)."""
        seg = self.seg_at(ep_local, min(offset + self.cfg.decision_pre_frames, len(self.frame_seg[ep_local]) - 1))
        if self.is_still[seg] or self.exclude[seg]:
            return False
        return self.start[seg] - self.cfg.decision_pre_frames <= offset < self.start[seg] + self.cfg.decision_post_frames

    def boundary_rows(self, windows: np.ndarray, valid_end: np.ndarray, num_frames: int) -> np.ndarray:
        """(window_index, ep_local, offset, raw_len) for every truncated sub-mode clip reachable
        from a kept window, over all delays 0..rtc_max_delay_steps (sorted, unique)."""
        rows = set()
        for idx, (ep, off) in enumerate(windows.tolist()):
            if self.exclude[self.seg_at(ep, off)]:
                continue
            full = min(num_frames, int(valid_end[ep]) - off)
            for d in range(min(self.cfg.rtc_max_delay_steps, full - 1) + 1):
                _, raw_len, _ = self.resolve(ep, off, d, full)
                if raw_len < full:
                    rows.add((idx, ep, off, raw_len))
        return np.asarray(sorted(rows), dtype=np.int64).reshape(-1, 4)


def load_table(path) -> pd.DataFrame:
    return pd.read_parquet(path)
