"""Marvin Pro real-robot LeRobot v3 reader for OpenWAM-alpha fine-tuning.

The converted dataset stores 76-D source vectors, but training consumes the
62-D physical contract below before scattering it into OpenWAM-alpha's fixed
80-D space::

    [left EEF xyz+rot6d (9), left hand joints (22),
     right EEF xyz+rot6d (9), right hand joints (22)]

The independent gripper slots and the final two hand slots on each side are
intentionally absent and therefore remain masked after the unified scatter.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import ClassVar, Optional

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.normalization import STAT_KEYS, apply_normalization
from openwam.dataloader.utils.subtask import SubtaskConfig, SubtaskIndex, load_table

logger = logging.getLogger(__name__)

SOURCE_VECTOR_DIM = 76
RAW_ACTION_DIM = 62

LEFT_HAND = slice(14, 36)
RIGHT_HAND = slice(36, 58)
LEFT_EEF = slice(58, 67)
RIGHT_EEF = slice(67, 76)

LEFT_ROT6D_RAW = tuple(range(3, 9))
RIGHT_ROT6D_RAW = tuple(range(34, 40))
ROT6D_RAW_DIMS = LEFT_ROT6D_RAW + RIGHT_ROT6D_RAW

OPENWAM_ROT6D_LAYOUT = "[R00,R10,R20,R01,R11,R21]"
OPENWAM_ROT6D_CONVENTION = "first_two_columns"


def marvin_raw_vector(source: np.ndarray) -> np.ndarray:
    """Select and order absolute EEF poses plus absolute hand joints."""

    source = np.asarray(source, dtype=np.float32)
    if source.shape[-1] != SOURCE_VECTOR_DIM:
        raise ValueError(f"Marvin source vector must be {SOURCE_VECTOR_DIM}-D, got shape {source.shape}")
    return np.concatenate(
        [
            source[..., LEFT_EEF],
            source[..., LEFT_HAND],
            source[..., RIGHT_EEF],
            source[..., RIGHT_HAND],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def rot6d_rows_to_columns(rows6d: np.ndarray) -> np.ndarray:
    """Convert legacy first-two-rows rot6d to OpenWAM first-two-columns.

    The source rows are orthonormalized before the third row is reconstructed.
    This makes the conversion robust to small interpolation/serialization drift.
    """

    rows6d = np.asarray(rows6d, dtype=np.float32)
    if rows6d.shape[-1] != 6:
        raise ValueError(f"rot6d input must end in 6 values, got shape {rows6d.shape}")
    row0 = rows6d[..., :3]
    row1 = rows6d[..., 3:]
    row0 = row0 / np.maximum(np.linalg.norm(row0, axis=-1, keepdims=True), 1e-8)
    row1 = row1 - np.sum(row0 * row1, axis=-1, keepdims=True) * row0
    row1 = row1 / np.maximum(np.linalg.norm(row1, axis=-1, keepdims=True), 1e-8)
    row2 = np.cross(row0, row1)
    return np.concatenate(
        [
            np.stack([row0[..., 0], row1[..., 0], row2[..., 0]], axis=-1),
            np.stack([row0[..., 1], row1[..., 1], row2[..., 1]], axis=-1),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _materialize_stats(raw: dict, *, label: str) -> dict:
    out = {}
    for key in STAT_KEYS:
        if key not in raw:
            raise KeyError(f"Marvin normalization block {label!r} is missing {key!r}")
        value = np.asarray(raw[key], dtype=np.float32)
        if value.shape != (RAW_ACTION_DIM,):
            raise ValueError(
                f"Marvin normalization block {label!r}.{key} must have shape "
                f"({RAW_ACTION_DIM},), got {value.shape}"
            )
        out[key] = value

    expected = {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0}
    idx = list(ROT6D_RAW_DIMS)
    stale = [key for key, value in expected.items() if not np.allclose(out[key][idx], value, atol=1e-6)]
    if stale:
        raise ValueError(
            f"Marvin normalization block {label!r} violates the rot6d identity contract "
            f"for {stale}; rerun scripts/convert_marvin_lerobot_v3.py"
        )
    return out


class MarvinDataset(LeRobotV3Reader):
    """Single-bucket Marvin Pro reader with the OpenWAM-alpha 80-D contract."""

    DATASET_NAME = "Marvin"
    NEEDED_COLS = ("action", "observation.state", "task_index")
    HEAD_CAMERA = "observation.images.left_head"
    LEFT_WRIST_CAMERA = "observation.images.left_wrist"
    RIGHT_WRIST_CAMERA = "observation.images.right_wrist"
    ACTION_DIM = RAW_ACTION_DIM
    DEPLOY_ACTION_MODE = "eef"
    DEFAULT_NORMALIZE_MODE = "min-max"

    CONFIG_KEYS: ClassVar[tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + ("subtask", "boundary_latent_cache_dir")

    def __init__(self, *args, subtask=None, boundary_latent_cache_dir: Optional[str] = None, **kwargs):
        # Set before the base init: its text-cache check enumerates the subtask prompts.
        self._subtask_cfg = SubtaskConfig.from_any(subtask)
        self._subtask_table = load_table(self._subtask_cfg.annotation_path) if self._subtask_cfg.enabled else None
        self._cache_view = None
        self._subtask = None
        self._forced_len = None
        super().__init__(*args, **kwargs)
        if not self._subtask_cfg.enabled:
            if boundary_latent_cache_dir:
                raise ValueError("boundary_latent_cache_dir requires subtask.annotation_path")
            return
        if self._split != "train":
            raise ValueError("subtask mode is a training-split feature")
        if self._window_stride != 1:
            raise ValueError("subtask mode assumes window_stride=1")
        self._subtask = SubtaskIndex(
            self._subtask_cfg,
            self._subtask_table,
            self._eps_df["episode_index"].to_numpy(),
            self._ep_valid_end - self._ep_valid_start,
        )
        if self._ep_valid_start.any():
            raise ValueError("subtask mode assumes episodes start at frame 0")
        windows = self.encode_cache_window_table().astype(np.int64)
        self._window_ep, self._window_off = windows[:, 0], windows[:, 1]
        keep = np.fromiter((self._subtask.window_ok(e, o) for e, o in windows.tolist()), bool, len(windows))
        decision = np.fromiter((self._subtask.decision_window(e, o) for e, o in windows.tolist()), bool, len(windows))
        base = np.flatnonzero(keep)
        extra = np.repeat(np.flatnonzero(keep & decision), self._subtask_cfg.decision_weight - 1)
        self._train_index = np.sort(np.concatenate([base, extra])).astype(np.int64)
        self._boundary_rows = self._subtask.boundary_rows(windows, self._ep_valid_end, self._num_frames)
        self._boundary_lookup = {(int(i), int(n)): k for k, (i, _, _, n) in enumerate(self._boundary_rows.tolist())}
        self._boundary_cache = None
        if boundary_latent_cache_dir:
            if self._latent_cache is None:
                raise ValueError("boundary_latent_cache_dir requires latent_cache_dir")
            from openwam.dataloader.utils.encode_cache import LatentCache

            self._boundary_cache = LatentCache(boundary_latent_cache_dir, self.encode_cache_view("boundary"))
        elif self._latent_cache is not None:
            raise ValueError("subtask mode with latent_cache_dir also needs boundary_latent_cache_dir")
        logger.info(
            "Marvin subtask mode: %d/%d windows kept, %d decision windows x%d -> %d samples; "
            "%d boundary clips; %d prompts",
            len(base), len(windows), int((keep & decision).sum()), self._subtask_cfg.decision_weight,
            len(self._train_index), len(self._boundary_rows), len(self.encode_cache_prompts()),
        )

    # ----- subtask sampling -------------------------------------------------

    def __len__(self) -> int:
        if self._subtask is None or self._cache_view == "main":
            return self._n_total
        if self._cache_view == "boundary":
            return len(self._boundary_rows)
        return len(self._train_index)

    def __getitem__(self, idx: int) -> dict:
        if self._subtask is None or self._cache_view == "main":
            return self._safe_get(idx)
        if self._cache_view == "boundary":
            window, _, _, raw_len = (int(v) for v in self._boundary_rows[idx])
            self._forced_len = raw_len
            try:
                sample = self._getitem_impl(window)
            finally:
                self._forced_len = None
            sample["window_index"] = idx  # builders verify rows against the view's own order
            return sample
        return self._safe_get(int(self._train_index[idx]))

    def _plan_window(self, idx: int, ep_local: int, offset: int, valid_end: int) -> dict:
        full = min(self._num_frames, valid_end - offset)
        if self._subtask is None or self._cache_view == "main":
            return {"raw_len": full}
        cfg = self._subtask_cfg
        if self._cache_view == "boundary":
            return {"raw_len": self._forced_len, "hold": cfg.hold}
        delay = random.randint(0, min(cfg.rtc_max_delay_steps, full - 1))
        extra = {"rtc_delay": delay} if cfg.rtc_max_delay_steps else {}
        if random.random() < cfg.main_prompt_prob:
            return {"raw_len": full, "prompt": cfg.main_prompt, "extra": {**extra, "prompt_mode": "main"}}
        seg, raw_len, delay = self._subtask.resolve(ep_local, offset, delay, full)
        if cfg.rtc_max_delay_steps:
            extra["rtc_delay"] = delay
        choices = self._subtask.prompt_choices(seg)
        kinds = [k for k in ("canonical", "nohand", "variants") if choices.get(k)]
        kind = random.choices(kinds, weights=[cfg.prompt_mix[k] for k in kinds])[0]
        plan = {
            "raw_len": raw_len,
            "prompt": random.choice(choices[kind]),
            "hold": cfg.hold,
            "supervise_padded_video": cfg.supervise_padded_video,
            "extra": {**extra, "prompt_mode": "still" if self._subtask.is_still[seg] else "sub"},
        }
        if raw_len < full and self._latent_cache is not None:
            row = self._boundary_lookup[(idx, raw_len)]
            plan["latent"] = lambda: self._boundary_cache.get(row)
        return plan

    # ----- encode-cache protocol (views) --------------------------------------

    def encode_cache_view(self, view: Optional[str]) -> "MarvinDataset":
        """``main``: every window, untruncated (the existing latent cache);
        ``boundary``: the truncated sub-mode clips (rows of the boundary cache)."""
        if view not in ("main", "boundary"):
            raise ValueError(f"unknown encode-cache view {view!r}")
        if view == "boundary" and self._subtask is None:
            raise ValueError("the boundary view requires subtask.annotation_path")
        out = copy.copy(self)
        out._cache_view = view
        return out

    def encode_cache_window_table(self) -> np.ndarray:
        if getattr(self, "_cache_view", None) == "boundary":
            return self._boundary_rows[:, 1:].astype(np.int32)  # (ep_local, offset, raw_len)
        return super().encode_cache_window_table()

    def encode_cache_latent_signature(self) -> dict:
        sig = super().encode_cache_latent_signature()
        if getattr(self, "_cache_view", None) == "boundary":
            cfg = self._subtask_cfg
            sig.update({"view": "boundary", "subtask_sha256": cfg.digest(), "hold": cfg.hold,
                        "num_windows": len(self._boundary_rows)})
        return sig

    def encode_cache_prompts(self) -> list:
        if self._subtask_table is None:
            return super().encode_cache_prompts()
        cfg = self._subtask_cfg
        out = {cfg.main_prompt, cfg.still_prompt}
        for r in self._subtask_table.itertuples():
            if r.exclude or r.type == "STILL":
                continue
            out.update(x for x in (r.prompt, r.prompt_nohand, *(r.variants if r.variants is not None else [])) if x)
        return sorted(out)

    def _resolve_cameras(self, info: dict):
        if info.get("codebase_version") != "v3.0":
            raise ValueError(
                "MarvinDataset requires the converted LeRobot v3.0 dataset; run "
                "scripts/convert_marvin_lerobot_v3.py first"
            )
        contract = info.get("openwam_contract", {})
        if contract.get("rotation_6d_layout") != OPENWAM_ROT6D_LAYOUT:
            raise ValueError(
                "Marvin dataset has the wrong rot6d layout: "
                f"{contract.get('rotation_6d_layout')!r}; expected {OPENWAM_ROT6D_LAYOUT}"
            )
        if contract.get("raw_action_dim") != RAW_ACTION_DIM:
            raise ValueError(
                f"Marvin dataset raw_action_dim must be {RAW_ACTION_DIM}, "
                f"got {contract.get('raw_action_dim')!r}"
            )
        return self.HEAD_CAMERA, self.LEFT_WRIST_CAMERA, self.RIGHT_WRIST_CAMERA

    def _load_stats(self, info: dict) -> Optional[dict]:
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if self._normalize_mode != "min-max":
            raise ValueError("OpenWAM-alpha Marvin fine-tuning requires normalize_mode='min-max'")

        stats_path = self._dataset_dir / "meta" / "normalization_stats.npy"
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Missing {stats_path}; rerun scripts/convert_marvin_lerobot_v3.py to generate training stats"
            )
        payload = np.load(stats_path, allow_pickle=True).item()
        action_stats = _materialize_stats(payload.get("eef", {}), label="eef")
        self._proprio_stats = _materialize_stats(payload.get("eef_state", {}), label="eef_state")
        self.normalization_stats_path = str(stats_path)
        return action_stats

    def _action_20d(self, win) -> np.ndarray:
        source = np.stack(win["action"].values).astype(np.float32)
        return apply_normalization(marvin_raw_vector(source), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win) -> np.ndarray:
        source = np.stack(win["observation.state"].values[:1]).astype(np.float32)
        return apply_normalization(marvin_raw_vector(source), self._proprio_stats, self._normalize_mode)


__all__ = [
    "MarvinDataset",
    "OPENWAM_ROT6D_CONVENTION",
    "OPENWAM_ROT6D_LAYOUT",
    "RAW_ACTION_DIM",
    "ROT6D_RAW_DIMS",
    "marvin_raw_vector",
    "rot6d_rows_to_columns",
]
