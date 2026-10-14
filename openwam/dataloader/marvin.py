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

from typing import ClassVar, Optional

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.normalization import STAT_KEYS, apply_normalization

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

    CONFIG_KEYS: ClassVar[tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS

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
