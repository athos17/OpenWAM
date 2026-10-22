"""AgiBotWorld-Beta reader for LeRobot v3 buckets.

The source exposes bimanual terminal-arm poses in the robot-base frame as
``[L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6)]``.  Grippered buckets insert
``[L_grip, R_grip]`` to form the canonical 20-D EEF layout; dexterous-hand
buckets instead assemble a 30-D pose-and-finger vector before scattering it
into the shared 80-D action space.  The pose endpoint is the rigid arm-side
attachment (flange-side), not a universal task TCP.

Action grippers use ``0=open, 1=closed`` and are inverted at the reader
boundary.  State grippers are converted from closing-actuator position to the
same ``0=closed, 1=open`` aperture convention.  Mobile-base x/yaw values map
to unified slots 68/70; the unsupported lateral slot remains masked.

Actions are already next-state relabeled, so rows are consumed without an
additional temporal shift.  The clamped terminal action of an episode is not
supervised.  Optional normalization uses separate action and proprio stats,
with rot6d dimensions pinned to identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.robocoin import GRIP_EXCLUDED_DIM_MASK, _build_dex_unify_map
from openwam.dataloader.utils.lerobotv3 import DataContractError
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_EEF20,
    apply_normalization,
    find_dataset_stats,
    materialize_eef_stats,
    pin_rot6d_identity,
)

logger = logging.getLogger(__name__)

# Fixed camera columns shared by every supported bucket.
HEAD_CAMERA = "observation.images.head"
LEFT_WRIST_CAMERA = "observation.images.hand_left"
RIGHT_WRIST_CAMERA = "observation.images.hand_right"


_DEX_PER_HAND = 6

# Native widths before optional mobile-base values are appended.
_EEF_RAW_DIM = 20
_DEX_RAW_DIM = 18 + 2 * _DEX_PER_HAND


# Differential drive publishes [forward, lateral, yaw]; lateral is unsupported.
_MOVE_SRC_DIMS = (0, 2)
_MOVE_DIM = len(_MOVE_SRC_DIMS)
_MOVE_SLOTS = (68, 70)  # shared 80-D forward/yaw slots
_MOVE_EPS = 1e-6


_GRIPPER_STATE_OPEN_POSITION_M = 0.035
_GRIPPER_STATE_CLOSED_POSITION_M = 0.125
# Persist the physical direction and state calibration with generated stats.
_GRIPPER_CONTRACT = {
    "schema_version": 1,
    "raw_action_semantics": "0_open_1_closed_command",
    "action_transform": "1-x",
    "raw_state_semantics": "closing_actuator_position_m",
    "state_transform": "clip((closed_position_m-x)/(closed_position_m-open_position_m),0,1)",
    "open_endpoint_m": _GRIPPER_STATE_OPEN_POSITION_M,
    "closed_endpoint_m": _GRIPPER_STATE_CLOSED_POSITION_M,
    "output_semantics": "0_closed_1_open_aperture_fraction",
}


# ``info.features`` declares both signal families, so bucket flavor is explicit.
_DEX_BUCKET_IDS = frozenset(
    {
        "475",
        "536",
        "549",
        "554",
        "577",
        "578",
        "595",
        "608",
        "620",
        "622",
        "660",
        "679",
        "705",
        "710",
        "727",
        "730",
        "731",
        "749",
        "753",
    }
)


_GRIP_COLS = (
    "task_index",
    "action.ee_base",
    "observation.state.ee_base",
    "action.gripper",
    "observation.state.gripper",
)
_DEX_COLS = (
    "task_index",
    "action.ee_base",
    "observation.state.ee_base",
    "action.dex",
    "observation.state.dex",
)

_POSE_ONLY_COLS = (
    "task_index",
    "action.ee_base",
    "observation.state.ee_base",
)

_MOVE_COLS = ("action.robot_velocity", "observation.state.robot_velocity")


# Dex raw layout: L pose9 + L fingers6 + R pose9 + R fingers6.
_ROT6D_DIMS_DEX30 = (3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23)

_STAT_FIELDS = ("min", "max", "mean", "std", "q01", "q99")


_UNIFIED_STATS_FILENAME = "stats_g2a.json"


_BUCKET_STATS_FILENAME = "stats.json"


def _bucket_base_motion_flags(dataset_dir) -> tuple[bool, bool]:
    """Return independent ``(action_moves, state_moves)`` flags for a bucket.

    Each flag is read from its OWN ``robot_velocity`` block in the bucket's
    shipped ``meta/stats.json`` and is true iff an exact min/max exceeds
    ``_MOVE_EPS``.  Keeping the streams separate matters for command-only
    mobile buckets: action slots 68/70 remain supervised while all-zero state
    slots are absent from the proprio mask.

    Missing or unreadable bucket stats fail immediately: silently treating a
    mobile bucket as stationary would remove real supervision.
    """

    p = Path(dataset_dir) / "meta" / _BUCKET_STATS_FILENAME
    try:
        with p.open(encoding="utf-8") as stream:
            st = json.load(stream)
    except (OSError, ValueError) as exc:
        raise DataContractError(f"Cannot load AgiBotWorld movement statistics from {p}: {exc}") from exc
    if not isinstance(st, dict):
        raise DataContractError(f"AgiBotWorld movement statistics {p} must contain a mapping.")

    flags = []
    for key in ("action.robot_velocity", "observation.state.robot_velocity"):
        blk = st.get(key)
        if not blk:
            flags.append(False)
            continue
        if not isinstance(blk, dict):
            raise DataContractError(f"{p}: {key} must contain movement bounds.")
        bounds = [np.asarray(blk.get(field), dtype=np.float64) for field in ("min", "max")]
        if any(values.shape != (3,) or not np.isfinite(values).all() for values in bounds):
            raise DataContractError(f"{p}: {key} min/max must contain three finite values.")
        if np.any(bounds[0] > bounds[1]):
            raise DataContractError(f"{p}: {key} min exceeds max.")
        flags.append(bool(max(np.abs(values).max() for values in bounds) > _MOVE_EPS))
    return flags[0], flags[1]


def _action_gripper_to_open_convention(grip: np.ndarray) -> np.ndarray:
    """Convert AgiBotWorld action gripper ``0=open,1=closed`` to ``0=closed,1=open``."""
    return (1.0 - np.asarray(grip, dtype=np.float32)).astype(np.float32, copy=False)


def _state_gripper_to_open_convention(grip: np.ndarray) -> np.ndarray:
    """Convert closing-actuator position to a clipped ``0=closed,1=open`` fraction."""
    raw = np.asarray(grip, dtype=np.float32)
    span = _GRIPPER_STATE_CLOSED_POSITION_M - _GRIPPER_STATE_OPEN_POSITION_M
    return np.clip((_GRIPPER_STATE_CLOSED_POSITION_M - raw) / span, 0.0, 1.0).astype(np.float32, copy=False)


def _eef18_to_eef20(ee18: np.ndarray, grip2: np.ndarray) -> np.ndarray:
    """Insert grippers into an already-rot6d 18-D pose → canonical 20-D EEF.

    Input layout::

        ee18:  [L_pos(3), L_rot6d(6), R_pos(3), R_rot6d(6)]   (already rot6d)
        grip2: [L_grip, R_grip]

    Output layout::

        [L_pos(3), L_rot6d(6), L_grip(1), R_pos(3), R_rot6d(6), R_grip(1)]

    Pure concatenation — NO euler→rot6d conversion (that is what distinguishes
    AgiBotWorld from RoboCOIN's :func:`eef14_to_eef20`).
    """

    l_pose9 = ee18[:, 0:9]
    r_pose9 = ee18[:, 9:18]
    l_grip = grip2[:, 0:1]
    r_grip = grip2[:, 1:2]
    return np.concatenate([l_pose9, l_grip, r_pose9, r_grip], axis=-1).astype(np.float32)


class AgiBotWorldDataset(LeRobotV3Reader):
    """Single-dataset reader for one AgiBotWorld-Beta task bucket (LeRobot v3).

    Bimanual EEF with real action/state supervision. All window / offset / video /
    prompt machinery is inherited from :class:`LeRobotV3Reader`; only the
    AgiBotWorld-specific bits are overridden below.
    """

    STRICT_BUCKET_LOADING = True
    DATASET_NAME = "AgiBotWorld"
    HEAD_CAMERA = HEAD_CAMERA
    LEFT_WRIST_CAMERA = LEFT_WRIST_CAMERA
    RIGHT_WRIST_CAMERA = RIGHT_WRIST_CAMERA

    PROMPT_FILE_REQUIRED = True

    DEFAULT_NORMALIZE_MODE = None

    WRIST_DECODE_TOLERATED = (Exception,)
    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("enable_action_supervision",)

    def __init__(
        self,
        dataset_dir,
        *,
        unify_action: bool = False,
        unify_action_map=None,
        enable_action_supervision: bool = True,
        **kwargs,
    ):
        """Detect the bucket flavor + base motion and wire the finger / move scatter.

        Under ``unify_action=True`` the reader builds the per-bucket map in code
        (OVERRIDING any config ``unify_action_map`` — the map varies per bucket, so a
        single yaml value cannot serve all). The raw width / map depend on:

        * Flavor — grippered raw ``EEF(20)`` (scatter ``0-9`` / ``34-43``) vs
          dexterous-hand raw ``[pose(18), fingers(12)]`` = 30 (RoboCOIN's dex scatter:
          pose→0-8/34-42, fingers→10-15/44-49).
        * Base motion — a 2-D ``robot_velocity`` x/yaw tail (→ move slots ``68`` / ``70``;
          y≡0 dropped) is appended when action OR state moves.  Independent raw
          dimension masks keep a command-only action valid while masking an absent
          all-zero state (and vice versa).

        Under ``unify_action=False`` no movement is emitted: grippered → 20-D EEF,
        dex → 20-D pose with grip slots masked.

        ``enable_action_supervision=False`` keeps the numeric values available but
        masks both action and proprio, and selects video-only training.
        """

        self._is_dex = Path(dataset_dir).name in _DEX_BUCKET_IDS
        self._enable_action_supervision = bool(enable_action_supervision)

        action_moves, state_moves = _bucket_base_motion_flags(dataset_dir) if unify_action else (False, False)
        self._action_has_move = bool(action_moves)
        self._proprio_has_move = bool(state_moves)
        self._has_move = self._action_has_move or self._proprio_has_move

        self.ACTION_DIM_MASK = None
        self.PROPRIO_DIM_MASK = None
        if unify_action:
            move_dim = _MOVE_DIM if self._has_move else 0
            move_slots = list(_MOVE_SLOTS) if self._has_move else []
            if self._is_dex:
                self.ACTION_DIM = _DEX_RAW_DIM + move_dim
                unify_action_map = _build_dex_unify_map(_DEX_PER_HAND, _DEX_PER_HAND) + move_slots
            else:
                self.ACTION_DIM = _EEF_RAW_DIM + move_dim
                unify_action_map = ["0-9", "34-43"] + move_slots
            if self._has_move:
                action_mask = np.ones(self.ACTION_DIM, dtype=bool)
                proprio_mask = np.ones(self.ACTION_DIM, dtype=bool)
                action_mask[-_MOVE_DIM:] = self._action_has_move
                proprio_mask[-_MOVE_DIM:] = self._proprio_has_move
                self.ACTION_DIM_MASK = action_mask
                self.PROPRIO_DIM_MASK = proprio_mask
        super().__init__(dataset_dir, unify_action=unify_action, unify_action_map=unify_action_map, **kwargs)

    def _load_excluded_episode_indices(self) -> set[int]:
        """The exported episodes already contain exactly the retained frames."""
        return set()

    def _resolve_cameras(self, info: dict):
        """Return the fixed head/left/right cameras, record robot_type, and pick
        the parquet columns + supervision mask for this bucket flavor.

        ``self._unify`` is already set by the base ``__init__`` before this hook
        runs, and the base reads the action/proprio masks after it.  The move-tail
        masks were set independently in ``__init__``; this hook only supplies the
        dex + unify-OFF grip mask (mirrors RoboCOIN).
        """

        self._robot_type = info.get("robot_type", "unknown")
        move_cols = _MOVE_COLS if self._has_move else ()
        if self._is_dex:
            if self._unify:
                self.NEEDED_COLS = _DEX_COLS + move_cols
            else:
                self.NEEDED_COLS = _POSE_ONLY_COLS
                self.ACTION_DIM_MASK = GRIP_EXCLUDED_DIM_MASK
                self.PROPRIO_DIM_MASK = GRIP_EXCLUDED_DIM_MASK
        else:
            self.NEEDED_COLS = (_GRIP_COLS + move_cols) if self._unify else _GRIP_COLS
        if not self._enable_action_supervision:
            # Set raw-width masks before the base class builds unified masks.
            # This covers both gripper/dex layouts and optional base motion.
            self.ACTION_DIM_MASK = np.zeros(self._raw_action_dim, dtype=bool)
            self.PROPRIO_DIM_MASK = np.zeros(self._raw_action_dim, dtype=bool)
        return self.HEAD_CAMERA, self.LEFT_WRIST_CAMERA, self.RIGHT_WRIST_CAMERA

    def _load_stats(self, info: dict):
        """Read the shipped family-root stats and assemble each signal layout.

        Action/state and mobile-base distributions remain separate. No frame
        selection or statistics generation is performed by the reader.
        """

        self._action_norm_stats = None
        self._proprio_norm_stats = None
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = find_dataset_stats(self._dataset_dir, _UNIFIED_STATS_FILENAME, family_root=self._stats_root)
        with open(stats_path, encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise DataContractError(f"{stats_path} must contain a statistics mapping.")
        if raw.get("gripper_contract") != _GRIPPER_CONTRACT:
            raise DataContractError(
                f"AgiBotWorld stats {stats_path} use gripper_contract="
                f"{raw.get('gripper_contract')!r}; expected {_GRIPPER_CONTRACT!r}. "
                "Use the stats shipped with the cleaned AgiBotWorld export."
            )

        def _mat(key: str, dim: int):
            if key not in raw:
                raise KeyError(f"AgiBotWorld bucket {self._dataset_id}: '{key}' missing from {stats_path}.")
            block = raw[key]
            if not isinstance(block, dict):
                raise DataContractError(f"{stats_path}: {key} must contain a statistics mapping.")
            for field in _STAT_FIELDS:
                if field in block:
                    values = np.asarray(block[field], dtype=np.float32)
                    if values.shape != (dim,) or not np.isfinite(values).all():
                        raise DataContractError(f"{stats_path}: {key}.{field} must contain {dim} finite values.")
            stats = materialize_eef_stats(
                block, self._normalize_mode, dim=dim, strict_minmax=False, source_hint=f"{stats_path}: {key}"
            )
            if np.any(stats["std"] < 0) or any(
                np.any(stats[low] > stats[high]) for low, high in (("min", "max"), ("q01", "q99"))
            ):
                raise DataContractError(f"{stats_path}: {key} has invalid normalization bounds or standard deviation.")
            return stats

        def _build(prefix: str):
            """Assemble the width-matched stats block for one stream ('action' or
            'observation.state')."""

            ee = _mat(f"{prefix}.ee_base", 18)
            combined = {}
            if self._is_dex and self._unify:
                fing = _mat(f"{prefix}.dex", 2 * _DEX_PER_HAND)
                for k in _STAT_FIELDS:
                    e, h = ee[k], fing[k]
                    combined[k] = np.concatenate(
                        [e[0:9], h[0:_DEX_PER_HAND], e[9:18], h[_DEX_PER_HAND : 2 * _DEX_PER_HAND]]
                    ).astype(np.float32)
                pin_rot6d_identity(combined, _ROT6D_DIMS_DEX30)
            else:
                g = _mat(f"{prefix}.gripper", 2)
                for k in _STAT_FIELDS:
                    e, gg = ee[k], g[k]
                    combined[k] = np.concatenate([e[0:9], gg[0:1], e[9:18], gg[1:2]]).astype(np.float32)
                pin_rot6d_identity(combined, ROT6D_DIMS_EEF20)
            if self._has_move:
                stream_moves = self._action_has_move if prefix == "action" else self._proprio_has_move
                if stream_moves:
                    vel = _mat(f"{prefix}.robot_velocity", 3)
                else:
                    vel = {
                        "mean": np.zeros(3, dtype=np.float32),
                        "std": np.ones(3, dtype=np.float32),
                        "min": -np.ones(3, dtype=np.float32),
                        "max": np.ones(3, dtype=np.float32),
                        "q01": -np.ones(3, dtype=np.float32),
                        "q99": np.ones(3, dtype=np.float32),
                    }
                src = list(_MOVE_SRC_DIMS)
                for k in _STAT_FIELDS:
                    combined[k] = np.concatenate([combined[k], vel[k][src]]).astype(np.float32)
            return combined

        self._action_norm_stats = _build("action")
        self._proprio_norm_stats = _build("observation.state")

        return self._action_norm_stats

    def _train_min_window_len(self) -> int:
        """Require two rows so every train window has a real next-state target."""
        return 2

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        """Mask the clamped final action in an episode-truncated tail window.

        The source is already next-state relabeled: its final episode row has no
        successor and therefore contains a fabricated/clamped target. A full
        ``num_frames`` window never emits that last row because ``T_action`` is
        one shorter; a truncated window does, so drop exactly its final row.
        """

        if actual_raw_len >= self._num_frames:
            return actual_raw_len
        return max(0, actual_raw_len - 1)

    def _normalize_array(self, arr: np.ndarray, stats) -> np.ndarray:
        """Apply per-stream normalization with the given stats. No-op when null."""
        return apply_normalization(arr, stats, self._normalize_mode)

    def _grip_or_zeros(self, win, col: str, n: int) -> np.ndarray:
        """Stack ``n`` rows of a gripper column, or zeros for dex-hand buckets.

        Dex-hand buckets under unify OFF read pose-only columns; the (n, 2) zero
        fill flows through :func:`_eef18_to_eef20` into the grip slots, which are
        masked out of supervision by ``GRIP_EXCLUDED_DIM_MASK``.
        """

        if not self._is_dex:
            grip = np.stack(win[col].values[:n]).astype(np.float32)
            if col == "action.gripper":
                grip = _action_gripper_to_open_convention(grip)
            else:
                grip = _state_gripper_to_open_convention(grip)
            return grip
        return np.zeros((n, 2), dtype=np.float32)

    def _dex_pose_fingers(self, ee18: np.ndarray, dex12: np.ndarray) -> np.ndarray:
        """Assemble a 30-D dex-hand pose+fingers vector (UN-normalized).

        ``[L_pos(3), L_rot6d(6), L_fingers(6), R_pos(3), R_rot6d(6), R_fingers(6)]``
        — pose from ``ee_base`` (already rot6d), fingers from ``dex`` (``[:6]`` left,
        ``[6:12]`` right). The base unify scatter places these into the 80-D pose +
        hand slots. Normalization is applied by the caller after the move tail is
        appended (so the whole raw vector normalizes in one pass).
        """

        l_pose9 = ee18[:, 0:9]
        r_pose9 = ee18[:, 9:18]
        l_fing = dex12[:, 0:_DEX_PER_HAND]
        r_fing = dex12[:, _DEX_PER_HAND : 2 * _DEX_PER_HAND]
        return np.concatenate([l_pose9, l_fing, r_pose9, r_fing], axis=-1).astype(np.float32)

    def _append_move(self, raw: np.ndarray, win, col: str, n: int) -> np.ndarray:
        """Append the base-movement tail for mobile buckets: robot_velocity's x/yaw
        (``_MOVE_SRC_DIMS``) only — the y column is ≡0 and is dropped."""

        if not self._has_move:
            return raw
        stream_moves = self._action_has_move if col.startswith("action.") else self._proprio_has_move
        if stream_moves:
            vel = np.stack(win[col].values[:n]).astype(np.float32)[:, _MOVE_SRC_DIMS]
        else:
            vel = np.zeros((n, _MOVE_DIM), dtype=np.float32)
        return np.concatenate([raw, vel], axis=-1)

    def _action_20d(self, win) -> np.ndarray:
        ee = np.stack(win["action.ee_base"].values).astype(np.float32)
        n = len(ee)
        if self._is_dex and self._unify:
            dex = np.stack(win["action.dex"].values).astype(np.float32)
            raw = self._dex_pose_fingers(ee, dex)
        else:
            grip = self._grip_or_zeros(win, "action.gripper", n)
            raw = _eef18_to_eef20(ee, grip)
        raw = self._append_move(raw, win, "action.robot_velocity", n)
        return self._normalize_array(raw, self._action_norm_stats)

    def _proprio_20d(self, win) -> np.ndarray:
        ee = np.stack(win["observation.state.ee_base"].values[:1]).astype(np.float32)
        if self._is_dex and self._unify:
            dex = np.stack(win["observation.state.dex"].values[:1]).astype(np.float32)
            raw = self._dex_pose_fingers(ee, dex)
        else:
            grip = self._grip_or_zeros(win, "observation.state.gripper", 1)
            raw = _eef18_to_eef20(ee, grip)
        raw = self._append_move(raw, win, "observation.state.robot_velocity", 1)
        return self._normalize_array(raw, self._proprio_norm_stats)

    def _getitem_impl(self, idx: int) -> dict:
        sample = super()._getitem_impl(idx)
        if not self._enable_action_supervision:
            # Explicit routing also removes action/proprio from video conditioning;
            # loss masks alone would still allow gradients through those streams.
            sample["training_modality"] = "video_only"
        return sample

    @property
    def robot_type(self):
        return self._robot_type

    @property
    def training_modality(self):
        return "video_action" if self._enable_action_supervision else "video_only"

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiAgiBotWorldDataset


class MultiAgiBotWorldDataset(MultiLeRobotV3Reader):
    """Aggregate of N AgiBotWorld-Beta per-task buckets."""

    def __init__(self, buckets: List[AgiBotWorldDataset]):
        super().__init__(buckets)
        modalities = {getattr(b, "training_modality", "video_action") for b in buckets}
        if len(modalities) != 1:
            raise DataContractError("AgiBot source buckets must share one training modality")
        self.training_modality = modalities.pop()
        # Converted datasets can package one shared 80-D action/state artifact.
        # Samples are already normalized; this path is only for checkpoint export.
        self.normalization_stats_path = None
        roots = {b._dataset_dir.parent.resolve() for b in self._buckets}
        modes = {b._normalize_mode for b in self._buckets}
        if len(roots) == 1 and all(b._unify for b in self._buckets) and modes.isdisjoint({None, "none", "null"}):
            root = next(iter(roots))
            manifest_path = root / "meta" / "normalization_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                if (manifest.get("schema_version") != 2 or manifest.get("action_key") != "eef"
                        or manifest.get("state_key") != "eef_state"):
                    raise DataContractError("Packaged AgiBot normalizer requires schema 2 with eef/eef_state")
                if modes != {manifest["normalize_mode"]} or manifest["unified_dim"] != 80:
                    raise DataContractError("Packaged AgiBot normalizer mode/dimension disagrees with reader")
                for filename, hash_key in (("stats_g2a.json", "stats_g2a_sha256"),
                                           ("normalization_stats.npy", "normalization_stats_sha256")):
                    path = root / "meta" / filename
                    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[hash_key]:
                        raise DataContractError(f"Packaged AgiBot normalizer changed: {path}")
                artifact = np.load(root / "meta" / "normalization_stats.npy", allow_pickle=True).item()
                if (not isinstance(artifact, dict) or "proprio_eef" in artifact
                        or not all(isinstance(artifact.get(key), dict) for key in ("eef", "eef_state"))):
                    raise DataContractError("Packaged AgiBot normalizer must contain eef/eef_state stats dicts")
                self.normalization_stats_path = str(root / "meta" / "normalization_stats.npy")
        n_dex = sum(1 for b in self._buckets if getattr(b, "_is_dex", False))
        logger.info(
            "MultiAgiBotWorldDataset: %d datasets (%d dex-hand, %d grippered), %d windows",
            len(self._buckets),
            n_dex,
            len(self._buckets) - n_dex,
            len(self),
        )

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return AgiBotWorldDataset.from_config(config, split)


__all__ = ["AgiBotWorldDataset", "MultiAgiBotWorldDataset"]
