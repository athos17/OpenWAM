"""Robot video/action/touch reader with an explicit, publication-bound contract.

The data publisher owns geometry conversion: raw joint/wrist measurements are
never guessed to be commands or canonical EEF components. This reader accepts
verified canonical component columns, maps them to 80 slots, aligns physical
time, and applies separate train-only action/state statistics.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from openwam.dataloader.touchwam_pretrain import LeRobotVideoIntensityDataset, _sha
from openwam.dataloader.utils.normalization import STAT_KEYS, apply_normalization
from openwam.dataloader.utils.unify_action import parse_unify_spec


@lru_cache(maxsize=4)
def _robot_table(path, columns):
    return pq.read_table(path, columns=list(columns))


def _required_text(obj, key):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Robot action contract requires explicit {key}")
    return value


class LeRobotRobotIntensityDataset(LeRobotVideoIntensityDataset):
    """VA/VAT only; never fall back to V/VT when robot actions are unavailable."""

    SOURCE_KIND = "robot"
    EXTRA_CONFIG_FIELDS = ("normalize_mode", "action_mode", "action_contract")

    def __init__(self, dataset_dir, *, normalize_mode="min-max", action_mode="eef",
                 action_contract="meta/action_contract.json", **kwargs):
        # Validate the action contract before video discovery/decoding.
        root = Path(dataset_dir).resolve(strict=True)
        path = root / action_contract
        if not path.is_file():
            raise FileNotFoundError(f"Robot action contract missing: {path}; FTP without verified actions is excluded")
        contract = json.loads(path.read_text())
        verification = contract.get("verification", {})
        if (contract.get("schema") != "touchwam_robot_action_v1"
                or contract.get("source_kind") != "robot"
                or verification.get("status") != "verified"
                or not verification.get("evidence")):
            raise ValueError("Robot actions require a verified touchwam_robot_action_v1 contract with evidence")
        if contract.get("dataset_info_sha256") != _sha(root / "meta/info.json"):
            raise ValueError("Robot action contract does not match dataset info fingerprint")
        self._action_contract = contract
        self._slots = {}
        for stream in ("action", "proprio"):
            spec = contract.get(stream, {})
            for key in ("column", "validity_column", "coordinate_frame", "representation"):
                _required_text(spec, key)
            if spec["representation"] != "openwam_canonical_components_v1":
                raise ValueError("Convert native geometry to verified canonical components before robot training")
            slots = parse_unify_spec(spec.get("slots", []))
            units = spec.get("units")
            if not isinstance(units, list) or len(units) != len(slots) or any(
                    not isinstance(unit, str) or not unit.strip() for unit in units):
                raise ValueError(f"{stream} requires a unit for every canonical component")
            self._slots[stream] = slots
        semantics = contract.get("action_semantics")
        if semantics not in ("recorded_command", "future_state_target"):
            raise ValueError("Declare recorded_command or explicitly verified future_state_target semantics")
        if not isinstance(contract.get("terminal_action_valid"), bool):
            raise ValueError("Explicit terminal_action_valid is required")
        if semantics == "future_state_target" and contract["action"]["column"] != contract["proprio"]["column"]:
            raise ValueError("future_state_target must explicitly reference the same measured state column")
        if semantics == "future_state_target" and not np.array_equal(self._slots["action"], self._slots["proprio"]):
            raise ValueError("future_state_target must preserve the measured state's canonical slot identities")
        norm = contract.get("normalization", {})
        if normalize_mode not in ("min-max", "quantile", "z-score") or norm.get("mode") != normalize_mode:
            raise ValueError("Robot reader normalization must match the published action contract")
        if norm.get("population") != "train" or norm.get("action_key") != action_mode:
            raise ValueError("Robot normalization requires separate train-population action and proprio statistics")
        if not action_mode or norm.get("proprio_key") != action_mode + "_state":
            raise ValueError("Robot proprio normalization key must be action_mode + '_state'")
        stats_path = (root / _required_text(norm, "file")).resolve(strict=True)
        if stats_path.suffix != ".npy" or _sha(stats_path) != norm.get("sha256"):
            raise ValueError("Robot normalization artifact must be a fingerprinted .npy statistics package")
        self._normalize_mode = normalize_mode
        raw_stats = np.load(stats_path, allow_pickle=True).item()
        self._stats = {}
        for stream, key in (("action", action_mode), ("proprio", action_mode + "_state")):
            block = raw_stats.get(key) if isinstance(raw_stats, dict) else None
            if not isinstance(block, dict) or set(STAT_KEYS) - set(block):
                raise ValueError("Robot statistics require separate complete action/proprio blocks for train and deploy")
            values = {name: np.asarray(block[name], dtype=np.float32) for name in STAT_KEYS}
            if any(value.shape != (80,) for value in values.values()):
                raise ValueError("Robot statistics require 80-D vectors")
            self._stats[stream] = values
        for stream, stats in self._stats.items():
            if not all(np.isfinite(v).all() for v in stats.values()):
                raise ValueError(f"Non-finite {stream} normalization statistics")
            lo, hi = ("q01", "q99") if normalize_mode == "quantile" else ("min", "max")
            if (stats[hi] < stats[lo]).any() or (stats["std"] < 0).any():
                raise ValueError(f"Invalid {stream} normalization bounds")
            identity = sorted((set(range(80)) - set(self._slots[stream])) | set([*range(3, 9), *range(37, 43)]))
            fields = dict(min=-1., max=1., q01=-1., q99=1., mean=0., std=1.)
            if any(not np.allclose(stats[key][identity], value) for key, value in fields.items()):
                raise ValueError("Unused slots and canonical rot6d components must retain identity normalization")
        super().__init__(root, **kwargs)
        if contract.get("source_id") != self.source_id:
            raise ValueError("Robot action contract source identity differs from the dataset")
        self.normalization_stats_path = str(stats_path)
        self.training_modality = "video_action_tactile" if self.touch else "video_action"
        self.action_offsets_s = (np.arange(self.num_frames - 1, dtype=np.float64)
                                 + (1 if semantics == "future_state_target" else 0)) / self.sampling_fps
        self._columns = tuple(sorted({"timestamp", "episode_index"} | {
            contract[stream][key] for stream in ("action", "proprio")
            for key in ("column", "validity_column")}))
        for stream in ("action", "proprio"):
            spec = contract[stream]
            feature = self.info.get("features", {}).get(spec["column"], {})
            if feature.get("shape") != [len(self._slots[stream])]:
                raise ValueError(f"{stream} feature shape disagrees with its canonical slot map")
        for chunk, file in self.episodes[["data/chunk_index", "data/file_index"]].drop_duplicates().itertuples(index=False):
            shard = root / self.info["data_path"].format(chunk_index=chunk, file_index=file)
            if set(self._columns) - set(pq.read_schema(shard).names):
                raise ValueError(f"Robot publication lacks required action/state/mask columns: {shard}")
        self.training_data_contract.update(
            schema="touchwam_lerobot_robot_v1", source_kind="robot",
            training_modality=self.training_modality, action_contract_sha256=_sha(path),
            action_contract=contract, normalization_sha256=norm["sha256"],
            action_times_s=self.action_offsets_s.tolist(),
            action_sampling="causal_hold" if semantics == "recorded_command" else "nearest_future_observation",
        )

    def _read_components(self, table, stream, row_valid=None):
        spec, slots = self._action_contract[stream], self._slots[stream]
        # Null values are never converted to observed zeros.
        rows = table[spec["column"]].to_pylist()
        values = np.asarray([row if row is not None else [np.nan] * len(slots) for row in rows], dtype=np.float32)
        raw_mask = table[spec["validity_column"]].to_pylist()
        import pyarrow as pa
        dtype = table[spec["validity_column"]].type
        vector_mask = pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype)
        masks = np.asarray([
            ([False] * len(slots) if row is None else [False if item is None else item for item in row])
            if vector_mask else (False if row is None else row) for row in raw_mask])
        if masks.dtype != np.bool_:
            raise ValueError(f"{stream} validity must be boolean and non-null within vector masks")
        if masks.shape == (len(table),):
            masks = np.broadcast_to(masks[:, None], values.shape)
        if values.shape != (len(table), len(slots)) or masks.shape != values.shape:
            raise ValueError(f"{stream} values/masks do not match the published component width")
        if row_valid is not None:
            masks = masks & np.asarray(row_valid, dtype=bool)[:, None]
        if not np.isfinite(values[masks]).all():
            raise ValueError(f"Observed {stream} components contain NaN/Inf")
        unified = np.zeros((len(table), 80), dtype=np.float32)
        valid = np.zeros_like(unified, dtype=bool)
        unified[:, slots] = np.where(masks, values, 0)
        valid[:, slots] = masks
        return unified, valid

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ei = int(np.searchsorted(self.cumulative, index, side="right") - 1)
        row = self.episodes.iloc[ei]
        length, offset = int(row.length), sample["episode_start_frame"]
        path = self.root / self.info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        table = _robot_table(str(path), self._columns).slice(int(row["_data_offset"]), length)
        if len(table) != length or not np.all(np.asarray(table["episode_index"]) == int(row.episode_index)):
            raise ValueError("Robot action shard crosses or truncates episode boundaries")
        # Parent validates the physical clock; exact frame indices avoid float32 timestamp drift.
        relative = self.action_offsets_s * self.fps
        if self._action_contract["action_semantics"] == "recorded_command":
            indices = offset + np.floor(relative + 1e-8).astype(np.int64)
        else:
            indices = offset + np.floor(relative + .5).astype(np.int64)
        in_episode = indices < length
        if self._action_contract["action_semantics"] == "future_state_target":
            in_episode &= indices > offset
        elif not self._action_contract["terminal_action_valid"]:
            in_episode &= indices < length - 1
        selected = table.take(np.clip(indices, 0, length - 1))
        actions, valid = self._read_components(selected, "action", in_episode)
        proprio, proprio_valid = self._read_components(table.slice(offset, 1), "proprio")
        if not valid.any() or not proprio_valid.any():
            raise ValueError("Robot window lacks observed action targets or current proprio; V/VT fallback is forbidden")
        actions = apply_normalization(actions, self._stats["action"], self._normalize_mode)
        proprio = apply_normalization(proprio, self._stats["proprio"], self._normalize_mode)
        actions[~valid] = 0
        proprio[~proprio_valid] = 0
        sample.update(action=torch.from_numpy(actions), action_mask=torch.from_numpy(valid),
                      proprio=torch.from_numpy(proprio), proprio_mask=torch.from_numpy(proprio_valid),
                      source_kind="robot", action_semantics=self._action_contract["action_semantics"],
                      action_time_offsets_s=torch.from_numpy(self.action_offsets_s.copy()).float(),
                      action_label_time_offsets_s=torch.from_numpy((indices - offset) / self.fps).float())
        return sample
