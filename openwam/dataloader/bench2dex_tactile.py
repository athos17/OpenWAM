"""Bench2Dex Alpha manifest reader with causal tactile history and future labels.

Reads existing numeric sidecars and HDF5 recordings. Dataset conversion and robot
kinematics are intentionally independent of model training.
"""

import bisect
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.unify_action import map_to_unify, parse_unify_spec
from openwam.tactile.sharpa import SCHEMA, SITE_NAMES, read_tactile_frames

CAMERAS = ("cam_stereo_left", "cam_wrist_left", "cam_wrist_right")
UNIFY_MAP = ["0-8", "10-29", "34-42", "44-63"]
DST = parse_unify_spec(UNIFY_MAP)


def _decode_rgb(value):
    encoded = np.frombuffer(value, dtype=np.uint8) if isinstance(value, bytes) else np.asarray(value, dtype=np.uint8)
    image = cv2.imdecode(encoded.reshape(-1), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode Bench2Dex camera frame")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def cache_key(source):
    return hashlib.sha256(source.encode()).hexdigest()[:24]


class Bench2DexTactileDataset(BaseDataset):
    history = 4
    map_size = 32
    future_frames = 8
    future_stride = 2
    cache_root = None

    def __init__(
        self,
        manifest_path,
        *,
        split="train",
        source_root=None,
        num_frames=33,
        video_stride=4,
        window_stride=1,
        height=384,
        width=320,
        normalize_mode="min-max",
        task_name=None,
    ):
        self.root = Path(manifest_path).resolve().parent
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest["schema"] != "bench2dex_alpha_v1" or num_frames != 33 or video_stride != 4:
            raise ValueError("Expected Alpha v1: 33 raw frames, stride 4, 32 actions")
        if split not in ("train", "val") or window_stride < 1:
            raise ValueError("Invalid split or window stride")
        if normalize_mode not in ("min-max", "quantile", "z-score", "none", None):
            raise ValueError(f"Invalid normalization: {normalize_mode}")
        self.source_root = Path(source_root or manifest["source_root"])
        self.episodes = [
            ep
            for ep in manifest["episodes"]
            if ep["split"] == split and (task_name is None or ep.get("task_name") == task_name)
        ]
        self.action_dim = 80
        self.normalization_stats_path = str(self.root / manifest["stats_path"])
        self.stats = np.load(self.normalization_stats_path, allow_pickle=True).item()
        self.normalize_mode = normalize_mode
        self.height, self.width = height, width
        self.windows, self.offsets, self.numeric = [], [0], []
        for idx, ep in enumerate(self.episodes):
            path = self.source_root / ep["source"]
            if path.stat().st_size != ep["bytes"]:
                raise ValueError(f"Source changed since conversion: {path}")
            with np.load(self.root / ep["numeric"]) as data:
                self.numeric.append((data["action"], data["state"]))
            for a, b in ep["segments"]:
                count = max(0, (b - a - num_frames) // window_stride + 1)
                if count:
                    self.windows.append((idx, a, window_stride))
                    self.offsets.append(self.offsets[-1] + count)
        if not self.windows:
            raise ValueError(f"No {split} windows")

    @classmethod
    def from_config(cls, config, split="train"):
        mapping = list(get_cfg(config, "unify_action_map", UNIFY_MAP))
        if not np.array_equal(parse_unify_spec(mapping), DST) or get_cfg(config, "action_mode", "eef") != "eef":
            raise ValueError("Config differs from converted Alpha action layout")
        tactile = {
            key: get_cfg(config, f"tactile_{key}", default)
            for key, default in (
                ("history", 4),
                ("map_size", 32),
                ("future_frames", 8),
                ("future_stride", 2),
            )
        }
        if any(type(value) is not int or value < 1 for value in tactile.values()) or tactile["map_size"] < 8:
            raise ValueError("Invalid tactile dimensions or horizon")
        if tactile["future_frames"] * tactile["future_stride"] > 32:
            raise ValueError("Future tactile horizon must lie inside the 32-step Alpha action window")
        if get_cfg(config, "tactile_fps", 20) != 20:
            raise ValueError("Bench2Dex Alpha tactile rows are at 20 Hz")
        dataset = cls(
            get_cfg(config, "manifest_path"),
            split=split,
            **{
                key: get_cfg(config, key, default)
                for key, default in (
                    ("source_root", None),
                    ("num_frames", 33),
                    ("video_stride", 4),
                    ("window_stride", 1),
                    ("height", 384),
                    ("width", 320),
                    ("normalize_mode", "min-max"),
                    ("task_name", None),
                )
            },
        )
        for key, value in tactile.items():
            setattr(dataset, key, value)
        root = get_cfg(config, "tactile_cache_root", None)
        dataset.cache_root = Path(root) if root else None
        if dataset.cache_root is not None:
            dataset._check_cache(get_cfg(config, "manifest_path"))
        return dataset

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        segment = bisect.bisect_right(self.offsets, idx) - 1
        episode_idx, first, stride = self.windows[segment]
        start = first + (idx - self.offsets[segment]) * stride
        ep = self.episodes[episode_idx]
        actions, states = self.numeric[episode_idx]
        action = apply_normalization(actions[start : start + 32], self.stats["eef"], self.normalize_mode)
        state = apply_normalization(states[start : start + 1], self.stats["eef_state"], self.normalize_mode)
        action, action_mask = map_to_unify(action, DST, unify_dim=80)
        state, state_mask = map_to_unify(state, DST, unify_dim=80)
        action_mask = np.broadcast_to(action_mask, action.shape).copy()
        state_mask = np.broadcast_to(state_mask, state.shape).copy()
        video = []
        with h5py.File(self.source_root / ep["source"], "r") as h5:
            for t in range(start, start + 33, 4):
                frames = {cam: _decode_rgb(h5[f"cameras/{cam}/rgb"][t]) for cam in CAMERAS}
                video.append(assemble_multiview_layout(frames, CAMERAS, self.height, self.width))
            extras = self.observation_extras(h5, ep, start, first)
        return {
            **extras,
            "video": video,
            "vace_video": None,
            "first_frame_image": [video[0]],
            "video_mask": torch.ones(9, dtype=torch.bool),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "proprio": torch.from_numpy(state),
            "proprio_mask": torch.from_numpy(state_mask),
            "prompt": ep["prompt"],
        }

    def _check_cache(self, manifest_path):
        meta = json.loads((self.cache_root / "manifest.json").read_text())
        digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        if (
            meta["schema"] != SCHEMA
            or meta["map_size"] != self.map_size
            or meta["source_manifest_sha256"] != digest
            or meta["site_names"] != list(SITE_NAMES)
        ):
            raise ValueError("Tactile cache contract/source manifest mismatch")
        for ep in self.episodes:
            if ep["source"] not in meta["episodes"]:
                raise ValueError(f"Tactile cache missing {ep['source']}; finish conversion first")
            stamp = (self.source_root / ep["source"]).stat()
            entry = meta["episodes"][ep["source"]]
            if stamp.st_size != entry["source_bytes"] or stamp.st_mtime_ns != entry["source_mtime_ns"]:
                raise ValueError(f"Source changed since tactile caching: {ep['source']}")

    def _read_tactile(self, h5, episode, start, stop):
        if self.cache_root is None:
            return read_tactile_frames(h5, start, stop, self.map_size)
        key = cache_key(episode["source"])
        maps = np.load(self.cache_root / f"{key}.maps.npy", mmap_mode="r")[start:stop]
        valid = np.load(self.cache_root / f"{key}.valid.npy", mmap_mode="r")[start:stop]
        if maps.shape != (stop - start, 10, 2, self.map_size, self.map_size) or valid.shape != (stop - start, 10):
            raise ValueError("Truncated or incompatible tactile cache")
        return maps, valid

    def observation_extras(self, h5, episode, start, segment_start):
        offsets = np.arange(1, self.future_frames + 1, dtype=np.int64) * self.future_stride
        end = start + int(offsets[-1]) + 1
        # Never cross invalid frames, timestamp gaps, homing or episode boundaries.
        if not any(a == segment_start and a <= start and end <= b for a, b in episode["segments"]):
            raise ValueError("Future tactile labels cross the current valid trajectory segment")
        first = max(segment_start, start - self.history + 1)
        history, valid = self._read_tactile(h5, episode, first, start + 1)
        maps = np.zeros((self.history, 10, 2, self.map_size, self.map_size), np.float32)
        mask = np.zeros((self.history, 10), bool)
        maps[-len(history) :], mask[-len(history) :] = history, valid
        future, future_valid = self._read_tactile(h5, episode, start + self.future_stride, end)
        return {
            "tactile_maps": torch.from_numpy(maps),
            "tactile_valid": torch.from_numpy(mask),
            "future_tactile_maps": torch.from_numpy(future[:: self.future_stride].copy()),
            "future_tactile_valid": torch.from_numpy(future_valid[:: self.future_stride].copy()),
            "future_tactile_offsets": torch.from_numpy(offsets),
        }


class Bench2DexTactileMixture(BaseDataset):
    """Balance tasks sharing one embodiment and one normalization artifact.

    This is a homogeneous task sampler. The heterogeneous MixtureDataset's
    per-source deployment/normalization package does not apply here.
    """

    dataset_class = Bench2DexTactileDataset
    action_dim = 80

    def __init__(self, datasets, names, *, seed=42, split="train"):
        self.split = split
        self.datasets = list(datasets)
        self.names = list(names)
        self._base_seed = self._seed = seed
        paths = {ds.normalization_stats_path for ds in datasets}
        if len(paths) != 1:
            raise ValueError("All tasks must use the same normalization statistics")
        self.normalization_stats_path = paths.pop()
        self._build_index_map()

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, index):
        task, sample = self._index_map[index]
        return self.datasets[int(task)][int(sample)]

    def set_epoch(self, epoch):
        seed = self._base_seed + int(epoch) * 7919
        if seed != self._seed:
            self._seed = seed
            self._build_index_map()

    def _build_index_map(self):
        rng = np.random.default_rng(self._seed)
        target = max(len(ds) for ds in self.datasets)
        parts = []
        for i, ds in enumerate(self.datasets):
            n = len(ds)
            count = target if self.split == "train" else n
            # Each full permutation covers every real window before repetition.
            indices = np.concatenate([rng.permutation(n) for _ in range((count + n - 1) // n)])[:count]
            parts.append(np.column_stack([np.full(count, i), indices]))
        combined = np.concatenate(parts).astype(np.int32)
        self._index_map = combined[rng.permutation(len(combined))]

    @classmethod
    def from_config(cls, config, split="train"):
        manifest = json.loads(Path(get_cfg(config, "manifest_path")).read_text())
        names = manifest.get("task_names")
        if not names:
            return cls.dataset_class.from_config(config, split)
        if len(names) != len(set(names)):
            raise ValueError("Manifest task_names must be unique")
        datasets = []
        for name in names:
            cfg = dict(config.items())
            cfg["task_name"] = name
            datasets.append(cls.dataset_class.from_config(cfg, split))
        dataset = cls(datasets, names, seed=int(get_cfg(config, "seed", 42)), split=split)
        from omegaconf import OmegaConf

        dataset.training_data_contract = {
            "schema_version": 1,
            "dataset_type": "bench2dex_tactile",
            "split": split,
            "manifest_sha256": hashlib.sha256(Path(get_cfg(config, "manifest_path")).read_bytes()).hexdigest(),
            "config": OmegaConf.to_container(OmegaConf.create(dict(config.items())), resolve=True),
            "sampling_algorithm": "equal_task_full_permutations_v1",
            "seed": dataset._base_seed,
        }
        return dataset
