"""LeRobot v3 video/touch sources and native Sharpa intensity integration.

All sources retain their native timestamps on disk. This video reader requires
and verifies a regular episode clock; irregular raw streams must be resampled
by their offline adapter first. The model sees explicit physical offsets;
human poses and unverified robot states never become actions.
"""

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.bases.multi_lerobot_v3_reader import MultiLeRobotV3Reader
from openwam.dataloader.egodex_video import EgoDexVideoDataset
from openwam.dataloader.sharpadex_native_tcp import SharpaDexNativeTCPDataset
from openwam.dataloader.sharpadex_tianji_global import MultiSharpaDexTianjiGlobalDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.lerobotv3 import (
    apply_info_splits, compute_file_local_offsets,
    parse_info_json, subsample_episodes_by_hours,
)
from openwam.dataloader.utils.video_io import decode_video_frames
from openwam.tactile.intensity_store import FUTURE_TIMES, HISTORY_TIMES, IntensityStore
from openwam.tactile.rgb_store import RGB_CODEC, RGBTactileStore


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_video_episodes(root, cameras):
    """Read only the selected video layout/index columns, never feature stats."""
    paths = sorted((Path(root) / "meta/episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata under {root}")
    required = {"episode_index", "length", "dataset_from_index", "data/chunk_index", "data/file_index"}
    optional = {"dataset_to_index", "tasks"}
    video_columns = {f"videos/{camera}/{field}" for camera in cameras
                     for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")}
    columns = set(pq.read_schema(paths[0]).names)
    if (required | video_columns) - columns:
        raise ValueError(f"Missing episode/video index columns: {sorted((required | video_columns) - columns)}")
    selected = sorted(required | video_columns | (optional & columns))
    table = pa.concat_tables([pq.read_table(path, columns=selected) for path in paths])
    return table.to_pandas().sort_values("episode_index").reset_index(drop=True)


@lru_cache(maxsize=4)
def _read_video_clock_table(path):
    """One process-wide four-shard cache shared by every source bucket."""
    names = pq.read_schema(path).names
    keys = [k for k in names if k in ("timestamp", "episode_index", "task_index", "task", "prompt")
            or k.endswith(".valid") or k.endswith("_valid")]
    return pq.read_table(path, columns=keys)


class HomogeneousPublishedBuckets(MultiLeRobotV3Reader):
    """A manifest-selected population: missing buckets fail rather than shrink."""

    def __init__(self, buckets):
        super().__init__(buckets)
        modes = {b.training_modality for b in buckets}
        if len(modes) != 1:
            raise ValueError("Split differing supervision modes into separate mixture sources")
        self.training_modality = modes.pop()
        codecs = {getattr(b, "tactile_codec", None) for b in buckets} - {None}
        if len(codecs) > 1:
            raise ValueError("A source must use one tactile representation; split RGB and intensity sources explicitly")
        self.tactile_codec = next(iter(codecs), None)
        shapes = {getattr(b, "tactile_observation_shape", None) for b in buckets}
        if len(shapes) != 1:
            raise ValueError("A source must use one tactile observation shape; split native shapes into separate sources")
        self.tactile_observation_shape = shapes.pop()
        self.normalization_stats_path = None
        paths = [getattr(b, "normalization_stats_path", None) for b in buckets]
        if any(paths):
            if not all(paths) or len({Path(p).resolve(strict=True) for p in paths}) != 1:
                raise ValueError("A normalized source must use one shared action/state statistics artifact across buckets")
            self.normalization_stats_path = str(Path(paths[0]).resolve(strict=True))

    def set_epoch(self, epoch):
        for b in self.buckets:
            if hasattr(b, "set_epoch"):
                b.set_epoch(epoch)

    @property
    def training_data_contract(self):
        return {"schema": "touchwam_published_buckets_v1", "training_modality": self.training_modality,
                "tactile_codec": self.tactile_codec,
                "buckets": [b.training_data_contract for b in self.buckets]}


class PublishedModalityBuckets(MultiLeRobotV3Reader):
    """A complete source whose buckets have different available supervision.

    MixtureDataset expands its homogeneous groups before global sampling. This
    permits incremental, verified tactile calibration without discarding the
    remaining video data or mixing different model graphs in a global batch.
    """

    training_modality = "mixed"
    normalization_stats_path = None
    tactile_codec = None

    def __init__(self, buckets):
        super().__init__(buckets)
        self.modality_groups = {
            mode: HomogeneousPublishedBuckets([b for b in buckets if b.training_modality == mode])
            for mode in sorted({b.training_modality for b in buckets})}

    @property
    def training_data_contract(self):
        return {"schema": "touchwam_published_modality_groups_v1",
                "groups": {mode: group.training_data_contract for mode, group in self.modality_groups.items()}}


class EgoDexReuseVideoDataset(EgoDexVideoDataset):
    """Read the existing accepted EgoDex video cache, without reconversion."""

    def __init__(self, *args, max_hours=None, include_episode_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        chosen = np.arange(len(self.lengths))
        if include_episode_ids is not None:
            wanted = set(include_episode_ids)
            chosen = np.asarray([i for i, key in enumerate(self.episodes["id"]) if key in wanted], dtype=np.int64)
            if {self.episodes["id"][i] for i in chosen} != wanted:
                raise ValueError("Requested EgoDex episode selection contains absent data")
        chosen = chosen[self.lengths[chosen] > self.video_steps[1]]
        if max_hours is not None:
            if not np.isfinite(max_hours) or max_hours <= 0:
                raise ValueError("total_hours must be positive or null")
            order = np.random.default_rng(self.seed).permutation(chosen)
            seconds = np.cumsum(self.lengths[order]) / float(self.info["fps"])
            count = min(len(order), int(np.searchsorted(seconds, max_hours * 3600)) + 1)
            chosen = np.sort(order[:count])
        if not len(chosen):
            raise ValueError("No selected EgoDex episode has a future video target")
        self.episodes = {key: [values[i] for i in chosen] for key, values in self.episodes.items()}
        self.lengths, self.video_offsets = self.lengths[chosen], self.video_offsets[chosen]
        # A pure V sample must contain a real future frame. The old cache and
        # its offsets stay unchanged; only useless terminal training starts go.
        starts = (self.lengths - self.video_steps[1] - 1) // self.window_stride + 1
        self.cumulative = np.concatenate(([0], np.cumsum(starts)))
        self.training_data_contract.update(
            dataset_type="egodex_reuse_video", total_windows=len(self),
            window_policy="require_one_future_video_frame_v1", requested_hours=max_hours,
            selected_episodes=len(chosen), selected_hours=float(self.lengths.sum() / self.info["fps"] / 3600),
            selected_episode_indices_sha256=hashlib.sha256(chosen.astype('<i8').tobytes()).hexdigest(),
            include_episode_ids=include_episode_ids)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        for key in ("action", "action_mask", "proprio", "proprio_mask"):
            sample.pop(key, None)
        return sample

    @classmethod
    def from_config(cls, config, split="train"):
        keys = ("num_frames", "video_stride", "window_stride", "height", "width", "color_jitter", "seed",
                "enable_action_supervision", "enable_proprio_supervision", "include_episode_ids")
        kwargs = {k: get_cfg(config, k) for k in keys if get_cfg(config, k) is not None}
        return cls(get_cfg(config, "dataset_dir"), split=split, max_hours=get_cfg(config, "total_hours"), **kwargs)


class LeRobotVideoIntensityDataset(BaseDataset):
    """Strictly V or VT. Does not manufacture robot action/proprio targets."""

    action_dim = 80
    normalization_stats = None
    normalization_stats_path = None
    SOURCE_KIND = "ego"
    EXTRA_CONFIG_FIELDS = ()

    def __init__(self, dataset_dir, *, split="train", sampling_fps=30, num_frames=33,
                 video_stride=4, window_stride=1, height=384, width=320,
                 camera_layout=None, tactile_mode="auto", tactile_map_size=32,
                 tactile_representation="intensity", tactile_rgb_manifest="meta/tactile_rgb.json",
                 tactile_rgb_map_size=128, tactile_observation_map_size=None, tactile_encoder_name=None,
                 history_times=HISTORY_TIMES, future_times=FUTURE_TIMES,
                 color_jitter=None, seed=42, max_hours=None):
        self.root = Path(dataset_dir).resolve(strict=True)
        self.info = parse_info_json(self.root)
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("Expected an actual LeRobot v3 dataset")
        definition = self.root / "meta/source_definition.json"
        self.definition = json.loads(definition.read_text()) if definition.exists() else {}
        robot = self.definition.get("source_kind") == "robot" or str(self.definition.get("source_id", "")).startswith("ftp1/")
        if robot and self.SOURCE_KIND != "robot":
            raise ValueError("Robot/FTP data requires an action reader; V/VT fallback is forbidden")
        if self.definition.get("training_ready") is False or self.definition.get("physical_time_verified") is False:
            raise ValueError(f"Source has unresolved training/time contract: {self.root}")
        self.fps, self.sampling_fps = float(self.info["fps"]), float(sampling_fps)
        if not np.isfinite([self.fps, self.sampling_fps]).all() or min(self.fps, self.sampling_fps) <= 0:
            raise ValueError("Verified positive native and sampling FPS required")
        if any(type(v) is not int or v < 1 for v in (num_frames, video_stride, window_stride, height, width)):
            raise ValueError("Window dimensions must be positive integers")
        if num_frames < 2:
            raise ValueError("Video learning requires a future target")
        self.num_frames, self.window_stride = num_frames, window_stride
        self.height, self.width = height, width
        self.video_offsets_s = np.arange(0, num_frames, video_stride) / self.sampling_fps
        if len(self.video_offsets_s) < 2:
            raise ValueError("Video stride leaves no future frame")
        self.seed, self.epoch = int(seed), 0
        self.source_id = str(self.definition.get("source_id", self.root.name))
        self.camera_layout = camera_layout or self.definition.get("camera_layout")
        if isinstance(self.camera_layout, dict):
            self.camera_layout = [self.camera_layout.get(k, f"__missing_{k}__")
                                  for k in ("head", "left_wrist", "right_wrist")]
        if self.camera_layout is None:
            cameras = [k for k, v in self.info.get("features", {}).items()
                       if v.get("dtype") == "video" and k.startswith("observation.images.")]
            if len(cameras) != 1:
                raise ValueError("Multi-camera sources require an explicit semantic camera_layout")
            self.camera_layout = [cameras[0], "__missing_left__", "__missing_right__"]
        if len(self.camera_layout) != 3 or not self.camera_layout[0]:
            raise ValueError("camera_layout requires head/left wrist/right wrist slots")
        self.cameras = [c for c in self.camera_layout if c and not c.startswith("__missing_")]
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("Do not duplicate one camera into multiple physical slots")
        for c in self.cameras:
            if self.info.get("features", {}).get(c, {}).get("dtype") != "video":
                raise ValueError(f"Missing configured camera: {c}")
        eps = _load_video_episodes(self.root, self.cameras)
        if eps.episode_index.duplicated().any() or (eps.length <= 0).any():
            raise ValueError("Duplicate or empty LeRobot episodes")
        # Compute physical shard offsets before selecting a split or budget.
        eps["_data_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
        eps = apply_info_splits(eps, split, self.info.get("splits", {}), source_name=self.source_id)
        if max_hours is not None:
            eps = subsample_episodes_by_hours(eps, float(max_hours), self.fps, self.seed)
        # No pure-video window with zero future supervision is sampled.
        last_margin = max(1, int(np.ceil(self.video_offsets_s[1] * self.fps - 1e-7)))
        eps = eps.loc[eps.length > last_margin].reset_index(drop=True)
        if eps.empty:
            raise ValueError(f"No usable {split} episodes in {self.root}")
        self.episodes = eps
        self.lengths = eps.length.to_numpy(np.int64)
        self.starts = (self.lengths - last_margin - 1) // window_stride + 1
        self.start_offsets = None
        validity_keys = [c + ".valid" for c in self.cameras if c + ".valid" in self.info["features"]]
        if validity_keys:
            # A missing recorded frame cannot be an input image or a black
            # target. Keep windows whose real camera observations all exist;
            # episode-tail padding is handled separately by video_mask.
            by_file = {}
            self.start_offsets = []
            relative = np.rint(self.video_offsets_s * self.fps).astype(np.int64)
            for _, row in eps.iterrows():
                key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
                if key not in by_file:
                    file_path = self.root / self.info["data_path"].format(chunk_index=key[0], file_index=key[1])
                    tab = pq.read_table(file_path, columns=validity_keys)
                    by_file[key] = np.logical_and.reduce([
                        np.asarray(tab[k].fill_null(False), dtype=bool) for k in validity_keys])
                first_row, length = int(row["_data_offset"]), int(row.length)
                measured = by_file[key][first_row:first_row + length]
                if len(measured) != length:
                    raise ValueError("Camera validity rows do not cover the episode")
                starts = np.arange(0, length - last_margin, window_stride, dtype=np.int64)
                keep = np.ones(len(starts), dtype=bool)
                for delta in relative:
                    indices = starts + delta
                    keep &= (indices >= length) | measured[indices.clip(max=length - 1)]
                self.start_offsets.append(starts[keep])
            self.starts = np.asarray([len(a) for a in self.start_offsets], dtype=np.int64)
            if self.starts.sum() == 0:
                raise ValueError(f"No complete observed camera windows: {self.root}")
        self.cumulative = np.concatenate(([0], np.cumsum(self.starts)))
        tasks_path = self.root / "meta/tasks.parquet"
        self.tasks = {}
        if tasks_path.exists():
            df = pq.read_table(tasks_path).to_pandas()
            for i, row in df.iterrows():
                tid = int(row.get("task_index", i))
                text = row.get("task", row.get("task_description", row.get("task_name")))
                if text is None and isinstance(i, str):
                    text = i
                if isinstance(text, str):
                    self.tasks[tid] = text
        if tactile_mode not in ("auto", "required", "off"):
            raise ValueError("tactile_mode must be auto, required or off")
        if tactile_representation not in ("intensity", "rgb"):
            raise ValueError("tactile_representation must be intensity or rgb")
        self.tactile_representation = tactile_representation
        manifest = self.root / (tactile_rgb_manifest if tactile_representation == "rgb" else "tactile/manifest.json")
        if tactile_mode == "required" and not manifest.exists():
            raise FileNotFoundError(f"Required tactile publication missing: {manifest}")
        store = RGBTactileStore if tactile_representation == "rgb" else IntensityStore
        map_size = tactile_rgb_map_size if tactile_representation == "rgb" else tactile_map_size
        self.touch = store(manifest, self.root, size=map_size, history_times=history_times, future_times=future_times,
                           observation_size=tactile_observation_map_size, encoder_name=tactile_encoder_name) \
            if tactile_mode != "off" and manifest.exists() else None
        self.tactile_codec = (RGB_CODEC if tactile_representation == "rgb" else "tactile_intensity_v1") if self.touch else None
        self.tactile_observation_shape = self.touch.observation_hw if self.touch else None
        if self.touch and any(str(int(e)) not in self.touch.episodes for e in eps.episode_index):
            raise ValueError("Tactile manifest does not cover every selected episode")
        self.training_modality = "video_tactile" if self.touch else "video_only"
        self.jitter = VideoColorJitter(**{k: get_cfg(color_jitter, k, d) for k, d in
                                       (("brightness", .2), ("contrast", .2), ("saturation", .2), ("hue", 0.))}) \
            if color_jitter_enabled(color_jitter) else None
        self.training_data_contract = {
            "schema": "touchwam_lerobot_video_v1", "dataset_dir": str(self.root),
            "info_sha256": _sha(self.root / "meta/info.json"),
            "episode_metadata": {str(p.relative_to(self.root)): _sha(p)
                                 for p in sorted((self.root / "meta/episodes").rglob("*.parquet"))},
            "definition_sha256": _sha(definition) if definition.exists() else None,
            "tactile_manifest_sha256": self.touch.digest if self.touch else None,
            "tactile_codec": self.tactile_codec, "tactile_representation": tactile_representation,
            "tactile_map_size": map_size if self.touch else None,
            **({"tactile_observation_map_size": list(self.touch.observation_hw),
                "tactile_encoder_name": tactile_encoder_name}
               if self.touch and self.touch.observation_hw is not None else {}),
            "split": split, "selected_episodes": eps.episode_index.astype(int).tolist(),
            "source_fps": self.fps, "sampling_fps": self.sampling_fps,
            "video_times_s": self.video_offsets_s.tolist(), "window_stride": window_stride,
            "height": height, "width": width, "camera_layout": list(self.camera_layout),
            "training_modality": self.training_modality, "windows": len(self), "seed": self.seed,
            "color_jitter": None if self.jitter is None else {k: getattr(self.jitter, k) for k in
                                                            ("brightness", "contrast", "saturation", "hue")},
        }

    def __len__(self):
        return int(self.cumulative[-1])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _table(self, chunk, file):
        p = self.root / self.info["data_path"].format(chunk_index=chunk, file_index=file)
        return _read_video_clock_table(str(p))

    @lru_cache(maxsize=128)
    def _episode_clock(self, chunk, file, data_offset, length, episode_id):
        """Verify the row-index/video-clock equivalence before decoding frames.

        LeRobot videos are encoded at a constant FPS. Using row indices with
        irregular timestamps silently changes the physical prediction horizon,
        so reject such publications instead of pretending they are synchronized.
        """
        table = self._table(chunk, file).slice(data_offset, length)
        if len(table) != length or "timestamp" not in table.column_names:
            raise ValueError("Episode is incomplete or lacks an explicit timestamp clock")
        times = np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
        if (not np.isfinite(times).all() or (np.diff(times) <= 0).any()
                or not np.allclose(times, times[0] + np.arange(length) / self.fps,
                                   rtol=0, atol=max(2e-5, 1e-3 / self.fps))):
            raise ValueError("Irregular episode timestamps: offline resampling is required before this video reader")
        if "episode_index" in table.column_names and not np.all(
            table["episode_index"].to_numpy(zero_copy_only=False) == episode_id
        ):
            raise ValueError("Episode metadata includes rows belonging to a different episode")
        return float(times[0]), float(times[0]) + (length - 1) / self.fps

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ei = int(np.searchsorted(self.cumulative, index, side="right") - 1)
        offset = int(index - self.cumulative[ei]) * self.window_stride
        if self.start_offsets is not None:
            offset = int(self.start_offsets[ei][index - self.cumulative[ei]])
        row = self.episodes.iloc[ei]
        length = int(row.length)
        clock_start, clock_end = self._episode_clock(
            int(row["data/chunk_index"]), int(row["data/file_index"]), int(row["_data_offset"]),
            length, int(row.episode_index),
        )
        table = self._table(int(row["data/chunk_index"]), int(row["data/file_index"]))
        first = table.slice(int(row["_data_offset"]) + offset, 1).to_pylist()[0]
        if "episode_index" in first and int(first["episode_index"]) != int(row.episode_index):
            raise ValueError("Episode metadata addresses a different physical episode")
        # _episode_clock has validated recorded timestamps against this grid.
        # A float32 timestamp just below t must not discard the current tactile frame.
        ref = clock_start + offset / self.fps
        relative = np.rint(self.video_offsets_s * self.fps).astype(np.int64)
        valid = offset + relative < length
        requested = (offset + relative).clip(max=length - 1)
        for camera in self.cameras:
            key = camera + ".valid"
            if key in table.column_names:
                observed = np.asarray(table[key].slice(int(row["_data_offset"]), length).fill_null(False), dtype=bool)
                valid &= observed[requested]
        if not valid[0] or not valid[1:].any():
            raise ValueError("Window has no observed image prefix or future video label")
        frames = {}
        for camera in self.cameras:
            prefix = f"videos/{camera}/"
            chunk, file = int(row[prefix + "chunk_index"]), int(row[prefix + "file_index"])
            video_fps = float(self.info["features"][camera].get("info", {}).get("video.fps", self.fps))
            start = float(row[prefix + "from_timestamp"]) * video_fps
            if not np.isfinite(start) or start < 0 or not np.isclose(start, round(start), atol=2e-3, rtol=0):
                raise ValueError("Video episode offset is not on a frame boundary")
            absolute = round(start) + np.rint(requested / self.fps * video_fps).astype(np.int64)
            path = self.root / self.info["video_path"].format(video_key=camera, chunk_index=chunk, file_index=file)
            frames[camera] = decode_video_frames(str(path), absolute.tolist(), self.height, self.width)
        canvases = [assemble_multiview_layout({k: v[i] for k, v in frames.items()},
                    self.camera_layout, self.height, self.width, return_missing_mask=True)
                    for i in range(len(requested))]
        video = [x[0] for x in canvases]
        if self.jitter:
            state = random.getstate()
            random.seed(f"{self.source_id}/{self.seed}/{self.epoch}/{index}")
            try:
                video = self.jitter.apply({"video": video,
                    "video_jitter_exclusion_masks": [x[1] for x in canvases]})["video"]
            finally:
                random.setstate(state)
        prompt = first.get("prompt", first.get("task"))
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = self.tasks.get(int(first.get("task_index", -1)))
        if not isinstance(prompt, str) or not prompt.strip():
            tasks = row.get("tasks")
            prompt = tasks[0] if isinstance(tasks, (list, np.ndarray)) and len(tasks) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Missing episode language")
        sample = {"video": video, "video_mask": torch.from_numpy(valid), "prompt": prompt,
                  "first_frame_image": [video[0]], "vace_video": None,
                  "training_modality": self.training_modality, "source_id": self.source_id,
                  "source_kind": self.SOURCE_KIND,
                  "episode_id": int(row.episode_index), "episode_start_frame": offset}
        if self.touch:
            sample.update(self.touch.window(int(row.episode_index), ref,
                          segment_start_s=clock_start, segment_end_s=clock_end))
        return sample

    @classmethod
    def from_config(cls, config, split="train"):
        root = Path(get_cfg(config, "dataset_dir"))
        fields = ("sampling_fps", "num_frames", "video_stride", "window_stride", "height", "width",
                  "camera_layout", "tactile_mode", "tactile_map_size", "history_times", "future_times",
                  "tactile_representation", "tactile_rgb_manifest", "tactile_rgb_map_size",
                  "tactile_observation_map_size", "tactile_encoder_name",
                  "color_jitter", "seed") + cls.EXTRA_CONFIG_FIELDS
        kwargs = {k: get_cfg(config, k) for k in fields if get_cfg(config, k) is not None}
        total_hours = get_cfg(config, "total_hours")
        if (root / "meta/info.json").exists():
            return cls(root, split=split, max_hours=total_hours, **kwargs)
        # Immutable published roots are inspected at construction only. Missing
        # expected entries are additionally rejected by the publication preflight.
        paths = sorted(p.parent.parent for p in root.rglob("meta/info.json"))
        if not paths:
            raise FileNotFoundError(f"No published LeRobot v3 buckets in {root}")
        includes = get_cfg(config, "include_buckets")
        if includes is not None:
            wanted = set(includes)
            paths = [p for p in paths if p.name in wanted]
            if {p.name for p in paths} != wanted:
                raise ValueError("Requested bucket selection contains absent data")
        from openwam.dataloader.utils.lerobotv3 import water_fill_hours
        budgets = [None] * len(paths)
        if total_hours is not None:
            hours = [sum(apply_info_splits(_load_video_episodes(p, []), split, parse_info_json(p).get("splits", {})).length)
                     / float(parse_info_json(p)["fps"]) / 3600
                     for p in paths]
            budgets = water_fill_hours(hours, float(total_hours))
        buckets = [cls(p, split=split, max_hours=h, **kwargs)
                   for p, h in zip(paths, budgets) if h is None or h > 0]
        if len({b.tactile_codec for b in buckets} - {None}) > 1:
            raise ValueError("A source cannot mix RGB and intensity tactile representations")
        if len({b.training_modality for b in buckets}) > 1:
            return PublishedModalityBuckets(buckets)
        return HomogeneousPublishedBuckets(buckets)


class MultiSharpaIntensityDataset(MultiSharpaDexTianjiGlobalDataset):
    training_modality = "video_action_tactile"
    tactile_codec = "tactile_intensity_v1"

    def __init__(self, buckets):
        super().__init__(buckets)
        shapes = {getattr(b, "tactile_observation_shape", None) for b in buckets}
        if len(shapes) != 1:
            raise ValueError("Sharpa buckets must use one tactile observation shape")
        self.tactile_observation_shape = shapes.pop()

    @property
    def training_data_contract(self):
        return {"schema": "sharpadex_native_intensity_v1",
                "buckets": [b.training_data_contract for b in self.buckets]}


class SharpaNativeIntensityDataset(SharpaDexNativeTCPDataset):
    training_modality = "video_action_tactile"
    tactile_codec = "tactile_intensity_v1"
    NEEDED_COLS = (*SharpaDexNativeTCPDataset.NEEDED_COLS, "timestamp")
    CONFIG_KEYS = (*SharpaDexNativeTCPDataset.CONFIG_KEYS, "tactile_map_size", "history_times", "future_times", "tactile_root",
                   "tactile_observation_map_size", "tactile_encoder_name", "tactile_observation_root")

    def __init__(self, *args, tactile_map_size=32, history_times=HISTORY_TIMES, future_times=FUTURE_TIMES,
                 tactile_root=None, tactile_observation_map_size=None, tactile_encoder_name=None,
                 tactile_observation_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        manifest = self._dataset_dir / "tactile/manifest.json"
        if tactile_root is not None:
            partition = self._dataset_dir.resolve(strict=True)
            manifest = Path(tactile_root) / "partitions" / partition.parent.name / partition.name / "tactile/manifest.json"
        observation_manifest = None
        if tactile_observation_root is not None:
            partition = self._dataset_dir.resolve(strict=True)
            observation_manifest = Path(tactile_observation_root) / "partitions" / partition.parent.name / partition.name / "observation.json"
        self.touch = IntensityStore(manifest, self._dataset_dir,
                                   size=tactile_map_size, history_times=history_times, future_times=future_times,
                                   observation_size=tactile_observation_map_size, encoder_name=tactile_encoder_name,
                                   observation_manifest=observation_manifest)
        self.tactile_observation_shape = self.touch.observation_hw
        if any(str(int(e)) not in self.touch.episodes for e in self._eps_df.episode_index):
            raise ValueError("Tactile manifest does not cover every selected Sharpa episode")
        self.training_data_contract = {**self.training_data_contract,
            "tactile_manifest_sha256": self.touch.digest, "tactile_history_times_s": list(history_times),
            "future_tactile_times_s": list(future_times), "tactile_map_size": tactile_map_size,
            "selected_episodes": self._eps_df.episode_index.astype(int).tolist(),
            **({"tactile_observation_map_size": list(self.touch.observation_hw),
                "tactile_encoder_name": tactile_encoder_name}
               if self.touch.observation_hw is not None else {})}
        if self.touch.video_history is not None:
            self.training_data_contract["tactile_observation_manifest_sha256"] = self.touch.video_history.digest
            self.training_data_contract["tactile_observation_storage"] = "sharpa_deform_video_history_v1"

    def _observation_extras(self, idx, row, offset, valid_start, valid_end, win):
        # The accepted native release and its sidecar share an episode-relative
        # regular clock. Parquet stores that clock as float32 while sidecars use
        # float64. Snap only after validating row/grid agreement: otherwise a
        # rounded-down timestamp loses the current causal observation, and a
        # rounded-up last timestamp falls outside its own episode.
        expected = (offset + np.arange(len(win), dtype=np.float64)) / self._fps
        recorded = win.timestamp.to_numpy(dtype=np.float64)
        if (not len(expected) or not np.isfinite(recorded).all()
                or not np.allclose(recorded, expected, rtol=0, atol=max(2e-5, 1e-3 / self._fps))):
            raise ValueError("Native Sharpa timestamps differ from the verified episode-relative frame clock")
        reference = float(expected[0])
        return {**self.touch.window(int(row.episode_index), reference,
            segment_start_s=valid_start / self._fps,
            segment_end_s=(valid_end - 1) / self._fps),
            "training_modality": self.training_modality, "source_id": "sharpadex",
            "episode_id": int(row.episode_index), "episode_start_frame": offset}

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiSharpaIntensityDataset
