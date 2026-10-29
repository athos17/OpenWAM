"""Versioned regional intensity sidecars sampled on a physical time axis.

Storage remains lossless with respect to the exported representation. Missing
measurements are metadata; a zero-valued valid image remains a measurement.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .registry import REGION_NAMES
from .regions import observation_shape


INTENSITY_CODEC = "tactile_intensity_v1"
HISTORY_TIMES = (-0.1, -2 / 30, -1 / 30, 0.0)
FUTURE_TIMES = tuple(i / 10 for i in range(1, 9))


def _time_grid(values, *, future):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (np.diff(a) <= 0).any():
        raise ValueError("Time offsets must be finite, nonempty and strictly increasing")
    if (a <= 0).any() if future else ((a > 0).any() or a[-1] != 0):
        raise ValueError("History ends at zero; future times must be strictly positive")
    return a


class IntensityStore:
    """Read [T,N,H,W] maps with explicit layout, validity and timestamps.

    Native timestamps can be irregular. History uses causal sample-and-hold;
    future supervision uses the closest *future* measurement within tolerance.
    No window can reach across an episode or a reader's segment boundary.
    """

    def __init__(self, manifest_path, dataset_dir, *, size=32,
                 history_times=HISTORY_TIMES, future_times=FUTURE_TIMES,
                 observation_size=None, encoder_name=None, observation_manifest=None):
        self.path = Path(manifest_path).resolve(strict=True)
        self.root = self.path.parent
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.meta = json.loads(self.path.read_text())
        self.history_times = _time_grid(history_times, future=False)
        self.future_times = _time_grid(future_times, future=True)
        if type(size) is not int or size < 1:
            raise ValueError("Map size must be a positive integer")
        self.size = size
        self.observation_hw = observation_shape(observation_size) if observation_size is not None else None
        if self.observation_hw is None and encoder_name is not None:
            raise ValueError("encoder_name requires observation_size")
        self.encoder_name = encoder_name
        digest = hashlib.sha256((Path(dataset_dir) / "meta/info.json").read_bytes()).hexdigest()
        if (self.meta.get("schema") not in ("tactile_regions_v1", INTENSITY_CODEC)
                or self.meta.get("codec") not in (INTENSITY_CODEC, "sharpa_deform_gray_v1")
                or self.meta.get("dataset_info_sha256") != digest):
            raise ValueError("Intensity sidecar schema/codec/dataset fingerprint mismatch")
        self.names = list(self.meta.get("region_names", []))
        if not self.names or len(set(self.names)) != len(self.names) or set(self.names) - set(REGION_NAMES):
            raise ValueError("Intensity regions must be unique members of the public registry")
        if any(k in self.meta for k in ("surface_ids", "tactile_surface_ids")):
            raise ValueError("surface_id is unsupported: compose one image per public region")
        self.sensor_names = self.meta.get("sensor_names")
        if self.sensor_names is None and self.meta.get("codec") == "sharpa_deform_gray_v1":
            if self.meta.get("sensor_ids") == [0] * len(self.names):
                self.sensor_names = ["sharpa_real"] * len(self.names)
        if (not isinstance(self.sensor_names, list) or len(self.sensor_names) != len(self.names)
                or any(not isinstance(n, str) or not n.strip() for n in self.sensor_names)):
            raise ValueError("Intensity sidecars require explicit sensor_names")
        declared_encoders = self.meta.get("encoder_names")
        if encoder_name is not None and declared_encoders is not None and declared_encoders != [encoder_name] * len(self.names):
            raise ValueError("Dataset encoder_name conflicts with intensity region encoder_names")
        self.encoder_names = [encoder_name] * len(self.names) if encoder_name is not None else declared_encoders
        if self.observation_hw is not None and (
            not isinstance(self.encoder_names, list) or len(self.encoder_names) != len(self.names)
            or any(not isinstance(name, str) or not name.strip() for name in self.encoder_names)
        ):
            raise ValueError("Heterogeneous intensity observations require explicit per-region encoder_name(s)")
        self.fps = float(self.meta["fps"])
        self.max_age = float(self.meta.get("history_max_age_s", 1.5 / self.fps)) if self.fps > 0 else -1
        self.future_tolerance = float(self.meta.get("future_tolerance_s", .51 / self.fps)) if self.fps > 0 else -1
        if (not np.isfinite([self.fps, self.max_age, self.future_tolerance]).all()
                or self.fps <= 0 or self.max_age < 0 or self.future_tolerance < 0):
            raise ValueError("Invalid native FPS or time-matching tolerance")
        self.episodes = self.meta["episodes"]
        if not isinstance(self.episodes, dict):
            raise ValueError("Manifest episodes must be an index mapping")
        self.video_history = None
        if observation_manifest is not None:
            if self.observation_hw is None or self.encoder_names != ["SharpaWave"] * 10:
                raise ValueError("Native Sharpa video history requires the SharpaWave observation encoder")
            from .sharpa_video_history import SharpaVideoHistory
            self.video_history = SharpaVideoHistory(
                observation_manifest, dataset_dir, self.path, self.meta, self.observation_hw)

    def _path(self, entry, key):
        p = (self.root / entry[key]).resolve(strict=True)
        if not p.is_relative_to(self.root):
            raise ValueError("Sidecar array escapes its manifest root")
        return p

    @lru_cache(maxsize=4)
    def arrays(self, episode):
        entry = self.episodes[str(int(episode))]
        maps, valid, times = [np.load(self._path(entry, k), mmap_mode="r", allow_pickle=False)
                              for k in ("maps", "valid", "timestamps")]
        n = len(self.names)
        present = np.asarray(entry["layout_present"])
        if (maps.ndim != 4 or maps.shape[1] != n or min(maps.shape) < 1
                or maps.dtype not in (np.uint8, np.float16, np.float32, np.float64)
                or valid.dtype != np.bool_ or valid.shape != maps.shape[:2]
                or times.shape != (len(maps),) or not np.isfinite(times).all()
                or (np.diff(times) <= 0).any() or present.dtype != np.bool_
                or present.shape != (n,) or (valid & ~present).any()):
            raise ValueError(f"Invalid intensity arrays for episode {episode}")
        if self.video_history is None and self.observation_hw is not None and any(
            target > native for target, native in zip(self.observation_hw, maps.shape[-2:])
        ):
            raise ValueError(
                f"Intensity sidecar contains only {tuple(maps.shape[-2:])}, but observation encoder requests "
                f"{self.observation_hw}; supply native-resolution deform/matrix storage, not upsampled labels"
            )
        spatial_path = entry.get("spatial_valid", self.meta.get("spatial_valid"))
        if spatial_path is None:
            spatial = np.ones((n, *maps.shape[-2:]), dtype=bool)
        else:
            spatial = np.load(self._path({"spatial_valid": spatial_path}, "spatial_valid"), allow_pickle=False)
            if spatial.dtype != np.bool_ or spatial.shape != (n, *maps.shape[-2:]):
                raise ValueError("Invalid spatial ROI")
        if self.video_history is not None and not spatial.all():
            raise ValueError("Native Sharpa video history requires independently published native ROI for masked caches")
        return maps, valid, times, present, spatial

    def window(self, episode, reference_time, *, segment_start_s=None, segment_end_s=None):
        maps, valid, times, present, spatial = self.arrays(episode)
        if not np.isfinite(reference_time):
            raise ValueError("Nonfinite window reference time")
        start = float(times[0]) if segment_start_s is None else max(float(times[0]), segment_start_s)
        end = float(times[-1]) if segment_end_s is None else min(float(times[-1]), segment_end_s)
        if not start - 1e-8 <= reference_time <= end + 1e-8:
            raise ValueError("Reference outside episode/segment")
        ht = reference_time + self.history_times
        hi = np.searchsorted(times, ht + 1e-8, side="right") - 1
        hc = hi.clip(0, len(times) - 1)
        hv = ((hi >= 0) & (times[hc] >= start - 1e-8) & (times[hc] <= ht + 1e-8)
              & (ht - times[hc] <= self.max_age + 1e-8))
        ft = reference_time + self.future_times
        right = np.searchsorted(times, ft).clip(0, len(times) - 1)
        left = (right - 1).clip(0)
        fi = np.where(np.abs(times[left] - ft) <= np.abs(times[right] - ft), left, right)
        fv = ((times[fi] > reference_time + 1e-8) & (times[fi] <= end + 1e-8)
              & (np.abs(times[fi] - ft) <= self.future_tolerance + 1e-8)
              & (ft <= end + 1e-8))

        def gather(rows, time_valid, target_hw):
            values = np.array(maps[rows], copy=True)
            enabled = np.array(valid[rows] & time_valid[:, None] & present[None], copy=True)
            px = enabled[..., None, None] & spatial[None]
            if values.dtype != np.uint8 and (not np.isfinite(values[px]).all()
                                             or (values[px] < 0).any() or (values[px] > 1).any()):
                raise ValueError("Observed intensity must be finite and in [0,1]")
            values[~px] = 0
            value = torch.from_numpy(values.astype(np.float32))
            if values.dtype == np.uint8:
                value /= 255
            h, n = value.shape[:2]
            value = F.interpolate(value.flatten(0, 1)[:, None], target_hw, mode="area")
            # Unmeasured ROI pixels are absent, not zero-valued contact. Divide
            # by their measured area so a partially covered output cell keeps
            # the source scale. A genuinely measured zero still contributes.
            support = F.interpolate(torch.from_numpy(spatial.copy()).float()[:, None],
                                    target_hw, mode="area")
            value = value.reshape(h, n, 1, *target_hw)
            value = torch.where(support[None] > 0, value / support[None].clamp_min(1e-12), 0)
            return value, torch.from_numpy(enabled)

        history, history_valid = gather(hc, hv, (self.size, self.size))
        future, future_valid = gather(fi, fv, (self.size, self.size))
        # Any-support keeps small measured cells; never treat unmeasured holes as labels.
        roi = F.interpolate(torch.from_numpy(spatial.copy()).float()[:, None],
                            (self.size, self.size), mode="area") > 0
        result = {
            "tactile_maps": history, "tactile_valid": history_valid,
            "future_tactile_maps": future, "future_tactile_valid": future_valid,
            "tactile_region_names": self.names, "tactile_sensor_names": self.sensor_names,
            "tactile_layout_present": torch.from_numpy(present.copy()),
            "tactile_spatial_valid": roi,
            "tactile_query_mask": torch.from_numpy(np.broadcast_to(present, future_valid.shape).copy()),
            "tactile_codec": INTENSITY_CODEC,
            "tactile_history_times_s": torch.tensor(self.history_times, dtype=torch.float32),
            "future_tactile_times_s": torch.tensor(self.future_times, dtype=torch.float32),
            "tactile_history_source_times_s": torch.tensor(np.where(hv, times[hc] - reference_time, np.nan)),
            "future_tactile_source_times_s": torch.tensor(np.where(fv, times[fi] - reference_time, np.nan)),
            "tactile_native_fps": self.fps,
        }
        if self.observation_hw is not None:
            if self.video_history is not None:
                observations, observation_roi = self.video_history.gather(
                    episode, hc, history_valid.numpy(), times)
            else:
                observations = gather(hc, hv, self.observation_hw)[0]
                observation_roi = F.interpolate(
                    torch.from_numpy(spatial.copy()).float()[:, None], self.observation_hw, mode="area"
                ) > 0
            result.update(
                tactile_observation_shape=self.observation_hw,
                tactile_observation_maps=observations,
                tactile_observation_spatial_valid=observation_roi,
                tactile_encoder_names=list(self.encoder_names),
            )
        return result
