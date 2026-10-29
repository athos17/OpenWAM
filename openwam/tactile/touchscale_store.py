"""Native asynchronous TouchScale normal-force matrices; no robot targets."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .intensity_store import HISTORY_TIMES, FUTURE_TIMES, INTENSITY_CODEC, _time_grid
from .registry import REGION_NAMES

SCHEMA = "touchscale_native_force_v1"
ENCODER = "TouchScaleForce"
SENSOR = "touchscale_glove"
SHAPE = (16, 10)


class TouchScaleForceStore:
    def __init__(self, manifest_path, dataset_dir, *, size=32, history_times=HISTORY_TIMES,
                 future_times=FUTURE_TIMES, observation_size=None, encoder_name=None):
        self.path = Path(manifest_path).resolve(strict=True)
        self.root = self.path.parent
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.meta = json.loads(self.path.read_text())
        m = self.meta
        if (m.get("schema") != SCHEMA or m.get("codec") != INTENSITY_CODEC
                or m.get("dataset_info_sha256") != hashlib.sha256((Path(dataset_dir) / "meta/info.json").read_bytes()).hexdigest()
                or m.get("normalization") != "max(force_N,0)/(max(force_N,0)+1N)"
                or m.get("encoder_name") != ENCODER or encoder_name != ENCODER
                or tuple(observation_size or ()) != SHAPE or size != 32):
            raise ValueError("TouchScale force publication/encoder/normalization mismatch")
        self.layout = m["layout"]
        self.names = [r["region"] for r in self.layout]
        if (len(self.names) != 30 or len(set(self.names)) != 30 or set(self.names) - set(REGION_NAMES)
                or [r["side"] for r in self.layout] != ["right"] * 15 + ["left"] * 15):
            raise ValueError("TouchScale requires the complete ordered bimanual native layout")
        for side in ("right", "left"):
            offset = 0
            for r in self.layout:
                if r["side"] != side:
                    continue
                h, w = r["native_shape"]
                if not (0 < h <= 16 and 0 < w <= 10 and r["offset"] == offset):
                    raise ValueError("TouchScale native taxel geometry mismatch")
                offset += h * w
            if offset != 880:
                raise ValueError("TouchScale hand must contain 880 native taxels")
        self.observation_hw = SHAPE
        self.episodes = m["episodes"]
        self.history_times = _time_grid(history_times, future=False)
        self.future_times = _time_grid(future_times, future=True)
        self.size = size
        self.spatial = np.zeros((30, 1, *SHAPE), dtype=bool)
        for i, r in enumerate(self.layout):
            h, w = r["native_shape"]
            self.spatial[i, 0, :h, :w] = True

    def _path(self, value):
        p = (self.root / value).resolve(strict=True)
        if not p.is_relative_to(self.root):
            raise ValueError("TouchScale array escaped publication")
        return p

    @lru_cache(maxsize=4)
    def arrays(self, episode, side):
        e = self.episodes[str(int(episode))][side]
        force = np.load(self._path(e["force"]), mmap_mode="r", allow_pickle=False)
        times = np.load(self._path(e["timestamps"]), mmap_mode="r", allow_pickle=False)
        if (force.dtype != np.float32 or force.shape != (len(times), 880) or len(times) < 2
                or times.dtype != np.float64 or not np.isfinite(times).all() or (np.diff(times) <= 0).any()):
            raise ValueError("Invalid native TouchScale force/clock arrays")
        return force, times

    def window(self, episode, reference_time, *, segment_start_s=None, segment_end_s=None):
        ep = self.episodes[str(int(episode))]
        start = max(0., segment_start_s or 0.)
        end = min(ep["duration_s"] - 1/30, segment_end_s if segment_end_s is not None else np.inf)
        if not np.isfinite(reference_time) or not start - 1e-7 <= reference_time <= end + 1e-7:
            raise ValueError("TouchScale reference outside episode")

        def gather(offsets, future):
            target = reference_time + offsets
            maps = np.zeros((len(offsets), 30, 1, *SHAPE), np.float32)
            valid = np.zeros((len(offsets), 30), bool)
            source_times = np.full((len(offsets), 30), np.nan, np.float64)
            for side in ("right", "left"):
                force, times = self.arrays(episode, side)
                if future:
                    right = np.searchsorted(times, target).clip(0, len(times)-1)
                    left = (right - 1).clip(0)
                    rows = np.where(abs(times[left]-target) <= abs(times[right]-target), left, right)
                    enabled = (abs(times[rows]-target) <= .025) & (times[rows] > reference_time + 1e-8)
                else:
                    raw = np.searchsorted(times, target + 1e-8, side="right")-1
                    rows = raw.clip(0, len(times)-1)
                    enabled = (raw >= 0) & (times[rows] <= target + 1e-8) & (target-times[rows] <= .05)
                enabled &= (target >= start-1e-8) & (target <= end+1e-8)
                enabled &= (times[rows] >= start-1e-8) & (times[rows] <= end+1e-8)
                for i, r in enumerate(self.layout):
                    if r["side"] != side:
                        continue
                    h, w = r["native_shape"]
                    values = np.asarray(force[rows, r["offset"]:r["offset"]+h*w])
                    good = enabled & np.isfinite(values).all(-1)
                    positive = np.maximum(np.where(good[:, None], values, 0), 0)
                    maps[:, i, 0, :h, :w] = (positive/(positive+1)).reshape(-1,h,w)
                    valid[:, i] = good
                    source_times[:, i] = np.where(good, times[rows]-reference_time, np.nan)
            return torch.from_numpy(maps), torch.from_numpy(valid), torch.from_numpy(source_times)

        history, hv, ht = gather(self.history_times, False)
        future, fv, ft = gather(self.future_times, True)
        spatial = torch.from_numpy(self.spatial.copy())
        support = F.interpolate(spatial.float(), (32,32), mode="area")
        def targets(x):
            y = F.interpolate(x.flatten(0,1), (32,32), mode="area").reshape(len(x),30,1,32,32)
            return torch.where(support > 0, y/support.clamp_min(1e-12), 0)
        return dict(tactile_maps=targets(history), tactile_valid=hv,
            future_tactile_maps=targets(future), future_tactile_valid=fv,
            tactile_observation_maps=history, tactile_observation_shape=SHAPE,
            tactile_observation_spatial_valid=spatial, tactile_spatial_valid=support>0,
            tactile_region_names=self.names, tactile_sensor_names=[SENSOR]*30,
            tactile_encoder_names=[ENCODER]*30, tactile_layout_present=torch.ones(30,dtype=torch.bool),
            tactile_query_mask=torch.ones((len(self.future_times),30),dtype=torch.bool),
            tactile_codec=INTENSITY_CODEC, tactile_representation="normalized_normal_force",
            tactile_history_times_s=torch.tensor(self.history_times,dtype=torch.float32),
            future_tactile_times_s=torch.tensor(self.future_times,dtype=torch.float32),
            tactile_history_source_times_s=ht, future_tactile_source_times_s=ft)
