"""Read explicitly aligned, lossless per-episode deform sidecars for LeRobot."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .regions import CODEC, REGION_NAMES, SCHEMA, resize_deform


def causal_match(target_times, source_times, max_age_s):
    """Latest source <= target, never nearest-future; return indices and validity."""
    target, source = np.asarray(target_times, np.float64), np.asarray(source_times, np.float64)
    if (
        target.ndim != 1
        or source.ndim != 1
        or not np.isfinite(target).all()
        or not np.isfinite(source).all()
        or (np.diff(target) <= 0).any()
        or (np.diff(source) <= 0).any()
        or not np.isfinite(max_age_s)
        or max_age_s < 0
    ):
        raise ValueError("Timestamps must be finite, strictly increasing; age must be nonnegative")
    index = np.searchsorted(source, target, side="right") - 1
    if not len(source):
        return index, np.zeros(len(target), dtype=bool)
    age = target - source[index.clip(min=0)]
    return index, (index >= 0) & (age <= max_age_s)


class AlignedDeformStore:
    """One manifest + uint8 [T,N,H,W] maps, bool [T,N] valid, [T] timestamps.

    Arrays are on the dataset frame grid, not the MCAP start-time grid. The
    exporter must retain causal source matching metadata in its provenance.
    No bag decoding, interpolation of missing packets, or label fabrication.
    """

    def __init__(self, manifest_path, dataset_dir, *, size, fps):
        path = Path(manifest_path)
        self.root, self.size = path.resolve().parent, size
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.meta = json.loads(path.read_text())
        info_digest = hashlib.sha256((Path(dataset_dir) / "meta/info.json").read_bytes()).hexdigest()
        if (
            self.meta.get("schema") != SCHEMA
            or self.meta.get("codec") != CODEC
            or self.meta.get("dataset_info_sha256") != info_digest
            or self.meta.get("fps") != fps
        ):
            raise ValueError("Deform sidecar schema/codec/dataset identity/fps mismatch")
        self.fps = fps
        self.names = list(self.meta["region_names"])
        if "surface_ids" in self.meta or "tactile_surface_ids" in self.meta:
            raise ValueError("surface_id was removed; rebuild the sidecar with one map per region")
        if not self.names or set(self.names) - set(REGION_NAMES) or len(set(self.names)) != len(self.names):
            raise ValueError("Invalid or duplicate sidecar region identities; compose native sensors first")
        self.sensor_ids = np.asarray(self.meta["sensor_ids"], dtype=np.int64)
        if self.sensor_ids.shape != (len(self.names),) or (self.sensor_ids != 0).any():
            raise ValueError("This real Sharpa sidecar requires sensor_id=0 for each region")
        self.episodes = self.meta["episodes"]

    def _path(self, entry, name):
        path = (self.root / entry[name]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Sidecar paths must remain inside the manifest directory")
        return path

    @lru_cache(maxsize=4)
    def arrays(self, episode):
        entry = self.episodes[str(int(episode))]
        maps = np.load(self._path(entry, "maps"), mmap_mode="r", allow_pickle=False)
        valid = np.load(self._path(entry, "valid"), mmap_mode="r", allow_pickle=False)
        times = np.load(self._path(entry, "timestamps"), mmap_mode="r", allow_pickle=False)
        present = np.asarray(entry["layout_present"])
        if (
            maps.dtype != np.uint8
            or maps.ndim != 4
            or maps.shape[1] != len(self.names)
            or min(maps.shape) < 1
            or valid.dtype != np.bool_
            or valid.shape != maps.shape[:2]
            or times.shape != (maps.shape[0],)
            or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any()
            or present.dtype != np.bool_
            or present.shape != (len(self.names),)
        ):
            raise ValueError(f"Invalid deform episode arrays: {episode}")
        if (valid & ~present[None]).any():
            raise ValueError("Uninstalled regions cannot contain valid observations")
        if not np.allclose(np.diff(times), 1 / self.fps, rtol=0, atol=2e-5):
            raise ValueError("Sidecar is not on the dataset frame grid")
        return maps, valid, times, present

    def window(
        self, episode, start, segment_start, segment_end, reference_time, *, history, future_frames, future_stride
    ):
        maps, valid, times, present = self.arrays(episode)
        offsets = np.arange(1, future_frames + 1) * future_stride
        future_rows = start + offsets
        if (
            start < segment_start
            or future_rows[-1] >= min(segment_end, len(maps))
            or not np.isclose(times[start], reference_time, rtol=0, atol=2e-5)
            or not present.any()
            or not valid[future_rows][:, present].all()
        ):
            raise ValueError("Incomplete or misaligned real future tactile window")
        n = len(self.names)
        hm = torch.zeros(history, n, 1, self.size, self.size)
        hv = torch.zeros(history, n, dtype=torch.bool)
        fm = torch.zeros(future_frames, n, 1, self.size, self.size)
        fv = torch.from_numpy(np.array(valid[future_rows], copy=True))
        for t, row in enumerate(range(start - history + 1, start + 1)):
            if row >= segment_start:
                for r in np.flatnonzero(valid[row]):
                    hm[t, r] = resize_deform(np.array(maps[row, r], copy=True), self.size)
                    hv[t, r] = True
        for t, row in enumerate(future_rows):
            for r in np.flatnonzero(valid[row]):
                fm[t, r] = resize_deform(np.array(maps[row, r], copy=True), self.size)
        return dict(
            tactile_maps=hm,
            tactile_valid=hv,
            future_tactile_maps=fm,
            future_tactile_valid=fv,
            tactile_region_names=self.names,
            tactile_layout_present=torch.from_numpy(present.copy()),
            tactile_sensor_names=["sharpa_real"] * len(self.names),
            tactile_codec=CODEC,
            tactile_fps=self.fps,
            future_tactile_offsets=torch.from_numpy(offsets),
        )
