"""Strict regional optical RGB reads from already published LeRobot v3 videos.

RGB is appearance, not calibrated pressure. Missing packets are represented by
validity metadata; pixel brightness never decides whether a packet exists.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, parse_info_json
from openwam.dataloader.utils.video_io import decode_video_frames
from .intensity_store import FUTURE_TIMES, HISTORY_TIMES, _time_grid
from .registry import REGION_NAMES
from .regions import RGB_CODEC


@lru_cache(maxsize=4)
def _clock_and_validity(path, columns):
    return pq.read_table(path, columns=list(columns))


@lru_cache(maxsize=128)
def _video_header(path):
    import av

    with av.open(path) as container:
        if len(container.streams.video) != 1:
            raise ValueError("RGB tactile publication requires one indexed video stream per feature")
        stream = container.streams.video[0]
        return stream.height, stream.width, float(stream.average_rate or 0), int(stream.frames)


def _text(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RGB tactile contract requires explicit {key}")
    return value


class RGBTactileStore:
    """One verified, uniquely identified RGB image per public tactile region.

    The first version deliberately requires row-aligned RGB videos at the
    dataset's native FPS. Different source rates require an explicit offline
    alignment contract rather than an implicit frame-index conversion.
    """

    def __init__(self, manifest_path, dataset_dir, *, size=128,
                 history_times=HISTORY_TIMES, future_times=FUTURE_TIMES):
        self.root = Path(dataset_dir).resolve(strict=True)
        self.path = Path(manifest_path).resolve(strict=True)
        if not self.path.is_relative_to(self.root):
            raise ValueError("RGB manifest must belong to the published dataset")
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.meta = json.loads(self.path.read_text())
        self.info = parse_info_json(self.root)
        self.history_times = _time_grid(history_times, future=False)
        self.future_times = _time_grid(future_times, future=True)
        if type(size) is not int or size < 1:
            raise ValueError("RGB map size must be a positive integer")
        self.size = size
        info_hash = hashlib.sha256((self.root / "meta/info.json").read_bytes()).hexdigest()
        if (self.meta.get("schema") != RGB_CODEC or self.meta.get("codec") != RGB_CODEC
                or self.meta.get("representation") != "optical_rgb"
                or self.meta.get("normalization") != "uint8_div_255"
                or self.meta.get("time_alignment") != "dataset_rows"
                or self.meta.get("dataset_info_sha256") != info_hash):
            raise ValueError("RGB tactile schema/representation/time/info fingerprint mismatch")
        proof = self.meta.get("verification", {})
        if (self.meta.get("status") != "verified" or not isinstance(proof.get("evidence"), list)
                or not proof["evidence"] or any(not isinstance(e, str) or not e.strip() for e in proof["evidence"])):
            raise ValueError("RGB geometry, source and clock mapping require verification evidence")
        definition = self.root / "meta/source_definition.json"
        source = json.loads(definition.read_text()).get("source_id", self.root.name) if definition.exists() else self.root.name
        if _text(self.meta, "source_id") != source:
            raise ValueError("RGB tactile source identity differs from the dataset")
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("RGB tactile requires published LeRobot v3 media")
        self.fps = float(self.info["fps"])
        self.max_age = float(self.meta.get("history_max_age_s", 1.5 / self.fps)) if self.fps > 0 else -1
        self.future_tolerance = float(self.meta.get("future_tolerance_s", .51 / self.fps)) if self.fps > 0 else -1
        if (not np.isfinite([self.fps, self.max_age, self.future_tolerance]).all()
                or self.fps <= 0 or self.max_age < 0 or self.future_tolerance < 0):
            raise ValueError("Invalid RGB native FPS or time matching tolerance")
        self.regions = self.meta.get("regions")
        if not isinstance(self.regions, list) or not self.regions:
            raise ValueError("RGB tactile requires explicit regional video mappings")
        self.names = [_text(entry, "region_name") for entry in self.regions]
        self.sensor_names = [_text(entry, "sensor_name") for entry in self.regions]
        if len(set(self.names)) != len(self.names) or set(self.names) - set(REGION_NAMES):
            raise ValueError("RGB regions must be unique members of the public registry")
        if any("surface_id" in entry or "surface_ids" in entry for entry in self.regions):
            raise ValueError("Compose native sensors into one image per region; surface_id is unsupported")
        self._native_roi, self._support, self._spatial = [], [], []
        for entry in self.regions:
            video = _text(entry, "video_key")
            validity = _text(entry, "validity_column")
            feature = self.info.get("features", {}).get(video, {})
            shape = feature.get("shape", [])
            if (not video.startswith("observation.tactile_rgb.") or feature.get("dtype") != "video"
                    or len(shape) != 3 or shape[2] != 3 or any(type(v) is not int or v < 1 for v in shape)
                    or feature.get("info", {}).get("video.is_depth_map", False)):
                raise ValueError("RGB codec requires a native three-channel tactile RGB video feature")
            fps = feature.get("info", {}).get("video.fps")
            if fps is None or not np.isclose(float(fps), self.fps, atol=1e-8, rtol=0):
                raise ValueError("RGB tactile video FPS must match the verified dataset row clock")
            mask_feature = self.info.get("features", {}).get(validity, {})
            if mask_feature.get("dtype") != "bool" or mask_feature.get("shape") != [1]:
                raise ValueError("RGB tactile requires an explicit scalar boolean validity feature")
            orientation = entry.get("orientation", {})
            turn = orientation.get("rotate_quadrants")
            if (type(turn) is not int or turn not in range(4)
                    or any(type(orientation.get(k)) is not bool for k in ("flip_horizontal", "flip_vertical"))):
                raise ValueError("RGB orientation requires rotation 0..3 and explicit horizontal/vertical flips")
            h, w = shape[:2]
            roi = entry.get("roi", {})
            if roi.get("type") == "full":
                valid_pixels = np.ones((h, w), dtype=bool)
            elif roi.get("type") == "rectangle":
                bounds = roi.get("bounds", [])
                if (not isinstance(bounds, list) or len(bounds) != 4 or any(type(v) is not int for v in bounds)
                        or not (0 <= bounds[0] < bounds[2] <= w and 0 <= bounds[1] < bounds[3] <= h)):
                    raise ValueError("RGB ROI rectangle must be an in-bounds nonempty [x0,y0,x1,y1]")
                valid_pixels = np.zeros((h, w), dtype=bool)
                valid_pixels[bounds[1]:bounds[3], bounds[0]:bounds[2]] = True
            elif roi.get("type") == "mask":
                mask_path = self._inside(self.root / _text(roi, "path"))
                if hashlib.sha256(mask_path.read_bytes()).hexdigest() != roi.get("sha256"):
                    raise ValueError("RGB spatial ROI mask fingerprint mismatch")
                valid_pixels = np.load(mask_path, allow_pickle=False)
                if valid_pixels.dtype != np.bool_ or valid_pixels.shape != (h, w) or not valid_pixels.any():
                    raise ValueError("RGB spatial ROI must be a nonempty bool mask in native image coordinates")
            else:
                raise ValueError("RGB tactile requires explicit full or rectangle or mask ROI")
            valid_pixels = self._orient(valid_pixels, entry)
            self._native_roi.append(torch.from_numpy(valid_pixels.copy())[None])
            support = F.interpolate(self._native_roi[-1][None].float(), (size, size), mode="area")[0]
            self._support.append(support)
            self._spatial.append(support > 0)
        self.spatial = torch.stack(self._spatial)
        metadata = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        required = {"episode_index", "length", "dataset_from_index", "data/chunk_index", "data/file_index"}
        for entry in self.regions:
            required |= {f"videos/{entry['video_key']}/{field}" for field in
                         ("chunk_index", "file_index", "from_timestamp", "to_timestamp")}
        tables = []
        for path in metadata:
            if required - set(pq.read_schema(path).names):
                raise ValueError("RGB publication lacks episode-specific video offsets")
            tables.append(pq.read_table(path, columns=sorted(required)))
        if not tables:
            raise ValueError("RGB publication has no episode metadata")
        eps = pa.concat_tables(tables).to_pandas().sort_values("episode_index").reset_index(drop=True)
        if eps.episode_index.duplicated().any() or (eps.length <= 0).any():
            raise ValueError("RGB publication contains duplicated or empty episodes")
        eps["_data_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
        self._episode_rows = {str(int(row.episode_index)): row for _, row in eps.iterrows()}
        self.episodes = self.meta.get("episodes", {})
        if not isinstance(self.episodes, dict) or set(self.episodes) != set(self._episode_rows):
            raise ValueError("RGB layout must explicitly cover exactly the published episode population")
        self._columns = tuple(sorted({"timestamp", "episode_index"} | {e["validity_column"] for e in self.regions}))
        checked = set()
        for key, row in self._episode_rows.items():
            present = self.episodes[key].get("layout_present")
            if not isinstance(present, list) or len(present) != len(self.names) or any(type(v) is not bool for v in present):
                raise ValueError("RGB episode layout_present must be an explicit boolean regional vector")
            path = self._data_path(row)
            if path not in checked:
                schema = pq.read_schema(path)
                if set(self._columns) - set(schema.names):
                    raise ValueError("RGB publication lacks row clocks or tactile validity columns")
                if any(not pa.types.is_boolean(schema.field(e["validity_column"]).type) for e in self.regions):
                    raise ValueError("RGB validity columns must be scalar boolean arrays")
                checked.add(path)
            for entry in self.regions:
                prefix = f"videos/{entry['video_key']}/"
                start, end = float(row[prefix + "from_timestamp"]), float(row[prefix + "to_timestamp"])
                if (not np.isfinite([start, end]).all() or start < 0
                        or not np.isclose(start * self.fps, round(start * self.fps), atol=2e-3, rtol=0)
                        or not np.isclose(end - start, int(row.length) / self.fps, atol=2e-5, rtol=0)):
                    raise ValueError("RGB episode video boundaries disagree with the row clock")
                video_path = self._video_path(row, entry)
                h, w, fps, frames = _video_header(str(video_path))
                expected_shape = self.info["features"][entry["video_key"]]["shape"]
                if (expected_shape[:2] != [h, w] or not np.isclose(fps, self.fps, atol=1e-6, rtol=0)
                        or frames < round(start * self.fps) + int(row.length)):
                    raise ValueError("RGB video media dimensions/FPS/frame inventory disagree with its publication")

    def _inside(self, path):
        path = path.resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("RGB media path escapes its published dataset")
        return path

    def _data_path(self, row):
        return self._inside(self.root / self.info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])))

    def _video_path(self, row, entry):
        prefix = f"videos/{entry['video_key']}/"
        return self._inside(self.root / self.info["video_path"].format(video_key=entry["video_key"],
            chunk_index=int(row[prefix + "chunk_index"]), file_index=int(row[prefix + "file_index"])))

    @staticmethod
    def _orient(values, entry):
        orientation = entry["orientation"]
        values = np.rot90(values, orientation["rotate_quadrants"], axes=(0, 1))
        if orientation["flip_horizontal"]:
            values = np.flip(values, axis=1)
        if orientation["flip_vertical"]:
            values = np.flip(values, axis=0)
        return values

    def window(self, episode, reference_time, *, segment_start_s=None, segment_end_s=None):
        key = str(int(episode))
        row = self._episode_rows[key]
        length = int(row.length)
        table = _clock_and_validity(str(self._data_path(row)), self._columns).slice(int(row["_data_offset"]), length)
        recorded = np.asarray(table["timestamp"], dtype=np.float64)
        if (len(table) != length or not np.isfinite(recorded).all() or (np.diff(recorded) <= 0).any()
                or not np.all(np.asarray(table["episode_index"]) == int(episode))):
            raise ValueError("RGB rows cross an episode boundary or have invalid timestamps")
        times = recorded[0] + np.arange(length, dtype=np.float64) / self.fps
        if not np.allclose(recorded, times, atol=max(2e-5, 1e-3 / self.fps), rtol=0):
            raise ValueError("RGB tactile needs a verified regular row clock; resample irregular media offline")
        start = times[0] if segment_start_s is None else max(times[0], segment_start_s)
        end = times[-1] if segment_end_s is None else min(times[-1], segment_end_s)
        if not np.isfinite(reference_time) or not start - 1e-8 <= reference_time <= end + 1e-8:
            raise ValueError("RGB reference time is outside its episode/segment")
        ht, ft = reference_time + self.history_times, reference_time + self.future_times
        hi = np.searchsorted(times, ht + 1e-8, side="right") - 1
        hc = hi.clip(0, length - 1)
        hv = ((hi >= 0) & (times[hc] >= start - 1e-8) & (times[hc] <= ht + 1e-8)
              & (ht - times[hc] <= self.max_age + 1e-8))
        right = np.searchsorted(times, ft).clip(0, length - 1)
        left = (right - 1).clip(0)
        fi = np.where(np.abs(times[left] - ft) <= np.abs(times[right] - ft), left, right)
        fv = ((times[fi] > reference_time + 1e-8) & (times[fi] <= end + 1e-8)
              & (np.abs(times[fi] - ft) <= self.future_tolerance + 1e-8) & (ft <= end + 1e-8))
        indices, time_valid = np.r_[hc, fi], np.r_[hv, fv]
        present = np.asarray(self.episodes[key]["layout_present"], dtype=bool)
        valid = np.zeros((len(indices), len(self.names)), dtype=bool)
        values = torch.zeros(len(indices), len(self.names), 3, self.size, self.size)
        for n, entry in enumerate(self.regions):
            measured = np.asarray(table[entry["validity_column"]].fill_null(False), dtype=bool)
            if not present[n] and measured.any():
                raise ValueError("RGB validity labels a sensor absent from the episode layout")
            enabled = time_valid & present[n] & measured[indices]
            valid[:, n] = enabled
            wanted = sorted(set(indices[enabled].tolist()))
            if not wanted:
                continue
            shape = self.info["features"][entry["video_key"]]["shape"]
            first = round(float(row[f"videos/{entry['video_key']}/from_timestamp"]) * self.fps)
            frames = decode_video_frames(str(self._video_path(row, entry)), [first + i for i in wanted], *shape[:2])
            converted = {}
            for frame_index, image in zip(wanted, frames):
                raw = np.asarray(image)
                if raw.dtype != np.uint8 or raw.shape != tuple(shape):
                    raise ValueError("RGB tactile decoding must preserve native uint8 three-channel images")
                tensor = torch.from_numpy(self._orient(raw, entry).copy()).permute(2, 0, 1).float() / 255
                mask = self._native_roi[n]
                support = self._support[n]
                resized = F.interpolate((tensor * mask)[None], (self.size, self.size), mode="area")[0]
                converted[frame_index] = torch.where(support > 0, resized / support.clamp_min(1e-12), 0)
            for i in np.flatnonzero(enabled):
                values[i, n] = converted[int(indices[i])]
        h = len(hc)
        return dict(
            tactile_maps=values[:h], tactile_valid=torch.from_numpy(valid[:h].copy()),
            future_tactile_maps=values[h:], future_tactile_valid=torch.from_numpy(valid[h:].copy()),
            tactile_region_names=self.names, tactile_sensor_names=self.sensor_names,
            tactile_layout_present=torch.from_numpy(present.copy()), tactile_spatial_valid=self.spatial.clone(),
            tactile_query_mask=torch.from_numpy(np.broadcast_to(present, valid[h:].shape).copy()),
            tactile_codec=RGB_CODEC, tactile_history_times_s=torch.tensor(self.history_times, dtype=torch.float32),
            future_tactile_times_s=torch.tensor(self.future_times, dtype=torch.float32),
            tactile_history_source_times_s=torch.tensor(np.where(hv, times[hc] - reference_time, np.nan)),
            future_tactile_source_times_s=torch.tensor(np.where(fv, times[fi] - reference_time, np.nan)),
            tactile_native_fps=self.fps,
        )
