"""Native Sharpa deform history, independently of the 32px prediction cache."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = "sharpa_deform_video_history_v1"
KEY = "observation.images.tactile_deform"
REGIONS = [f"{side}.{finger}.tip" for side in ("left", "right")
           for finger in ("thumb", "index", "middle", "ring", "pinky")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)) or not path.is_file():
        raise ValueError("Sharpa history media escapes its published dataset")
    return path


def decode_gray_frames(path, indices, fps):
    """Seek once, decode exact source PTS; never substitute a neighboring frame."""
    import av

    targets = set(map(int, indices))
    if not targets:
        return {}
    if min(targets) < 0:
        raise ValueError("Negative Sharpa source frame")
    found = {}
    with av.open(str(path), options={"hwaccel": "none"}) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Sharpa deform requires exactly one video stream")
        stream = container.streams.video[0]
        stream.thread_count = 1
        if ((stream.height, stream.width) != (480, 1200)
                or not np.isclose(float(stream.average_rate or 0), fps, atol=1e-6, rtol=0)):
            raise ValueError("Sharpa deform video shape/FPS changed")
        if min(targets):
            container.seek(max(0, int((min(targets) / fps) / stream.time_base)),
                           stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Sharpa deform frame has no physical PTS")
            position = float(frame.pts * stream.time_base) * fps
            index = round(position)
            if abs(position - index) > 2e-3:
                raise ValueError("Sharpa deform PTS differs from its verified frame grid")
            if index in targets:
                found[index] = frame.to_ndarray(format="gray")
            if index >= max(targets):
                break
    if targets - found.keys():
        raise ValueError(f"Missing valid Sharpa deform frames: {sorted(targets - found.keys())}")
    return found


class SharpaVideoHistory:
    def __init__(self, manifest_path, dataset_dir, tactile_path, tactile_meta, observation_hw):
        self.path = Path(manifest_path).resolve(strict=True)
        self.digest = sha256(self.path)
        self.meta = json.loads(self.path.read_text())
        self.root = Path(dataset_dir).resolve(strict=True)
        self.hw = tuple(observation_hw)
        m = self.meta
        if (m.get("schema") != SCHEMA or m.get("dataset_info_sha256") != sha256(self.root / "meta/info.json")
                or m.get("tactile_manifest_sha256") != sha256(tactile_path)
                or m.get("region_names") != REGIONS or tactile_meta.get("region_names") != REGIONS
                or tactile_meta.get("sensor_names") != ["sharpa_real"] * 10
                or m.get("encoder_name") != "SharpaWave" or m.get("video_key") != KEY
                or m.get("mosaic_shape") != [480, 1200] or m.get("tile_shape") != [240, 240]
                or m.get("normalization") != "pyav_gray_uint8_div_255"
                or m.get("resize") != "opencv_inter_area"
                or m.get("orientation") != "top_left_bottom_right_thumb_to_pinky"):
            raise ValueError("Sharpa native history publication identity/geometry mismatch")
        if self.hw != (224, 224):
            raise ValueError("This Sharpa native history release requires 224x224 observations")
        self.fps = float(m["fps"])
        if not np.isfinite(self.fps) or self.fps != 30 or self.fps != tactile_meta["fps"]:
            raise ValueError("Sharpa native history clock mismatch")
        self.episodes = m["episodes"]
        if set(self.episodes) != set(tactile_meta["episodes"]):
            raise ValueError("Sharpa native history episode population mismatch")
        for key, entry in self.episodes.items():
            old = tactile_meta["episodes"][key]
            offset = float(entry["from_timestamp"])
            if (entry["video"] != old["source_video"] or entry["frames"] != old["frames"]
                    or not np.isfinite(offset) or offset < 0
                    or abs(offset - old["source_video_from_timestamp"]) > 1e-8
                    or abs(offset * self.fps - round(offset * self.fps)) > 2e-3
                    or entry["video"] not in m["videos"]):
                raise ValueError("Sharpa native history differs from original episode video offsets")
        # Stat identity detects ordinary file replacement/writes, not payload bit rot.
        for relative in m["videos"]:
            self._video(relative)

    def _video(self, relative):
        path = inside(self.root, relative)
        st = path.stat()
        expected = self.meta["videos"][relative]
        if st.st_size != expected["bytes"] or st.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Sharpa native history video changed after publication")
        return path

    def gather(self, episode, rows, enabled, times):
        import cv2

        entry = self.episodes[str(int(episode))]
        if (np.asarray(rows) < 0).any() or (np.asarray(rows) >= entry["frames"]).any():
            raise ValueError("Sharpa history selection crosses an episode boundary")
        if not np.allclose(times[rows], np.asarray(rows) / self.fps, atol=2e-5, rtol=0):
            raise ValueError("Sharpa history cache clock differs from source video")
        indices = round(entry["from_timestamp"] * self.fps) + np.asarray(rows)
        requested = indices[np.asarray(enabled).any(axis=1)]
        decoded = decode_gray_frames(self._video(entry["video"]), requested, self.fps)
        values = np.zeros((len(rows), 10, 1, *self.hw), dtype=np.float32)
        for i, index in enumerate(indices):
            if not enabled[i].any():
                continue
            gray = decoded[int(index)]
            for region in np.flatnonzero(enabled[i]):
                y, x = divmod(int(region), 5)
                tile = gray[y * 240:(y + 1) * 240, x * 240:(x + 1) * 240]
                values[i, region, 0] = cv2.resize(tile, self.hw[::-1], interpolation=cv2.INTER_AREA)
        values *= 1 / 255
        return torch.from_numpy(values), torch.ones((10, 1, *self.hw), dtype=torch.bool)
