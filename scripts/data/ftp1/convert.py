#!/usr/bin/env python3
"""Stream the FTP-1 tar release to native LeRobot v3 buckets.

Each source/task is a bucket; every original episode and numeric field is retained.
Optical tactile is lossless RGB video, not an invented pressure map. No commands
are inferred from state. Missing Zarr chunks produce explicit validity columns.
The source archives are never extracted or modified. Completed videos and tasks
are restartable; source fingerprint and conversion options must match.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import re
import subprocess
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numcodecs import get_codec

VERSION = "ftp1-lrv3-1"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def log(event, **kw):
    print(json.dumps({"time": time.time(), "event": event, **kw}), flush=True)


class Parts(io.RawIOBase):
    def __init__(self, files):
        self.files, self.ends, self.handles, self.pos = files, [], {}, 0
        total = 0
        for path in files:
            total += path.stat().st_size
            self.ends.append(total)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.ends[-1] + offset
        if self.pos < 0:
            raise ValueError("Negative archive offset")
        return self.pos

    def read(self, size=-1):
        if size < 0:
            size = self.ends[-1] - self.pos
        chunks = []
        while size and self.pos < self.ends[-1]:
            i = bisect.bisect_right(self.ends, self.pos)
            start = 0 if i == 0 else self.ends[i - 1]
            if i not in self.handles:
                self.handles[i] = self.files[i].open("rb")
            handle = self.handles[i]
            handle.seek(self.pos - start)
            chunk = handle.read(min(size, self.ends[i] - self.pos))
            if not chunk:
                break
            chunks.append(chunk)
            self.pos += len(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        super().close()


def index_group(files, group, path, archive_factory=None, source_fingerprint=None):
    fingerprint = hashlib.sha256(
        json.dumps([(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]).encode()
    ).hexdigest()
    fingerprint = source_fingerprint or fingerprint
    archive_factory = archive_factory or (lambda: Parts(files))
    if path.exists():
        out = json.loads(path.read_text())
        if out["archive_fingerprint"] != fingerprint:
            raise ValueError(f"Source changed: {path}")
        return out
    out = {"archive_fingerprint": fingerprint, "arrays": {}}
    prefix = group["group"] + "/"
    with archive_factory() as f:
        f.seek(group["offset"])
        with tarfile.open(fileobj=f, mode="r:") as tar:
            while True:
                member = tar.next()
                if member is None:
                    break
                tar.members.clear()  # discard TarInfo cache; our compact index is sufficient
                if member.offset >= group["end_offset"]:
                    break
                if not member.name.startswith(prefix) or not member.isfile():
                    continue
                rel = member.name[len(prefix) :]
                if rel.endswith("/.zarray"):
                    key = rel[: -len("/.zarray")]
                    cfg = json.loads(tar.extractfile(member).read())
                    out["arrays"].setdefault(key, {})["config"] = cfg
                elif "/" in rel and not rel.rsplit("/", 1)[1].startswith("."):
                    key, chunk = rel.rsplit("/", 1)
                    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", chunk):
                        out["arrays"].setdefault(key, {}).setdefault("chunks", {})[chunk] = [
                            member.offset_data,
                            member.size,
                        ]
    for key, value in out["arrays"].items():
        if "config" not in value:
            raise ValueError(f"Missing Zarr metadata: {group['group']}/{key}")
        value.setdefault("chunks", {})
    atomic_json(path, out)
    return out


def resolve_selection(group):
    """Validate a pilot census without changing the immutable source census."""
    original = group.get("source_census", group)
    ends = [int(x) for x in original["episode_ends"]]
    if not ends or any(b <= a for a, b in zip([0] + ends[:-1], ends)) or ends[-1] != original["frames"]:
        raise ValueError("Invalid original episode boundaries")
    ids = group.get("selected_source_episode_indices", list(range(len(ends))))
    if not ids or ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < len(ends) for i in ids):
        raise ValueError("Selected episode IDs must be unique sorted original episode indices")
    ranges = [[0 if i == 0 else ends[i - 1], ends[i]] for i in ids]
    lengths = [b - a for a, b in ranges]
    if "source_census" in group:
        if group.get("selection_schema") != "touchwam.ftp1.episode_selection.v1":
            raise ValueError("Unknown episode selection schema")
        for key in ("group", "offset", "end_offset"):
            if group[key] != original[key]:
                raise ValueError("Pilot/source census identity mismatch")
        if group["frames"] != sum(lengths) or group["episode_ends"] != np.cumsum(lengths).tolist():
            raise ValueError("Pilot frame counts do not match selected source episodes")
        if group.get("episodes") != len(ids):
            raise ValueError("Pilot episode count mismatch")
    return original, ranges, ids


class Array:
    """Bounded-memory Zarr v2 array decoder over indexed tar members."""

    def __init__(self, stream, spec, frame_ranges=None):
        self.stream, self.spec = stream, spec
        self.cfg = spec["config"]
        self.source_shape = tuple(self.cfg["shape"])
        self.frame_ranges = frame_ranges or [[0, self.source_shape[0]]]
        if any(a < 0 or b <= a or b > self.source_shape[0] for a, b in self.frame_ranges):
            raise ValueError("Selected frames outside source array")
        self.shape = (sum(b - a for a, b in self.frame_ranges),) + self.source_shape[1:]
        self.chunk_shape = tuple(self.cfg["chunks"])
        self.dtype = np.dtype(self.cfg["dtype"])
        self.codec = get_codec(self.cfg["compressor"]) if self.cfg.get("compressor") else None
        if self.cfg.get("filters"):
            raise ValueError("Unexpected Zarr filters: explicit support required")
        if self.chunk_shape[1:] != self.shape[1:]:
            raise ValueError(f"Non-time-only chunking requires separate decoder: {self.shape}/{self.chunk_shape}")

    def blocks(self):
        out_begin, previous_chunk, previous_data, previous_present = 0, None, None, None
        for start, stop in self.frame_ranges:
            for begin in range(start // self.chunk_shape[0] * self.chunk_shape[0], stop, self.chunk_shape[0]):
                chunk_id = ".".join([str(begin // self.chunk_shape[0])] + ["0"] * (len(self.shape) - 1))
                if chunk_id != previous_chunk:
                    entry = self.spec["chunks"].get(chunk_id)
                    n = min(self.chunk_shape[0], self.source_shape[0] - begin)
                    if entry is None:
                        data = np.full((n,) + self.shape[1:], self.cfg.get("fill_value", 0), dtype=self.dtype)
                    else:
                        self.stream.seek(entry[0])
                        raw = self.stream.read(entry[1])
                        if len(raw) != entry[1]:
                            raise EOFError(f"Short Zarr chunk {chunk_id}")
                        raw = self.codec.decode(raw) if self.codec else raw
                        data = np.frombuffer(raw, dtype=self.dtype).reshape(
                            self.chunk_shape, order=self.cfg.get("order", "C")
                        )[:n]
                    previous_chunk, previous_data, previous_present = chunk_id, data, entry is not None
                a, b = max(start, begin) - begin, min(stop, begin + len(previous_data)) - begin
                block = previous_data[a:b]
                yield out_begin, block, previous_present
                out_begin += len(block)

    def read(self):
        out = np.empty(self.shape, dtype=self.dtype)
        valid = np.zeros(self.shape[0], dtype=bool)
        for begin, data, present in self.blocks():
            out[begin : begin + len(data)] = data
            valid[begin : begin + len(data)] = present
        return out, valid


class ReusedArray:
    """Select exact original frames from an already validated v3 bucket."""

    def __init__(self, bucket, table, raw_key, spec, ranges, fps):
        self.bucket, self.table, self.raw_key, self.spec = Path(bucket), table, raw_key, spec
        self.ranges, self.fps = ranges, fps
        self.dtype = np.dtype(spec["config"]["dtype"])
        self.shape = (sum(b - a for a, b in ranges), *spec["config"]["shape"][1:])
        self.indices = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in ranges])
        self.is_rgb = len(self.shape) in (4, 5) and self.shape[-1] == 3 and self.dtype == np.uint8

    def read(self):
        key = "observation.ftp1." + self.raw_key
        selected = self.table.select([key, key + ".valid"]).take(pa.array(self.indices))
        value = np.asarray(selected[key].to_pylist(), dtype=self.dtype).reshape(self.shape)
        valid = np.asarray(selected[key + ".valid"].to_numpy(), dtype=bool)
        return value, valid

    def video_blocks(self, sensor):
        import av

        feature = "observation.tactile_rgb." if "tactile_data" in self.raw_key else "observation.images."
        feature += self.raw_key + (f".{sensor}" if sensor is not None else "")
        valid = self.table[feature + ".valid"].to_numpy()
        path = self.bucket / VIDEO_PATH.format(video_key=feature, chunk_index=0, file_index=0)
        out_begin = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for start, stop in self.ranges:
                container.seek(int((start / self.fps) / stream.time_base), stream=stream, backward=True)
                wanted = start
                for frame in container.decode(video=0):
                    index = round(float(frame.time) * self.fps)
                    if index < start:
                        continue
                    if index >= stop:
                        break
                    if index != wanted:
                        raise ValueError(f"Reused video frame discontinuity: {path}: {index} != {wanted}")
                    yield out_begin, frame.to_ndarray(format="rgb24")[None], bool(valid[index])
                    out_begin += 1
                    wanted += 1
                if wanted != stop:
                    raise ValueError(f"Reused video ended before selected episode: {path}")


def bucket_path(output_root, name, group):
    task = group["group"].split("/", 1)[1].removesuffix(".zarr")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task).strip("_")
    return Path(output_root) / name / (safe + "-" + hashlib.sha256(task.encode()).hexdigest()[:8])


def arrow_column(value):
    if value.ndim <= 1:
        return pa.array(value.tolist() if value.dtype.kind == "U" else value)
    # LeRobot numeric vector columns are lists with a shape declared in info.
    return pa.array(value.reshape(len(value), -1).tolist())


def video_stream(array, sensor, output, fps, encoder_threads):
    import imageio_ffmpeg

    expected = array.shape[0]
    if output.exists() and output.with_suffix(".complete.json").exists():
        complete = json.loads(output.with_suffix(".complete.json").read_text())
        if complete["frames"] == expected and complete["fps"] == fps:
            return complete
        raise ValueError(f"Incompatible existing video: {output}")
    height, width = array.shape[-3:-1]
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".tmp.mp4")
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264rgb",
        "-preset",
        "ultrafast",
        "-crf",
        "0",
        "-threads",
        str(encoder_threads),
        "-g",
        str(max(1, round(fps))),
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    valid_ranges = []
    count = 0
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            blocks = array.video_blocks(sensor) if isinstance(array, ReusedArray) else array.blocks()
            for begin, data, present in blocks:
                frames = data if sensor is None or isinstance(array, ReusedArray) else data[:, sensor]
                process.stdin.write(np.ascontiguousarray(frames).tobytes())
                count += len(frames)
                if present:
                    valid_ranges.append([begin, begin + len(frames)])
            process.stdin.close()
            err = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(f"Video encoding failed: {err}")
        except BaseException:
            process.kill()
            process.wait()
            raise
    if count != expected:
        raise ValueError(f"Video length {count} != {expected}")
    # PyAV reads the completed container index, without decoding the whole clip.
    import av

    with av.open(str(tmp)) as container:
        if container.streams.video[0].frames != expected:
            raise ValueError(f"Encoded video frame count mismatch: {output}")
    tmp.replace(output)
    complete = {
        "frames": count,
        "fps": fps,
        "width": width,
        "height": height,
        "valid_ranges": valid_ranges,
        "codec": "h264",
        "pix_fmt": "gbrp",
        "lossless_rgb": True,
    }
    atomic_json(output.with_suffix(".complete.json"), complete)
    return complete


def _feature(value):
    native_shape = list(value.shape[1:])
    feature = {
        "dtype": "string" if value.dtype.kind == "U" else str(value.dtype),
        "shape": [int(np.prod(native_shape))] if native_shape else [1],
        "names": None,
    }
    if len(native_shape) > 1:
        feature["native_shape"] = native_shape
    return feature


def convert_group(job):
    source_root, output_root, definition_path, group, encoder_threads = job[:5]
    staged_path = job[5] if len(job) > 5 else None
    options = job[6] if len(job) > 6 else {}
    reuse_bucket = options.get("reuse_bucket")
    started = time.time()
    definition = json.loads(Path(definition_path).read_text())
    name = definition["source"]
    task = group["group"].split("/", 1)[1].removesuffix(".zarr")
    bucket = bucket_path(output_root, name, group)
    if group["frames"] == 0:
        return {"source": name, "task": task, "status": "empty", "frames": 0}
    original, frame_ranges, source_episode_ids = resolve_selection(group)
    pilot = "source_census" in group
    if reuse_bucket and not pilot:
        raise ValueError("Reuse is only supported into a separate pilot namespace")
    if reuse_bucket and staged_path:
        raise ValueError("Choose either validated v3 reuse or sparse raw staging")
    index = None
    reuse_table = None
    if reuse_bucket:
        from validate import asset_identity

        reuse_bucket = Path(reuse_bucket)
        if reuse_bucket.resolve() == bucket.resolve():
            raise ValueError("Pilot output must not overwrite reused source")
        saved = json.loads((reuse_bucket / "meta/conversion_complete.json").read_text())
        validation = json.loads((reuse_bucket / "meta/validation.json").read_text())
        info = json.loads((reuse_bucket / "meta/info.json").read_text())
        index = json.loads((reuse_bucket / "meta/archive_index.json").read_text())
        expected = hashlib.sha256(
            json.dumps([VERSION, definition, index["archive_fingerprint"], original], sort_keys=True).encode()
        ).hexdigest()
        if saved["signature"] != expected or validation.get("conversion_signature") != expected:
            raise ValueError("Reuse definition/source conversion signature mismatch")
        if validation.get("assets") != asset_identity(reuse_bucket, info):
            raise ValueError("Reused v3 files changed since validation")
        if saved["frames"] != original["frames"] or info["total_episodes"] != len(original["episode_ends"]):
            raise ValueError("Reused v3 source population mismatch")
        reuse_table = pq.read_table(reuse_bucket / DATA_PATH.format(chunk_index=0, file_index=0))

        def archive_factory():
            return nullcontext(None)
    elif staged_path:
        from stage import RangeArchive, SparseArchive, load_manifest

        staged = load_manifest(staged_path, verify=True)
        if staged["group"] != group or staged["source"] != name:
            raise ValueError("Staged task identity differs from selected census group")
        files = []

        def archive_factory():
            return (
                SparseArchive(staged_path, staged) if "archive_index" in staged else RangeArchive(staged_path, staged)
            )

        if "archive_index" in staged:
            index = staged["archive_index"]
        source_fingerprint = staged["source_fingerprint"]
    else:
        files = sorted(p for p in (Path(source_root) / name).iterdir() if ".tar" in p.name)

        def archive_factory():
            return Parts(files)

        source_fingerprint = None
    if index is None:
        index = index_group(
            files, original, bucket / "meta" / "archive_index.json", archive_factory, source_fingerprint
        )
    elif not (bucket / "meta/archive_index.json").exists():
        atomic_json(bucket / "meta/archive_index.json", index)
    signature = hashlib.sha256(
        json.dumps([VERSION, definition, index["archive_fingerprint"], group], sort_keys=True).encode()
    ).hexdigest()
    complete_path = bucket / "meta" / "conversion_complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete["signature"] != signature:
            raise ValueError(f"Existing conversion definition changed: {bucket}")
        return complete
    if pilot:
        selection_receipt = bucket / "meta/pilot_selection.json"
        if selection_receipt.exists():
            previous = json.loads(selection_receipt.read_text())
            if previous.get("conversion_signature") != signature:
                raise ValueError("Incomplete pilot output belongs to a different immutable selection/definition")
    log("group_started", source=name, task=task, frames=group["frames"], bucket=str(bucket))
    arrays = index["arrays"]
    n = group["frames"]
    ends = np.array(group["episode_ends"], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    lengths = ends - starts
    episode_index = np.repeat(np.arange(len(ends), dtype=np.int64), lengths)
    frame_index = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    # Unknown-rate sources remain archival-only. 1 fps is a declared frame-index
    # container timescale, never accepted as a physical sampling rate by readers.
    fps = definition["nominal_fps"] or 1
    features = {}
    columns = {
        "timestamp": pa.array(frame_index / fps),
        "frame_index": pa.array(frame_index),
        "episode_index": pa.array(episode_index),
        "index": pa.array(np.arange(n, dtype=np.int64)),
    }
    for k, value in [
        ("timestamp", frame_index / fps),
        ("frame_index", frame_index),
        ("episode_index", episode_index),
        ("index", np.arange(n, dtype=np.int64)),
    ]:
        features[k] = _feature(value)
    if pilot:
        source_frames = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in frame_ranges])
        source_eps = np.repeat(np.asarray(source_episode_ids, dtype=np.int64), lengths)
        for key, value in [("source_frame_index", source_frames), ("source_episode_index", source_eps)]:
            key = "observation.ftp1." + key
            columns[key], features[key] = pa.array(value), _feature(value)
        atomic_json(
            bucket / "meta/pilot_selection.json",
            {
                "schema": group["selection_schema"],
                "conversion_signature": signature,
                "source_group": group["group"],
                "selected_source_episode_indices": source_episode_ids,
                "source_frame_ranges": frame_ranges,
                "source_total_frames": original["frames"],
                "selected_frames": n,
                "source_archive_fingerprint": index["archive_fingerprint"],
                "selection_sha256": hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest(),
                "reused_from": str(reuse_bucket) if reuse_bucket else None,
            },
        )
    videos, raw_numeric, instructions = {}, [], None
    with archive_factory() as stream:
        for key, spec in sorted(arrays.items()):
            if not key.startswith("data/"):
                continue
            raw_key = key[len("data/") :]
            if spec["config"]["shape"][0] != original["frames"]:
                raise ValueError(f"Original source array length mismatch: {key}")
            array = (
                ReusedArray(reuse_bucket, reuse_table, raw_key, spec, frame_ranges, fps)
                if reuse_bucket
                else Array(stream, spec, frame_ranges)
            )
            if array.shape[0] != n:
                raise ValueError(f"Length mismatch {key} {array.shape[0]} != {n}")
            is_rgb = len(array.shape) in (4, 5) and array.shape[-1] == 3 and array.dtype == np.uint8
            if is_rgb:
                tactile = "tactile_data" in raw_key
                sensors = range(array.shape[1]) if len(array.shape) == 5 else [None]
                for sensor in sensors:
                    feature = (
                        ("observation.tactile_rgb." if tactile else "observation.images.")
                        + raw_key
                        + (f".{sensor}" if sensor is not None else "")
                    )
                    output = bucket / VIDEO_PATH.format(video_key=feature, chunk_index=0, file_index=0)
                    result = video_stream(array, sensor, output, fps, encoder_threads)
                    videos[feature] = result
                    valid = np.zeros(n, dtype=bool)
                    for a, b in result["valid_ranges"]:
                        valid[a:b] = True
                    columns[feature + ".valid"] = pa.array(valid)
                    features[feature + ".valid"] = _feature(valid)
                    features[feature] = {
                        "dtype": "video",
                        "shape": [result["height"], result["width"], 3],
                        "names": ["height", "width", "channels"],
                        "info": {
                            "video.height": result["height"],
                            "video.width": result["width"],
                            "video.channels": 3,
                            "video.fps": fps,
                            "video.codec": "h264",
                            "video.pix_fmt": "gbrp",
                            "video.is_depth_map": False,
                            "has_audio": False,
                        },
                    }
                log("array_converted", source=name, task=task, key=key, videos=len(videos))
            else:
                value, valid = array.read()
                feature = "observation.ftp1." + raw_key
                columns[feature] = arrow_column(value)
                columns[feature + ".valid"] = pa.array(valid)
                features[feature], features[feature + ".valid"] = _feature(value), _feature(valid)
                raw_numeric.append(feature)
                if raw_key == "sub_task_instruction":
                    instructions = value
    if instructions is None:
        instructions = np.full(n, task)
    prompts, task_indices = np.unique(instructions, return_inverse=True)
    columns["task_index"] = pa.array(task_indices.astype(np.int64))
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    data_file = bucket / DATA_PATH.format(chunk_index=0, file_index=0)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), data_file.with_suffix(".tmp.parquet"), compression="zstd", row_group_size=2048)
    data_file.with_suffix(".tmp.parquet").replace(data_file)
    # Pandas index named 'task' is LeRobot v3's standard task-index convention.
    import pandas as pd

    tasks = pd.DataFrame(
        {"task_index": np.arange(len(prompts), dtype=np.int64)}, index=pd.Index(prompts.tolist(), name="task")
    )
    tasks.to_parquet(bucket / "meta" / "tasks.parquet")
    eps = []
    for i, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        row = {
            "episode_index": i,
            "length": b - a,
            "tasks": sorted(set(instructions[a:b].tolist())),
            "dataset_from_index": a,
            "dataset_to_index": b,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "source/group": group["group"],
            "source/episode_index": source_episode_ids[i],
            "source/from_frame": frame_ranges[i][0],
            "source/to_frame": frame_ranges[i][1],
        }
        for feature in videos:
            row.update(
                {
                    f"videos/{feature}/chunk_index": 0,
                    f"videos/{feature}/file_index": 0,
                    f"videos/{feature}/from_timestamp": a / fps,
                    f"videos/{feature}/to_timestamp": b / fps,
                }
            )
        eps.append(row)
    ep_file = bucket / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(eps), ep_file)
    images = [k for k in videos if k.startswith("observation.images.")]

    def select(*candidates):
        return next(("observation.images." + x for x in candidates if "observation.images." + x in images), None)

    head = select("camera_ego_rgb", "camera_main_rgb", "right_wrist_camera_rgb", "left_wrist_camera_rgb")
    left_wrist = select("left_wrist_camera_rgb")
    right_wrist = select("right_wrist_camera_rgb")
    layout = {
        "head": head,
        "left_wrist": left_wrist if left_wrist != head else None,
        "right_wrist": right_wrist if right_wrist != head else None,
    }
    definition.update(
        {
            "source_id": "ftp1/" + name,
            "source_group": group["group"],
            "raw_numeric_features": raw_numeric,
            "raw_tactile_features": [k for k in videos if k.startswith("observation.tactile_rgb.")],
            "camera_layout": layout,
            "camera_provenance": images,
            "physical_time_valid": definition["nominal_fps"] is not None,
            "physical_time_verified": definition["nominal_fps"] is not None,
            "container_fps": fps,
            "training_ready": definition["nominal_fps"] is not None,
            "training_modalities": ["video"],
            "state80_status": "not_emitted_unverified_native_pose_and_joint_semantics",
        }
    )
    regions_path = Path(definition_path).parent / "regions" / f"{name}.json"
    if regions_path.exists():
        definition["tactile_region_definition"] = json.loads(regions_path.read_text())
    if pilot:
        definition["selection_scope"] = "pilot_episodes"
        definition["selected_source_episode_indices"] = source_episode_ids
    atomic_json(bucket / "meta" / "source_definition.json", definition)
    info = {
        "codebase_version": "v3.0",
        "robot_type": definition["embodiment"],
        "total_episodes": len(ends),
        "total_frames": n,
        "total_tasks": len(prompts),
        "total_videos": len(videos),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(ends)}", "val": "0:0"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
        "source_id": definition["source_id"],
        "physical_time_valid": definition["physical_time_valid"],
        "conversion_version": VERSION,
    }
    atomic_json(bucket / "meta" / "info.json", info)
    # Never invent intensity labels from raw images or force vectors.
    atomic_json(
        bucket / "meta" / "tactile_conversion_status.json",
        {
            "enabled": False,
            "reason": definition["intensity_status"],
            "raw_preserved": True,
            "raw_features": definition["raw_tactile_features"],
        },
    )
    complete = {
        "signature": signature,
        "source": name,
        "task": task,
        "bucket": str(bucket),
        "status": "complete",
        "training_ready": definition["training_ready"],
        "frames": n,
        "episodes": len(ends),
        "videos": len(videos),
        "seconds": time.time() - started,
        **(
            {
                "selection_scope": "pilot_episodes",
                "selected_source_episode_indices": source_episode_ids,
                "reused_from": str(reuse_bucket) if reuse_bucket else None,
            }
            if pilot
            else {}
        ),
    }
    atomic_json(complete_path, complete)
    log("group_complete", **complete)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--definitions", default="configs/data_sources/ftp1")
    parser.add_argument("--census", default="configs/data_sources/ftp1/archive_census.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--sources", nargs="*")
    parser.add_argument(
        "--staged-manifest", help="Convert one verified PFS task range; original archive mount is unnecessary"
    )
    parser.add_argument(
        "--reuse-root", help="Validated full v3 root; pilot episodes reuse existing videos and numeric data"
    )
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--exclude-sources", nargs="*", default=[])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--run-id", help="Separate progress receipts when disjoint shards share one output root")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must lie in [0,num-shards)")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        parser.error("run-id must be a simple path component")
    run_id = args.run_id or (f"shard-{args.shard_index}-of-{args.num_shards}" if args.num_shards > 1 else None)
    progress_root = Path(args.output) / "_runs" / run_id if run_id else Path(args.output)
    jobs = []
    for source in json.loads(Path(args.census).read_text()):
        name = source["dataset"]
        if name == "sharpa" or name in args.exclude_sources or (args.sources and name not in args.sources):
            continue
        for group in source["groups"]:
            if int(hashlib.sha256(group["group"].encode()).hexdigest(), 16) % args.num_shards != args.shard_index:
                continue
            jobs.append(
                (args.input, args.output, str(Path(args.definitions) / f"{name}.json"), group, args.encoder_threads)
            )
    if args.staged_manifest:
        from stage import load_manifest

        staged = load_manifest(args.staged_manifest, verify=False)
        jobs = [(*job, args.staged_manifest) for job in jobs if job[3] == staged["group"]]
        if len(jobs) != 1:
            parser.error("Staged manifest must match exactly one selected census task")
    if args.reuse_root:
        reused = []
        for job in jobs:
            candidate = bucket_path(args.reuse_root, Path(job[2]).stem, job[3])
            if (candidate / "meta/validation.json").exists():
                job = (*job[:5], job[5] if len(job) > 5 else None, {"reuse_bucket": str(candidate)})
            reused.append(job)
        jobs = reused
    jobs.sort(key=lambda job: job[3]["frames"])
    if args.limit_groups is not None:
        jobs = jobs[: args.limit_groups]
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(convert_group, job): job[3]["group"] for job in jobs}
        for future in as_completed(pending):
            group = pending[future]
            try:
                results.append(future.result())
            except Exception as exc:
                import traceback

                error = {"group": group, "error": repr(exc), "traceback": traceback.format_exc()}
                errors.append(error)
                log("group_error", **error)
            atomic_json(
                progress_root / "conversion_progress.json",
                {"expected_groups": len(jobs), "complete": results, "errors": errors},
            )
    atomic_json(
        progress_root / "dataset_catalog.json",
        {
            "schema_version": VERSION,
            "excluded_sources": ["sharpa"],
            "datasets": results,
            "errors": errors,
            "complete": len(results) == len(jobs) and not errors,
        },
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
