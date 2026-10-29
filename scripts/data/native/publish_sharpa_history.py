#!/usr/bin/env python3
"""Publish metadata for original-resolution Sharpa history; never rewrite source data.

Videos and 32px label arrays stay in their existing immutable PFS releases.
The small new publication binds both releases and records native video offsets.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import av
import numpy as np

from openwam.dataloader.publication import source_publication_contract
from openwam.tactile.intensity_store import IntensityStore
from openwam.tactile.sharpa_video_history import KEY, REGIONS, SCHEMA, inside, sha256
from scripts.data.native import catalog_sharpa_tactile as catalog


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2) + "\n"
    if path.exists() and path.read_text() == text:
        return
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text)
    os.replace(temporary, path)


def build_manifest(bucket, tactile_path):
    # All episode identities, source offsets, array headers, timestamps and masks.
    checked = catalog.validate_bucket(bucket, tactile_path)
    old = json.loads(tactile_path.read_text())
    videos = {}
    episodes = {}
    for key, entry in old["episodes"].items():
        relative = entry["source_video"]
        if relative not in videos:
            path = inside(bucket, relative)
            st = path.stat()
            with av.open(str(path)) as container:
                if len(container.streams.video) != 1:
                    raise ValueError(f"Expected one deform stream: {path}")
                stream = container.streams.video[0]
                if ((stream.height, stream.width) != (480, 1200)
                        or not np.isclose(float(stream.average_rate or 0), 30)
                        or not stream.frames or stream.start_time not in (None, 0)):
                    raise ValueError(f"Invalid native deform media header: {path}")
                videos[relative] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                                        frames=int(stream.frames), fps=30, shape=[480, 1200])
        offset = float(entry["source_video_from_timestamp"])
        first = round(offset * 30)
        if first < 0 or abs(first - offset * 30) > 2e-3 or first + entry["frames"] > videos[relative]["frames"]:
            raise ValueError(f"Episode is outside original video: {bucket}:{key}")
        episodes[key] = dict(video=relative, from_timestamp=offset, frames=entry["frames"])
    manifest = dict(schema=SCHEMA, dataset_info_sha256=checked["source_info_sha256"],
                    tactile_manifest_sha256=sha256(tactile_path), fps=30,
                    region_names=REGIONS, encoder_name="SharpaWave", video_key=KEY,
                    mosaic_shape=[480, 1200], tile_shape=[240, 240], observation_shape=[224, 224],
                    normalization="pyav_gray_uint8_div_255", resize="opencv_inter_area",
                    orientation="top_left_bottom_right_thumb_to_pinky",
                    video_integrity_scope="immutable PFS files, size/mtime and video headers; no new full payload hashes",
                    videos=videos, episodes=episodes)
    return manifest, checked


def publish(source, tactile, output):
    source, tactile, output = [Path(p).resolve() for p in (source, tactile, output)]
    config = dict(dataset_dir=str(source / "buckets"), tactile_root=str(tactile))
    original = source_publication_contract(config)
    if output.exists():
        return source_publication_contract({**config, "tactile_observation_root": str(output)})["tactile_observation"]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, tactile)):
        raise ValueError("Publish history into a separate version directory")
    stage = output.with_name(output.name + ".building")
    stage.mkdir(parents=True, exist_ok=True)
    with (stage / ".publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries = {}
        receipts = {}
        t0 = time.monotonic()
        old_catalog = json.loads((tactile / "tactile_catalog.json").read_text())["bucket_manifest_by_name"]
        for i, name in enumerate(original["buckets"]):
            bucket = (source / "buckets" / name).resolve(strict=True)
            relative = Path("partitions") / bucket.parent.name / bucket.name
            labels = tactile / relative / "tactile/manifest.json"
            document, checked = build_manifest(bucket, labels)
            if checked["array_inventory_sha256"] != old_catalog[name]["array_inventory_sha256"]:
                raise ValueError(f"Existing label cache changed since its publication: {name}")
            manifest = stage / relative / "observation.json"
            write(manifest, document)
            store = IntensityStore(labels, bucket, observation_size=224, encoder_name="SharpaWave",
                                   observation_manifest=manifest)
            # Start, middle and end of a nonempty episode in EVERY partition.
            episode = max(document["episodes"], key=lambda k: document["episodes"][k]["frames"])
            n = document["episodes"][episode]["frames"]
            samples = []
            for frame in sorted({0, n // 2, n - 1}):
                sample = store.window(int(episode), frame / 30)
                values = sample["tactile_observation_maps"]
                if tuple(values.shape) != (4, 10, 1, 224, 224) or not np.isfinite(values.numpy()).all():
                    raise ValueError(f"Invalid native history sample: {name}")
                samples.append(dict(episode=int(episode), frame=frame,
                                    valid_history=int(sample["tactile_valid"].sum()),
                                    valid_future=int(sample["future_tactile_valid"].sum())))
            entries[name] = dict(manifest=str(manifest.relative_to(stage)), sha256=sha256(manifest))
            receipts[name] = dict(episodes=checked["episodes"], frames=checked["frames"],
                                  video_files=len(document["videos"]), samples=samples,
                                  array_inventory_sha256=checked["array_inventory_sha256"])
            if i % 20 == 0 or i + 1 == len(original["buckets"]):
                print(json.dumps(dict(buckets_completed=i + 1, buckets_expected=len(original["buckets"]),
                                      elapsed_s=round(time.monotonic() - t0, 1))), flush=True)
        # Recheck source/label publication identities before committing the new release.
        if source_publication_contract(config) != original:
            raise ValueError("Source publication changed while preparing native history")
        report = dict(status="passed", scope="all metadata/array clocks/masks/video headers; 3 history windows per partition",
                      full_video_payload_audit=False, buckets=receipts,
                      elapsed_s=time.monotonic() - t0)
        write(stage / "validation.json", report)
        publication = dict(schema=SCHEMA, status="complete", training_ready=True,
                           source_root=str(source), tactile_root=str(tactile),
                           source_manifest_sha256=original["sha256"],
                           tactile_catalog_sha256=original["tactile"]["binding"]["catalog_sha256"],
                           episodes=original["episodes"], frames=original["frames"],
                           buckets=entries, validation_sha256=sha256(stage / "validation.json"),
                           storage="metadata only; existing PFS videos and 32px labels remain required",
                           observation_shape=[224, 224], label_shape=[32, 32])
        write(stage / "publication.json", publication)
        (stage / "README.md").write_text(
            "# Sharpa 高分辨率触觉历史\n\n"
            "本目录保存历史观测发布索引，训练在线解码原始 deform，逐指裁剪 240×240 → 224×224。\n"
            "未来标签继续读取原有 32×32 缓存。没有复制或改写视频、数值数据、缓存或划分。\n\n"
            f"原始机器人数据与视频：`{source}`\n\n32×32 标签：`{tactile}`\n\n"
            "上述两个目录必须在训练时保持可读。索引不是独立媒体副本，也不在 FTP 临时清理范围内。\n"
            "训练配置：train_touchwam_ftp1；TOUCHWAM_SHARPA_HISTORY_ROOT 可覆盖本发布路径。\n"
            "验收范围见 validation.json；完整模型多 GPU 训练由训练验收另行确认。\n")
        source_publication_contract({**config, "tactile_observation_root": str(stage)})
        os.rename(stage, output)
    return dict(path=str(output), status="complete", buckets=len(entries),
                episodes=original["episodes"], frames=original["frames"],
                bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--tactile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.tactile, args.output)), flush=True)
