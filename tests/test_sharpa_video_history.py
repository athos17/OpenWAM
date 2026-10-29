"""Encoded packed videos: exact offsets, causal history, labels and release binding."""
import json
import shutil

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openwam.dataloader.publication import source_publication_contract
from openwam.tactile.intensity_store import IntensityStore
from openwam.tactile.sharpa_video_history import KEY, decode_gray_frames, sha256
from scripts.data.native import catalog_sharpa_tactile as catalog
from scripts.data.native.extract_sharpa_deform import extract_partition
from scripts.data.native.publish_sharpa_history import build_manifest, publish


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def release(tmp_path):
    source, cache, output = [tmp_path / x for x in ("source", "labels", "history")]
    bucket = source / "partitions/task/season"
    (source / "buckets").mkdir(parents=True)
    (source / "buckets/task--season").symlink_to("../partitions/task/season", target_is_directory=True)
    video = bucket / f"videos/{KEY}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = 1200, 480, "yuv420p"
        s.codec_context.gop_size = 12
        for t in range(72):
            pixels = np.zeros((480, 1200, 3), np.uint8)
            pattern = (np.indices((240, 240)).sum(0) % 4 == 0) * 30
            for region in range(10):
                y, x = divmod(region, 5)
                pixels[y * 240:(y + 1) * 240, x * 240:(x + 1) * 240] = (region * 15 + t + pattern)[..., None]
            for packet in s.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    info = dict(codebase_version="v3.0", fps=30, total_episodes=2, total_frames=48,
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                features={KEY: dict(shape=[480, 1200, 3])})
    dump(bucket / "meta/info.json", info)
    ep = bucket / "meta/episodes/chunk-000/file-000.parquet"
    ep.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        dict(episode_index=e, length=24, **{f"videos/{KEY}/chunk_index": 0,
             f"videos/{KEY}/file_index": 0, f"videos/{KEY}/from_timestamp": start / 30})
        for e, start in ((0, 4), (1, 37))]), ep)
    np.save(source / "normalization_stats.npy", np.arange(3))
    dump(source / "dataset_manifest.json", dict(version="sharpadex_native_tcp_v1_20261026",
         status="data_and_reader_validated", partitions=1, episodes=2, frames=48,
         normalization_sha256=sha256(source / "normalization_stats.npy")))
    shutil.copytree(source, cache, symlinks=True)
    extract_partition(cache / "partitions/task/season")
    catalog.catalog(source, cache, {})
    publish(source, cache, output)
    labels = cache / "partitions/task/season/tactile/manifest.json"
    history = output / "partitions/task/season/observation.json"
    return source, cache, output, bucket, labels, history, video


def test_native_history_reads_correct_packed_offset_and_keeps_future_labels(release):
    source, cache, output, bucket, labels, history, video = release
    original = IntensityStore(labels, bucket)
    store = IntensityStore(labels, bucket, observation_size=224, encoder_name="SharpaWave", observation_manifest=history)
    sample = store.window(1, 10 / 30)
    old = original.window(1, 10 / 30)
    for key in old:
        if torch.is_tensor(old[key]):
            torch.testing.assert_close(old[key], sample[key], rtol=0, atol=0, equal_nan=True)
        else:
            assert old[key] == sample[key]
    # Independent sequential decode, starting from frame zero (reader seeks).
    with av.open(str(video)) as c:
        frames = {i: f.to_ndarray(format="gray") for i, f in enumerate(c.decode(video=0)) if 44 <= i <= 47}
    expected = np.stack([np.stack([
        cv2.resize(frames[t][(r // 5) * 240:(r // 5 + 1) * 240,
                             (r % 5) * 240:(r % 5 + 1) * 240], (224, 224), interpolation=cv2.INTER_AREA)
        for r in range(10)]) for t in range(44, 48)]).astype(np.float32) * (1 / 255)
    torch.testing.assert_close(sample["tactile_observation_maps"][:, :, 0], torch.from_numpy(expected), rtol=0, atol=0)
    assert sample["tactile_encoder_names"] == ["SharpaWave"] * 10
    beginning = store.window(1, 0)
    assert not beginning["tactile_valid"][:3].any()
    assert not beginning["tactile_observation_maps"][:3].any()
    end = store.window(1, 23 / 30)
    assert not end["future_tactile_valid"].any()
    assert end["tactile_valid"].all()


def test_missing_native_history_never_upsamples_old_labels(release):
    _, _, _, bucket, labels, _, _ = release
    store = IntensityStore(labels, bucket, observation_size=224, encoder_name="SharpaWave")
    with pytest.raises(ValueError, match="not upsampled"):
        store.window(0, 0)


@pytest.mark.parametrize("change", ["offset", "region", "source", "video", "deleted_video"])
def test_rejects_changed_offsets_identities_and_media(release, change):
    _, _, _, bucket, labels, history, video = release
    document = json.loads(history.read_text())
    if change == "offset":
        document["episodes"]["1"]["from_timestamp"] = 0
    elif change == "region":
        document["region_names"].reverse()
    elif change == "source":
        document["dataset_info_sha256"] = "incorrect"
    elif change == "video":
        with video.open("ab") as f:
            f.write(b"changed")
    else:
        video.unlink()
    dump(history, document)
    with pytest.raises((ValueError, FileNotFoundError)):
        IntensityStore(labels, bucket, observation_size=224, encoder_name="SharpaWave", observation_manifest=history)


def test_publication_is_metadata_only_idempotent_and_binds_manifests(release):
    source, cache, output, bucket, labels, history, video = release
    before = {p: sha256(p) for root in (source, cache) for p in root.rglob("*") if p.is_file()}
    publish(source, cache, output)
    assert before == {p: sha256(p) for p in before}
    assert not list(output.rglob("*.mp4")) and not list(output.rglob("*.npy"))
    config = dict(dataset_dir=str(source / "buckets"), tactile_root=str(cache), tactile_observation_root=str(output))
    assert source_publication_contract(config)["tactile_observation"]["buckets_verified"] == 1
    with history.open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="manifest changed"):
        source_publication_contract(config)


def test_decoder_fails_for_missing_valid_source_frames(release):
    with pytest.raises(ValueError, match="Missing valid"):
        decode_gray_frames(release[-1], [72], 30)
