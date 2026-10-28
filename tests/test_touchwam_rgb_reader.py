"""Real v3 RGB video reads retain color, clocks and missing-packet semantics."""
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openwam.dataloader.robot_intensity import LeRobotRobotIntensityDataset
from openwam.dataloader.touchwam_pretrain import HomogeneousPublishedBuckets, LeRobotVideoIntensityDataset
from openwam.tactile.rgb_store import RGB_CODEC, RGBTactileStore
from openwam.tactile.regions import collate_regions
from tests.test_touchwam_robot_intensity import robot_bucket


def rgb_bucket(root, *, missing=()):
    av = pytest.importorskip("av")
    root, action = robot_bucket(root)
    info_path, ep_path = root / "meta/info.json", root / "meta/episodes/chunk-000/file-000.parquet"
    info, episodes = json.loads(info_path.read_text()), pq.read_table(ep_path).to_pylist()
    data_path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_path)
    regions, originals = [], []
    for n, region in enumerate(("right.gripper.jaw_a.surface", "right.gripper.jaw_b.surface")):
        key = f"observation.tactile_rgb.right_tactile_data_gripper.{n}"
        shape = [8, 12, 3]
        info["features"][key] = dict(dtype="video", shape=shape, info={"video.fps": 10, "video.channels": 3})
        info["features"][key + ".valid"] = dict(dtype="bool", shape=[1])
        valid = [True] * 40
        for frame in missing:
            valid[20 + frame] = False
        table = table.append_column(key + ".valid", pa.array(valid))
        pixels = np.zeros((40, *shape), dtype=np.uint8)
        pixels[:20, ..., 0] = 240  # A different physical episode in the same MP4.
        for frame in range(20):
            pixels[20 + frame] = [frame * 10, 30 + n * 20, 60 + n * 20]
        pixels[20] = 0  # A real, valid dark image, not a missing packet.
        originals.append(pixels)
        path = root / "videos" / key / "chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264rgb", rate=10)
            stream.width, stream.height, stream.pix_fmt = 12, 8, "rgb24"
            stream.options = {"crf": "0", "preset": "ultrafast", "g": "10"}
            for frame in pixels:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        for episode in episodes:
            prefix = f"videos/{key}/"
            episode.update({prefix + "chunk_index": 0, prefix + "file_index": 0,
                            prefix + "from_timestamp": int(episode["episode_index"]) * 2.,
                            prefix + "to_timestamp": (int(episode["episode_index"]) + 1) * 2.})
        regions.append(dict(region_name=region, sensor_name="unit", video_key=key, validity_column=key + ".valid",
                            orientation=dict(rotate_quadrants=0, flip_horizontal=False, flip_vertical=False),
                            roi={"type": "full"}))
    pq.write_table(pa.Table.from_pylist(episodes), ep_path)
    pq.write_table(table, data_path)
    info_path.write_text(json.dumps(info))
    info_hash = hashlib.sha256(info_path.read_bytes()).hexdigest()
    action["dataset_info_sha256"] = info_hash
    (root / "meta/action_contract.json").write_text(json.dumps(action))
    manifest = dict(schema=RGB_CODEC, codec=RGB_CODEC, representation="optical_rgb", normalization="uint8_div_255",
                    time_alignment="dataset_rows", source_id="ftp1/Synthetic", dataset_info_sha256=info_hash,
                    status="verified", verification={"evidence": ["Synthetic known colors, layouts and 10 Hz clock"]},
                    regions=regions, episodes={str(i): {"layout_present": [True, True]} for i in range(2)})
    (root / "meta/tactile_rgb.json").write_text(json.dumps(manifest))
    return root, manifest, originals


def rgb_reader(root, **kwargs):
    return LeRobotRobotIntensityDataset(root, height=48, width=32, tactile_mode="required",
                                       tactile_representation="rgb", **kwargs)


def write_manifest(root, manifest):
    (root / "meta/tactile_rgb.json").write_text(json.dumps(manifest))


def test_rgb_robot_sample_preserves_video_offsets_colors_and_actions(tmp_path):
    root, _, _ = rgb_bucket(tmp_path / "robot")
    ds = rgb_reader(root)
    sample = ds[0]
    assert ds.tactile_codec == RGB_CODEC and sample["training_modality"] == "video_action_tactile"
    assert sample["episode_id"] == 1
    assert sample["tactile_maps"].shape == (4, 2, 3, 128, 128)
    assert sample["future_tactile_maps"].shape == (8, 2, 3, 128, 128)
    assert sample["tactile_spatial_valid"].shape == (2, 1, 128, 128)
    assert sample["action"].shape == (32, 80) and sample["action_mask"].any()
    assert not sample["tactile_valid"][:-1].any()  # never read the previous episode
    assert sample["tactile_valid"][-1].all()
    assert not sample["tactile_maps"][-1].any()  # dark is a valid RGB observation
    torch.testing.assert_close(sample["future_tactile_maps"][0, 0, :, 0, 0], torch.tensor([10., 30., 60.]) / 255)
    batch = collate_regions([sample], history=4, future_frames=8, map_size=128,
                            codec=RGB_CODEC, sensor_names=["unit"])
    assert batch["tactile_maps"].shape == (1, 4, 2, 3, 128, 128)
    assert batch["tactile_sensor_ids"].tolist() == [[0, 0]]
    assert HomogeneousPublishedBuckets([ds]).tactile_codec == RGB_CODEC


def test_missing_rgb_packet_is_masked_without_changing_query_or_action(tmp_path):
    root, _, _ = rgb_bucket(tmp_path / "robot", missing=[1])
    sample = rgb_reader(root)[0]
    assert not sample["future_tactile_valid"][0].any()
    assert not sample["future_tactile_maps"][0].any()
    assert sample["future_tactile_valid"][1].all()
    assert sample["tactile_query_mask"].all() and sample["action_mask"].any()


def test_rgb_history_is_causal_and_future_never_relabels_current_packet(tmp_path):
    root, _, _ = rgb_bucket(tmp_path / "robot")
    store = RGBTactileStore(root / "meta/tactile_rgb.json", root, size=8,
                           history_times=[-.05, 0], future_times=[.01, .05, .1])
    zero = store.window(1, 0)
    assert not zero["future_tactile_valid"][:2].any()
    assert zero["future_tactile_valid"][2].all()
    later = store.window(1, .1)
    assert later["tactile_valid"].all()
    assert not later["tactile_maps"][0].any()  # t=.05 holds observed dark t=0
    assert later["tactile_history_source_times_s"].tolist() == pytest.approx([-.1, 0])


def test_rgb_float32_episode_tail_keeps_real_last_frame_without_future_padding(tmp_path):
    root, _, _ = rgb_bucket(tmp_path / "robot")
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.column_names.index("timestamp"), "timestamp", table["timestamp"].cast(pa.float32()))
    pq.write_table(table, path)
    ds = rgb_reader(root)
    assert ds[11]["future_tactile_valid"].all()
    tail = ds[len(ds) - 1]
    assert tail["future_tactile_valid"][:2].all()
    assert not tail["future_tactile_valid"][2:].any()
    assert not tail["future_tactile_maps"][2:].any()


def test_rgb_orientation_transforms_roi_and_preserves_supported_color(tmp_path):
    root, manifest, _ = rgb_bucket(tmp_path / "robot")
    entry = manifest["regions"][0]
    entry["roi"] = dict(type="rectangle", bounds=[0, 0, 6, 8])
    entry["orientation"]["flip_horizontal"] = True
    write_manifest(root, manifest)
    sample = rgb_reader(root, tactile_rgb_map_size=12)[1]
    roi = sample["tactile_spatial_valid"][0, 0]
    assert not roi[:, :6].any() and roi[:, 6:].all()
    assert not sample["tactile_maps"][-1, 0, :, :, :6].any()
    torch.testing.assert_close(sample["tactile_maps"][-1, 0, :, 0, 6], torch.tensor([10., 30., 60.]) / 255)


def test_rgb_irregular_roi_mask_is_fingerprinted_and_not_a_pressure_map(tmp_path):
    root, manifest, _ = rgb_bucket(tmp_path / "robot")
    pixels = np.zeros((8, 12), dtype=bool)
    pixels[2:6, 3:9] = True
    path = root / "meta/roi.npy"
    np.save(path, pixels)
    manifest["regions"][0]["roi"] = dict(type="mask", path="meta/roi.npy",
                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_manifest(root, manifest)
    sample = rgb_reader(root)[1]
    assert sample["tactile_spatial_valid"][0].any() and not sample["tactile_spatial_valid"][0].all()
    assert not sample["tactile_maps"][-1, 0, :, 0, 0].any()
    torch.testing.assert_close(sample["tactile_maps"][-1, 0, :, 64, 64], torch.tensor([10., 30., 60.]) / 255)
    np.save(path, ~pixels)
    with pytest.raises(ValueError, match="ROI mask fingerprint"):
        rgb_reader(root)


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m.update(dataset_info_sha256="wrong"), "fingerprint"),
    (lambda m: m.update(source_id="different"), "source identity"),
    (lambda m: m.update(status="pending"), "verification evidence"),
    (lambda m: m["regions"][0].pop("orientation"), "orientation"),
    (lambda m: m["regions"][0].update(roi={}), "explicit full or rectangle"),
    (lambda m: m["regions"][1].update(region_name="right.gripper.jaw_a.surface"), "unique"),
    (lambda m: m["regions"][0].update(video_key="observation.images.camera"), "native three-channel tactile"),
    (lambda m: m["episodes"].pop("0"), "episode population"),
])
def test_rgb_requires_verified_explicit_source_geometry_and_population(tmp_path, mutation, match):
    root, manifest, _ = rgb_bucket(tmp_path / "robot")
    mutation(manifest)
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match=match):
        rgb_reader(root)


def test_rgb_requires_action_contract_and_does_not_use_existing_intensity_sidecar(tmp_path):
    root, _, _ = rgb_bucket(tmp_path / "robot")
    (root / "meta/action_contract.json").unlink()
    with pytest.raises(FileNotFoundError, match="action contract"):
        rgb_reader(root)
    with pytest.raises(ValueError, match="action reader"):
        LeRobotVideoIntensityDataset(root, tactile_representation="rgb")


def test_ego_rgb_uses_vt_without_action_or_proprio_placeholders(tmp_path):
    root, manifest, _ = rgb_bucket(tmp_path / "ego")
    definition = root / "meta/source_definition.json"
    identity = json.loads(definition.read_text())
    identity.update(source_id="synthetic_ego", source_kind="ego")
    definition.write_text(json.dumps(identity))
    manifest["source_id"] = "synthetic_ego"
    write_manifest(root, manifest)
    ds = LeRobotVideoIntensityDataset(root, height=48, width=32, tactile_representation="rgb", tactile_mode="required")
    sample = ds[0]
    assert sample["training_modality"] == "video_tactile" and sample["source_kind"] == "ego"
    assert sample["tactile_codec"] == RGB_CODEC
    assert not {"action", "action_mask", "proprio", "proprio_mask"} & set(sample)


def test_rgb_media_shape_is_checked_before_roi_or_resizing_can_hide_mismatch(tmp_path):
    root, manifest, _ = rgb_bucket(tmp_path / "robot")
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"][manifest["regions"][0]["video_key"]]["shape"][0] = 16
    info_path.write_text(json.dumps(info))
    fingerprint = hashlib.sha256(info_path.read_bytes()).hexdigest()
    manifest["dataset_info_sha256"] = fingerprint
    write_manifest(root, manifest)
    action_path = root / "meta/action_contract.json"
    action = json.loads(action_path.read_text())
    action["dataset_info_sha256"] = fingerprint
    action_path.write_text(json.dumps(action))
    with pytest.raises(ValueError, match="media dimensions"):
        rgb_reader(root)


def test_source_config_passes_rgb_reader_options_and_publication_still_gates(tmp_path):
    from openwam.dataloader.registry import build_dataset

    root, _, _ = rgb_bucket(tmp_path / "robot")
    config = dict(type="lerobot_robot_intensity", dataset_dir=str(root), source_kind="robot", source_family="ftp1",
                  expected_modalities=["video_action_tactile"], tactile_mode="required", tactile_representation="rgb",
                  tactile_rgb_map_size=16, tactile_rgb_manifest="meta/tactile_rgb.json", height=48, width=32)
    ds = build_dataset(config)
    assert ds[0]["tactile_maps"].shape[-3:] == (3, 16, 16)
    assert ds.training_data_contract["tactile_representation"] == "rgb"
    assert ds.training_data_contract["tactile_manifest_sha256"] == hashlib.sha256((root / "meta/tactile_rgb.json").read_bytes()).hexdigest()
    (root / "publication.json").write_text(json.dumps(dict(status="converting", episodes=2, frames=40)))
    with pytest.raises(ValueError, match="not complete"):
        build_dataset(dict(config, require_publication=True))


def test_homogeneous_source_cannot_hide_mixed_rgb_intensity_buckets(tmp_path):
    root, _, _ = rgb_bucket(tmp_path / "rgb")
    rgb = rgb_reader(root)
    intensity_root, _ = robot_bucket(tmp_path / "intensity", tactile=True)
    intensity = LeRobotRobotIntensityDataset(intensity_root, height=48, width=32, tactile_mode="required")
    with pytest.raises(ValueError, match="one tactile representation"):
        HomogeneousPublishedBuckets([rgb, intensity])


def test_real_rgb_video_reader_reaches_rgb_and_shared_model_backward(tmp_path, monkeypatch):
    from scripts.validation.touchwam_multisource_gpu import tiny_model
    from tests.test_touchwam import open_gates

    root, _, _ = rgb_bucket(tmp_path / "robot")
    sample = rgb_reader(root)[1]
    model = tiny_model(torch.device("cpu"), torch.float32, rgb_enabled=True)
    model.tactile_spec["sensor_names"] = ["unit", "bench2dex_sim"]
    # Real video I/O, robot reader, collation, Wan/ActionDiT and tactile flow;
    # only expensive scene/text encoders use fixed-shape synthetic latents.
    monkeypatch.setattr(model, "preprocess", lambda **kw: dict(
        input_latents=torch.randn(1, 4, 3, 4, 4), context=torch.randn(1, 4, 32),
        first_frame_latents=torch.zeros(1, 4, 1, 4, 4), context_mask=torch.ones(1, 4, dtype=torch.bool),
        fuse_vae_embedding_in_latents=True))
    prepared = model.prepare_inputs([sample])
    assert prepared["tactile_codec"] == RGB_CODEC
    open_gates(model)
    prepared["use_gradient_checkpointing"] = True
    losses = model.compute_loss(**prepared)
    losses["loss"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["loss_action"] > 0 and losses["loss_tactile_rgb"] > 0
    assert model.touch_v2.rgb_input_projection[0].weight.grad.abs().sum() > 0
    assert model.touch_v2.history_encoder.layers[0].linear1.weight.grad.abs().sum() > 0
    assert model.touch_v2.input_projection[0].weight.grad is None
