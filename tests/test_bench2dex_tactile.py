import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from openwam.dataloader.bench2dex_tactile import Bench2DexTactileDataset, cache_key
from openwam.dataloader.registry import build_dataset
from openwam.tactile.sharpa import SITE_NAMES, read_tactile_frames


def test_history_future_alignment_boundaries_and_cache(tmp_path):
    source = tmp_path / "episode.h5"
    with h5py.File(source, "w") as h5:
        values = np.broadcast_to(np.arange(20)[:, None, None] * 0.0001, (20, 8, 8)).astype(np.float32)
        for site in SITE_NAMES:
            h5.create_dataset(f"robot/tactile/distance_along_normal_m/{site}", data=values)
            h5.create_dataset(f"robot/tactile/contact_mask/{site}", data=np.ones_like(values, dtype=bool))
    dataset = object.__new__(Bench2DexTactileDataset)
    dataset.history, dataset.map_size = 4, 8
    dataset.future_frames, dataset.future_stride = 3, 2
    episode = dict(source=source.name, segments=[[4, 20]])
    with h5py.File(source) as h5:
        raw = dataset.observation_extras(h5, episode, start=5, segment_start=4)
        assert not raw["tactile_valid"][:2].any()
        torch.testing.assert_close(raw["tactile_maps"][2:, 0, 0, 0, 0], torch.tensor([0.0004, 0.0005]))
        torch.testing.assert_close(raw["future_tactile_maps"][:, 0, 0, 0, 0], torch.tensor([0.0007, 0.0009, 0.0011]))
        assert raw["future_tactile_offsets"].tolist() == [2, 4, 6]
        with pytest.raises(ValueError, match="cross"):
            dataset.observation_extras(h5, episode, start=15, segment_start=4)
        maps, valid = read_tactile_frames(h5, 0, 20, 8)
        key = cache_key(source.name)
        np.save(tmp_path / f"{key}.maps.npy", maps)
        np.save(tmp_path / f"{key}.valid.npy", valid)
        dataset.cache_root = tmp_path
        cached = dataset.observation_extras(h5, episode, start=5, segment_start=4)
    for key in raw:
        torch.testing.assert_close(raw[key], cached[key], rtol=0, atol=0)


def test_manifest_to_full_training_sample_and_balanced_mixture(tmp_path):
    """Exercise config -> registry -> mixture -> RGB/action/history/future labels."""
    stats = {
        key: np.ones(58, np.float32) if key in ("std", "max", "q99") else np.zeros(58, np.float32)
        for key in ("mean", "std", "min", "max", "q01", "q99")
    }
    np.save(tmp_path / "stats.npy", {"eef": stats, "eef_state": stats})
    entries = []
    for task, frames in (("task_a", 33), ("task_b", 35)):
        source = tmp_path / f"{task}.h5"
        with h5py.File(source, "w") as h5:
            encoded = cv2.imencode(".jpg", np.full((8, 8, 3), 120, np.uint8))[1]
            for camera in ("cam_stereo_left", "cam_wrist_left", "cam_wrist_right"):
                ds = h5.create_dataset(f"cameras/{camera}/rgb", (frames,), dtype=h5py.vlen_dtype(np.uint8))
                for t in range(frames):
                    ds[t] = encoded
            values = np.broadcast_to(np.arange(frames)[:, None, None] * 0.0001, (frames, 8, 8)).astype(np.float32)
            for site in SITE_NAMES:
                h5.create_dataset(f"robot/tactile/distance_along_normal_m/{site}", data=values)
                h5.create_dataset(f"robot/tactile/contact_mask/{site}", data=np.ones_like(values, dtype=bool))
        numeric = f"{task}.npz"
        np.savez(
            tmp_path / numeric,
            action=np.full((frames, 58), 0.25, np.float32),
            state=np.full((frames, 58), 0.5, np.float32),
        )
        entries.append(
            dict(
                source=source.name,
                numeric=numeric,
                bytes=source.stat().st_size,
                segments=[[0, frames]],
                task_name=task,
                split="train",
                prompt=task,
            )
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                schema="bench2dex_alpha_v1",
                source_root=str(tmp_path),
                stats_path="stats.npy",
                task_names=["task_a", "task_b"],
                episodes=entries,
            )
        )
    )
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(
            config_name="train_touchwam",
            overrides=[
                f"dataloader.manifest_path={manifest}",
                "dataloader.height=16",
                "dataloader.width=16",
            ],
        )
    dataset = build_dataset(cfg.dataloader)
    assert len(dataset) == 6  # One-window task is repeated to match the three-window task.
    assert sorted(np.bincount(dataset._index_map[:, 0]).tolist()) == [3, 3]
    sample = dataset.datasets[0][0]
    assert len(sample["video"]) == 9 and sample["video"][0].size == (16, 16)
    assert sample["tactile_maps"].shape == (4, 10, 2, 32, 32)
    assert sample["future_tactile_maps"].shape == (8, 10, 2, 32, 32)
    assert sample["action"].shape == (32, 80) and sample["action_mask"].sum() == 32 * 58
    assert sample["proprio_mask"].sum() == 58
    assert not sample["tactile_valid"][:3].any()
    assert sample["tactile_valid"][3].all()
    torch.testing.assert_close(sample["future_tactile_maps"][:, 0, 0, 0, 0], torch.arange(2, 17, 2).float() * 0.0001)
    assert sample["future_tactile_offsets"].tolist() == list(range(2, 17, 2))
    from openwam.deploy.model_loader import _build_normalizer
    from openwam.train.utils.checkpointing import save_normalization_stats, verify_resume_normalization_stats

    output = tmp_path / "checkpoint"
    output.mkdir()
    save_normalization_stats(str(output), dataset)
    verify_resume_normalization_stats(str(output), dataset)
    assert (output / "normalization_stats.npy").is_file()
    assert not (output / "normalization_sources.json").exists()
    normalizer = _build_normalizer(cfg, str(output))
    raw = normalizer.unnormalize(sample["action"].numpy())
    np.testing.assert_allclose(raw, np.full((32, 58), 0.25, np.float32), atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(tactile_future_frames=17, tactile_future_stride=2),
        dict(tactile_fps=30),
        dict(tactile_history=0),
        dict(tactile_map_size=4),
    ],
)
def test_invalid_tactile_config_rejected_before_reading_files(overrides):
    with pytest.raises(ValueError):
        Bench2DexTactileDataset.from_config(dict(manifest_path="/missing/manifest.json", **overrides))
