import hashlib
import json

import numpy as np
import pytest

from openwam.dataloader.touchwam_pretrain import LeRobotVideoIntensityDataset
from scripts.data.freeze_episode_selection import freeze
from tests.test_touchwam_publication_reader import make_video_dataset


def selected_config(tmp_path):
    root = tmp_path / "buckets"
    bucket = make_video_dataset(root / "first")
    (root / "publication.json").write_text(json.dumps({"status": "complete", "episodes": 2, "frames": 40}))
    path = tmp_path / "selection.json"
    record = freeze(root, path, max_hours=100)
    return root, bucket, path, record, dict(dataset_dir=str(root), tactile_mode="off", height=48, width=32,
        episode_selection_manifest=str(path), episode_selection_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_frozen_selection_preserves_physical_offsets_and_ignores_new_buckets(tmp_path):
    root, _, _, record, config = selected_config(tmp_path)
    assert record["selected_episodes"] == 1  # Validation episode is excluded.
    assert record["selected_frames"] == 20
    make_video_dataset(root / "added_after_freeze")
    ds = LeRobotVideoIntensityDataset.from_config(config)
    assert len(ds.buckets) == 1
    sample = ds[0]
    assert sample["episode_id"] == 1
    assert np.asarray(sample["video"][0])[:30].mean() == pytest.approx(100, abs=2)
    assert ds.buckets[0].training_data_contract["episode_selection"]["sha256"] == config["episode_selection_sha256"]


def test_frozen_selection_rejects_metadata_drift_and_changed_selection(tmp_path):
    _, bucket, path, _, config = selected_config(tmp_path)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256"):
        LeRobotVideoIntensityDataset.from_config(config)
    config["episode_selection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    info = bucket / "meta/info.json"
    info.write_text(info.read_text() + " ")
    with pytest.raises(ValueError, match="metadata changed"):
        LeRobotVideoIntensityDataset.from_config(config)


def test_frozen_selection_cannot_be_resampled_or_used_as_validation(tmp_path):
    _, _, _, _, config = selected_config(tmp_path)
    with pytest.raises(ValueError, match="replaces"):
        LeRobotVideoIntensityDataset.from_config({**config, "total_hours": 1})
    with pytest.raises(ValueError, match="split"):
        LeRobotVideoIntensityDataset.from_config(config, split="val")
