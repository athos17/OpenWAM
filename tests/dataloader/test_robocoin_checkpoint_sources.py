from types import SimpleNamespace

import numpy as np
import pytest

from openwam.dataloader.robocoin import MultiRobotCOINDataset
from openwam.dataloader.utils.lerobotv3 import DataContractError
from openwam.dataloader.utils.normalization_package import (
    read_normalization_package, save_normalization_package, verify_normalization_package,
)


def bucket(robot, width, mapping=None):
    stats = {key: np.full(width, i, dtype=np.float32)
             for i, key in enumerate(("mean", "std", "min", "max", "q01", "q99"))}
    return SimpleNamespace(robot_type=robot, _normalize_mode="quantile", _unify=True,
                           _unify_dst_index=np.asarray(mapping or list(range(width))),
                           _unify_state_dst_index=np.asarray(mapping or list(range(width))),
                           _normalization_stats=stats)


def sources(*buckets):
    reader = object.__new__(MultiRobotCOINDataset)
    reader._buckets = list(buckets)
    config = dict(type="robocoin", normalize_mode="quantile", unify_action=True)
    return {"robocoin/" + key: value for key, value in reader.normalization_sources_for_config(config).items()}


def test_per_robot_checkpoint_roundtrip_and_transform_drift(tmp_path):
    records = sources(bucket("gripper", 20), bucket("dex", 32))
    save_normalization_package(tmp_path, records)
    verify_normalization_package(tmp_path, records)
    save_normalization_package(tmp_path, records)
    package = read_normalization_package(tmp_path)
    for name, width in [("gripper", 20), ("dex", 32)]:
        entry = package["sources"]["robocoin/" + name]
        saved = np.load(tmp_path / entry["stats_file"], allow_pickle=True).item()
        assert saved["eef"]["mean"].shape == (width,)
        assert entry["config"]["unify_action_map"] == list(range(width))
        assert entry["config"]["action_mode"] == "eef"
    records["robocoin/dex"]["stats_payload"]["eef"]["mean"][0] = 9
    with pytest.raises(ValueError, match="differ"):
        verify_normalization_package(tmp_path, records)
    entry = package["sources"]["robocoin/gripper"]
    (tmp_path / entry["stats_file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        read_normalization_package(tmp_path)


def test_same_robot_cannot_hide_different_statistics_or_maps():
    first, second = bucket("robot", 20), bucket("robot", 20)
    second._normalization_stats["q01"][0] += 1
    with pytest.raises(DataContractError, match="incompatible"):
        sources(first, second)
    second = bucket("robot", 20, list(range(1, 21)))
    with pytest.raises(DataContractError, match="incompatible"):
        sources(first, second)


def test_payload_and_path_cannot_be_ambiguous(tmp_path):
    records = sources(bucket("robot", 20))
    records["robocoin/robot"]["stats_path"] = "unused.npy"
    with pytest.raises(ValueError, match="both"):
        save_normalization_package(tmp_path, records)
