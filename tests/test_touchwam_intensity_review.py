"""Regression tests for physical-time and partial-ROI audit findings."""

import hashlib
import json

import numpy as np
import pyarrow as pa
import pytest
import torch

from openwam.dataloader.touchwam_pretrain import LeRobotVideoIntensityDataset
from openwam.tactile.intensity_store import IntensityStore


def test_roi_area_resize_preserves_measured_intensity_and_real_zero(tmp_path):
    (tmp_path / "meta").mkdir()
    info = tmp_path / "meta/info.json"
    info.write_text('{}')
    maps = np.full((5, 1, 4, 4), np.nan, dtype=np.float32)
    maps[:, 0, 0, 0] = .8
    maps[2, 0, 0, 0] = 0  # A measured zero must remain a valid target.
    np.save(tmp_path / "maps.npy", maps)
    np.save(tmp_path / "valid.npy", np.ones((5, 1), dtype=bool))
    np.save(tmp_path / "times.npy", np.arange(5) / 10)
    roi = np.zeros((1, 4, 4), dtype=bool)
    roi[0, 0, 0] = True
    np.save(tmp_path / "roi.npy", roi)
    manifest = dict(schema="tactile_intensity_v1", codec="tactile_intensity_v1", fps=10,
                    dataset_info_sha256=hashlib.sha256(info.read_bytes()).hexdigest(),
                    region_names=["right.thumb.tip"], sensor_names=["test"], spatial_valid="roi.npy",
                    episodes={"0": dict(maps="maps.npy", valid="valid.npy", timestamps="times.npy", layout_present=[True])})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    store = IntensityStore(path, tmp_path, size=2, history_times=[0], future_times=[.1, .2])
    window = store.window(0, .1)
    assert window["tactile_valid"].all() and window["future_tactile_valid"].all()
    torch.testing.assert_close(window["tactile_maps"][0, 0, 0, 0, 0], torch.tensor(.8))
    torch.testing.assert_close(window["future_tactile_maps"][0], torch.zeros(1, 1, 2, 2))
    torch.testing.assert_close(window["future_tactile_maps"][1, 0, 0, 0, 0], torch.tensor(.8))
    assert window["tactile_spatial_valid"].sum() == 1


@pytest.mark.parametrize("times,episodes,accepted", [
    ([0, .1, .2, .3], [0]*4, True),
    ([10, 10.1, 10.2, 10.3], [0]*4, True),
    ([0, .1, .24, .3], [0]*4, False),
    ([0, .1, .1, .3], [0]*4, False),
    ([0, .1, .2, .3], [0, 0, 1, 1], False),
])
def test_reader_rejects_irregular_or_mixed_episode_clock(times, episodes, accepted):
    dataset = object.__new__(LeRobotVideoIntensityDataset)
    dataset.fps = 10
    dataset._table = lambda chunk, file: pa.table({"timestamp": times, "episode_index": episodes})
    if accepted:
        assert dataset._episode_clock(0, 0, 0, 4, 0) == (times[0], times[-1])
    else:
        with pytest.raises(ValueError):
            dataset._episode_clock(0, 0, 0, 4, 0)


def test_reader_rejects_half_written_episode_clock():
    dataset = object.__new__(LeRobotVideoIntensityDataset)
    dataset.fps = 10
    dataset._table = lambda chunk, file: pa.table({"timestamp": [0., .1], "episode_index": [0, 0]})
    with pytest.raises(ValueError, match="incomplete"):
        dataset._episode_clock(0, 0, 0, 4, 0)


def test_episode_metadata_reader_projects_only_required_columns(tmp_path):
    import pyarrow.parquet as pq
    from openwam.dataloader.touchwam_pretrain import _load_video_episodes

    folder = tmp_path / 'meta/episodes/chunk-000'
    folder.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0], "length": [4], "dataset_from_index": [0],
                            "data/chunk_index": [0], "data/file_index": [0],
                            "stats/huge/unneeded": [[1.0] * 100], "unneeded_extra": ["metadata"]}), folder / 'file-000.parquet')
    frame = _load_video_episodes(tmp_path, [])
    assert set(frame.columns) == {"episode_index", "length", "dataset_from_index", "data/chunk_index", "data/file_index"}


def test_clock_table_cache_is_shared_and_bounded(tmp_path):
    import pyarrow.parquet as pq
    from openwam.dataloader.touchwam_pretrain import _read_video_clock_table

    _read_video_clock_table.cache_clear()
    for i in range(6):
        path = tmp_path / f'{i}.parquet'
        pq.write_table(pa.table({"timestamp": [0.], "episode_index": [i]}), path)
        _read_video_clock_table(str(path))
    assert _read_video_clock_table.cache_info().currsize == 4
