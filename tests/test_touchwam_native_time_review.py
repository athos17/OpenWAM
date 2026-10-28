"""Native float32 row clocks must address their exact float64 tactile frames."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from openwam.dataloader.touchwam_pretrain import SharpaNativeIntensityDataset
from openwam.tactile.intensity_store import IntensityStore


def reader(tmp_path):
    (tmp_path / "meta").mkdir()
    info = tmp_path / "meta/info.json"
    info.write_text("{}")
    times = np.arange(34, dtype=np.float64) / 30
    np.save(tmp_path / "timestamps.npy", times)
    np.save(tmp_path / "valid.npy", np.ones((34, 1), dtype=bool))
    np.save(tmp_path / "maps.npy", np.arange(34, dtype=np.float32)[:, None, None, None] / 33)
    manifest = {"schema": "tactile_intensity_v1", "codec": "tactile_intensity_v1", "fps": 30,
                "dataset_info_sha256": hashlib.sha256(info.read_bytes()).hexdigest(),
                "region_names": ["right.thumb.tip"], "sensor_names": ["sharpa_real"],
                "episodes": {"0": {"timestamps": "timestamps.npy", "valid": "valid.npy",
                                     "maps": "maps.npy", "layout_present": [True]}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    ds = object.__new__(SharpaNativeIntensityDataset)
    ds._fps = 30
    ds.touch = IntensityStore(path, tmp_path, size=1)
    return ds


@pytest.mark.parametrize("offset", [31, 33])
def test_native_clock_rounding_keeps_current_frame_and_episode_tail(tmp_path, offset):
    ds = reader(tmp_path)
    # 31/30 rounds down, 33/30 rounds up. Both must select this exact frame.
    recorded = np.asarray([offset / 30], dtype=np.float32)
    result = ds._observation_extras(0, SimpleNamespace(episode_index=0), offset, 0, 34,
                                    pd.DataFrame({"timestamp": recorded}))
    assert result["tactile_valid"][-1, 0]
    assert result["tactile_maps"][-1, 0, 0, 0, 0].item() == pytest.approx(offset / 33)
    assert result["tactile_history_source_times_s"][-1].item() == 0
    assert not result["future_tactile_valid"].any()


def test_native_clock_rejects_real_offset_and_preserves_segment_start(tmp_path):
    ds = reader(tmp_path)
    row = SimpleNamespace(episode_index=0)
    with pytest.raises(ValueError, match="verified episode-relative frame clock"):
        ds._observation_extras(0, row, 30, 30, 34, pd.DataFrame({"timestamp": [1.01]}))
    good = ds._observation_extras(0, row, 30, 30, 34,
                                  pd.DataFrame({"timestamp": np.asarray([1.0], dtype=np.float32)}))
    assert not good["tactile_valid"][:-1].any()
    assert good["tactile_valid"][-1].all()


def test_preflight_accepts_action_supervision_in_native_video_tail():
    from scripts.preflight_touchwam import validate_sample

    sample = {"training_modality": "video_action", "prompt": "Move the object.",
              "video": [Image.new("RGB", (32, 48))] * 9,
              "video_mask": torch.tensor([True] + [False] * 8),
              "action": torch.zeros(32, 80), "action_mask": torch.zeros(32, 80, dtype=torch.bool),
              "proprio": torch.zeros(1, 80), "proprio_mask": torch.ones(1, 80, dtype=torch.bool)}
    sample["action_mask"][0, :62] = True
    assert validate_sample(sample, "video_action")["video_observed_frames"] == 1
    sample["action_mask"].zero_()
    with pytest.raises(ValueError, match="no future"):
        validate_sample(sample, "video_action")
