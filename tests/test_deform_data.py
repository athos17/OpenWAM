import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from openwam.dataloader.bench2dex_deform import Bench2DexDeformDataset
from openwam.dataloader.marvin_deform import MarvinDeformDataset
from openwam.tactile.deform_store import AlignedDeformStore, causal_match
from openwam.tactile.regions import CODEC, FINGERTIP_NAMES, SCHEMA
from openwam.tactile.sharpa import SITE_NAMES


def sidecar(root, info_path, episodes=(1, 2), frames=40):
    root.mkdir()
    entries = {}
    for ep in episodes:
        maps = np.broadcast_to(
            np.arange(frames, dtype=np.uint8)[:, None, None, None] + ep * 50, (frames, 10, 8, 8)
        ).copy()
        valid = np.ones((frames, 10), bool)
        if ep == 1:
            valid[8, 0] = False
        arrays = dict(maps=maps, valid=valid, timestamps=np.arange(frames, dtype=np.float64) / 30)
        entry = dict(layout_present=[True] * 10)
        for name, value in arrays.items():
            path = root / f"{ep}.{name}.npy"
            np.save(path, value)
            entry[name] = path.name
        entries[str(ep)] = entry
    meta = dict(
        schema=SCHEMA,
        codec=CODEC,
        fps=30,
        region_names=list(FINGERTIP_NAMES),
        sensor_ids=[0] * 10,
        dataset_info_sha256=hashlib.sha256(info_path.read_bytes()).hexdigest(),
        episodes=entries,
    )
    path = root / "manifest.json"
    path.write_text(json.dumps(meta))
    return path


def test_causal_match_rejects_future_and_stale_samples():
    idx, valid = causal_match([0, 0.05, 0.1, 1], [0.02, 0.08, 0.11], 0.04)
    assert idx.tolist() == [-1, 0, 1, 2]
    assert valid.tolist() == [False, True, True, False]
    with pytest.raises(ValueError, match="increasing"):
        causal_match([0], [0.1, 0.1], 0.04)


def test_sidecar_alignment_boundary_and_missing_future(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    info = meta / "info.json"
    info.write_text("{}")
    path = sidecar(tmp_path / "touch", info)
    store = AlignedDeformStore(path, tmp_path, size=8, fps=30)
    args = dict(history=4, future_frames=2, future_stride=3)
    value = store.window(1, 0, 0, 40, 0, **args)
    assert value["tactile_valid"].sum() == 10
    torch.testing.assert_close(value["future_tactile_maps"][:, 0, 0, 0, 0], torch.tensor([53, 56]) / 255)
    with pytest.raises(ValueError, match="Incomplete"):
        store.window(1, 5, 0, 40, 5 / 30, **args)
    with pytest.raises(ValueError, match="misaligned"):
        store.window(1, 0, 0, 40, 1 / 30, **args)
    with pytest.raises(ValueError, match="Incomplete"):
        store.window(1, 37, 0, 40, 37 / 30, **args)
    info.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity"):
        AlignedDeformStore(path, tmp_path, size=8, fps=30)


def test_bench2dex_reads_real_saved_gray_not_geometry(tmp_path):
    ds = object.__new__(Bench2DexDeformDataset)
    ds.history, ds.map_size, ds.future_frames, ds.future_stride = 4, 8, 2, 2
    with h5py.File(tmp_path / "sim.h5", "w") as f:
        f.create_dataset(f"robot/tactile/distance_along_normal_m/{SITE_NAMES[0]}", data=np.ones((10, 8, 8)))
        with pytest.raises(ValueError, match="No stored"):
            ds.observation_extras(f, {"segments": [[0, 10]]}, 1, 0)
        gray = np.broadcast_to(np.arange(10, dtype=np.uint8)[:, None, None], (10, 8, 8))
        f.create_dataset(f"robot/tactile/tacmap/{SITE_NAMES[0]}", data=gray)
        sample = ds.observation_extras(f, {"segments": [[0, 10]]}, 1, 0)
    assert sample["tactile_layout_present"].sum() == 1
    assert sample["tactile_valid"][:, 0].tolist() == [False, False, True, True]
    assert sample["tactile_valid"][2, 0]  # valid zero intensity
    torch.testing.assert_close(sample["future_tactile_maps"][:, 0, 0, 0, 0], torch.tensor([3, 5]) / 255)
    assert sample["tactile_codec"] == CODEC


def test_marvin_actual_window_mapping_survives_retry(tmp_path, monkeypatch):
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    info_path = tmp_path / "meta/info.json"
    info_path.write_text(
        json.dumps(
            dict(
                fps=30,
                codebase_version="v3.0",
                splits={"train": "0:3"},
                openwam_contract=dict(raw_action_dim=62, rotation_6d_layout="[R00,R10,R20,R01,R11,R21]"),
            )
        )
    )
    pd.DataFrame(dict(task_index=[0], task=["pick"])).to_parquet(tmp_path / "meta/tasks.parquet")
    episodes, rows = [], []
    for ep in range(3):
        record = dict(
            episode_index=ep, length=40, dataset_from_index=ep * 40, **{"data/chunk_index": 0, "data/file_index": 0}
        )
        for cam in ["observation.images.left_head", "observation.images.left_wrist", "observation.images.right_wrist"]:
            record[f"videos/{cam}/chunk_index"] = 0
            record[f"videos/{cam}/file_index"] = 0
        episodes.append(record)
        for t in range(40):
            values = np.full(76, ep + t / 100, dtype=np.float32).tolist()
            rows.append(dict(action=values, timestamp=t / 30, task_index=0, **{"observation.state": values}))
    pd.DataFrame(episodes).to_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "data/chunk-000/file-000.parquet")
    path = sidecar(tmp_path / "touch", info_path)
    calls = []

    def decode(self, row, ep_local, offset, local_indices, idx, **kwargs):
        calls.append((int(row["episode_index"]), offset))
        if len(calls) == 1:
            raise OSError("one transient video failure")
        return [Image.new("RGB", (16, 16)) for _ in range(9)]

    monkeypatch.setattr(MarvinDeformDataset, "_decode_window_video", decode)
    ds = MarvinDeformDataset(
        str(tmp_path),
        tactile_manifest_path=str(path),
        tactile_map_size=8,
        normalize_mode="none",
        unify_action=True,
        unify_action_map=["0-8", "10-31", "34-42", "44-65"],
        unify_state_map=["0-8", "10-31", "34-42", "44-65"],
        height=16,
        width=16,
    )
    assert ds._touch_indices.min() >= 40  # episode 0 had no touch, never fabricated
    sample = ds[0]
    ep, offset = calls[-1]
    assert ep == 1 and offset != calls[0][1]
    assert sample["action"].shape == (32, 80)
    assert sample["action"][0, 0] == pytest.approx(ep + offset / 100)
    assert sample["tactile_maps"][-1, 0, 0, 0, 0] == pytest.approx((ep * 50 + offset) / 255)
    assert sample["future_tactile_maps"][0, 0, 0, 0, 0] == pytest.approx((ep * 50 + offset + 3) / 255)
    assert ds.training_data_contract["selected_windows"] == len(ds)
