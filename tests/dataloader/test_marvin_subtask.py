"""Marvin subtask (language-following) mode: window plans, views and RTC delays.

Needs a converted Marvin bucket at ``$OPENWAM_TEST_MARVIN_DIR`` (default
``data/pick_place_2_3_openwam_v3``); skipped otherwise. The annotation is synthetic.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from openwam.model.training_rtc import RTCConfig, condition_training_actions

MARVIN_DIR = Path(os.environ.get("OPENWAM_TEST_MARVIN_DIR", "data/pick_place_2_3_openwam_v3"))
needs_data = pytest.mark.skipif(not (MARVIN_DIR / "meta" / "info.json").is_file(), reason="Marvin bucket not available")
SEG = 120  # synthetic segment length
MAX_DELAY = 8


def test_explicit_rtc_delay_is_clamped_to_contiguous_valid_steps():
    actions = torch.randn(3, 6, 2)
    noisy = torch.randn_like(actions)
    pad = torch.zeros_like(actions, dtype=torch.bool)
    pad[1, 2:] = True  # two real steps -> at most one prefix step
    _, times, mask = condition_training_actions(
        actions, noisy, torch.tensor([500.0, 500.0, 500.0]), pad, RTCConfig(True, 4), delay=torch.tensor([3, 3, 0])
    )
    assert times[:, :4].eq(0).sum(1).tolist() == [3, 1, 0]
    assert mask[0, :3].all() and not mask[0, 3:].any()
    with pytest.raises(ValueError):
        condition_training_actions(actions, noisy, torch.full((3,), 500.0), pad, RTCConfig(True, 4), delay=[-1, 0, 0])


def _annotation(path: Path) -> Path:
    eps = pd.read_parquet(MARVIN_DIR / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    rows = []
    for ep, n in zip(eps.episode_index, eps.length):
        for k, start in enumerate(range(0, int(n), SEG)):
            end = min(int(n), start + SEG)
            still = k % 4 == 3
            rows.append({
                "episode_index": int(ep), "start": start, "end": end, "type": "STILL" if still else "PICK",
                "prompt": None if still else f"Pick up object {k} with the right hand and put it in the bin.",
                "prompt_nohand": None if still else f"Put object {k} into the bin.",
                "variants": [] if still else [f"Grab object {k} and drop it into the bin."],
                "exclude": k == 2,  # third segment of every episode is excluded
            })
    out = path / "subtasks.parquet"
    pd.DataFrame(rows).to_parquet(out)
    return out


def _dataset(annotation=None, **subtask):
    from openwam.dataloader.marvin import MarvinDataset

    cfg = {
        "dataset_dir": str(MARVIN_DIR), "unify_action": True,
        "unify_action_map": ["0-8", "10-31", "34-42", "44-65"], "unify_state_map": ["0-8", "10-31", "34-42", "44-65"],
        "normalize_mode": "min-max", "num_frames": 33, "video_stride": 4, "window_stride": 1,
        "height": 384, "width": 320, "multiview": True, "color_jitter": {"enabled": False},
        "camera_layout": ["observation.images.left_head", "observation.images.left_wrist", "observation.images.right_wrist"],
    }
    if annotation is not None:
        cfg["subtask"] = {"annotation_path": str(annotation), "rtc_max_delay_steps": MAX_DELAY, **subtask}
    return MarvinDataset.from_config(cfg, split="train")


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return _dataset(_annotation(tmp_path_factory.mktemp("subtask")))


@needs_data
def test_training_index_skips_excluded_segments_and_weights_decisions(ds):
    idx = ds._train_index
    seg = np.asarray([ds._subtask.seg_at(e, o) for e, o in zip(ds._window_ep[idx], ds._window_off[idx])])
    assert not ds._subtask.exclude[seg].any()
    assert len(idx) > len(np.unique(idx))  # decision windows are repeated
    assert len(ds) == len(idx) and len(ds.encode_cache_view("main")) == ds._n_total


@needs_data
def test_sub_plans_follow_the_first_generated_step(ds):
    random.seed(0)
    prompts = set(ds.encode_cache_prompts())
    lookup = set(map(tuple, ds._boundary_rows[:, [0, 3]].tolist()))
    seen = {"sub": 0, "main": 0, "still": 0}
    for idx in random.sample(range(ds._n_total), 3000):
        ep, off = int(ds._window_ep[idx]), int(ds._window_off[idx])
        if not ds._subtask.window_ok(ep, off):
            continue
        valid_end = int(ds._ep_valid_end[ep])
        full = min(33, valid_end - off)
        plan = ds._plan_window(idx, ep, off, valid_end)
        mode = plan["extra"]["prompt_mode"]
        seen[mode] += 1
        assert plan["prompt"] in prompts
        d = plan["extra"]["rtc_delay"]
        assert 0 <= d <= MAX_DELAY and d < plan["raw_len"]
        if mode == "main":
            assert plan["raw_len"] == full and not plan.get("hold")
            continue
        s = ds._subtask.seg_at(ep, off + d)
        assert not ds._subtask.exclude[s]
        assert plan["raw_len"] == min(full, int(ds._subtask.end[s]) - off) and plan["hold"]
        if plan["raw_len"] < full:
            assert (idx, plan["raw_len"]) in lookup
    assert all(seen.values()), seen


@needs_data
def test_boundary_view_holds_actions_and_masks_padded_video(ds):
    view = ds.encode_cache_view("boundary")
    k = int(np.flatnonzero(ds._boundary_rows[:, 3] == 10)[0])
    window, ep, off, raw_len = (int(v) for v in ds._boundary_rows[k])
    s = view[k]
    assert s["window_index"] == k
    assert s["video_mask"].tolist() == [True] * 3 + [False] * 6  # sampled frames 0,4,8 are real
    a, m = s["action"].numpy(), s["action_mask"].numpy()
    dims = m.any(0)
    assert m.any(-1).all()  # every step supervised
    np.testing.assert_array_equal(a[raw_len:, dims], np.broadcast_to(a[raw_len - 1, dims], a[raw_len:, dims].shape))
    full = ds.encode_cache_view("main")[window]
    np.testing.assert_array_equal(full["action"].numpy()[:raw_len], a[:raw_len])
    assert view.encode_cache_window_table().shape == (len(ds._boundary_rows), 3)
    assert view.encode_cache_latent_signature()["view"] == "boundary"


@needs_data
def test_main_view_matches_plain_reader(ds):
    plain = _dataset()
    assert len(plain) == ds._n_total
    np.testing.assert_array_equal(plain.encode_cache_window_table(), ds.encode_cache_window_table())
    assert plain.encode_cache_latent_signature() == ds.encode_cache_view("main").encode_cache_latent_signature()
    a, b = plain[12345], ds.encode_cache_view("main")[12345]
    torch.testing.assert_close(a["action"], b["action"])
    assert a["action_mask"].equal(b["action_mask"]) and a["prompt"] == b["prompt"]
