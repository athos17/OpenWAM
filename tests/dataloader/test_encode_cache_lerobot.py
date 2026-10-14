"""Offline encoder caches on a LeRobot-v3 reader (MarvinDataset).

Needs a converted Marvin bucket at ``$OPENWAM_TEST_MARVIN_DIR`` (default
``outputs/sharpa_data/pick_place_2_openwam_v3``); skipped otherwise. Cache rows
are synthetic and tiny — the real encoders are checked on GPU by
``scripts/precompute/build_encode_cache.py check``.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from openwam.dataloader.utils import encode_cache as ec
from tests.dataloader.test_encode_cache import _write_cache

MARVIN_DIR = Path(os.environ.get("OPENWAM_TEST_MARVIN_DIR", "outputs/sharpa_data/pick_place_2_openwam_v3"))
pytestmark = pytest.mark.skipif(not (MARVIN_DIR / "meta" / "info.json").is_file(), reason="Marvin bucket not available")
ROW = (3, 2, 2, 2)  # tiny stand-in for (48, 3, 24, 20)


def _config(**extra):
    cfg = {
        "dataset_dir": str(MARVIN_DIR),
        "action_mode": "eef",
        "unify_action": True,
        "unify_action_map": ["0-8", "10-31", "34-42", "44-65"],
        "unify_state_map": ["0-8", "10-31", "34-42", "44-65"],
        "normalize_mode": "min-max",
        "num_frames": 33,
        "video_stride": 4,
        "window_stride": 1,
        "height": 384,
        "width": 320,
        "multiview": True,
        "camera_layout": ["observation.images.left_head", "observation.images.left_wrist", "observation.images.right_wrist"],
        "color_jitter": {"enabled": False},
        "total_hours": 0.02,  # a few episodes keep the synthetic cache small
    }
    cfg.update(extra)
    return cfg


def _dataset(**extra):
    from openwam.dataloader.marvin import MarvinDataset

    return MarvinDataset.from_config(_config(**extra), split="train")


def _latent_row(r: int) -> torch.Tensor:
    return torch.randn(*ROW, generator=torch.Generator().manual_seed(7000 + r)).to(torch.bfloat16)


def _text_row(prompt_len: int) -> torch.Tensor:
    row = torch.randn(30, 16, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    row[prompt_len:] = 0
    return row


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    root = tmp_path_factory.mktemp("marvin_cache")
    online = _dataset()
    latent = root / "latent"
    latent.mkdir()
    np.save(latent / "windows.npy", ec.window_table(online))
    _write_cache(latent, "latent", online, ROW, _latent_row, {"windows.npy": None, "_rows": range(len(online))},
                 ec.latent_signature(online), "vae-fp")
    prompts = ec.candidate_prompt_list(online)
    text = root / "text"
    text.mkdir()
    (text / "prompts.json").write_text(json.dumps(prompts))
    np.save(text / "seq_lens.npy", np.asarray([11] * len(prompts), dtype=np.int32))
    _write_cache(text, "text", online, (30, 16), lambda r: _text_row(11), {"prompts.json": None, "seq_lens.npy": None, "_rows": prompts},
                 ec.text_signature(online), "t5-fp", tok="tok")
    return {"latent": latent, "text": text, "online": online, "prompts": prompts}


def test_window_table_matches_getitem(caches):
    ds = caches["online"]
    table = ec.window_table(ds)
    assert table.shape == (len(ds), 2)
    for idx in [0, 1, len(ds) // 2, len(ds) - 1]:
        ep = int(np.searchsorted(ds._cum_n_starts, idx, side="right") - 1)
        offset = int(ds._ep_valid_start[ep]) + (idx - int(ds._cum_n_starts[ep])) * ds._window_stride
        assert tuple(table[idx]) == (ep, offset)
    assert ds[5]["window_index"] == 5


def test_cached_samples_match_online(caches):
    online = caches["online"]
    cached = _dataset(latent_cache_dir=str(caches["latent"]), text_cache_dir=str(caches["text"]))
    assert len(cached) == len(online)
    for idx in [0, 3, len(online) // 2, len(online) - 1] + random.Random(1).sample(range(len(online)), 6):
        random.seed(idx)
        a = online[idx]
        state = random.getstate()
        random.seed(idx)
        b = cached[idx]
        assert random.getstate() == state
        for key in ("action", "action_mask", "video_mask", "proprio", "proprio_mask"):
            assert torch.equal(a[key], b[key]), key
        assert a["prompt"] == b["prompt"] and a["window_index"] == b["window_index"] == idx
        assert b["video"] is None and b["first_frame_image"] is None and b["first_frame_from_latent"]
        assert b["video_hw"] == (384, 320) and b["num_video_frames"] == len(a["video"]) == 9
        assert torch.equal(b["latent"], _latent_row(idx))
        assert b["text_len"] == 11 and torch.equal(b["text_embed"], _text_row(11)[:11])


def test_latent_cache_refuses_jitter_and_mismatch(caches):
    with pytest.raises(ValueError, match="color_jitter"):
        _dataset(latent_cache_dir=str(caches["latent"]), color_jitter={"enabled": True})
    with pytest.raises(ValueError, match="does not match"):
        _dataset(latent_cache_dir=str(caches["latent"]), num_frames=17)


def test_dataloader_pins_cached_tensors(caches):
    cached = _dataset(latent_cache_dir=str(caches["latent"]), text_cache_dir=str(caches["text"]))
    loader = torch.utils.data.DataLoader(cached, batch_size=4, num_workers=2, collate_fn=list,
                                         pin_memory=torch.cuda.is_available())
    batch = next(iter(loader))
    for i, sample in enumerate(batch):
        assert sample["window_index"] == i and torch.equal(sample["latent"], _latent_row(i))
        if torch.cuda.is_available():
            assert sample["latent"].is_pinned() and sample["text_embed"].is_pinned()
