"""Offline encoder cache: format, verification and dataset equivalence.

The dataset-level tests need a small RoboTwin copy (a few clean_50 episodes of
any tasks plus ``meta/robotwin_clean_50_normalization_stats.npy``) at
``$OPENWAM_TEST_ROBOTWIN_DIR`` (default ``outputs/robotwin_subset``); they are
skipped when it is absent. Cache rows are synthetic — the real encoders are
checked on GPU by ``scripts/precompute/build_encode_cache.py check``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from openwam.dataloader.robotwin import _resolve_prompt, candidate_prompts
from openwam.dataloader.utils import encode_cache as ec
from openwam.model.video_backbone.wan import encode as wan_encode

ROBOTWIN_DIR = Path(os.environ.get("OPENWAM_TEST_ROBOTWIN_DIR", "outputs/robotwin_subset"))
needs_data = pytest.mark.skipif(
    not (ROBOTWIN_DIR / "meta" / "robotwin_clean_50_normalization_stats.npy").is_file(),
    reason="RoboTwin subset not available",
)
NUM_SHARDS = 3


# ----------------------------------------------------------------------------
# prompt enumeration
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instr",
    [
        {"seen": ["a", "b", "c"], "unseen": ["z"]},
        {"seen": [], "unseen": ["u1", "u2"]},
        {"seen": [], "instruction": "only"},
        {"seen": [], "instruction": None},
        {"seen": ["x", None]},
        "plain string",
        ["first", "second"],
        [3],
        [],
        None,
    ],
)
@pytest.mark.parametrize("split", ["train", "val"])
def test_candidate_prompts_cover_resolve_prompt(instr, split):
    instructions = {} if instr is None else {"episode7.json": instr}
    candidates = set(candidate_prompts(instructions, "/x/episode7.hdf5", split, "stack cups"))
    rng_state = random.getstate()
    try:
        for seed in range(50):
            random.seed(seed)
            assert _resolve_prompt(instructions, "/x/episode7.hdf5", split, "stack cups") in candidates
    finally:
        random.setstate(rng_state)


# ----------------------------------------------------------------------------
# text row round trip (lossless trim + zero pad)
# ----------------------------------------------------------------------------


def test_load_cached_text_restores_online_context_bitwise():
    g = torch.Generator().manual_seed(0)
    lens = [3, 17, 40, 1]
    context = torch.randn(len(lens), 512, 64, generator=g).to(torch.bfloat16)
    for i, n in enumerate(lens):
        context[i, n:] = 0  # what encode_text guarantees
    lmax = max(lens)
    rows = [context[i, :lmax].clone() for i in range(len(lens))]
    rebuilt, seq_lens = wan_encode.load_cached_text([r[:n] for r, n in zip(rows, lens)], lens, device="cpu")
    assert rebuilt.dtype == torch.bfloat16
    assert torch.equal(rebuilt, context)
    assert seq_lens.dtype == torch.long and seq_lens.tolist() == lens


# ----------------------------------------------------------------------------
# dataset-level
# ----------------------------------------------------------------------------


def _config(**extra):
    cfg = {
        "dataset_dir": str(ROBOTWIN_DIR),
        "embodiment": "aloha-agilex",
        "variant": "clean_50",
        "action_mode": "eef",
        "unify_action": True,
        "unify_action_map": ["0-9", "34-43"],
        "num_frames": 33,
        "video_stride": 4,
        "window_stride": 1,
        "height": 384,
        "width": 320,
        "multiview": True,
        "camera_layout": ["head_camera", "left_camera", "right_camera"],
        "normalize_mode": "min-max",
        "color_jitter": {"enabled": False},
    }
    cfg.update(extra)
    return cfg


def _dataset(**extra):
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset

    return MultiTaskRoboTwinDataset.from_config(_config(**extra), split="train")


def _write_cache(out: Path, kind: str, dataset, row_shape, rows_fn, side: dict, signature, fingerprint, tok=None):
    out.mkdir(parents=True, exist_ok=True)
    n = len(side["_rows"])
    for i in range(NUM_SHARDS):
        start, end = ec.shard_bounds(n, i, NUM_SHARDS)
        path = out / ec.shard_name(i, NUM_SHARDS)
        mm = np.memmap(path, dtype=np.int16, mode="w+", shape=(end - start, *row_shape))
        for r in range(start, end):
            mm[r - start] = rows_fn(r).view(torch.int16).numpy()
        mm.flush()
        del mm
        ec.write_shard_record(
            out,
            i,
            NUM_SHARDS,
            start,
            end,
            path,
            extra={
                "signature_sha256": ec.digest(signature),
                "encoder_fingerprint": fingerprint,
                "tokenizer_sha256": tok,
                "row_shape": list(row_shape),
                "num_rows": n,
                "encode_seconds": 0.0,
            },
        )
    files = {name: ec.sha256_file(out / name) for name in side if name != "_rows"}
    ec.finalize_meta(
        out, kind=kind, signature=signature, num_rows=n, row_shape=row_shape, count=NUM_SHARDS, side_files=files
    )
    ec.verify_cache(out, workers=2)


def _latent_row(r: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + r)
    return torch.randn(48, 3, 24, 20, generator=g).to(torch.bfloat16)


def _text_len(prompt: str) -> int:
    return 5 + len(prompt) % 30


def _text_row(r: int, prompts) -> torch.Tensor:
    g = torch.Generator().manual_seed(r)
    row = torch.randn(35, 64, generator=g).to(torch.bfloat16)
    row[_text_len(prompts[r]) :] = 0
    return row


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    if not (ROBOTWIN_DIR / "meta" / "robotwin_clean_50_normalization_stats.npy").is_file():
        pytest.skip("RoboTwin subset not available")
    root = tmp_path_factory.mktemp("encode_cache")
    dataset = _dataset()
    windows = ec.window_table(dataset)
    np.save(root / "windows.npy", windows)
    latent_dir = root / "latent"
    latent_dir.mkdir()
    np.save(latent_dir / "windows.npy", windows)
    _write_cache(
        latent_dir,
        "latent",
        dataset,
        (48, 3, 24, 20),
        _latent_row,
        {"windows.npy": None, "_rows": range(len(dataset))},
        ec.latent_signature(dataset),
        "vae-fp",
    )
    prompts = ec.candidate_prompt_list(dataset)
    text_dir = root / "text"
    text_dir.mkdir()
    (text_dir / "prompts.json").write_text(__import__("json").dumps(prompts))
    np.save(text_dir / "seq_lens.npy", np.asarray([_text_len(p) for p in prompts], dtype=np.int32))
    _write_cache(
        text_dir,
        "text",
        dataset,
        (35, 64),
        lambda r: _text_row(r, prompts),
        {"prompts.json": None, "seq_lens.npy": None, "_rows": prompts},
        ec.text_signature(dataset),
        "t5-fp",
        tok="tok",
    )
    return {"root": root, "latent": latent_dir, "text": text_dir, "prompts": prompts, "online": dataset}


@needs_data
def test_cached_samples_match_online_samples(caches):
    online = caches["online"]
    cached = _dataset(latent_cache_dir=str(caches["latent"]), text_cache_dir=str(caches["text"]))
    prompts = caches["prompts"]
    indices = [0, 1, len(online) // 2, len(online) - 1] + random.Random(0).sample(range(len(online)), 12)
    for idx in indices:
        random.seed(idx)
        a = online[idx]
        state_a = random.getstate()
        random.seed(idx)
        b = cached[idx]
        assert random.getstate() == state_a, "cached path consumed RNG differently"
        for key in ("action", "action_mask", "video_mask", "proprio", "proprio_mask"):
            assert torch.equal(a[key], b[key]), key
        for key in ("prompt", "episode_path", "start_frame", "end_frame", "episode_length", "task_name"):
            assert a[key] == b[key], key
        assert b["video"] is None and b["first_frame_image"] is None and b["first_frame_from_latent"]
        assert b["video_hw"] == (384, 320) and b["num_video_frames"] == len(a["video"]) == 9
        assert torch.equal(b["latent"], _latent_row(idx))
        row = prompts.index(b["prompt"])
        n = _text_len(b["prompt"])
        assert b["text_len"] == n and torch.equal(b["text_embed"], _text_row(row, prompts)[:n])


@needs_data
def test_text_only_cache_keeps_pixels(caches):
    online = caches["online"]
    cached = _dataset(text_cache_dir=str(caches["text"]))
    random.seed(3)
    a = online[5]
    random.seed(3)
    b = cached[5]
    assert "latent" not in b and len(b["video"]) == 9
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a["video"], b["video"]))
    assert b["text_len"] == _text_len(a["prompt"])


@needs_data
def test_dataloader_workers_and_pinning(caches):
    cached = _dataset(latent_cache_dir=str(caches["latent"]), text_cache_dir=str(caches["text"]))
    loader = torch.utils.data.DataLoader(
        cached, batch_size=4, num_workers=2, collate_fn=list, pin_memory=torch.cuda.is_available()
    )
    batch = next(iter(loader))
    assert isinstance(batch, list) and len(batch) == 4
    for i, sample in enumerate(batch):
        assert torch.equal(sample["latent"], _latent_row(i))
        if torch.cuda.is_available():
            assert sample["latent"].is_pinned() and sample["text_embed"].is_pinned()


@needs_data
def test_cache_rejects_tampering_and_mismatch(caches, tmp_path):
    import shutil

    latent = tmp_path / "latent"
    shutil.copytree(caches["latent"], latent)
    shard = latent / ec.shard_name(1, NUM_SHARDS)
    with open(shard, "r+b") as f:
        f.seek(100)
        byte = f.read(1)
        f.seek(100)
        f.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        ec.verify_cache(latent, workers=2)
    with open(shard, "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="size changed"):
        ec.load_verified_meta(latent, "latent")
    # A different window enumeration must be refused.
    with pytest.raises(ValueError, match="does not match"):
        _dataset(latent_cache_dir=str(caches["latent"]), window_stride=2)
    with pytest.raises(ValueError, match="color_jitter"):
        _dataset(latent_cache_dir=str(caches["latent"]), color_jitter={"enabled": True})
    # A stamp from another host / manifest is not accepted.
    other = tmp_path / "unverified"
    shutil.copytree(caches["text"], other)
    for stamp in other.glob("VERIFIED-*.json"):
        stamp.unlink()
    with pytest.raises(ValueError, match="run verify"):
        ec.load_verified_meta(other, "text")
