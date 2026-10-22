"""TextEmbedCache round trip / lookup parity and batched video preprocessing parity (CPU)."""

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.model.video_backbone.wan.preprocess import preprocess_video, preprocess_video_batch
from openwam.model.video_backbone.wan.text_embed_cache import TextEmbedCache

D, MAX_LEN = 8, 16


def _online(prompts, table):
    """What ``encode_text`` returns: valid rows then exact zeros up to MAX_LEN."""
    ctx = torch.zeros(len(prompts), MAX_LEN, D, dtype=torch.bfloat16)
    for i, p in enumerate(prompts):
        ctx[i, : table[p].shape[0]] = table[p]
    return ctx, torch.tensor([table[p].shape[0] for p in prompts])


@pytest.fixture
def table():
    g = torch.Generator().manual_seed(0)
    lens = {"pick the cup": 3, "open the drawer": 5, "fold": 1, "x" * 10: MAX_LEN}
    return {p: torch.randn(n, D, generator=g).to(torch.bfloat16) for p, n in lens.items()}


def test_save_load_lookup_matches_online(tmp_path, table):
    path = str(tmp_path / "cache.safetensors")
    prompts = sorted(table)
    TextEmbedCache.save(path, prompts, [table[p] for p in prompts], max_length=MAX_LEN, meta={"encode_batch_size": 2})
    cache = TextEmbedCache.load(path)
    assert cache.meta["encode_batch_size"] == 2 and len(cache) == len(table)
    batch = ["fold", "x" * 10, "pick the cup", "fold", "open the drawer"]
    ctx, lens = cache.lookup(batch)
    ref_ctx, ref_lens = _online(batch, table)
    assert torch.equal(ctx, ref_ctx) and torch.equal(lens, ref_lens)


def test_missing_and_online_fallback(table):
    prompts = sorted(table)
    rows = [table[p] for p in prompts]
    offsets = torch.tensor([0] + [r.shape[0] for r in rows]).cumsum(0)
    cache = TextEmbedCache(prompts, torch.cat(rows), offsets, max_length=MAX_LEN)
    assert cache.missing(["fold", "new", "new"]) == ["new"]
    with pytest.raises(KeyError):
        cache.lookup(["new"])
    table["new"] = torch.ones(2, D, dtype=torch.bfloat16)
    cache.add("new", table["new"])
    ctx, lens = cache.lookup(["new", "fold"])
    ref_ctx, ref_lens = _online(["new", "fold"], table)
    assert torch.equal(ctx, ref_ctx) and torch.equal(lens, ref_lens)


def _clip(rng, t, h, w):
    return [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for _ in range(t)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_preprocess_video_batch_bitwise(dtype):
    rng = np.random.default_rng(0)
    clips = [_clip(rng, 5, 24, 20) for _ in range(3)]
    legacy = torch.cat([preprocess_video(c, dtype=dtype, device="cpu") for c in clips], dim=0)
    fast = preprocess_video_batch(clips, dtype=dtype, device="cpu")
    assert fast.shape == legacy.shape == (3, 3, 5, 24, 20)
    assert fast.is_contiguous() and torch.equal(fast, legacy)


def test_preprocess_video_batch_non_pil_takes_legacy_path():
    rng = np.random.default_rng(1)
    clips = [[np.asarray(im) for im in _clip(rng, 2, 8, 8)] for _ in range(2)]  # ndarray frames
    legacy = torch.cat([preprocess_video(c, dtype=torch.float32, device="cpu") for c in clips], dim=0)
    assert torch.equal(preprocess_video_batch(clips, dtype=torch.float32, device="cpu"), legacy)
