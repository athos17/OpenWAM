"""Wan train-input fast paths must be bitwise identical to the reference paths."""

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.preprocess import preprocess_video, preprocess_video_batch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _Tokenizer:
    """Right-padded ids/mask like the UMT5 HuggingfaceTokenizer (max_length padding)."""

    def __init__(self, lengths, max_length=12):
        self.lengths = lengths
        self.max_length = max_length

    def __call__(self, prompts, **kwargs):
        ids = torch.zeros(len(prompts), self.max_length, dtype=torch.long)
        mask = torch.zeros(len(prompts), self.max_length, dtype=torch.long)
        for i, n in enumerate(self.lengths):
            ids[i, :n] = torch.arange(1, n + 1) + 7 * i
            mask[i, :n] = 1
        return ids, mask


class _TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(64, 8)

    def forward(self, ids, mask):
        return self.emb(ids) + 0.5  # nonzero at padded positions too


def _reference_encode_text(prompts, *, tokenizer, text_encoder, device):
    ids, mask = tokenizer(prompts)
    ids, mask = ids.to(device), mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[i, v:] = 0
    return context, seq_lens


@pytest.mark.parametrize("device", DEVICES)
def test_encode_text_matches_reference(device):
    tokenizer = _Tokenizer([3, 12, 1, 7])
    encoder = _TextEncoder().to(device)
    prompts = ["a", "b", "c", "d"]
    with torch.no_grad():
        ref, ref_lens = _reference_encode_text(prompts, tokenizer=tokenizer, text_encoder=encoder, device=device)
        out, lens = wan_encode.encode_text(prompts, tokenizer=tokenizer, text_encoder=encoder, device=device)
    assert torch.equal(out, ref)
    assert torch.equal(lens, ref_lens) and lens.device == ref_lens.device


def _clip(rng, frames=5, h=8, w=6):
    return [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for _ in range(frames)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_preprocess_video_batch_bitwise(device, dtype):
    rng = np.random.default_rng(0)
    clips = [_clip(rng) for _ in range(3)]
    ref = torch.cat([preprocess_video(c, dtype=dtype, device=device) for c in clips])
    out = preprocess_video_batch(clips, dtype=dtype, device=device)
    assert out.shape == ref.shape == (3, 3, 5, 8, 6)
    assert out.dtype == ref.dtype and out.is_contiguous()
    assert torch.equal(out, ref)


def test_preprocess_video_batch_falls_back_on_ragged_clips():
    rng = np.random.default_rng(1)
    clips = [_clip(rng, frames=4), _clip(rng, frames=6)]
    # Different clip lengths skip the uint8 stack; the per-clip path then fails on
    # the concat exactly as the reference loop would.
    with pytest.raises(RuntimeError):
        preprocess_video_batch(clips, dtype=torch.float32, device="cpu")
