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

    def __call__(self, prompts, padding="max_length", **kwargs):
        lengths = [self.lengths[p] for p in prompts]
        width = self.max_length if padding == "max_length" else max(lengths)
        ids = torch.zeros(len(prompts), width, dtype=torch.long)
        mask = torch.zeros(len(prompts), width, dtype=torch.long)
        for i, (p, n) in enumerate(zip(prompts, lengths)):
            ids[i, :n] = torch.arange(1, n + 1) + 7 * p
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


# Prompts are ints so the fake tokenizer can give each one a fixed length.
LENGTHS = {0: 3, 1: 12, 2: 1, 3: 7}
PROMPTS = [0, 1, 2, 3, 1, 0, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_encode_text_matches_reference(device):
    tokenizer = _Tokenizer(LENGTHS)
    encoder = _TextEncoder().to(device)
    with torch.no_grad():
        ref, ref_lens = _reference_encode_text(PROMPTS, tokenizer=tokenizer, text_encoder=encoder, device=device)
        out, lens = wan_encode.encode_text(PROMPTS, tokenizer=tokenizer, text_encoder=encoder, device=device)
    assert torch.equal(out, ref)
    assert torch.equal(lens, ref_lens) and lens.device == ref_lens.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("padding", ["max_length", "longest"])
def test_encode_text_dedup_and_longest(device, padding):
    tokenizer = _Tokenizer(LENGTHS)
    encoder = _TextEncoder().to(device)
    with torch.no_grad():
        ref, ref_lens = _reference_encode_text(PROMPTS, tokenizer=tokenizer, text_encoder=encoder, device=device)
        out, lens = wan_encode.encode_text(
            PROMPTS, tokenizer=tokenizer, text_encoder=encoder, device=device, padding=padding, dedup=True
        )
    width = ref.shape[1] if padding == "max_length" else max(LENGTHS.values())
    assert out.shape == (len(PROMPTS), width, ref.shape[2])
    # The row-wise fake encoder makes both paths exact; with UMT5 they differ only
    # by bf16 kernel rounding (padded keys are masked).
    assert torch.equal(out, ref[:, :width]) and not ref[:, width:].any()
    assert torch.equal(lens, ref_lens)


def test_encode_text_rejects_unknown_padding():
    with pytest.raises(ValueError):
        wan_encode.encode_text([0], tokenizer=_Tokenizer(LENGTHS), text_encoder=_TextEncoder(), device="cpu", padding="x")


def test_cross_attention_ignores_trimmed_padding():
    """Masked padded keys contribute nothing: a 512-padded context and the context
    cut to the batch's longest prompt give the same Wan cross-attention output, with
    the proprio token appended after the padding as in ``_append_proprio_context_token``."""
    from openwam.model.video_backbone.wan.models.dit import CrossAttention

    torch.manual_seed(0)
    attn = CrossAttention(dim=32, num_heads=4).eval()
    x = torch.randn(2, 10, 32)
    seq_lens = torch.tensor([5, 9])
    full = torch.randn(2, 16, 32)  # nonzero padding, like text_embedding(0) = bias
    proprio = torch.randn(2, 1, 32)

    def run(ctx_len):
        ctx = torch.cat([full[:, :ctx_len], proprio], dim=1)
        text_mask = torch.arange(ctx_len)[None] < seq_lens[:, None]
        mask = torch.cat([text_mask, torch.ones(2, 1, dtype=torch.bool)], dim=1)
        with torch.no_grad():
            return attn(x, ctx, ctx_mask=mask[:, None, None, :].expand(-1, 1, x.shape[1], -1))

    torch.testing.assert_close(run(9), run(16), rtol=1e-6, atol=1e-6)


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
