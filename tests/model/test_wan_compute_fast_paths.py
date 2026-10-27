"""Wan compute fast paths: CausalConv3d without the padded input copy, fp32 RoPE."""

import pytest
import torch
import torch.nn.functional as F

from openwam.model.video_backbone.wan.models import dit as wan_dit
from openwam.model.video_backbone.wan.models.vae import CausalConv3d

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _reference_causal_conv(conv, x, cache_x=None):
    """The original CausalConv3d forward: F.pad over all dims, then an unpadded conv."""
    padding = list(conv._padding)
    if cache_x is not None and conv._padding[4] > 0:
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)
    return F.conv3d(x, conv.weight, conv.bias, conv.stride, 0, conv.dilation, conv.groups)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kernel,padding,stride",
    [(3, 1, 1), ((3, 1, 1), (1, 0, 0), 1), (1, 0, 1), ((3, 1, 1), (0, 0, 0), (2, 1, 1))],
)
@pytest.mark.parametrize("cache_frames", [None, 1, 2])
def test_causal_conv3d_matches_padded_reference(device, kernel, padding, stride, cache_frames):
    torch.manual_seed(0)
    conv = CausalConv3d(4, 6, kernel, padding=padding, stride=stride).to(device, torch.float64)
    x = torch.randn(2, 4, 4, 9, 7, device=device, dtype=torch.float64)
    cache = None if cache_frames is None else torch.randn(2, 4, cache_frames, 9, 7, device=device, dtype=torch.float64)
    out = conv(x, cache)
    ref = _reference_causal_conv(conv, x, cache)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-12)


def _freqs(seq_len, half_dim):
    angles = torch.outer(torch.arange(seq_len).double(), torch.rand(half_dim).double() * 3)
    return torch.polar(torch.ones_like(angles), angles).view(seq_len, 1, half_dim)  # complex128 like dit.freqs


def test_rope_fp32_matches_fp64_reference(monkeypatch):
    torch.manual_seed(0)
    heads, head_dim, seq = 4, 16, 10
    x = torch.randn(3, seq, heads * head_dim, dtype=torch.float32)
    freqs = _freqs(seq, head_dim // 2)
    ref = wan_dit.rope_apply(x, freqs, heads)
    monkeypatch.setattr(wan_dit, "_ROPE_IMPL", "fp32")
    monkeypatch.setattr(wan_dit, "_rope_fp32_compiled", False)  # eager: no compiler toolchain in unit tests
    out = wan_dit.rope_apply(x, freqs, heads)
    assert out.dtype == x.dtype and out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)


def test_rope_fp32_gradient_matches_fp64(monkeypatch):
    torch.manual_seed(0)
    heads, head_dim, seq = 2, 8, 6
    freqs = _freqs(seq, head_dim // 2)
    x = torch.randn(2, seq, heads * head_dim, dtype=torch.float32, requires_grad=True)
    w = torch.randn(2, seq, heads * head_dim)
    (wan_dit.rope_apply(x, freqs, heads) * w).sum().backward()
    ref_grad = x.grad.clone()
    x.grad = None
    monkeypatch.setattr(wan_dit, "_ROPE_IMPL", "fp32")
    monkeypatch.setattr(wan_dit, "_rope_fp32_compiled", False)
    (wan_dit.rope_apply(x, freqs, heads) * w).sum().backward()
    torch.testing.assert_close(x.grad, ref_grad, rtol=1e-6, atol=1e-6)


def test_set_rope_impl_validates():
    with pytest.raises(ValueError):
        wan_dit.set_rope_impl("bf16")
