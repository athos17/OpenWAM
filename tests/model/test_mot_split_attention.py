"""Split (unmasked) MoT attention must equal the masked joint attention."""

import types

import pytest
import torch

from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver, plan_split_attention
from openwam.model.architectures.utils.mask_modes import VALID_ATTENTION_MASK_MODES, build_cross_modal_attention_mask
from openwam.model.video_backbone.wan_backbone import WanBase

S_VIDEO, S_ACTION, TOKENS_PER_FRAME, HEADS, HEAD_DIM = 12, 5, 4, 2, 8


def _mask(mode, video_mode):
    vb = types.SimpleNamespace(video_attention_mask_mode=video_mode)
    vb.build_video_to_video_mask = lambda **kw: WanBase.build_video_to_video_mask(vb, **kw)
    return build_cross_modal_attention_mask(
        vb,
        s_video=S_VIDEO,
        s_action=S_ACTION,
        video_tokens_per_frame=TOKENS_PER_FRAME,
        mode=mode,
        device=torch.device("cpu"),
    )


def _driver():
    driver = DualSystemMoTDriver.__new__(DualSystemMoTDriver)
    driver.num_heads = HEADS
    return driver


@pytest.mark.parametrize("video_mode", ["first_frame_causal", "per_frame_causal", "bidirectional"])
@pytest.mark.parametrize("mode", VALID_ATTENTION_MASK_MODES)
def test_split_matches_masked_attention(mode, video_mode):
    mask = _mask(mode, video_mode)
    plan = plan_split_attention(mask)
    if mode in ("mutual", "action_sees_video") and video_mode != "per_frame_causal":
        assert plan is not None  # the production layouts must take the fast path
    if plan is None:
        pytest.skip(f"{mode}/{video_mode} has non-contiguous rows; masked path is kept")
    covered = [q for q0, q1, _, _ in plan for q in range(q0, q1)]
    assert covered == list(range(S_VIDEO + S_ACTION))

    torch.manual_seed(0)
    shape = (2, S_VIDEO + S_ACTION, HEADS * HEAD_DIM)
    q, k, v = (torch.randn(shape, dtype=torch.float64, requires_grad=True) for _ in range(3))
    driver = _driver()
    ref = driver._mixed_attention(q, k, v, mask)
    out = driver._mixed_attention(q, k, v, plan)
    torch.testing.assert_close(out, ref)
    grad = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(ref, (q, k, v), grad)
    out_grads = torch.autograd.grad(out, (q, k, v), grad)
    for a, b in zip(out_grads, ref_grads):
        torch.testing.assert_close(a, b)


def test_plan_rejects_gaps_and_empty_rows():
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[1, 1] = False  # row 1 sees keys {0, 2, 3}: not contiguous
    assert plan_split_attention(mask) is None
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[2] = False
    assert plan_split_attention(mask) is None


def test_plan_groups_rows():
    mask = _mask("mutual", "first_frame_causal")
    # First-frame video rows see only the first frame; every other row sees everything.
    assert plan_split_attention(mask) == (
        (0, TOKENS_PER_FRAME, 0, TOKENS_PER_FRAME),
        (TOKENS_PER_FRAME, S_VIDEO + S_ACTION, 0, S_VIDEO + S_ACTION),
    )
