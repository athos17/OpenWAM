"""Per-latent-frame Wan time modulation must match the per-token reference (values and grads)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from openwam.model.video_backbone.wan import dit_forward
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan21

B, C, F_LAT, H, W, DIM, HEADS = 2, 4, 3, 4, 6, 32, 4
TOKENS_PER_FRAME = (H // 2) * (W // 2)


def _backbone(ti2v=True, num_layers=2):
    torch.manual_seed(0)
    model = WanModel(
        dim=DIM,
        in_dim=C,
        ffn_dim=64,
        out_dim=C,
        text_dim=16,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=HEADS,
        num_layers=num_layers,
        has_image_input=False,
        seperated_timestep=ti2v,
        fuse_vae_embedding_in_latents=ti2v,
    ).double()
    pipe = SimpleNamespace(dit=model, motion_controller=None, vace=None, use_unified_sequence_parallel=False)
    return Wan21(pipe), model


def _inputs():
    torch.manual_seed(1)
    latents = torch.randn(B, C, F_LAT, H, W, dtype=torch.float64)
    return dict(
        latents=latents,
        timestep=torch.tensor([300.0, 800.0], dtype=torch.float64),
        context=torch.randn(B, 5, 16, dtype=torch.float64),
        fuse_vae_embedding_in_latents=True,
        num_clean_prefix_frames=1,
        force_per_token_t_mod=True,
        zero_clean_prefix_t_mod=True,
        first_frame_latents=latents[:, :, :1].clone(),
    )


def _per_token(t_mod):
    """(B, F, 1, ...) per frame -> (B, F*T, ...) per token."""
    return t_mod.expand(t_mod.shape[0], t_mod.shape[1], TOKENS_PER_FRAME, *t_mod.shape[3:]).flatten(1, 2)


@pytest.mark.parametrize("ti2v", [True, False])
def test_build_time_modulation_per_frame_matches_per_token(ti2v):
    _, model = _backbone(ti2v=ti2v)
    kw = dict(
        patch_size=(1, 2, 2),
        fuse_vae_embedding_in_latents=ti2v,
        force_per_token_t_mod=True,
        num_clean_prefix_frames=1,
        zero_clean_prefix_t_mod=True,
        has_first_frame_latents=True,
    )
    x = _inputs()
    ref_embed, ref_mod = dit_forward.build_time_modulation(model, x["timestep"], x["latents"], **kw)
    embed, mod = dit_forward.build_time_modulation(model, x["timestep"], x["latents"], per_frame=True, **kw)
    assert embed.shape == (B, F_LAT, 1, DIM) and mod.shape == (B, F_LAT, 1, 6, DIM)
    torch.testing.assert_close(_per_token(embed), ref_embed)
    torch.testing.assert_close(_per_token(mod), ref_mod)


def _video_forward(backbone, per_frame):
    backbone.set_frame_time_modulation(per_frame)
    state = backbone.prepare(**_inputs())
    assert state.time_mod.dim() == (5 if per_frame else 4)
    for block_id in range(len(backbone.dit.blocks)):
        state = backbone.run_block(block_id, state)
    return backbone.finalize(state)


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_video_blocks_per_frame_match_per_token():
    backbone, model = _backbone()
    w = None
    results = []
    for per_frame in (False, True):
        model.zero_grad(set_to_none=True)
        out = _video_forward(backbone, per_frame)
        if w is None:
            w = torch.randn_like(out)
        (out * w).sum().backward()
        results.append((out.detach(), _grads(model)))
    (ref, ref_grads), (out, grads) = results
    torch.testing.assert_close(out, ref)
    assert grads.keys() == ref_grads.keys() and any("time_projection" in n for n in grads)
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], msg=name)


def _mot_forward(backbone, per_frame):
    """Wan side of one MoT layer per block: pre half -> plain self-attention -> post half."""
    backbone.set_frame_time_modulation(per_frame)
    state = backbone.prepare(**_inputs())
    for layer_id in range(len(backbone.dit.blocks)):
        q, k, v, post = backbone.pre_attn_at_layer(layer_id, state)
        q, k, v = (rearrange(t, "b s (n d) -> b n s d", n=HEADS) for t in (q, k, v))
        attn = rearrange(F.scaled_dot_product_attention(q, k, v), "b n s d -> b s (n d)")
        state = backbone.post_attn_at_layer(layer_id, state, attn, post)
    return backbone.finalize(state)


def test_mot_halves_per_frame_match_per_token():
    backbone, model = _backbone()
    results = []
    torch.manual_seed(2)
    w = None
    for per_frame in (False, True):
        model.zero_grad(set_to_none=True)
        out = _mot_forward(backbone, per_frame)
        if w is None:
            w = torch.randn_like(out)
        (out * w).sum().backward()
        results.append((out.detach(), _grads(model)))
    (ref, ref_grads), (out, grads) = results
    torch.testing.assert_close(out, ref)
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], msg=name)


def test_trainer_rejects_frame_time_modulation_without_support():
    from openwam.train.openwam_trainer import OpenWAMTrainer

    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg = SimpleNamespace(training=SimpleNamespace(get=lambda k, d=None: {"frame_time_modulation": True}.get(k, d)))
    backbone, _ = _backbone()
    arch = SimpleNamespace(video_backbone=backbone)  # no supports_frame_time_modulation
    with pytest.raises(ValueError, match="supports_frame_time_modulation"):
        trainer._configure_compute_options(arch)
    arch.supports_frame_time_modulation = True
    trainer._configure_compute_options(arch)
    assert backbone._frame_time_modulation is True
