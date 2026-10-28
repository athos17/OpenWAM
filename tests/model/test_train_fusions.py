"""Train-time fusions (training.dit_block_fusion, training.vae_fused_norm) must match the eager path."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from openwam.model.video_backbone.wan import fused_block
from openwam.model.video_backbone.wan.models import vae as wan_vae
from openwam.model.video_backbone.wan.models.dit import WanModel, rope_apply
from openwam.model.video_backbone.wan_backbone import Wan21

B, C, F_LAT, H, W, DIM, HEADS = 2, 4, 3, 4, 6, 32, 4


def _backbone(num_layers=2):
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
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
    ).double()
    pipe = SimpleNamespace(dit=model, motion_controller=None, vace=None, use_unified_sequence_parallel=False)
    return Wan21(pipe), model


def _inputs(gc=False):
    torch.manual_seed(1)
    latents = torch.randn(B, C, F_LAT, H, W, dtype=torch.float64)
    return dict(
        latents=latents,
        timestep=torch.tensor([300.0, 800.0], dtype=torch.float64),
        context=torch.randn(B, 5, 16, dtype=torch.float64),
        context_mask=torch.tensor([[True] * 5, [True] * 3 + [False] * 2]),
        fuse_vae_embedding_in_latents=True,
        num_clean_prefix_frames=1,
        force_per_token_t_mod=True,
        zero_clean_prefix_t_mod=True,
        first_frame_latents=latents[:, :, :1].clone(),
        use_gradient_checkpointing=gc,
    )


def _video_forward(backbone, gc):
    state = backbone.prepare(**_inputs(gc))
    for block_id in range(len(backbone.dit.blocks)):
        state = backbone.run_block(block_id, state)
    return backbone.finalize(state)


def _mot_forward(backbone, gc):
    """Wan side of one MoT layer per block: pre half -> plain self-attention -> post half."""
    state = backbone.prepare(**_inputs(gc))
    for layer_id in range(len(backbone.dit.blocks)):
        q, k, v, post = backbone.pre_attn_at_layer(layer_id, state)
        q, k, v = (rearrange(t, "b s (n d) -> b n s d", n=HEADS) for t in (q, k, v))
        attn = rearrange(F.scaled_dot_product_attention(q, k, v), "b n s d -> b s (n d)")
        state = backbone.post_attn_at_layer(layer_id, state, attn, post)
    return backbone.finalize(state)


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_rope_real_matches_complex_rope_apply():
    torch.manual_seed(3)
    backbone, _ = _backbone()
    freqs = backbone.prepare(**_inputs()).rope_freqs
    x = torch.randn(B, freqs.shape[0], DIM, dtype=torch.float64)
    cos, sin = fused_block.rope_cos_sin(freqs, torch.float64)
    torch.testing.assert_close(fused_block.rope_real(x, cos, sin, HEADS), rope_apply(x, freqs, HEADS))


@pytest.mark.parametrize("path", ["video_blocks", "mot_halves"])
@pytest.mark.parametrize("per_frame", [False, True])
@pytest.mark.parametrize("gc", [False, True])
def test_block_fusion_matches_eager(path, per_frame, gc):
    backbone, model = _backbone()
    backbone.set_frame_time_modulation(per_frame)
    forward = _video_forward if path == "video_blocks" else _mot_forward
    w = None
    results = []
    for fused in (False, True):
        backbone.set_block_fusion(fused)
        model.zero_grad(set_to_none=True)
        out = forward(backbone, gc)
        if w is None:
            torch.manual_seed(2)
            w = torch.randn_like(out)
        (out * w).sum().backward()
        results.append((out.detach(), _grads(model)))
    (ref, ref_grads), (out, grads) = results
    # RMSNorm computes in float32 even for a float64 model, and the fused kernel sums its
    # reduction in another order: agreement is at the float32 level.
    tol = dict(rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out, ref, **tol)
    assert grads.keys() == ref_grads.keys()
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], msg=name, **tol)


def test_block_fusion_compiles_after_a_flop_counted_first_step():
    """The trainer counts FLOPs on step 1 under FlopCounterMode; that call must not keep
    the fused regions eager for the rest of training."""
    import torch._dynamo.utils as dynamo_utils
    from torch.utils.flop_counter import FlopCounterMode

    torch._dynamo.reset()
    fused_block._compiled.clear()
    dynamo_utils.counters.clear()
    backbone, _ = _backbone()
    backbone.set_block_fusion(True)
    with FlopCounterMode(display=False):
        _video_forward(backbone, gc=False)
    assert dynamo_utils.counters["stats"]["unique_graphs"] == 0
    _video_forward(backbone, gc=False).sum().backward()
    assert dynamo_utils.counters["stats"]["unique_graphs"] == 3


def _small_vae():
    torch.manual_seed(4)
    model = wan_vae.VideoVAE38_(dim=16, z_dim=4, dec_dim=16).eval()
    scale = [torch.randn(4), torch.rand(4) + 0.5]
    return model, scale


@pytest.mark.parametrize("channels_last", [False, True])
def test_vae_fused_norm_matches_eager(channels_last):
    model, scale = _small_vae()
    torch.manual_seed(5)
    x = torch.rand(2, 3, 9, 32, 32) * 2 - 1
    if channels_last:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv3d):
                m.to(memory_format=torch.channels_last_3d)
        x = x.contiguous(memory_format=torch.channels_last_3d)
    with torch.no_grad():
        ref = model.encode(x, scale)
        with wan_vae.vae_fusion(True):
            out = model.encode(x, scale)
    assert not wan_vae._VAE_FUSION  # scoped to the context
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def _trainer(options):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg = SimpleNamespace(training=SimpleNamespace(get=lambda k, d=None: options.get(k, d)))
    return trainer


def test_trainer_wires_fusion_options():
    backbone, _ = _backbone()
    _trainer({"dit_block_fusion": True})._configure_compute_options(SimpleNamespace(video_backbone=backbone))
    assert backbone._block_fusion is True
    with pytest.raises(ValueError, match="dit_block_fusion"):
        _trainer({"dit_block_fusion": True})._configure_compute_options(SimpleNamespace(video_backbone=None))

    seen = {}
    vb = SimpleNamespace(set_vae_encode_options=lambda **kw: seen.update(kw))
    _trainer({"vae_fused_norm": True})._configure_input_fast_paths(SimpleNamespace(video_backbone=vb))
    assert seen == {"cudnn_benchmark": False, "channels_last": False, "fused_norm": True}
