"""Train-time fused Wan DiT block (``training.dit_block_fusion``).

A Wan block between its two attention calls is a chain of small memory-bound ops
(LayerNorm, AdaLN modulate, RMSNorm on q/k, RoPE, gates, residual adds, dtype casts)
around the q/k/v/o/FFN GEMMs. Here each stretch runs as one ``torch.compile`` region,
so inductor fuses those ops (forward and backward) while the GEMMs stay on cuBLAS:

    pre_attn        norm1 + modulate -> q/k/v (+ RMSNorm, RoPE)
    (self / joint attention, eager)
    post_self_attn  o-proj + gate + residual -> norm3 -> cross-attn q (+ RMSNorm)
    (cross-attention k/v over the text and SDPA, eager: text length varies per batch)
    post_cross_attn cross o-proj + residual -> norm2 + modulate -> FFN + gate

RoPE is applied as a real rotation in the dtype of ``cos``/``sin``: float64 matches
``rope_apply``'s complex128 product, float32 the ``rope_impl=fp32`` kernel. Same math
as ``DiTBlock.forward``; fusion keeps fp32 intermediates, so results change at the
bf16 rounding level. ``inline_inbuilt_nn_modules`` lets all blocks share one graph per
input shape.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from openwam.model.compile_options import compile_outside_dispatch_modes, raise_recompile_limit
from openwam.model.video_backbone.wan.models.dit import flash_attention, modulate, split_time_modulation


def rope_real(x: Tensor, cos: Tensor, sin: Tensor, num_heads: int) -> Tensor:
    b, s, _ = x.shape
    xr = x.to(cos.dtype).reshape(b, s, num_heads, -1, 2)
    x1, x2 = xr[..., 0], xr[..., 1]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(2).to(x.dtype)


def rope_cos_sin(freqs: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    return freqs.real.to(dtype).contiguous(), freqs.imag.to(dtype).contiguous()


def pre_attn(block, x: Tensor, time_mod: Tensor, cos: Tensor, sin: Tensor):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split_time_modulation(block.modulation, time_mod)
    attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
    sa = block.self_attn
    q = rope_real(sa.norm_q(sa.q(attn_input)), cos, sin, sa.num_heads)
    k = rope_real(sa.norm_k(sa.k(attn_input)), cos, sin, sa.num_heads)
    v = sa.v(attn_input)
    return q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp


def post_self_attn(block, x: Tensor, gate_msa: Tensor, attn_out: Tensor):
    x = block.gate(x, gate_msa, block.self_attn.o(attn_out))
    ca = block.cross_attn
    return x, ca.norm_q(ca.q(block.norm3(x)))


def cross_attn_core(block, q: Tensor, context: Tensor, context_mask: Optional[Tensor]) -> Tensor:
    ca = block.cross_attn
    k = ca.norm_k(ca.k(context))
    v = ca.v(context)
    return flash_attention(q, k, v, num_heads=ca.num_heads, attn_mask=context_mask)


def post_cross_attn(block, x: Tensor, cross_out: Tensor, shift_mlp: Tensor, scale_mlp: Tensor, gate_mlp: Tensor):
    x = x + block.cross_attn.o(cross_out)
    return block.gate(x, gate_mlp, block.ffn(modulate(block.norm2(x), shift_mlp, scale_mlp)))


_FNS = {"pre_attn": pre_attn, "post_self_attn": post_self_attn, "post_cross_attn": post_cross_attn}
_compiled: dict = {}


def compiled(name: str):
    fn = _compiled.get(name)
    if fn is None:
        # One specialization per (robot / video-only, data source) token layout.
        raise_recompile_limit(64)
        fn = compile_outside_dispatch_modes(_FNS[name], dynamic=False)
        _compiled[name] = fn
    return fn


def block_forward(
    block,
    x: Tensor,
    context: Tensor,
    context_mask: Optional[Tensor],
    time_mod: Tensor,
    cos: Tensor,
    sin: Tensor,
    self_attn_mask: Optional[Tensor] = None,
) -> Tensor:
    """``DiTBlock.forward`` on the fused regions (video-only block loop)."""
    if context_mask is not None and context_mask.dim() == 3:
        context_mask = context_mask.unsqueeze(1)
    q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp = compiled("pre_attn")(block, x, time_mod, cos, sin)
    attn = block.self_attn.attn(q, k, v, attn_mask=self_attn_mask)
    x, cross_q = compiled("post_self_attn")(block, x, gate_msa, attn)
    cross_out = cross_attn_core(block, cross_q, context, context_mask)
    return compiled("post_cross_attn")(block, x, cross_out, shift_mlp, scale_mlp, gate_mlp)


__all__ = ["block_forward", "compiled", "cross_attn_core", "rope_cos_sin", "rope_real"]
