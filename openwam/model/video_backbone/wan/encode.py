"""Wan VAE / text-encoder IO with native-VAE ↔ external-encoder routing.

Free functions — ``vae`` / ``encoder`` / ``tokenizer`` / ``text_encoder`` passed
explicitly so the backbone class holds no IO logic. ``encoder is not None``
selects the external-encoder path; otherwise the native Wan VAE is used.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.wan.preprocess import preprocess_video as _preprocess_video_native
from openwam.model.video_backbone.wan.preprocess import preprocess_video_batch as _preprocess_video_batch_native
from openwam.model.video_backbone.wan.preprocess import vae_output_to_video


TEXT_PADDING_MODES = ("max_length", "longest")


def encode_text(
    prompts: list, *, tokenizer, text_encoder, device, padding: str = "max_length", dedup: bool = False
) -> Tuple[Tensor, Tensor]:
    """UMT5-encode ``prompts`` -> ``(context [B, L, D], seq_lens [B])``, zero past each length.

    ``padding="max_length"`` pads every prompt to 512 tokens (the reference path);
    ``"longest"`` pads to the longest prompt in the batch, so ``L`` shrinks. Padded
    keys are masked in UMT5 and in every downstream cross-attention (``seq_lens`` ->
    context mask), so both give the same result up to bf16 kernel rounding.
    ``dedup`` encodes each distinct prompt once and gathers the rows back.
    """
    if padding not in TEXT_PADDING_MODES:
        raise ValueError(f"text padding must be one of {TEXT_PADDING_MODES}, got {padding!r}")
    if dedup:
        rows = {p: i for i, p in enumerate(dict.fromkeys(prompts))}
        if len(rows) < len(prompts):
            context, seq_lens = encode_text(
                list(rows), tokenizer=tokenizer, text_encoder=text_encoder, device=device, padding=padding
            )
            index = torch.tensor([rows[p] for p in prompts], dtype=torch.long).to(device)
            return context.index_select(0, index), seq_lens.index_select(0, index)
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=512,
        padding=padding,
        truncation=True,
    )
    # Lengths and the zero-fill mask come from the CPU-side tokenizer mask: slicing
    # ``context[i, v:]`` with a device scalar ``v`` would block the host until the
    # encoder finishes, serializing the CPU video preprocessing behind it.
    seq_lens = mask.gt(0).sum(dim=1).long()
    pad = torch.arange(mask.shape[1]).unsqueeze(0) >= seq_lens.unsqueeze(1)
    ids = ids.to(device)
    mask = mask.to(device)
    pad = pad.to(device)
    seq_lens = seq_lens.to(device)
    context = text_encoder(ids, mask)
    context.masked_fill_(pad.unsqueeze(-1), 0)
    return context, seq_lens


def encode_text_for_inference(
    prompt, *, vace_cache, prompt_embed_cache, tokenizer, text_encoder, device
) -> Tuple[Tensor, Tensor]:
    """Deploy ``(context, seq_lens)``, reusing a prompt-keyed cached embed."""
    if vace_cache and vace_cache.get("populated") and vace_cache.get("prompt_key") == prompt:
        return vace_cache["context"], vace_cache["seq_lens"]
    if prompt_embed_cache is not None and prompt in prompt_embed_cache:
        return prompt_embed_cache[prompt]
    context, seq_lens = encode_text([prompt], tokenizer=tokenizer, text_encoder=text_encoder, device=device)
    if prompt_embed_cache is not None:
        prompt_embed_cache[prompt] = (context, seq_lens)
    return context, seq_lens


def preprocess_video(frames, *, encoder=None, dtype, device) -> Tensor:
    if encoder is not None:
        return encoder.preprocess_video(frames)
    return _preprocess_video_native(frames, dtype=dtype, device=device)


def preprocess_video_batch(clips, *, encoder=None, dtype, device) -> Tensor:
    """Batch of clips -> stacked ``[B, C, T, H, W]`` pixels (one uint8 upload on the native path)."""
    if encoder is not None:
        return torch.cat([encoder.preprocess_video(c) for c in clips], dim=0)
    return _preprocess_video_batch_native(clips, dtype=dtype, device=device)


def encode_video(video_tensor: Tensor, *, vae, encoder=None) -> Tensor:
    if encoder is not None:
        return encoder.batch_encode(video_tensor)
    return vae.batch_encode(video_tensor, device=video_tensor.device)


def encode_video_for_vace(
    pixels: Tensor,
    *,
    vae,
    encoder=None,
    device,
    tiled: bool,
    tile_size: tuple,
    tile_stride: tuple,
) -> Tensor:
    """Tiled-aware VAE encode for the VACE pixel→latent helper, returning
    ``(B, z_dim, T_lat, H_lat, W_lat)``. ``tiled=False`` (training) batches in
    one call; ``tiled=True`` (deploy) loops per-sample with bounded peak memory,
    mirroring the vendored unit so large-frame deploy does not OOM.
    """
    if not tiled:
        return encode_video(pixels, vae=vae, encoder=encoder).to(dtype=pixels.dtype, device=pixels.device)
    if encoder is not None:
        # No generic tiled-encode contract; fall back to batch_encode.
        # Unreachable today (VACE + external_encoder is fail-fast).
        return encoder.batch_encode(pixels).to(dtype=pixels.dtype, device=pixels.device)
    # Native Wan VAE: per-sample tiled encode (deploy is B=1).
    outs = []
    for i in range(pixels.shape[0]):
        lat = vae.encode(
            [pixels[i]],
            device=device,
            tiled=True,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        outs.append(lat)
    return torch.cat(outs, dim=0).to(dtype=pixels.dtype, device=pixels.device)


def decode_latents(latents: Tensor, *, vae, encoder=None, device, tiled: bool = True) -> Tensor:
    if encoder is not None:
        return encoder.decode(latents.to(device), tiled=tiled)
    return vae.decode(latents.to(device), device=device, tiled=tiled)


def latents_to_frames(video_tensor: Tensor, *, encoder=None) -> list:
    if encoder is not None:
        return encoder.to_frames(video_tensor)
    return vae_output_to_video(video_tensor)
