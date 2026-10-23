"""Compact tactile flow stream and additive bridges for TouchWAM V2.

The codec is a fixed, invertible patch layout, not a learned VAE. This keeps
the flow targets stationary on small datasets and preserves sparse contacts.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class TactilePatchCodec(nn.Module):
    def __init__(self, map_size=32, patch_size=16, distance_scale_m=0.015):
        super().__init__()
        if map_size < 8 or patch_size < 1 or map_size % patch_size or distance_scale_m <= 0:
            raise ValueError("Invalid tactile map/patch size or distance scale")
        self.map_size, self.patch_size = map_size, patch_size
        self.distance_scale_m = distance_scale_m
        self.patches_per_site = (map_size // patch_size) ** 2
        self.tokens_per_frame = 10 * self.patches_per_site
        self.patch_dim = 2 * patch_size**2

    def patchify(self, maps):
        return rearrange(
            maps,
            "b t n c (h p) (w q) -> b (t n h w) (c p q)",
            p=self.patch_size,
            q=self.patch_size,
        )

    def unpatchify(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] % self.tokens_per_frame or tokens.shape[2] != self.patch_dim:
            raise ValueError("Invalid tactile patch tensor")
        grid = self.map_size // self.patch_size
        return rearrange(
            tokens,
            "b (t n h w) (c p q) -> b t n c (h p) (w q)",
            n=10,
            h=grid,
            w=grid,
            c=2,
            p=self.patch_size,
            q=self.patch_size,
        )

    def encode(self, maps, valid):
        if maps.ndim != 6 or maps.shape[2:] != (10, 2, self.map_size, self.map_size):
            raise ValueError("Tactile maps must be [B,T,10,2,map_size,map_size]")
        if valid.shape != maps.shape[:3] or valid.dtype != torch.bool:
            raise ValueError("Tactile validity must be bool [B,T,10]")
        valid = valid & torch.isfinite(maps).flatten(3).all(-1)
        maps = torch.where(valid[..., None, None, None], maps, 0)
        contact = maps[:, :, :, 1].clamp(0, 1)
        distance = (maps[:, :, :, 0] / self.distance_scale_m).clamp(-1, 1) * (contact > 0)
        normalized = torch.stack((distance, 2 * contact - 1), dim=3)
        normalized = normalized.masked_fill(~valid[..., None, None, None], 0)
        token_valid = valid.flatten(1).repeat_interleave(self.patches_per_site, dim=1)
        return self.patchify(normalized), token_valid

    def decode(self, tokens):
        maps = self.unpatchify(tokens)
        contact = ((maps[:, :, :, 1] + 1) / 2).clamp(0, 1)
        distance = maps[:, :, :, 0].clamp(-1, 1) * self.distance_scale_m
        # Probabilities stay continuous; geometry is only defined on contact.
        distance = distance.masked_fill(contact < 0.5, 0)
        return torch.stack((distance, contact), dim=3)


class CrossAttention(nn.Module):
    """Small attention space, independent of Alpha's shared VA attention."""

    def __init__(self, query_dim, context_dim, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(query_dim, width)
        self.k = nn.Linear(context_dim, width)
        self.v = nn.Linear(context_dim, width)
        self.out = nn.Linear(width, query_dim)

    def forward(self, query, context, valid):
        # A permanently masked-in zero slot avoids NaNs for absent sensors.
        context = context.masked_fill(~valid[..., None], 0)
        context = torch.cat((context.new_zeros(context.shape[0], 1, context.shape[-1]), context), dim=1)
        null = ~valid.any(dim=1, keepdim=True)
        safe_valid = torch.cat((null, valid), dim=1)
        q, k, v = [
            rearrange(x, "b s (h d) -> b h s d", h=self.heads)
            for x in (self.q(query), self.k(context), self.v(context))
        ]
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_valid[:, None, None])
        result = self.out(rearrange(result, "b h s d -> b s (h d)"))
        return result.masked_fill(~valid.any(dim=1)[:, None, None], 0)


class TactileFlowBlock(nn.Module):
    """Future touch reads clean history and VA; gated touch writes back to VA.

    History is read-only. The clean visual prefix never receives a touch
    write, so noisy future touch cannot contaminate conditioning tokens.
    """

    def __init__(self, width, heads, video_dim, action_dim):
        super().__init__()
        self.norm_self = nn.LayerNorm(width, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 4 * width))
        self.self_attn = CrossAttention(width, width, width, heads)
        self.video_read = nn.Sequential(nn.LayerNorm(video_dim), nn.Linear(video_dim, width))
        self.action_read = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, width))
        self.read_norm = nn.LayerNorm(width)
        self.read_va = CrossAttention(width, width, width, heads)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.video_norm = nn.LayerNorm(video_dim)
        self.action_norm = nn.LayerNorm(action_dim)
        self.write_video = CrossAttention(video_dim, width, width, heads)
        self.write_action = CrossAttention(action_dim, width, width, heads)
        self.video_gate = nn.Parameter(torch.zeros(()))
        self.action_gate = nn.Parameter(torch.zeros(()))

    def forward(self, future, history, future_valid, history_valid, time, video, action, clean_video_tokens):
        shift_a, scale_a, shift_f, scale_f = self.modulation(time)[:, None].chunk(4, dim=-1)
        query = self.norm_self(future) * (1 + scale_a) + shift_a
        valid = torch.cat((history_valid, future_valid), dim=1)
        future = future + self.self_attn(query, torch.cat((history, query), dim=1), valid)
        va = torch.cat((self.video_read(video), self.action_read(action)), dim=1)
        va_valid = torch.ones(va.shape[:2], dtype=torch.bool, device=va.device)
        future = future + self.read_va(self.read_norm(future), va, va_valid)
        future = future + self.ffn(self.norm_ffn(future) * (1 + scale_f) + shift_f)
        future = future.masked_fill(~future_valid[..., None], 0)
        memory = torch.cat((history, future), dim=1)
        dv = self.write_video(self.video_norm(video), memory, valid)
        # No in-place mutation of activations: checkpoint recomputation is safe.
        writable = torch.arange(video.shape[1], device=video.device) >= clean_video_tokens
        dv = dv * writable[None, :, None]
        video = video + self.video_gate.tanh() * dv
        action = action + self.action_gate.tanh() * self.write_action(self.action_norm(action), memory, valid)
        return future, video, action


class TactileFlowStream(nn.Module):
    def __init__(self, spec, video_dim, action_dim):
        super().__init__()
        self.spec = spec
        width, heads = spec["width"], spec["heads"]
        self.codec = TactilePatchCodec(spec["map_size"], spec["patch_size"], spec["distance_scale_m"])
        self.input_projection = nn.Sequential(
            nn.Linear(self.codec.patch_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.site_embedding = nn.Embedding(10, width)
        self.patch_embedding = nn.Embedding(self.codec.patches_per_site, width)
        self.frame_embedding = nn.Embedding(spec["history"] + spec["future_frames"], width)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(layer, spec["history_layers"], enable_nested_tensor=False)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(
            [TactileFlowBlock(width, heads, video_dim, action_dim) for _ in spec["bridge_layers"]]
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.codec.patch_dim))

    def embed(self, tokens, start_frame, num_frames):
        device = tokens.device
        p = self.codec.patches_per_site
        position = (
            self.frame_embedding(torch.arange(start_frame, start_frame + num_frames, device=device))[:, None, None]
            + self.site_embedding(torch.arange(10, device=device))[None, :, None]
            + self.patch_embedding(torch.arange(p, device=device))[None, None]
        ).reshape(1, num_frames * 10 * p, -1)
        return self.input_projection(tokens) + position

    def encode_history(self, maps, valid):
        tokens, valid = self.codec.encode(maps, valid)
        history = self.spec["history"]
        if maps.shape[1] != history:
            raise ValueError(f"Expected {history} tactile history frames")
        tokens = self.embed(tokens, 0, history)
        tokens = torch.cat((tokens.new_zeros(tokens.shape[0], 1, tokens.shape[-1]), tokens), dim=1)
        keys = torch.cat((torch.ones(valid.shape[0], 1, dtype=torch.bool, device=valid.device), valid), dim=1)
        times = torch.cat(
            (
                torch.tensor([-1], device=tokens.device),
                torch.arange(history, device=tokens.device).repeat_interleave(self.codec.tokens_per_frame),
            )
        )
        causal = times[None, :] > times[:, None]
        tokens = self.history_encoder(tokens, mask=causal, src_key_padding_mask=~keys)[:, 1:]
        return tokens.masked_fill(~valid[..., None], 0), valid

    def prepare_future(self, noisy, timestep):
        expected = self.spec["future_frames"] * self.codec.tokens_per_frame
        if noisy.ndim != 3 or noisy.shape[1:] != (expected, self.codec.patch_dim):
            raise ValueError("noisy_tactile does not match the configured future horizon/patch layout")
        width = self.spec["width"]
        frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=noisy.device).float() / (width // 2))
        phase = timestep.float().reshape(-1, 1) * frequencies[None]
        time = self.time_mlp(torch.cat((phase.cos(), phase.sin()), dim=-1).to(noisy.dtype))
        return self.embed(noisy, self.spec["history"], self.spec["future_frames"]), time


def tactile_flow_loss(codec, prediction, target, clean, valid, *, positive_weight=5.0, background_weight=0.1):
    """Masked velocity MSE; contact pixels are not drowned out by empty maps."""
    error = codec.unpatchify(prediction.float() - target.float()).square()
    contact = codec.unpatchify(clean.float())[:, :, :, 1] > 0
    sensor_valid = valid.reshape(valid.shape[0], -1, 10, codec.patches_per_site).all(-1)
    pixels = sensor_valid[..., None, None].expand_as(contact)
    contact_weight = pixels * torch.where(contact, positive_weight, 1.0)
    distance_weight = pixels * torch.where(contact, 1.0, background_weight)
    loss_contact = (error[:, :, :, 1] * contact_weight).sum() / contact_weight.sum().clamp_min(1)
    loss_distance = (error[:, :, :, 0] * distance_weight).sum() / distance_weight.sum().clamp_min(1)
    return loss_contact + loss_distance, loss_contact, loss_distance
