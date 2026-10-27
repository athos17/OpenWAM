"""Deformation patch flow with variable, named regions and optional palms."""

import math

import torch
from einops import rearrange
from torch import nn

from openwam.tactile.regions import CODEC

from .modules import TactileFlowBlock


class DeformPatchCodec(nn.Module):
    """Single-channel [0,1] intensity, never pressure or metric depth."""

    def __init__(self, map_size=32, patch_size=16):
        super().__init__()
        if map_size < 1 or patch_size < 1 or map_size % patch_size:
            raise ValueError("Map size must be a positive multiple of patch size")
        self.map_size, self.patch_size = map_size, patch_size
        self.patches_per_site = (map_size // patch_size) ** 2
        self.patch_dim = patch_size**2

    def patchify(self, maps):
        return rearrange(maps, "b t n c (h p) (w q) -> b (t n h w) (c p q)", p=self.patch_size, q=self.patch_size)

    def unpatchify(self, tokens, num_regions):
        return rearrange(
            tokens,
            "b (t n h w) (c p q) -> b t n c (h p) (w q)",
            n=num_regions,
            c=1,
            h=self.map_size // self.patch_size,
            w=self.map_size // self.patch_size,
            p=self.patch_size,
            q=self.patch_size,
        )

    def pixel_mask(self, valid, spatial):
        return self.patchify(valid[..., None, None, None] & spatial[:, None])

    def encode(self, maps, valid, spatial):
        if maps.ndim != 6 or maps.shape[3:] != (1, self.map_size, self.map_size):
            raise ValueError("Deform maps must be [B,T,N,1,size,size]")
        if valid.dtype != torch.bool or valid.shape != maps.shape[:3]:
            raise ValueError("Observation validity must be bool [B,T,N]")
        if spatial.dtype != torch.bool or spatial.shape != (maps.shape[0], maps.shape[2], *maps.shape[3:]):
            raise ValueError("Spatial validity must be bool [B,N,1,size,size]")
        if maps.dtype == torch.uint8:
            maps = maps.float() / 255
        elif not maps.is_floating_point():
            raise ValueError("Deform maps must be uint8 or float [0,1]")
        finite = (torch.isfinite(maps) | ~spatial[:, None]).flatten(3).all(-1)
        valid = valid & finite & spatial.flatten(2).any(-1)[:, None]
        pixels = valid[..., None, None, None] & spatial[:, None]
        if ((maps < 0) | (maps > 1)).masked_select(pixels).any():
            raise ValueError("Floating deform intensities must be in [0,1]; no per-frame normalization")
        normalized = torch.where(pixels, 2 * maps - 1, 0)
        return self.patchify(normalized), valid

    def decode(self, tokens, num_regions):
        return ((self.unpatchify(tokens, num_regions) + 1) / 2).clamp(0, 1)


CODECS = {CODEC: DeformPatchCodec}


class RegionalTactileFlowStream(nn.Module):
    def __init__(self, spec, video_dim, action_dim):
        super().__init__()
        self.spec = spec
        width = spec["width"]
        self.codec = CODECS[spec["codec"]](spec["map_size"], spec["patch_size"])
        self.input_projection = nn.Sequential(
            nn.Linear(self.codec.patch_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.region_embedding = nn.Embedding(len(spec["region_names"]), width)
        self.sensor_embedding = nn.Embedding(len(spec["sensor_names"]), width)
        self.spatial_projection = nn.Linear(self.codec.patch_dim, width, bias=False)
        self.position_projection = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width, spec["heads"], 4 * width, dropout=0, activation="gelu", batch_first=True, norm_first=True
        )
        self.history_encoder = nn.TransformerEncoder(layer, spec["history_layers"], enable_nested_tensor=False)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(
            [TactileFlowBlock(width, spec["heads"], video_dim, action_dim) for _ in spec["bridge_layers"]]
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.codec.patch_dim))

    def embed(self, tokens, ids, sensors, spatial, times, surface_ids=None):
        b, n = ids.shape
        p = self.codec.patches_per_site
        side = self.spec["map_size"] // self.spec["patch_size"]
        centers = (torch.arange(side, device=tokens.device, dtype=tokens.dtype) + 0.5) / side * 2 - 1
        y, x = torch.meshgrid(centers, centers, indexing="ij")
        position = torch.stack((x.flatten(), y.flatten()), dim=-1)
        position = position[None].expand(len(times), -1, -1)
        position = torch.cat((position, times[:, None, None].expand(-1, p, 1)), dim=-1)
        positional = self.position_projection(position)[None, :, None]
        identity = (self.region_embedding(ids.clamp_min(0)) + self.sensor_embedding(sensors))[:, None, :, None]
        if surface_ids is not None:
            # Stable per-region instance IDs; no expanding learned identity table.
            # ID 0 is exactly zero, preserving the original twelve-region model.
            width = self.spec["width"]
            frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=tokens.device).float() / (width // 2))
            phase = surface_ids.float()[..., None] * frequencies
            instance = torch.cat((phase.sin(), phase.cos() - 1), dim=-1).to(tokens.dtype)
            identity = identity + instance[:, None, :, None]
        spatial_tokens = self.codec.patchify(spatial[:, None]).to(tokens.dtype)
        spatial_features = self.spatial_projection(spatial_tokens).reshape(b, 1, n, p, -1)
        return self.input_projection(tokens) + (positional + identity + spatial_features).reshape(
            b, -1, self.spec["width"]
        )

    def encode_history(self, maps, valid, ids, sensors, spatial, surface_ids=None):
        tokens, observed = self.codec.encode(maps, valid, spatial)
        pixels = self.codec.pixel_mask(observed, spatial)
        mask = pixels.any(-1)
        times = torch.arange(1 - self.spec["history"], 1, device=maps.device, dtype=tokens.dtype) / self.spec["fps"]
        tokens = self.embed(tokens, ids, sensors, spatial, times, surface_ids).masked_fill(~mask[..., None], 0)
        tokens = torch.cat((tokens.new_zeros(tokens.shape[0], 1, tokens.shape[-1]), tokens), dim=1)
        keys = torch.cat((torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device), mask), dim=1)
        frames = torch.cat(
            (
                torch.tensor([-1], device=mask.device),
                torch.arange(len(times), device=mask.device).repeat_interleave(
                    ids.shape[1] * self.codec.patches_per_site
                ),
            )
        )
        result = self.history_encoder(tokens, mask=frames[None] > frames[:, None], src_key_padding_mask=~keys)[:, 1:]
        return result.masked_fill(~mask[..., None], 0), mask, observed.any(1)

    def prepare_future(self, noisy, timestep, ids, sensors, spatial, surface_ids=None):
        width = self.spec["width"]
        frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=noisy.device).float() / (width // 2))
        phase = timestep.float().reshape(-1, 1) * frequencies[None]
        time = self.time_mlp(torch.cat((phase.cos(), phase.sin()), dim=-1).to(noisy.dtype))
        times = torch.arange(1, self.spec["future_frames"] + 1, device=noisy.device, dtype=noisy.dtype)
        times = times * self.spec["future_stride"] / self.spec["fps"]
        return self.embed(noisy, ids, sensors, spatial, times, surface_ids), time


def regional_flow_loss(codec, prediction, target, clean, pixel_valid, num_regions, foreground_weight=1.0):
    """Pixel mean -> region mean -> labeled sample mean, including distributed ranks."""
    delta = torch.where(pixel_valid, prediction.float() - target.float(), 0)
    intensity = ((clean.float() + 1) / 2).clamp(0, 1)
    weight = pixel_valid * (1 + (foreground_weight - 1) * intensity)
    error = codec.unpatchify(delta.square() * weight, num_regions).sum((1, 3, 4, 5))
    counts = codec.unpatchify(weight, num_regions).sum((1, 3, 4, 5))
    region_loss = error / counts.clamp_min(1)
    region_valid = counts > 0
    samples = (region_loss * region_valid).sum(1) / region_valid.sum(1).clamp_min(1)
    count = region_valid.any(1).sum().to(samples.dtype)
    numerator = samples.sum()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # DDP averages gradients; normalize by the global labeled-sample count.
        torch.distributed.all_reduce(count)
        return numerator * torch.distributed.get_world_size() / count.clamp_min(1)
    return numerator / count.clamp_min(1)
