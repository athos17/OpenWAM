"""Deformation patch flow with variable, named regions and optional palms."""

import math

import torch
from einops import rearrange
from torch import nn

from openwam.tactile.regions import CODEC, INTENSITY_CODEC, RGB_CODEC

from .modules import TactileFlowBlock


class DeformPatchCodec(nn.Module):
    """Single-channel [0,1] intensity, never pressure or metric depth."""

    channels = 1

    def __init__(self, map_size=32, patch_size=16):
        super().__init__()
        if map_size < 1 or patch_size < 1 or map_size % patch_size:
            raise ValueError("Map size must be a positive multiple of patch size")
        self.map_size, self.patch_size = map_size, patch_size
        self.patches_per_site = (map_size // patch_size) ** 2
        self.patch_dim = self.channels * patch_size**2

    def patchify(self, maps):
        return rearrange(maps, "b t n c (h p) (w q) -> b (t n h w) (c p q)", p=self.patch_size, q=self.patch_size)

    def unpatchify(self, tokens, num_regions):
        return rearrange(
            tokens,
            "b (t n h w) (c p q) -> b t n c (h p) (w q)",
            n=num_regions,
            c=self.channels,
            h=self.map_size // self.patch_size,
            w=self.map_size // self.patch_size,
            p=self.patch_size,
            q=self.patch_size,
        )

    def pixel_mask(self, valid, spatial):
        pixels = valid[..., None, None, None] & spatial[:, None]
        return self.patchify(pixels.expand(-1, -1, -1, self.channels, -1, -1))

    def encode(self, maps, valid, spatial):
        if maps.ndim != 6 or maps.shape[3:] != (self.channels, self.map_size, self.map_size):
            raise ValueError(f"Tactile maps must be [B,T,N,{self.channels},size,size]")
        if valid.dtype != torch.bool or valid.shape != maps.shape[:3]:
            raise ValueError("Observation validity must be bool [B,T,N]")
        if spatial.dtype != torch.bool or spatial.shape != (maps.shape[0], maps.shape[2], 1, self.map_size, self.map_size):
            raise ValueError("Spatial validity must be bool [B,N,1,size,size]")
        if maps.dtype == torch.uint8:
            maps = maps.float() / 255
        elif not maps.is_floating_point():
            raise ValueError("Tactile maps must be uint8 or float [0,1]")
        finite = (torch.isfinite(maps) | ~spatial[:, None]).flatten(3).all(-1)
        valid = valid & finite & spatial.flatten(2).any(-1)[:, None]
        pixels = valid[..., None, None, None] & spatial[:, None]
        if ((maps < 0) | (maps > 1)).masked_select(pixels).any():
            raise ValueError("Floating tactile values must be in [0,1]; no per-frame normalization")
        normalized = torch.where(pixels, 2 * maps - 1, 0)
        return self.patchify(normalized), valid

    def decode(self, tokens, num_regions):
        return ((self.unpatchify(tokens, num_regions) + 1) / 2).clamp(0, 1)


class IntensityPatchCodec(DeformPatchCodec):
    """Source-normalized contact/deformation strength, not calibrated force.

    Normalization belongs to the versioned offline source adapter. This codec
    only maps its fixed [0,1] scale to flow space [-1,1] and patchifies it.
    """


class RGBPatchCodec(DeformPatchCodec):
    """Original optical RGB values, not indentation or pressure estimates."""

    channels = 3


CODECS = {CODEC: DeformPatchCodec, INTENSITY_CODEC: IntensityPatchCodec, RGB_CODEC: RGBPatchCodec}


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
        if spec.get("rgb_enabled", False):
            self.rgb_codec = RGBPatchCodec(spec["rgb_map_size"], spec["rgb_patch_size"])
            if self.rgb_codec.patches_per_site != self.codec.patches_per_site:
                raise ValueError("RGB and intensity must have equal patches per region")
            self.rgb_input_projection = nn.Sequential(
                nn.Linear(self.rgb_codec.patch_dim, width), nn.SiLU(), nn.Linear(width, width)
            )
            self.rgb_spatial_projection = nn.Linear(self.rgb_codec.patch_size**2, width, bias=False)
            self.rgb_output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.rgb_codec.patch_dim))
        if spec.get("observation_encoder"):
            from .heterogeneous import HeterogeneousObservationEncoder
            if self.codec.patches_per_site != 4:
                raise ValueError("FTP spatial adapter currently requires exactly four tokens per region")
            self.observation_encoder = HeterogeneousObservationEncoder(spec["observation_encoder"], width, spec["sensor_names"])

    def get_codec(self, name=None):
        if name in (None, self.spec["codec"]):
            return self.codec
        if name == RGB_CODEC and self.spec.get("rgb_enabled", False):
            return self.rgb_codec
        raise ValueError(f"Tactile codec is not enabled in this checkpoint: {name}")

    def predict(self, future, codec_name=None):
        self.get_codec(codec_name)  # fail closed on invalid representation
        return (self.rgb_output if codec_name == RGB_CODEC else self.output)(future)

    def identity_features(self, tokens, ids, sensors, spatial, times, codec_name=None):
        b, n = ids.shape
        codec = self.get_codec(codec_name)
        p = codec.patches_per_site
        side = codec.map_size // codec.patch_size
        centers = (torch.arange(side, device=tokens.device, dtype=tokens.dtype) + 0.5) / side * 2 - 1
        y, x = torch.meshgrid(centers, centers, indexing="ij")
        position = torch.stack((x.flatten(), y.flatten()), dim=-1)
        position = position[None].expand(len(times), -1, -1)
        position = torch.cat((position, times[:, None, None].expand(-1, p, 1)), dim=-1)
        positional = self.position_projection(position)[None, :, None]
        identity = (self.region_embedding(ids.clamp_min(0)) + self.sensor_embedding(sensors))[:, None, :, None]
        spatial_tokens = codec.patchify(spatial[:, None]).to(tokens.dtype)
        spatial_projection = self.rgb_spatial_projection if codec_name == RGB_CODEC else self.spatial_projection
        spatial_features = spatial_projection(spatial_tokens).reshape(b, 1, n, p, -1)
        return (positional + identity + spatial_features).reshape(b, -1, self.spec["width"])

    def embed(self, tokens, ids, sensors, spatial, times, codec_name=None):
        input_projection = self.rgb_input_projection if codec_name == RGB_CODEC else self.input_projection
        return input_projection(tokens) + self.identity_features(tokens, ids, sensors, spatial, times, codec_name)

    def encode_history(self, maps, valid, ids, sensors, spatial, codec_name=None, *,
                       observation_maps=None, observation_spatial=None, encoder_ids=None):
        codec = self.get_codec(codec_name)
        if hasattr(self, "observation_encoder"):
            return self._encode_heterogeneous_history(valid, ids, sensors, spatial, codec_name,
                                                       observation_maps, observation_spatial, encoder_ids)
        tokens, observed = codec.encode(maps, valid, spatial)
        if ids.shape[1] == 0:
            return tokens.new_zeros(maps.shape[0], 0, self.spec["width"]), valid.new_zeros(maps.shape[0], 0), observed.any(1)
        pixels = codec.pixel_mask(observed, spatial)
        mask = pixels.any(-1)
        times = torch.arange(1 - self.spec["history"], 1, device=maps.device, dtype=tokens.dtype) / self.spec["fps"]
        tokens = self.embed(tokens, ids, sensors, spatial, times, codec_name).masked_fill(~mask[..., None], 0)
        tokens = torch.cat((tokens.new_zeros(tokens.shape[0], 1, tokens.shape[-1]), tokens), dim=1)
        keys = torch.cat((torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device), mask), dim=1)
        frames = torch.cat(
            (
                torch.tensor([-1], device=mask.device),
                torch.arange(len(times), device=mask.device).repeat_interleave(
                    ids.shape[1] * codec.patches_per_site
                ),
            )
        )
        result = self.history_encoder(tokens, mask=frames[None] > frames[:, None], src_key_padding_mask=~keys)[:, 1:]
        return result.masked_fill(~mask[..., None], 0), mask, observed.any(1)

    def _encode_heterogeneous_history(self, valid, ids, sensors, spatial, codec_name,
                                      observation_maps, observation_spatial, encoder_ids):
        if ids.shape[1] == 0:
            return spatial.new_zeros(ids.shape[0], 0, self.spec["width"], dtype=self.region_embedding.weight.dtype), valid.new_zeros(ids.shape[0], 0), valid.any(1)
        if observation_maps is None or observation_spatial is None or encoder_ids is None:
            raise ValueError("Heterogeneous history requires original-resolution observation maps, ROI and encoder IDs")
        tokens, mask, available = self.observation_encoder(observation_maps, valid, observation_spatial, encoder_ids, sensors)
        times = torch.arange(1 - self.spec["history"], 1, device=tokens.device, dtype=tokens.dtype) / self.spec["fps"]
        tokens = (tokens + self.identity_features(tokens, ids, sensors, spatial, times, codec_name)).masked_fill(~mask[..., None], 0)
        tokens = torch.cat((tokens.new_zeros(tokens.shape[0], 1, tokens.shape[-1]), tokens), dim=1)
        keys = torch.cat((torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device), mask), 1)
        frames = torch.cat((torch.tensor([-1], device=mask.device),
                            torch.arange(len(times), device=mask.device).repeat_interleave(ids.shape[1] * 4)))
        result = self.history_encoder(tokens, mask=frames[None] > frames[:, None], src_key_padding_mask=~keys)[:, 1:]
        return result.masked_fill(~mask[..., None], 0), mask, available

    def prepare_future(self, noisy, timestep, ids, sensors, spatial, codec_name=None):
        width = self.spec["width"]
        frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=noisy.device).float() / (width // 2))
        phase = timestep.float().reshape(-1, 1) * frequencies[None]
        time = self.time_mlp(torch.cat((phase.cos(), phase.sin()), dim=-1).to(noisy.dtype))
        times = torch.arange(1, self.spec["future_frames"] + 1, device=noisy.device, dtype=noisy.dtype)
        times = times * self.spec["future_stride"] / self.spec["fps"]
        return self.embed(noisy, ids, sensors, spatial, times, codec_name), time


def regional_flow_loss(codec, prediction, target, clean, pixel_valid, num_regions, foreground_weight=1.0):
    """Pixel mean -> region mean -> labeled sample mean, including distributed ranks."""
    if num_regions == 0:
        # All ranks still enter the same collective, including empty layouts.
        return globally_normalized_sum(prediction.sum() * 0, prediction.new_zeros(()))
    delta = torch.where(pixel_valid, prediction.float() - target.float(), 0)
    if isinstance(codec, RGBPatchCodec):
        # RGB brightness is not contact intensity: never weight bright pixels
        # as if they represented larger physical force/deformation.
        weight = pixel_valid.float()
    else:
        intensity = ((clean.float() + 1) / 2).clamp(0, 1)
        weight = pixel_valid * (1 + (foreground_weight - 1) * intensity)
    error = codec.unpatchify(delta.square() * weight, num_regions).sum((1, 3, 4, 5))
    counts = codec.unpatchify(weight, num_regions).sum((1, 3, 4, 5))
    region_loss = error / counts.clamp_min(1)
    region_valid = counts > 0
    samples = (region_loss * region_valid).sum(1) / region_valid.sum(1).clamp_min(1)
    count = region_valid.any(1).sum().to(samples.dtype)
    numerator = samples.sum()
    return globally_normalized_sum(numerator, count)


def globally_normalized_sum(numerator, count):
    """DDP gradient mean of the globally labeled sample population."""
    count = count.detach().clone().to(device=numerator.device, dtype=torch.float32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # DDP averages gradients; normalize by the global labeled-sample count.
        torch.distributed.all_reduce(count)
        return numerator * torch.distributed.get_world_size() / count.clamp_min(1)
    return numerator / count.clamp_min(1)
