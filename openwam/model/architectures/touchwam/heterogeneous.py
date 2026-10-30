"""FTP-1/T3 compatible *observation* encoders with spatial TouchWAM adapters.

Architecture adapted from FTP-1 (Apache-2.0) and T3 (MIT), see
THIRD_PARTY_NOTICES.md. State-dict names match the published timm ViT weights.
No original FTP region embedding, CLS projection or action expert is reused.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, width, heads, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=eps)
        self.attn = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width, eps=eps)
        self.mlp = Mlp(width)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ImageFrontend(nn.Module):
    def __init__(self, size, patch, width, depth, heads):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, width, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, (size // patch) ** 2 + 1, width))
        self.blocks = nn.ModuleList([Block(width, heads, 1e-6) for _ in range(depth)])
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), 1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return x


class ImageTrunk(nn.Module):
    def __init__(self, width, depth, heads):
        super().__init__()
        # FTP's shared TransformerTrunk uses the default nn.LayerNorm epsilon.
        self.blocks = nn.ModuleList([Block(width, heads, 1e-5) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class MatrixEncoder(nn.Module):
    """FTP Fourier+CNN architecture; preserve spatial cells instead of flattening."""
    def __init__(self, shape, width, output_width, fourier_dim=8):
        super().__init__()
        self.shape, self.fourier_dim = tuple(shape), fourier_dim
        layers, channels = [], 1 + fourier_dim
        h, w = shape
        for _ in range(3):
            layers += [nn.Conv2d(channels, width, 3, padding=1), nn.GELU()]
            if (h > 4 or w > 4) and min(h, w) >= 2:
                layers.append(nn.MaxPool2d(2, 2))
                h, w = h // 2, w // 2
            channels = width
        self.cnn = nn.Sequential(*layers)
        self.projection = nn.Linear(width, output_width)

    def forward(self, x, spatial):
        # Fourier evaluation in float32, then convolution in model dtype.
        periods = 1e-3 * 1000 ** torch.linspace(0, 1, self.fourier_dim // 2, device=x.device)
        phase = x.float() * (2 * torch.pi / periods)[None, :, None, None]
        expanded = torch.cat((x.float(), phase.sin(), phase.cos()), 1)
        expanded = expanded.masked_fill(~spatial, 0).to(x.dtype)
        features = self.cnn(expanded)
        return self.projection(F.adaptive_avg_pool2d(features, 2).flatten(2).transpose(1, 2))


def _plain_config(value):
    """Detach nested Hydra containers while their interpolation context exists."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def normalize_encoder_spec(value, source_names):
    """Validate the versioned model contract without opening checkpoint files."""
    # A shallow dict(DictConfig) leaves sensors, weights and normalizations as
    # DictConfig/ListConfig objects, which cannot be serialized in safetensors
    # metadata. Also avoid mutating the caller's source mappings below.
    s = _plain_config(value)
    defaults = dict(kind="ftp1_heterogeneous_v1", initialization="pretrained", image_size=224,
                    image_patch_size=16, image_width=768, image_front_layers=3,
                    image_shared_layers=9, image_heads=12, matrix_width=1024,
                    matrix_fourier_dim=8, freeze_pretrained=True, weights_root="", weights={},
                    sensors=[], source_to_encoder={}, normalizations={})
    if set(s) - set(defaults):
        raise ValueError(f"Unknown observation encoder settings: {sorted(set(s) - set(defaults))}")
    defaults.update(s)
    s = defaults
    if s["kind"] != "ftp1_heterogeneous_v1" or s["initialization"] not in ("pretrained", "random"):
        raise ValueError("Unknown observation encoder kind/initialization")
    for key in ("image_size", "image_patch_size", "image_width", "image_front_layers", "image_shared_layers",
                "image_heads", "matrix_width", "matrix_fourier_dim"):
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (s["image_size"] % s["image_patch_size"] or s["image_size"] // s["image_patch_size"] < 2
            or s["image_width"] % s["image_heads"] or s["matrix_fourier_dim"] % 2):
        raise ValueError("Invalid patch, attention or Fourier dimensions")
    if type(s["freeze_pretrained"]) is not bool:
        raise ValueError("freeze_pretrained must be boolean")
    names = [x["name"] for x in s["sensors"]]
    if not names or len(names) != len(set(names)) or any(not n or "." in n for n in names):
        raise ValueError("Encoder names must be nonempty, unique module keys")
    for sensor in s["sensors"]:
        if sensor.get("kind") not in ("image", "matrix"):
            raise ValueError("Encoder kind must be image or matrix")
        if sensor["kind"] == "image" and sensor.get("input_representation") not in ("rgb", "deform"):
            raise ValueError("Image encoder requires rgb/deform input_representation")
        if sensor["kind"] == "matrix":
            shape = sensor.get("input_shape", [])
            if len(shape) != 2 or any(type(n) is not int or n < 2 for n in shape):
                raise ValueError("Matrix encoder requires explicit native input_shape [H,W], each >=2")
    for source, encoders in s["source_to_encoder"].items():
        encoders = [encoders] if isinstance(encoders, str) else list(encoders)
        if source not in source_names or not encoders or set(encoders) - set(names):
            raise ValueError(f"Unknown source/encoder mapping: {source}")
        s["source_to_encoder"][source] = encoders
        for name in encoders:
            sensor = s["sensors"][names.index(name)]
            if sensor["kind"] != "image":
                continue
            norm = s["normalizations"].get(source, {}).get(name)
            if norm is None:
                raise ValueError(f"Explicit source-specific normalization required: {source}/{name}")
            if (len(norm.get("mean", [])) != 3 or len(norm.get("std", [])) != 3
                    or not torch.isfinite(torch.tensor([*norm["mean"], *norm["std"], norm.get("pre_scale", 2), norm.get("pre_bias", -1)])).all()
                    or any(x <= 0 for x in norm["std"])):
                raise ValueError(f"Invalid image normalization: {source}/{name}")
    if s["initialization"] == "pretrained":
        required = {x["name"] for x in s["sensors"] if x["kind"] == "image"}
        if required:
            required.add("shared")
        for key in required:
            item = s["weights"].get(key, {})
            if not item.get("file") or len(item.get("sha256", "")) != 64:
                raise ValueError(f"Pinned pretrained file and SHA256 required for {key}")
    return s


class HeterogeneousObservationEncoder(nn.Module):
    # RegionalTouchWAMArchitecture.configure_training_fastpath(validate_inputs=False) clears it.
    _validate_inputs = True

    def __init__(self, spec, output_width, source_names):
        super().__init__()
        self.spec, self.source_names = spec, list(source_names)
        self.encoder_names = [x["name"] for x in spec["sensors"]]
        self.image_frontends, self.matrix_encoders = nn.ModuleDict(), nn.ModuleDict()
        for sensor in spec["sensors"]:
            name = sensor["name"]
            if sensor["kind"] == "image":
                self.image_frontends[name] = ImageFrontend(spec["image_size"], spec["image_patch_size"],
                    spec["image_width"], spec["image_front_layers"], spec["image_heads"])
            else:
                self.matrix_encoders[name] = MatrixEncoder(sensor["input_shape"], spec["matrix_width"],
                                                           output_width, spec["matrix_fourier_dim"])
        if self.image_frontends:
            self.image_trunk = ImageTrunk(spec["image_width"], spec["image_shared_layers"], spec["image_heads"])
            self.image_projection = nn.Linear(spec["image_width"], output_width)
        self.output_width = output_width
        # Persists in full checkpoints/resume; Alpha warm start must load FTP explicitly.
        self.register_buffer("weights_initialized", torch.tensor(spec["initialization"] == "random"))
        self.apply_freeze_policy()

    def apply_freeze_policy(self):
        if self.spec["freeze_pretrained"]:
            for module in [*self.image_frontends.values(), *([self.image_trunk] if self.image_frontends else [])]:
                module.requires_grad_(False)

    def observation_shape(self, encoder_ids):
        shapes = set()
        for idx in torch.as_tensor(encoder_ids).flatten().unique().tolist():
            if not 0 <= idx < len(self.encoder_names):
                raise ValueError("Unknown tactile encoder ID")
            entry = self.spec["sensors"][idx]
            shapes.add((self.spec["image_size"],) * 2 if entry["kind"] == "image" else tuple(entry["input_shape"]))
        if len(shapes) > 1:
            raise ValueError("A microbatch requires one observation spatial shape; bucket by encoder shape")
        return next(iter(shapes), (self.spec["image_size"],) * 2)

    def load_pretrained(self):
        """Strict key/shape/hash audit; deliberately exclude only FTP's CLS projection."""
        from safetensors import safe_open
        from safetensors.torch import load_file

        if self.spec["initialization"] == "random":
            return {"initialization": "explicit_random", "reused_parameters": 0}
        self.weights_initialized.fill_(False)
        targets = [(name, module, "vit_encoder.", set()) for name, module in self.image_frontends.items()]
        if self.image_frontends:
            targets += [("shared", self.image_trunk, "shared_chunk_encoder.", {"image_proj.weight", "image_proj.bias"})]
        audits = []
        for name, module, prefix, ignored in targets:
            entry = self.spec["weights"][name]
            path = Path(self.spec["weights_root"]) / entry["file"]
            digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise RuntimeError(f"FTP-1 checksum mismatch: {name}")
            expected = module.state_dict()
            with safe_open(path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                required = {prefix + key for key in expected}
                if keys != required | ignored:
                    raise RuntimeError(f"FTP-1 key mismatch: {name}: missing={required - keys}, extra={keys - required - ignored}")
                for key, value in expected.items():
                    if tuple(f.get_slice(prefix + key).get_shape()) != tuple(value.shape):
                        raise RuntimeError(f"FTP-1 shape mismatch: {name}/{key}")
                if ignored and (f.get_slice("image_proj.weight").get_shape() != [512, self.spec["image_width"]]
                                or f.get_slice("image_proj.bias").get_shape() != [512]):
                    raise RuntimeError("Unexpected FTP CLS projection shape")
            checkpoint = load_file(str(path), device="cpu")
            module.load_state_dict({key: checkpoint[prefix + key] for key in expected}, strict=True)
            audits.append(dict(name=name, file=str(path), sha256=digest, tensors=len(expected),
                               parameters=sum(x.numel() for x in expected.values()), ignored=sorted(ignored)))
        self.weights_initialized.fill_(True)
        self.apply_freeze_policy()
        return dict(initialization="ftp1_pretrained", files=audits,
                    reused_parameters=sum(x["parameters"] for x in audits),
                    source_revision="d6e5b73e473d3e70fb5f53132a2b1c35b5031156")

    def forward(self, maps, valid, spatial, encoder_ids, source_ids):
        b, t, n, c, h, w = maps.shape
        if encoder_ids.shape != (b, n) or encoder_ids.dtype != torch.long:
            raise ValueError("tactile_encoder_ids must be int64 [B,N]")
        if source_ids.shape != (b, n) or spatial.shape != (b, n, 1, h, w) or spatial.dtype != torch.bool:
            raise ValueError("Observation source IDs/spatial ROI differs from layout")
        if valid.shape != (b, t, n) or valid.dtype != torch.bool:
            raise ValueError("Observation validity must be bool [B,T,N]")
        if not n:
            return maps.new_zeros(b, 0, self.output_width), valid.new_zeros(b, 0), valid.any(1)
        supported = spatial.flatten(2).any(-1)
        if self._validate_inputs:
            if ((encoder_ids < -1) | (encoder_ids >= len(self.encoder_names))
                    | ((encoder_ids < 0) & (supported | valid.any(1)))).any():
                raise ValueError("Unknown tactile encoder ID")
            real_ids = encoder_ids[supported]
            if real_ids.numel() and tuple((h, w)) != self.observation_shape(real_ids):
                raise ValueError("Observation maps must preserve the configured image/matrix spatial resolution")
        pixel_valid = valid[..., None, None, None] & spatial[:, None]
        finite = (torch.isfinite(maps) | ~pixel_valid).flatten(3).all(-1)
        observed = valid & finite & spatial.flatten(2).any(-1)[:, None]
        pixel_valid &= observed[..., None, None, None]
        if self._validate_inputs and ((maps < 0) | (maps > 1)).masked_select(pixel_valid).any():
            raise ValueError("Observation values must be source-normalized [0,1]")
        maps = torch.where(pixel_valid, maps, 0)
        result = maps.new_zeros(b * t * n, 4, self.output_width)
        flat = maps.reshape(b * t * n, c, h, w)
        roi = spatial[:, None].expand(b, t, n, 1, h, w).reshape(-1, 1, h, w)
        encoders = encoder_ids[:, None].expand(b, t, n).reshape(-1)
        sources = source_ids[:, None].expand(b, t, n).reshape(-1)
        enabled = observed.reshape(-1)
        for idx, sensor in enumerate(self.spec["sensors"]):
            name = sensor["name"]
            for source_id in source_ids.unique().tolist():
                selection = (encoders == idx) & (sources == source_id) & enabled
                if not selection.any():
                    continue
                source = self.source_names[source_id]
                if name not in self.spec["source_to_encoder"].get(source, []):
                    raise ValueError(f"Unregistered source/encoder pair: {source}/{name}")
                x, mask = flat[selection], roi[selection]
                if sensor["kind"] == "image":
                    if self._validate_inputs and not self.weights_initialized.item():
                        raise RuntimeError("FTP-1 observation weights not initialized; load pinned weights or full checkpoint")
                    expected_channels = 3 if sensor["input_representation"] == "rgb" else 1
                    if c != expected_channels:
                        raise ValueError(f"Wrong observation channels for {name}: expected {expected_channels}")
                    x = x.expand(-1, 3, -1, -1) if c == 1 else x
                    norm = self.spec["normalizations"][source][name]
                    mean, std = x.new_tensor(norm["mean"])[None, :, None, None], x.new_tensor(norm["std"])[None, :, None, None]
                    x = ((x * norm.get("pre_scale", 2) + norm.get("pre_bias", -1) - mean) / std).masked_fill(~mask, 0)
                    features = self.image_trunk(self.image_frontends[name](x))[:, 1:]
                    side = self.spec["image_size"] // self.spec["image_patch_size"]
                    features = features.transpose(1, 2).reshape(-1, self.spec["image_width"], side, side)
                    patch_roi = F.adaptive_avg_pool2d(mask.float(), (side, side))
                    pooled = F.adaptive_avg_pool2d(features * patch_roi, 2) / F.adaptive_avg_pool2d(patch_roi, 2).clamp_min(1e-8)
                    output = self.image_projection(pooled.to(features.dtype).flatten(2).transpose(1, 2))
                else:
                    if c != 1:
                        raise ValueError("Pressure encoder requires one normalized pressure channel")
                    output = self.matrix_encoders[name](x, mask)
                result[selection] = output
        output_valid = F.adaptive_max_pool2d(roi.float(), 2).flatten(1).bool() & enabled[:, None]
        result = result.masked_fill(~output_valid[..., None], 0)
        return result.reshape(b, t * n * 4, -1), output_valid.reshape(b, -1), observed.any(1)
