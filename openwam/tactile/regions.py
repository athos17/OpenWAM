"""Named tactile surfaces. Availability is metadata, never inferred from intensity."""

from collections import deque

import numpy as np
import torch
from torch.nn import functional as F

from .registry import FINGERTIP_NAMES, REGION_NAMES, validate_registry

SCHEMA = "tactile_regions_v1"
CODEC = "sharpa_deform_gray_v1"
INTENSITY_CODEC = "tactile_intensity_v1"
RGB_CODEC = "tactile_rgb_v1"
SUPPORTED_CODECS = (CODEC, INTENSITY_CODEC, RGB_CODEC)


def observation_shape(size):
    """Normalize an explicit observation shape without inventing native resolution."""
    try:
        shape = (size, size) if type(size) is int else tuple(size)
    except TypeError as exc:
        raise ValueError("Observation map size must be a positive integer or [height, width]") from exc
    if len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError("Observation map size must be a positive integer or [height, width]")
    return shape


def observation_batch_shape(samples, specification):
    """Resolve the explicit frontend shape for a homogeneous observation batch."""
    sensors = list(specification.get("sensors", ()))
    names = [item["name"] for item in sensors]
    if not names or len(set(names)) != len(names):
        raise ValueError("Heterogeneous observation encoders must have unique names")
    shapes = set()
    for sample in samples:
        count = len(sample.get("tactile_region_names", ()))
        if not count:
            continue
        if "tactile_encoder_names" in sample:
            selected = list(sample["tactile_encoder_names"])
            if "tactile_encoder_ids" in sample or len(selected) != count or set(selected) - set(names):
                raise ValueError("Ambiguous or unregistered tactile encoder names")
            indices = [names.index(name) for name in selected]
        else:
            ids = torch.as_tensor(sample.get("tactile_encoder_ids", []))
            if ids.dtype != torch.long or ids.shape != (count,) or ((ids < 0) | (ids >= len(names))).any():
                raise ValueError("Heterogeneous observations require explicit registered encoder IDs/names")
            indices = ids.tolist()
        for i in indices:
            sensor = sensors[i]
            if sensor["kind"] == "image":
                shape = observation_shape(specification.get("image_size", 224))
            elif sensor["kind"] == "matrix":
                if "input_shape" not in sensor:
                    raise ValueError("Matrix encoders require explicit native input_shape")
                shape = observation_shape(sensor["input_shape"])
            else:
                raise ValueError("Observation encoder kind must be image or matrix")
            shapes.add(shape)
    if len(shapes) > 1:
        raise ValueError("A microbatch must use one observation spatial shape; bucket native matrix shapes separately")
    return next(iter(shapes), observation_shape(specification.get("image_size", 224)))


def resize_deform(image, size):
    """uint8 SDK grayscale -> float [0,1], area resize without per-frame scaling."""
    value = torch.as_tensor(np.asarray(image))
    if value.dtype != torch.uint8 or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("Expected a nonempty uint8 mono deformation image")
    if type(size) is not int or size < 1:
        raise ValueError("Invalid deformation image size")
    return F.interpolate(value[None, None].float() / 255, (size, size), mode="area")[0]


def collate_regions(
    samples,
    *,
    map_size,
    history,
    future_frames,
    region_names=REGION_NAMES,
    sensor_names=("sharpa_real", "bench2dex_sim"),
    codec=CODEC,
    observation_map_size=None,
    encoder_names=None,
):
    """Collate one regional tactile representation per microbatch.

    Empty layouts have N=0 when the entire batch is non-tactile. Mixed layouts
    use masked padding only. Query availability is independent of target labels.
    ``tactile_intensity_v1`` also accepts already normalized legacy deform maps;
    this is a representation compatibility rule, not sensor calibration.
    RGB maps preserve three optical channels; [0,1] means pixel range, not force.
    Their spatial ROI remains one channel and applies to all three color channels.
    """
    if not samples:
        raise ValueError("Cannot collate an empty tactile batch")
    validate_registry(region_names)
    if codec not in SUPPORTED_CODECS:
        raise ValueError(f"Unsupported tactile_codec: {codec}")
    accepted = (CODEC, INTENSITY_CODEC) if codec == INTENSITY_CODEC else (codec,)
    channels = 3 if codec == RGB_CODEC else 1
    observation_hw = observation_shape(observation_map_size) if observation_map_size is not None else None
    if observation_hw is not None:
        encoder_names = tuple(encoder_names or ())
        if not encoder_names or len(set(encoder_names)) != len(encoder_names) or any(
            not isinstance(name, str) or not name.strip() for name in encoder_names
        ):
            raise ValueError("Heterogeneous observations require registered encoder_names")
    ns = []
    for sample in samples:
        names = list(sample.get("tactile_region_names", ()))
        if "tactile_surface_ids" in sample:
            raise ValueError("surface_id was removed; compose one map per region before collation")
        if set(names) - set(region_names):
            raise ValueError("Region names must be registered")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate region; compose native sensors first")
        if names:
            if sample.get("tactile_codec") not in accepted:
                raise ValueError(f"Expected explicit tactile_codec in {accepted}")
            if codec in (INTENSITY_CODEC, RGB_CODEC) and not any(k in sample for k in ("tactile_sensor_ids", "tactile_sensor_names")):
                raise ValueError("Intensity/RGB maps require explicit tactile sensor/source identity")
        elif any(torch.as_tensor(sample[k]).numel() for k in (
            "tactile_maps", "tactile_valid", "future_tactile_maps", "future_tactile_valid",
            "tactile_observation_maps", "tactile_observation_spatial_valid", "tactile_encoder_ids"
        ) if k in sample):
            raise ValueError("Nonempty tactile tensors require named regional layout")
        ns.append(len(names))
    n = max(ns)
    specs = {
        "tactile_maps": ((history, n, channels, map_size, map_size), torch.float32, 0),
        "tactile_valid": ((history, n), torch.bool, False),
        "future_tactile_maps": ((future_frames, n, channels, map_size, map_size), torch.float32, 0),
        "future_tactile_valid": ((future_frames, n), torch.bool, False),
        "tactile_layout_present": ((n,), torch.bool, False),
        "tactile_query_mask": ((future_frames, n), torch.bool, False),
        "tactile_spatial_valid": ((n, 1, map_size, map_size), torch.bool, False),
        "tactile_sensor_ids": ((n,), torch.long, 0),
        "tactile_region_ids": ((n,), torch.long, -1),
    }
    if observation_hw is not None:
        specs.update({
            "tactile_observation_maps": ((history, n, channels, *observation_hw), torch.float32, 0),
            "tactile_observation_spatial_valid": ((n, 1, *observation_hw), torch.bool, False),
            "tactile_encoder_ids": ((n,), torch.long, -1),
        })
    device = next((torch.as_tensor(s["tactile_maps"]).device for s in samples if "tactile_maps" in s), "cpu")
    out = {key: torch.full((len(samples), *shape), fill, dtype=dtype, device=device)
           for key, (shape, dtype, fill) in specs.items()}
    for b, (sample, count) in enumerate(zip(samples, ns)):
        if not count:
            continue
        ids = [region_names.index(name) for name in sample["tactile_region_names"]]
        out["tactile_region_ids"][b, :count] = torch.tensor(ids, device=device)
        layout = torch.as_tensor(sample["tactile_layout_present"], device=device)
        if ("future_tactile_maps" in sample) != ("future_tactile_valid" in sample):
            raise ValueError("Future maps and their independent validity must be provided together")
        defaults = {
            "tactile_query_mask": layout[None].expand(future_frames, -1),
            "tactile_spatial_valid": torch.ones(count, 1, map_size, map_size, dtype=torch.bool, device=device),
            "tactile_sensor_ids": torch.zeros(count, dtype=torch.long, device=device),
            "future_tactile_maps": torch.zeros(future_frames, count, channels, map_size, map_size, device=device),
            "future_tactile_valid": torch.zeros(future_frames, count, dtype=torch.bool, device=device),
        }
        for key, (shape, dtype, _) in specs.items():
            if key == "tactile_region_ids":
                continue
            axis = 1 if key in (
                "tactile_maps", "tactile_valid", "future_tactile_maps", "future_tactile_valid", "tactile_query_mask",
                "tactile_observation_maps"
            ) else 0
            expected = list(shape)
            expected[axis] = count
            if key == "tactile_encoder_ids" and "tactile_encoder_names" in sample:
                names = list(sample["tactile_encoder_names"])
                if "tactile_encoder_ids" in sample or len(names) != count or set(names) - set(encoder_names):
                    raise ValueError("Ambiguous or unregistered tactile encoder names")
                value = torch.tensor([encoder_names.index(name) for name in names], device=device, dtype=torch.long)
            elif key == "tactile_sensor_ids" and "tactile_sensor_names" in sample:
                names = list(sample["tactile_sensor_names"])
                if "tactile_sensor_ids" in sample or len(names) != count or set(names) - set(sensor_names):
                    raise ValueError("Ambiguous or unregistered tactile sensor names")
                value = torch.tensor([sensor_names.index(name) for name in names], device=device, dtype=torch.long)
            else:
                if key not in sample and key not in defaults:
                    raise ValueError(f"Regional tactile sample requires {key}")
                value = torch.as_tensor(sample[key] if key in sample else defaults[key], device=device)
            if tuple(value.shape) != tuple(expected):
                raise ValueError(f"{key}: expected {tuple(expected)}, got {tuple(value.shape)}")
            if dtype in (torch.bool, torch.long) and value.dtype != dtype:
                raise ValueError(f"{key} requires {dtype}")
            if key == "tactile_sensor_ids" and ((value < 0) | (value >= len(sensor_names))).any():
                raise ValueError("Unregistered tactile sensor IDs")
            if key == "tactile_encoder_ids" and ((value < 0) | (value >= len(encoder_names))).any():
                raise ValueError("Unregistered tactile encoder IDs")
            if key.endswith("maps"):
                if value.dtype == torch.uint8:
                    value = value.float() / 255
                elif not value.is_floating_point():
                    raise ValueError("Maps must be uint8 or floating point in [0,1]")
            target = [b] + [slice(None)] * len(shape)
            target[axis + 1] = slice(0, count)
            out[key][tuple(target)] = value
    return out


class DeformHistory:
    """Causal, explicitly configured per-region live history at a fixed sample rate.

    Missing packets leave a valid-layout region unobserved; a black packet is
    valid. Append at every observation tick, not just when requesting a plan.
    """

    def __init__(
        self,
        region_names=FINGERTIP_NAMES,
        *,
        history=4,
        size=32,
        fps=30,
        sensor_id=0,
        registry_names=REGION_NAMES,
    ):
        validate_registry(registry_names)
        if not region_names or set(region_names) - set(registry_names) or len(set(region_names)) != len(region_names):
            raise ValueError("Invalid or duplicate region layout; compose native sensors first")
        if history < 1 or size < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid history settings")
        self.names = tuple(region_names)
        self.registry_names = tuple(registry_names)
        self.history, self.size, self.fps, self.sensor_id = history, size, fps, sensor_id
        self.frames = deque(maxlen=history)
        self.last_time = None

    def reset(self):
        self.frames.clear()
        self.last_time = None

    def append(self, timestamp_s, images_by_region):
        if not np.isfinite(timestamp_s) or set(images_by_region) - set(self.names):
            raise ValueError("Invalid timestamp or unexpected region")
        if self.last_time is not None and not np.isclose(
            timestamp_s - self.last_time, 1 / self.fps, rtol=0.1, atol=1e-6
        ):
            self.reset()
        maps = torch.zeros(len(self.names), 1, self.size, self.size)
        valid = torch.zeros(len(self.names), dtype=torch.bool)
        for i, name in enumerate(self.names):
            if images_by_region.get(name) is not None:
                maps[i] = resize_deform(images_by_region[name], self.size)
                valid[i] = True
        self.frames.append((maps, valid))
        self.last_time = float(timestamp_s)
        history = torch.zeros(self.history, *maps.shape)
        mask = torch.zeros(self.history, len(self.names), dtype=torch.bool)
        for i, (value, enabled) in enumerate(self.frames, start=self.history - len(self.frames)):
            history[i], mask[i] = value, enabled
        return dict(
            tactile_maps=history.numpy(),
            tactile_valid=mask.numpy(),
            tactile_region_ids=np.array([self.registry_names.index(name) for name in self.names], dtype=np.int64),
            tactile_layout_present=np.ones(len(self.names), dtype=bool),
            tactile_sensor_ids=np.full(len(self.names), self.sensor_id, dtype=np.int64),
        )
