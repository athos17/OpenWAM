"""Named tactile surfaces. Availability is metadata, never inferred from intensity."""

from collections import deque

import numpy as np
import torch
from torch.nn import functional as F

from .registry import FINGERTIP_NAMES, REGION_NAMES, validate_registry

SCHEMA = "tactile_regions_v1"
CODEC = "sharpa_deform_gray_v1"


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
):
    """Pad variable region lists, preserving identity and separate availability.

    Returns only tactile fields; VA preprocessing remains architecture-owned.
    All samples in this version use the same, explicit deform codec/grid.
    """
    if not samples:
        raise ValueError("Cannot collate an empty tactile batch")
    validate_registry(region_names)
    ns = []
    for sample in samples:
        if sample.get("tactile_codec") != CODEC:
            raise ValueError("Expected explicit tactile_codec=sharpa_deform_gray_v1")
        names = list(sample["tactile_region_names"])
        if not names or set(names) - set(region_names):
            raise ValueError("Region names must be nonempty and registered")
        if "tactile_surface_ids" in sample:
            raise ValueError("surface_id was removed; compose one map per region before collation")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate region; compose native sensors first")
        ns.append(len(names))
    n = max(ns)
    out = {}
    specs = {
        "tactile_maps": ((history, n, 1, map_size, map_size), torch.float32, 0),
        "tactile_valid": ((history, n), torch.bool, False),
        "future_tactile_maps": ((future_frames, n, 1, map_size, map_size), torch.float32, 0),
        "future_tactile_valid": ((future_frames, n), torch.bool, False),
        "tactile_layout_present": ((n,), torch.bool, False),
        "tactile_query_mask": ((future_frames, n), torch.bool, False),
        "tactile_spatial_valid": ((n, 1, map_size, map_size), torch.bool, False),
        "tactile_sensor_ids": ((n,), torch.long, 0),
        "tactile_region_ids": ((n,), torch.long, -1),
    }
    device = torch.as_tensor(samples[0]["tactile_maps"]).device
    for key, (shape, dtype, fill) in specs.items():
        out[key] = torch.full((len(samples), *shape), fill, dtype=dtype, device=device)
    for b, (sample, count) in enumerate(zip(samples, ns)):
        ids = [region_names.index(name) for name in sample["tactile_region_names"]]
        out["tactile_region_ids"][b, :count] = torch.tensor(ids, device=device)
        layout = torch.as_tensor(sample["tactile_layout_present"], device=device)
        for key, (shape, dtype, _) in specs.items():
            if key == "tactile_region_ids":
                continue
            axis = (
                1
                if key
                in (
                    "tactile_maps",
                    "tactile_valid",
                    "future_tactile_maps",
                    "future_tactile_valid",
                    "tactile_query_mask",
                )
                else 0
            )
            expected = list(shape)
            expected[axis] = count
            defaults = {
                "tactile_query_mask": layout[None].expand(future_frames, -1),
                "tactile_spatial_valid": torch.ones(count, 1, map_size, map_size, dtype=torch.bool, device=device),
                "tactile_sensor_ids": torch.zeros(count, dtype=torch.long, device=device),
            }
            if key == "tactile_sensor_ids" and "tactile_sensor_names" in sample:
                names = list(sample["tactile_sensor_names"])
                if "tactile_sensor_ids" in sample or len(names) != count or set(names) - set(sensor_names):
                    raise ValueError("Ambiguous or unregistered tactile sensor names")
                value = torch.tensor([sensor_names.index(name) for name in names], device=device, dtype=torch.long)
            else:
                value = torch.as_tensor(sample[key] if key in sample else defaults[key], device=device)
            if tuple(value.shape) != tuple(expected):
                raise ValueError(f"{key}: expected {tuple(expected)}, got {tuple(value.shape)}")
            if dtype in (torch.bool, torch.long) and value.dtype != dtype:
                raise ValueError(f"{key} requires {dtype}")
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
