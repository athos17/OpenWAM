"""Sharpa tactile input contract. Distance is raycast geometry, NOT calibrated force."""

from collections import deque

import numpy as np
import torch
from torch.nn import functional as F

SCHEMA = "sharpa_tactile_v1"
SITE_NAMES = tuple(
    f"{side}_{finger}_elastomer"
    for side in ("right", "left")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
)


def pool_tactile(distance, contact, size=32):
    """Contact-preserving pooling of [...,H,W] arrays, returns [...,2,size,size].

    Channel 0 = mean distance of contacting pixels in metres; channel 1 =
    ANY contact in the cell. A single contacting raw pixel survives reduction.
    The caller separately marks missing/invalid sensor frames.
    """
    distance = torch.as_tensor(np.asarray(distance), dtype=torch.float32)
    contact = torch.as_tensor(np.asarray(contact), dtype=torch.bool)
    if distance.shape != contact.shape or distance.ndim < 2 or size < 1:
        raise ValueError("Expected matching distance/contact maps")
    shape = distance.shape[:-2]
    valid = torch.isfinite(distance).flatten(-2).all(-1)
    distance = torch.nan_to_num(distance).reshape(-1, 1, *distance.shape[-2:])
    contact = contact.reshape_as(distance).float()
    occupancy = F.adaptive_avg_pool2d(contact, size)
    mean = F.adaptive_avg_pool2d(distance * contact, size) / occupancy.clamp_min(1e-12)
    present = F.adaptive_max_pool2d(contact, size)
    maps = torch.cat((mean, present), dim=1).reshape(*shape, 2, size, size)
    maps = maps.masked_fill(~valid[..., None, None, None], 0)
    return maps.numpy(), valid.numpy()


def read_tactile_frames(h5, start, stop, size=32):
    """Read canonical named sensors, independent of HDF5 group iteration order."""
    maps = np.zeros((stop - start, 10, 2, size, size), dtype=np.float32)
    valid = np.zeros((stop - start, 10), dtype=bool)
    for i, site in enumerate(SITE_NAMES):
        dk, ck = f"robot/tactile/distance_along_normal_m/{site}", f"robot/tactile/contact_mask/{site}"
        if dk not in h5 or ck not in h5:
            continue
        if len(h5[dk]) < stop or len(h5[ck]) < stop:
            raise ValueError(f"Truncated tactile stream for {site}")
        maps[:, i], valid[:, i] = pool_tactile(h5[dk][start:stop], h5[ck][start:stop], size)
    return maps, valid


class SharpaTactileHistory:
    """Client-side 20 Hz buffer. Append EVERY observation, including executed
    open-loop actions; send the resulting fields only when requesting a plan.
    Call reset between episodes. Gaps/repeated timestamps reset history.
    """

    def __init__(self, history=4, size=32, fps=20):
        if history < 1 or size < 1 or fps <= 0:
            raise ValueError("Invalid history settings")
        self.history, self.size, self.dt = history, size, 1.0 / fps
        self.frames = deque(maxlen=history)
        self.last_time = None

    def reset(self):
        self.frames.clear()
        self.last_time = None

    def append(self, timestamp_s, distance_by_site, contact_by_site):
        if not np.isfinite(timestamp_s):
            raise ValueError("Tactile timestamps must be finite seconds")
        if self.last_time is not None and not np.isclose(timestamp_s - self.last_time, self.dt, rtol=0.1, atol=1e-6):
            self.reset()
        maps = np.zeros((10, 2, self.size, self.size), np.float32)
        valid = np.zeros(10, bool)
        for i, site in enumerate(SITE_NAMES):
            if site in distance_by_site and site in contact_by_site:
                maps[i], valid[i] = pool_tactile(distance_by_site[site], contact_by_site[site], self.size)
        self.frames.append((maps, valid))
        self.last_time = float(timestamp_s)
        out = np.zeros((self.history, *maps.shape), np.float32)
        mask = np.zeros((self.history, 10), bool)
        offset = self.history - len(self.frames)
        for i, (value, enabled) in enumerate(self.frames):
            out[offset + i], mask[offset + i] = value, enabled
        return {"tactile_maps": out, "tactile_valid": mask}
