"""Bench2Dex's stored grayscale tacmap, not its distance/contact channels."""

import numpy as np
import torch

from openwam.dataloader.bench2dex_tactile import Bench2DexTactileDataset, Bench2DexTactileMixture
from openwam.tactile.regions import CODEC, FINGERTIP_NAMES, resize_deform
from openwam.tactile.sharpa import SITE_NAMES


class Bench2DexDeformDataset(Bench2DexTactileDataset):
    def _check_cache(self, manifest_path):
        raise ValueError("Geometry/contact caches cannot train deform; read the stored uint8 tacmap")

    def observation_extras(self, h5, episode, start, segment_start):
        offsets = np.arange(1, self.future_frames + 1, dtype=np.int64) * self.future_stride
        end = start + int(offsets[-1]) + 1
        if not any(a == segment_start and a <= start and end <= b for a, b in episode["segments"]):
            raise ValueError("Future deform labels cross a valid trajectory segment")
        history_rows = np.arange(start - self.history + 1, start + 1)
        future_rows = start + offsets
        history = torch.zeros(self.history, 10, 1, self.map_size, self.map_size)
        future = torch.zeros(self.future_frames, 10, 1, self.map_size, self.map_size)
        observed = torch.zeros(self.history, 10, dtype=torch.bool)
        labels = torch.zeros(self.future_frames, 10, dtype=torch.bool)
        present = torch.zeros(10, dtype=torch.bool)
        for i, site in enumerate(SITE_NAMES):
            key = f"robot/tactile/tacmap/{site}"
            if key not in h5:
                continue
            ds = h5[key]
            if ds.dtype != np.uint8 or ds.ndim != 3 or len(ds) < end:
                raise ValueError(f"Invalid or truncated stored deformation stream: {key}")
            present[i] = True
            for t, row in enumerate(history_rows):
                if row >= segment_start:
                    history[t, i] = resize_deform(ds[int(row)], self.map_size)
                    observed[t, i] = True
            for t, row in enumerate(future_rows):
                future[t, i] = resize_deform(ds[int(row)], self.map_size)
                labels[t, i] = True
        if not present.any():
            raise ValueError("No stored uint8 tacmap in episode; export real simulated deform maps first")
        return dict(
            tactile_maps=history,
            tactile_valid=observed,
            future_tactile_maps=future,
            future_tactile_valid=labels,
            future_tactile_offsets=torch.from_numpy(offsets),
            tactile_layout_present=present,
            tactile_region_names=list(FINGERTIP_NAMES),
            tactile_codec=CODEC,
            tactile_sensor_names=["bench2dex_sim"] * 10,
            tactile_fps=20,
        )


class Bench2DexDeformMixture(Bench2DexTactileMixture):
    dataset_class = Bench2DexDeformDataset

    @classmethod
    def from_config(cls, config, split="train"):
        result = super().from_config(config, split)
        if hasattr(result, "training_data_contract"):
            result.training_data_contract["dataset_type"] = "bench2dex_deform"
            result.training_data_contract["tactile_codec"] = CODEC
        return result
