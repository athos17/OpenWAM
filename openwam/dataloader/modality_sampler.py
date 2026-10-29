"""Align each global microbatch's modality, codec, and observation shape across parallel workers.

ZeRO-2's gradient buckets must follow the same graph on every rank. Robot
sources may mix within a batch; strict video-only samples use separate batches.
All virtual windows are retained, with at most one padded batch per observation bucket.
"""
import math

import numpy as np
from torch.utils.data import Sampler

from openwam.tactile.regions import observation_shape

MODALITY_CODES = {"video_action": 0, "video_only": 1, "video_tactile": 2, "video_action_tactile": 3}
TACTILE_CODEC_CODES = {None: 0, "tactile_intensity_v1": 1, "sharpa_deform_gray_v1": 2, "tactile_rgb_v1": 3}


def batch_tactile_codec(samples):
    """One selected tactile representation; never mix RGB/intensity in a batch."""
    if any("tactile" in sample.get("training_modality", "video_action")
           and sample.get("tactile_codec", "tactile_intensity_v1") is None for sample in samples):
        raise ValueError("Tactile microbatch requires a non-null tactile codec")
    codecs = {sample.get("tactile_codec", "tactile_intensity_v1")
              if "tactile" in sample.get("training_modality", "video_action") else None for sample in samples}
    if len(codecs) != 1 or next(iter(codecs)) not in TACTILE_CODEC_CODES:
        raise ValueError("Microbatch contains mixed or unsupported tactile codecs")
    return next(iter(codecs))


def batch_tactile_observation_shape(samples):
    """Return None for legacy/N=0 or one (H,W) observation shape.

    Shape is separate from encoder identity: compatible frontend routes may
    coexist within a microbatch, while native matrices and image observations
    with different resolutions cannot be stacked together.
    """
    shapes = set()
    for sample in samples:
        maps = sample.get("tactile_observation_maps")
        declared = sample.get("tactile_observation_shape")
        has_tactile = "tactile" in sample.get("training_modality", "video_action")
        if maps is None:
            if declared is not None:
                raise ValueError("Declared tactile observation shape requires observation maps")
            shapes.add(None)
            continue
        shape = getattr(maps, "shape", ())
        codec = sample.get("tactile_codec", "tactile_intensity_v1")
        channels = 3 if codec == "tactile_rgb_v1" else 1
        if len(shape) != 5 or shape[2] != channels or shape[0] < 1 or min(shape[-2:]) < 1:
            raise ValueError("Tactile observations require [T,N,C,H,W] consistent with their codec")
        if not has_tactile:
            if shape[1] != 0:
                raise ValueError("Non-tactile samples cannot contain observation regions")
            shapes.add(None)
            continue
        if shape[1] < 1:
            raise ValueError("Tactile samples require a nonempty observation layout")
        actual = tuple(int(v) for v in shape[-2:])
        if declared is not None and observation_shape(declared) != actual:
            raise ValueError("Declared tactile observation shape differs from its maps")
        shapes.add(actual)
    if len(shapes) != 1:
        raise ValueError("Microbatch contains mixed tactile observation shapes; use global observation buckets")
    return next(iter(shapes))


def batch_modality_code(samples):
    """One deterministic code on every rank; -1 is an invalid mixed batch."""
    codes = {MODALITY_CODES.get(s.get("training_modality", "video_action"), -1) for s in samples}
    return codes.pop() if len(codes) == 1 else -1


class GlobalModalitySampler(Sampler):
    def __init__(self, dataset, batch_size: int, world_size: int):
        if batch_size < 1 or world_size < 1:
            raise ValueError("batch_size and world_size must be positive")
        self.dataset = dataset
        self.global_batch_size = int(batch_size) * int(world_size)
        if any(m not in MODALITY_CODES for m in dataset.source_modalities):
            raise ValueError("Unregistered training modality")
        codecs = getattr(dataset, "source_tactile_codecs", [
            "tactile_intensity_v1" if "tactile" in mode else None for mode in dataset.source_modalities])
        if len(codecs) != len(dataset.source_modalities) or any(
            (codec is not None) != ("tactile" in mode) for mode, codec in zip(dataset.source_modalities, codecs)
        ):
            raise ValueError("Every tactile source requires a codec; sources without tactile use null")
        if any(codec not in TACTILE_CODEC_CODES for codec in codecs):
            raise ValueError("Unregistered tactile codec")
        self._source_codes = np.asarray([4 * MODALITY_CODES[m] + TACTILE_CODEC_CODES[c]
                                         for m, c in zip(dataset.source_modalities, codecs)])
        shapes = list(getattr(dataset, "source_tactile_observation_shapes", [None] * len(codecs)))
        if len(shapes) != len(codecs):
            raise ValueError("Observation shapes must match the number of mixture sources")
        shapes = [observation_shape(shape) if shape is not None else None for shape in shapes]
        if any(shape is not None and "tactile" not in mode
               for shape, mode in zip(shapes, dataset.source_modalities)):
            raise ValueError("Sources without tactile must use a null observation shape")
        self.observation_shape_batching = any(shape is not None for shape in shapes)
        if self.observation_shape_batching:
            # Compress sorted composite keys. The legacy path retains the exact
            # previous flag ordering and random schedule when no shape is used.
            keys = [(int(code), *(shape or (0, 0))) for code, shape in zip(self._source_codes, shapes)]
            order = {key: i for i, key in enumerate(sorted(set(keys)))}
            self._source_codes = np.asarray([order[key] for key in keys])
        self._modalities = tuple(sorted(set(self._source_codes.tolist())))

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def _counts(self):
        by_source = self.dataset.dataset_sample_counts()
        return [sum(by_source[name] for name, mode in zip(self.dataset.names, self._source_codes) if mode == flag)
                for flag in self._modalities]

    def __len__(self):
        block = self.global_batch_size
        return sum(math.ceil(n / block) * block for n in self._counts())

    def __iter__(self):
        flags = self._source_codes[self.dataset._index_map[:, 0]]
        positions = [np.flatnonzero(flags == flag) for flag in self._modalities]
        block = self.global_batch_size
        schedule = np.concatenate([np.full(math.ceil(len(p) / block), i, dtype=np.int64)
                                   for i, p in enumerate(positions)])
        np.random.RandomState(self.dataset._seed ^ 0x4D4F44).shuffle(schedule)
        cursors = [0] * len(positions)
        for modality in schedule:
            modality = int(modality)
            pos = positions[modality]
            start = cursors[modality]
            chunk = pos[start:start + block]
            cursors[modality] += len(chunk)
            if len(chunk) < block:
                chunk = np.concatenate((chunk, np.resize(pos, block - len(chunk))))
            yield from map(int, chunk)
