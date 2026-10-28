"""Align each global microbatch's modality across data-parallel workers.

ZeRO-2's gradient buckets must follow the same graph on every rank. Robot
sources may mix within a batch; strict video-only samples use separate batches.
All virtual windows are retained, with at most one padded global batch per modality.
"""
import math

import numpy as np
from torch.utils.data import Sampler

MODALITY_CODES = {"video_action": 0, "video_only": 1, "video_tactile": 2, "video_action_tactile": 3}


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
        self._source_codes = np.asarray([MODALITY_CODES[m] for m in dataset.source_modalities])
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
        schedule = np.concatenate([np.full(math.ceil(len(p) / block), i, dtype=np.uint8)
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
