"""Align each global microbatch's modality across data-parallel workers.

ZeRO-2's gradient buckets must follow the same graph on every rank. Robot
sources may mix within a batch; strict video-only samples use separate batches.
All virtual windows are retained, with at most one padded global batch per modality.
"""
import math

import numpy as np
from torch.utils.data import Sampler


class GlobalModalitySampler(Sampler):
    def __init__(self, dataset, batch_size: int, world_size: int):
        if batch_size < 1 or world_size < 1:
            raise ValueError("batch_size and world_size must be positive")
        self.dataset = dataset
        self.global_batch_size = int(batch_size) * int(world_size)
        self._video_sources = np.asarray([m == "video_only" for m in dataset.source_modalities])

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def _counts(self):
        by_source = self.dataset.dataset_sample_counts()
        return [sum(by_source[name] for name, video in zip(self.dataset.names, self._video_sources) if video == flag)
                for flag in (False, True)]

    def __len__(self):
        block = self.global_batch_size
        return sum(math.ceil(n / block) * block for n in self._counts())

    def __iter__(self):
        flags = self._video_sources[self.dataset._index_map[:, 0]]
        positions = [np.flatnonzero(flags == flag) for flag in (False, True)]
        block = self.global_batch_size
        schedule = np.concatenate([np.full(math.ceil(len(p) / block), i, dtype=np.uint8)
                                   for i, p in enumerate(positions)])
        np.random.RandomState(self.dataset._seed ^ 0x4D4F44).shuffle(schedule)
        cursors = [0, 0]
        for modality in schedule:
            modality = int(modality)
            pos = positions[modality]
            start = cursors[modality]
            chunk = pos[start:start + block]
            cursors[modality] += len(chunk)
            if len(chunk) < block:
                chunk = np.concatenate((chunk, np.resize(pos, block - len(chunk))))
            yield from map(int, chunk)
