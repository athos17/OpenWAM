"""Manifest-bound EgoDex video/language input with no human action targets."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.video_io import decode_video_frames


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class EgoDexVideoDataset(BaseDataset):
    """One cumulative episode index, with the same padded train starts as LeRobotV3Reader.

    No raw pose Parquet or directory discovery is used. Missing video or invalid
    metadata fails explicitly rather than silently substituting another episode.
    """
    action_dim = 80
    normalization_stats = None
    normalization_stats_path = None
    training_modality = 'video_only'
    CAMERA_LAYOUT = ['observation.images.camera', '__missing_left__', '__missing_right__']

    def __init__(self, dataset_dir, *, split='train', num_frames=33, video_stride=4,
                 window_stride=1, height=384, width=320, color_jitter=None,
                 enable_action_supervision=False, enable_proprio_supervision=False, seed=42):
        if enable_action_supervision or enable_proprio_supervision:
            raise ValueError('EgoDex is strictly video-only; action/proprio supervision is forbidden')
        if split != 'train':
            raise ValueError('This frozen EgoDex release contains official train only')
        if (num_frames, video_stride, height, width) != (33, 4, 384, 320):
            raise ValueError('EgoDex Alpha contract requires 33 steps, stride 4, height 384, width 320')
        if int(window_stride) != window_stride or window_stride < 1:
            raise ValueError('window_stride must be a positive integer')
        self.root = Path(dataset_dir).resolve(strict=True)
        self.info = json.loads((self.root/'dataset_info.json').read_text())
        if self.info['source_split'] != 'train' or self.info['training_modality'] != 'video_only':
            raise ValueError('Invalid dataset modality or split')
        index_path = self.root/'episodes.parquet'
        actual_hash = file_sha256(index_path)
        if actual_hash != self.info['index_sha256']:
            raise ValueError('EgoDex episode index checksum mismatch')
        self.episodes = pq.read_table(index_path).to_pydict()
        self.lengths = np.asarray(self.episodes['length'], dtype=np.int64)
        self.video_offsets = np.asarray(self.episodes['video_from_frame'], dtype=np.int64)
        ids = self.episodes['id']
        if len(set(ids)) != len(ids) or len(ids) != self.info['total_episodes']:
            raise ValueError('Duplicate or inconsistent episode identity')
        if np.any(self.lengths <= 0) or int(self.lengths.sum()) != self.info['total_frames']:
            raise ValueError('Invalid selected frame counts')
        if np.any(self.video_offsets < 0) or not all(x == 'train' for x in self.episodes['source_split']):
            raise ValueError('Invalid source split or video offsets')
        if not all(isinstance(p, str) and p.strip() for p in self.episodes['prompt']):
            raise ValueError('Missing episode language')
        self.video_files = {}
        for rel in set(self.episodes['video_path']):
            candidate = self.root/rel
            if Path(rel).is_absolute() or '..' in Path(rel).parts or candidate.is_symlink():
                raise ValueError(f'Video must be held inside the published dataset: {rel}')
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(self.root) or not resolved.is_file():
                raise ValueError(f'External or missing video: {rel}')
            self.video_files[rel] = str(resolved)
        self.window_stride = int(window_stride)
        self.seed, self.epoch = int(seed), 0
        self.num_frames, self.height, self.width = num_frames, height, width
        self.video_steps = np.arange(0, num_frames, video_stride, dtype=np.int64)
        # Match the base train reader: min_window_len=1, including masked tails.
        starts = (self.lengths - 1)//self.window_stride + 1
        self.cumulative = np.concatenate(([0], np.cumsum(starts)))
        self.color_jitter = None
        if color_jitter_enabled(color_jitter):
            self.color_jitter = VideoColorJitter(**{
                key: get_cfg(color_jitter, key, default)
                for key, default in [('brightness', .2), ('contrast', .2), ('saturation', .2), ('hue', 0.)]
            })
        self.training_data_contract = {
            'dataset_type': 'egodex_video', 'selection_id': self.info['selection_id'],
            'source_revision': self.info['source_revision'],
            'selection_sha256': self.info['selection_sha256'], 'index_sha256': actual_hash,
            'training_modality': 'video_only', 'source_split': 'train', 'normalization': 'none',
            'num_frames': num_frames, 'video_stride': video_stride, 'window_stride': self.window_stride,
            'height': height, 'width': width, 'camera_layout': self.CAMERA_LAYOUT,
            'color_jitter': None if self.color_jitter is None else {
                key: getattr(self.color_jitter, key) for key in ['brightness', 'contrast', 'saturation', 'hue']},
            'total_windows': len(self),
            'augmentation_seed': self.seed, 'augmentation_rng': 'per_seed_epoch_window_v1',
        }

    def __len__(self):
        return int(self.cumulative[-1])

    def locate(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.cumulative, index, side='right')-1)
        offset = (index-int(self.cumulative[episode]))*self.window_stride
        return episode, offset

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        episode, offset = self.locate(index)
        valid_length = min(self.num_frames, int(self.lengths[episode])-offset)
        valid = self.video_steps < valid_length
        frame_indices = (self.video_offsets[episode]+offset+self.video_steps[valid]).tolist()
        path = self.video_files[self.episodes['video_path'][episode]]
        head = decode_video_frames(path, frame_indices, 256, 320)
        head += [head[-1]]*(len(self.video_steps)-len(head))
        composed = [assemble_multiview_layout(
            {self.CAMERA_LAYOUT[0]: frame}, self.CAMERA_LAYOUT,
            out_h=self.height, out_w=self.width, return_missing_mask=True) for frame in head]
        video = [x[0] for x in composed]
        if self.color_jitter is not None:
            # Resume/worker scheduling must not change the augmentation for a
            # given window within an epoch. Preserve the caller's global RNG.
            state = random.getstate()
            random.seed(f'egodex/{self.seed}/{self.epoch}/{int(index)}')
            try:
                video = self.color_jitter.apply({
                    'video': video, 'video_jitter_exclusion_masks': [x[1] for x in composed]})['video']
            finally:
                random.setstate(state)
        return {
            'video': video, 'video_mask': torch.from_numpy(valid),
            'action': torch.zeros((32, 80)), 'action_mask': torch.zeros((32, 80), dtype=torch.bool),
            'proprio': torch.zeros((1, 80)), 'proprio_mask': torch.zeros((1, 80), dtype=torch.bool),
            'prompt': self.episodes['prompt'][episode], 'first_frame_image': [video[0]], 'vace_video': None,
            'training_modality': 'video_only', 'source_id': 'egodex',
            'episode_id': self.episodes['id'][episode], 'episode_start_frame': offset,
        }

    @classmethod
    def from_config(cls, config, split='train'):
        for key in ['enable_action_supervision', 'enable_proprio_supervision']:
            if get_cfg(config, key, False):
                raise ValueError(f'EgoDex forbids {key}')
        return cls(get_cfg(config, 'dataset_dir'), split=split, **{
            key: get_cfg(config, key, default) for key, default in [
                ('num_frames', 33), ('video_stride', 4), ('window_stride', 1),
                ('height', 384), ('width', 320), ('color_jitter', None), ('seed', 42)]})
