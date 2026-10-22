"""Converted SharpaDex with Tianji absolute EEF + SharpaWave joint supervision."""
import numpy as np
import pandas as pd
from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.normalization import STAT_KEYS, apply_normalization

ROT_DIMS = list(range(3, 9)) + list(range(34, 40))


def tianji_raw62(source):
    source = np.asarray(source, dtype=np.float32)
    if source.shape[-1] != 76:
        raise ValueError(f"Expected 76-D Tianji source vector: {source.shape}")
    return np.concatenate([source[...,58:67], source[...,14:36], source[...,67:76], source[...,36:58]], axis=-1)


class SharpaDexTianjiDataset(LeRobotV3Reader):
    DATASET_NAME = 'SharpaDexTianji'
    NEEDED_COLS = ('action', 'observation.state', 'task_index')
    ACTION_DIM = 62
    DEPLOY_ACTION_MODE = 'eef'
    DEFAULT_NORMALIZE_MODE = 'min-max'
    # Require all selected cameras to decode during the pilot.
    WRIST_DECODE_TOLERATED = ()

    def _load_prompts(self):
        tasks = pd.read_parquet(self._dataset_dir / 'meta/tasks.parquet')
        texts = tasks['task'].tolist() if 'task' in tasks.columns else tasks.index.tolist()
        if tasks['task_index'].duplicated().any() or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Duplicate task indices or missing task text')
        self._task_idx_to_text = dict(zip(tasks['task_index'].tolist(), texts))

    def _resolve_cameras(self, info):
        c = info.get('openwam_contract', {})
        if (info.get('codebase_version') != 'v3.0' or c.get('raw_action_dim') != 62
            or c.get('rotation_6d_layout') != '[R00,R10,R20,R01,R11,R21]'
            or c.get('action_pose_frame') != 'base_link'
            or c.get('action_representation') != 'absolute_eef_and_absolute_hand_joints'):
            raise ValueError('Unexpected converted Tianji dataset contract')
        return ('observation.images.left_head', 'observation.images.left_wrist', 'observation.images.right_wrist')

    def _load_stats(self, info):
        self._proprio_stats = None
        if self._normalize_mode in (None, 'none', 'null'):
            return None
        if self._normalize_mode != 'min-max':
            raise ValueError('Tianji pilot uses min-max normalization')
        path = self._dataset_dir / 'meta/normalization_stats.npy'
        payload = np.load(path, allow_pickle=True).item()
        result = {}
        identity = dict(mean=0., std=1., min=-1., max=1., q01=-1., q99=1.)
        for name in ['eef', 'eef_state']:
            block = {k: np.asarray(payload[name][k], dtype=np.float32) for k in STAT_KEYS}
            for k,v in block.items():
                if v.shape != (62,) or not np.isfinite(v).all() or not np.allclose(v[ROT_DIMS], identity[k]):
                    raise ValueError(f'Invalid {name}.{k} statistics / rot6d convention')
            if np.any(block['max'] < block['min']):
                raise ValueError(f'Invalid range in {name}')
            result[name] = block
        self.normalization_stats_path = str(path)
        self._proprio_stats = result['eef_state']
        return result['eef']

    def _action_20d(self, win):
        return apply_normalization(tianji_raw62(np.stack(win['action'])), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win):
        return apply_normalization(tianji_raw62(np.stack(win['observation.state'].values[:1])), self._proprio_stats, self._normalize_mode)
