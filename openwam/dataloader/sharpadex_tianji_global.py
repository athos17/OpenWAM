"""Global EEF62 pretraining data with source video offsets and command masks."""
from pathlib import Path
import numpy as np
from openwam.dataloader.sharpadex_tianji import SharpaDexTianjiDataset
from openwam.dataloader.bases.multi_lerobot_v3_reader import MultiLeRobotV3Reader
from openwam.dataloader.utils.normalization import apply_normalization


class MultiSharpaDexTianjiGlobalDataset(MultiLeRobotV3Reader):
    """Expose the one shared statistics artifact for checkpoint export.

    Samples are already normalized by each bucket. Keep normalization_stats=None
    to prevent a second normalization; only expose the artifact's source path.
    """

    def __init__(self, buckets):
        super().__init__(buckets)
        paths = [getattr(bucket, 'normalization_stats_path', None) for bucket in buckets]
        self.normalization_stats_path = None
        if all(path is None for path in paths):
            return
        if any(path is None for path in paths):
            raise ValueError('Global SharpaDex mixes normalized and unnormalized buckets')
        resolved = {Path(path).resolve(strict=True) for path in paths}
        if len(resolved) != 1:
            raise ValueError('Global SharpaDex requires one shared normalization file')
        self.normalization_stats_path = str(resolved.pop())


class SharpaDexTianjiGlobalDataset(SharpaDexTianjiDataset):
    DATASET_NAME='SharpaDexTianjiGlobal'
    NEEDED_COLS=('action','observation.state','action_validity','task_index')

    def _resolve_cameras(self,info):
        if info.get('openwam_contract',{}).get('storage_vector_dim')!=62:
            raise ValueError('Expected native EEF62 global pretraining data')
        return super()._resolve_cameras(info)

    def _add_episode_offsets(self,eps):
        self._add_data_offsets(eps)
        for cam in self._video_cameras():
            values=eps[f'videos/{cam}/from_timestamp'].to_numpy(dtype=np.float64)*self._fps
            if not np.isfinite(values).all() or np.any(values<0) or not np.allclose(values,np.rint(values),atol=1e-3,rtol=0):
                raise ValueError('Invalid source video offset')
            eps[self._video_offset_col(cam)]=np.rint(values).astype(np.int64)

    def _action_20d(self,win):
        x=np.stack(win['action']).astype(np.float32)
        if x.shape[1]!=62 or not np.isfinite(x).all():raise ValueError('Invalid EEF62 action')
        self._window_action_validity=np.repeat(np.stack(win['action_validity']),[9,22,9,22],axis=1)
        return apply_normalization(x,self._normalization_stats,self._normalize_mode)

    def _proprio_20d(self,win):
        x=np.stack(win['observation.state'].values[:1]).astype(np.float32)
        if x.shape[1]!=62 or not np.isfinite(x).all():raise ValueError('Invalid EEF62 state')
        return apply_normalization(x,self._proprio_stats,self._normalize_mode)

    def _finalize_action(self,action,actual_raw_len,**kwargs):
        # OpenWAM-fast's reader passes the optional hold argument here.
        values,mask=super()._finalize_action(action,actual_raw_len,**kwargs)
        n=min(len(self._window_action_validity),len(mask))
        if self._unify:
            valid=np.zeros((n,self._unify_dim),dtype=bool)
            valid[:,self._unify_dst_index]=self._window_action_validity[:n]
        else:valid=self._window_action_validity[:n]
        mask[:n]&=valid
        values[~mask]=0.
        return values,mask

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiSharpaDexTianjiGlobalDataset
