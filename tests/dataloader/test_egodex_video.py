import hashlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openwam.dataloader.egodex_video import EgoDexVideoDataset
from openwam.train.utils.checkpointing import save_training_data_contract, verify_resume_training_data_contract


@pytest.fixture
def published(tmp_path):
    (tmp_path/'video.mp4').write_bytes(b'unit-test-decoder-is-injected')
    rows=[dict(id='train/task/episode_000000',task='task',source_split='train',length=7,
               video_from_frame=10,video_to_frame=17,video_path='video.mp4',prompt='pick up'),
          dict(id='train/task/episode_000001',task='task',source_split='train',length=40,
               video_from_frame=30,video_to_frame=70,video_path='video.mp4',prompt='put down')]
    path=tmp_path/'episodes.parquet';pq.write_table(pa.Table.from_pylist(rows),path)
    info=dict(source_split='train',training_modality='video_only',total_frames=47,total_episodes=2,
              index_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selection_id='test',
              source_revision='fixed',selection_sha256='selection')
    (tmp_path/'dataset_info.json').write_text(json.dumps(info))
    return tmp_path


def test_boundaries_padding_and_no_pose_loading(published, monkeypatch):
    calls=[]
    def decode(path,indices,height,width):
        calls.append(indices)
        return [Image.new('RGB',(width,height),(i,i,i)) for i in indices]
    monkeypatch.setattr('openwam.dataloader.egodex_video.decode_video_frames',decode)
    ds=EgoDexVideoDataset(published)
    assert len(ds)==47
    first=ds[0]
    assert calls[-1]==[10,14]
    assert first['video_mask'].tolist()==[True,True]+[False]*7
    assert np.array_equal(np.asarray(first['video'][1]),np.asarray(first['video'][-1]))
    assert first['video'][0].size==(320,384)
    assert np.asarray(first['video'][0])[256:].sum()==0
    assert not first['action_mask'].any() and not first['proprio_mask'].any()
    assert first['action'].shape==(32,80) and first['proprio'].shape==(1,80)
    assert first['training_modality']=='video_only'
    assert ds[6]['episode_id'].endswith('000000') and calls[-1]==[16]
    assert ds[7]['episode_id'].endswith('000001') and calls[-1]==list(range(30,63,4))
    assert ds[46]['video_mask'].sum()==1 and calls[-1]==[69]
    for index in [-1,47]:
        with pytest.raises(IndexError):ds[index]


def test_jitter_reproducible_and_missing_slots_black(published,monkeypatch):
    monkeypatch.setattr('openwam.dataloader.egodex_video.decode_video_frames',
        lambda path,indices,height,width:[Image.new('RGB',(width,height),(90,120,160)) for _ in indices])
    ds=EgoDexVideoDataset(published,color_jitter={'brightness':.2,'contrast':.2,'saturation':.2},seed=17)
    first=ds[0]['video'];again=ds[0]['video']
    assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(first,again))
    assert all(np.asarray(f)[256:].sum()==0 for f in first)
    assert np.array_equal(np.asarray(first[0]),np.asarray(first[-1]))
    ds.set_epoch(1)
    assert not np.array_equal(np.asarray(first[0]),np.asarray(ds[0]['video'][0]))


def test_fail_closed_on_supervision_and_changed_index(published):
    with pytest.raises(ValueError,match='supervision'):
        EgoDexVideoDataset(published,enable_action_supervision=True)
    with pytest.raises(ValueError,match='train only'):
        EgoDexVideoDataset(published,split='val')
    with (published/'episodes.parquet').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        EgoDexVideoDataset(published)


def test_resume_binds_selection_and_sampling(published,tmp_path):
    ds=EgoDexVideoDataset(published)
    checkpoint=tmp_path/'checkpoint'
    save_training_data_contract(str(checkpoint),ds)
    verify_resume_training_data_contract(str(checkpoint),ds)
    changed=SimpleNamespace(training_data_contract={**ds.training_data_contract,'index_sha256':'changed'})
    with pytest.raises(ValueError,match='differs'):
        verify_resume_training_data_contract(str(checkpoint),changed)
    with pytest.raises(ValueError,match='no longer'):
        verify_resume_training_data_contract(str(checkpoint),object())
