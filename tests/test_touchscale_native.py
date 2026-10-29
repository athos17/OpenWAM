import json
from pathlib import Path
import hashlib

import av
import numpy as np
import pytest
import torch

from openwam.tactile.touchscale_store import TouchScaleForceStore
from scripts.data.touchscale.convert import clock_selection, layout, video


def fixture(root):
    (root/'meta').mkdir(parents=True);(root/'meta/info.json').write_text('{}')
    data=root/'tactile';data.mkdir()
    entry={'duration_s':1.}
    for side,shift in [('right',0),('left',.007)]:
        times=np.arange(51,dtype=np.float64)*.02+shift
        force=np.broadcast_to((np.arange(51)*.1).astype(np.float32)[:,None],(51,880)).copy()
        force[:,0]=-1
        np.save(data/f'{side}.npy',force);np.save(data/f'{side}_t.npy',times)
        entry[side]=dict(force=f'{side}.npy',timestamps=f'{side}_t.npy')
    manifest=dict(schema='touchscale_native_force_v1',codec='tactile_intensity_v1',
          dataset_info_sha256=hashlib.sha256(b'{}').hexdigest(),encoder_name='TouchScaleForce',layout=layout(),
          normalization='max(force_N,0)/(max(force_N,0)+1N)',episodes={'0':entry})
    p=data/'manifest.json';p.write_text(json.dumps(manifest))
    return TouchScaleForceStore(p,root,observation_size=[16,10],encoder_name='TouchScaleForce')


def test_native_taxels_independent_clocks_and_causal_normalization(tmp_path):
    store=fixture(tmp_path);sample=store.window(0,.2)
    maps=sample['tactile_observation_maps']
    assert maps.shape==(4,30,1,16,10)
    assert sample['tactile_observation_spatial_valid'].sum()==1760
    # Right .20s maps to 1N, left .187s maps to .9N; no future-left .207s.
    assert maps[-1,0,0,0,1]==pytest.approx(.5)
    assert maps[-1,15,0,0,1]==pytest.approx(.9/1.9)
    assert maps[-1,0,0,0,0]==0  # negative offset is not positive compression
    assert np.load(tmp_path/'tactile/right.npy')[10,0]==-1  # raw signal preserved
    t=sample['tactile_history_source_times_s'];v=sample['tactile_valid']
    assert (t[v] <= sample['tactile_history_times_s'][:,None].expand_as(t)[v]+1e-7).all()
    assert sample['tactile_valid'].all()
    assert not sample['future_tactile_valid'][-1].any()


def test_boundaries_missing_hand_packet_and_unmeasured_padding(tmp_path):
    store=fixture(tmp_path)
    first=store.window(0,0)
    assert not first['tactile_valid'][:-1].any()
    assert first['tactile_valid'][-1,:15].all() and not first['tactile_valid'][-1,15:].any()
    end=store.window(0,29/30)
    assert not end['future_tactile_valid'].any()
    assert end['tactile_query_mask'].all()
    values=first['tactile_observation_maps']
    roi=first['tactile_observation_spatial_valid'][None].expand_as(values)
    assert not values[~roi].any()
    force=np.load(tmp_path/'tactile/right.npy');force[10,160:192]=np.nan;np.save(tmp_path/'tactile/right.npy',force)
    store.arrays.cache_clear();sample=store.window(0,.2)
    assert not sample['tactile_valid'][-1,1] and sample['tactile_valid'][-1,0]


def test_stale_measurements_and_episode_boundaries(tmp_path):
    store=fixture(tmp_path)
    t=np.load(tmp_path/'tactile/right_t.npy');t[5:]=t[5:]+.3;np.save(tmp_path/'tactile/right_t.npy',t)
    assert not store.window(0,.2)['tactile_valid'][-1,:15].any()
    with pytest.raises(ValueError,match='outside episode'):store.window(0,-.1)


def test_real_video_causal_resampling_is_lossless(tmp_path):
    source=tmp_path/'src.mp4'
    with av.open(str(source),'w') as out:
        stream=out.add_stream('libx264',rate=30);stream.width=32;stream.height=24;stream.pix_fmt='yuv420p'
        stream.options={'crf':'0','preset':'ultrafast'}
        for i in range(12):
            f=av.VideoFrame.from_ndarray(np.full((24,32,3),i*19,np.uint8),format='rgb24')
            for p in stream.encode(f):out.mux(p)
        for p in stream.encode():out.mux(p)
    with av.open(str(source)) as inp:frames=[f.to_ndarray(format='yuv420p') for f in inp.decode(video=0)]
    indices=np.array([1,2,2,4,5,6,8])
    dest=tmp_path/'aligned.mp4';receipt=video(source,dest,indices,12,1)
    assert receipt['storage']=='lossless_causal_resample'
    with av.open(str(dest)) as inp:actual=[f.to_ndarray(format='yuv420p') for f in inp.decode(video=0)]
    for a,i in zip(actual,indices):np.testing.assert_array_equal(a,frames[i])
    copy=tmp_path/'copied.mp4';receipt=video(source,copy,np.arange(2,9),12,1)
    assert receipt['from_timestamp']==2/30 and copy.read_bytes()==source.read_bytes()


def test_clock_selection_masks_missing_without_borrowing_future():
    times=np.array([.01,.04,.07,.2]);grid=np.arange(8)/30
    idx,valid=clock_selection(times,grid,.05)
    assert not valid[0] and not valid[4]
    assert (times[idx[valid]]<=grid[valid]+1e-8).all()
    with pytest.raises(ValueError):clock_selection(np.array([0,0]),grid)


def test_preflight_checks_each_glove_clock_and_model_configuration(tmp_path):
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from scripts.preflight_touchwam import validate_tactile_sample,validate_source_contract
    from openwam.model.architectures.touchwam.heterogeneous import normalize_encoder_spec,HeterogeneousObservationEncoder
    with initialize_config_dir(config_dir=str(Path('configs').resolve()),version_base=None):
        cfg=compose(config_name='train_touchwam_ftp1')
    spec=OmegaConf.to_container(cfg.model.architecture.tactile,resolve=True)
    source=OmegaConf.to_container(cfg.dataloader.datasets.touchscale,resolve=True)
    assert validate_source_contract('touchscale',source)['source_kind']=='ego'
    sample=fixture(tmp_path).window(0,.2)
    assert validate_tactile_sample(sample,spec)[0]['observation_shape']==[1,4,30,1,16,10]
    sample['tactile_history_source_times_s'][-1,0]=.001
    with pytest.raises(ValueError,match='borrow future'):validate_tactile_sample(sample,spec)
    normalized=normalize_encoder_spec(spec['observation_encoder'],spec['sensor_names'])
    normalized.update(initialization='random',sensors=[normalized['sensors'][-1]],
                      source_to_encoder={'touchscale_glove':['TouchScaleForce']},matrix_width=8)
    model=HeterogeneousObservationEncoder(normalized,16,['touchscale_glove'])
    output,mask,_=model(torch.rand(1,4,30,1,16,10),torch.ones(1,4,30,dtype=torch.bool),
          torch.as_tensor(fixture(tmp_path/'other').spatial)[None],torch.zeros(1,30,dtype=torch.long),torch.zeros(1,30,dtype=torch.long))
    assert output.shape==(1,480,16) and torch.isfinite(output).all()
    output.sum().backward();assert model.matrix_encoders['TouchScaleForce'].projection.weight.grad is not None
