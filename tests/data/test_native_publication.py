"""Native release invariants: identity, offsets, missing labels and mosaic layout."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/data/native'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def test_immutable_publication_rewrites_internal_links(tmp_path):
    publish=module('publish_native').publish
    src=tmp_path/'source';(src/'partitions/a/meta').mkdir(parents=True);(src/'buckets').mkdir()
    (src/'dataset_manifest.json').write_text('{}');(src/'normalization.npy').write_bytes(b'normalization')
    (src/'partitions/a/meta/info.json').write_text('{"codebase_version":"v3.0"}')
    (src/'partitions/a/meta/stats.npy').symlink_to(src/'normalization.npy')
    (src/'buckets/a').symlink_to(src/'partitions/a',target_is_directory=True)
    dst=tmp_path/'output';report=publish(src,dst)
    assert (dst/'buckets/a').resolve()==dst/'partitions/a'
    assert (dst/'partitions/a/meta/stats.npy').resolve()==dst/'normalization.npy'
    assert (dst/'normalization.npy').samefile(src/'normalization.npy')
    assert report['hardlinked_files']==3
    assert publish(src,dst)['referenced_bytes']==report['referenced_bytes']


def test_publication_rejects_existing_changed_bytes(tmp_path):
    src=tmp_path/'src';src.mkdir();(src/'dataset_manifest.json').write_text('{}')
    dst=tmp_path/'dst';dst.mkdir();(dst/'dataset_manifest.json').write_text('bad')
    with pytest.raises(ValueError,match='not source hardlink'):module('publish_native').publish(src,dst)


def test_mosaic_order_is_not_hand_mirrored():
    pytest.importorskip('cv2');pytest.importorskip('pyarrow')
    m=module('extract_sharpa_deform')
    a=np.zeros((480,1200),np.uint8)
    for i in range(10):a[(i//5)*240:(i//5+1)*240,(i%5)*240:(i%5+1)*240]=i*20
    out=m.split_mosaic(a)
    assert out.shape==(10,32,32)
    assert np.all(out[:,0,0]==np.arange(10)*20)
    assert m.REGIONS[0]=='left.thumb.tip' and m.REGIONS[9]=='right.pinky.tip'
    with pytest.raises(ValueError):m.split_mosaic(a[:10])


def write_video(path,frames):
    av=pytest.importorskip('av');path.parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(path),'w') as c:
        s=c.add_stream('libx264',rate=30);s.width=frames[0].shape[1];s.height=frames[0].shape[0];s.pix_fmt='yuv420p'
        for a in frames:
            frame=av.VideoFrame.from_ndarray(np.repeat(a[...,None],3,axis=2),format='rgb24')
            for packet in s.encode(frame):c.mux(packet)
        for packet in s.encode():c.mux(packet)


def test_egodex_v3_preserves_video_offsets_without_action(tmp_path):
    pa=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet')
    m=module('convert_egodex_video');src=tmp_path/'source';src.mkdir()
    write_video(src/'original.mp4',[np.full((32,32),v,np.uint8) for v in [0,20,40,60,80,100]])
    rows=[dict(id='train/a/episode_1',task='a',source_split='train',length=2,prompt='a',video_path='original.mp4',video_from_frame=1,video_to_frame=3,source_dataset='train/a',storage_origin='cache'),
          dict(id='train/a/episode_2',task='a',source_split='train',length=2,prompt='b',video_path='original.mp4',video_from_frame=4,video_to_frame=6,source_dataset='train/a',storage_origin='cache')]
    pq.write_table(pa.Table.from_pylist(rows),src/'episodes.parquet');(src/'dataset_info.json').write_text(json.dumps({'total_frames':4,'total_episodes':2}))
    dst=tmp_path/'out';report=m.convert(src,dst);info=json.loads((dst/'meta/info.json').read_text())
    assert report['episodes']==2 and report['frames']==4
    assert 'action' not in info['features'] and 'observation.state' not in info['features']
    eps=pq.read_table(dst/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    assert eps[0]['videos/observation.images.camera/from_timestamp']==1/30
    assert eps[1]['videos/observation.images.camera/from_timestamp']==4/30
    table=pq.read_table(dst/'data/chunk-000/file-000.parquet').to_pydict()
    assert table['episode_index']==[0,0,1,1] and table['frame_index']==[0,1,0,1]
    assert m.convert(src,dst)['signature']==report['signature']


def test_sharpa_extraction_preserves_crop_and_missing(tmp_path):
    pa=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet');pytest.importorskip('cv2')
    m=module('extract_sharpa_deform');p=tmp_path/'partition';(p/'meta/episodes/chunk-000').mkdir(parents=True)
    video='videos/observation.images.tactile_deform/chunk-000/file-000.mp4'
    write_video(p/video,[np.full((480,1200),v,np.uint8) for v in [0,30,60,90,120]])
    info={'fps':30,'features':{m.KEY:{'shape':[480,1200,3]}},'video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    (p/'meta/info.json').write_text(json.dumps(info))
    eps=[{'episode_index':0,'length':3,f'videos/{m.KEY}/from_timestamp':2/30,f'videos/{m.KEY}/chunk_index':0,f'videos/{m.KEY}/file_index':0},
         {'episode_index':1,'length':2,f'videos/{m.KEY}/from_timestamp':5/30,f'videos/{m.KEY}/chunk_index':0,f'videos/{m.KEY}/file_index':0}]
    pq.write_table(pa.Table.from_pylist(eps),p/'meta/episodes/chunk-000/file-000.parquet')
    result=m.extract_partition(p);assert result['frames']==5 and result['valid_frames']==3
    maps=np.load(p/'tactile/episode_000000/maps.npy');assert np.allclose(maps.mean((1,2,3)),[60,90,120],atol=2)
    assert np.load(p/'tactile/episode_000000/valid.npy').all()
    assert not np.load(p/'tactile/episode_000001/valid.npy').any()
    assert np.allclose(np.load(p/'tactile/episode_000000/timestamps.npy'),[0,1/30,2/30])


def test_staging_recovers_rename_before_receipt(tmp_path):
    pytest.importorskip('pyarrow');m=module('stage_egodex_pool');src=tmp_path/'src';dst=tmp_path/'dst';src.write_bytes(b'source');dst.write_bytes(b'source')
    record=m.copy_verified(src,dst,None);assert record['bytes']==6 and record['sha256']==m.hash_file(src)
    dst.write_bytes(b'differ')
    with pytest.raises(ValueError,match='differs from source'):m.copy_verified(src,dst,None)
