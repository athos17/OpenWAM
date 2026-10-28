"""Real encoded mosaics exercise the independent all-partition audit."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


def fixture(tmp_path,monkeypatch):
    av=pytest.importorskip('av');pa=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet');pytest.importorskip('cv2')
    scripts=Path(__file__).resolve().parents[2]/'scripts/data/native';monkeypatch.syspath_prepend(str(scripts))
    import validate_sharpa_all as audit
    import extract_sharpa_deform as extractor
    source=tmp_path/'source';cache=tmp_path/'cache';bucket=source/'buckets/task';meta=bucket/'meta'
    (meta/'episodes/chunk-000').mkdir(parents=True);(source/'dataset_manifest.json').write_text('{}')
    key=audit.KEY;video=f'videos/{key}/chunk-000/file-000.mp4';path=bucket/video;path.parent.mkdir(parents=True)
    with av.open(str(path),'w') as c:
        stream=c.add_stream('libx264',rate=30);stream.width=1200;stream.height=480;stream.pix_fmt='yuv420p'
        for frame_index in range(8):
            image=np.zeros((480,1200),np.uint8)
            for region in range(10):
                y,x=divmod(region,5);tile=np.full((240,240),region*20+frame_index*2,np.uint8)
                tile[20:60,40:100]+=15;image[y*240:(y+1)*240,x*240:(x+1)*240]=tile
            frame=av.VideoFrame.from_ndarray(np.repeat(image[...,None],3,axis=2),format='rgb24')
            for packet in stream.encode(frame):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    info={'fps':30,'total_episodes':3,'total_frames':6,'features':{key:{'shape':[480,1200,3]}},'video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    (meta/'info.json').write_text(json.dumps(info))
    rows=[{'episode_index':idx,'length':2,f'videos/{key}/chunk_index':0,f'videos/{key}/file_index':0,f'videos/{key}/from_timestamp':start/30} for idx,start in ((7,2),(10,5),(11,8))]
    pq.write_table(pa.Table.from_pylist(rows),meta/'episodes/chunk-000/file-000.parquet')
    new=cache/'buckets/task';shutil.copytree(bucket,new);extractor.extract_partition(new)
    return audit,source,cache,bucket,new/'tactile/manifest.json',tmp_path/'audit'


def test_readback_uses_original_offsets_all_surfaces_and_missing_masks(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest,out=fixture(tmp_path,monkeypatch)
    before={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    receipt=out/'receipts/task.json';result=m.audit_bucket(bucket,manifest,receipt)
    assert result['status']=='passed',result
    assert result['sample_count']==6 and result['matched_sample_count']==4 and result['masked_source_unavailable_count']==2
    assert result['selected_episode_indices']==[7,10,11]
    sample=result['samples'][0]
    assert sample['source_video_from_timestamp']==2/30 and sample['decoded_pts']==2/30
    assert sample['region_max_uint8_errors']==[0]*10 and len(sample['region_mean_uint8'])==10
    assert before=={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert m.audit_bucket(bucket,manifest,receipt)['reused']


def test_readback_detects_payload_corruption_and_stale_receipt(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest,out=fixture(tmp_path,monkeypatch);receipt=out/'receipts/task.json'
    assert m.audit_bucket(bucket,manifest,receipt)['status']=='passed'
    maps=np.load(manifest.parent/'episode_000007/maps.npy',mmap_mode='r+');maps[0,4,3,6]^=255;maps.flush();del maps
    aggregate=m.aggregate(source,cache,out)
    assert aggregate['status']=='incomplete' and len(aggregate['stale'])==1
    result=m.audit_bucket(bucket,manifest,receipt)
    assert result['status']=='failed' and 'Source pixel mismatch 7:0' in result['error']


def test_aggregate_requires_every_original_partition(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest,out=fixture(tmp_path,monkeypatch)
    m.audit_bucket(bucket,manifest,out/'receipts/task.json')
    assert m.aggregate(source,cache,out)['status']=='passed'
    (source/'buckets/missing').mkdir()
    result=m.aggregate(source,cache,out)
    assert result['status']=='incomplete' and result['buckets_expected']==2 and result['pending']==['missing']


def test_readback_rejects_mismatched_video_metadata(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest,out=fixture(tmp_path,monkeypatch)
    doc=json.loads(manifest.read_text());doc['episodes']['7']['source_video_from_timestamp']=0;manifest.write_text(json.dumps(doc))
    result=m.audit_bucket(bucket,manifest,out/'receipts/task.json')
    assert result['status']=='failed' and 'metadata differs' in result['error']
