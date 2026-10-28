"""An external tactile catalog must bind original episodes without writing them."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_catalog(monkeypatch):
    pytest.importorskip('pyarrow')
    scripts=Path(__file__).resolve().parents[2]/'scripts/data/native'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('native_catalog_tested',scripts/'catalog_sharpa_tactile.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_bucket(tmp_path,monkeypatch):
    m=load_catalog(monkeypatch)
    import pyarrow as pa
    import pyarrow.parquet as pq
    source=tmp_path/'source';cache=tmp_path/'cache'
    bucket=source/'buckets/task_session';meta=bucket/'meta'
    (meta/'episodes/chunk-000').mkdir(parents=True)
    (source/'dataset_manifest.json').write_text('{}')
    (meta/'info.json').write_text(json.dumps({'fps':30,'total_episodes':2,'total_frames':5,'video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}))
    prefix=f'videos/{m.KEY}/'
    rows=[{'episode_index':idx,'length':n,prefix+'chunk_index':0,prefix+'file_index':0,prefix+'from_timestamp':offset} for idx,n,offset in [(7,2,1/30),(10,3,5/30)]]
    pq.write_table(pa.Table.from_pylist(rows),meta/'episodes/chunk-000/file-000.parquet')
    tactile=cache/'buckets/task_session/tactile';tactile.mkdir(parents=True)
    entries={}
    for idx,n in ((7,2),(10,3)):
        d=tactile/f'episode_{idx}';d.mkdir()
        np.save(d/'maps.npy',np.zeros((n,10,32,32),np.uint8))
        valid=np.ones((n,10),bool)
        if idx==10:valid[-1]=False
        np.save(d/'valid.npy',valid)
        np.save(d/'timestamps.npy',np.arange(n)/30)
        entries[str(idx)]={k:f'episode_{idx}/{k}.npy' for k in ('maps','valid','timestamps')}
        entries[str(idx)]['layout_present']=[True]*10
        entries[str(idx)].update(frames=n,source_video=f'videos/{m.KEY}/chunk-000/file-000.mp4',source_video_from_timestamp=(1 if idx==7 else 5)/30,source_episode_index=idx)
    manifest=tactile/'manifest.json'
    manifest.write_text(json.dumps({'schema':'tactile_regions_v1','codec':'tactile_intensity_v1','map_size':[32,32],'dataset_info_sha256':m.digest(meta/'info.json'),'region_names':m.REGIONS,'fps':30,'frames':5,'valid_frames':4,'episodes':entries}))
    return m,source,cache,bucket,manifest


def test_catalog_preserves_missing_masks_and_original_files(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest=fixture_bucket(tmp_path,monkeypatch)
    before={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert m.catalog(source,cache,{})['status']=='complete'
    pub=json.loads((cache/'publication.json').read_text())
    catalog=json.loads((cache/'tactile_catalog.json').read_text())
    assert pub['training_ready'] and pub['valid_frames']==4 and pub['frames']==5
    assert catalog['bucket_manifest_by_name']['task_session']['valid_region_frames']==40
    assert catalog['bucket_manifest_by_name']['task_session']['manifest']=='buckets/task_session/tactile/manifest.json'
    assert before=={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_catalog_binds_original_metadata(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest=fixture_bucket(tmp_path,monkeypatch)
    info=bucket/'meta/info.json';info.write_text(info.read_text()+' ')
    with pytest.raises(ValueError,match='identity differs'):m.catalog(source,cache,{})
    assert not json.loads((cache/'publication.json').read_text())['training_ready']


def test_catalog_rejects_escaped_array_paths(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest=fixture_bucket(tmp_path,monkeypatch)
    doc=json.loads(manifest.read_text());doc['episodes']['7']['maps']='../../../../outside.npy';manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='escapes'):m.catalog(source,cache,{})


def test_catalog_does_not_publish_partial_or_empty_dataset(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest=fixture_bucket(tmp_path,monkeypatch)
    manifest.unlink()
    assert m.catalog(source,cache,{})['status']=='pending_tactile'
    assert not json.loads((cache/'publication.json').read_text())['training_ready']
    empty=tmp_path/'empty';(empty/'buckets').mkdir(parents=True)
    with pytest.raises(ValueError,match='no buckets'):m.catalog(empty,tmp_path/'empty_cache',{})


def test_catalog_rejects_clock_mismatch(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest=fixture_bucket(tmp_path,monkeypatch)
    np.save(manifest.parent/'episode_10/timestamps.npy',np.arange(3)/60)
    with pytest.raises(ValueError,match='clock'):m.catalog(source,cache,{})


@pytest.mark.parametrize('damage',['missing','truncated','payload_write'])
def test_catalog_rechecks_arrays_even_with_cached_metadata(tmp_path,monkeypatch,damage):
    m,source,cache,bucket,manifest=fixture_bucket(tmp_path,monkeypatch);verified={}
    m.catalog(source,cache,verified)
    path=manifest.parent/'episode_7/maps.npy'
    if damage=='missing':path.unlink()
    elif damage=='truncated':
        with path.open('r+b') as f:f.truncate(path.stat().st_size-1)
    else:
        a=np.load(path,mmap_mode='r+');a[0,0,0,0]=253;a.flush();del a
    with pytest.raises((ValueError,FileNotFoundError)):m.catalog(source,cache,verified)
    pub=json.loads((cache/'publication.json').read_text())
    assert pub['status']=='validation_failed' and not pub['training_ready']


def test_catalog_rejects_appended_array_and_wrong_masks(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest=fixture_bucket(tmp_path,monkeypatch)
    path=manifest.parent/'episode_7/maps.npy'
    with path.open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='length/header'):m.validate_bucket(bucket,manifest)
    with path.open('r+b') as f:f.truncate(path.stat().st_size-len(b'corruption'))
    np.save(manifest.parent/'episode_7/valid.npy',[[True]+[False]*9]*2)
    with pytest.raises(ValueError,match='measurement mask'):m.validate_bucket(bucket,manifest)


def test_catalog_checks_source_video_offsets_and_parquet_changes(tmp_path,monkeypatch):
    m,source,cache,bucket,manifest=fixture_bucket(tmp_path,monkeypatch);verified={}
    m.catalog(source,cache,verified)
    import pyarrow as pa
    import pyarrow.parquet as pq
    path=bucket/'meta/episodes/chunk-000/file-000.parquet';rows=pq.read_table(path).to_pylist()
    rows[0][f'videos/{m.KEY}/from_timestamp']=4/30;pq.write_table(pa.Table.from_pylist(rows),path)
    with pytest.raises(ValueError,match='metadata differs'):m.catalog(source,cache,verified)
    assert not json.loads((cache/'publication.json').read_text())['training_ready']


def test_pending_workers_skip_locked_or_published_tasks(tmp_path,monkeypatch):
    load_catalog(monkeypatch)
    import fcntl
    from run_native_shards import job
    (tmp_path/'conversion_locks').mkdir()
    with (tmp_path/'conversion_locks/task__session.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert job('sharpa',None,tmp_path,'task/session',True)['status']=='owned_by_another_worker'
    manifest=tmp_path/'partitions/task/session/tactile/manifest.json'
    manifest.parent.mkdir(parents=True);manifest.write_text('{}')
    assert job('sharpa',None,tmp_path,'task/session',True)['status']=='already_published'
    assert manifest.read_text()=='{}'
