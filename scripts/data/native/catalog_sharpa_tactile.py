#!/usr/bin/env python3
"""Bind an external tactile cache to existing Sharpa LeRobot buckets.

The existing dataset is read-only. Only the external cache catalog/publication
are written. All original video/action files and normalization assets are reused.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from extract_sharpa_deform import REGIONS,KEY


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(v,indent=2)+'\n'
    if p.exists() and p.read_text()==text:return
    tmp=p.with_name(p.name+f'.partial.{os.getpid()}');tmp.write_text(text);os.replace(tmp,p)

def source_episode_files(bucket):
    return sorted(bucket.glob('meta/episodes/*/*.parquet'))

def source_episode_signature(bucket):
    return hashlib.sha256(json.dumps([(str(p.relative_to(bucket)),digest(p)) for p in source_episode_files(bucket)]).encode()).hexdigest()

def array_inventory(manifest,document=None):
    """Cheap immutable-file identity, checked even when metadata is cached.

    This detects normal writes, replacement, deletion and truncation after a
    receipt. It is not a cryptographic checksum of every array payload; raw
    uint8 corruption present before the first receipt needs source comparison.
    """
    document=read(manifest) if document is None else document
    root=manifest.parent.resolve();records=[];seen=set()
    for episode,e in sorted(document['episodes'].items()):
        for key in ('maps','valid','timestamps'):
            path=(root/e[key]).resolve()
            if not path.is_relative_to(root):raise ValueError('Sidecar array escapes manifest directory')
            if path in seen:raise ValueError('Multiple episodes or fields alias one sidecar array')
            seen.add(path);s=path.stat()
            if not path.is_file():raise ValueError('Sidecar array is not a regular file')
            records.append((episode,key,str(path.relative_to(root)),s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
    return hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()

def validate_bucket(bucket,manifest):
    info_path=bucket/'meta/info.json';info=read(info_path);m=read(manifest)
    if m['schema']!='tactile_regions_v1' or m['dataset_info_sha256']!=digest(info_path) or m['codec']!='tactile_intensity_v1' or m['map_size']!=[32,32] or m['region_names']!=REGIONS or m['fps']!=info['fps'] or info['fps']<=0:
        raise ValueError(f'Sidecar identity differs from original bucket: {bucket}')
    lengths={};sources={};prefix=f'videos/{KEY}/'
    for p in source_episode_files(bucket):
        for e in pq.read_table(p).to_pylist():
            key=str(e['episode_index']);n=int(e['length'])
            if key in lengths or n<=0:raise ValueError(f'Duplicate or empty original episode: {bucket}:{key}')
            lengths[key]=n
            sources[key]={'video':info['video_path'].format(video_key=KEY,chunk_index=e[prefix+'chunk_index'],file_index=e[prefix+'file_index']),
                          'offset':float(e[prefix+'from_timestamp']),'source_episode_index':int(e.get('source_episode_index',e['episode_index']))}
            if not np.isfinite(sources[key]['offset']) or sources[key]['offset']<0:raise ValueError('Invalid original video offset')
    if set(lengths)!=set(m['episodes']) or len(lengths)!=info['total_episodes']:raise ValueError(f'Original episode set differs: {bucket}')
    frames=0;valid_frames=0;valid_cells=0
    for episode,e in m['episodes'].items():
        paths=[(manifest.parent/e[key]).resolve() for key in ('maps','valid','timestamps')]
        if any(not p.is_relative_to(manifest.parent.resolve()) for p in paths):raise ValueError('Sidecar array escapes manifest directory')
        maps,valid,ts=[np.load(p,mmap_mode='r',allow_pickle=False) for p in paths]
        n=lengths[episode]
        source=sources[episode]
        sidecar_offset=float(e['source_video_from_timestamp'])
        if e['frames']!=n or e['source_video']!=source['video'] or not np.isfinite(sidecar_offset) or abs(sidecar_offset-source['offset'])>1e-8 or e['source_episode_index']!=source['source_episode_index']:
            raise ValueError(f'Sidecar video/episode metadata differs from original: {bucket}:{episode}')
        if maps.dtype!=np.uint8 or maps.shape!=(n,10,32,32) or valid.shape!=(n,10) or valid.dtype!=np.bool_ or ts.shape!=(n,) or ts.dtype!=np.float64:raise ValueError(f'Invalid sidecar shape/dtype: {bucket}:{episode}')
        for p,a in zip(paths,(maps,valid,ts)):
            if p.stat().st_size!=a.offset+a.nbytes:raise ValueError(f'Sidecar NPY length/header differs: {p}')
        if not np.allclose(ts,np.arange(n)/info['fps'],atol=2e-5,rtol=0):raise ValueError(f'Invalid sidecar clock: {bucket}:{episode}')
        present=np.asarray(e['layout_present'])
        if present.dtype!=np.bool_ or present.shape!=(10,) or not present.all() or (valid.any(axis=1)!=valid.all(axis=1)).any():raise ValueError('Invalid Sharpa mosaic layout/measurement mask')
        frames+=n;valid_frames+=int(valid.all(axis=1).sum());valid_cells+=int(valid.sum())
        del maps,valid,ts
    if frames!=info['total_frames'] or frames!=m['frames'] or valid_frames!=m['valid_frames']:raise ValueError(f'Sidecar frame totals differ: {bucket}')
    return {'source_bucket':str(bucket),'source_info_sha256':digest(info_path),'source_episode_metadata_sha256':source_episode_signature(bucket),
            'array_inventory_sha256':array_inventory(manifest,m),'array_integrity_scope':'NPY header/shape/length, all timestamps/masks, immutable file stat identity; not full payload hashes',
            'episodes':len(lengths),'frames':frames,'valid_frames':valid_frames,'valid_region_frames':valid_cells,
            'manifest_sha256':digest(manifest),'decode_errors':m.get('decode_errors',[])}

def _catalog(source,cache,verified):
    buckets=sorted(p for p in (source/'buckets').iterdir() if p.is_dir());entries={};pending=[];total_frames=0;total_episodes=0
    if not buckets:raise ValueError(f'Original Sharpa dataset has no buckets: {source}')
    previous=read(cache/'tactile_catalog.json').get('bucket_manifest_by_name',{}) if (cache/'tactile_catalog.json').exists() else {}
    for bucket in buckets:
        info_path=bucket/'meta/info.json';info=read(info_path);total_frames+=info['total_frames'];total_episodes+=info['total_episodes']
        manifest=cache/'buckets'/bucket.name/'tactile/manifest.json'
        if not manifest.exists():pending.append(bucket.name);continue
        metadata_identity=(digest(manifest),digest(info_path),source_episode_signature(bucket))
        inventory=array_inventory(manifest)
        prior=previous.get(bucket.name,{})
        if prior.get('manifest_sha256')==metadata_identity[0] and prior.get('array_inventory_sha256') not in (None,inventory):
            raise ValueError(f'Published immutable sidecar arrays changed after receipt: {bucket.name}')
        identity=(*metadata_identity,inventory)
        if bucket.name not in verified or verified[bucket.name][0]!=identity:verified[bucket.name]=(identity,validate_bucket(bucket,manifest))
        entries[bucket.name]={**verified[bucket.name][1],'manifest':str(manifest.relative_to(cache))}
    complete=len(entries)==len(buckets)
    result={'schema':'touchwam_sharpa_external_tactile_catalog_v1','status':'complete' if complete else 'pending_tactile',
            'source_dataset':str(source),'source_manifest_sha256':digest(source/'dataset_manifest.json'),'tactile_root':str(cache),
            'source_policy':'Reuse existing native video/action/state/normalization without rewriting old data; only tactile cache is new',
            'buckets_expected':len(buckets),'buckets_completed':len(entries),'episodes':total_episodes,'frames':total_frames,
            'completed_frames':sum(e['frames'] for e in entries.values()),'valid_frames':sum(e['valid_frames'] for e in entries.values()),
            'validation':'all episodes, NPY header/length/shape/dtype, original metadata and video offsets, physical timestamps/masks, immutable array inventory; source pixel comparison is separate',
            'bucket_manifest_by_name':entries,'pending':pending}
    write(cache/'tactile_catalog.json',result)
    pub_path=cache/'publication.json';pub=read(pub_path) if pub_path.exists() else {}
    pub.update({k:v for k,v in result.items() if k not in ('bucket_manifest_by_name','pending')})
    pub['tactile_catalog']='tactile_catalog.json';pub['tactile_catalog_sha256']=digest(cache/'tactile_catalog.json')
    pub['training_ready']=complete;pub['raw_frames']=total_frames;pub['reuse_existing_native_dataset']=True
    pub.pop('validation_error',None)
    write(pub_path,pub)
    return {k:result[k] for k in ('status','buckets_expected','buckets_completed','episodes','frames','completed_frames','valid_frames')}

def catalog(source,cache,verified):
    try:return _catalog(source,cache,verified)
    except Exception as exc:
        # An old successful publication must not survive a failed recheck.
        pub_path=cache/'publication.json';pub=read(pub_path) if pub_path.exists() else {}
        pub.update(status='validation_failed',training_ready=False,validation_error=repr(exc))
        write(pub_path,pub)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--cache',required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();verified={}
    while True:
        result=catalog(Path(a.source),Path(a.cache),verified);print(json.dumps(result),flush=True)
        if result['status']=='complete':break
        if not a.wait:raise SystemExit(2)
        time.sleep(60)
