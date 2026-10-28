#!/usr/bin/env python3
"""Independent source-video readback for every original Sharpa partition.

Only --output is written. Original data and tactile sidecars are read-only.
Each partition selects first/middle/last episodes and first/middle/last rows
inside them. Every selected row compares all ten 32x32 maps. Source video paths
and offsets come from original LeRobot metadata, not sidecar assertions.

Fixed shard indices partition the sorted ORIGINAL bucket inventory; do not
repartition only pending tasks. No distributed flock or work stealing is used.
Full array payloads are not hashed: structural checks cover every episode,
while source pixel comparison is sampled and explicitly reported as such.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from catalog_sharpa_tactile import array_inventory,source_episode_signature,validate_bucket


KEY='observation.images.tactile_deform'
REGIONS=['left.thumb.tip','left.index.tip','left.middle.tip','left.ring.tip','left.pinky.tip',
         'right.thumb.tip','right.index.tip','right.middle.tip','right.ring.tip','right.pinky.tip']
VERSION='sharpa_source_readback_v1'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.partial.{os.getpid()}')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def oracle_tiles(frame):
    """Independent per-tile crop oracle; never calls the frozen converter.

    The converter resizes the entire mosaic and reshapes it; this oracle crops
    each original 240x240 surface first and resizes the ten crops separately.
    Both implement the same declared video-range convention.
    """
    import cv2
    if (frame.height,frame.width)!=(480,1200):raise ValueError('Source tactile video is not 480x1200')
    luma=frame.format.name in ('yuv420p','yuvj420p')
    if luma:
        gray=np.frombuffer(frame.planes[0],np.uint8).reshape(frame.height,frame.planes[0].line_size)[:,:frame.width]
        limited=frame.color_range!=2 and frame.format.name!='yuvj420p'
    else:gray=frame.to_ndarray(format='gray');limited=False
    regions=[]
    for hand_row in (0,1):
        for finger_col in range(5):
            crop=gray[hand_row*240:(hand_row+1)*240,finger_col*240:(finger_col+1)*240]
            tile=cv2.resize(crop,(32,32),interpolation=cv2.INTER_AREA)
            if limited:tile=np.rint(np.clip((tile.astype(np.float32)-16)*255/219,0,255)).astype(np.uint8)
            regions.append(tile)
    return np.stack(regions)


def source_episodes(bucket):
    rows=[]
    for path in sorted(bucket.glob('meta/episodes/*/*.parquet')):rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows,key=lambda e:int(e['episode_index']))


def source_video(info,episode):
    prefix=f'videos/{KEY}/'
    return info['video_path'].format(video_key=KEY,chunk_index=episode[prefix+'chunk_index'],file_index=episode[prefix+'file_index'])


def source_identity(bucket,manifest):
    info=read(bucket/'meta/info.json');episodes=source_episodes(bucket);videos=[]
    for name in sorted({source_video(info,e) for e in episodes}):
        path=bucket/name;s=path.stat()
        videos.append([name,s.st_size,s.st_mtime_ns,s.st_ctime_ns])
    identity={'version':VERSION,'validator_sha256':digest(__file__),
              'structural_validator_sha256':digest(Path(__file__).with_name('catalog_sharpa_tactile.py')),
              'source_info_sha256':digest(bucket/'meta/info.json'),'source_episode_metadata_sha256':source_episode_signature(bucket),
              'manifest_sha256':digest(manifest),'array_inventory_sha256':array_inventory(manifest),
              'source_video_inventory_sha256':hashlib.sha256(json.dumps(videos).encode()).hexdigest()}
    identity['signature']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return identity,info,episodes


def decode_target(container,stream,target,fps):
    container.seek(int(max(0,target-1/fps)/stream.time_base),stream=stream,backward=True,any_frame=False)
    decoded=0
    for frame in container.decode(stream):
        decoded+=1
        if frame.pts is None:raise ValueError('Source video frame lacks physical PTS')
        pts=float(frame.pts*stream.time_base)
        if abs(pts-target)<=2e-4:return frame,pts,decoded
        if pts>target+2e-4:break
    return None,None,decoded


def audit_bucket(bucket,manifest,receipt,reuse=True):
    import av,cv2
    cv2.setNumThreads(1)
    bucket=Path(bucket);manifest=Path(manifest);receipt=Path(receipt);started=time.monotonic()
    if not manifest.exists():
        result={'bucket':bucket.name,'status':'pending_manifest','version':VERSION}
        atomic_json(receipt,result);return result
    try:
        identity,info,episodes=source_identity(bucket,manifest)
        if reuse and receipt.exists():
            old=read(receipt)
            if old.get('status')=='passed' and old.get('signature')==identity['signature']:
                return {**old,'reused':True}
        structural=validate_bucket(bucket,manifest);document=read(manifest)
        if document['region_names']!=REGIONS:raise ValueError('Ten-finger identity/order mismatch')
        chosen=sorted({0,len(episodes)//2,len(episodes)-1});samples=[];containers={};decode_frames=0
        try:
            for index in chosen:
                ep=episodes[index];epid=str(ep['episode_index']);n=int(ep['length']);entry=document['episodes'][epid]
                maps=np.load(manifest.parent/entry['maps'],mmap_mode='r',allow_pickle=False)
                valid=np.load(manifest.parent/entry['valid'],mmap_mode='r',allow_pickle=False)
                times=np.load(manifest.parent/entry['timestamps'],mmap_mode='r',allow_pickle=False)
                video=source_video(info,ep);offset=float(ep[f'videos/{KEY}/from_timestamp'])
                if video not in containers:
                    c=av.open(str(bucket/video));s=c.streams.video[0];s.thread_count=1;containers[video]=(c,s)
                c,s=containers[video]
                for row in sorted({0,n//2,n-1}):
                    target=offset+row/info['fps'];frame,pts,decoded=decode_target(c,s,target,info['fps']);decode_frames+=decoded
                    record={'episode_index':int(epid),'episode_length':n,'row':row,'source_video':video,
                            'source_video_from_timestamp':offset,'target_video_timestamp':target,'decoded_pts':pts,
                            'sidecar_timestamp':float(times[row]),'valid_regions':valid[row].tolist()}
                    if abs(float(times[row])-row/info['fps'])>2e-5:raise ValueError('Sample sidecar episode time differs')
                    if frame is None:
                        if valid[row].any():raise ValueError(f'Source PTS unavailable but sample is valid: {epid}:{row}')
                        record['sample_status']='masked_source_unavailable'
                    else:
                        if not valid[row].all():raise ValueError(f'Decodable source PTS marked invalid: {epid}:{row}')
                        expected=oracle_tiles(frame);actual=np.asarray(maps[row]);diff=np.abs(expected.astype(np.int16)-actual.astype(np.int16))
                        errors=diff.max(axis=(1,2)).tolist()
                        if any(errors):raise ValueError(f'Source pixel mismatch {epid}:{row}; per-region max uint8 error={errors}')
                        record.update(sample_status='matched',pixel_format=frame.format.name,color_range=int(frame.color_range),
                                      region_max_uint8_errors=errors,region_mean_uint8=actual.mean(axis=(1,2)).tolist(),
                                      oracle_map_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),sidecar_map_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
                    samples.append(record)
        finally:
            for c,s in containers.values():c.close()
        final_identity,_,_=source_identity(bucket,manifest)
        if final_identity['signature']!=identity['signature']:raise ValueError('Source or sidecar changed during readback')
        result={'bucket':bucket.name,'status':'passed',**identity,'structural':structural,'region_names':REGIONS,
                'selected_episode_indices':[int(episodes[i]['episode_index']) for i in chosen],
                'samples':samples,'sample_count':len(samples),'matched_sample_count':sum(x['sample_status']=='matched' for x in samples),
                'masked_source_unavailable_count':sum(x['sample_status']=='masked_source_unavailable' for x in samples),
                'decoded_frames_including_seek_preroll':decode_frames,'elapsed_seconds':time.monotonic()-started,
                'scope':'All episode metadata/array headers/timestamps/masks; source pixels sampled at first/middle/last episodes and rows, all ten surfaces. No full payload checksum or full-frame replay.'}
    except Exception as exc:
        result={'bucket':bucket.name,'status':'failed','version':VERSION,'error':repr(exc),'elapsed_seconds':time.monotonic()-started}
    atomic_json(receipt,result);return result


def aggregate(source,cache,output):
    source=Path(source);cache=Path(cache);output=Path(output)
    buckets=sorted(p for p in (source/'buckets').iterdir() if p.is_dir())
    if not buckets:raise ValueError('No original Sharpa buckets')
    passed=[];pending=[];failed=[];stale=[];receipts=[]
    for bucket in buckets:
        path=output/'receipts'/f'{bucket.name}.json'
        if not path.exists():pending.append(bucket.name);continue
        result=read(path)
        if result.get('status')=='pending_manifest':pending.append(bucket.name);continue
        if result.get('status')!='passed':failed.append({'bucket':bucket.name,'error':result.get('error')});continue
        try:
            identity,_,_=source_identity(bucket,cache/'buckets'/bucket.name/'tactile/manifest.json')
            if result.get('signature')!=identity['signature']:raise ValueError('Receipt source/sidecar/code identity changed')
        except Exception as exc:stale.append({'bucket':bucket.name,'error':repr(exc)});continue
        passed.append(bucket.name);receipts.append(result)
    result={'schema':'touchwam_sharpa_all_source_readback_v1','status':'passed' if len(passed)==len(buckets) else 'incomplete',
            'source_dataset':str(source),'tactile_cache':str(cache),'source_manifest_sha256':digest(source/'dataset_manifest.json'),
            'buckets_expected':len(buckets),'buckets_passed':len(passed),'pending':pending,'failed':failed,'stale':stale,
            'samples':sum(x['sample_count'] for x in receipts),'matched_samples':sum(x['matched_sample_count'] for x in receipts),
            'masked_source_unavailable_samples':sum(x['masked_source_unavailable_count'] for x in receipts),
            'region_pixel_comparisons':sum(x['matched_sample_count'] for x in receipts)*10*32*32,
            'decoded_frames_including_seek_preroll':sum(x['decoded_frames_including_seek_preroll'] for x in receipts),
            'scope':'Independent sampled source pixel/offset comparison for every partition; structural validation for all episodes; not full frame-by-frame source replay.'}
    atomic_json(output/'summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--limit',type=int,help='Pilot only: first N buckets in this fixed shard; never global completion')
    p.add_argument('--aggregate',action='store_true',help='Check all original buckets against existing receipts; no video decoding')
    p.add_argument('--no-reuse',action='store_true');a=p.parse_args()
    if a.workers<1 or not 0<=a.shard_index<a.shard_count:p.error('Invalid workers or fixed shard index/count')
    if a.limit is not None and a.limit<1:p.error('--limit must be positive')
    source=Path(a.source);cache=Path(a.cache);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if a.aggregate:
        result=aggregate(source,cache,output);print(json.dumps(result),flush=True);raise SystemExit(0 if result['status']=='passed' else 2)
    buckets=sorted(p for p in (source/'buckets').iterdir() if p.is_dir())[a.shard_index::a.shard_count]
    if a.limit is not None:buckets=buckets[:a.limit]
    failures=0;pending=0;passed=0
    with ProcessPoolExecutor(a.workers) as pool:
        futures={pool.submit(audit_bucket,b,cache/'buckets'/b.name/'tactile/manifest.json',output/'receipts'/f'{b.name}.json',not a.no_reuse):b for b in buckets}
        for future in as_completed(futures):
            result=future.result();failures+=result['status']=='failed';pending+=result['status']=='pending_manifest';passed+=result['status']=='passed'
            print(json.dumps({k:result[k] for k in ('bucket','status','sample_count','matched_sample_count','elapsed_seconds','error','reused') if k in result}),flush=True)
    summary={'scope':'fixed_shard_pilot' if a.limit else 'fixed_shard','shard_index':a.shard_index,'shard_count':a.shard_count,'buckets_selected':len(buckets),'passed':passed,'failed':failures,'pending':pending}
    atomic_json(output/f'shard_{a.shard_index:03d}_of_{a.shard_count:03d}.json',summary)
    print(json.dumps(summary),flush=True);raise SystemExit(1 if failures else 2 if pending else 0)
