#!/usr/bin/env python3
"""Decode SharpaDex's synchronized 2x5 deform mosaic once into regional sidecars.

Higher stored gray intensity means stronger SDK contact-deformation response;
this is a relative intensity and has no N/mm/Pa calibration. No per-frame minmax
or inferred force is used. A decoded black tile is valid zero contact; decode
failure is missing data. Video timestamps retain the published source offsets.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REGIONS=[f'{side}.{finger}.tip' for side in ('left','right') for finger in ('thumb','index','middle','ring','pinky')]
KEY='observation.images.tactile_deform'
VERSION='sharpa_deform_mosaic_v1_20261028'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def split_mosaic(gray):
    """Preserve native orientation; top left->right = left thumb..little."""
    import cv2
    if gray.shape!=(480,1200) or gray.dtype!=np.uint8:raise ValueError(f'Unexpected deform mosaic: {gray.shape}/{gray.dtype}')
    small=cv2.resize(gray,(160,64),interpolation=cv2.INTER_AREA)
    return small.reshape(2,32,5,32).transpose(0,2,1,3).reshape(10,32,32)

def split_video_frame(frame):
    # Decode luma without a full 1200x480 swscale conversion per frame. The
    # source is grayscale H.264/yuv420p; normalize its declared video range.
    import cv2
    if frame.format.name not in ('yuv420p','yuvj420p'):
        return split_mosaic(frame.to_ndarray(format='gray'))
    if (frame.height,frame.width)!=(480,1200):raise ValueError('Unexpected video shape')
    y=np.frombuffer(frame.planes[0],dtype=np.uint8).reshape(frame.height,frame.planes[0].line_size)[:,:frame.width]
    small=cv2.resize(y,(160,64),interpolation=cv2.INTER_AREA)
    if frame.color_range!=2 and frame.format.name!='yuvj420p':
        small=np.rint(np.clip((small.astype(np.float32)-16)*255/219,0,255)).astype(np.uint8)
    return small.reshape(2,32,5,32).transpose(0,2,1,3).reshape(10,32,32)

def extract_partition(path):
    import av,cv2
    cv2.setNumThreads(1)
    path=Path(path);troot=path/'tactile';troot.mkdir(exist_ok=True)
    info_path=path/'meta/info.json';info=json.loads(info_path.read_text()); info_hash=digest(info_path)
    if info['fps']!=30 or info['features'][KEY]['shape']!=[480,1200,3]:raise ValueError(f'Incompatible source {path}')
    signature=hashlib.sha256((VERSION+info_hash+digest(__file__)).encode()).hexdigest()
    manifest=troot/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get('conversion_signature')!=signature:raise ValueError(f'Existing sidecar signature differs: {path}')
        if all((troot/e[k]).exists() for e in old['episodes'].values() for k in ('maps','valid','timestamps')):
            return {'partition':str(path),'status':'reused','episodes':len(old['episodes']),'frames':old['frames'],'valid_frames':old['valid_frames']}
        raise ValueError(f'Published manifest has missing arrays: {path}')
    eps=[]
    for f in sorted(path.glob('meta/episodes/*/*.parquet')):eps.extend(pq.read_table(f).to_pylist())
    grouped={};entries={};frame_count=0;valid_frames=0;errors=[]
    for ep in eps:
        idx=int(ep['episode_index']);n=int(ep['length']);offset=float(ep[f'videos/{KEY}/from_timestamp'])
        start=round(offset*30)
        if n<1 or abs(start/30-offset)>1e-4:raise ValueError(f'Invalid video offset {ep}')
        vrel=info['video_path'].format(video_key=KEY,chunk_index=ep[f'videos/{KEY}/chunk_index'],file_index=ep[f'videos/{KEY}/file_index'])
        grouped.setdefault(vrel,[]).append((start,start+n,idx,n))
        entries[str(idx)]={'maps':f'episode_{idx:06d}/maps.npy','valid':f'episode_{idx:06d}/valid.npy','timestamps':f'episode_{idx:06d}/timestamps.npy',
                          'layout_present':[True]*10,'source_video':vrel,'source_video_from_timestamp':offset,
                          'source_episode_index':int(ep.get('source_episode_index',idx)), 'frames':n}
        frame_count+=n
    for vrel,items in sorted(grouped.items()):
        items.sort();video=path/vrel;last_end=-1
        for start,end,idx,n in items:
            if start<last_end:raise ValueError(f'Overlapping source video episodes {vrel}')
            last_end=end
        # Limit open arrays to one episode and publish an episode only after complete video pass.
        c=av.open(str(video));s=c.streams.video[0];s.thread_count=1
        c.seek(int(max(0,items[0][0]/30-1)/s.time_base),stream=s)
        position=0;arrays=None;written=0;previous_pts=None
        def open_episode(item):
            start,end,idx,n=item;d=troot/f'episode_{idx:06d}';d.mkdir(exist_ok=True)
            maps=np.lib.format.open_memmap(d/'maps.npy.partial',mode='w+',dtype=np.uint8,shape=(n,10,32,32))
            valid=np.lib.format.open_memmap(d/'valid.npy.partial',mode='w+',dtype=np.bool_,shape=(n,10));valid[:]=False
            return maps,valid,d
        def finish_episode(arr,item):
            nonlocal valid_frames
            maps,valid,d=arr;start,end,idx,n=item
            valid_frames+=int(valid.all(axis=1).sum());maps.flush();valid.flush()
            # Actual timestamps use the published dataset's episode-relative 30 Hz grid.
            np.save(d/'timestamps.npy.partial',np.arange(n,dtype=np.float64)/30,allow_pickle=False)
            del maps,valid
            os.replace(d/'maps.npy.partial',d/'maps.npy');os.replace(d/'valid.npy.partial',d/'valid.npy')
            os.replace(d/'timestamps.npy.partial.npy',d/'timestamps.npy')
        try:
            for frame in c.decode(s):
                pts=float(frame.pts*s.time_base)
                if previous_pts is not None and pts<=previous_pts:raise ValueError(f'Nonmonotonic PTS in {video}')
                previous_pts=pts
                frame_index=round(pts*30)
                if abs(frame_index/30-pts)>2e-4:raise ValueError(f'Off-grid PTS in {video}: {pts}')
                while position<len(items) and frame_index>=items[position][1]:
                    if arrays is None:arrays=open_episode(items[position])
                    finish_episode(arrays,items[position]);arrays=None;position+=1
                if position>=len(items):break
                start,end,idx,n=items[position]
                if frame_index<start:continue
                if arrays is None:arrays=open_episode(items[position])
                row=frame_index-start
                arrays[0][row]=split_video_frame(frame)
                arrays[1][row]=True;written+=1
        except (av.error.FFmpegError,EOFError) as exc:
            errors.append({'video':vrel,'error':str(exc),'policy':'undecoded frames invalid'})
        finally:c.close()
        while position<len(items):
            if arrays is None:arrays=open_episode(items[position])
            finish_episode(arrays,items[position]);arrays=None;position+=1
    meta={'schema':'tactile_regions_v1','codec':'tactile_intensity_v1','fps':30.0,'dataset_info_sha256':info_hash,
          'region_names':REGIONS,'sensor_ids':[0]*10,'sensor_names':['sharpa_real']*10,'map_size':[32,32],
          'normalization':{'source':'SDK uint8 deform grayscale','mapping':'intensity=gray/255','higher_means':'stronger relative SDK deformation response','physical_units':None,'per_frame_rescaling':False},
          'spatial_transform':{'mosaic_shape':[480,1200],'tile_shape':[240,240],'layout':'2 rows x 5 columns; top left hand, bottom right hand; thumb,index,middle,ring,little left to right','rotation_degrees':0,'mirror':False,'resampling':'area in luma plane then fixed video-range conversion; at most rounding-level difference from grayscale-first area'},
          'timing':{'clock':'episode relative seconds','source':'source video PTS minus metadata from_timestamp','resampling':'exact same 30Hz grid; no future interpolation','missing':'valid=false'},
          'provenance':{'source_dataset':str(path),'conversion_version':VERSION,'converter_sha256':digest(__file__)},
          'conversion_signature':signature,'frames':frame_count,'valid_frames':valid_frames,'decode_errors':errors,'episodes':entries}
    atomic_json(manifest,meta)
    return {'partition':str(path),'status':'complete','episodes':len(eps),'frames':frame_count,'valid_frames':valid_frames,'decode_errors':errors}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--workers',type=int,default=24);p.add_argument('--limit',type=int);a=p.parse_args()
    paths=sorted(Path(a.dataset).glob('partitions/*/*'))
    if a.limit:paths=paths[:a.limit]
    started=time.time();totals=[]
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        futures={pool.submit(extract_partition,p):p for p in paths}
        for f in concurrent.futures.as_completed(futures):
            try:r=f.result();totals.append(r);print(json.dumps(r),flush=True)
            except Exception as e:print(json.dumps({'partition':str(futures[f]),'status':'error','error':repr(e)}),flush=True)
    report={'partitions_expected':len(paths),'partitions_completed':len(totals),'episodes':sum(t['episodes'] for t in totals),'frames':sum(t['frames'] for t in totals),'valid_frames':sum(t['valid_frames'] for t in totals),'elapsed_seconds':time.time()-started,'status':'complete' if len(totals)==len(paths) else 'incomplete'}
    atomic_json(Path(a.dataset)/'tactile_conversion.json',report);print(json.dumps(report),flush=True)
    if report['status']!='complete':raise SystemExit(1)
