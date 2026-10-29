"""Restartable TouchScale WDS -> native LeRobot v3 on PFS, no robot actions."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = 'touchscale_lerobot_v3_v1'
REVISION = '4bc32c64c48d3711e99e1483a2520bdceda50c85'
CAMERAS = ['rgb_head', 'wrist_left', 'wrist_right']
VIDEO_KEYS = ['observation.images.' + c for c in CAMERAS]
KEEP = {'json', 'episode.h5', *[c + suffix for c in CAMERAS for suffix in ('.csv','.mp4')]}
DATA_PATH = 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
VIDEO_PATH = 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def layout():
    root = Path(__file__).resolve().parents[3]
    rows = json.loads((root / 'configs/tactile_layouts/touchscale_full_hand.json').read_text())['surfaces']
    offsets = dict(left=0, right=0)
    result = []
    for r in rows:
        side = r['native_key'].split('/')[-2]
        h,w = r['native_shape']
        result.append(dict(region=r['region'], side=side, native_key=r['native_key'],
                           native_shape=[h,w], offset=offsets[side], native_name=r['native_name']))
        offsets[side] += h*w
    assert offsets == dict(left=880,right=880)
    return result


def plan(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = source / 'metadata/episodes.parquet'
    table = pq.read_table(metadata).to_pandas()
    assert not table.episode_id.duplicated().any()
    groups = []
    for shard, group in table.groupby('shard', sort=True):
        path = source / shard
        groups.append(dict(shard=shard, bytes=path.stat().st_size,
                           episode_ids=sorted(group.episode_id.tolist())))
    identity = dict(schema=SCHEMA, source_revision=REVISION, source_root=str(source),
                    source_metadata_sha256=sha(metadata), episodes=len(table),
                    source_hours=float(table.duration_s.sum()/3600), groups=groups,
                    layout=layout(), camera_max_age_s=.1, force_history_max_age_s=.05,
                    force_future_tolerance_s=.025, video_encoding='copy if contiguous; otherwise lossless h264 yuv420p CRF0',
                    split='official train; no synthetic evaluation split')
    target = output / 'conversion_plan.json'
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError('Existing TouchScale plan differs; use a new output release')
    write(target, identity)
    provenance = output/'provenance';provenance.mkdir(exist_ok=True)
    for rel in ['README.md','metadata/episodes.parquet','metadata/glove_geometry_left.json','metadata/glove_geometry_right.json']:
        dest = provenance / Path(rel).name
        if dest.exists():
            if sha(dest) != sha(source/rel): raise ValueError('Source provenance changed')
        else: shutil.copyfile(source/rel, dest)
    if not (output/'publication.json').exists():
        write(output/'publication.json', dict(schema=SCHEMA,status='building',training_ready=False,
              buckets_expected=len(groups),buckets_completed=0,episodes=len(table),frames=0))
    return identity


def clock_selection(times, grid, max_age=.1):
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or (np.diff(times)<=0).any()):
        raise ValueError('Invalid physical source clock')
    rows = np.searchsorted(times, grid+1e-8, side='right')-1
    clipped = rows.clip(0,len(times)-1)
    valid = (rows>=0)&(grid-times[clipped]<=max_age)&(times[clipped]<=grid+1e-8)
    return clipped, valid


def video(source, dest, indices, total, threads):
    """Verify all source frames; use original payload when a regular slice suffices."""
    contiguous = bool(np.all(np.diff(indices)==1))
    dest.parent.mkdir(parents=True,exist_ok=True)
    writer = stream = None
    cursor = 0; count = 0
    if not contiguous:
        writer = av.open(str(dest), 'w')
        stream = writer.add_stream('libx264', rate=30)
        stream.pix_fmt = 'yuv420p'
        stream.thread_count = threads
        stream.options = {'crf':'0','preset':'veryfast','threads':str(threads)}
    try:
        with av.open(str(source)) as container:
            src = container.streams.video[0];src.thread_count=1
            if len(container.streams.video)!=1 or float(src.average_rate)!=30 or src.frames!=total:
                raise ValueError('Video clock/header differs from CSV')
            shape = [src.height,src.width,3]
            if stream is not None:
                stream.height=src.height;stream.width=src.width
            for i,frame in enumerate(container.decode(src)):
                if frame.pts is None or abs(float(frame.pts*src.time_base)*30-i)>2e-3:
                    raise ValueError('Video PTS is not its declared 30Hz frame grid')
                count+=1
                if stream is not None:
                    while cursor<len(indices) and indices[cursor]==i:
                        frame.pts=cursor;frame.time_base=Fraction(1,30)
                        for packet in stream.encode(frame):writer.mux(packet)
                        cursor+=1
        if count!=total or indices.max()>=count:
            raise ValueError('Truncated source video')
        if stream is not None:
            if cursor!=len(indices):raise ValueError('Missing requested camera frames')
            for packet in stream.encode():writer.mux(packet)
    finally:
        if writer is not None:writer.close()
    if contiguous:
        shutil.copyfile(source,dest)
    # Read back the full encoded output when it differs from the source bytes.
    if not contiguous:
        with av.open(str(dest)) as check:
            check.streams.video[0].thread_count=1
            if sum(1 for _ in check.decode(video=0))!=len(indices):
                raise ValueError('Incomplete synchronized video')
    return dict(shape=shape,from_timestamp=float(indices[0]/30) if contiguous else 0.,
                storage='original_payload' if contiguous else 'lossless_causal_resample',sha256=sha(dest))


def convert_episode(raw, root, row, ep, task_id, geometry, threads=2):
    key = row['episode_id'];receipt=root/'receipts'/f'{ep:06d}.json'
    if receipt.exists():
        result=json.loads(receipt.read_text())
        if result['source_episode_id']!=key:raise ValueError('Episode identity changed')
        for name,entry in result['files'].items():
            p=root/name
            if not p.is_file() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:
                raise ValueError('Completed episode changed')
        return result
    paths={ext:raw/f'{key}.{ext}' for ext in KEEP}
    source_sha={ext:sha(p) for ext,p in paths.items()}
    chunk,file=divmod(ep,1000)
    frames={}; force_entries={};rawvalues=[];force_valid=[]
    h5dest=root/'provenance/episodes'/f'{ep:06d}.h5';h5dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(paths['episode.h5'],h5dest)
    with h5py.File(h5dest) as h:
        if int(h['metadata'].attrs['has_action'])!=0:raise ValueError('Unexpected robot action source')
        original_ns=h['time/timestamp_ns'][:]
        if len(original_ns)<6 or (np.diff(original_ns)<=0).any():raise ValueError('Invalid episode timeline')
        epoch=float(original_ns[0]/1e9)
        n=int(np.floor((original_ns[-1]-original_ns[0])/1e9*30+1e-6))+1
        grid=np.arange(n,dtype=np.float64)/30
        # Epoch-relative arithmetic avoids Unix-float32 precision loss.
        if n<6:raise ValueError('Too short for video supervision')
        for side in ['right','left']:
            group=h[f'obs/tactile/raw/{side}']
            times=np.asarray(group['t_epoch'][:],dtype=np.float64)-epoch
            for r in geometry:
                if r['side']==side and h[r['native_key']].shape!=(len(times),*r['native_shape']):
                    raise ValueError('Source pad shape differs from native geometry')
            force=np.concatenate([np.asarray(h[r['native_key']][:],dtype=np.float32).reshape(len(times),-1)
                                  for r in geometry if r['side']==side],axis=1)
            if force.shape!=(len(times),880):raise ValueError('Invalid hand geometry')
            selected,valid=clock_selection(times,grid,.05)
            vals=force[selected].copy()
            pv=[]
            for r in geometry:
                if r['side']!=side:continue
                count=int(np.prod(r['native_shape']));pv.append(valid&np.isfinite(vals[:,r['offset']:r['offset']+count]).all(1))
            force_valid.append(np.stack(pv,1));rawvalues.append(vals)
            tactile=root/'tactile/episodes'/f'{ep:06d}';tactile.mkdir(parents=True,exist_ok=True)
            np.save(tactile/f'{side}_force.npy',force,allow_pickle=False)
            np.save(tactile/f'{side}_timestamps.npy',times,allow_pickle=False)
            force_entries[side]=dict(force=f'episodes/{ep:06d}/{side}_force.npy',timestamps=f'episodes/{ep:06d}/{side}_timestamps.npy')
        for c in CAMERAS:
            data=np.genfromtxt(paths[c+'.csv'],delimiter=',',names=True)
            if data.dtype.names!=('frame_index','timestamp_s') or not np.array_equal(data['frame_index'],np.arange(len(data))):
                raise ValueError('Video CSV frame identity mismatch')
            ts=np.asarray(data['timestamp_s'],dtype=np.float64)-epoch
            indices,valid=clock_selection(ts,grid,.1)
            dest=root/VIDEO_PATH.format(video_key='observation.images.'+c,chunk_index=chunk,file_index=file)
            check=video(paths[c+'.mp4'],dest,indices,len(data),threads)
            check.update(source_frame_indices=indices,source_timestamps=ts[indices],valid=valid)
            frames[c]=check
    columns=dict(timestamp=pa.array(grid),episode_index=pa.array(np.full(n,ep,np.int64)),
                 frame_index=pa.array(np.arange(n,dtype=np.int64)),index=pa.array(np.arange(n,dtype=np.int64)),
                 task_index=pa.array(np.full(n,task_id,np.int64)),prompt=pa.array([row['instruction']]*n))
    normal=np.concatenate(rawvalues,axis=1)
    columns['observation.tactile.normal_force']=pa.FixedSizeListArray.from_arrays(pa.array(normal.reshape(-1)),1760)
    valid=np.concatenate(force_valid,axis=1)
    columns['observation.tactile.valid']=pa.FixedSizeListArray.from_arrays(pa.array(valid.reshape(-1)),30)
    for c in CAMERAS:
        prefix='observation.images.'+c
        columns[prefix+'.valid']=pa.array(frames[c]['valid'])
        columns[prefix+'.source_frame_index']=pa.array(frames[c]['source_frame_indices'])
        columns[prefix+'.source_timestamp']=pa.array(frames[c]['source_timestamps'])
    dest=root/DATA_PATH.format(chunk_index=chunk,file_index=file);dest.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.table(columns),dest,compression='zstd')
    provenance=root/'provenance/episodes'/f'{ep:06d}.json'
    write(provenance,dict(source_metadata=row,source_payload_sha256=source_sha,epoch_s=epoch,
                         timeline='regular 30Hz inside source H5 interval',source_units='newton_assumed_from_vendor_calibrated_output'))
    for c in CAMERAS:shutil.copyfile(paths[c+'.csv'],provenance.with_suffix('.'+c+'.csv'))
    files={}
    for p in [dest,h5dest,provenance,*provenance.parent.glob(f'{ep:06d}.*.csv'),
              *(root/'tactile/episodes'/f'{ep:06d}').glob('*.npy')]:
        files[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    for c,check in frames.items():
        p=root/VIDEO_PATH.format(video_key='observation.images.'+c,chunk_index=chunk,file_index=file)
        files[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=check['sha256'])
    result=dict(source_episode_id=key,episode_index=ep,frames=n,task_index=task_id,
                tactile={**force_entries,'duration_s':n/30},files=files,
                camera={c:{k:v for k,v in check.items() if k not in ('source_frame_indices','source_timestamps','valid')}
                        for c,check in frames.items()},camera_valid_frames={c:int(x['valid'].sum()) for c,x in frames.items()},
                valid_force_regions=int(valid.sum()),elapsed_complete=time.time())
    write(receipt,result)
    return result


def finish_bucket(root, group, rows, geometry, plan_sha):
    receipts=[json.loads((root/'receipts'/f'{i:06d}.json').read_text()) for i in range(len(rows))]
    features={k:dict(dtype=d,shape=[1]) for k,d in [('timestamp','float64'),('episode_index','int64'),
              ('frame_index','int64'),('index','int64'),('task_index','int64')]}
    features['prompt']=dict(dtype='string',shape=[1])
    features['observation.tactile.normal_force']=dict(dtype='float32',shape=[1760],units='N_assumed_vendor_calibrated')
    features['observation.tactile.valid']=dict(dtype='bool',shape=[30])
    for c in CAMERAS:
        key='observation.images.'+c
        shapes={tuple(x['camera'][c]['shape']) for x in receipts}
        if len(shapes)!=1:raise ValueError('Changing camera shape within a bucket')
        features[key]=dict(dtype='video',shape=list(shapes.pop()),names=['height','width','channels'],
                           info={'video.fps':30,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False})
        for suffix,dtype in [('.valid','bool'),('.source_frame_index','int64'),('.source_timestamp','float64')]:
            features[key+suffix]=dict(dtype=dtype,shape=[1])
    total=sum(x['frames'] for x in receipts)
    info=dict(codebase_version='v3.0',robot_type='human_tactile_gloves',fps=30,total_episodes=len(rows),total_frames=total,
              total_tasks=len({r['task_id'] for r in rows}),total_videos=3*len(rows),chunks_size=1000,
              data_files_size_in_mb=100,video_files_size_in_mb=500,splits={'train':f'0:{len(rows)}'},
              data_path=DATA_PATH,video_path=VIDEO_PATH,features=features)
    write(root/'meta/info.json',info)
    write(root/'meta/source_definition.json',dict(source_id='touchscale',source_kind='ego',training_ready=True,
          physical_time_verified=True,camera_layout=VIDEO_KEYS,training_modality='video_tactile',has_action=False,
          source_revision=REVISION,source_shard=group['shard'],conversion_plan_sha256=plan_sha))
    episodes=[];start=0
    for i,(row,rec) in enumerate(zip(rows,receipts)):
        chunk,file=divmod(i,1000);n=rec['frames']
        path=root/DATA_PATH.format(chunk_index=chunk,file_index=file)
        tab=pq.read_table(path)
        # LeRobot index is global within this bucket; one data file per episode.
        tab=tab.set_column(tab.schema.get_field_index('index'),'index',pa.array(np.arange(start,start+n,dtype=np.int64)))
        pq.write_table(tab,path,compression='zstd')
        rec['files'][str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=sha(path));write(root/'receipts'/f'{i:06d}.json',rec)
        ep=dict(episode_index=i,length=n,dataset_from_index=start,dataset_to_index=start+n,tasks=[row['instruction']],
                **{'data/chunk_index':chunk,'data/file_index':file})
        for c in CAMERAS:
            prefix='videos/observation.images.'+c+'/'
            offset=rec['camera'][c]['from_timestamp']
            ep.update({prefix+'chunk_index':chunk,prefix+'file_index':file,prefix+'from_timestamp':offset,prefix+'to_timestamp':offset+n/30})
        episodes.append(ep);start+=n
    epfile=root/'meta/episodes/chunk-000/file-000.parquet';epfile.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episodes),epfile,compression='zstd')
    tasks={r['task_index']:rows[i]['task_name'] for i,r in enumerate(receipts)}
    pq.write_table(pa.Table.from_pylist([dict(task_index=i,task=t) for i,t in sorted(tasks.items())]),root/'meta/tasks.parquet')
    write(root/'tactile/manifest.json',dict(schema='touchscale_native_force_v1',codec='tactile_intensity_v1',
          dataset_info_sha256=sha(root/'meta/info.json'),encoder_name='TouchScaleForce',layout=geometry,
          normalization='max(force_N,0)/(max(force_N,0)+1N)',episodes={str(i):r['tactile'] for i,r in enumerate(receipts)}))
    # Production reader checks real images/touch at both boundaries and middle.
    from openwam.dataloader.touchwam_pretrain import LeRobotVideoIntensityDataset
    reader=LeRobotVideoIntensityDataset(root,tactile_mode='required',tactile_representation='touchscale_force',
             tactile_observation_map_size=[16,10],tactile_encoder_name='TouchScaleForce')
    checked=[]
    for idx in sorted({0,len(reader)//2,len(reader)-1}):
        sample=reader[idx]
        if sample['tactile_observation_maps'].shape!=(4,30,1,16,10) or 'action' in sample:
            raise ValueError('Production reader violated human tactile contract')
        checked.append(dict(index=idx,episode=sample['episode_id'],frame=sample['episode_start_frame'],
                            valid_history=int(sample['tactile_valid'].sum()),valid_future=int(sample['future_tactile_valid'].sum())))
    inventory={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob('*') if p.is_file()}
    # A second full payload read detects partial writes before publication.
    for rel,entry in inventory.items():
        if sha(root/rel)!=entry['sha256']:raise ValueError('PFS payload readback differs')
    write(root/'validation.json',dict(status='passed',episodes=len(rows),frames=total,reader_windows=len(reader),
          reader_samples=checked,all_source_videos_decoded=True,all_output_payloads_readback_sha256=True,
          conversion_plan_sha256=plan_sha,files=inventory))
    return dict(status='complete',episodes=len(rows),frames=total,reader_windows=len(reader),
                validation_sha256=sha(root/'validation.json'),info_sha256=sha(root/'meta/info.json'),
                tactile_sha256=sha(root/'tactile/manifest.json'))


def run_shard(args):
    output,index,threads=args
    output=Path(output);planfile=output/'conversion_plan.json';p=json.loads(planfile.read_text());g=p['groups'][index]
    name=f'shard-{index:05d}';final=output/'buckets'/name
    if final.exists():
        return json.loads((final/'complete.json').read_text())
    root=output/'_building'/name;root.mkdir(parents=True,exist_ok=True)
    raw=root/'_raw';raw.mkdir(exist_ok=True)
    table=pq.read_table(output/'provenance/episodes.parquet').to_pandas()
    taskmap={s:i for i,s in enumerate(sorted(table.task_id.unique()))}
    subset=table[table.shard==g['shard']].sort_values('episode_id')
    rows=json.loads(subset.to_json(orient='records'))
    byid={r['episode_id']:(i,r) for i,r in enumerate(rows)}
    src=Path(p['source_root'])/g['shard']
    if src.stat().st_size!=g['bytes']:raise ValueError('Source tar changed')
    seen=set();current=None;parts=set()
    def finish():
        if current is None:return
        if parts!=KEEP:raise ValueError(f'Incomplete source episode {current}: {KEEP-parts}')
        i,row=byid[current]
        convert_episode(raw,root,row,i,taskmap[row['task_id']],p['layout'],threads)
        for ext in KEEP:(raw/f'{current}.{ext}').unlink()
        seen.add(current)
    with tarfile.open(src,mode='r:') as archive:
        for m in archive:
            if (output/'STOP').exists():raise RuntimeError('TouchScale STOP requested; completed receipts retained')
            if not m.isfile():continue
            if '/' in m.name:raise ValueError('Unexpected nested WDS member')
            key,ext=m.name.split('.',1)
            if key not in byid:raise ValueError('Archive episode missing from source metadata')
            if key!=current:
                finish();current=key;parts=set()
                if key in seen:raise ValueError('Duplicate noncontiguous episode in tar')
            if ext not in KEEP:continue
            if ext in parts:raise ValueError('Duplicate archive member')
            with archive.extractfile(m) as inp,(raw/m.name).open('wb') as out:shutil.copyfileobj(inp,out,1024*1024)
            if (raw/m.name).stat().st_size!=m.size:raise ValueError('Short archive read')
            parts.add(ext)
        finish()
    if seen!=set(g['episode_ids']):raise ValueError('Missing source episodes')
    raw.rmdir()
    receipt=finish_bucket(root,g,rows,p['layout'],sha(planfile));receipt.update(shard_index=index,source_shard=g['shard'])
    write(root/'complete.json',receipt)
    final.parent.mkdir(exist_ok=True);os.rename(root,final)
    print(json.dumps(dict(event='shard_complete',**receipt)),flush=True)
    return receipt


def finalize(output):
    output=Path(output);p=json.loads((output/'conversion_plan.json').read_text());buckets={}
    for i,g in enumerate(p['groups']):
        root=output/'buckets'/f'shard-{i:05d}'
        r=json.loads((root/'complete.json').read_text())
        assert r['status']=='complete' and r['episodes']==len(g['episode_ids'])
        assert r['validation_sha256']==sha(root/'validation.json')
        buckets[root.name]=r
    episodes=sum(r['episodes'] for r in buckets.values());frames=sum(r['frames'] for r in buckets.values())
    assert episodes==p['episodes']
    result=dict(schema=SCHEMA,status='complete',training_ready=True,buckets_expected=len(buckets),buckets_completed=len(buckets),
                episodes=episodes,frames=frames,hours=frames/30/3600,buckets=buckets,source_revision=REVISION,
                conversion_plan_sha256=sha(output/'conversion_plan.json'),source_hours=p['source_hours'],
                source_kind='ego',training_modality='video_tactile',external_dependencies=[],
                excluded_training_modalities=['head_depth','IMU','tangential_force'],
                retained_auxiliary='source H5 contains native shear, IMU, calibration; original depth stays in BOS')
    write(output/'publication.json',result)
    return {k:v for k,v in result.items() if k!='buckets'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['plan','worker','finalize'])
    parser.add_argument('--source');parser.add_argument('--output',required=True)
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--world',type=int,default=1)
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--encode-threads',type=int,default=2)
    parser.add_argument('--shard',type=int)
    a=parser.parse_args()
    if a.operation=='plan':
        p=plan(a.source,a.output);print(json.dumps({k:v for k,v in p.items() if k not in ('groups','layout')}));return
    if a.operation=='finalize':print(json.dumps(finalize(a.output)));return
    p=json.loads((Path(a.output)/'conversion_plan.json').read_text())
    indices=[a.shard] if a.shard is not None else list(range(a.rank,len(p['groups']),a.world))
    if a.workers==1:
        for i in indices:run_shard((a.output,i,a.encode_threads))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(run_shard,(a.output,i,a.encode_threads)):i for i in indices}
            for f in as_completed(futures):
                try:f.result()
                except Exception as e:
                    write(Path(a.output)/'errors'/f'shard-{futures[f]:05d}.json',dict(error=str(e),time=time.time()))
                    raise


if __name__=='__main__':main()
