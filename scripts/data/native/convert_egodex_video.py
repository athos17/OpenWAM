#!/usr/bin/env python3
"""Materialize the accepted EgoDex video cache as actual LeRobot v3.

Copies no human pose into a robot action slot and creates no dummy action field.
Video bytes are hardlinked, with their original per-episode offsets preserved.
This publishes the pre-existing cache selection, not a new training selection.
The complete official pool is catalogued separately by inventory_egodex.py.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERA='observation.images.camera'
VERSION='egodex_video_lerobot_v3_20261028'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.json.partial');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)

def convert(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    identity={'version':VERSION,'episodes_sha256':digest(source/'episodes.parquet'),'dataset_info_sha256':digest(source/'dataset_info.json')}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if (output/'publication.json').exists():
        report=json.loads((output/'publication.json').read_text())
        if report['signature']!=signature:raise ValueError('Existing EgoDex publication identity differs')
        return report
    if output.exists():raise ValueError('Refusing incomplete output; investigate staging directory')
    stage=output.with_name(output.name+'.building');stage.mkdir(parents=True,exist_ok=True)
    journal=stage/'build_identity.json'
    if journal.exists() and json.loads(journal.read_text())!=identity:raise ValueError('Staging identity differs')
    write_json(journal,identity)
    rows=pq.read_table(source/'episodes.parquet').to_pylist();source_info=json.loads((source/'dataset_info.json').read_text())
    video_paths=sorted({row['video_path'] for row in rows});vmap={name:divmod(i,1000) for i,name in enumerate(video_paths)}
    for name,(chunk,file) in vmap.items():
        src=source/name;dst=stage/'videos'/CAMERA/f'chunk-{chunk:03d}'/f'file-{file:03d}.mp4';dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if not os.path.samefile(src,dst):raise ValueError(f'Unexpected existing video {dst}')
        else:os.link(src,dst)
    prompts=sorted({r['prompt'] for r in rows});tasks={s:i for i,s in enumerate(prompts)}
    metas=[];global_index=0;frames=0; shard_size=1000
    for shard,start in enumerate(range(0,len(rows),shard_size)):
        batch=rows[start:start+shard_size];columns={k:[] for k in ('timestamp','frame_index','episode_index','index','task_index')}
        for j,row in enumerate(batch):
            ep=start+j;n=int(row['length']);first=int(row['video_from_frame']);last=int(row['video_to_frame'])
            if n<1 or last-first!=n or row['source_split']!='train':raise ValueError(f'Invalid source episode {row}')
            cols={'timestamp':np.arange(n,dtype=np.float32)/30,'frame_index':np.arange(n,dtype=np.int64),
                  'episode_index':np.full(n,ep,dtype=np.int64),'index':np.arange(global_index,global_index+n,dtype=np.int64),
                  'task_index':np.full(n,tasks[row['prompt']],dtype=np.int64)}
            for k,v in cols.items():columns[k].append(v)
            vc,vf=vmap[row['video_path']]
            metas.append({'episode_index':ep,'tasks':[row['prompt']],'length':n,'dataset_from_index':global_index,'dataset_to_index':global_index+n,
                          'data/chunk_index':shard//1000,'data/file_index':shard%1000,
                          f'videos/{CAMERA}/chunk_index':vc,f'videos/{CAMERA}/file_index':vf,
                          f'videos/{CAMERA}/from_timestamp':first/30,f'videos/{CAMERA}/to_timestamp':last/30,
                          'source_episode_id':row['id'],'source_dataset':row['source_dataset'],'source_task':row['task'],'source_split':row['source_split']})
            global_index+=n;frames+=n
        path=stage/'data'/f'chunk-{shard//1000:03d}'/f'file-{shard%1000:03d}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
        table=pa.table({k:np.concatenate(v) for k,v in columns.items()})
        if path.exists():
            old=pq.read_table(path)
            if not old.equals(table):raise ValueError(f'Existing data shard differs {path}')
        else:pq.write_table(table,path,compression='zstd')
        if pq.read_metadata(path).num_rows!=len(table):raise ValueError(f'Parquet row count differs {path}')
        print(json.dumps({'event':'shard','shard':shard,'episodes':start+len(batch),'frames':frames}),flush=True)
    for shard,start in enumerate(range(0,len(metas),1000)):
        path=stage/'meta/episodes'/f'chunk-{shard//1000:03d}'/f'file-{shard%1000:03d}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist(metas[start:start+1000]),path,compression='zstd')
    pq.write_table(pa.table({'task_index':np.arange(len(prompts),dtype=np.int64),'task':prompts}),stage/'meta/tasks.parquet',compression='zstd')
    features={k:{'dtype':'float32' if k=='timestamp' else 'int64','shape':[1],'names':None} for k in columns}
    # Inspect the actual first stream rather than assuming a source resolution.
    import av
    with av.open(str(stage/'videos'/CAMERA/'chunk-000/file-000.mp4')) as container:
        s=container.streams.video[0];width,height=s.width,s.height;codec=s.codec_context.name
        codec={'libdav1d':'av1','libaom-av1':'av1','libvpx-vp9':'vp9'}.get(codec,codec)
    features[CAMERA]={'dtype':'video','shape':[height,width,3],'names':['height','width','channels'],
                      'info':{'video.height':height,'video.width':width,'video.codec':codec,'video.fps':30,'video.channels':3,'video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}}
    info={'codebase_version':'v3.0','robot_type':None,'fps':30,'total_episodes':len(rows),'total_frames':frames,'total_tasks':len(prompts),
          'total_videos':len(video_paths),'chunks_size':1000,'splits':{'train':f'0:{len(rows)}'},
          'data_path':'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
          'video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4','features':features,
          'touchwam_contract':{'supervision':['video'],'action_supervision':False,'proprio_supervision':False,'tactile_supervision':False,'camera_timing':'preserved source offsets','selection':'pre-existing accepted 300h cache; not current training allocation'}}
    write_json(stage/'meta/info.json',info)
    if frames!=source_info['total_frames'] or len(rows)!=source_info['total_episodes']:raise ValueError('Source selection totals differ')
    report={'schema':'touchwam_native_publication_v1','signature':signature,**identity,'source':str(source),'episodes':len(rows),'frames':frames,'hours':frames/30/3600,
            'videos':len(video_paths),'selection_changed':False,'action_fields':[],'tactile_fields':[],'status':'complete','video_storage':'immutable hardlinks; offsets preserved'}
    write_json(stage/'publication.json',report);output.parent.mkdir(parents=True,exist_ok=True);os.rename(stage,output)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(convert(a.source,a.output)),flush=True)
