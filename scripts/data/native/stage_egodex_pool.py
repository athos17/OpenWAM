#!/usr/bin/env python3
"""Resume-capable immutable staging of the entire official EgoDex v3 train pool.

No new episode selection. The source revision and each byte SHA256 are recorded.
Only validated, fully copied buckets enter catalog.ready. Official test remains
an explicit separate source and never enters the training pool. Human transforms
are preserved for provenance but disabled as model action/proprio labels.
"""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,time
from pathlib import Path
import pyarrow.parquet as pq


def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(obj,indent=2)+'\n');os.replace(t,p)

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()

def copy_verified(src,dst,record):
    st=src.stat();size=st.st_size
    if dst.exists() and record and record['bytes']==size and dst.stat().st_size==size:
        if record['source_mtime_ns']==st.st_mtime_ns:
            return record
    if dst.exists():
        # A crash may occur after atomic rename and before the receipt flush.
        if dst.stat().st_size==size:
            expected=hash_file(src)
            if hash_file(dst)==expected:
                return {'bytes':size,'sha256':expected,'source_mtime_ns':st.st_mtime_ns}
        raise ValueError(f'Existing immutable file differs from source: {dst}')
    dst.parent.mkdir(parents=True,exist_ok=True);part=dst.with_name(dst.name+'.partial');h=hashlib.sha256()
    with open(src,'rb') as fi,open(part,'wb') as fo:
        while True:
            b=fi.read(8<<20)
            if not b:break
            fo.write(b);h.update(b)
    if part.stat().st_size!=size or hash_file(part)!=h.hexdigest():raise ValueError(f'Staged bytes mismatch: {src}')
    if src.stat().st_mtime_ns!=st.st_mtime_ns:raise ValueError(f'Source changed while copying: {src}')
    os.replace(part,dst)
    return {'bytes':size,'sha256':h.hexdigest(),'source_mtime_ns':st.st_mtime_ns}

def stage_bucket(source,output,name):
    src=source/'train'/name;dst=output/'train'/name
    receipt_path=output/'receipts'/f'{name}.json';receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'files':{}}
    info=json.loads((src/'meta/info.json').read_text())
    if info['codebase_version']!='v3.0' or info['fps']!=30:raise ValueError(f'Unexpected EgoDex definition {src}')
    receipt.update(source=str(src),output=str(dst),status='copying',episodes=info['total_episodes'],frames=info['total_frames'],hours=info['total_frames']/30/3600)
    files=[]
    for root,dirs,names in os.walk(src):
        dirs[:]=sorted(d for d in dirs if not d.startswith('.'))
        for filename in sorted(names):
            if filename.startswith('.'):continue
            f=Path(root)/filename
            if f.is_symlink():raise ValueError(f'Unexpected source symlink: {f}')
            files.append(f)
    receipt['files_expected']=len(files);atomic_json(receipt_path,receipt)
    # Metadata first lets status and downstream independent checks remain useful.
    files.sort(key=lambda f:(not str(f.relative_to(src)).startswith('meta/'),str(f)))
    for i,f in enumerate(files):
        rel=str(f.relative_to(src));record=copy_verified(f,dst/rel,receipt['files'].get(rel));receipt['files'][rel]=record
        if i%16==0 or i+1==len(files):
            receipt['files_completed']=len(receipt['files']);receipt['bytes_completed']=sum(x['bytes'] for x in receipt['files'].values());atomic_json(receipt_path,receipt)
            print(json.dumps({'event':'copy','bucket':name,'files_completed':receipt['files_completed'],'files_expected':len(files),'bytes_completed':receipt['bytes_completed']}),flush=True)
    eps=[]
    for path in sorted(dst.glob('meta/episodes/*/*.parquet')):eps.extend(pq.read_table(path,columns=['episode_index','length']).to_pylist())
    if len(eps)!=info['total_episodes'] or sum(e['length'] for e in eps)!=info['total_frames']:raise ValueError(f'Full pool metadata mismatch {name}')
    data_rows=sum(pq.read_metadata(p).num_rows for p in dst.glob('data/*/*.parquet'))
    if data_rows!=info['total_frames']:raise ValueError(f'Full pool parquet row mismatch {name}')
    receipt.update(status='complete',files_completed=len(files),bytes_completed=sum(x['bytes'] for x in receipt['files'].values()),verified_data_rows=data_rows)
    atomic_json(receipt_path,receipt)
    return {k:v for k,v in receipt.items() if k!='files'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    source,output=Path(a.source).resolve(),Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    names=sorted(p.name for p in (source/'train').iterdir() if p.is_dir())
    catalog={'schema':'touchwam_egodex_full_pool_v1','source_revision':source.name,'source':str(source),'output':str(output),'status':'staging',
             'source_split':'official_train','reserved_test':str(source/'test'),'supervision':['video'],'action_supervision':False,'proprio_supervision':False,'tactile_supervision':False,
             'selection_policy':'all official train episodes; downstream sample allocation is not defined','mutually_exclusive_with':['accepted_cache_300h'],
             'buckets_expected':len(names),'ready':[],'errors':[]}
    # Inventory independent of copying and without enumerating multi-terabyte bodies.
    for n in names:
        info=json.loads((source/'train'/n/'meta/info.json').read_text())
        catalog.setdefault('inventory',[]).append({'bucket':n,'episodes':info['total_episodes'],'frames':info['total_frames'],'fps':info['fps']})
    catalog['total_episodes']=sum(i['episodes'] for i in catalog['inventory']);catalog['total_frames']=sum(i['frames'] for i in catalog['inventory']);catalog['total_hours']=sum(i['frames']/i['fps']/3600 for i in catalog['inventory'])
    atomic_json(output/'catalog.json',catalog);print(json.dumps({k:v for k,v in catalog.items() if k!='inventory'}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        futures={pool.submit(stage_bucket,source,output,name):name for name in names}
        for future in concurrent.futures.as_completed(futures):
            try:r=future.result();catalog['ready'].append(r);print(json.dumps({'event':'bucket_complete',**r}),flush=True)
            except Exception as e:catalog['errors'].append({'bucket':futures[future],'error':repr(e)});print(json.dumps(catalog['errors'][-1]),flush=True)
            atomic_json(output/'catalog.json',catalog)
    catalog['status']='complete' if len(catalog['ready'])==len(names) else 'incomplete';atomic_json(output/'catalog.json',catalog)
    if catalog['status']!='complete':raise SystemExit(1)
