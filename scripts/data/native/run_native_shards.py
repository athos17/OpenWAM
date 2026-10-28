#!/usr/bin/env python3
"""Disjoint native conversion shards for independent CPU allocations.

Use identical --shard-count on every allocation and a unique --shard-index.
Each process additionally owns a filesystem advisory lock per bucket. PFS must
support POSIX advisory locks if allocations share overlapping task lists.
This launcher does not publish global completion; catalog_sharpa_tactile.py
checks all Sharpa buckets. Existing finished output is reused by the underlying
converter. --pending-only lets an idle allocation help unfinished Sharpa tasks
without waiting for locks held by the fixed shards.
"""
from __future__ import annotations
import argparse,concurrent.futures,fcntl,json,os
from pathlib import Path
from extract_sharpa_deform import extract_partition
from stage_egodex_pool import stage_bucket


def job(kind,source,output,item,skip_completed=False):
    locks=Path(output)/'conversion_locks';locks.mkdir(parents=True,exist_ok=True)
    lock_name=item.replace('/','__')+'.lock'
    with open(locks/lock_name,'a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'owned_by_another_worker','item':item}
        if skip_completed and (Path(output)/'partitions'/item/'tactile/manifest.json').exists():
            return {'status':'already_published','item':item}
        lock.seek(0);lock.truncate();lock.write(json.dumps({'pid':os.getpid(),'host':os.uname().nodename}));lock.flush()
        if kind=='sharpa':return extract_partition(Path(output)/'partitions'/item)
        return stage_bucket(Path(source),Path(output),item)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['sharpa','egodex'],required=True);p.add_argument('--source');p.add_argument('--output',required=True)
    p.add_argument('--shard-index',type=int,required=True);p.add_argument('--shard-count',type=int,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--pending-only',action='store_true',help='Sharpa only: skip published manifests before scheduling and again under the shared lock')
    a=p.parse_args()
    if not 0<=a.shard_index<a.shard_count:p.error('shard-index must lie in [0, shard-count)')
    if a.pending_only and a.kind!='sharpa':p.error('--pending-only applies only to Sharpa')
    root=Path(a.output)
    if a.kind=='sharpa':items=[str(p.relative_to(root/'partitions')) for p in sorted((root/'partitions').glob('*/*')) if p.is_dir()]
    else:
        if not a.source:p.error('EgoDex requires --source')
        items=sorted(p.name for p in (Path(a.source)/'train').iterdir() if p.is_dir())
    selected=items[a.shard_index::a.shard_count]
    if a.pending_only:selected=[item for item in selected if not (root/'partitions'/item/'tactile/manifest.json').exists()]
    errors=[]
    print(json.dumps({'kind':a.kind,'shard_index':a.shard_index,'shard_count':a.shard_count,'tasks':len(selected),'workers':a.workers,'pending_only':a.pending_only}),flush=True)
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        futures={pool.submit(job,a.kind,a.source,a.output,item,a.pending_only):item for item in selected}
        for f in concurrent.futures.as_completed(futures):
            try:print(json.dumps(f.result()),flush=True)
            except Exception as e:
                record={'item':futures[f],'status':'error','error':repr(e)};errors.append(record);print(json.dumps(record),flush=True)
    if errors:raise SystemExit(1)
