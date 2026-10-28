#!/usr/bin/env python3
"""Publish fail-closed native readiness after conversion/staging is complete.

A pending publication is deliberately not train-ready. This script never edits
source data or in-place hardlinks; JSON records are atomically replaced. Run
again after interruption, or use --wait while the authorized converters run.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(v,indent=2)+'\n'
    if p.exists() and p.read_text()==text:return
    tmp=p.with_suffix(f'.json.partial.{os.getpid()}');tmp.write_text(text);os.replace(tmp,p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stamp_definitions(root,kind,definition_path,infos):
    definition=read(definition_path)
    for ip in infos:
        info=read(ip);effective={**definition,'bucket':str(ip.parent.parent.relative_to(root)),'fps':info['fps'],
          'source_info_sha256':digest(ip),'native_robot_type':info.get('robot_type'),'native_openwam_contract':info.get('openwam_contract'),
          'actual_features':sorted(info['features'])}
        if kind=='agibot':
            effective['embodiment']=info.get('openwam_contract',{}).get('embodiment',info.get('robot_type'))
            effective['end_effector']=info.get('openwam_contract',{}).get('end_effector','feature-driven reader')
            effective['base_validity']='Original agibotworld reader feature/statistics-driven rule; no override from this definition'
        write(ip.parent/'source_definition.json',effective)
    return definition

def finalize(root,definitions):
    reports={}
    for kind in ('agibot','sharpadex'):
        base=root/kind;pub=base/'publication.json'
        if not pub.exists():reports[kind]={'status':'not_published'};continue
        old=read(pub)
        if old.get('status')=='complete':reports[kind]=old;continue
        infos=sorted(base.glob('*/meta/info.json')) if kind=='agibot' else sorted(base.glob('partitions/*/*/meta/info.json'))
        expected=75 if kind=='agibot' else 495
        if len(infos)!=expected:raise ValueError(f'{kind} expected {expected} buckets, got {len(infos)}')
        stamp_definitions(base,kind,definitions/f'{kind}.json',infos)
        report={**old,'status':'pending_tactile' if kind=='sharpadex' else 'complete','buckets_expected':expected,'buckets_completed':0,
                'episodes':sum(read(p)['total_episodes'] for p in infos),'frames':sum(read(p)['total_frames'] for p in infos),
                'hours':sum(read(p)['total_frames']/read(p)['fps']/3600 for p in infos),
                'source_definition_sha256':digest(definitions/f'{kind}.json'),'sampling':'not selected; full published pool available',
                'file_verification':'every hardlink inode+size; native source validated previously; unchanged source metadata identity',
                'conversion_scope':'existing native selection; no new resampling or episode filtering'}
        if kind=='agibot':report['buckets_completed']=expected
        else:
            manifests=[p.parent.parent/'tactile/manifest.json' for p in infos]
            report['buckets_completed']=sum(p.exists() for p in manifests)
            if report['buckets_completed']==expected:
                records=[];valid_frames=0;total_frames=0
                for ip,mp in zip(infos,manifests):
                    m=read(mp)
                    if m['dataset_info_sha256']!=digest(ip) or m['codec']!='tactile_intensity_v1':raise ValueError(f'Invalid tactile identity {mp}')
                    for ep,e in m['episodes'].items():
                        maps=np.load(mp.parent/e['maps'],mmap_mode='r',allow_pickle=False);valid=np.load(mp.parent/e['valid'],mmap_mode='r',allow_pickle=False);ts=np.load(mp.parent/e['timestamps'],mmap_mode='r',allow_pickle=False)
                        if maps.dtype!=np.uint8 or maps.shape[1:]!=(10,32,32) or valid.shape!=maps.shape[:2] or valid.dtype!=np.bool_ or ts.shape!=(len(maps),):raise ValueError(f'Invalid sidecar arrays {mp}:{ep}')
                        if not np.allclose(ts,np.arange(len(ts))/30,atol=2e-5,rtol=0):raise ValueError(f'Invalid timestamps {mp}:{ep}')
                        valid_frames+=int(valid.all(axis=1).sum());total_frames+=len(maps)
                    records.append({'path':str(mp.relative_to(base)),'sha256':digest(mp),'decode_errors':m.get('decode_errors',[])})
                if total_frames!=report['frames']:raise ValueError('Tactile total frames differ from LeRobot')
                report.update(status='complete',tactile_frames=total_frames,tactile_valid_frames=valid_frames,tactile_manifests=records,
                              tactile_validation='all array shapes/dtypes, episode frame counts, strictly aligned times and valid counts read back')
        write(pub,report);reports[kind]=report
    base=root/'egodex';catalog=base/'full_pool/catalog.json'
    if catalog.exists():
        c=read(catalog)
        # Independent allocation shards write per-bucket receipts. Aggregate
        # them without trusting one launcher's process-local completed list.
        receipts=[]
        expected={x['bucket'] for x in c['inventory']}
        for rp in (base/'full_pool/receipts').glob('*.json'):
            if rp.stem not in expected:continue
            rr=read(rp)
            if rr.get('status')=='complete':receipts.append({k:v for k,v in rr.items() if k!='files'})
        c['ready']=receipts
        if len(receipts)==c['buckets_expected']:
            if sum(r['frames'] for r in receipts)!=c['total_frames'] or sum(r['episodes'] for r in receipts)!=c['total_episodes']:raise ValueError('EgoDex receipt totals differ')
            c['status']='complete';c['errors']=[];write(catalog,c)
        report={'schema':'touchwam_native_publication_v1','status':'complete' if c['status']=='complete' else 'pending_copy',
          'source':c['source'],'source_revision':c['source_revision'],'buckets_expected':c['buckets_expected'],'buckets_completed':len(c['ready']),
          'episodes':c['total_episodes'],'frames':c['total_frames'],'hours':c['total_hours'],'official_train_only':True,
          'sampling':'all official training episodes; sampling hours deferred','excluded_derivative':'accepted_cache_300h; same origin, mutually exclusive',
          'file_verification':'SHA256 while copying + independent destination readback; per-file receipts; all episode lengths and parquet row totals',
          'catalog_sha256':digest(catalog),'errors':c['errors']}
        if c['status']=='complete':
            infos=sorted((base/'full_pool/train').glob('*/meta/info.json'));stamp_definitions(base/'full_pool','egodex',definitions/'egodex.json',infos)
            write(base/'full_pool/publication.json',report)
        write(base/'publication.json',report);reports['egodex']=report
    cache=base/'accepted_cache_300h/meta/info.json'
    if cache.exists():stamp_definitions(base/'accepted_cache_300h','egodex',definitions/'egodex.json',[cache])
    summary={k:{f:v for f,v in r.items() if f in ('status','buckets_expected','buckets_completed','episodes','frames','hours')} for k,r in reports.items()}
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--definitions',required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    while True:
        result=finalize(Path(a.root),Path(a.definitions));print(json.dumps(result),flush=True)
        if len(result)==3 and all(r.get('status')=='complete' for r in result.values()):break
        if not a.wait:raise SystemExit(2)
        time.sleep(60)
