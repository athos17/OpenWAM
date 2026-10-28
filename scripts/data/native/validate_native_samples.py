#!/usr/bin/env python3
"""Read real published samples and compare tactile maps to source video PTS."""
import argparse,json
from pathlib import Path
import av,numpy as np,pyarrow.parquet as pq
from extract_sharpa_deform import split_video_frame,REGIONS,KEY


def validate_sharpa(root,count):
    manifests=sorted(root.glob('partitions/*/*/tactile/manifest.json'))
    selected=manifests[::max(1,len(manifests)//count)][:count];results=[]
    for mp in selected:
        m=json.loads(mp.read_text());p=mp.parent.parent
        if m['region_names']!=REGIONS:raise ValueError('Region ordering differs')
        for ep,e in list(m['episodes'].items())[:2]:
            maps=np.load(mp.parent/e['maps'],mmap_mode='r');valid=np.load(mp.parent/e['valid']);times=np.load(mp.parent/e['timestamps'])
            rows=sorted({0,len(maps)//2,len(maps)-1});maxdiff=0
            with av.open(str(p/e['source_video'])) as c:
                s=c.streams.video[0];s.thread_count=1
                for row in rows:
                    target=e['source_video_from_timestamp']+row/30;c.seek(int(max(0,target-0.1)/s.time_base),stream=s)
                    matched=False
                    for frame in c.decode(s):
                        t=float(frame.pts*s.time_base)
                        if t>target+1e-4:break
                        if abs(t-target)<=1e-4:
                            expected=split_video_frame(frame);diff=int(np.max(np.abs(expected.astype(int)-maps[row].astype(int))))
                            maxdiff=max(maxdiff,diff);matched=True;break
                    if matched:
                        if not valid[row].all() or maxdiff:raise ValueError(f'Raw video vs sidecar mismatch {mp}:{ep}:{row}')
                    elif valid[row].any():raise ValueError('Unavailable source frame marked valid')
                    if abs(times[row]-row/30)>2e-5:raise ValueError('Episode clock differs')
            results.append({'partition':str(p.relative_to(root)),'episode':int(ep),'rows':rows,'max_uint8_error':maxdiff})
    return results


def validate_egodex_cache(root,count):
    info=json.loads((root/'meta/info.json').read_text());forbidden=['action','observation.state']
    if any(f in info['features'] for f in forbidden):raise ValueError('Ego video release has robot labels')
    files=sorted(root.glob('meta/episodes/*/*.parquet'));results=[]
    for epfile in files[::max(1,len(files)//count)][:count]:
        ep=pq.read_table(epfile).slice(0,1).to_pylist()[0];key='observation.images.camera';prefix=f'videos/{key}/'
        path=root/info['video_path'].format(video_key=key,chunk_index=ep[prefix+'chunk_index'],file_index=ep[prefix+'file_index'])
        with av.open(str(path)) as c:
            s=c.streams.video[0];s.thread_count=1;target=ep[prefix+'from_timestamp'];c.seek(int(max(0,target-0.1)/s.time_base),stream=s)
            for frame in c.decode(s):
                if abs(float(frame.pts*s.time_base)-target)<1e-4:break
                if float(frame.pts*s.time_base)>target+1e-4:raise ValueError('Video from_timestamp unavailable')
            else:raise ValueError('Video cannot be decoded at source offset')
        results.append({'episode':ep['episode_index'],'video_from_timestamp':target,'decoded_shape':[frame.height,frame.width],'source_episode_id':ep['source_episode_id']})
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--buckets',type=int,default=5);a=p.parse_args();root=Path(a.root)
    report={'sharpa_samples':validate_sharpa(root/'sharpadex',a.buckets),'egodex_samples':validate_egodex_cache(root/'egodex/accepted_cache_300h',a.buckets),
            'scope':'sampled real byte/PTS/mapping readback; full conversion readiness is separate publication gate','status':'passed'}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
