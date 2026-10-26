"""Compare the comparison runs against base: same data per step, loss curves, speed and MFU.

    python3 compare_runs.py [--variants base s7d s7d_vae s7e_vae] [--window 50] [--skip 10]

Reads each run dir from results/<variant>_rank0.json: metrics.jsonl (rank-0 sample ids),
debug_loss_history.csv (per-step losses) and step_timing.jsonl (timing / MFU summaries).
Losses are compared on the steps every run reached. Writes results/comparison.json.
"""
import argparse,csv,json,statistics
from pathlib import Path
R=Path('/mnt/pfs/v89atp/workspace/setup/accel_cmp_32gpu_20261026')
METRICS=('loss','loss_video','loss_action','grad_norm')
p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',default=['base','s7d','s7d_vae','s7e_vae'])
p.add_argument('--window',type=int,default=50);p.add_argument('--skip',type=int,default=0)
p.add_argument('--timing-from',type=int,default=50,help='ignore timing summaries before this step');a=p.parse_args()

runs={}
for v in a.variants:
    if not (R/'results'/f'{v}_rank0.json').exists():continue
    rep=json.loads((R/'results'/f'{v}_rank0.json').read_text());out=Path(rep['output_path'])
    with (out/'debug_loss_history.csv').open() as f:
        loss={int(r['step']):{m:float(r[m]) for m in METRICS if r.get(m) not in (None,'')} for r in csv.DictReader(f)}
    samples={}
    for line in (out/'metrics.jsonl').read_text().splitlines():
        r=json.loads(line);samples[r['step']]=(r['sample_ids'],r['lr'])
    timing=[json.loads(l) for l in (out/'step_timing.jsonl').read_text().splitlines()] if (out/'step_timing.jsonl').exists() else []
    peaks=[json.loads(q.read_text()) for q in (R/'results').glob(f'{v}_rank*.json')]
    runs[v]={'dir':str(out),'loss':loss,'samples':samples,'timing':timing,'status':rep['status'],'last_step':max(loss) if loss else 0,
             'peak_alloc_GiB':max(r['peak_gpu_memory_bytes'] for r in peaks)/2**30,
             'peak_reserved_GiB':max(r.get('peak_gpu_reserved_bytes',0) for r in peaks)/2**30}
names=list(runs);ref=names[0];b=runs[ref]
result={'reference':ref,'runs':{}}

common=sorted(set.intersection(*(set(r['samples']) for r in runs.values())))
steps=[k for k in sorted(set.intersection(*(set(r['loss']) for r in runs.values()))) if k>a.skip]
print(f"reference {ref}; steps reached by every run: {len(common)} (loss steps {steps[0]}..{steps[-1]})")
for v in names[1:]:
    r=runs[v]
    same_data=sum(b['samples'][k][0]==r['samples'][k][0] for k in common)
    same_lr=sum(b['samples'][k][1]==r['samples'][k][1] for k in common)
    result['runs'][v]={'identical_sample_ids':same_data,'identical_lr':same_lr,'common_steps':len(common)}
    print(f"  {v}: identical rank-0 sample ids {same_data}/{len(common)}, identical LR {same_lr}/{len(common)}")

windows=[steps[i:i+a.window] for i in range(0,len(steps),a.window)]
mean=lambda run,m,chunk:statistics.fmean(run['loss'][k][m] for k in chunk)
for m in METRICS[:3]:
    print(f"\n{m}: window means (relative gap to {ref})")
    print('window'.ljust(12)+''.join(f'{v:>22s}' for v in names))
    for chunk in windows:
        mb=mean(b,m,chunk)
        print(f'{chunk[0]:>5d}-{chunk[-1]:<6d}'+f'{mb:22.5f}'+''.join(f'{mean(runs[v],m,chunk):13.5f} ({(mean(runs[v],m,chunk)-mb)/abs(mb):+7.2%})' for v in names[1:]))
print(f'\nmean / max |relative gap| of window means vs {ref}, and of single steps')
for v in names[1:]:
    g={}
    for m in METRICS:
        rel=[abs(mean(runs[v],m,c)-mean(b,m,c))/max(abs(mean(b,m,c)),1e-12) for c in windows]
        step_rel=[abs(runs[v]['loss'][k][m]-b['loss'][k][m])/max(abs(b['loss'][k][m]),1e-12) for k in steps]
        g[m]={'window_mean':statistics.fmean(rel),'window_max':max(rel),'step_median':statistics.median(step_rel),'step_max':max(step_rel)}
    result['runs'][v]['gap']=g
    print(f'  {v:8s} '+'  '.join(f"{m} {g[m]['window_mean']:.2%}/{g[m]['window_max']:.2%} (step med {g[m]['step_median']:.2%})" for m in METRICS))

def speed(run):
    rows=[t for t in run['timing'] if t['step']>=a.timing_from] or run['timing']
    if not rows:return {}
    def med(key,stage=True):
        xs=[(t['stages'].get(key,{}).get('median_of_ranks') if stage else t.get(key)) for t in rows]
        xs=[x for x in xs if x is not None]
        return statistics.median(xs) if xs else None
    return {'summaries':len(rows),'iteration_s':med('iteration_cpu_s'),'samples_per_sec':med('samples_per_sec',False),
            'mfu':med('mfu',False),'mfu_incl_encoders':med('mfu_incl_encoders',False),
            'model_tflop_per_step':med('model_tflop_per_step'),
            **{k:med(k+'_gpu_s') for k in ('encode_text','encode_video','prepare_inputs','compute_loss','ds_backward','ds_step')}}
print(f"\nspeed (median of {a.timing_from}+ step summaries)")
print(f"{'run':9s}{'status':9s}{'steps':>6s}{'s/step':>8s}{'speedup':>9s}{'samples/s':>10s}{'MFU':>7s}{'MFU+enc':>8s}{'text':>6s}{'vae':>6s}{'fwd':>6s}{'bwd':>6s}{'alloc':>7s}{'resv':>7s}")
base_it=None
for v,run in runs.items():
    sp=speed(run);result['runs'].setdefault(v,{}).update({'status':run['status'],'dir':run['dir'],'last_step':run['last_step'],'speed':sp,
        'peak_alloc_GiB':round(run['peak_alloc_GiB'],2),'peak_reserved_GiB':round(run['peak_reserved_GiB'],2)})
    if not sp.get('iteration_s'):print(v,'no timing yet');continue
    base_it=base_it or sp['iteration_s'];sp['speedup']=base_it/sp['iteration_s']
    fwd=sp['compute_loss']-sp['prepare_inputs']
    print(f"{v:9s}{run['status']:9s}{run['last_step']:6d}{sp['iteration_s']:8.2f}{sp['speedup']:8.3f}x{sp['samples_per_sec']:10.1f}{sp['mfu']:7.1%}{sp['mfu_incl_encoders']:8.1%}"
          f"{sp['encode_text']:6.2f}{sp['encode_video']:6.2f}{fwd:6.2f}{sp['ds_backward']:6.2f}{run['peak_alloc_GiB']:7.1f}{run['peak_reserved_GiB']:7.1f}")
(R/'results/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
