"""Compare the base and s7d runs: same data per step, loss curves, speed and MFU.

    python3 compare_runs.py [--window 50] [--skip 10]

Reads each run dir from results/<variant>_rank0.json: metrics.jsonl (rank-0 sample ids),
debug_loss_history.csv (per-step losses) and step_timing.jsonl (timing / MFU summaries).
Writes results/comparison.json.
"""
import argparse,csv,json,statistics
from pathlib import Path
R=Path('/mnt/pfs/v89atp/workspace/setup/accel_cmp_32gpu_20261026')
METRICS=('loss','loss_video','loss_action','grad_norm')
p=argparse.ArgumentParser();p.add_argument('--window',type=int,default=50);p.add_argument('--skip',type=int,default=10)
p.add_argument('--timing-from',type=int,default=50,help='ignore timing summaries before this step');a=p.parse_args()

runs={}
for v in ('base','s7d'):
    rep=json.loads((R/'results'/f'{v}_rank0.json').read_text());out=Path(rep['output_path'])
    with (out/'debug_loss_history.csv').open() as f:
        loss={int(r['step']):{m:float(r[m]) for m in METRICS if r.get(m) not in (None,'')} for r in csv.DictReader(f)}
    samples={}
    for line in (out/'metrics.jsonl').read_text().splitlines():
        r=json.loads(line);samples[r['step']]=(r['sample_ids'],r['lr'])
    timing=[json.loads(l) for l in (out/'step_timing.jsonl').read_text().splitlines()] if (out/'step_timing.jsonl').exists() else []
    peaks=[json.loads(q.read_text()) for q in (R/'results').glob(f'{v}_rank*.json')]
    runs[v]={'dir':str(out),'loss':loss,'samples':samples,'timing':timing,'status':rep['status'],
             'peak_alloc_GiB':max(r['peak_gpu_memory_bytes'] for r in peaks)/2**30,
             'peak_reserved_GiB':max(r.get('peak_gpu_reserved_bytes',0) for r in peaks)/2**30}
b,s=runs['base'],runs['s7d']

common=sorted(set(b['samples'])&set(s['samples']))
same_data=sum(b['samples'][k][0]==s['samples'][k][0] for k in common)
same_lr=sum(b['samples'][k][1]==s['samples'][k][1] for k in common)
print(f"steps in both runs: {len(common)}; identical rank-0 sample ids: {same_data}; identical LR: {same_lr}")

steps=[k for k in sorted(set(b['loss'])&set(s['loss'])) if k>a.skip]
result={'common_steps':len(common),'identical_sample_ids':same_data,'identical_lr':same_lr,'windows':[],'gap':{}}
if steps:
    print(f"\nwindow means, steps {steps[0]}..{steps[-1]} (window {a.window})")
    print('window'.ljust(12)+''.join(f'{m+" base":>18s}{m+" s7d":>16s}{"rel":>9s}' for m in METRICS[:3]))
    gaps={m:[] for m in METRICS}
    for i in range(0,len(steps),a.window):
        chunk=steps[i:i+a.window];row={'from':chunk[0],'to':chunk[-1]};line=f'{chunk[0]:>5d}-{chunk[-1]:<6d}'
        for m in METRICS:
            mb=statistics.fmean(b['loss'][k][m] for k in chunk);ms=statistics.fmean(s['loss'][k][m] for k in chunk)
            rel=(ms-mb)/abs(mb) if mb else 0.0;gaps[m].append(abs(rel));row[m]={'base':mb,'s7d':ms,'rel':rel}
            if m!='grad_norm':line+=f'{mb:18.5f}{ms:16.5f}{rel:9.2%}'
        result['windows'].append(row);print(line)
    for m in METRICS:
        result['gap'][m]={'mean_abs_rel':statistics.fmean(gaps[m]),'max_abs_rel':max(gaps[m])}
    print('\nmean / max |relative gap| of window means: '+', '.join(f"{m} {result['gap'][m]['mean_abs_rel']:.2%} / {result['gap'][m]['max_abs_rel']:.2%}" for m in METRICS))

def speed(run):
    rows=[t for t in run['timing'] if t['step']>=a.timing_from] or run['timing']
    if not rows:return {}
    med=lambda xs:statistics.median(x for x in xs if x is not None) if any(x is not None for x in xs) else None
    return {'summaries':len(rows),'iteration_s':med([t['stages'].get('iteration_cpu_s',{}).get('median_of_ranks') for t in rows]),
            'samples_per_sec':med([t.get('samples_per_sec') for t in rows]),'mfu':med([t.get('mfu') for t in rows]),
            'mfu_incl_encoders':med([t.get('mfu_incl_encoders') for t in rows]),
            'model_tflop_per_step':med([t['stages'].get('model_tflop_per_step',{}).get('median_of_ranks') for t in rows])}
print()
for v,run in runs.items():
    sp=speed(run);result[v]={'status':run['status'],'dir':run['dir'],'speed':sp,
        'peak_alloc_GiB':round(run['peak_alloc_GiB'],2),'peak_reserved_GiB':round(run['peak_reserved_GiB'],2)}
    print(f"{v:5s} {run['status']:8s} "+(f"{sp['iteration_s']:.2f} s/step  {sp['samples_per_sec']:.1f} samples/s  MFU {sp['mfu']:.1%}  (incl. encoders {sp['mfu_incl_encoders']:.1%})  " if sp.get('mfu') is not None else '')
          +f"peak alloc {run['peak_alloc_GiB']:.1f} GiB, reserved {run['peak_reserved_GiB']:.1f} GiB")
if result['base']['speed'].get('iteration_s') and result['s7d']['speed'].get('iteration_s'):
    result['speedup']=result['base']['speed']['iteration_s']/result['s7d']['speed']['iteration_s']
    print(f"speedup {result['speedup']:.3f}x")
(R/'results/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
