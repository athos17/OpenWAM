"""Compare the s7d formal run with the 2026-10-22 8-node run step by step (CPU only).

    python3 compare_with_baseline.py [--window 500]

Both runs log every step to <run>/metrics.jsonl (rank-0 sample ids, LR, losses, step time).
Same seed and data order, so equal sample ids at a step mean the same batch. Step time is the
per-step wall time (1 / steps_per_sec); steps over 3x the median (checkpoint saves) are dropped.
Writes results/baseline_comparison.json.
"""
import argparse,json,statistics
from pathlib import Path
R=Path('/mnt/pfs/v89atp/workspace/setup/action_random_64gpu_s7d_20261026')
BASELINE=Path('/mnt/pfs/v89atp/workspace/outputs/action_random_64gpu/2026-10-22_22-50-01')
METRICS=('loss_total','loss_video','loss_action','grad_norm')
p=argparse.ArgumentParser();p.add_argument('--window',type=int,default=500);a=p.parse_args()

def load(run):
    out={}
    with (run/'metrics.jsonl').open() as f:
        for line in f:
            try:r=json.loads(line)
            except ValueError:continue  # a partially written last line
            out[r['step']]=r
    return out
new_dir=Path(json.loads((R/'results/formal_rank0.json').read_text())['output_path'])
old,new=load(BASELINE),load(new_dir)
steps=sorted(set(old)&set(new))
same_ids=sum(old[k]['sample_ids']==new[k]['sample_ids'] for k in steps)
same_lr=sum(abs(old[k]['lr']-new[k]['lr'])<=1e-12*max(abs(old[k]['lr']),1e-30) for k in steps)
first_diff=next((k for k in steps if old[k]['sample_ids']!=new[k]['sample_ids']),None)
print(f'baseline {BASELINE.name} ({len(old)} steps) vs s7d {new_dir.name} ({len(new)} steps)')
print(f'common steps {len(steps)}: identical rank-0 sample ids {same_ids}, identical LR {same_lr}, first differing batch {first_diff}')

result={'baseline':str(BASELINE),'run':str(new_dir),'common_steps':len(steps),'identical_sample_ids':same_ids,
        'identical_lr':same_lr,'first_differing_batch':first_diff,'windows':[]}
print(f"\nwindow means (window {a.window}); rel = s7d vs baseline")
print('window'.ljust(14)+''.join(f'{m:>34s}' for m in METRICS))
for i in range(0,len(steps),a.window):
    chunk=steps[i:i+a.window];row={'from':chunk[0],'to':chunk[-1]};line=f'{chunk[0]:>6d}-{chunk[-1]:<7d}'
    for m in METRICS:
        mo=statistics.fmean(old[k]['metrics'][m] for k in chunk);mn=statistics.fmean(new[k]['metrics'][m] for k in chunk)
        rel=(mn-mo)/abs(mo) if mo else 0.0;row[m]={'baseline':mo,'s7d':mn,'rel':rel}
        line+=f'{mo:12.5f} {mn:11.5f} {rel:+8.2%}'
    result['windows'].append(row);print(line)

def step_times(run,keys):
    t=[1/run[k]['steps_per_sec'] for k in keys if k>20 and run[k]['steps_per_sec']>0]
    med=statistics.median(t) if t else None
    return med,[x for x in t if x<=3*med] if t else []
for name,run in (('baseline',old),('s7d',new)):
    med,kept=step_times(run,steps)
    if not kept:continue
    q=statistics.quantiles(kept,n=10)
    result[name+'_step_s']={'median':statistics.median(kept),'p10':q[0],'p90':q[-1],'mean':statistics.fmean(kept),'steps':len(kept)}
print()
for name in ('baseline','s7d'):
    s=result.get(name+'_step_s')
    if s:print(f"{name:9s} step time over common steps: median {s['median']:.2f} s  p10 {s['p10']:.2f}  p90 {s['p90']:.2f}  mean {s['mean']:.2f} (saves dropped)")
if result.get('baseline_step_s') and result.get('s7d_step_s'):
    result['speedup']=result['baseline_step_s']['median']/result['s7d_step_s']['median']
    print(f"speedup (median step) {result['speedup']:.3f}x")
timing=new_dir/'step_timing.jsonl'
if timing.exists():
    rows=[json.loads(l) for l in timing.read_text().splitlines() if l.strip()][1:]
    if rows:
        med=lambda f:statistics.median(f(r) for r in rows)
        result['s7d_timing']={'iteration_s':med(lambda r:r['stages']['iteration_cpu_s']['median_of_ranks']),
            'mfu':med(lambda r:r['mfu']),'mfu_incl_encoders':med(lambda r:r['mfu_incl_encoders']),
            'peak_reserved_gib':max(r['stages']['peak_reserved_gib']['rank_max'] for r in rows)}
        t=result['s7d_timing'];print(f"s7d step timing: {t['iteration_s']:.2f} s/step, MFU {t['mfu']:.1%} (incl. encoders {t['mfu_incl_encoders']:.1%}), peak reserved {t['peak_reserved_gib']:.1f} GiB")
(R/'results/baseline_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
