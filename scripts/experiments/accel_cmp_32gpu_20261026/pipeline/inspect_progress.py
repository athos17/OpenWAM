"""Read shared-file progress; no cluster credentials or GPU imports required."""
import json,time
from pathlib import Path
R=Path('/mnt/pfs/v89atp/workspace/setup/accel_cmp_32gpu_20261026')
for variant in ('base','s7d','s7d_vae','s7e_vae'):
    reports=[json.loads(p.read_text()) for p in sorted((R/'results').glob(f'{variant}_rank*.json'))]
    if not reports:continue
    steps=[r['history'][-1]['step'] if r['history'] else 0 for r in reports]
    rank0=json.loads((R/'results'/f'{variant}_rank0.json').read_text())
    last=rank0['history'][-1] if rank0['history'] else {}
    timing={}
    path=Path(rank0['output_path'])/'step_timing.jsonl'
    if path.exists():
        lines=path.read_text().splitlines()
        if lines:
            s=json.loads(lines[-1])
            timing={k:s.get(k) for k in ('step','samples_per_sec','mfu','mfu_incl_encoders')}
            timing['iteration_s']=s['stages'].get('iteration_cpu_s',{}).get('median_of_ranks')
    print(json.dumps({'variant':variant,'reports':len(reports),'passed':sum(r['status']=='passed' for r in reports),
        'seconds_since_rank0_update':round(time.time()-(R/'results'/f'{variant}_rank0.json').stat().st_mtime,1),
        'step_range':[min(steps),max(steps)],'max_steps':rank0.get('max_steps'),'lr':last.get('lr'),
        'metrics':last.get('metrics'),'timing':timing,'elapsed_seconds':round(rank0['elapsed_seconds']),
        'peak_gpu_GiB':round(max(r['peak_gpu_memory_bytes'] for r in reports)/2**30,2),
        'peak_reserved_GiB':round(max(r.get('peak_gpu_reserved_bytes',0) for r in reports)/2**30,2),
        'output_path':rank0['output_path']},ensure_ascii=False))
