"""Read shared-file progress; no cluster credentials or GPU imports required."""
import json,time
from pathlib import Path
R=Path('/mnt/pfs/v89atp/workspace/setup/action_random_64gpu_s7d_20261026')
for phase in ('smoke','resume','formal'):
    reports=[json.loads(p.read_text()) for p in sorted((R/'results').glob(f'{phase}_rank*.json'))]
    if not reports:continue
    steps=[r['history'][-1]['step'] if r['history'] else 0 for r in reports]
    rank0=json.loads((R/'results'/f'{phase}_rank0.json').read_text())
    last=rank0['history'][-1] if rank0['history'] else {}
    print(json.dumps({'phase':phase,'reports':len(reports),'passed':sum(r['status']=='passed' for r in reports),
        'seconds_since_rank0_update':round(time.time()-(R/'results'/f'{phase}_rank0.json').stat().st_mtime,1),
        'step_range':[min(steps),max(steps)],'optimizer_step':last.get('optimizer_step'),
        'metrics':last.get('metrics'),'elapsed_seconds':rank0['elapsed_seconds'],
        'peak_gpu_GiB':round(max(r['peak_gpu_memory_bytes'] for r in reports)/2**30,2),
        'output_path':rank0['output_path']},ensure_ascii=False))
