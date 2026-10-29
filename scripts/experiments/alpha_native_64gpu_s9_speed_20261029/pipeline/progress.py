"""Progress from the shared files only (no cluster credentials, no GPU imports)."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from runtime import ROOT, settings  # noqa: E402

reports = [json.loads(p.read_text()) for p in sorted((ROOT / 'results').glob('speed_rank*.json'))]
nccl = len(list((ROOT / 'results').glob('nccl_rank*.json')))
if not reports:
    print(json.dumps({'nccl_ranks_passed': nccl, 'speed_reports': 0}))
    raise SystemExit
rank0 = ROOT / 'results/speed_rank0.json'
metrics = Path(json.loads(rank0.read_text())['output_path']) / 'metrics.jsonl'
last = json.loads(metrics.read_text().splitlines()[-1]) if metrics.exists() else {}
print(json.dumps({
    'nccl_ranks_passed': nccl, 'reports': len(reports), 'passed': sum(r['status'] == 'passed' for r in reports),
    'step': last.get('step'), 'of': settings()['steps'], 'metrics': last.get('metrics'),
    'last_step_s': round(1 / last['steps_per_sec'], 2) if last else None,
    'seconds_since_update': round(time.time() - rank0.stat().st_mtime, 1),
    'peak_gpu_reserved_GiB': round(max(r['peak_gpu_reserved_bytes'] for r in reports) / 2**30, 2),
}, ensure_ascii=False))
