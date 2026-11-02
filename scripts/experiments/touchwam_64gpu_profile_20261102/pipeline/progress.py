"""Progress from the shared files only (no cluster credentials, no GPU imports)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runtime import PHASE, ROOT, output_root  # noqa: E402

results = ROOT / 'results'
report = {'nodes_started': len(list(results.glob(f'{PHASE}_node*.json'))),
          'nccl_ranks_passed': len(list(results.glob('nccl_rank*.json'))),
          'completed': len(list(results.glob('completed_node*.json'))),
          'failures': [json.loads(p.read_text()) for p in results.glob('failure_node*.json')]}
runs = sorted(output_root().glob('*/config.yaml')) if output_root().exists() else []
if runs:
    run = runs[-1].parent
    report['run_dir'] = str(run)
    metrics = run / 'training_metrics.jsonl'
    if metrics.exists():
        lines = metrics.read_text().splitlines()
        report['last_metrics'] = json.loads(lines[-1]) if lines else None
        report['seconds_since_update'] = round(time.time() - metrics.stat().st_mtime, 1)
print(json.dumps(report, ensure_ascii=False, indent=2))
