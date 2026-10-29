"""Sample node CPU / memory and GPU usage every 15 s (shows whether the DataLoader is CPU-bound)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from runtime import ROOT  # noqa: E402

phase = sys.argv[1]
rank = int(os.environ['OPENWAM_NODE_RANK'])
previous = None
while True:
    record = {'time': time.time(), 'node_rank': rank, 'phase': phase}
    try:
        fields = ['index', 'utilization.gpu', 'memory.used', 'memory.total', 'power.draw']
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'], text=True, timeout=10)
        record['gpus'] = [dict(zip(fields, [float(x.strip()) for x in line.split(',')]))
                          for line in output.strip().splitlines()]
        for base in (Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/memory')):
            for name in ('memory.current', 'memory.max', 'memory.usage_in_bytes', 'memory.limit_in_bytes'):
                if (base / name).exists():
                    record[name] = (base / name).read_text().strip()
        cpu = Path('/sys/fs/cgroup/cpu.stat')
        if cpu.exists():
            stats = dict(line.split() for line in cpu.read_text().splitlines())
            current = (time.monotonic(), int(stats.get('usage_usec', 0)))
            if previous:
                record['cpu_cores_used'] = (current[1] - previous[1]) / 1e6 / (current[0] - previous[0])
            previous = current
        if Path('/sys/fs/cgroup/cpu.max').exists():
            record['cpu.max'] = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
        record['loadavg'] = Path('/proc/loadavg').read_text().strip()
    except Exception as exc:
        record['error'] = str(exc)
    with (ROOT / 'results' / f'{phase}_resources_node{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    time.sleep(15)
