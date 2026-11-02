"""AIHC node entry: 8 nodes x 8 ranks; NCCL + gloo probe, then the profile run from the frozen release."""
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from runtime import PHASE, ROOT, config_name, require_preflight, write_json  # noqa: E402

rank = int(os.environ.get('NODE_RANK', os.environ.get('RANK', '-1')))
assert int(os.environ['WORLD_SIZE']) == 8 and 0 <= rank < 8
require_preflight()
os.environ['OPENWAM_NODE_RANK'] = str(rank)
write_json(ROOT / 'results' / f'{PHASE}_node{rank}.json', {'host': socket.gethostname(), 'node_rank': rank})
torchrun = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=8', '--nproc-per-node=8',
            f'--node-rank={rank}', '--master-addr=' + os.environ['MASTER_ADDR'],
            '--master-port=' + os.environ.get('MASTER_PORT', '29500'), '--max-restarts=0']
os.chdir(ROOT / 'release')
monitor = subprocess.Popen([sys.executable, str(ROOT / 'pipeline/node_monitor.py'), PHASE])
try:
    with (ROOT / 'results' / f'{PHASE}_node{rank}.log').open('a', buffering=1) as log:
        subprocess.run(torchrun + [str(ROOT / 'pipeline/nccl_probe.py')], check=True, timeout=600,
                       stdout=log, stderr=subprocess.STDOUT)
        # 45 steps take about half an hour; a hung collective must not hold 64 GPUs indefinitely.
        subprocess.run(torchrun + ['scripts/train.py', '--config-name', config_name()], check=True, timeout=2 * 3600,
                       stdout=log, stderr=subprocess.STDOUT)
    write_json(ROOT / 'results' / f'completed_node{rank}.json', {'status': 'completed'})
except BaseException as exc:
    write_json(ROOT / 'results' / f'failure_node{rank}.json', {'status': 'failed', 'error': repr(exc)})
    raise
finally:
    monitor.terminate()
    monitor.wait(timeout=15)
