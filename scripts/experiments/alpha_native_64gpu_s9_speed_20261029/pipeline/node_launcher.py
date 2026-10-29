"""AIHC node entry: 8 nodes x 8 ranks; NCCL probe, then the speed run, from the frozen release."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from runtime import ROOT, verify_release, write_json  # noqa: E402

phase = sys.argv[1]
if phase != 'speed':
    raise ValueError(phase)
keys = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'NODE_RANK', 'NCCL_IB_DISABLE']
env = {key: os.environ.get(key) for key in keys}
rank = int(os.environ.get('NODE_RANK', os.environ.get('RANK', '-1')))
nodes = int(os.environ.get('WORLD_SIZE', '0'))
if nodes != 8 or not 0 <= rank < 8 or not os.environ.get('MASTER_ADDR'):
    raise RuntimeError(f'Expected eight AIHC replicas; got {env}')
verify_release()
os.environ['OPENWAM_NODE_RANK'] = str(rank)
write_json(ROOT / 'results' / f'{phase}_node{rank}.json', {'host': socket.gethostname(), 'env': env})
torchrun = [str(Path(sys.executable).parent / 'torchrun'), '--nnodes=8', '--nproc-per-node=8', f'--node-rank={rank}',
            '--master-addr=' + os.environ['MASTER_ADDR'], '--master-port=' + os.environ.get('MASTER_PORT', '29500'),
            '--max-restarts=0']
os.chdir(ROOT / 'release')
os.environ['PYTHONPATH'] = str(ROOT / 'release')
monitor = subprocess.Popen([sys.executable, str(ROOT / 'pipeline/node_monitor.py'), phase])
try:
    with (ROOT / 'results' / f'{phase}_node{rank}.log').open('a', buffering=1) as output:
        for script, timeout in (('nccl_probe.py', 600), ('train_experiment.py', 4 * 3600)):
            command = torchrun + [str(ROOT / 'pipeline' / script)]
            print('LAUNCH', json.dumps(command), flush=True)
            subprocess.run(command, check=True, timeout=timeout, stdout=output, stderr=subprocess.STDOUT)
finally:
    monitor.terminate()
    monitor.wait(timeout=15)
