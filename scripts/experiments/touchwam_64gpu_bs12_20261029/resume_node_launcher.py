"""Continue the exact accepted population from its verified 140-step checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from runtime import ROOT, CONFIG_NAME, require_preflight, write_json

rank = int(os.environ.get('NODE_RANK', os.environ.get('RANK', '-1')))
assert int(os.environ['WORLD_SIZE']) == 8 and 0 <= rank < 8
require_preflight()
acceptance = json.loads((ROOT/'results/checkpoint_acceptance.json').read_text())
assert acceptance['status'] == 'passed' and acceptance['world_size'] == 64
assert acceptance['final_step'] == 140 and acceptance['per_rank_restored_state_comparison']
accepted_root = Path(acceptance['acceptance_root'])
assert hashlib.sha256((accepted_root/'results/release_manifest.json').read_bytes()).hexdigest() == acceptance['acceptance_manifest_sha256']
accepted_files = json.loads((accepted_root/'results/release_manifest.json').read_text())['files']
# The validated training implementation/config must be byte-identical; later
# launcher, tests or documentation changes do not affect these checks.
for name, digest in accepted_files.items():
    if name.startswith(('release/openwam/', 'release/configs/')) or name == 'release/scripts/train.py':
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
run = Path(acceptance['run_dir'])
assert (run/'accel_state_step_140/trainer_state.json').is_file()
os.environ['OPENWAM_NODE_RANK'] = str(rank)
write_json(ROOT/'results'/f'full_node{rank}.json', dict(host=socket.gethostname(), node_rank=rank))
torchrun = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=8', '--nproc-per-node=8',
            f'--node-rank={rank}', '--master-addr='+os.environ['MASTER_ADDR'],
            '--master-port='+os.environ.get('MASTER_PORT','29500'), '--max-restarts=0']
os.chdir(ROOT/'release')
monitor = subprocess.Popen([sys.executable, str(ROOT/'pipeline/node_monitor.py'), 'full'])
try:
    with (ROOT/'results'/f'full_node{rank}.log').open('a', buffering=1) as log:
        subprocess.run(torchrun+[str(ROOT/'pipeline/nccl_probe.py')], check=True, timeout=600,
                       stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(torchrun+['scripts/train.py', '--config-name', CONFIG_NAME,
            f'training.resume_ckpt_path={run}', 'training.finetune_ckpt_path=null',
            'training.initialize_touch_from_openwam=false'],
            check=True, stdout=log, stderr=subprocess.STDOUT)
    write_json(ROOT/'results'/f'completed_node{rank}.json', dict(status='completed'))
except BaseException as exc:
    write_json(ROOT/'results'/f'failure_node{rank}.json', dict(status='failed', error=repr(exc)))
    raise
finally:
    monitor.terminate()
    monitor.wait(timeout=15)
