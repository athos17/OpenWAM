"""Run bounded production training, then a fresh 64-process resume invocation."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import json

from runtime import ROOT, CONFIG_NAME, require_preflight, write_json

rank = int(os.environ.get('NODE_RANK', os.environ.get('RANK', '-1')))
assert int(os.environ['WORLD_SIZE']) == 8 and 0 <= rank < 8
require_preflight()
os.environ['OPENWAM_NODE_RANK'] = str(rank)
os.environ['TOUCHWAM_ACCEPTANCE_REPORTS'] = str(ROOT/'results')
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
        for phase, stop in [('fresh', 120), ('resume', 140)]:
            os.environ['TOUCHWAM_ACCEPTANCE_PHASE'] = phase
            command = torchrun+['scripts/validation/touchwam_checkpoint_acceptance.py', '--config-name', CONFIG_NAME,
                f'training.output_path={ROOT}/training_output', f'training.stop_after_steps={stop}',
                'training.save_steps=60', 'training.log_flush_every=1']
            if phase == 'resume':
                run = json.loads((ROOT/'results/fresh-rank0.json').read_text())['run_dir']
                command += [f'training.resume_ckpt_path={run}', 'training.finetune_ckpt_path=null',
                            'training.initialize_touch_from_openwam=false']
            subprocess.run(command, check=True, timeout=5400, stdout=log, stderr=subprocess.STDOUT)
    write_json(ROOT/'results'/f'completed_node{rank}.json', dict(status='completed'))
except BaseException as exc:
    write_json(ROOT/'results'/f'failure_node{rank}.json', dict(status='failed', error=repr(exc)))
    raise
finally:
    monitor.terminate()
    monitor.wait(timeout=15)
