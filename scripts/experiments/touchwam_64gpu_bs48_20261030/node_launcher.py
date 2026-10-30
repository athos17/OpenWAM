"""Eight-node NCCL acceptance followed by production rebatch continuation."""
import os
from pathlib import Path
import socket
import subprocess
import sys

from runtime import ROOT, require_preflight, write_json

rank=int(os.environ.get('NODE_RANK',os.environ.get('RANK','-1')))
assert int(os.environ['WORLD_SIZE'])==8 and 0<=rank<8
proof=require_preflight()
run=Path(proof['run_dir'])
os.environ['OPENWAM_NODE_RANK']=str(rank)
write_json(ROOT/'results'/f'full_node{rank}.json',dict(host=socket.gethostname(),node_rank=rank,run_dir=str(run)))
torchrun=[sys.executable,'-m','torch.distributed.run','--nnodes=8','--nproc-per-node=8',
    f'--node-rank={rank}','--master-addr='+os.environ['MASTER_ADDR'],
    '--master-port='+os.environ.get('MASTER_PORT','29500'),'--max-restarts=0']
os.chdir(ROOT/'release')
monitor=subprocess.Popen([sys.executable,str(ROOT/'pipeline/node_monitor.py'),'full'])
try:
    with (ROOT/'results'/f'full_node{rank}.log').open('a',buffering=1) as log:
        subprocess.run(torchrun+[str(ROOT/'pipeline/nccl_probe.py')],check=True,timeout=600,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(torchrun+['scripts/train.py','--config-path',str(run),'--config-name','config'],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
    write_json(ROOT/'results'/f'completed_node{rank}.json',dict(status='completed'))
except BaseException as exc:
    write_json(ROOT/'results'/f'failure_node{rank}.json',dict(status='failed',error=repr(exc)))
    raise
finally:
    monitor.terminate();monitor.wait(timeout=15)
