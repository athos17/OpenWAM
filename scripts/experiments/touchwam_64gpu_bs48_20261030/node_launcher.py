"""Eight-node NCCL acceptance followed by fresh full OpenWAM + FTP training."""
import os
from pathlib import Path
import socket
import subprocess
import sys

from runtime import ROOT, CONFIG_NAME, require_preflight, write_json

rank=int(os.environ.get('NODE_RANK',os.environ.get('RANK','-1')))
assert int(os.environ['WORLD_SIZE'])==8 and 0<=rank<8
proof=require_preflight()
os.environ['TOUCHWAM_OUTPUT_ROOT']=proof['output_root']
os.environ['OPENWAM_NODE_RANK']=str(rank)
write_json(ROOT/'results'/f'full_node{rank}.json',dict(host=socket.gethostname(),node_rank=rank,output_root=proof['output_root']))
torchrun=[sys.executable,'-m','torch.distributed.run','--nnodes=8','--nproc-per-node=8',
    f'--node-rank={rank}','--master-addr='+os.environ['MASTER_ADDR'],
    '--master-port='+os.environ.get('MASTER_PORT','29500'),'--max-restarts=0']
os.chdir(ROOT/'release')
monitor=subprocess.Popen([sys.executable,str(ROOT/'pipeline/node_monitor.py'),'full'])
try:
    with (ROOT/'results'/f'full_node{rank}.log').open('a',buffering=1) as log:
        subprocess.run(torchrun+[str(ROOT/'pipeline/nccl_probe.py')],check=True,timeout=600,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(torchrun+['scripts/train.py','--config-name',CONFIG_NAME],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
    write_json(ROOT/'results'/f'completed_node{rank}.json',dict(status='completed'))
except BaseException as exc:
    write_json(ROOT/'results'/f'failure_node{rank}.json',dict(status='failed',error=repr(exc)))
    raise
finally:
    monitor.terminate();monitor.wait(timeout=15)
