import hashlib,json,os,socket,subprocess,sys
from pathlib import Path
R=Path('/mnt/pfs/v89atp/workspace/setup/action_random_64gpu_20261022')
phase=sys.argv[1]
keys=['MASTER_ADDR','MASTER_PORT','RANK','WORLD_SIZE','NODE_RANK','GROUP_RANK','LOCAL_RANK','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NCCL_IB_DISABLE']
env={key:os.environ.get(key) for key in keys}
print('NODE_LAUNCH',json.dumps({'host':socket.gethostname(),'phase':phase,'env':env}),flush=True)
rank=int(os.environ.get('NODE_RANK',os.environ.get('RANK','-1')))
nodes=int(os.environ.get('WORLD_SIZE','0'))
assert nodes==8 and 0<=rank<8,(nodes,rank,env)
assert os.environ.get('MASTER_ADDR'),env
os.environ['OPENWAM_NODE_RANK']=str(rank)
(R/'results'/f'{phase}_node{rank}.json').write_text(json.dumps({'host':socket.gethostname(),'env':env},indent=2)+'\n')
args=[str(Path(sys.executable).parent/'torchrun'),'--nnodes=8','--nproc-per-node=8',f'--node-rank={rank}',
      '--master-addr='+os.environ['MASTER_ADDR'],'--master-port='+os.environ.get('MASTER_PORT','29500'),'--max-restarts=0']
if phase=='verify':phases=['nccl','smoke','resume']
elif phase in ('formal','nccl'):phases=[phase]
else:raise ValueError(phase)
monitor=subprocess.Popen([sys.executable,str(R/'pipeline/node_monitor.py'),phase])
try:
    with (R/'results'/f'{phase}_node{rank}.log').open('a',buffering=1) as output:
        for target in phases:
            if target=='nccl':
                command=args+[str(R/'pipeline/nccl_probe.py')];timeout=600
            else:
                release='release'
                manifest_name='release_manifest.json'
                manifest=json.loads((R/'results'/manifest_name).read_text())
                for name,expected in manifest['files'].items():
                    assert hashlib.sha256((R/release/name).read_bytes()).hexdigest()==expected,name
                for name,expected in manifest.get('runtime_files',{}).items():
                    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected,name
                os.chdir(R/release);os.environ['PYTHONPATH']=str(R/release)
                command=args+[str(R/'pipeline/train_experiment.py'),target]
                timeout=7200 if target!='formal' else None
            print('LAUNCH',json.dumps(command),flush=True)
            subprocess.run(command,check=True,timeout=timeout,stdout=output,stderr=subprocess.STDOUT)
finally:
    monitor.terminate()
    monitor.wait(timeout=15)
