import hashlib,json,os,socket,subprocess,sys
from pathlib import Path
R=Path('/mnt/pfs/v89atp/workspace/setup/accel_cmp_32gpu_20261026')
NODES=4
variant=sys.argv[1]
assert variant in ('base','s7d'),variant
keys=['MASTER_ADDR','MASTER_PORT','RANK','WORLD_SIZE','NODE_RANK','GROUP_RANK','LOCAL_RANK','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NCCL_IB_DISABLE']
env={key:os.environ.get(key) for key in keys}
print('NODE_LAUNCH',json.dumps({'host':socket.gethostname(),'variant':variant,'env':env}),flush=True)
rank=int(os.environ.get('NODE_RANK',os.environ.get('RANK','-1')))
nodes=int(os.environ.get('WORLD_SIZE','0'))
assert nodes==NODES and 0<=rank<NODES,(nodes,rank,env)
assert os.environ.get('MASTER_ADDR'),env
os.environ['OPENWAM_NODE_RANK']=str(rank)
(R/'results'/f'{variant}_node{rank}.json').write_text(json.dumps({'host':socket.gethostname(),'env':env},indent=2)+'\n')
args=[str(Path(sys.executable).parent/'torchrun'),f'--nnodes={NODES}','--nproc-per-node=8',f'--node-rank={rank}',
      '--master-addr='+os.environ['MASTER_ADDR'],'--master-port='+os.environ.get('MASTER_PORT','29500'),'--max-restarts=0']
monitor=subprocess.Popen([sys.executable,str(R/'pipeline/node_monitor.py'),variant])
try:
    with (R/'results'/f'{variant}_node{rank}.log').open('a',buffering=1) as output:
        for target in ('nccl','train'):
            if target=='nccl':
                command=args+[str(R/'pipeline/nccl_probe.py'),variant];timeout=600
            else:
                manifest=json.loads((R/'results/release_manifest.json').read_text())
                for name,expected in manifest['files'].items():
                    assert hashlib.sha256((R/'release'/name).read_bytes()).hexdigest()==expected,name
                for name,expected in manifest.get('runtime_files',{}).items():
                    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected,name
                os.chdir(R/'release');os.environ['PYTHONPATH']=str(R/'release')
                command=args+[str(R/'pipeline/train_experiment.py'),variant];timeout=None
            print('LAUNCH',json.dumps(command),flush=True)
            subprocess.run(command,check=True,timeout=timeout,stdout=output,stderr=subprocess.STDOUT)
finally:
    monitor.terminate()
    monitor.wait(timeout=15)
