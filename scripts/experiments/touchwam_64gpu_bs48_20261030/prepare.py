"""Prepare a frozen release and auditably migrate one complete source checkpoint."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from omegaconf import OmegaConf

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--setup',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--step',required=True,type=int)
    parser.add_argument('--previous-setup',required=True)
    args=parser.parse_args()
    root=Path(args.setup);previous=Path(args.previous_setup)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit the release before preparation')
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in ('results','pipeline','training_output'):(root/name).mkdir()
    spec=importlib.util.spec_from_file_location('stage_rebatch',REPO/'scripts/validation/stage_touchwam_rebatch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run=root/'training_output'/'rebatch_bs48'
    migration=module.stage(args.source,run,args.step)
    cfg=OmegaConf.to_container(OmegaConf.load(run/'config.yaml'),resolve=True)
    source_contract=json.loads((Path(args.source)/'training_data_contract.json').read_text())
    target_contract=json.loads((run/'training_data_contract.json').read_text())
    if {k:v for k,v in source_contract.items() if k!='training_layout'} != {
            k:v for k,v in target_contract.items() if k!='training_layout'}:
        raise RuntimeError('Rebatch changed the source population')
    previous_preflight=previous/'results/preflight.json'
    old=json.loads(previous_preflight.read_text())
    if old['status']!='passed' or not old['training_launch_allowed']:
        raise RuntimeError('Source production data preflight was not passed')
    if cfg['dataloader']!=old['config']['dataloader']:
        raise RuntimeError('Data settings differ from the validated source production configuration')
    old_model=old['config']['model'];new_model=cfg['model']
    # Saved reconstruction metadata adds components; architecture behavior only
    # changes the explicitly requested mixed-attention recomputation flag.
    architecture=dict(new_model['architecture']);architecture['mot_checkpoint_mixed_attn']=False
    if architecture!=old_model['architecture']:
        raise RuntimeError('Unexpected architecture change')
    source_state=Path(args.source)/f'accel_state_step_{args.step}'
    target_state=run/'accel_state_step_0'
    tensor_files=[p for p in source_state.rglob('*') if p.is_file() and p.name not in ('scheduler.bin','trainer_state.json')]
    if not all(p.samefile(target_state/p.relative_to(source_state)) for p in tensor_files):
        raise RuntimeError('Model, Adam or RNG storage was not preserved')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    subprocess.run(['git','archive','--format=tar','--prefix=release/',f'--output={root}/release.tar',commit],cwd=REPO,check=True)
    with tarfile.open(root/'release.tar') as archive:archive.extractall(root,filter='data')
    common=REPO/'scripts/experiments/touchwam_64gpu_bs12_20261029'
    current=Path(__file__).parent
    for name in ('cluster_job.py','cluster_bootstrap.sh','nccl_probe.py','node_monitor.py'):
        shutil.copy2(common/name,root/'pipeline'/name)
    for name in ('runtime.py','node_launcher.py'):shutil.copy2(current/name,root/'pipeline'/name)
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for folder in ('release','pipeline') for p in (root/folder).rglob('*') if p.is_file()}
    write(root/'results/release_manifest.json',dict(commit=commit,files=files))
    proof=dict(status='passed',training_launch_allowed=True,run_dir=str(run),config=cfg,
        scope='Source data readiness reused; rebatch population, tensor storage and LR migration verified. GPU startup pending.',
        previous_preflight=str(previous_preflight),previous_preflight_sha256=hashlib.sha256(previous_preflight.read_bytes()).hexdigest(),
        unchanged_tensor_files=len(tensor_files),plan=migration['plan'],
        run_config_sha256=hashlib.sha256((run/'config.yaml').read_bytes()).hexdigest(),
        data_contract_sha256=hashlib.sha256((run/'training_data_contract.json').read_bytes()).hexdigest())
    write(root/'results/preflight.json',proof)
    write(root/'results/launch_config_resolved.json',cfg)
    write(root/'results/preparation.json',{k:v for k,v in proof.items() if k!='config'})
    print(json.dumps({k:v for k,v in proof.items() if k!='config'},indent=2))


if __name__=='__main__':main()
