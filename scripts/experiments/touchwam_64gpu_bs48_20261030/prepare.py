"""Freeze a fresh batch-48 release and verify its real data/Alpha initialization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

REPO=Path(__file__).resolve().parents[3]
CONFIG='train_touchwam_six_source_64gpu_bs48_20261030'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--setup',required=True)
    args=parser.parse_args()
    root=Path(args.setup)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit the release before preparation')
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in ('results','pipeline','training_output'):(root/name).mkdir()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    subprocess.run(['git','archive','--format=tar','--prefix=release/',f'--output={root}/release.tar',commit],cwd=REPO,check=True)
    with tarfile.open(root/'release.tar') as archive:archive.extractall(root,filter='data')
    common=REPO/'scripts/experiments/touchwam_64gpu_bs12_20261029'
    for name in ('cluster_job.py','cluster_bootstrap.sh','nccl_probe.py','node_monitor.py'):
        shutil.copy2(common/name,root/'pipeline'/name)
    for name in ('runtime.py','node_launcher.py'):shutil.copy2(Path(__file__).parent/name,root/'pipeline'/name)
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for folder in ('release','pipeline') for p in (root/folder).rglob('*') if p.is_file()}
    (root/'results/release_manifest.json').write_text(json.dumps(dict(commit=commit,files=files),indent=2)+'\n')
    env=dict(os.environ,PYTHONPATH=str(root/'release'),PYTHONDONTWRITEBYTECODE='1',
        TOUCHWAM_OUTPUT_ROOT=str(root/'training_output'),OMP_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    audit=root/'results/alpha_bitwise_audit.json'
    checkpoint='/mnt/pfs/v89atp/workspace/models/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors'
    commands=[('alpha_audit.log',[sys.executable,'scripts/validation/touchwam_alpha_init.py',
        '--checkpoint',checkpoint,'--output',str(audit),'--config',CONFIG]),
        ('preflight.log',[sys.executable,'scripts/preflight_touchwam.py','--config',CONFIG,
        '--output',str(root/'results/preflight.json'),'--alpha-audit',str(audit)])]
    for logname,cmd in commands:
        with (root/'results'/logname).open('w',buffering=1) as log:
            subprocess.run(cmd,cwd=root/'release',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    proof=json.loads((root/'results/preflight.json').read_text())
    (root/'results/launch_config_resolved.json').write_text(json.dumps(proof['config'],indent=2)+'\n')
    print(json.dumps(dict(status=proof['status'],commit=commit,setup=str(root),
        output_root=str(root/'training_output'),sources=list(proof['sources'])),indent=2))


if __name__=='__main__':main()
