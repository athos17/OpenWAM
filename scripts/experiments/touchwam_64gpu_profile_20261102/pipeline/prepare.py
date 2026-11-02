"""Freeze one profile's experiment directory from a clean checkout of this repository. Submits nothing.

    python scripts/experiments/touchwam_64gpu_profile_20261102/pipeline/prepare.py --variant ds|hsdp [--dest DIR]
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from runtime import BASE, VARIANTS, sha256, write_json  # noqa: E402

HERE = Path(__file__).absolute().parents[1]
REPO = HERE.parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=VARIANTS)
    p.add_argument('--dest')
    args = p.parse_args()
    dest = Path(args.dest or BASE / 'setup' / f'{HERE.name}_{args.variant}')
    if dest.exists():
        raise SystemExit(f'{dest} exists; a frozen experiment is never overwritten, pick a new --dest')
    git = lambda *a: subprocess.check_output(['git', '-C', str(REPO), *a], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('Tracked files have uncommitted changes; commit them so the release names one commit')
    commit = git('rev-parse', 'HEAD')

    (dest / 'release').mkdir(parents=True)
    archive = subprocess.run(['git', '-C', str(REPO), 'archive', commit], check=True, capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(dest / 'release')], input=archive, check=True)
    shutil.copytree(HERE / 'pipeline', dest / 'pipeline', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('launch.sh', 'README.md'):
        shutil.copy2(HERE / name, dest / name)
    write_json(dest / 'settings.json', {'variant': args.variant, 'commit': commit, 'source': str(REPO)})

    files = {str(path.relative_to(dest)): sha256(path)
             for path in sorted((dest / 'release').rglob('*')) if path.is_file()}
    runtime = {str(path.relative_to(dest)): sha256(path)
               for path in [dest / 'launch.sh', dest / 'settings.json', *sorted((dest / 'pipeline').glob('*'))]
               if path.is_file()}
    write_json(dest / 'results/release_manifest.json', {'commit': commit, 'files': files, 'runtime_files': runtime})
    print(json.dumps({'dest': str(dest), 'variant': args.variant, 'commit': commit, 'release_files': len(files)}))


if __name__ == '__main__':
    main()
