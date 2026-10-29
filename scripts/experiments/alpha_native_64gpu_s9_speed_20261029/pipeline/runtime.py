"""Shared paths and release checks for the 8-node s9 speed test. No cluster or GPU imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
# dev-cpu sees the PFS as /mnt/pfs/ls3r34 too; AIHC pods mount only /mnt/pfs/v89atp.
if ROOT.is_relative_to('/mnt/pfs/ls3r34'):
    ROOT = Path('/mnt/pfs/v89atp') / ROOT.relative_to('/mnt/pfs/ls3r34')
BASE = Path('/mnt/pfs/v89atp/workspace')
CONFIG_NAME = 'train_alpha_native_64gpu_aug_s9'
BASE_CONFIG_NAME = 'train_alpha_native_64gpu_aug'
# The formal run this test reproduces (same data order and LR schedule) and compares against.
BASELINE_SETUP = BASE / 'setup/sharpa_native_64gpu_aug_20261026'
BASELINE_RUN = BASE / 'outputs/alpha_native_64gpu_aug/2026-10-26_23-07-36'
OUTPUT = BASE / 'outputs/alpha_native_64gpu_aug_s9_speed'
SAMPLE_KEYS = ('episode_id', 'episode_start_frame', '_dataset_name', '_dataset_index')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def settings():
    return json.loads((ROOT / 'settings.json').read_text())


def release_hash():
    return sha256(ROOT / 'results/release_manifest.json')


def verify_release():
    manifest = json.loads((ROOT / 'results/release_manifest.json').read_text())
    for group in ('files', 'runtime_files'):
        for name, expected in manifest[group].items():
            if sha256(ROOT / name) != expected:
                raise RuntimeError(f'{name} changed after prepare; prepare a new experiment directory')
    return manifest


def require_preflight():
    verify_release()
    report = json.loads((ROOT / 'results/preflight.json').read_text())
    if report.get('status') != 'passed' or report.get('release_manifest_sha256') != release_hash():
        raise RuntimeError('Run `launch.sh preflight` for this exact release before submission')
