"""Helpers for the frozen setup/pipeline copy of this launch bundle."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = 'train_touchwam_six_source_64gpu_bs12_20261029'


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def verify_release():
    manifest = json.loads((ROOT/'results/release_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen launch file changed: {name}')
    return manifest


def require_preflight():
    verify_release()
    proof = json.loads((ROOT/'results/preflight.json').read_text())
    if proof['status'] != 'passed' or not proof['training_launch_allowed']:
        raise RuntimeError('Full production-data preflight must pass before submission')
    t = proof['config']['training']
    a = proof['config']['model']['architecture']
    assert t['batch_size'] == 12 and t['gradient_accumulation_steps'] == 1
    assert t['learning_rate'] == 5e-5 and t['num_epochs'] == 1
    assert t['max_steps'] is None and t['stop_after_steps'] is None
    assert t['save_full_states_for_resume'] and t['save_steps'] == 2000
    assert not t['use_gradient_checkpointing'] and not t['use_gradient_checkpointing_offload']
    assert not t['gradient_checkpointing_layers'] and not t['gradient_checkpointing_layers_video_only']
    assert not a['mot_checkpoint_mixed_attn']
    return proof
