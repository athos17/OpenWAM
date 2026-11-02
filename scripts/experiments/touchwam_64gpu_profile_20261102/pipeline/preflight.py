"""CPU checks before submission (dev-cpu, no GPU):

1. the frozen base config composes to r2's saved config.yaml (every key r2 has, same value; keys it
   lacks are new trainer options at their train.yaml defaults);
2. the profile config differs from it only in the backend / timing / run-length / profile options listed below;
3. the weights exist and every source loads one sample through the frozen release.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runtime import (BASE_CONFIG_NAME, BASELINE_RUN, ROOT, config_name, output_root, release_hash,  # noqa: E402
                     variant, verify_release, write_json)

sys.path.insert(0, str(ROOT / 'release'))
import yaml  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

COMMON_KEYS = {
    'training.stop_after_steps', 'training.save_steps', 'training.save_full_states_for_resume',
    'training.touchwam_p1.deferred_modality_sync', 'training.touchwam_p1.deferred_blocking_steps',
    'training.touchwam_p1.input_validation_steps', 'training.touchwam_p1.pinned_tactile_h2d',
    'training.step_timing.record_steps', 'training.step_timing.peak_per_step', 'training.output_path', 'project.name',
    'training.torch_profile.enabled', 'training.torch_profile.wait', 'training.torch_profile.warmup',
    'training.torch_profile.active', 'training.torch_profile.ranks',
}
HSDP_KEYS = {
    'training.backend', 'training.fsdp2.param_dtype', 'training.fsdp2.reduce_dtype', 'training.fsdp2.forward_prefetch',
    'training.fsdp2.local_optimizer', 'training.optimizer_fused', 'training.overlap_param_allgather',
}
# Written into the saved config by the trainer, not composed.
RUNTIME_PREFIXES = ('model.video_backbone.components', 'model.video_backbone.tokenizer.', 'project.output_dir')


def flatten(value, prefix=''):
    out = {}
    for key, val in value.items():
        if isinstance(val, dict) and val:
            out.update(flatten(val, f'{prefix}{key}.'))
        else:
            out[f'{prefix}{key}'] = val
    return out


def resolved(name):
    from scripts.train import _inject_project_seed  # project.seed -> dataloader.seed, as at launch

    with initialize_config_dir(config_dir=str(ROOT / 'release/configs'), version_base=None):
        cfg = compose(config_name=name)
    _inject_project_seed(cfg)
    return cfg, flatten(OmegaConf.to_container(cfg, resolve=False))


def main():
    verify_release()
    report = {'status': 'running', 'release_manifest_sha256': release_hash(), 'variant': variant()}

    baseline = flatten(yaml.safe_load((BASELINE_RUN / 'config.yaml').read_text()))
    _, base = resolved(BASE_CONFIG_NAME)
    mismatched = {k: [baseline[k], base.get(k, '<missing>')] for k in baseline
                  if not k.startswith(RUNTIME_PREFIXES) and base.get(k, '<missing>') != baseline[k]}
    assert not mismatched, f'base config differs from r2: {mismatched}'
    report['base_equals_baseline_keys'] = len(baseline)
    report['base_new_keys'] = sorted(set(base) - set(baseline))

    cfg, profile = resolved(config_name())
    changed = {k: [base.get(k, '<missing>'), profile.get(k, '<missing>')] for k in sorted(set(base) | set(profile))
               if base.get(k, '<missing>') != profile.get(k, '<missing>')}
    allowed = COMMON_KEYS | (HSDP_KEYS if variant() == 'hsdp' else set())
    unexpected = sorted(set(changed) - allowed)
    assert not unexpected, f'profile config changes more than the listed options: {unexpected}'
    report['profile_changes'] = changed
    t = cfg.training
    assert t.backend == {'ds': 'deepspeed', 'hsdp': 'fsdp2'}[variant()]
    assert t.overlap_param_allgather == (variant() == 'ds')
    assert t.touchwam_p1.deferred_modality_sync and t.step_timing.enabled and t.torch_profile.enabled
    assert list(t.torch_profile.ranks) == [*range(8), *range(8, 64, 8)]
    assert (t.torch_profile.wait, t.torch_profile.warmup, t.torch_profile.active) == (20, 2, 20)
    assert t.batch_size == 48 and t.gradient_accumulation_steps == 1 and t.dataset_num_workers == 8
    assert t.num_epochs == 1 and t.max_steps is None and t.stop_after_steps == 45 and not t.save_steps
    assert t.gradient_checkpointing_layers == list(range(20)) and cfg.project.seed == 42
    assert Path(t.output_path) == output_root(), t.output_path
    assert Path(t.finetune_ckpt_path).is_file(), t.finetune_ckpt_path
    weights = Path(cfg.model.architecture.tactile.observation_encoder.weights_root)
    assert weights.is_dir(), weights

    from openwam.train.fsdp2.accelerator import FSDP2Accelerator  # the backend imports on this environment
    report['fsdp2_backend'] = FSDP2Accelerator.__name__

    from openwam.dataloader.registry import build_dataset
    start = time.time()
    dataset = build_dataset(cfg.dataloader)
    report['dataset'] = {'names': list(dataset._names), 'length': len(dataset), 'build_seconds': time.time() - start}
    samples = {}
    for name, source in zip(dataset._names, dataset._datasets):
        start = time.time()
        sample = source[len(source) // 2]
        samples[name] = {'frames': len(sample['video']), 'seconds': round(time.time() - start, 2),
                         'modality': sample.get('training_modality'), 'episode_id': str(sample.get('episode_id'))}
    report['samples'] = samples
    report['status'] = 'passed'
    write_json(ROOT / 'results/preflight.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'base_equals_baseline_keys', 'dataset', 'samples')}, indent=2))
    print('profile changes:', json.dumps(changed))


if __name__ == '__main__':
    main()
