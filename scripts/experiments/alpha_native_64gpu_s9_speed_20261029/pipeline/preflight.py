"""CPU checks before submission (dev-cpu, no GPU):

1. the frozen base config composes to the baseline run's saved config.yaml (every key the
   baseline has, same value; keys it lacks are new trainer options at their train.yaml defaults);
2. the s9 config differs from it only in the acceleration / timing options listed below;
3. the three readers load one ColorJitter'd sample each through the frozen release.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runtime import (BASE_CONFIG_NAME, BASELINE_RUN, CONFIG_NAME, ROOT, release_hash, settings,  # noqa: E402
                     verify_release, write_json)

sys.path.insert(0, str(ROOT / 'release'))
import yaml  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

S9_KEYS = {
    'training.output_path', 'training.deepspeed_zero.overlap_comm', 'training.deepspeed_zero.contiguous_gradients',
    'training.deepspeed_zero.reduce_bucket_size', 'training.deepspeed_zero.allgather_bucket_size',
    'training.text_padding', 'training.text_dedup', 'training.video_uint8_in_workers', 'training.log_flush_every',
    'training.vae_cudnn_benchmark', 'training.vae_channels_last', 'training.frame_time_modulation',
    'training.gradient_checkpointing_layers', 'training.dit_block_fusion', 'training.vae_fused_norm',
    'training.overlap_param_allgather', 'training.loss_log_csv', 'training.step_timing.enabled',
    'training.step_timing.summary_every', 'training.step_timing.mfu', 'training.step_timing.peak_tflops',
    'model.architecture.mot_split_attention', 'project.name',
}


def flatten(value, prefix=''):
    out = {}
    for key, val in value.items():
        if isinstance(val, dict) and val:
            out.update(flatten(val, f'{prefix}{key}.'))
        else:
            out[f'{prefix}{key}'] = val
    return out


# Written into the saved config by the trainer, not composed: the Wan component / tokenizer
# specs come from the Alpha checkpoint's config.yaml (propagate_component_specs).
RUNTIME_PREFIXES = ('model.video_backbone.components', 'model.video_backbone.tokenizer.', 'project.output_dir')


def resolved(name):
    from scripts.train import _inject_project_seed  # project.seed -> dataloader.seed, as at launch

    with initialize_config_dir(config_dir=str(ROOT / 'release/configs'), version_base=None):
        cfg = compose(config_name=name)
    _inject_project_seed(cfg)
    return cfg, flatten(OmegaConf.to_container(cfg, resolve=False))


def main():
    verify_release()
    report = {'status': 'running', 'release_manifest_sha256': release_hash(), 'settings': settings()}

    baseline = flatten(yaml.safe_load((BASELINE_RUN / 'config.yaml').read_text()))
    _, base = resolved(BASE_CONFIG_NAME)
    mismatched = {k: [baseline[k], base.get(k, '<missing>')] for k in baseline
                  if not k.startswith(RUNTIME_PREFIXES) and base.get(k, '<missing>') != baseline[k]}
    assert not mismatched, f'base config differs from the baseline run: {mismatched}'
    report['base_equals_baseline_keys'] = len(baseline)
    report['base_new_keys'] = {k: base[k] for k in sorted(set(base) - set(baseline))}

    cfg, s9 = resolved(CONFIG_NAME)
    changed = {k: [base.get(k, '<missing>'), s9.get(k, '<missing>')] for k in sorted(set(base) | set(s9))
               if base.get(k, '<missing>') != s9.get(k, '<missing>')}
    unexpected = sorted(set(changed) - S9_KEYS)
    assert not unexpected, f's9 config changes more than the acceleration options: {unexpected}'
    report['s9_changes'] = changed
    t = cfg.training
    assert t.batch_size == 48 and t.gradient_accumulation_steps == 1 and t.num_epochs == 1 and t.max_steps is None
    assert Path(t.finetune_ckpt_path).is_file(), t.finetune_ckpt_path

    from openwam.dataloader.registry import build_dataset
    start = time.time()
    dataset = build_dataset(cfg.dataloader)
    report['dataset'] = {'names': list(dataset._names), 'length': len(dataset), 'build_seconds': time.time() - start}
    samples = {}
    for name, source in zip(dataset._names, dataset._datasets):
        start = time.time()
        sample = source[len(source) // 2]
        video = sample['video']
        samples[name] = {'frames': len(video), 'size': list(video[0].size), 'seconds': round(time.time() - start, 2),
                         'episode_id': str(sample.get('episode_id'))}
        assert len(video) == 33 and video[0].size == (320, 384), samples[name]
    report['samples'] = samples
    report['status'] = 'passed'
    write_json(ROOT / 'results/preflight.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'base_equals_baseline_keys', 'dataset', 'samples')}, indent=2))
    print('s9 changes:', json.dumps(changed))


if __name__ == '__main__':
    main()
