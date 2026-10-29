"""Production trainer on the baseline's schedule and data order: `settings.json` steps
(speed test, no checkpoints) or, with steps 0, the whole run with the config's checkpointing.

Records the baseline's per-step fields (sample ids, lr, losses, step rate) in
<output>/metrics.jsonl so compare_with_baseline.py can match the runs step by step.
"""
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).parent))
from runtime import (CONFIG_NAME, ROOT, SAMPLE_KEYS, output_root, phase, release_hash, settings,  # noqa: E402
                     write_json)

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: E402
from openwam.train.utils.seeding import seed_everything  # noqa: E402
from scripts.train import _apply_gpu_memory_cap, _build_accelerator, _inject_project_seed  # noqa: E402


class SpeedTrainer(OpenWAMTrainer):
    def __init__(self, *args, **kwargs):
        self.started = time.time()
        self.last = None
        super().__init__(*args, **kwargs)
        assert self.accelerator.num_processes == 64

    def compute_loss(self, batch):
        self.sources = [s['_dataset_name'] for s in batch]
        self.sample_ids = [{k: str(s.get(k)) for k in SAMPLE_KEYS} for s in batch]
        self.video_only = all(s.get('training_modality') == 'video_only' for s in batch)
        losses = super().compute_loss(batch)
        assert all(torch.isfinite(v).all() for v in losses.values()), self.sample_ids
        self.losses = {k: float(v.detach()) for k, v in losses.items()}
        return losses

    def log_step(self, **kw):
        if self.accelerator.is_main_process:
            super().log_step(**kw)
        record = {'step': kw['global_step'], 'optimizer_step': kw['opt_step'], 'lr': kw['lr'],
                  'sources': self.sources, 'sample_ids': self.sample_ids, 'video_only': self.video_only,
                  'local_losses': self.losses, 'metrics': {k: float(v) for k, v in kw['metrics'].items()},
                  'steps_per_sec': kw['steps_per_sec'], 'time': time.time()}
        if self.accelerator.is_main_process:
            with (Path(kw['output_path']) / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print('TRAIN_PROGRESS ' + json.dumps({k: record[k] for k in ('step', 'lr', 'metrics', 'steps_per_sec')}),
                  flush=True)
        self.last = record
        if kw['global_step'] % 10 == 0:
            self.write_report('running', kw['output_path'])

    def finish_training(self, output_path, *args, **kwargs):
        super().finish_training(output_path, *args, **kwargs)
        self.write_report('passed', output_path)

    def write_report(self, status, output_path):
        write_json(ROOT / 'results' / f'{phase()}_rank{self.accelerator.process_index}.json', {
            'status': status, 'release_manifest_sha256': release_hash(), 'output_path': str(output_path),
            'last_step': self.last and self.last['step'], 'elapsed_seconds': time.time() - self.started,
            'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved()})


def main():
    run = settings()
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.WARNING)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=1200))
    overrides = [f'training.output_path={output_root()}', f'training.dataset_num_workers={run["dataset_num_workers"]}']
    if phase() == 'speed':
        overrides += [f'training.stop_after_steps={run["steps"]}', 'training.save_steps=0',
                      'training.save_full_states_for_resume=false']
    with initialize_config_dir(config_dir=str(ROOT / 'release/configs'), version_base=None):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
    t = cfg.training
    # The full run's schedule: one epoch, no max_steps (stop_after_steps does not shorten it).
    assert t.batch_size == 48 and t.gradient_accumulation_steps == 1 and t.num_epochs == 1 and t.max_steps is None
    _inject_project_seed(cfg)
    seed_everything(cfg.project.seed, rank=int(os.environ['RANK']))
    _apply_gpu_memory_cap(cfg)
    accelerator = _build_accelerator(cfg)
    dataset = build_dataset(cfg.dataloader)
    SpeedTrainer(cfg, accelerator=accelerator, dataset=dataset).train()
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # A failed rank must take its node down: otherwise peers block in NCCL and the
        # job keeps its GPUs until the collective timeout.
        traceback.print_exc()
        os._exit(1)
