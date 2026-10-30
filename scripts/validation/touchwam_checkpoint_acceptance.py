"""Instrument the actual production trainer for a 64-rank save/resume check.

Uses the full production population, sampler, model and optimizer. Only stopping,
checkpoint frequency and log flushing differ. Reports are separate from metrics.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.utils.checkpoint as checkpoint_module


def forbid_recompute(*args, **kwargs):
    raise RuntimeError("Activation recomputation invoked during no-recompute acceptance")


checkpoint_module.checkpoint = forbid_recompute
from openwam.train import openwam_trainer as trainer_module

REPORTS = Path(os.environ["TOUCHWAM_ACCEPTANCE_REPORTS"])
PHASE = os.environ["TOUCHWAM_ACCEPTANCE_PHASE"]
RANK = int(os.environ["RANK"])


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def fingerprint(value):
    """Bounded deterministic samples from every tensor, plus exact scalar state.

    This is a restoration diagnostic, not a full tensor/file checksum.
    """
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1)
        sample = flat[::max(1, flat.numel() // 32)][:32].contiguous().cpu()
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sampled_sha256=hashlib.sha256(sample.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported acceptance state: {type(value)}')


class AcceptanceTrainer(trainer_module.OpenWAMTrainer):
    def __init__(self, cfg, *args, **kwargs):
        assert int(os.environ['WORLD_SIZE']) == 64
        assert cfg.training.batch_size == 12 and cfg.training.gradient_accumulation_steps == 1
        assert cfg.training.max_steps is None and cfg.training.num_epochs == 1
        assert cfg.training.learning_rate == 5e-5
        assert cfg.training.stop_after_steps == (120 if PHASE == 'fresh' else 140)
        assert cfg.training.save_full_states_for_resume
        assert not cfg.training.use_gradient_checkpointing
        assert not cfg.model.architecture.mot_checkpoint_mixed_attn
        self.acceptance_steps = []
        self.acceptance_resume_verified = False
        super().__init__(cfg, *args, **kwargs)
        if PHASE == 'fresh':
            init = self.architecture.initialization_provenance
            assert init['openwam_tensors_preserved'] is True
            assert init['tactile_observation_encoder']['initialization'] == 'ftp1_pretrained'

    def prepare_accelerate(self, *args):
        result = super().prepare_accelerate(*args)
        self.acceptance_scheduler = result[3]
        return result

    def snapshot(self):
        arch = self.accelerator.unwrap_model(self.architecture)
        return dict(model=fingerprint(arch.state_dict()),
                    optimizer=fingerprint(self._modality_optimizer.state_dict()),
                    scheduler=fingerprint(self.acceptance_scheduler.state_dict()))

    def resume_if_configured(self, *args):
        result = super().resume_if_configured(*args)
        assert result[:2] == (120, 120), result
        expected = json.loads((REPORTS/f'fresh-rank{RANK}.json').read_text())['snapshot']
        actual = self.snapshot()
        mismatch = [k for k in expected if expected[k] != actual[k]]
        if mismatch:
            write(REPORTS/f'resume-mismatch-rank{RANK}.json', dict(expected=expected, actual=actual))
            raise RuntimeError(f'Restored state differs: {mismatch}')
        assert bool(self.accelerator.unwrap_model(self.architecture).touch_v2.observation_encoder.weights_initialized)
        self.acceptance_resume_verified = True
        return result

    def log_step(self, **kwargs):
        assert all(math.isfinite(float(v)) for v in kwargs['metrics'].values())
        assert kwargs['opt_step'] == kwargs['global_step']
        self.acceptance_steps.append(kwargs['global_step'])
        super().log_step(**kwargs)

    def finish_training(self, output_path, global_step, *args, **kwargs):
        super().finish_training(output_path, global_step, *args, **kwargs)
        start, stop = (1, 120) if PHASE == 'fresh' else (121, 140)
        assert self.acceptance_steps == list(range(start, stop + 1))
        assert PHASE == 'fresh' or self.acceptance_resume_verified
        folder = Path(output_path)
        assert (folder/f'checkpoint_step_{stop}.safetensors').is_file()
        meta = json.loads((folder/f'accel_state_step_{stop}/trainer_state.json').read_text())
        assert meta['global_step'] == meta['opt_step'] == stop
        assert meta['batches_per_epoch'] == 138386
        write(REPORTS/f'{PHASE}-rank{RANK}.json', dict(status='passed', rank=RANK,
              phase=PHASE, first_step=start, last_step=stop, run_dir=str(folder),
              resume_verified=self.acceptance_resume_verified, snapshot=self.snapshot()))
        self.accelerator.wait_for_everyone()


trainer_module.OpenWAMTrainer = AcceptanceTrainer
from scripts.train import main

if __name__ == '__main__':
    main()
