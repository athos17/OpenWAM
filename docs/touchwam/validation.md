# Migration validation — 2026-10-23

Target baseline: `96f4b6ec4d8d5d92fd65572ba23c15f2e1481f3e` on dev-cpu.
Validation uses `/root/miniforge/envs/openwam/bin/python` with CUDA disabled,
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. No training job or simulator was launched.

## Results

- **422 passed, 2 skipped** in the combined CPU regression run below.
- Skips: CUDA batch-device offset validation and the existing CUDA RoPE migration check.
- Ruff 0.14.2: all changed Python files pass; new Python files formatted.
- `git diff --check`: pass.
- Existing OpenWAM working directory remains clean at the baseline commit.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_touchwam.py tests/test_touchwam_integration.py \
  tests/test_bench2dex_tactile.py tests/test_random_action_initialization.py \
  tests/test_finetune_config_override.py tests/test_openwam_trainer.py \
  tests/test_training_utils.py tests/test_optimizer_groups.py tests/test_checkpointing.py \
  tests/test_repr_contract.py tests/test_deployment_changes.py tests/test_policy_server_cli.py \
  tests/test_obs_preprocess.py tests/test_schedule_equivalence.py \
  tests/test_prepare_accelerate_unwrap.py tests/test_mot_driver.py \
  tests/test_wam_architecture.py tests/test_architecture_variants.py \
  tests/test_public_config_defaults.py tests/test_imports.py \
  tests/model/test_video_only_routing.py tests/model/test_2d_action_mask.py \
  tests/dataloader/test_mixed_training_contract.py \
  tests/dataloader/test_mixture_from_config.py \
  tests/dataloader/test_mixture_trainer_dataloader.py -q --disable-warnings
```

## Direct comparison with the source implementation

On ys, the original `tests.test_touchwam_v2.pair` fixture constructed a small model
with seed 137; its gates were opened using `open_gates`, and its checkpoint was
saved with the original `touchwam_v2` metadata. The same weights were strictly
loaded into the migrated architecture on dev-cpu. Fixed inputs came from the
original `inputs` fixture; losses used seed 913. Generation used schedule
`[(1000,1000),(500,1000),(0,500),(0,0)]`, CFG 2.0 and the same observation.

All **13 compared tensors are exactly equal**, with maximum absolute error zero:
three forward predictions, six total/component losses, generated actions, future
maps, validity and offsets. See [legacy_compatibility.json](legacy_compatibility.json).
The direct comparison uses a tiny video backbone and real ActionDiT. Separate
regression tests run actual small Wan TI2V and ActionDiT forward/backward in FP32
and BF16. This is not a full-size GPU numerical or closed-loop success-rate test.

Temporary reference weights and outputs are retained outside the repository in
`/mnt/pfs/ls3r34/workspace/setup/touchwam_migration_20261023/reference/`.
Only compact verification results and source SHA256s are committed.

## Scope checks

- Existing random ActionDiT initialization, mixed video/action routing, model
  defaults, checkpoint utilities and deployment contracts pass their regressions.
- Alpha -> TouchWAM initialization preserves every base weight and initializes new
  touch weights identically across rank-local RNG seeds without altering the
  caller's CPU RNG state.
- Full V2 finetuning honors the live training policy and retains frozen preprocessors.
- Bench2Dex's shared normalization is saved, verified on resume, and used by the
  existing deploy normalizer to recover the original 58-dimensional actions.
- Synthetic HDF5 tests exercise RGB, action/state slots, equal-task sampling,
  history padding, future offsets, cache equivalence and segment boundaries.

No real dataset, production checkpoint, GPU optimizer resume or robot/simulator
rollout was migrated or executed as part of this code change.
