# TouchWAM six-source recipe, 2026-10-29

Use `train_touchwam_six_source_20261029`. This recipe preserves the full
OpenWAM Alpha initialization, pretrained FTP observation encoder, and existing
regional TouchWAM model. Earlier recipes are unchanged.

## Dataset population and sampling

| Dataset | Selected hours | Training modalities | Sampling share |
| --- | ---: | --- | ---: |
| EgoDex | 300.000000 | V | 30.123% |
| TouchScale | 99.650796 | VT | 10.006% |
| Native SharpaDex | 242.136528 | VAT | 24.313% |
| AgiBot | 136.188889 | VA | 13.675% |
| RoboCOIN | 200.000000 | VA | 20.082% |
| FTP-1 with verified actions | 17.935301 | VA and VAT | 1.801% |
| **Total** | **995.911514** | | **100%** |

The manual mixture weights are physical hours, not native frame counts.
FTP recordings have different FPS; proportional window-count sampling would
otherwise assign different probabilities to the same duration of footage.
These are source-selection probabilities; globally homogeneous modality batches
and integer rounding can slightly change finite-run realized percentages.

EgoDex retains the previous 300h index, AgiBot its accepted release, and Sharpa
its accepted train split. Sharpa validation episodes remain excluded. RoboCOIN
now uses `robocoin_200h_v1_20261029`: all original 20h tasks plus approximately
180h of new complete tasks. It retains all 15 robot types, with normalization
statistics recomputed separately for each type over the new 200h population.
See `docs/data/robocoin_200h_20261029.md`. The standalone older 20h recipe and
published data remain available.

FTP's frozen archive contains 25.914875 nominal hours. Only 17.935301h with
verified action profiles enters this recipe: FreeTacMan, RDP_Bimanual,
VLA_touch, ViTaMIn and exUMI use VAT; MotionTrans and the three RH20T sources
use VA. Their raw force/torque or uSkin vectors are preserved in storage but are
not interpreted as scalar pressure maps. Unit_Bimanual, D-WHEEL and QINGLOONG
remain archived until their action semantics are verified. RDP and Unit have
no completed buckets in this snapshot. No FTP robot source is treated as ego.

FTP paths are rooted at
`/mnt/pfs/v89atp/workspace/dataset/ftp1_frozen25h_v1_20261029`.
Its source publications must be complete before training or preflight succeeds;
the recipe never scans the still-growing conversion output.

## TouchScale: freeze the existing 100h release without copying media

The complete accepted TouchScale release contains **99.65079629629629h**:
430 buckets, 15,324 train episodes, 10,762,286 video frames at 30 FPS.
All fit within the 100h budget. There is no duplication or sampling up to 100h.

The exact episode list and bound metadata are stored at:

```
/mnt/pfs/v89atp/workspace/dataset/_manifests/touchwam_five_source_20261029/touchscale_selection.json
SHA256: 4fb3d03113430ec399b74fe069c77d909c7d024ebad8b395fb1f12f0d40dd11d
```

The manifest directory was named before RoboCOIN was added to the recipe;
it identifies this immutable TouchScale selection, not the number of sources.
It records seed 42, the requested 100h budget, actual duration, publication
fingerprint, bucket-relative episode IDs, frame counts, and hashes of the
episode metadata, source definition, info and tactile manifests. The recipe
pins the selection hash. The reader verifies these hashes and selects the
listed episodes after computing physical file offsets and applying the split.
It rejects missing episodes, changed metadata, zero-target episodes, and a
second `total_hours`/`include_buckets` selection. Newly added buckets cannot
silently expand the selected population. Original publication validation still
applies independently.

For a future release, write a **new manifest path** and deliberately update its
recipe hash. This command refuses to overwrite a different existing selection:

```bash
python -m scripts.data.freeze_episode_selection \
  --dataset-root /mnt/pfs/v89atp/workspace/dataset/touchscale_lerobot_v3_full_v1_20261029/buckets \
  --output /path/to/a/new/touchscale_selection.json \
  --max-hours 100 --seed 42
```

The generic selector applies the hour budget once over the global episode
population. If the full train population exceeds the budget, a seeded random
episode permutation is taken until the target is reached, with at most one
episode of overshoot. For this release all episodes are selected. No videos,
parquet frames or native pressure arrays are copied.

TouchScale emits 30 identified hand regions, historical native matrices
`[4,30,1,16,10]` with taxel-validity masks, and future normalized targets
`[8,30,1,32,32]`. It is ego VT and has no fabricated action/proprio supervision.

## Validation evidence

The selection hashes and the original publication were checked on dev-cpu.
Three representative TouchScale buckets were decoded at their first, middle
and last reader windows: all nine real samples passed the production sample
validator and tactile collator. Evidence is in
`/mnt/pfs/v89atp/workspace/setup/touchwam_five_source_ready_20261029/touchscale_acceptance.json`.
The same directory contains `touchscale_tactile_sample.pt` for model smoke tests.

The targeted reader/selection/TouchScale tests passed (20 tests). They include
correct media offsets after episode selection, selection hash and metadata
drift rejection, incompatible split rejection, and protection against a later
bucket addition. Data-reader acceptance alone is not full distributed-training
acceptance; use the final run's combined preflight and model smoke evidence.

## Entry point

On the training host, activate the existing OpenWAM environment and run:

```bash
cd /mnt/pfs/v89atp/workspace/OpenWAM-touchwam-training
export PYTHON_BIN=/mnt/pfs/v89atp/workspace/envs/openwam/bin/python
bash scripts/train_touchwam_six_source.sh check
```

The wrapper selects this frozen recipe and the corresponding full Alpha audit.
It performs real source reads, tactile collation, deployment normalization checks
and the current-model/Alpha binding check. `check` does not start training.
`PYTHON_BIN` also selects the distributed training interpreter, so preflight
and training use the same environment.
For a deliberately budgeted training run, use the same wrapper with `train`
and the desired Hydra schedule overrides, e.g. `training.max_steps=<microsteps>`.
Resources and `NNODES`/`NPROC_PER_NODE`/`NODE_RANK` follow `scripts/train.sh`.
The retained default is one virtual epoch; choose a run budget before submitting
a long job. This change prepares the recipe; it does not submit formal training.

Existing manifests and normalization statistics refer to both PFS aliases,
`/mnt/pfs/v89atp` and `/mnt/pfs/ls3r34`. A new compute container that only mounts
the former must expose the latter as the same directory before launching:

```bash
if [[ ! -e /mnt/pfs/ls3r34 ]]; then
  ln -s /mnt/pfs/v89atp /mnt/pfs/ls3r34
fi
[[ /mnt/pfs/ls3r34 -ef /mnt/pfs/v89atp ]] || exit 1
```

This does not copy data or replace an existing mount. The bounded validation
bootstrap also checks this alias before reading any fixture or model file.

All six-source readiness evidence is collected under
`/mnt/pfs/v89atp/workspace/setup/touchwam_six_source_ready_20261029`.
The bounded GPU validation uses seven real reader fixtures (including two
optical encoders), full Alpha/T5/VAE, and eight GPU ranks. It repeats one window
per source across ranks to exercise the real training path with a small,
explicit budget; this does not measure long-run stability or model quality.

### Original acceptance (20h RoboCOIN recipe)

- Combined preflight passed all 14 reader sources and 120 decoded windows;
  action statistics, tactile collation and the full Alpha audit matched.
- Resource-pool job `job-n5qi0sd67s89` succeeded on 8 A800 80GB GPUs, using
  frozen code commit `055db8394943d2fe828e15cfcbb00d9750993fce`.
- All seven real optimizer updates completed, with finite losses and gradient
  norms. All ranks followed the same modality/codec/observation-size schedule;
  each of the seven fixture sources appeared eight times globally.
- Full Alpha initialization and pretrained FTP encoder assertions passed.
  Peak PyTorch allocated memory was 38.54–38.69 GiB per GPU; reserved memory
  was 39.77–42.70 GiB. These are bounded-smoke measurements, not an upper bound
  for other data windows or a complete training run.
- `full_gpu_smoke/report.json`, `full_gpu_smoke/initialization.json`,
  `gpu_status.json` and `code_manifest.json` retain the evidence. The smoke
  disabled checkpoint writes; exact interrupted/resumed optimizer behavior
  was separately checked by the existing tiny production-trainer validation.
- CPU resource-pool job `job-bjxqv67sxwp8` also succeeded: the complete
  86,837,962-window mixture initialized in 74.97 seconds, and the production
  loader with four workers decoded/validated 16 batches using rank 0's
  eight-rank sampling layout. Parent-process peak RSS was 4.07 GiB. This
  complements the GPU fixture test without loading the model on the CPU job;
  it is not a measurement of eight simultaneous full-population processes.
  Evidence: `population_v2_report.json` and `population_v2_status.json`.

The later launcher-only fix makes explicit `PYTHON_BIN` apply to torchrun as
well as preflight. It does not change the validated model or training loop.
No long training job was started.

### RoboCOIN expansion to 200h

On 2026-10-29 the user increased RoboCOIN to 200h. The inherited S9 recipe
uses the same expanded data configuration. Model settings are unchanged; the
new source selection and statistics intentionally change the training-data
contract. Existing 20h-run full optimizer states cannot be strictly resumed
against this changed population.

The current 399-task release, refreshed source preflight, and new 200h
normalization checks are recorded under
`/mnt/pfs/v89atp/workspace/setup/robocoin_200h_20261029`. The original GPU report
above used the earlier 20h selection; it is not a new GPU run on all 200h.
