# TouchWAM six-source recipe, 2026-10-29

Use `train_touchwam_six_source_20261029`. This recipe preserves the full
OpenWAM Alpha initialization, pretrained FTP observation encoder, and existing
regional TouchWAM model. Earlier recipes are unchanged.

## Dataset population and sampling

| Dataset | Selected hours | Training modalities | Sampling share |
| --- | ---: | --- | ---: |
| EgoDex | 300.000000 | V | 36.769% |
| TouchScale | 99.650796 | VT | 12.213% |
| Native SharpaDex | 242.136528 | VAT | 29.677% |
| AgiBot | 136.188889 | VA | 16.692% |
| RoboCOIN | 20.000157 | VA | 2.451% |
| FTP-1 with verified actions | 17.935301 | VA and VAT | 2.198% |
| **Total** | **815.911671** | | **100%** |

The manual mixture weights are physical hours, not native frame counts.
FTP recordings have different FPS; proportional window-count sampling would
otherwise assign different probabilities to the same duration of footage.
These are source-selection probabilities; globally homogeneous modality batches
and integer rounding can slightly change finite-run realized percentages.

EgoDex retains the previous 300h index, AgiBot its accepted release, and Sharpa
its accepted train split. Sharpa validation episodes remain excluded. RoboCOIN
retains all 15 robot types and their separate normalization statistics.

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
For a deliberately budgeted training run, use the same wrapper with `train`
and the desired Hydra schedule overrides, e.g. `training.max_steps=<microsteps>`.
Resources and `NNODES`/`NPROC_PER_NODE`/`NODE_RANK` follow `scripts/train.sh`.
The retained default is one virtual epoch; choose a run budget before submitting
a long job. This change prepares the recipe; it does not submit formal training.

All six-source readiness evidence is collected under
`/mnt/pfs/v89atp/workspace/setup/touchwam_six_source_ready_20261029`.
The bounded GPU validation uses seven real reader fixtures (including two
optical encoders), full Alpha/T5/VAE, and eight GPU ranks. It repeats one window
per source across ranks to exercise the real training path with a small,
explicit budget; this does not measure long-run stability or model quality.
