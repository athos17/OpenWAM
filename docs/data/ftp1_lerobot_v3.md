# FTP-1 → TouchWAM LeRobot v3

This adapter covers the 14 FTP-1 sources other than `sharpa`. The original tar
archives remain read-only. All nonempty task groups become independent LeRobot
v3 buckets, preserving original episode boundaries. The converter supports a
complete release, but **the current requested scope is per-source pilot examples**.
The full conversion and staging jobs were stopped on 2026-10-28; do not restart
them merely to finish the original census. Future bulk storage is planned on BOS.

The complete census has **366 task groups (363 nonempty), 16,632 episodes,
9,855,802 frames**. The source-specific nominal clocks give **115.65 hours**;
this is exported trajectory duration and does not sum simultaneous sensors.
The 14 archive inputs occupy about **5.67 TB**. Conversion completion is always
read from the publication receipts; these census totals do not mean an ongoing
conversion has already finished.

At cancellation, 143 task buckets / 414,828 frames had complete conversion and
v3 validation receipts; 5 buckets were partial. Completed buckets occupied
130.254 GiB, partial buckets 10.339 GiB, and temporary staged archive ranges
134.537 GiB (logical file sizes). These are historical observations, not current
storage ownership or a cleanup instruction. Staged originals remain present in
the source archive, and their manifests/receipts are sufficient to audit a future
explicit cleanup. No complete source is claimed train-ready from these pilots.

## Frozen source contracts

`configs/data_sources/ftp1/<source>.json` records embodiment, time convention,
raw numeric schema, source/arm identity, frequency evidence and available
supervision. `regions/<source>.json` separately records spatial region mappings
and auxiliary signals. Those semantic definitions do not turn uncalibrated raw
signals into pressure labels.

| Source | Embodiment | Nominal export Hz | Tactile representation |
|---|---|---:|---|
| FreeTacMan | Hand-held single gripper | 30 | Two optical RGB surfaces |
| MotionTrans | Inspire dexterous hand | 20 | Finger/thumb joint force/torque; no spatial map |
| RDP | Single gripper | 24 | GelSightMini, MCTac RGB, grip force |
| RDP_Bimanual | Two grippers | 24 | Same sensors on two manipulators; some arrays are missing chunks |
| RH20TCfg5Franka | Single gripper | 10 | Six-axis wrench |
| RH20TCfg6ATIAxia | Single gripper | 10 | Six-axis wrench |
| RH20TCfg7Tactile | Single gripper | 10 | Two 4×4×3 uSkin arrays and wrench; baseline/sign/calibration required |
| Unit | Single gripper | 10 | One GelSightMini RGB surface |
| Unit_Bimanual | Two grippers | 10 | One GelSightMini surface per gripper |
| VLA_touch | Single gripper | 10 | One GelSightMini RGB surface |
| ViTaMIn | Hand-held single gripper | 60 | Two AllTact RGB views of compliant Fin Ray fingers |
| VisuoTactile_D-WHEEL | Bimanual mobile manipulator | 30 | Two optical surfaces per gripper |
| VisuoTactile_QINGLOONG | Bimanual humanoid | 30 | Two optical surfaces per gripper |
| exUMI | Hand-held single gripper | 60 | One image concatenating two 9DTact encoded deformation representations |

These are **nominal export grids**, not reconstructed exact wall-clock time.
Original timestamps are retained under `observation.ftp1.timestamps`, including
integer frame indices and quantized/low-precision clocks. RH20T raw milliseconds
retain sampling jitter. RDP's float32 epoch clock cannot recover subsecond time.

The FTP-1 author states that exports keep each source's default frequency:
<https://github.com/michaelyuancb/ftp1-policy/issues/9#issuecomment-5437253164>.

exUMI's official GoPro setup QR specifies `p60`; its alignment code timestamps
frames with `start_timestamp + i_frame / fps`, and its replay export keeps every
video frame. The checked QR is saved in `ftp1_evidence/exumi_gopro_60fps.png`.

ViTaMIn's paper specifies a 60 Hz GoPro and 30 Hz tactile cameras. In the actual
FTP export, 16 sampled 7-frame tactile chunks each have three exactly repeated
within-chunk adjacent image pairs, while camera chunks have none. This supports
the original visual clock at 60 Hz; the evidence is saved in
`ftp1_evidence/vitamin_duplicate_audit.json`. The frequency is explicitly marked
as inferred from original acquisition plus the local duplicate-pattern audit.
It is not inferred from the robot's 10 Hz deployment loop.

## What the conversion preserves

Each bucket contains:

```
meta/info.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
meta/source_definition.json
meta/archive_index.json
meta/conversion_complete.json
meta/validation.json
meta/stats.json
data/chunk-000/file-000.parquet
videos/observation.images.<original_camera>/chunk-000/file-000.mp4
videos/observation.tactile_rgb.<original_field>.<sensor>/chunk-000/file-000.mp4
```

The parquet includes standard `timestamp`, `index`, `episode_index`,
`frame_index`, and `task_index`. Every original nonvideo data array is retained
as `observation.ftp1.<name>`. Tensor columns are flattened to one-dimensional
vectors for LeRobot's numeric feature schema; `native_shape` separately preserves
the original sensor tensor axes. Scalar dtype and original values are preserved.
Each raw field and video has a `.valid` column derived from actual chunk
presence. A fill-value array is never silently interpreted as valid sensing.

The video codec is **lossless RGB H.264 (`libx264rgb`, CRF 0)**. This avoids
introducing another lossy transformation to tactile images. Native source
compression artifacts remain. One video file can span several episodes; v3
`from_timestamp`/`to_timestamp` and data offsets identify each episode exactly.
The converter never requires a training worker to scan tar archives.

`camera_layout` in `source_definition.json` maps `head`, `left_wrist`, and
`right_wrist` to actual features. Single-arm tactile sensors use the canonical
**right** gripper A/B regions; dual-arm sources retain left/right identity.
exUMI's `[460,680,3]` image must be split into two `[460,340]` ROIs before mapping
A/B. The original source code rotates the two sensors in opposite directions
before horizontal concatenation.

## Honest supervision boundaries

The current native conversion publishes visual supervision, retains all raw
state/tactile data, and does not create action commands or pressure maps.

- Measured wrist/joint trajectories are **not** verified control commands.
  No `action` column or artificial `state80` mapping is generated.
- Optical RGB grayscale divided by 255 is **not** normal force or indentation.
  A valid sensor calibration or separately approved optical representation is
  required before producing an intensity sidecar.
- ViTaMIn's AllTact camera sees global Fin Ray deformation, so image pixels are
  not already coordinates on a local contact pressure surface.
- exUMI stores an encoded representation: after BGR→RGB, channels correspond to
  `7×positive darkening`, `7×positive brightening`, and current grayscale.
  The original uint8 arithmetic can wrap; arbitrary channel magnitude cannot
  be treated as monotonic contact strength without checking this encoding.
- Wrench, scalar grip force, and joint torque cannot be broadcast into a
  spatial map. uSkin has a spatial grid, but needs a validated zero-contact
  baseline, axis/sign convention and fixed normalization.

Valid future intensity caches must follow `tactile_regions_v1` /
`tactile_intensity_v1` and the shared `openwam.tactile.intensity_store` contract.
Until then, `tactile_enabled=false` is explicit; retained raw data is available
for a later calibrated converter without repeating the tar→v3 conversion.

## Running and resuming

Use a separate tools environment or `pip --target` for `numcodecs`; do not
modify a live training environment. Required packages: numpy, pyarrow, pandas,
numcodecs, imageio-ffmpeg, av. The current server uses OpenWAM's interpreter plus
an isolated `tools/python` target for numcodecs.

```bash
python scripts/data/ftp1/convert.py \
  --input /dataset/huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/main/d287c9c532514a08b2eb29bf1259997bb32f4f3b \
  --output /mnt/pfs/v89atp/workspace/dataset/touchwam_pretrain_v1_20261028/ftp1 \
  --workers 4 --encoder-threads 1 \
  --exclude-sources VisuoTactile_D-WHEEL VisuoTactile_QINGLOONG

python scripts/data/ftp1/validate.py \
  /mnt/pfs/v89atp/workspace/dataset/touchwam_pretrain_v1_20261028/ftp1 \
  --census configs/data_sources/ftp1/archive_census.json --workers 2 --watch
```

Completed videos/tasks are resumed only when source fingerprints and definitions
match. Changing FPS is not a safe metadata-only edit: video PTS, parquet timestamps
and all episode offsets must be regenerated consistently. The time definitions
for exUMI/ViTaMIn were resolved before either source began conversion.

`--sources` and `--limit-groups` are explicit conversion debug/scope controls.
They must not be confused with full-release completion. The validator compares
against the complete 14-source census before a source can be marked ready.

`<source>/publication.json` contains expected/completed/validated group counts,
frames/episodes, errors, time status and available modalities. `ready=true`
requires every nonempty group to be converted and independently validated.
Root publication is ready only when all 14 sources are ready. Partial datasets
must not silently enter full pretraining.

Validation checks all episode/data ranges and indices, parquet lengths, flat
numeric shapes, validity dtypes, prompts, video container frame counts/FPS,
decoded first/last frames, and video offsets. Cached validation binds the
metadata/parquet/video sizes and modification timestamps. Publication requires
the exact census task identities and per-task counts, not only matching totals.
Scalar stats are exact; optional image stats explicitly sample the first and last
frame of each bucket. Tests additionally check a frame-exact lossless roundtrip,
split-tar boundaries, missing chunks and resume safety.

### Bounded PFS staging and CPU queue

`dev-cpu` has an actual cgroup allowance of 8 CPU / 16 GiB; its host-wide CPU count
must not be used to size workers. The two large VisuoTactile sources are handled
by CPU cluster workers. Their source tar archives are not mounted in those jobs.
`stage.py` copies one contiguous task range, retaining original absolute tar
offsets, original archive identity, and a SHA256 for each <=1 GiB segment and
the complete range. Partial segments are recopied; complete segments are checked
and reused. The consumer verifies all hashes before conversion.

`stage_queue.py` bounds ready-but-unfinished task ranges to two. It stages the
smallest tasks first and skips an already completed output only after checking
its source/definition signature and validation. Queue **v2** freezes a `run_id`
and `worker_count`; each task belongs only to `int(task_hash, 16) % worker_count`.
Consumers must supply the matching run identity and an explicit unique
`RANK/WORLD_SIZE` (or CLI equivalents). Mismatches and legacy v1 queues are
rejected before conversion. Local POSIX locks additionally protect same-host
duplicate processes; distributed correctness does not rely on shared `flock`.
Launch exactly one node per rank. Complete video/task receipts allow resume.

```bash
python scripts/data/ftp1/stage_queue.py producer \
  --input "$FTP_SOURCE" --staging-root "$BASE/ftp1_staging" \
  --queue "$BASE/ftp1_staging/visuo_queue" --output "$BASE/ftp1" \
  --definitions "$BASE/tools/ftp1/definitions" \
  --census "$BASE/tools/ftp1/definitions/archive_census.json" \
  --sources VisuoTactile_D-WHEEL VisuoTactile_QINGLOONG \
  --max-pending 2 --segment-mib 1024 \
  --worker-count 2 --run-id example-pilot-run

# The same command runs on each independent 2 CPU / 16 GiB consumer.
python "$BASE/tools/ftp1/scripts/stage_queue.py" worker \
  --queue "$BASE/ftp1_staging/visuo_queue" --encoder-threads 2 \
  --run-id example-pilot-run
```

Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 BLOSC_NTHREADS=1` in both jobs.
The producer needs the archive mount and PFS; consumers require only PFS and
the dependency runtime. The queue exits after producer completion and every task
has a validated completion receipt; three failures on a task stop consumers with
the recorded error. `--max-pending 2` limits concurrent pending ranges, not total
retained storage, since completed staged bytes are deliberately not deleted.

A real 166,993,920-byte RH20T task range was tested with 4 MiB segments and no
original archive mount: conversion signature, complete numeric parquet, and
absolute archive index match the direct conversion. Its independent CPU-cluster
compatibility job `job-o9jpa0ug0qp3` succeeded on 2026-10-28. The production queue
was resized to two consumers to match the pending limit and available CPU quota.
Its current status is external job state, not an assertion that all data is
already published. Persistent queue paths preserve the `/mnt/pfs/v89atp` mount
spelling: resolving symlinks on the login host can produce a different alias
unavailable inside a cluster container. The regression test covers this case.

The cancelled v1 production attempt also recorded `FileNotFoundError` for
`recycle_the_waste_into_the_recycling_bin`'s first `file-000.tmp.mp4` immediately
after ffmpeg completed. A competing writer renaming a shared temporary file is
consistent with that symptom; distributed flock exclusion was not established.
This motivated v2's fixed-rank partition. A regression runs two workers across
40 synthetic tasks with flock disabled, checks every task is written once, and
checks resumed workers skip all completed receipts. No production conversion was
restarted for this test. Existing v1 queues must not be resumed by old workers.

### Small pilot scope and the 200-hour target

The 11 sources with gripper **surface** tactile sensing contain at most
**84.23684 nominal trajectory hours**, of which **74.58089 hours** are optical
and **9.65594 hours** are uSkin. This excludes the 4.46017-hour MotionTrans
dexterous-hand source and 26.94861 hours of RH20T configurations 5/6 containing
wrist wrench without surface tactile arrays. Simultaneous left/right sensors
are counted once. No global content-level duplicate audit has been completed.
Calibration, invalid chunks, and train/evaluation exclusions can reduce these
upper bounds further. The current FTP archive cannot provide 200 independent
gripper-tactile hours: even using every candidate leaves **115.76316 hours** to
obtain elsewhere. Repetition changes training exposure, not unique data hours.

The current converter selects task groups, not episode intervals. To meet the
revised 5–10 episodes/source and <=10 GiB total pilot budget, add a frame-window
view over `Array`: derive `[start, end)` from original `episode_ends`, read only
intersecting original chunks, rebase local frame/timestamp indices, retain
original source episode IDs, and encode only that selected interval. Bind the
selection into a separate `pilot` namespace/signature and publish a scoped pilot
manifest rather than a full-source readiness receipt. Direct archive range reads
avoid copying whole 50–280 GiB tasks to PFS. This episode-selector extension was
scoped but deliberately not implemented as part of the cancelled bulk operation.

## Calibrated intensity/depth adapter

`scripts/data/ftp1/build_intensity.py` converts a **validated** sensor definition
to the shared sidecar format. Supported numeric paths are a measured spatial
normal-axis grid, a calibrated grayscale lookup, or an explicitly verified
monotonic video channel. `depth_reconstruction` additionally loads a fingerprinted
local Python reconstruction backend and fingerprinted calibration assets.

The backend signature is:

```python
reconstruct(*, image, baseline, calibration, options) -> float32[H,W]
```

It must return spatially registered positive indentation depth in declared units.
It must not return a whole-gripper force painted into an image. The adapter then
applies fixed sensor normalization, orientation and area resizing to 32×32.
Every native validity bit is preserved, episode timestamps restart at zero, and
no history/future label crosses episode boundaries. Each region has a completion
receipt for resumable construction, and the manifest is published atomically
only after all regions and episodes validate.

`depth_plugins/dtact_lut.py` implements a **numerical backend only** for a matched
DTact response lookup and zero-contact baseline. It does not assert that the
public repository's calibration belongs to the FTP-1 recording. The example in
`intensity_examples/depth_reconstruction.json` intentionally stays `unverified`
and cannot run until real assets, hashes and calibration evidence are supplied.

RGB reference difference also has an explicit `optical_change_proxy` mode, but
it is disabled unless the definition separately records approved proxy use.
It cannot declare itself normal force. No current source silently enables it.

### uSkin normal-axis evidence and the remaining baseline requirement

XELA's [sensor technology documentation](https://xelarobotics.com/technology/)
explicitly identifies **X/Y as tangential and Z as normal** to the surface. It
also distinguishes calibrated outputs, per-taxel response calibration, and
temperature/magnetic compensation. This supports a Z-normal candidate adapter;
it does not prove that FTP's final array dimension has not been permuted or
that raw readings are zeroed positive contact magnitude.

RH20T's [official modality table](https://rh20t.github.io/) describes fingertip
data as `2 × 16 × 3` at native 200 Hz (FTP resamples to its 10 Hz export grid).
At API commit `aa3124434729ed622109a29b2cbb9f3bbb1c5eeb`,
[`RH20TScene.tactile_data`](https://github.com/rh20t/rh20t_api/blob/aa3124434729ed622109a29b2cbb9f3bbb1c5eeb/rh20t_api/scene.py#L295)
is explicitly raw; tactile preprocessing is listed as TODO and `get_tactile`
raises `NotImplementedError`. The force/torque zeroing code in that API is for
the separate six-axis wrench and must not be applied to the 96 uSkin values.

Actual converted samples confirm a large nonzero sensor offset. For the first
frame of `task_0001_Top-down_button_press`, the two 4×4 grids span X 31,756–33,448,
Y 32,324–33,684 and third-axis 38,281–40,115. In
`task_0006_Gripper_push_block`, the full-trajectory third-axis 1st–99th percentiles
are 38,003–47,862.77. These are retained raw sensor readings, not newtons or
already zero-contact-normalized values.

Before publishing uSkin as `normal_contact_intensity`, verify the exported axis
order and contact sign, identify a **known empty-contact reference for each
taxel/sensor** (including session drift), and freeze a normalization range using
the eventual training split. The existing `spatial_axis` adapter can then use
`max(0, sign × (Z − baseline))` with the fixed range and the A/B region maps.
Absolute force units are not required for TouchWAM's normalized representation;
zero-contact and monotonic normal-contact semantics still are. An arbitrary
first frame or `raw / 65535` is not a validated substitute for that reference.
