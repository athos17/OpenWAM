# Native data reuse and tactile cache

**Current accepted workflow:** reuse `egodex_300h_video_v1`, `sharpadex_native_tcp_v1_20261026/buckets`, and `agibot_pretrain_v1` exactly as prepared for the earlier training run. Duplicate EgoDex/AgiBot publications and the partial full EgoDex copy were deleted at the user's request; do not recreate them. Only the previously missing Sharpa ten-fingertip tactile sidecar is active work. Its standalone `catalog_sharpa_tactile.py` validates against original metadata and publishes an external `tactile_catalog.json` without editing old data. Training and validation use existing PFS datasets.

The conversion tools below describe the retained optional utilities and work performed before that correction; they are not prerequisites or default inputs for this training.

These tools publish EgoDex, SharpaDex and AgiBot without changing the frozen source datasets. Dataset selection and mixture hours remain unset; `configs/data_sources/native/*.json` defines semantics, identities and supervision, not the eventual training allocation.

## Historical optional layouts (not current preparation instructions)

Under the configured release root:

- `agibot/`: existing 75 native LeRobot v3 buckets, independently hardlinked. All 75 buckets in this release are A2D with dual grippers; the general AgiBot reader can also support dexterous variants. Its existing feature-driven base-motion masks remain authoritative.
- `sharpadex/buckets/`: 495 LeRobot v3 buckets referencing `sharpadex/partitions/`. The validated native TCP62 reader scatters into OpenWAM80. New `tactile/manifest.json` and per-episode `maps.npy`, `valid.npy`, `timestamps.npy` are appended to the independent publication only.
- `egodex/full_pool/train/`: all 129 official training partitions, 314,839 episodes, 78,282,306 frames (724.8361667 h at 30 Hz). This is byte-verified PFS staging of the actual v3 release; official test is reserved and excluded. Human transforms are retained as original data, but provide **no robot action or proprio supervision**.
- `egodex/accepted_cache_300h/`: the already accepted video cache materialized as a genuine LeRobot v3 dataset, 130,390 episodes / 32,400,000 frames. It has **no action or proprio columns**. This is a derived subset for smoke tests and an optional later data selection. Never concatenate it with the full pool.

`publication.json` is the release readiness gate. `complete` means the work owned by this publisher has finished; `pending_copy` or `pending_tactile` must fail training preflight. A reader being able to find one valid bucket is insufficient. Full GPU training acceptance is a separate model/runtime check.

## Historical conversion commands (do not rerun for current native data)

Use the prepared Python environment with NumPy, PyArrow, PyAV and OpenCV. Limit BLAS threads before running converters. The server exposes 128 logical CPUs but, when audited on 2026-10-28, `/sys/fs/cgroup/cpu.max` limited this process group to **8 CPU cores**. Size concurrency using the quota, and account for other converters.

```bash
python scripts/data/native/publish_native.py --source OLD_AGIBOT --output RELEASE/agibot
python scripts/data/native/publish_native.py --source OLD_SHARPA_NATIVE --output RELEASE/sharpadex
python scripts/data/native/extract_sharpa_deform.py --dataset RELEASE/sharpadex --workers 8
python scripts/data/native/convert_egodex_video.py --source ACCEPTED_EGO_CACHE --output RELEASE/egodex/accepted_cache_300h
python scripts/data/native/stage_egodex_pool.py --source ORIGINAL_EGODEX_REVISION --output RELEASE/egodex/full_pool --workers 4
python scripts/data/native/finalize_native.py --root RELEASE --definitions configs/data_sources/native --wait
```

`publish_native.py` verifies immutable hardlink identity when resumed. It rewrites internal bucket/statistics symlinks to the new publication. Never edit shared files in place.

EgoDex staging uses atomic `.partial` files and source/destination SHA256 verification. Each bucket has a receipt. A crash between rename and receipt write is recovered by checking both files. A bucket enters `catalog.ready` only after all files, episode counts and Parquet row counts pass. Resume with the same command. Newly written `meta/source_definition.json` files do not modify original metadata.

Sharpa extraction resumes completed manifests only when the code/data signature matches. Interrupted unpublished episodes are regenerated atomically. No data can be considered fully ready before all 495 bucket manifests are present and final readback passes.

## Sharpa deform meaning and mapping

The original 480 × 1200 mosaic has two rows of five 240 × 240 tiles. Top row is **left** hand and bottom row **right** hand. Each row is thumb, index, middle, ring, little from left to right; the canonical registry calls little `pinky`. Preserve each tile's original image orientation. There is no `surface_id`.

This order was independently checked in 20 partitions / 1,598 synchronized samples: each tile's mean grayscale is maximally correlated with its matching finger's absolute XYZ force sum, with diagonal correlations 0.849–0.998. The check supports sensor identity and intensity direction; it is not force calibration. Reproduce with `audit_sharpa_mosaic.py --dataset SOURCE --output AUDIT_DIR`.

The codec is `tactile_intensity_v1`, with sensor identity `sharpa_real`. Decode grayscale video luma, area-resize to 32 × 32 per region, and perform fixed declared-video-range conversion to uint8. Training divides by 255. Black is zero relative contact-deformation response; larger values mean stronger SDK response. This is **not calibrated pressure, force or millimeter depth**. Do not perform per-frame min-max normalization. The fast luma conversion can differ from full-resolution grayscale-then-resize by rounding-level quantization; it preserves the same fixed direction and dynamic range.

A decoded zero tile is a valid zero-contact observation. Missing/decode-failed frames are masked invalid. Installed regions come from the source layout, never inferred from intensity. This cannot detect a source device dropout that was itself encoded as ordinary black video; the source does not provide that validity signal.

Episode source video offsets are preserved exactly. Frames are matched by their PTS on the 30 Hz grid, and sidecar timestamps are episode-relative. The downstream reader samples physical history/future offsets and masks unavailable targets; it must not cross an episode boundary or interpolate from future measurements.

## Validation

`tests/data/test_native_publication.py` covers source immutability, tile identities and extraction masks. `tests/data/test_native_catalog.py` verifies original video offsets, all array headers/lengths/timestamps/masks, and detection of missing, truncated or modified arrays even when metadata was cached. Failed rechecks revoke `training_ready`; an old ready publication must not survive an error.

`catalog_sharpa_tactile.py` records immutable array stat identities (inode, size, mtime, ctime); it does not compute cryptographic checksums of the entire 297 GB map payload. Changes after the receipt are rejected. Before the first receipt, arbitrary valid-shaped uint8 corruption cannot be identified from headers alone.

`validate_sharpa_all.py` independently reads original video PTS at first/middle/last episodes and first/middle/last frames inside each selected episode, up to 9 samples per partition. It derives video paths and offsets from original metadata, crops each of the 10 original tiles independently, and compares every sampled pixel exactly. The frozen extractor's resize function is never called. Every episode also receives structural/time/mask validation. This covers every partition but remains sampled source replay, not every source frame. Receipts bind source metadata/video stat identities, sidecar arrays, and validator code; stale receipts do not count as passed.

```bash
# Pilot: only AUDIT_DIR is written; originals and sidecars remain read-only.
OPENBLAS_NUM_THREADS=1 python scripts/data/native/validate_sharpa_all.py \
  --source EXISTING_SHARPA_NATIVE_ROOT --cache EXTERNAL_TACTILE_ROOT \
  --output AUDIT_DIR --workers 1 --limit 3

# Full validation. Separate CPU nodes may use distinct --shard-index I and
# identical --shard-count N; shards use the sorted ORIGINAL bucket inventory.
OPENBLAS_NUM_THREADS=1 python scripts/data/native/validate_sharpa_all.py \
  --source EXISTING_SHARPA_NATIVE_ROOT --cache EXTERNAL_TACTILE_ROOT \
  --output AUDIT_DIR --workers 2

# Final aggregate requires fresh passed receipts for ALL original buckets.
python scripts/data/native/validate_sharpa_all.py \
  --source EXISTING_SHARPA_NATIVE_ROOT --cache EXTERNAL_TACTILE_ROOT \
  --output AUDIT_DIR --aggregate
```

An aggregate with missing, failed or stale partitions returns exit code 2.
Individual runners return 1 for failures and 2 for pending manifests. Matching
successful receipts are reused; rerun after remaining extraction completes.

For independent CPU allocations, stop the unsharded Sharpa runner first, then
run `run_native_shards.py --kind sharpa --output RELEASE/sharpadex
--shard-index I --shard-count N --workers W` with distinct indices and identical
`N`. Current work relies on disjoint fixed shards, not cross-node advisory-lock
work stealing. Do not run the
older unsharded CLI concurrently on the same output, because that CLI predates
the lock wrapper. `catalog_sharpa_tactile.py` aggregates all Sharpa manifests
regardless of which allocation produced them. The older aggregate native
finalizer and EgoDex sharded launcher are not part of the active workflow.

The resource pool ends peer replicas when the master exits successfully.
`cluster_sharpa_cpu.py` therefore writes per-rank exit receipts under the unique
`AIHC_JOB_ID` and holds rank 0 until every shard completes. Old-job receipts
cannot satisfy this barrier. Scheduler `Succeeded` alone never proves that all
495 manifests exist; the catalog remains the global readiness check.

Current external-cache acceptance command:

```bash
python scripts/data/native/catalog_sharpa_tactile.py --source EXISTING_SHARPA_NATIVE_ROOT --cache EXTERNAL_TACTILE_ROOT --wait
```
