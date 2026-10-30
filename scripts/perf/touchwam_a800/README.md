# TouchWAM P0/P1 on 2×8 A800 (a800_1 + a800_2)

**What this measures:**
- **P0:** the production 64-GPU bs48 recipe (`job-v19j6o92tot0`, DeepSpeed ZeRO-2), run on two nodes.
- **P1:** the same recipe with the host-sync removals of `training.touchwam_p1`.

**Everything else is identical:**
- Same model, seed, samples, timesteps and noise per step, and the same LR (production 34,599-step schedule).
- Only the data population is smaller: median-size buckets copied from bdy, one to three per source.
- So P1 must reproduce P0's per-step losses, and any timing difference comes from P1.

## What P1 changes

All switches default to off.

| Switch | Before (each step) | After |
|---|---|---|
| `deferred_modality_sync` | Two blocking checks. A layout all_gather plus `.tolist()` before `prepare_inputs`. An encoder-id `unique()`, an all_reduce and `.tolist()` right after the forward. | Layout: gloo all_gather in the background. Encoder activity: an `[E]` bitmask all-reduced on the stream and copied to pinned memory. Both are resolved between DeepSpeed `engine.backward` and `engine.step`, which is when the optimizer reads them. The first 20 steps stay blocking. |
| `input_validation_steps: 20` | Pure input checks read GPU values back: `.any()`, `.item()`, `unique()`, `masked_select().any()`. | They run for the first 20 steps, then are skipped. Values entering the model are unchanged. |
| `pinned_tactile_h2d` | Every collated tactile tensor is moved with a blocking copy, which syncs the stream. | One non-blocking copy from pinned memory. |

`p1v` adds fewer recompute layers on video-only steps (0–14 instead of 0–19). That is a memory-for-speed trade, so it is a separate run.

## Prerequisites

All of these are already on vePFS:
- Data: `/gaozt-test1/lyh/datasets/touchwam_subset_v1`. Progress is in `.copy_state/pull.log`, which ends with `COMPLETE`.
- FTP-1 encoder: `/gaozt-test1/lyh/models/FTP-1-Tactile-Encoder/d6e5…`.
- Alpha checkpoint: `/gaozt-test1/lyh/models/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors`.
- Code: `/gaozt-test1/lyh/OpenWAM-touchwam`, branch `perf/touchwam-p1`.
- Env: the shared `/gaozt-test1/zhanxch/miniconda3/envs/openwam`. It has torch 2.7.1, DeepSpeed 0.18.9 and accelerate 1.14.0, the same versions as bdy, but Python 3.10 instead of bdy's 3.12. The FTP-1 checksum no longer needs `hashlib.file_digest`, which is 3.11+.

## Running the whole queue

The two nodes cannot ssh into each other. Instead, one worker per node reads the same queue file and the workers meet at file barriers on vePFS. Start both workers from any host that has the ssh aliases. `WAIT_FOR_DATA=1` holds the first run until both copies report `COMPLETE`.

```bash
for r in 0 1; do ssh -n a800_$((r+1)) "cd /gaozt-test1/lyh/OpenWAM-touchwam && (NODE_RANK=$r WAIT_FOR_DATA=1 setsid nohup bash scripts/perf/touchwam_a800/queue_worker.sh scripts/perf/touchwam_a800/queue_p0_p1.txt > /dev/null 2>&1 < /dev/null &)"; done
```

`queue_p0_p1.txt` runs, in order (short runs first, so a broken setup fails early):
1. `nccl`: busbw for world (16), intra-node (8) and inter-node (2) groups, for the ZeRO-2 → HSDP projection.
2. `p0_prof`, `p1_prof`: torch.profiler on 8 steps, ranks 0 and 8.
3. `p0`, `p1`: 1700 steps each.
   - One pass over the subset is ~1724 steps (1.324M samples, 768 per step). So every sample is seen at most once, and there is no epoch-rollover loader restart.
   - The runs cover 98% of the production warmup (1729 steps, peak LR 1e-4), where loss divergence would show.
   - With the subset's modality mix, that is roughly 940 V, 480 VA, 150 VAT and 130 VT steps.
4. `p1v`: 1000 steps. Recompute changes only memory and speed, so this is mainly the per-step peak memory over many batches.
5. `p0r`: P0 again for 500 steps. This is the noise floor: how much speed and loss move between two identical runs.

Expected time:
- Steps take ~4.5 s (V) to ~8 s (VT) here, ~6 s on average. Each run also spends ~5 min starting up plus a ~60 s first step.
- `p0` and `p1` take ~3 h each, `p1v` ~1.8 h and `p0r` ~1 h. The whole queue takes ~9.5–10.5 h.
- A run that fails (e.g. OOM) is recorded in its `.done` file, and the queue moves on to the next one.

Logs:
- Queue logs: `/gaozt-test1/lyh/perf_runs/touchwam_a800/queue.node{0,1}.log`.
- Node logs: `<TAG>.node{0,1}.log`.
- Run directories: `<TAG>/<timestamp>/`.

A run that finished leaves `<TAG>.node*.done`. Delete those files to repeat it.

To run a single job instead of the queue: `bash scripts/perf/touchwam_a800/start.sh <TAG> <CONFIG> [PROFILE=1] [-- overrides]`.

## Reading the results

Every run is cut to the steps all listed runs have, because step k is the same batch in each run. So compare in two passes:

```bash
python scripts/perf/touchwam_a800/compare.py p0 p1 --json /gaozt-test1/lyh/perf_runs/touchwam_a800/p1_1700.json
python scripts/perf/touchwam_a800/compare.py p0 p0r p1 p1v --json /gaozt-test1/lyh/perf_runs/touchwam_a800/p1_500.json
```
- The first pass covers all 1700 steps.
- The second covers steps 1–500 and puts P1's gaps next to the `p0r` noise floor.

**Per modality (V/VA/VAT/VT):**
- Median over steps of the median-rank stage times.
- The fastest-rank forward.
- The slowest-rank iteration.
- Per-step peak memory.

**Overall:** the mean iteration reweighted to the production mix (V 0.30, VA 0.37, VAT 0.27, VT 0.06).

**Losses:**
- Step by step against `p0`. Nondeterministic kernels start at ~1e-5 relative, and in bf16 the gap compounds over hundreds of steps.
- The verdict therefore uses per-modality loss means over 100-step windows (`--window`). A gap clearly above `p0r`'s is a bug.
- Later changes (FSDP2/TorchTitan, P2) are compared the same way against this `p0`.

**Profiler traces:** `<TAG>/<timestamp>/torch_profile/rank{0,8}.json.gz`, plus op tables in `rank*_ops.txt`.

## How the subset stays readable

Production readers bind the full published population. The subset therefore needs a few extra pieces. `prepare_data.sh` runs them once, before the first queued run:

- **Publication contracts:** turned off with `require_publication: false` on every source.
- **EgoDex:** the reader indexes `episodes.parquet` globally and needs every listed video. It reads `egodex_300h_video_v1_sub` instead, built by `make_egodex_subset.py`. That root holds hardlinks of the copied videos and the matching filtered index.
- **RoboCOIN and AgiBot:** the pooled stats validate every contributor bucket's `meta/`. So all of `meta/` was copied (RoboCOIN 199 MB, AgiBot 26 MB), and `include_buckets` picks the buckets whose data was copied: four for RoboCOIN, two for AgiBot. AgiBot uses the generic LeRobot v3 root mode, which already had `include_buckets`; it was added to the RoboCOIN root mode.
- **SharpaDex history:** the manifests bind each video's `mtime_ns`. `sync_mtimes.py` restores the bdy modification times.
- **TouchScale:** the frozen selection covers every episode, so it is dropped.
- **Bucket discovery:** FTP sources and TouchScale find buckets with `rglob`, so they see exactly what was copied.

## Caveats

- **Memory:** two nodes shard ZeRO-2 16-way rather than 64-way, so the optimizer state costs about 4–5 GiB more per GPU than in the 64-GPU job. The production peak of 74.3 GiB reserved can therefore approach 80 GiB here.
  - If P0 OOMs or slows down near the limit, rerun P0 and P1 with the same override, e.g. `training.gradient_checkpointing_layers=[0,...,21]`. The comparison stays fair.
  - For p1v, fitting here implies fitting on 8 nodes.
- **No GA4 run:** a GA4 run (global batch 3072 on 16 GPUs) would not reproduce production semantics. Every micro-step has one global modality, so an accumulation window would mix four modalities in one optimizer step. P0 at GA1 is therefore the loss reference for the later TorchTitan/FSDP2 port (P2).
- **gloo (`deferred_modality_sync`):** this path uses a gloo group, so `GLOO_SOCKET_IFNAME` must name a routable interface (eth1 here). On bdy, set it before using P1 in production.
- **Before the copy finishes:** `link_complete_buckets.py` builds a symlink tree of the finished buckets. Point `TOUCHWAM_SUBSET_ROOT` at it to start short runs early.
