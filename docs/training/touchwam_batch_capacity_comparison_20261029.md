# TouchWAM batch / recomputation capacity comparison — 2026-10-29

All three bounded resource-pool jobs succeeded on one node with eight NVIDIA A800 80GB GPUs. Each ran the full TouchWAM model, full Alpha initialization, the pretrained FTP encoder, real T5/VAE preprocessing, BF16, ZeRO-2 and 56 optimizer updates per rank. Gradient accumulation was 1. Production defaults remain batch 8 and peak LR 1e-4; this benchmark branch does not switch the training checkout or start a production run.

Batch 8 and 12 disabled every explicit activation-checkpoint path and rejected checkpoint calls at runtime. Batch 48 restored the original S9 policy: outer checkpointing on, full-layer recomputation for layers 0–19, no separate video-only override, mixed-attention checkpointing on, and CPU offload off. The runtime tracer observed 1,768 checkpoint calls on every batch-48 rank.

| Per-GPU batch | Recompute | Weighted cached iteration | Estimated 8-GPU samples/s | Peak allocated / reserved GiB | Job |
| ---: | --- | ---: | ---: | ---: | --- |
| 8 | off | 1.164 s | 55.00 | 51.29 / 54.49 | `job-1kwse33hujfg` |
| 12 | off | 1.513 s | 63.47 | 60.18 / 63.72 | `job-stm3c9kq4vli` |
| 48 | S9 | 5.614 s | 68.40 | 76.51 / 78.10 | `job-9ch1002apm15` |

Batch 12 improves estimated recipe throughput by 15.4% over batch 8. Batch 48 improves it by 24.4% over batch 8 and 7.8% over batch 12. TouchScale reverses that trend: batch 48 reaches 48.05 samples/s versus 55.70 at batch 12 (13.7% slower). The measured card exposes 79.325 GiB; batch 48 reserves up to 78.102 GiB in PyTorch alone, leaving little room for longer/different inputs and allocations outside PyTorch. These samples passed; this is not a population-wide memory upper bound.

## Stable source timings

Each cell is cached iteration seconds / aggregate 8-GPU samples per second.

| Source | Batch 8, no recompute | Batch 12, no recompute | Batch 48, S9 recompute |
| --- | ---: | ---: | ---: |
| robocoin | 1.146 / 55.87 | 1.466 / 65.49 | 5.216 / 73.62 |
| egodex | 1.016 / 63.00 | 1.310 / 73.28 | 4.650 / 82.59 |
| sharpadex | 1.319 / 48.53 | 1.710 / 56.14 | 6.357 / 60.41 |
| agibot | 1.163 / 55.01 | 1.523 / 63.02 | 5.324 / 72.13 |
| ftp_freetacman | 1.221 / 52.40 | 1.562 / 61.45 | 5.357 / 71.68 |
| ftp_motiontrans | 1.150 / 55.65 | 1.479 / 64.92 | 5.076 / 75.65 |
| ftp_rdp_bimanual | 1.206 / 53.05 | 1.532 / 62.67 | 5.218 / 73.59 |
| ftp_rh20tcfg5franka | 1.132 / 56.52 | 1.445 / 66.42 | 4.956 / 77.49 |
| ftp_rh20tcfg6atiaxia | 1.137 / 56.27 | 1.458 / 65.83 | 4.995 / 76.87 |
| ftp_rh20tcfg7tactile | 1.146 / 55.85 | 1.472 / 65.22 | 5.045 / 76.11 |
| ftp_vla_touch | 1.215 / 52.68 | 1.553 / 61.82 | 5.295 / 72.53 |
| ftp_vitamin | 1.218 / 52.52 | 1.550 / 61.93 | 5.268 / 72.89 |
| ftp_exumi | 1.210 / 52.91 | 1.531 / 62.72 | 5.195 / 73.91 |
| touchscale | 1.268 / 50.46 | 1.723 / 55.70 | 7.992 / 48.05 |

## Measurement scope

- Four explicit coverage cycles over 14 enabled sources; 8, 12 or 48 distinct real reader windows per source, according to batch size. Windows repeat across ranks and cycles. The source configuration and sampling procedure match, but individual selected windows and prompt lengths can differ across batch sizes.
- Discard the first cycle. Take the maximum rank time for each iteration, then the median across the remaining cycles (three values per source, two for the final TouchScale source because no next iteration follows the last step).
- StepTimer iteration time at step k describes compute/update at k−1 plus logging and fetching batch k. Analysis assigns it to k−1. It includes frozen encoders, forward, backward, optimizer, collective operations and cached-loader overhead.
- Source weighting uses the current physical-hour recipe. Weighted time is the weighted mean of per-source medians; estimated throughput is global microbatch divided by that time. This is an estimate from source-by-source measurements, not a timed randomly mixed production epoch.
- Decoded fixtures replace the production loader. Full-population PFS reads/video decoding, checkpoint save/resume, and 64-rank inter-node traffic are not benchmarked. Do not multiply these results by eight and present them as measured 64-GPU throughput. ZeRO-2 also changes per-rank optimizer/gradient memory with world size.
- All ranks completed all updates with finite recorded losses and gradient metrics. The runs remained early in warmup and do not establish convergence or validate a peak learning rate.

## Learning-rate recommendation

For the intended 64-GPU run, effective batch = per-GPU batch × 64 × accumulation. The previous saved pretraining run used batch 48, accumulation 1, effective batch 3072 and peak LR 1e-4.

| Per-GPU batch | Effective batch at 64 GPUs | Suggested initial peak LR |
| ---: | ---: | ---: |
| 8 | 512 | 4e-5 |
| 12 | 768 | 5e-5 |
| 48 | 3072 | 1e-4 |

The smaller-batch suggestions use the AdamW square-root heuristic, LR_new = 1e-4 × sqrt(B_new / 3072), rounded for batch 8. This is a tuning starting point, not a requirement or a convergence result. Keep the reference 5% warmup and cosine schedule initially. Linear scaling would give 1.67e-5 / 2.5e-5, but is not automatically the correct rule for this model. Accumulation 6 at batch 8 or 4 at batch 12 restores the old effective batch, at the cost of more microsteps per optimizer update.

The benchmark configs retain peak LR 1e-4 to isolate the batch/recomputation comparison; no production LR was changed. Given the 8-GPU result, batch 12 without recomputation is a balanced starting point. Batch 48 offers the highest recipe-weighted throughput in these samples but should receive a representative 64-GPU production-loader acceptance check before committing a long run.

## Reproduction and evidence

Frozen code: batch 8 `ddb1276`, batch 12 `baf95b7`, batch 48 `1a4b46e`. The batch-12/48 changes affect the validation harness and candidate configs; model/trainer implementation remains identical. Model-config/checkpoint identity was checked before reusing the relevant complete Alpha tensor audit.

PFS evidence directories under `/mnt/pfs/v89atp/workspace/setup/`: `touchwam_bs8_norecompute_20261029`, `touchwam_bs12_norecompute_20261029`, `touchwam_bs48_s9recompute_20261029`. Reports: `capacity/report.json`, `capacity/steps-rank*.jsonl`, `capacity/run/*/step_timing.jsonl`; combined analysis: `touchwam_bs48_s9recompute_20261029/comparison.json`.

Candidate configs are `train_touchwam_six_source_bs12_norecompute_20261029` and `train_touchwam_six_source_bs48_s9recompute_20261029`. The latter requires `--recompute-policy config` in `scripts/validation/touchwam_capacity.py`. Use separately prepared, matching-size fixtures. The existing production launcher is unchanged.
