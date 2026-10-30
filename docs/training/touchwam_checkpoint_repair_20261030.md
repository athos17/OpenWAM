# TouchWAM checkpoint serialization repair and 64-GPU acceptance

The first production job, `job-tq51vxdotsub`, completed 2,000 optimizer updates
but failed on 2026-10-30 at 00:52 CST when writing its first checkpoint.
`dict(DictConfig)` had preserved nested Hydra containers in the heterogeneous
tactile encoder specification. Serializing that specification into safetensors
metadata raised `TypeError: Object of type DictConfig is not JSON serializable`.
No usable weights or optimizer checkpoint was published by that run.

## Repair

Commit `ad2dd0b` recursively resolves and detaches Hydra containers, including
nested dictionaries and lists, before encoder-spec validation. It also avoids
mutating the source configuration while normalizing source-to-encoder mappings.
Regional model construction evaluates checkpoint metadata before training can
start, so an unsupported metadata value fails early.

The metadata format, model topology, pretrained initialization, data mixture,
optimizer and loss definitions are unchanged. No data was reconverted.

Regression checks passed: 89 related local tests (with oneDNN disabled for this
workstation's CPU BF16 backward limitation), and 62 core tests in the actual
dev-cpu training environment. These include Hydra-backed heterogeneous model
save/load, strict metadata roundtrips, checkpoint retention, atomic completion
markers and real Accelerate optimizer/scheduler resume. The full production
tactile specification was also serialized into a small safetensors header;
that header check does not replace full-model GPU checkpoint IO.

## Acceptance procedure

The bounded job uses 8 nodes x 8 A800 80GB, batch 12 per GPU, global batch 768,
BF16/ZeRO-2, full OpenWAM Alpha and FTP weights, and the actual full population
reader and sampler. Activation recomputation is forbidden at runtime.

1. All 64 ranks pass the NCCL collective and DDP backward probe.
2. Train through step 120, with full checkpoints at steps 60 and 120.
3. Exit all training processes, then start a new 64-rank invocation.
4. Load step 120 and compare deterministic samples from every model/optimizer
   tensor, exact scalar optimizer fields and scheduler state, separately on
   each rank. This is a sampled state diagnostic, not a full-tensor checksum.
5. Continue through step 140 and save both deploy weights and full resume state.
6. Require reports from all 64 ranks and all 8 nodes, continuous optimizer
   counters, finite metrics, all 14 enabled sources, and all four modalities.

The short stop does not shorten the cosine schedule: it retains the production
138,386-update horizon and 6,919-update warmup. A production continuation uses
the accepted full state, retaining those updates, and restores the normal
2,000-step save interval with no short-run stop.

`scripts/validation/verify_touchwam_checkpoint_acceptance.py` produces the
acceptance report only after these requirements pass. The production resume
launcher verifies that its model, reader, trainer and configuration files are
byte-identical to the accepted release.

The first acceptance attempt, `job-hdqaa2ey6phg`, stopped before its first
training step because the validation wrapper indexed a three-value return at
index 3. Commit `d004f3b` fixes the wrapper and adds interface/state-diagnostic
tests (both passed locally and on dev-cpu). Its replacement is
`job-7r68sw9rzzba`, with frozen release `d004f3b`, under
`/mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_checkpoint_acceptance_20261030_r2`.

## Monitoring

Commit `272e03b` adds failed/completed node counts, recent resource-node counts,
and metric-file age to the independent TensorBoard bridge. A reported node
failure is published once and terminates the bridge; stale GPU observations
must not be mistaken for continued training. The original failed run's terminal
health was published, and its old bridge was stopped without stopping the
shared TensorBoard server.

Acceptance completion and production submission must be confirmed from their
saved reports; the existence of a prepared launcher alone is not a successful
acceptance or a submitted training job.

## Verified acceptance result

At 09:07 CST, all 8 pods of `job-7r68sw9rzzba` had succeeded. All 64 fresh and
64 resumed rank reports passed. The verifier published
`results/checkpoint_acceptance.json` with `status: passed`.

- Real updates 1–120 completed, followed by a new process invocation that
  restored update 120 and completed updates 121–140.
- All 14 enabled sources were sampled; V, VA, VT and VAT paths and RGB tactile
  supervision all executed, with finite metrics throughout.
- Weights and complete resume state were published at updates 60, 120 and 140.
- Each weights artifact is about 23.66 GiB and contains 2,874 tensors. The
  complete state at update 60 is about 103.91 GiB, including 64 optimizer shards
  and 64 RNG files, with a final atomic `trainer_state.json` completion marker.
- Every rank's sampled model/optimizer tensors and exact scalar/scheduler state
  matched after loading update 120. The model samples also agreed across all
  64 ranks before the restart.

The production continuation uses the same frozen commit `d004f3b`
under `/mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_bs12_20261030_r3`.
It resumes the validated update-140 state, removes the short-run stop, and uses
the original 2,000-update checkpoint interval and three-checkpoint retention.
The output is also available through the symlink
`/mnt/pfs/v89atp/workspace/outputs/touchwam_six_source_64gpu_bs12_20261030`.
This links to the validated run; it does not duplicate checkpoint data.

## Production continuation

Job `job-399y3w4cxhmg` (`touchwam-64gpu-bs12-20261030-r3`) was submitted at
09:08 CST after acceptance passed. Its 64 ranks subsequently passed the NCCL
probe. It uses batch 12, accumulation 1, peak LR `5e-5`, no recomputation, and
the original one-epoch horizon. The resolved launch settings, including the
resume path and disabled fresh-initialization flag, are recorded in
`results/launch_config_resolved.json` in the r3 setup directory. The run's
original `config.yaml` records its initial short phase; the live production
training overrides remove that stop and restore the 2,000-step save interval.

The shared dashboard remains <http://127.0.0.1:16006/#scalars>:

- `touchwam-64gpu-bs12-r3`: accepted short phase followed by production updates.
- `touchwam-64gpu-bs12-r3-system`: production node/GPU resource and health data.
- `touchwam-64gpu-bs12`: the earlier failed 2,000-step run, preserved for review.

The new CPU bridge is recorded in the r3 setup's
`tensorboard_monitor/services.json`; the existing shared TensorBoard server and
local tunnel are reused. No additional dashboard server is needed.

## Verified production startup

At 09:30 CST, all 8 production pods were Running and all 8 node logs had reached
update 219. Rank 0 confirmed restoration at `global_step=140 opt_step=140`.
The first persisted production metrics covered every update from 141 through
190 without gaps; optimizer counters matched and every numeric metric was
finite. This window exercised V (12 updates), VA (23), VT (4) and VAT (11).
No node failure, traceback or out-of-memory error was present.

The 64-rank timing window ending at update 200 covered 50 updates: median rank
iteration time was 2.305 s and aggregate throughput was about 333 samples/s.
Maximum allocated CUDA memory was 54.03 GiB and maximum reserved memory was
58.46 GiB; the sampled device-level usage was approximately 59–60 GiB per GPU.
These are early production measurements, not an end-to-end epoch benchmark.

`results/production_startup_verified.json` records the startup checks. The
shared TensorBoard API returned new optimizer updates through 190 for
`touchwam-64gpu-bs12-r3`; its lag behind the node logs reflects the metric-file
flush interval and bridge polling. Production continues toward 138,386 updates.
The running release remains frozen at `d004f3b`; subsequent documentation-only
commits do not modify that release.
