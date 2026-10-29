# TouchWAM batch 8 without activation recomputation

The six-source launcher now defaults to
`train_touchwam_six_source_bs8_norecompute_20261029`; its default operation is
still `check`. The data population, full OpenWAM Alpha initialization and FTP
tactile encoder weights are unchanged from the accepted six-source recipe
(including the 200h RoboCOIN release).

## Requested settings

The reference is the saved configuration of the previous eight-node run:
`outputs/alpha_native_64gpu_aug/2026-10-26_23-07-36/config.yaml`.

| Setting | Current value |
| --- | --- |
| Per-GPU batch / accumulation | 8 / 1 |
| Learning rate | 0.0001 |
| AdamW betas / weight decay / gradient clipping | [0.9, 0.95] / 0.01 / 1.0 |
| Schedule | cosine, warmup 0.05, minimum LR ratio 0.01 |
| Precision / optimizer sharding | BF16 / ZeRO-2 |
| Training horizon | one virtual mixture epoch; no explicit max_steps |

At 64 GPUs the global batch is 512, compared with 3072 in the reference run
(48 per GPU). The peak LR is intentionally unchanged; this does not reproduce
the original batch size or exact number of warmup updates. Current video/action
loss weights, timestep range and pretrained ActionDiT initialization are kept.

All explicit activation-checkpoint paths are disabled:

```yaml
training:
  use_gradient_checkpointing: false
  use_gradient_checkpointing_offload: false
  gradient_checkpointing_layers: []
  gradient_checkpointing_layers_video_only: []
model:
  architecture:
    mot_checkpoint_mixed_attn: false
```

The mixed-attention switch is independent of the outer checkpointing switch;
both must be false. Tactile bridge recomputation follows the outer switch.
Other S9 acceleration options remain enabled.

## Actual GPU result

Resource-pool job `job-1kwse33hujfg` ran the frozen validation code `ddb1276`
on **one node with eight A800 80GB GPUs**. All 14 enabled sources each supplied
eight distinct real reader windows. Four coverage cycles completed **56 real
optimizer updates**, with batch 8 on every rank and no gradient accumulation.
Each window was reused across ranks/cycles; this tests model capacity and
modality switching rather than full-population I/O throughput or learning quality.

- Every rank completed the source schedule and all 56 optimizer updates.
- Losses and gradient metrics were finite.
- A runtime guard rejected any call to explicit PyTorch activation checkpointing;
  no such calls occurred.
- Complete Alpha and pretrained FTP initialization assertions passed.
- Maximum allocated GPU memory across ranks/steps was approximately **51.29 GiB**;
  maximum reserved GPU memory was approximately **54.49 GiB**.

| Input family | Maximum allocated memory across tested ranks/steps |
| --- | ---: |
| EgoDex video | 47.40 GiB |
| AgiBot video/action | 49.83 GiB |
| RoboCOIN video/action | 49.51 GiB |
| Sharpa video/action/deform | 51.11 GiB |
| FTP video/action/RGB (maximum among sources) | 49.92 GiB |
| TouchScale video/normal-force matrices | 51.29 GiB |

These values describe the sampled windows, not an upper bound on every possible
prompt/validity/layout. The run did not test 64-rank inter-node communication,
checkpoint writing/resume, or full-dataset loader throughput. Checkpoint I/O was
disabled for this capacity probe. Existing smaller optimizer-resume regression
tests do not substitute for a full-sized checkpoint test.

The capacity harness uses deterministic source coverage and an approximately
207,575-update horizon derived from the 64-card batch-8 population. It stops at
56 updates without shortening that LR horizon. The production config continues
to use its real population sampler and one-epoch schedule. Timings from cached
fixtures should not be used as formal training throughput estimates.

The separate full Alpha audit verified 2,089 tensors bit-for-bit. The new
configuration's production preflight passed all 14 sources with no errors.
Local small-model batch-8 forward/backward checks additionally covered all four
modalities and RGB/intensity codecs while forbidding activation checkpoint calls.

Evidence and fixture provenance are under:
`/mnt/pfs/v89atp/workspace/setup/touchwam_bs8_norecompute_20261029`.
The GPU report is `capacity/report.json`; per-rank per-source measurements are
`capacity/steps-rank*.jsonl`.

## Entry point

```bash
cd /mnt/pfs/v89atp/workspace/OpenWAM-touchwam-training
export PYTHON_BIN=/mnt/pfs/v89atp/workspace/envs/openwam/bin/python
bash scripts/train_touchwam_six_source.sh check
# On allocated GPU workers with the intended distributed environment:
bash scripts/train_touchwam_six_source.sh train
```

The wrapper selects this config's matching full Alpha audit automatically.
For the previous batch-1 S9 recipe, explicitly set
`TOUCHWAM_CONFIG=train_touchwam_six_source_20261029_s9`.
No production long-running training job was submitted by the capacity check.
