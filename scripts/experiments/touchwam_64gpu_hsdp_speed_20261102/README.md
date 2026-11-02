# 8 机 64 卡 A800：TouchWAM 生产配方换 HSDP（F1 配置），测速短跑

**对照：** 刚结束的生产训练 r2（`job-v19j6o92tot0`，`setup/touchwam_64gpu_bs48_20261030_r2`，DeepSpeed ZeRO-2）。它的前 1000 步有逐 rank 的 step trace。

**本次配置：** `configs/train_touchwam_six_source_64gpu_bs48_hsdp_20261102.yaml`。
- 与 r2 相同：数据和权重、seed 42、每卡 batch 48、每卡 8 个 worker、前 20 层重算、Alpha + FTP 初始化、学习率调度（`stop_after_steps` 不改变总步数，逐步的数据和 LR 不变）、资源（8×8 A800、116 CPU、928 GiB 内存、128 GiB 共享内存）、镜像、队列、系统包。
- 不同：`training.backend=fsdp2`（HSDP 8 副本 × 8 分片）、F1 的开关（forward prefetch、bf16 规约、本地分片优化器、fused AdamW）、`training.touchwam_p1`（r2 没有，单独约 −1.4%）。
- 300 步，不保存 checkpoint。提交后先跑 NCCL + gloo 探测（gloo 是第一次在这些 pod 上用）。

## 准备（dev-cpu，不申请 GPU）

```bash
cd /mnt/pfs/ls3r34/workspace/OpenWAM-accel        # 干净的 perf/touchwam-fsdp2 检出
python3 scripts/experiments/touchwam_64gpu_hsdp_speed_20261102/pipeline/prepare.py
T=/mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_hsdp_speed_20261102
bash $T/launch.sh preflight     # 配置与 r2 逐项一致、HSDP 只改列出的选项、每个数据源读一个样本
bash $T/launch.sh request       # 只打印作业请求
```

## 提交

```bash
bash $T/launch.sh create
```

## 查看

```bash
bash $T/launch.sh status | logs | progress
bash $T/launch.sh compare       # 与 r2 比较，结果写入 $T/results/baseline_comparison.json
bash $T/launch.sh stop
```

- 输出目录：`outputs/touchwam_64gpu_hsdp_speed_20261102/<timestamp>`。
- 每节点日志和资源采样：`$T/results/speed_node*.log`、`speed_resources_node*.jsonl`。
