# 8 机 64 卡 A800：alpha_native 正式配方 + s9 加速，测速短跑

**对照：**2026-10-26 起运行的正式训练 `alpha_native_64gpu_aug`。
- 实验目录：`setup/sharpa_native_64gpu_aug_20261026`。
- 输出：`outputs/alpha_native_64gpu_aug/2026-10-26_23-07-36`。
- 速度：第 21–22837 步稳定在约 10.0 s/step（中位 10.35），几乎没有等数据的长步。

**本次测试的配置：**`configs/train_alpha_native_64gpu_aug_s9.yaml`。
- 正式配方本身完全不变：
  - 数据：同样的三源数据与权重，原始 SharpaDex TCP。
  - 三源都开 ColorJitter。
  - Alpha 全量初始化，seed 42。
  - 每卡 batch 48，GA 1，全局 batch 3072。
  - 一个 epoch 的 cosine 调度，共 23,845 步；每卡 8 个 DataLoader worker。
- 只额外打开在 2×8 A800 上验证过的 s9 加速选项：
  - 来自 s7d：overlap_comm、text 按最长 prompt 补齐并去重、MoT split attention。
  - 来自 s7p：worker 内把帧堆成 uint8。
  - 来自 s8a/s8b：VAE 的 cuDNN autotune 和 channels_last、逐帧 time modulation。
  - gc20：前 20 层重算。
  - 来自 s9：Wan block 编译融合、VAE 融合 norm、参数 all-gather 与下一步重叠。

**短跑方式：**`training.stop_after_steps` 在第 N 步（默认 300）退出。
- 学习率调度仍按完整的 23,845 步计算，所以逐步的数据顺序和 LR 与对照完全相同。
- `compare_with_baseline.py` 逐步核对：
  - sample id 和 LR 是否一致；
  - 每 50 步的 loss 差；
  - 第 21 步之后的 s/step、加速比、按全程 23,845 步折算的耗时；
  - 各阶段耗时、data wait、每节点 CPU 占用（cgroup）。
- 测试不保存任何权重。

资源与对照一致：每节点 8×A800、116 CPU、928 GiB 内存、128 GiB 共享内存，开启 RDMA；镜像、队列和系统包相同。

## 完整训练

`prepare.py --steps 0` 冻结一个完整训练目录（建议 `--dest .../setup/alpha_native_64gpu_s9_full_<日期>`）：
- 跑完整的 23,845 步，不设 stop_after_steps。
- 按配置每 2000 步保存一次权重和恢复状态，全部保留；输出到 `outputs/alpha_native_64gpu_aug_s9/<timestamp>`。
- 结果文件以 `full_` 开头，其余命令（preflight / create / status / progress / compare / stop）用法与测速相同。

## 准备（dev-cpu，不申请 GPU）

```bash
cd /mnt/pfs/ls3r34/workspace/OpenWAM-accel        # 干净的 perf/train-accel-3src 检出
python3 scripts/experiments/alpha_native_64gpu_s9_speed_20261029/pipeline/prepare.py --steps 300 --workers 8
T=/mnt/pfs/v89atp/workspace/setup/alpha_native_64gpu_s9_speed_20261029
bash $T/launch.sh preflight     # 配置与对照逐项一致、s9 只改加速项、三源各读一个样本
bash $T/launch.sh request       # 只打印作业请求
```

`prepare.py` 的作用：
- 把当前提交冻结到 `$T/release`，并把 `settings.json` 和 `results/release_manifest.json` 一起写入 `$T`。
- 已存在的目录不会被覆盖。要改步数或 worker 数，请换一个 `--dest`。

## 提交

```bash
# 等对照正式训练结束（每 5 分钟查询一次）后自动提交测速作业：
nohup bash $T/launch.sh wait-submit > $T/wait_submit.log 2>&1 < /dev/null &
# 或在队列空闲时直接提交：
bash $T/launch.sh create
```

## 查看

```bash
bash $T/launch.sh status | logs | progress
bash $T/launch.sh compare       # 结果写入 $T/results/baseline_comparison.json
bash $T/launch.sh stop
```

- 输出目录：`outputs/alpha_native_64gpu_aug_s9_speed/<timestamp>`，内容包括 `metrics.jsonl`、`step_timing.jsonl`、`debug_loss_history.csv`。
- 每节点的资源采样：`$T/results/speed_resources_node*.jsonl`。
