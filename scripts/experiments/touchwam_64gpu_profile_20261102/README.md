# 8 机 64 卡 A800：DeepSpeed 与 HSDP 各一次 profile

**目的：** 定位 P5 里两个没解释的问题：反向为什么两个后端都比 2 节点多约 0.7 s，长尾尖峰来自哪里。

**两次运行（配方都是 r2，都开 `training.touchwam_p1`，所以只有后端不同）：**
- `ds`：`configs/train_touchwam_six_source_64gpu_bs48_prof_ds_20261102.yaml`（ZeRO-2）。
- `hsdp`：`configs/train_touchwam_six_source_64gpu_bs48_prof_hsdp_20261102.yaml`（HSDP 8×8 + F1 开关）。

**方式：**
- 45 步，第 23–42 步（共 20 步）开 torch.profiler。第 21 步之后才生效的 P1 延迟同步不受影响；第 43–45 步写 trace 时会变慢。
- 追踪 15 个 rank：节点 0 的全部 8 个 rank（同一时钟，能看出谁迟到），其他节点各 1 个。
- 每步 trace 里新增 `gc_pause_cpu_s`、`gc_gen2_count`（所有 64 个 rank，不依赖 profiler）。
- 不保存 checkpoint。资源、镜像、队列都与 r2 相同。

**不要拿这两次运行的 s/step 当速度结果：** 开 profiler 的步变慢，而且 15 个 rank 慢、其余不慢，会放大 rank 间等待。这里只看各阶段的相对组成。

## 准备（dev-cpu，不申请 GPU）

```bash
cd /mnt/pfs/ls3r34/workspace/OpenWAM-accel        # 干净的 perf/touchwam-fsdp2 检出
P=scripts/experiments/touchwam_64gpu_profile_20261102/pipeline/prepare.py
python3 $P --variant ds
python3 $P --variant hsdp
bash /mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_profile_20261102_ds/launch.sh preflight
bash /mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_profile_20261102_hsdp/launch.sh preflight
```

## 提交（队列一次只放得下一个，先 `ds`，跑完再 `hsdp`）

```bash
bash /mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_profile_20261102_ds/launch.sh create
bash /mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_profile_20261102_hsdp/launch.sh create
```

## 查看

```bash
T=/mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_profile_20261102_ds     # 或 _hsdp
bash $T/launch.sh status | logs | progress | stop
tail -F $T/results/profile_node0.log | tr '\r' '\n'
bash $T/launch.sh analyze       # 结果写入 <输出目录>/torch_profile/summary.json
```

输出目录：`outputs/touchwam_64gpu_profile_20261102_{ds,hsdp}/<timestamp>`。
