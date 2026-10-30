# TouchWAM 64 卡训练速度审计（2026-10-30）

## 结论与边界

审计对象是正在运行的 `job-399y3w4cxhmg`，冻结代码 `d004f3b`；对照为
`OpenWAM-accel` 的 `b326236` 及其实际 64 卡 S9 短跑记录。
本次只读取代码、配置、checkpoint 验证摘要和日志，没有修改或重启训练。

未发现 S9 核心加速整体漏合并或未启用。当前吞吐量确实低于原 S9 的 batch-48
三源实验，但不能直接归因于触觉：batch、重计算策略、数据配方、优化器分组和
分布式执行路径均不同。已有证据明确显示触觉开销、无触觉步骤的固定开销，
以及少数长步同时存在。尚无同一批输入上关闭触觉的 64 卡 A/B trace，因此
不能给出各因素严格相加的因果贡献。

## 比较口径

| 实验 | GPU / 每卡 batch | 重计算 / 输入 | 耗时和吞吐 |
|---|---|---|---|
| OpenWAM S9 原始实测 | 64 / 48 | 前 20 层重计算，真实三源 loader | steps 21–300 平均 5.783 s，531.2 样本/s |
| TouchWAM 容量测试 | 8 / 12 | 全关，缓存的真实窗口 fixture | 来源加权中位数 1.513 s，约 63.47 样本/s（仅 8 卡） |
| TouchWAM 正式续训 | 64 / 12 | 全关，六类数据真实 loader | steps 201–640 平均 2.5208 s，304.7 样本/s |

正式训练在这个固定的 440 步窗口中，逐步耗时中位数为 2.1945 s。200–650 步
各 50 步窗口的 `iteration_cpu_s` 中位数平均为 2.2532 s。
之前引用的 2.29 s 来自窗口中位数，不能当作含长步的墙钟平均值。
按实际平均吞吐，当前比 batch-48 S9 低约 42.6%；处理相同 3072 个样本约需
4 × 2.5208 = 10.083 s。两者没有相同的训练目标和输入分布。

当前 epoch 为 138386 × 768 = 106280448 个采样窗口位置；原 S9 为
23845 × 3072 = 73251840，当前约为其 1.451 倍。这是 sampler 的训练窗口数，
不是独立轨迹数或新增原始数据量。总时长还受这个训练量差异影响。

## 加速合并与运行状态

- 冻结 release 的 Wan 目录中，53 个共同 Python 文件有 52 个逐字节相同。
  唯一变化是 `models/dit.py` 支持 TouchWAM 的精确分块 attention mask。
- `fused_block.py`、`models/vae.py`、`compile_options.py`、
  `deferred_allgather.py`、`step_timing.py` 与原 S9 相同。
- MoT driver 的变化是补上 no-action 路径的 clean-prefix mask；split attention
  仍启用，并非退回整体 masked attention。
- 实际启动配置及日志确认 text longest/dedup、worker uint8、VAE cuDNN /
  channels-last / fused norm、逐帧调制、Wan block fusion、通信 overlap、
  deferred parameter all-gather 均启用。
- 64 个进程均记录了异步 all-gather 安装；DeepSpeed throughput timer 的强制
  synchronize 已关闭。未发现 Dynamo cache-limit/recompile-limit/fallback 警告。
  日志未报警不等于 kernel trace 证明每个动态分支始终命中编译缓存。
- 关闭全部重计算是当前配置的明确选择，不是漏合并。

## 分模态观测（固定窗口 steps 201–640）

| 批次模态 | 步数 | 中位数 s/step | 平均 s/step |
|---|---:|---:|---:|
| 纯视频（EgoDex） | 107 | 1.718 | 2.061 |
| 视频＋动作，无触觉 | 167 | 2.101 | 2.675 |
| 视频＋触觉（TouchScale） | 53 | 2.276 | 2.507 |
| 视频＋动作＋强度/形变触觉 | 107 | 2.438 | 2.753 |
| 视频＋动作＋RGB 触觉 | 6 | 2.480 | 2.403 |

带触觉与无触觉的中位差约 0.34–0.56 s，但这些不是相同样本的开关实验。
提示词、动作分支、区域数、图像内容均不同；RGB 组仅 6 步，不能推广。
触觉图像编码、历史/未来 token 和 bridge 确实增加计算，不能仅凭新增参数量
推算耗时。当前日志实际记录 touch_v2 总参数 294.4M、可训练 76.6M。

无触觉路径保留 N=0 bypass；它不会执行一套虚构的十指触觉 forward。
因此无触觉步骤也比单机缓存实验慢，不能由“每步都计算触觉”解释。

## 计算阶段与优先排查点

以下为正式任务 steps 200–650 的窗口中位数平均；CUDA event 包括阶段内
等待，阶段嵌套，不能直接相加当作精确总时间。

| 阶段 | 秒 |
|---|---:|
| VAE encode（包含在 prepare/compute_loss 内） | 0.305 |
| T5 encode（同上） | 0.071 |
| prepare_inputs（包含在 compute_loss 内） | 0.383 |
| compute_loss，含输入准备与前向 | 0.914 |
| DeepSpeed backward | 0.921 |
| DeepSpeed step | 0.267 |
| 指标归约 | 0.005 |
| 本 rank 数据等待中位数 | 0.007 |

**优化器与同步开销值得优先 profiling。** 原 S9 `ds_step_gpu_s` 为 0.060 s，
而当前为 0.267 s；同一 TouchWAM 的单机 batch-12 缓存实验约为 0.14–0.15 s。
原配置没有不同 LR 的分组，采用一个扁平 optimizer group；TouchWAM checkpoint
实际有 9 组，分别管理视频、动作、proprio、触觉主体、两种输出表示、图像投影
以及两种矩阵编码器。异步 gather 实现按 group 发起通信，还增加了模态/encoder
一致性 collective。这些是可检查的具体差异，但现有聚合 timer 不能分开 Adam
kernel、参数复制、通信等待和 launch gap，因此 0.207 s 不能全算作分组成本。

分组用于防止未参与当前模态的参数被 Adam momentum/weight decay 更新；不能为
速度直接撤掉。优化应保留 inactive-state 语义并测试 checkpoint 兼容性。

**长步是另一独立问题。** 固定 440 步中有 20 步超过 5 s，累计 149.21 s；
剩余 420 步平均为 2.2856 s，全部步骤平均为 2.5208 s。长步没有固定落在
50 步日志刷新边界，也发生在无触觉批次，因此并非只由触觉或 checkpoint 引起。
该窗口尚未到 2000 步保存点。

数据等待的窗口中位数约 7 ms 不能排除 PFS/视频解码尾延迟：某一 rank 的 loader
慢，会让其他 ranks 在下一次模态检查/collective 中等待，且窗口中位数掩盖尖峰。
当前每卡 4 workers，原 S9 为 8；不能只据此断言增加 workers 一定有效。

## 下一步验证顺序

### 无触觉路径专项核对（10:08 CST，steps 201–1090）

| 模态 | 步数 | 当前中位数 | 当前平均 | 昨日同配置 TouchWAM 中位数 / 平均 |
|---|---:|---:|---:|---:|
| V | 247 | 1.7305 s | 2.1337 s | 1.7263 / 2.1527 s |
| VA | 320 | 2.0997 s | 2.6321 s | 2.0766 / 2.5967 s |

昨日对照为 `outputs/touchwam_six_source_64gpu_bs12_20261029/2026-10-29_23-28-07`，
同为 64 卡、batch12、关闭重计算，取完全相同的步数范围。约 1% 的速度差异不
支持“checkpoint 修复/恢复使无触觉路径明显降速”的判断。

原 S9 在 steps 21–300 的 V 批次（127 步）平均 5.1408 s、597.6 样本/s；VA
批次（153 步）平均 6.3161 s、486.4 样本/s。当前 V / VA 实际吞吐约为
359.9 / 291.8 样本/s，两者均低约 40%。但 S9 每卡 batch48 且开启部分重计算，
VA 来源分布也不同，不能把这 40% 当作同配置代码回归的测量。

源码进一步确认：**N=0 只绕过触觉网络，尚未绕过所有触觉管理路径。**

- `RegionalTouchWAMArchitecture.compute_loss` 仍进入 `_condition` 的空布局校验、
  空未来目标/噪声准备；真正的网络 bypass 位于 `forward`。
- loss 侧仍分别归约 V、A、T 的有效样本数。无触觉时 T 的计数为零；纯 V 时 A
  的计数也为零，仍调用归约。
- trainer 仍调用 `global_observation_encoder_ids`，对全零的 encoder 激活向量
  执行 all-reduce，再转回 Python。相比原基础 loss，这两个位置合计增加四次
  all-reduce 调用（V/A/T 计数各一次，encoder 激活一次），不含 ZeRO 自身通信。
- 原 S9 已有模态一致性检查，不能把当前的布局 all-gather 整体算作新增通信。

还核对了实际环境的 DeepSpeed 源码：
`envs/openwam/lib/python3.12/site-packages/deepspeed/runtime/zero/stage_1_and_2.py`。
`scaled_global_norm`（2055 行）遍历每个参数组，并在 `get_grad_norm_direct`
（1897 行）分别做 DP all-reduce。`step`（2098 行）仍逐组整理梯度、调用内部
optimizer 和复制分片，之后把所有参数组传入 all-gather。当前的异步 patch
也逐组发起 gather，没有读取 `touchwam_active_in_window`。因此 9 组都会参与
这些管理路径；inactive Adam 更新被跳过，不代表外层 ZeRO 工作被跳过。
这是已确认的执行差异，具体时间占比仍需要 trace。

567 个无触觉步骤中，29 个超过 5 s。数据等待尖峰/straggler 的根因仍未测定；
这些长步应独立于触觉 forward 分析。

优先优化候选是：在保持全局标注归一化语义的前提下，让 N=0 路径跳过触觉
布局/目标处理及空激活归约，合并有效样本数与分组梯度范数归约，并验证未激活
组的参数同步能否安全省去。参数组本身用于保护 inactive Adam 状态，不应简单
合并成一组。正式任务尚未改动。

### 对照实验

1. 在单独的短任务增加逐 rank、逐 step 的 stage 记录及短 profiler 窗口，标注
   layout collective、encoder collective、tactile encoder、bridge、Adam、gather。
   同时记录数据来源、区域数、文字长度，避免仅看聚合中位数。
2. 相同 64 卡、batch12、相同 VA/V 输入、相同重计算策略下，对比原 S9 与 TouchWAM
   的无触觉路径；再用同一机器人样本对比触觉开/关，量出纯触觉增量。
3. 优先验证多 group 的通信/optimizer 调度和不必要 CPU–GPU 同步，保留 inactive
   参数语义；根据 trace 再决定 worker/预取是否需要修改。
4. 如果固定开销主导，再测试更大的 microbatch 与选择性重计算。以前 batch48
   缓存测量已接近显存上限，不能直接把正式任务改为 48。

本次没有启动额外 GPU 实验，也没有改变正式训练。按 2.5208 s 的实际平均速度，
当前计划的纯训练约 97 小时，checkpoint 和后续长步波动另计；先前按 2.29 s
推算总时长偏乐观。

## 原始证据

- 当前冻结配置：`setup/touchwam_64gpu_bs12_20261030_r3/results/launch_config_resolved.json`
- 当前日志：同目录 `full_node0.log` 至 `full_node7.log`
- 当前逐步指标/计时：`setup/touchwam_64gpu_checkpoint_acceptance_20261030_r2/training_output/2026-10-30_08-40-24/{training_metrics,step_timing}.jsonl`
- optimizer group 实证：`setup/touchwam_64gpu_checkpoint_acceptance_20261030_r2/results/fresh-rank0.json`
- 原 S9：`setup/alpha_native_64gpu_s9_speed_20261029/results/baseline_comparison.json`
- 单机 TouchWAM：`setup/touchwam_bs12_norecompute_20261029/capacity/run/*/step_timing.jsonl`

以上远程相对路径均位于 `dev-cpu:/mnt/pfs/v89atp/workspace/`。
