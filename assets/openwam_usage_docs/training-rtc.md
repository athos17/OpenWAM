# Marvin Training RTC

实现 Training-Time Action Conditioning for Efficient Real-Time Chunking
(arXiv:2512.05964v2)。首版训练和部署支持 `dual_system/joint_self_attn`。
原有 `sync` / `async` 模式仍可使用旧 checkpoint。

## 默认启动会做什么

本文的“RTC 脚本默认值”指运行 `bash scripts/train_marvin_rtc.sh`、没有额外覆盖时的
**实际生效值**。它与直接运行 `scripts/train.py` 使用的 `configs/train.yaml` 默认值不同。
所有路径占位符都需要替换；示例中的训练预算不是已经验证过的最优超参数。

| 项目 | RTC 脚本默认值 |
| --- | --- |
| 机器数 / 每台训练进程数 | `1` / `4`（通常一进程一 GPU，不自动检测卡数） |
| 每卡 batch / 梯度累积 | `32` / `4` |
| 有效全局 batch | `32 × 4 × 1 × 4 = 512` 个训练样本/权重更新 |
| 训练多久 | `num_epochs=5`，`max_steps=null`，即默认跑 **5 个 epoch** |
| 学习率 | `5e-5` |
| RTC 训练前缀长度 | 每样本均匀采样整数 `d ∈ {0,...,16}` |
| 保存间隔 / 保留数量 | 每 `2000` 个 micro-step 保存；保留最近 `1` 份权重 |
| 计算精度 / ZeRO | `bf16` / `2` |
| 初始化权重 / 数据目录 | **必须提供** `RTC_CKPT_DIR` / `DATASET_DIR`，没有默认路径 |

## 训练参数怎么设置

**入口区别：** `train_marvin.sh` 是原有普通微调脚本，默认不启用 RTC，且默认训练
20 epoch；本文的 5 epoch 等默认值针对 `train_marvin_rtc.sh`。不要混用两者的默认参数。
普通脚本现按项目所在位置选择两台训练机各自的环境、foundation 权重和数据路径；
RTC 脚本仍通过下面的环境变量显式指定路径。

两台训练机使用相同的 `scripts/train_marvin_rtc.sh` 和核心源码。环境变量放在
`bash` 前或提前 `export`；Hydra 配置覆盖放在脚本名后，写成 `training.xxx=value`。
优先级为：**脚本末尾的 Hydra 覆盖 > 脚本设置（包含环境变量取值）> YAML 默认值**。

```bash
export OPENWAM_ENV_PREFIX=/path/to/envs/openwam
export RTC_CKPT_DIR=/path/to/self-contained-marvin-checkpoint
export DATASET_DIR=/path/to/pick_place_2_openwam_v3

# 默认训练：4 GPU、每卡 32、累积 4、5 epoch。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash scripts/train_marvin_rtc.sh
```

| 机器 | 项目目录（先 cd 到这里） | `OPENWAM_ENV_PREFIX` |
| --- | --- | --- |
| `ssh h20` | `/gaozt-test1/zdx/OpenWAM` | `/gaozt-test1/zhanxch/miniconda3/envs/openwam` |
| `ssh zhangdaoxuan@10.249.43.100` | `/data_all/zhangdaoxuan/vla-base-model/OpenWAM` | `/data_all/zhangdaoxuan/miniconda3/envs/openwam` |

### 启动脚本环境变量

| 环境变量 | 默认值 | 含义与设置方法 |
| --- | --- | --- |
| `RTC_CKPT_DIR` | 无，必填 | 自包含 checkpoint 目录，含权重、`config.yaml`、归一化统计；映射到 `training.finetune_ckpt_path` |
| `DATASET_DIR` | 无，必填 | 转换后的 Marvin 数据集目录；映射到 `dataloader.dataset_dir` |
| `OPENWAM_ENV_PREFIX` | 不设置 | 设置后将其 `bin` 加到 PATH 前端；否则使用当前 PATH 中的 `python` |
| `CUDA_VISIBLE_DEVICES` | 脚本不设置，继承当前环境 | CUDA 可见的 GPU，如 `0,1,2,3`；先用 `nvidia-smi` 确认可用卡 |
| `NPROC_PER_NODE` | `4` | 每台机器训练进程数；通常等于本次使用的 GPU 数，不能超过可见 GPU 数 |
| `NNODES` | `1` | 同一个分布式训练作业的机器数 |
| `NODE_RANK` | `0` | 当前机器在同一个作业中的编号，从 0 开始 |
| `MASTER_ADDR` | `127.0.0.1` | 分布式 rank 0 地址；单机保持默认，多机必须填所有节点可访问的 rank 0 IP |
| `MASTER_PORT` | `29500` | 分布式通信端口；同机并行启动独立作业时使用不同端口 |
| `LEARNING_RATE` | `5e-5` | 基础学习率；对应 `training.learning_rate` |
| `BATCH_SIZE` | `32` | **每卡**每个 micro-step 的样本数；对应 `training.batch_size` |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | 累积多少个 micro-batch 再更新权重；对应同名小写 training 配置 |
| `MAX_STEPS` | `null` | `global_step` 上限，单位为 **micro-step**；对应 `training.max_steps` |

两台机器同步代码后，默认仍是各自独立的单机训练，`NNODES=1`。
代码同步不会将两台机器自动组成一个作业，也不会启动训练。
只有确实启动同一个多机作业时才设 `NNODES=2`，分别设 `NODE_RANK=0/1`，
并统一 `MASTER_ADDR` 和 `MASTER_PORT`，确保数据、初始化权重和配置一致。

**此 RTC 脚本不读取 `NPROC`、`NUM_EPOCHS`、`SAVE_STEPS`、`KEEP_LAST_K_CKPTS`、
`ZERO_STAGE`、`MIXED_PRECISION`、`DATASET_NUM_WORKERS` 这些环境变量。**
进程数请用 `NPROC_PER_NODE`；其余请用下面表格中的 Hydra 配置名。
例如 `bash scripts/train_marvin_rtc.sh training.num_epochs=10 training.save_steps=1000`。

### 训练配置默认值

下表左列均可放在脚本后覆盖。除特别注明外，RTC 脚本与 YAML 默认值相同。

| 配置名 | RTC 脚本默认值 | 含义 / YAML 差异 |
| --- | --- | --- |
| `training.rtc.enabled` | `true` | 开启训练时动作前缀条件；YAML 原始默认 `false` |
| `training.rtc.max_delay_steps` | `16` | 可采样的最大前缀长度，**含上界**；必须小于动作窗口 H |
| `training.rtc.delay_distribution` | `uniform` | 当前支持均匀采样整数 `0..max_delay_steps` |
| `training.num_epochs` | `5` | 数据集遍历轮数；`null` 表示仅由正整数 `max_steps` 控制结束 |
| `training.max_steps` | `null` | 不限制 micro-step；由 `MAX_STEPS` 设置；与 epoch 同时设置时先到者结束 |
| `training.batch_size` | `32` | 每卡 micro-batch；YAML 原始默认 `16` |
| `training.gradient_accumulation_steps` | `4` | 梯度累积次数；YAML 原始默认 `1` |
| `training.learning_rate` | `5e-5` | AdamW 基础学习率；YAML 原始默认 `1e-4` |
| `training.adam_betas` | `[0.9,0.95]` | AdamW 两个动量系数 |
| `training.weight_decay` | `0.01` | 权重衰减 |
| `training.max_grad_norm` | `1.0` | 梯度裁剪阈值 |
| `training.lr_scheduler` | `cosine` | 带线性 warmup 的余弦调度；`null` 为固定学习率 |
| `training.warmup_ratio` | `0.05` | 学习率 warmup 占计划更新步数的比例；与 checkpoint warm-start 不同 |
| `training.lr_min_ratio` | `0.01` | 最低学习率/基础学习率；默认基础 LR 下为 `5e-7` |
| `training.action_lr` | `null` | 动作模型专用 LR；`null` 使用基础 LR |
| `training.video_lr` | `null` | 视频模型专用 LR；`null` 使用基础 LR |
| `training.mixed_precision` | `bf16` | 计算精度，可选 `bf16`、`fp16`、`no` |
| `training.zero_stage` | `2` | DeepSpeed ZeRO 阶段，当前支持 `1`、`2` |
| `training.use_gradient_checkpointing` | `true` | 用重算激活降低显存占用，会增加计算量 |
| `training.use_gradient_checkpointing_offload` | `false` | 是否将 checkpointed 激活卸载到 CPU |
| `training.initialize_model_on_cpu` | `false` | 是否先在 CPU 构造/加载模型，再由 DeepSpeed 放到 GPU |
| `training.offload_optimizer_device` | `none` | 优化器状态卸载设备；可改为 `cpu` |
| `training.lambda_video` | `1.0` | 视频损失系数 |
| `training.lambda_action` | `1.0` | 动作损失系数；RTC 只计算有效后缀动作损失 |
| `training.min_timestep_boundary` | `0.0` | flow-matching 噪声时间采样下界比例 |
| `training.max_timestep_boundary` | `1.0` | flow-matching 噪声时间采样上界比例；不是 RTC 延迟 |
| `training.dataset_num_workers` | `8` | **每个训练进程**的数据加载 worker 数；4 进程通常共 32 个 |
| `training.output_path` | `outputs/openwam_checkpoints` | checkpoint 根目录；脚本切到项目目录后启动，路径相对项目目录 |
| `training.save_steps` | `2000` | 每多少 micro-step 保存；`0` 或 `null` 会同时关闭定期及结束时权重保存 |
| `training.keep_last_k_ckpts` | `1` | 滚动保留最近 K 份权重；想比较中间结果可改为 `5` |
| `training.save_full_states_for_resume` | `true` | 运行中保存优化器、调度器、RNG 等完整状态；YAML 原始默认 `false`；结束清理行为见下文 |
| `training.finetune_ckpt_path` | `RTC_CKPT_DIR` 的值 | warm-start 权重目录；YAML 原始默认 `null` |
| `training.resume_ckpt_path` | `null` | 完整恢复目录；与 `finetune_ckpt_path` 互斥 |
| `training.debug` | `false` | `true` 强制使用 `max_steps=20`、保存间隔 `10`、固定 LR，用于冒烟测试 |
| `project.seed` | `42` | 随机种子 |
| `project.wandb.project` | `openwam` | W&B 项目名 |
| `project.wandb.run_name` | `null` | 不指定实验名称，可自行设置便于区分两机实验 |
| `project.wandb.entity` | `null` | 不指定 W&B entity，使用运行环境的配置 |
| `project.name` | `openwam` | 项目名称 |
| `project.output_dir` | `outputs/${now:%Y-%m-%d_%H-%M-%S}` | Hydra 运行/日志目录配置；模型保存位置看 `training.output_path` |

### 到底训练多少步，batch 怎么算

当前代码每处理一次 dataloader micro-batch 就执行 `global_step += 1`。
`MAX_STEPS`、`training.save_steps` 和权重文件名里的 step 均按此计数，**不是权重更新次数**。
多卡同步处理各卡的 batch，`global_step` 不再乘 GPU 数。

```text
有效全局 batch = BATCH_SIZE × NPROC_PER_NODE × NNODES × GRADIENT_ACCUMULATION_STEPS
权重更新次数 ≈ global_step / GRADIENT_ACCUMULATION_STEPS
默认总 micro-step = 5 × 分布式切分后每个进程的 dataloader 长度
```

例如默认 `32 × 4 × 1 × 4 = 512`；`MAX_STEPS=8000`、累积 4 次时约为 **2000 次权重更新**。
若目标是约 8000 次权重更新、累积仍为 4，应设 `MAX_STEPS=32000`。
epoch 末尾不足一个累积窗口时的处理会影响精确更新次数，以日志 `opt_step` 为准。
建议将步数上限和保存间隔设为梯度累积次数的整数倍。

默认 5 epoch 不能直接换算成固定步数：它取决于数据集实际训练样本/窗口数、batch、GPU 数和
分布式补齐规则，并不等于 5 次遍历机器人 episode 文件就更新 5 次。
如果只想按步数结束，务必同时设 `training.num_epochs=null`，否则可能先达到 5 epoch。

### 可直接修改的训练示例

以下命令均假设已设置前面的路径环境变量，并位于相应机器的 OpenWAM 目录。

```bash
# 固定预算示例：8000 micro-step，约 2000 次更新；不受默认 5 epoch 限制。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=4 LEARNING_RATE=5e-5 MAX_STEPS=8000 \
  bash scripts/train_marvin_rtc.sh \
    training.num_epochs=null training.save_steps=1000 training.keep_last_k_ckpts=5

# 显存不足：每卡 batch 降为 16，累积增到 8，有效 batch 仍为 512。
# 同时将 micro-step 预算翻倍，保持约 2000 次更新。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=8 MAX_STEPS=16000 \
  bash scripts/train_marvin_rtc.sh training.num_epochs=null

# 按 epoch 训练：明确清除可能继承的 MAX_STEPS，跑 10 epoch。
MAX_STEPS=null bash scripts/train_marvin_rtc.sh training.num_epochs=10

# 单卡小规模冒烟：最多 20 micro-step，仅验证训练管线。
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 \
  bash scripts/train_marvin_rtc.sh training.debug=true training.num_epochs=null
```

`NPROC_PER_NODE` 根据实际分配到的 GPU 数设置，单卡用 `1`，四卡用 `4`，八卡用 `8`。
增减卡数会改变有效 batch；要保持它不变，需要相应调整每卡 batch 或累积次数。
训练预算、学习率和 batch 最终应通过验证集损失、动作连续性和任务成功率决定。

### 初始化 checkpoint、warm-start 和 resume

**warm-start** 表示只加载已有模型权重，然后从 `global_step=0` 开始新实验，
重新创建优化器和学习率调度器。默认选择 `RTC_CKPT_DIR` 中步数最大的
`checkpoint_step_*.safetensors`，新输出目录为
`outputs/openwam_checkpoints/YYYY-MM-DD_HH-MM-SS`（可通过 `training.output_path` 改根目录）。

如果已有质量较好的非 RTC Marvin 微调 checkpoint，优先用它继续做 RTC 微调；
也可以从适配当前模型架构的 foundation checkpoint 开始，但需要完成相应任务适配。
入口要求自包含目录，不能只给一个孤立权重文件。第一次从非 RTC 切到 RTC 应使用 warm-start。

**resume** 恢复权重、优化器、调度器、RNG 和已有 step，复用原输出目录；
RTC 配置必须与原训练保存时一致，不能通过 resume 把普通训练改成 RTC，或修改训练最大延迟。
此时 `max_steps` 是恢复后的总 step 上限，不是额外再跑多少步。

```bash
# RTC_CKPT_DIR 和 DATASET_DIR 仍需设置：脚本启动前会检查它们。
# finetune=null 取消脚本默认的 warm-start，选择真正的完整状态恢复。
MAX_STEPS=16000 bash scripts/train_marvin_rtc.sh \
  training.finetune_ckpt_path=null training.resume_ckpt_path=/path/to/rtc-output \
  training.num_epochs=null
```

**当前保存实现的限制：** `save_full_states_for_resume=true` 会在运行中的保存点写出
`accel_state_step_*`，供中断后恢复；正常达到步数/epoch 上限后，会删除这些完整状态目录，
只保留最近 K 份权重。因此正常结束的输出通常只能 warm-start，不能再完整 resume。
`training.save_steps=0` 或 `null` 连结束时的权重也不保存，不适合需要产出模型的训练。

### RTC 最大延迟与训练行为

每样本独立采样前缀长度。前缀使用归一化的干净动作和 timestep=0，
后缀沿用原 flow matching 加噪；动作损失只对有效后缀元素计算。
短样本保留至少一个有效后缀步，无动作监督样本不制造监督。
视频损失与原 timestep 权重继续沿用。不增加模型参数。

`training.rtc.max_delay_steps` 默认 `16`，可以调高或调低，例如：

```bash
bash scripts/train_marvin_rtc.sh training.rtc.max_delay_steps=12
```

它以动作步计数，30 Hz 下 16 步约为 0.533 秒；它表示训练覆盖的前缀长度上限，
不是服务端推理必定能在 0.533 秒内完成的保证。调高会让模型接触更长的固定前缀，
同时缩短这些样本的可监督后缀；调低会缩小部署允许的延迟范围。
更改后需要按新配置训练，不能只改部署参数扩大旧 checkpoint 的能力。

当前 Marvin 配置的动作窗口 **H=32**，来自数据窗口 `num_frames=33`，不是由部署 horizon 决定。
训练要求 `0 <= max_delay_steps < H`。部署还必须满足 `d + s <= H`，其中 s 是每轮采用的新动作步数。
所以 H=32、s=16 时，即使训练最大延迟调到 24，部署实际 d 仍不能超过 16；
需要同时缩短 s 才能容纳更长前缀。

## 服务端

必须使用 **已经完成 RTC 微调** 的 checkpoint。能力信息来自加载 checkpoint 时的
训练配置，不能靠 deploy.yaml 把普通模型标记成 RTC。旧 checkpoint 启用新模式会明确报错。

### 服务端参数与默认值

下表为 `configs/deploy.yaml` 的实际默认值，命令行具名参数可覆盖它们。
优化参数也可在命令末尾使用 `optimization.xxx=value` 覆盖。

| 参数 | 默认值 | RTC 中的含义 |
| --- | --- | --- |
| `--config` | 未指定时读取 `configs/deploy.yaml` | 部署配置文件；下列默认值以此文件为准 |
| `--ckpt-dir` | `checkpoint_path=null`，无默认目录 | 必须提供 RTC checkpoint 目录，或在部署 YAML 配置路径 |
| `--ckpt-name` | `null` | 未指定时选择目录中步数最大的权重 |
| `--device` | `cuda:0` | 推理设备 |
| `--host` | `0.0.0.0` | 服务端监听地址；示例显式绑定机器人可访问的 `192.168.50.1` |
| `--port` | `8848` | WebSocket 端口 |
| `--inference-mode` | `sync` | **必须显式改为 `training_rtc`** 才启用此模式 |
| `--inference-horizon` | `null` | RTC 每轮约定采用的新动作数 s；建议显式设为 `16`，与客户端一致 |
| `--denoise-steps` | `10` | 每次生成的去噪迭代次数；不是动作步数，也不是 RTC 延迟 |
| `--denoise-mode` | `sync` | 视频/动作去噪轨迹，可选 `sync`、`async`；与 RTC 执行模式独立 |
| `--lead-modality` | `video` | 仅 `denoise-mode=async` 时选择领先模态，可选 `video`、`action` |
| `--variance-shift-alpha` | `1.0` | 仅异步去噪时的领先轨迹偏移参数 |
| `--linear-offset` | `0.0` | 仅异步去噪时的滞后轨迹起点偏移 |
| `--inference-delay-steps` | `null` | 原有 `inference-mode=async` 的预期延迟；RTC 请求时机由客户端 lead 控制 |
| `--compile-enabled` | `true` | 启用编译加速；下面首次验证示例显式改成 `false` |
| `optimization.compile.self_attn.torch_mode` | `default` | 当前 joint self-attention 架构编译模式 |
| `optimization.compile.self_attn.dynamic` | `false` | 默认使用固定形状编译 |
| `optimization.dit_cache.enabled` | `true` | DiT 速度缓存；下面示例显式改为 `false` |
| `optimization.dit_cache.cosine_threshold` | `0.99` | 缓存相似度阈值 |
| `optimization.dit_cache.max_skips` | `3` | 最多连续跳过的视频 DiT 计算次数 |
| `optimization.decode_video` | `false` | 默认不做视频 VAE 解码，只返回动作，video 为 null |
| `optimization.prompt_embed_cache.enabled` | `true` | 缓存文本嵌入 |
| `optimization.prompt_embed_cache.maxsize` | `32` | 文本嵌入缓存容量 |

**RTC 中 `inference-horizon=16` 不表示只生成 16 步。** 服务端仍生成并返回完整 H=32 步，
其中前 d 步固定为客户端已承诺动作；客户端只采用 `[d:d+16]`。
服务端的 horizon 用于握手约定 s，并检查 `d+s<=H`。
若保持 `null`，服务端不指定这项 s 约定，客户端仍按自己的 s 校验和执行；
推荐两端显式设置相同值，避免配置歧义。

```bash
python scripts/deploy.py --ckpt-dir /path/to/rtc-output \
  --host 192.168.50.1 --port 8848 \
  --inference-mode training_rtc --inference-horizon 16 \
  --denoise-mode sync --compile-enabled false optimization.dit_cache.enabled=false
```

首次验证建议关闭 compile/cache；完成动作质量基线后分别打开进行对照。
`denoise_mode` 是视频与动作去噪轨迹，与 RTC 的异步播放独立。
前缀走 action 统计和 action scatter map，不能走 proprio 统计。
前缀在每次前向和每次去噪更新后恢复，返回时物理空间前缀也精确保留。

## 机器人客户端

在机器人 `192.168.50.2` 的 `/home/nvidia/sharpa-marvin-client` 中运行。
客户端连接的是推理机 `192.168.50.1`，不要把客户端 `--host` 填成机器人自身 IP。

### 客户端参数与默认值

没有传 CLI 值时，部分参数从 `--config` 指定的 YAML 读取；下表区分了这些来源。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--config` | 项目下 `config/marvin_qp.yaml` | 机器人、网络、发布频率、滤波等配置文件 |
| `--execution-mode` | `single` | **必须显式设为 `training_rtc`** |
| `--execute-horizon` | CLI 未指定；**RTC 有效默认 `16`** | 每轮采用并追加的新动作数 s；普通模式的 YAML `execution_horizon=32` 不是 RTC 默认值 |
| `--rtc-lead-steps` | `10` | 已承诺队列剩余多少动作时发起下一轮请求 |
| `--control-hz` | CLI 未指定；YAML `30.0` | 新动作消费频率，单位 Hz；RTC 的步数以此为时间尺度 |
| `rates.publisher_hz`（YAML） | `200.0` | 目标重发频率；不用于 RTC 动作步数计数 |
| `--host` | YAML `openwam_policy.host` | 机器人部署配置为 `192.168.50.1`；本地源码 YAML 为 `172.16.10.32`，建议显式传 IP |
| `--port` | CLI 未指定；YAML `8848` | 推理服务端口 |
| `--connect-timeout-sec` | CLI 未指定；YAML `10.0` | 建立网络连接超时秒数 |
| `--request-timeout-sec` | CLI 未指定；YAML `300.0` | 网络请求超时；不代表 RTC 可等 300 秒，队列 deadline 会更早终止运行 |
| `--warmup-timeout-sec` | `15.0` | 启动阶段等待机器人观测就绪的时间上限，单位秒 |
| `--fresh-observation-timeout-sec` | `1.0` | 等待新鲜观测的时间上限，单位秒 |
| `--skip-policy-warmup` | `false`（不传） | 默认执行真实观测的模型 warm-up；传此开关跳过请求，仍重置策略状态 |
| `--max-loops` | `0` | `0` 不限轮数；RTC 按采用的动作 chunk 轮数计数，不是单个动作数 |
| `--video-crop` | `None` | 默认不额外指定裁剪；传 `X Y WIDTH HEIGHT`，下面数值是当前任务示例而非默认 |
| `--task` | `None`，读取 YAML `instruction` | 默认配置指令为 `pick all items from the table and place them in the bin` |
| `--log-dir` | `/home/nvidia/marvin_gr00t_runtime/logs` | 客户端日志目录 |
| `--execute` | `false`（不传） | 默认 dry-run；传入后才允许发布运动目标，且仍需满足配置和操作门控 |
| `command_filter.enabled`（YAML） | `true` | 已承诺目标使用滤波后的结果 |
| `command_filter.eef_alpha`（YAML） | `0.3` | 末端目标滤波系数 |
| `command_filter.hand_alpha`（YAML） | `0.3` | 手部目标滤波系数 |

模型 **warm-up** 是推理前做一次请求以预热运行路径/编译，与训练的 **warm-start**
（加载已有权重开始新实验）和学习率 **warmup**（逐渐增加 LR）是三件不同的事。

```bash
./run_openwam_client_with_env.sh --host 192.168.50.1 --port 8848 \
  --execution-mode training_rtc \
  --execute-horizon 16 --rtc-lead-steps 10 \
  --video-crop 184 120 600 480 --max-loops 20
```

以上是 dry-run。真机播放使用既有 `--execute` 开关和操作门控。
RTC 默认执行 16 步；`--rtc-lead-steps` 是队列剩余多少动作时开始请求，默认 10。
相机年龄也会计入前缀，因此 lead 必须小于 checkpoint 支持的最大 d。
实际 `d` 由图像时间戳和发布器已消费的 30Hz 动作索引计算，不使用 200Hz 重发次数，
也不使用服务端时钟。相机时间戳必须与机器人系统 wall clock 一致。

当前 H=32，s=16，需要 `d <= min(训练最大延迟, H-s)`。
启动参数还要求 `0 < rtc_lead_steps < s`，且
`rtc_lead_steps < min(训练最大延迟, H-s)`，为图像年龄留出余量。
默认 30 Hz 下，s=16 对应每轮约 0.533 秒的新动作，lead=10 对应队列约剩 0.333 秒时请求。
实际 d 还包括图像年龄，不能简单认为 `d=rtc_lead_steps`；
较小的 lead 会缩短推理可用时间，较小的 s 会提高请求频率。
需要让取观测、推理、通信和预处理在已承诺动作耗尽前完成。

每次生成完整 H 步，仅采用 `[d:d+s]`。客户端只预处理这 s 步并追加到已承诺队列；
未采用尾部不经过滤波器。下一次前缀来自实际承诺的滤波/限幅后目标，转换回模型坐标和列式 ROT6D。
这避免了预处理整块后丢弃尾部导致的滤波状态污染，也无需回滚 native worker。

单个后台任务依次处理观测、网络和后缀预处理；独立进程继续发布已有目标。
结果提前到达时追加到固定前缀后，不提前改写前缀。发布进程验证每段起始索引，
连续播放中若队列耗尽，立即停止客户端目标发布并报 deadline miss。
迟到、越界 d、断线、非法回复和机器人门控故障均终止本次 RTC 运行；首版不自动重启运动。
停止会先停止发布器，再清理后台请求，旧结果不会重新入队。
正常退出 `--max-loops` 会播放完最后已承诺段。

日志 `rtc_cycle` 包含 episode/request id、观测步、切换步、前缀长度、推理时间、
预处理时间和剩余队列；`command` 记录实际采用目标，`rtc_abort` 记录停止原因。

### 排查 `RTC deadline missed before enqueue`

这个错误表示新动作入队前，旧动作的待消费队列已经耗尽。它与握手失败不同，
也不能通过增大网络 `request-timeout-sec` 解决。RTC 的可用时间取决于剩余动作，
取观测、图片编码、网络、服务端推理、动作预处理及线程调度都要计入预算。
预热成功仅表示模型请求成功，不代表持续运行一定满足这个预算。

2026-10-08 在机器人 `192.168.50.2` 和推理机 `192.168.50.1`，使用
`10-08/checkpoint_step_3975.safetensors`、H=32、s=16、30 Hz、10 次去噪及 compile/cache 开启时：

| `--rtc-lead-steps` | 约可用时间 | 当次 dry-run 结果 |
| --- | --- | --- |
| `10`（代码默认值） | 333 ms | 用户日志中完成 6 轮后，在入队前超时 |
| `12` | 400 ms | 图像年龄叠加后出现 d=17，超过训练最大前缀 16，被拒绝 |
| `11` | 367 ms | 完成 30 轮，无 abort；后续轮次 d=13–16 |

`lead=11` 的这次测试中，客户端推理调用耗时约 224–282 ms（含编码、通信和解析），
服务端推理约 170–183 ms，动作预处理约 2–35 ms。此结果只覆盖该次负载和输入，
不能保证长期运行无 deadline miss，也不代表已验证真实运动质量。

当前机器可先用下面的命令重复有限轮次的 dry-run；`11` 是本次实测设置，
**代码默认值仍为 10**：

```bash
./run_openwam_client_with_env.sh --host 192.168.50.1 --port 8848 \
  --execution-mode training_rtc --execute-horizon 16 --rtc-lead-steps 11 \
  --video-crop 184 120 600 480 --max-loops 30
```

如果仍超时，应结合 `rtc_cycle.jsonl` 的 `inference_latency_sec`、`server_latency_ms`、
`prepare_metrics.wall_ms` 和 `prefix_length` 排查处理延迟与图像年龄。
仅已成功入队的轮次会记录 `rtc_cycle`，因此不能把最后成功一轮的耗时当成失败一轮的耗时。
不能盲目继续增大 lead：当前 checkpoint 的 d 上限为 16，增大 s 或网络超时也不会扩大它。
保留迟到拒绝和队列断流停止行为，不应通过采用过时动作来消除报错。

## 协议 v1

`pong`：`inference_mode=training_rtc`、`supported_requests=[obs_rtc,reset,ping]`、
`rtc_protocol_version=1`、`action_horizon`、`rtc_max_delay_steps`、表示契约。

`obs_rtc`：标准 images/state/prompt，另加 episode_id、request_id、observation_step、
commit_step、prefix_length、action_prefix（[d,62]）；必须 `commit_step-observation_step=d`。
返回 action_chunk（[H,62]），回显全部时间标识。客户端校验标识、形状和前缀一致性。
`d=0` 用于 warm-up 和启动；后续请求至少固定一个动作。

## 验证

`tests/test_training_rtc.py`：mask 与梯度、d=0 等价、MoT timestep/compile 路径、
前缀去噪不变性、action/state 统计区分、协议拒绝、真实小架构训练步和 checkpoint 往返。
机器人 `tests/test_rtc.py`：图像时间对齐、滤波状态接续、连续入队、超时、错序、停止。
代码与测试同步不代表完成真实数据微调或真机任务成功率评估。

本文参数来源：`scripts/train_marvin_rtc.sh`、`configs/train.yaml`、
`openwam/train/openwam_trainer.py`、`openwam/train/utils/checkpointing.py`、
`configs/deploy.yaml`、`openwam/deploy/server.py`，以及机器人客户端的
`sharpa_marvin_client/config.py`、`sharpa_marvin_client/rtc.py` 和 `config/marvin_qp.yaml`。
