# TouchWAM

新增 **单通道 deform、逐区域预测、可选掌面** 的实现为 `touchwam_regions`，
训练入口是 `configs/train_touchwam_regions.yaml`。
数据合同、真机／仿真 reader、旧权重迁移和推理协议见 [regions.md](regions.md)。
下文记录保留的 `touchwam_v2` 距离／接触版；它与 deform 是不同表示，不能混用缓存。

在现有 OpenWAM 上增加触觉生成分支。模型保留原视觉／动作主干及 Alpha 80 维动作表示，
新增 6 个触觉交互块，预测未来距离图和接触概率。默认联合微调视觉、动作、状态与触觉；
VAE、文本编码器等预处理模块继续冻结。

## 代码组织与兼容性

| 文件 | 职责 |
| --- | --- |
| `openwam/model/architectures/touchwam/architecture.py` | 组装、前向、损失、训练策略和权重兼容检查 |
| `openwam/model/architectures/touchwam/modules.py` | 固定 patch codec、因果历史编码、触觉交互块 |
| `openwam/model/architectures/touchwam/inference.py` | 三分支联合 Euler 去噪与 CFG |
| `openwam/tactile/sharpa.py` | 离线／在线共用降采样、传感器顺序和历史缓冲 |
| `openwam/dataloader/bench2dex_tactile.py` | 已转换 Bench2Dex 数据读取、未来标签和等任务采样 |
| `configs/model/touchwam.yaml` | 模型结构及触觉配置 |
| `configs/train_touchwam.yaml` | 唯一训练入口配置，续训／消融用参数覆盖 |

为兼容已有权重，注册名和 checkpoint 元数据继续使用 `touchwam_v2`，新增权重仍在
`touch_v2.*` 下，输入 schema 仍为 `sharpa_tactile_v1`。这些名称表示保存格式，
不依赖 V1 模型实现。官方 Alpha 初始化只允许缺失新增触觉权重；V2 自身权重严格校验
参数以及 model version、codec 和 tactile spec。改变模型结构时不能静默忽略不匹配项。

本次迁移未引入 V1 adapter、Compact Action、旧关机／切换／监控启动器、评测视频录制和
网页展示代码。原本的多份训练 YAML 合并为一个配置；不带机器专属路径、权重和数据。
Bench2Dex 只保留现成 manifest／numeric sidecar／HDF5 的读取，原始数据转换及机器人
运动学工具仍由数据准备工程负责。共享归一化的等任务采样独立于当前异构数据 MixtureDataset，
保持 checkpoint 根目录的 `normalization_stats.npy` 格式。

原 OpenWAM 的模型默认值、混合视频训练、随机 ActionDiT 初始化及启动脚本保持原有行为。
当前 TouchWAM 仅接受带动作监督、显式触觉有效性标记的样本；未支持纯视频混合训练。

## 数据约定

| 字段 | 单样本形状／含义 |
| --- | --- |
| `tactile_maps` | `[4,10,2,32,32]`，从 t-3 到 t 的历史 |
| `tactile_valid` | `[4,10]` bool，传感器存在且数值有效 |
| `future_tactile_maps` | `[8,10,2,32,32]`，仅用于训练监督 |
| `future_tactile_valid` | `[8,10]` bool |
| `future_tactile_offsets` | `[2,4,6,8,10,12,14,16]`，以原始数据行为单位 |

原始数据为 20 Hz，未来监督覆盖 0.1～0.8 秒。10 个指尖依次为右手、左手的
thumb/index/middle/ring/pinky。两通道为 `distance_along_normal_m` 和 `contact_mask`；
距离是射线几何量。没有接触仍是有效观测，缺失传感器必须使用 `valid=False`。

历史不读未来，标签不跨有效轨迹片段。32×32 图切为 4 个 16×16 patch，历史 160 个
和未来 320 个 token。触觉没有学习型 VAE。默认全尺寸新增参数为 30,522,892。

训练读取现有 `bench2dex_alpha_v1` manifest、numeric sidecars 和归一化文件。
`source_root` 可覆盖 manifest 中旧机器的 HDF5 路径。可选复用 `sharpa_tactile_v1`
逐帧缓存；缓存契约与源文件指纹不匹配会报错。代码迁移不包含数据迁移。

## 训练与继续微调

从已有 Alpha checkpoint 初始化，显式提供 checkpoint 和数据位置：

```bash
NPROC_PER_NODE=4 bash scripts/train.sh --config-name train_touchwam \
  training.finetune_ckpt_path=/path/to/alpha/checkpoint_step_N.safetensors \
  dataloader.manifest_path=/path/to/bench2dex/manifest.json \
  dataloader.source_root=/path/to/hdf5
```

checkpoint 同目录需要 `config.yaml`、tokenizer 及原加载器要求的组件产物。
新增触觉权重使用统一 `project.seed` 初始化，避免 rank 本地训练 seed 导致初始权重不同。
原 ActionDiT 权重保留；不能同时请求 `training.action_backbone_init=random`。

从完整 V2 checkpoint 继续微调，使用同一命令并覆盖：

```text
training.finetune_ckpt_path=/path/to/touchwam/checkpoint_step_N.safetensors
training.initialize_touch_from_openwam=false
```

这是权重热启动，会创建新优化器。实时 `model.architecture.training_mode` 优先于
旧 checkpoint 的训练模式；默认 `joint_finetune`。显式消融可选 `new_only` 或
`action_finetune`。精确 resume 需要已有 accelerate 完整状态，同时设置
`training.resume_ckpt_path=/path/to/run`、`training.finetune_ckpt_path=null` 和
`training.initialize_touch_from_openwam=false`，沿用原训练器的恢复流程。

训练日志新增 `loss_tactile`、`loss_tactile_contact`、`loss_tactile_distance`。
各模态独立采样训练噪声，触觉接触像素和距离背景采用独立加权速度 MSE。

## 推理

沿用现有部署入口，明确关闭不支持三分支的优化：

```bash
python scripts/deploy.py --ckpt-dir /path/to/touchwam/run \
  --compile-enabled false optimization.dit_cache.enabled=false
```

服务握手增加 `tactile` 输入契约。客户端将 `tactile_maps`、`tactile_valid` 放入 observation；
`JointInferenceEngine` 传给模型。需要从原始传感器构建历史时使用
`openwam.tactile.sharpa.SharpaTactileHistory`，每个 20 Hz 观察帧都调用 `append`，
episode 重置时调用 `reset`。不能将四次重规划请求当作连续四帧触觉。

`architecture.generate`／`engine.generate` 除原视频和动作外，返回未来触觉地图、valid、
offsets 和 fps。WebSocket policy 接口仍返回执行动作；不自动传输大型未来地图。
触觉缺失时模型支持显式消融旁路，未来触觉结果标记无效。正式触觉评测应由客户端验证
传感器持续有效。当前推理支持 Wan 布局、单观察和 CFG；预测未来触觉不包含 MPC 搜索。

## 验证和迁移记录

CPU 核心与集成检查：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_touchwam.py tests/test_touchwam_integration.py \
  tests/test_bench2dex_tactile.py -q
```

覆盖 codec、因果性、未来标签隔离、零门控等价、缺失传感器、三种训练策略的实际参数
更新、梯度检查点、FP32/BF16 小尺寸真实 Wan+ActionDiT、严格权重加载、跨 rank 初始化、
联合推理、部署传参、采样／缓存时间对齐和归一化保存恢复。CUDA 专项在 CPU 主机上跳过。
回归结果和旧源码数值对照见 [validation.md](validation.md)。全尺寸 GPU 训练及闭环成功率
仍需单独验证。

源代码来自 `ys:/data_all/intern10/workspace/TouchWAM/OpenWAM` 的实际工作区，包含未提交
V2 文件；源 HEAD 为 `4a9af3e5bf6a992b0a41ccc19d33acd224f55ed1`，文件 SHA256 见
[source_manifest.json](source_manifest.json)。目标以 `96f4b6e` 为基础，使用同一仓库的
`codex/touchwam` 分支及独立 `OpenWAM-touchwam` worktree；原工作目录保持原分支。
