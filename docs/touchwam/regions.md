# 按区域预测 deform，选择性使用掌面

`touchwam_regions` 在原 OpenWAM 视觉／动作主干和 TouchWAM 三分支交互上，
增加单通道 deform 输入与预测。十个指尖共享编码器、交互块和输出投影，
通过区域身份区分左右手和手指。掌面使用同一分支，以可选区域参与。
旧的 `touchwam_v2` 距离／接触双通道实现和权重加载路径保留。

## 实现范围

- 支持每条样本不同数量、不同顺序的已注册区域；batch 内 padding。
- 默认使用固定 64 类公共区域；保留原十指／双掌 ID。实际输入、输出只包含选中的表面。
- 完整公共定义、TouchScale／夹爪映射和 12→64 迁移见 [canonical_regions.md](canonical_regions.md)。
- 布局、历史观测、未来预测请求、未来标签、空间 ROI 分别设 mask。
- `palm_mode=auto` 按布局启用掌面，`off` 屏蔽全部掌面历史及未来预测。
- 带掌面样本和不带掌面样本可以混合训练；不会给缺失掌面补黑图监督。
- 当前每个模型／batch 使用同一种单通道 deform codec 和统一图尺寸。
  尚未实现压力图、RGB 视触觉等异构 codec 混合，也未支持缺少动作监督的 ego 样本。
- 默认注册掌面身份只提供结构容量。模型仍需实际掌面样本训练，才能评估掌面预测能力；
  仅用十指数据训练，不能声称获得了可靠的掌面预测。

deform 是传感器输出的形变灰度强度，不是校准压力、法向位移米值或六维力。
仿真与真机统一的是表示和数据接口，物理一致性仍需采集标定和实验验证。
`sensor_names` 区分真机与仿真来源，不能替代标定。

## 数据和掩码

默认 `H=4, F=8, S=32`。单样本由 reader 返回，下表不含 batch 维度。

| 字段 | 形状／含义 |
| --- | --- |
| `tactile_maps` | `[H,N,1,S,S]`，从 t-H+1 到 t 的历史，uint8 或 float `[0,1]` |
| `tactile_valid` | `[H,N]` bool，该时间的该区域是否实际观测到 |
| `future_tactile_maps` | `[F,N,1,S,S]`，训练标签，不进入干净历史条件 |
| `future_tactile_valid` | `[F,N]` bool，未来标签是否真实有效 |
| `tactile_region_names` | N 个名称，存在于固定注册表；可对应同一区域的不同表面 |
| `tactile_surface_ids` | 可选 `[N]` int64，默认 0；与区域 ID 组成唯一表面身份 |
| `tactile_layout_present` | `[N]` bool，采集设备实际包含哪些区域 |
| `tactile_query_mask` | 可选 `[F,N]` bool；默认预测布局中所有启用区域 |
| `tactile_spatial_valid` | 可选 `[N,1,S,S]` bool，传感器图中的有效像素；默认全图 |
| `tactile_sensor_names` | 可选 N 个来源名称，映射到 checkpoint 的 sensor 注册表 |
| `tactile_codec` | 必须显式为 `sharpa_deform_gray_v1` |
| `tactile_fps` | 必须与模型一致 |
| `future_tactile_offsets` | 未来帧相对 t 的原始行偏移，必须与配置一致 |

`collate_regions` 将名称转换为 `tactile_region_ids [B,N]`；padding ID 是 `-1`，
padding 的全部 mask 为 false。在线模型接口使用 ID，客户端应从服务握手读取
注册表。来源也可以使用 `tactile_sensor_ids`，但不能同时传 names 和 IDs。

默认区域表的前 12 项保持原顺序：右手五指 `.tip`、左手五指 `.tip`、左右掌。
其余 ID 固定覆盖指节、掌面细分、夹爪和通用接触面，详见公共定义。
后续传感器通过 layout 映射和 surface_id 接入，不向公共表追加名字。

| 情况 | layout | observed | query／loss |
| --- | --- | --- | --- |
| 没装掌面传感器 | false 或省略区域 | false | 不预测，不计算 loss |
| 装了传感器但该帧丢包 | true | false | query 独立决定；训练仍需真实未来标签 |
| 有效黑图，无明显形变 | true | true | 正常输入和监督，不根据亮度删除 |
| 显式关闭掌面 | 原始布局不变，模型内禁用 | 屏蔽 | 不预测，不计算 loss |
| 有效图的 ROI 外像素 | true | 由观测决定 | 像素屏蔽，不参与 loss |

归一化固定为原始 uint8 除以 255，再在线性 codec 中映射到 `[-1,1]`。
禁止逐帧 min-max。原始 240×240 等灰度图使用 area resize 到 S×S；S 可配置，
增大 S 的训练显存和细小接触保留效果需要另测。当前没有学习型触觉 VAE。

流匹配损失先按区域内有效像素／时间求平均，再对区域和有标签样本平均；
避免掌面像素多或传感器多的样本自然获得更大权重。DDP 按全局有标签样本数归一化。
可用 `foreground_weight` 按灰度连续加权，这不等价于接触概率标签。

## 掌面与缺失观测训练

```yaml
model:
  architecture:
    tactile:
      palm_mode: auto
      palm_dropout: 0.0
      history_dropout: 0.0
      allow_missing_history: false
```

`palm_dropout` 在训练时按样本随机移除全部掌面，包括历史、预测请求和损失，
用于让带掌面数据训练的模型适应只有指尖的设备。仅有十指数据时无需启用。

`history_dropout` 独立随机移除整个区域的历史，但保留该区域未来请求与真实标签。
这用于训练“传感器暂时不可见，仍根据视觉／动作等条件预测”的能力。
只有设置了非零 history dropout，才能配置 `allow_missing_history=true`；
配置许可本身不证明模型已经学会，需要训练和评估。

默认推理遇到请求区域完全没有有效历史会报错。客户端可以选择排除该区域的
query，或使用经过缺失历史训练的 checkpoint。不能把丢包改成有效黑图。
训练时所有启用 query 的未来标签必须完整；不完整窗口由 reader 提前过滤，
也可由具备明确采样策略的 reader 显式选择 query。模型不伪造缺失未来标签。

## 真机 LeRobot 数据

`MarvinDeformDataset` 读取转换后的 Marvin LeRobot v3 数据，保留原 76 维存储到
62 维 EEF／手部、再映射到统一 80 维空间的约定，要求 column rot6d 合同。
它通过独立 deform sidecar 读取触觉，不直接解析 bag。
本次没有提取实际 bag、生成正式触觉缓存或启动训练。

每个 episode 的无损 `.npy`：

- `maps`：uint8 `[T,N,Hraw,Wraw]`。
- `valid`：bool `[T,N]`。
- `timestamps`：`[T]`，与对应 LeRobot episode 的 timestamp 同基准、同帧网格。

manifest 示例（简写为两个区域；真实十指需列出完整区域及相应数组）：

```json
{
  "schema": "tactile_regions_v1",
  "codec": "sharpa_deform_gray_v1",
  "fps": 30,
  "dataset_info_sha256": "<sha256 of dataset/meta/info.json>",
  "region_names": ["right.index.tip", "left.palm"],
  "sensor_ids": [0, 0],
  "episodes": {
    "16": {
      "maps": "episode_000016.maps.npy",
      "valid": "episode_000016.valid.npy",
      "timestamps": "episode_000016.timestamps.npy",
      "layout_present": [true, false]
    }
  },
  "provenance": {
    "alignment": "latest source timestamp <= dataset timestamp",
    "max_age_s": 0.04,
    "note": "Exporter must record actual bag/topic/clock mapping and file fingerprints."
  }
}
```

示例中的 max age 仅演示字段，需依据真实采样率和时钟审计确定。导出器需要先把
相机、动作和 bag 的时间映射到相同基准，再用 `causal_match` 选择不晚于目标时刻的
最近消息；过期或缺失消息置无效。不能直接拿 bag 起始时刻替代 episode 起始时刻，
也不能用未来消息补历史。

reader 检查 schema、codec、fps、`info.json` SHA、数组类型、路径、帧间隔和窗口起点
时间；训练记录保存 manifest SHA 与筛选窗口 SHA。**这些校验不验证原始 bag 到
dataset 的时钟映射，也没有自动校验 `.npy` 内容 SHA。**缓存必须保持不可变，
正式导出应在 provenance 留存文件指纹并独立验证；仅 `info.json` 相同不足以证明
两批数据完全相同。

reader 只采样有完整未来触觉的窗口，历史不跨 episode 边界；视频失败重试后，
按实际选中的 episode／时间读取触觉，避免动作与触觉错配。
此前数据审计的 `pick_place_2_3` 中，零基 episode 16–66、72–86 共 66 条含触觉，
实际哪些窗口可用仍以提取后的 valid 为准。当前 reader 是完整 episode 模式，
尚未接入其他开发分支的 subtask／latent-cache 数据模式。

## 训练入口与权重迁移

服务器代码目录：`/mnt/pfs/v89atp/workspace/OpenWAM-touchwam`。
使用服务器训练环境，在实际 GPU 节点运行既有 `scripts/train.sh`。
下面路径中的 checkpoint 和触觉 manifest 必须替换为已经准备好的产物。

从 OpenWAM Alpha 初始化：

```bash
NPROC_PER_NODE=4 bash scripts/train.sh --config-name train_touchwam_regions \
  training.finetune_ckpt_path=/path/to/alpha/checkpoint_step_N.safetensors \
  dataloader.dataset_dir=/mnt/pfs/v89atp/workspace/dataset/posttrain/pick_place_2_3_openwam_v3 \
  dataloader.tactile_manifest_path=/path/to/aligned_deform/manifest.json
```

从已有距离／接触版 TouchWAM V2 初始化，在同一命令覆盖：

```text
training.finetune_ckpt_path=/path/to/legacy_touchwam/checkpoint_step_N.safetensors
training.initialize_touch_from_openwam=false
training.initialize_touch_from_legacy=true
```

迁移严格检查旧模型 schema 及视觉／动作权重键和形状，复用这些权重，并选择复用
兼容的触觉历史 encoder、交互块和时间 MLP。单通道投影、区域／来源 embedding、
位置／ROI 投影及输出头重新初始化，V/A 回写 gate 归零。迁移报告记录在训练输出
`initialization.json`；这是新优化器的热启动，不能恢复旧触觉优化器状态。
`project.seed` 控制新增权重，在各 rank 保持一致。

从 regional checkpoint 继续微调：使用同样模型结构、注册表和 codec，关闭两个
initialize 标志，再设 `finetune_ckpt_path`。从 accelerate 完整状态精确 resume：
关闭两个 initialize 标志、设 `finetune_ckpt_path=null` 和
`resume_ckpt_path=/path/to/run`，使用与保存时一致的配置和数据合同。
checkpoint 同目录仍须有原加载器需要的 `config.yaml` 和组件产物。

默认真机 30 Hz、未来 stride=3，预测 0.1–0.8 秒。原始 Bench2Dex 20 Hz 使用：

```bash
NPROC_PER_NODE=4 bash scripts/train.sh --config-name train_touchwam_regions \
  dataloader=bench2dex_deform \
  dataloader.manifest_path=/path/to/bench2dex/manifest.json \
  dataloader.source_root=/path/to/hdf5 \
  model.architecture.tactile.fps=20 \
  model.architecture.tactile.future_stride=2 \
  training.finetune_ckpt_path=/path/to/alpha/checkpoint_step_N.safetensors
```

此 reader 必须读取 HDF5 中实际保存的 `robot/tactile/tacmap/<site>` uint8 图，
不会用旧 `distance/contact` 缓存冒充 deform；没有 tacmap 会报错。
当前单个模型配置要求统一 fps 和 offsets，20 Hz 仿真与 30 Hz 真机若要在同一
batch 混合，需先因果重采样到共同时间网格，不能直接混入原始两种频率。

## 在线输入与逐区域显示

部署仍使用原入口，关闭三分支尚未支持的 compile 和 DiT cache：

```bash
python scripts/deploy.py --ckpt-dir /path/to/regional/run \
  --compile-enabled false optimization.dit_cache.enabled=false
```

`DeformHistory` 在每个固定 observation tick 更新，不能只在请求动作规划时更新。
构造时指定实际区域，并使用 checkpoint 的 `region_names` 注册表和 sensor ID。
输入各区域原始 uint8 图；丢包省略该区域或传 None。时间间隔异常会清空历史。
这是规则采样 tick 缓冲，不负责原始异步消息的时间对齐；对齐应由客户端先完成。

同步 policy 请求设置 `include_tactile: true`，返回额外 `tactile` 字段：

- `future_tactile_maps [F,N_requested,1,S,S]`，float `[0,1]`。
- `tactile_region_names`／`tactile_region_ids`／`tactile_surface_ids`，按区域和实例分别显示。
- `future_tactile_valid` 表示本次请求生成的区域和时间，不是实测接触状态。
- `tactile_history_available` 和 `tactile_spatial_valid`，用于说明条件缺失及 ROI。
- `future_tactile_offsets`、`future_tactile_times_s` 和 `tactile_fps`。
- `prediction_id`、`observation_timestamp_s`、`action_index`，标识对应的动作计划。

客户端需要传 `timestamp_s` 才能返回有效的来源时间。执行同一动作 chunk 时，
触觉图仍锚定生成该计划时的观测，只有 action_index 增长；不会为显示另采样一次。
未来时间相对于该来源观测，而不是当前请求时刻。无请求区域返回空区域维度。
本次提供逐区域输出协议，未增加显示 UI。异步 executor 暂不支持附带触觉结果，
显式请求会报错；默认动作响应保持不变。

## 验证

测试覆盖 10／11／12 区域、混合布局 loss／backward、掌面屏蔽和 dropout、
黑图／丢包／NaN、区域置换、旧版回归、严格 checkpoint 与迁移、CFG 合并等价、
小尺寸真实 Wan+ActionDiT 的 FP32/BF16、模拟数据 reader 和真实格式 LeRobot fixture。
synthetic fixture 验证代码对齐逻辑，不证明尚未导出的真实 bag 已正确对齐。

核心验证命令（禁用环境中无关 pytest 插件；CPU oneDNN 不支持部分 BF16 backward
时仅在测试进程中关闭 mkldnn）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:third_party OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -c 'import torch, pytest; torch.backends.mkldnn.enabled=False; raise SystemExit(pytest.main([
"tests/test_touchwam.py", "tests/test_touchwam_integration.py",
"tests/test_touchwam_regions.py", "tests/test_touchwam_regions_integration.py",
"tests/test_deform_data.py", "tests/test_bench2dex_tactile.py", "-q"]))'
```

尚未运行全尺寸 GPU 训练、真机闭环或跨域效果评估。
