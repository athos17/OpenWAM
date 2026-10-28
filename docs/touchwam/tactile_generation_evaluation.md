# 触觉生成与留出集评估

训练和生成使用相同的四个模态名称。新 `generation_modality` 是显式选择；不传时保留原来生成动作的 API 行为。

| 名称 | 输入／生成流 | 机器人 proprio／动作 |
| --- | --- | --- |
| `video_only` | V | 不接收、不构造 |
| `video_action` | VA | 真实 proprio，生成动作 |
| `video_tactile` | VT，适用于 ego 触觉 | 不接收、不构造 |
| `video_action_tactile` | VAT，适用于机器人触觉 | 真实 proprio，生成动作和触觉 |

本轮 FTP 来源按机器人数据处理，需有真实且定义明确的动作监督；没有动作的 FTP 来源应从训练配方排除。VT 接口服务 ego 触觉，不是把缺少动作的机器人数据自动降级为 VT。

```python
result = model.generate(
    schedule=[(1000, 1000), (500, 1000), (500, 500), (0, 0)],
    prompt="pick up the object",
    generation_modality="video_tactile",
    first_frame_image=scene_image,
    tactile_maps=history,                    # [4,N,1,32,32], float [0,1] 或 uint8
    tactile_valid=history_valid,             # [4,N], bool
    tactile_region_ids=region_ids,           # [N], 模型公共区域编号
    tactile_layout_present=layout_present,   # [N], bool
    tactile_sensor_ids=sensor_ids,           # [N], 模型来源编号
    tactile_spatial_valid=roi,               # [N,1,32,32], bool
    decode_video=False,
)
assert result["actions"] is None
```

VA/VAT 的二元 schedule 仍为 `(video timestep, action timestep)`，触觉跟随动作的 sigma。VT 的第二轴直接是触觉 timestep，动作调度不参与；V 的第二轴不用作生成流。两个轴均从各自 `num_train_timesteps` 到 0，允许平台段，CFG 的合并和逐次调用都支持无动作。当前各 scheduler 通常为 1000。使用 `schedule_sync(model.video_scheduler, model.tactile_scheduler, ...)` 可生成 VT 调度。改变某 scheduler 的时间尺度时，触觉使用其自身 timestep，Euler 更新按相应 sigma 差计算。

Wan 原预处理只产生有条件文本；TouchWAM 在 `cfg_scale>1` 时额外编码空文本，并为有条件／无条件分支分别保留文本 padding mask。默认仍为 `cfg_scale=1`，CFG 是否改善预测需用留出集测量。

## 评估 API

`scripts.validation.touchwam_tactile_eval.evaluate_sample` 接收一条已经由训练 reader 处理好的留出集样本。它只把**第一帧场景图、历史触觉、布局和 VAT 的归一化 proprio**送入 `model.generate`。未来触觉真值只用于采样完成后的评估；不会把流匹配 velocity loss 解释成图像误差。

接受 reader 的 `first_frame_image=[PIL.Image]` 格式，也接受单张 PIL；评估后恢复调用前的模型 training/eval 状态。

```python
from scripts.validation.touchwam_tactile_eval import evaluate_sample
from openwam.deploy.denoise_schedule import schedule_sync

auxiliary = model.tactile_scheduler if sample["training_modality"] == "video_tactile" else model.action_scheduler
schedule = schedule_sync(model.video_scheduler, auxiliary, num_steps=20)
report = evaluate_sample(model, sample, schedule=schedule, output_dir="eval/episode_10", seed=42)
```

要求：区域/来源身份显式；物理时间网格与 checkpoint 匹配；强度固定为 `[0,1]`；未来测量、历史测量、空间 ROI、区域布局各自有 mask。VAT 样本还需有真实动作（只取其长度及有效槽位，不把未来动作值作为输入）和**已归一化到训练空间的 proprio**。

输出包括：

- `metrics.json`：生成图、全零图、保持最近一次有效历史图的 MAE/RMSE；分区域、分预测时间。
- `predictions.npz`：预测图、真值、实际评分像素 mask、hold-last 图与共同支持 mask。
- `region_XX.png`：每个区域独立显示真值、预测、全零、hold-last 和绝对误差；列为未来时刻，灰度范围始终 `[0,1]`，粉色为缺失/不评分区域。

没有有效历史的区域不能建立 hold-last 基线。报告同时提供 `all_target_support`（模型／zero 在全部可评分标签上，hold-last 仅在可建立的区域上）及 `common_history_support`（三者在完全相同的支持上比较）。数值汇总用真实像素数和误差和，避免平均不同大小样本的 MAE。这里的单位是归一化强度，不是 N、Pa 或 mm。全零接触图仍是有效观测。

## CLI 留出样本格式

CLI 使用少量 JSON + NPZ 样例，可从 reader 的 validation/test 样本导出，不需要重新转换整个数据集。样本划分必须先按 episode／采集会话完成；CLI 记录清单及来源，但不能证明清单与训练集没有重叠。

`samples.json`：

```json
{
  "schema": "touchwam_tactile_eval_samples_v1",
  "split": "validation",
  "samples": [{
    "arrays": "sample_000.npz",
    "first_frame_image": "sample_000.png",
    "source_id": "ego_touch",
    "episode_index": 10,
    "reference_time_s": 1.0,
    "training_modality": "video_tactile",
    "tactile_codec": "tactile_intensity_v1",
    "prompt": "pick up the object",
    "num_frames": 9
  }]
}
```

NPZ 保存 `tactile_maps`、`tactile_valid`、`future_tactile_maps`、`future_tactile_valid`、`tactile_layout_present`、`tactile_region_names`、`tactile_sensor_names`、`tactile_history_times_s`、`future_tactile_times_s`。可选 `tactile_spatial_valid` 和 `tactile_query_mask`。名字用 NumPy Unicode 数组，禁止 object/pickle。VAT 另外保存归一化 `proprio`、真实 `action`、`action_mask`。PNG 是训练 reader 同样排布/裁剪后的首帧场景图。

```bash
python -m scripts.validation.touchwam_tactile_eval \
  --checkpoint /path/to/run/checkpoint_step_1000.safetensors \
  --manifest /path/to/heldout/samples.json \
  --output /path/to/eval --device cuda --dtype bf16 --steps 20
```

checkpoint 旁需有自包含 `config.yaml` 和 tokenizer 等资产。CLI 严格加载训练后的区域 TouchWAM 参数，使用样本已经归一化的 proprio，因此 **ego VT 不需要选择机器人 normalizer 或 action source**。它不是启动训练的入口，也不修改输入样本。输出 `summary.json` 记录样本清单 SHA256、checkpoint 文件信息、种子、步数和聚合指标。

机器人策略部署仍使用正常的 source normalizer：`video_action` 和 `video_action_tactile` 都可选；V/VT 来源不能冒充机器人动作归一化来源。
