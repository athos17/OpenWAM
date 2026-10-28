# 机器人触觉数据的动作读取合同

本轮来源明确分为 ego V、ego VT、机器人 VA、机器人 VAT。FTP 必须有已核实的动作监督；缺动作不降级成 V/VT，不用零动作补齐。带触觉的 ego 仍可使用 `lerobot_video_intensity`，FTP 使用 `lerobot_robot_intensity`。

这份合同是训练 reader 的交付接口，不表示现有 FTP 原始关节/位姿已经核实为动作。`configs/data_sources/ftp1/*.json` 中原有的 `action_enabled: false` 记录保留；数据 agent 需要完成源语义核对及数值字段转换后发布下述文件。只补数值列和元数据即可，不需要重做 RGB 视频或旧 EgoDex/Sharpa/AgiBot。

## 磁盘与配置

每个实际 LeRobot v3 bucket 提供：

1. Parquet 中的动作、当前观测状态及各自有效性列。值为 `[D]` 浮点向量，mask 为 bool 标量或 `[D]`；缺失整行或分量可用 null/false 表达。有效分量的 NaN/Inf 会报错。
2. `meta/info.json` 的 features 声明上述数值向量宽度。
3. `meta/action_contract.json`，绑定当前 info 的 SHA256，声明源身份、槽位、语义、时间关系及验证证据。
4. 同一来源各 bucket 共用一个 `normalization_stats.npy`，合同可用相对于 bucket 的路径引用。统计仅来自 train，不从验证集重算。内容为独立的 `eef` 与 `eef_state` 块，每块完整包含 `min/max/mean/std/q01/q99` 六个长度 80 的向量。

配置示例：

```yaml
type: lerobot_robot_intensity
source_kind: robot
source_family: ftp1
expected_modalities: [video_action_tactile]
dataset_dir: /path/to/published/source
require_publication: true
action_contract: meta/action_contract.json
action_mode: eef
normalize_mode: min-max
unify_action: false  # reader 已输出 canonical80，不再做第二次散射
tactile_mode: required
```

机器人无触觉数据可显式使用 `tactile_mode: off` 与 `[video_action]`。本轮选中的 FTP 夹爪触觉配置使用 `required`，未完成触觉发布时不能静默变成普通视频。

## 单臂 canonical EEF10 合同示例

下面的单臂示例映射右臂 34–43：xyz、rot6d、gripper。它是接口模板，不是任何 FTP 来源已通过验证的声明。发布者需替换摘要、来源与验证证据，确认坐标变换、末端定义、长度/角度单位、rotation 6D 顺序和夹爪开度方向。

```json
{
  "schema": "touchwam_robot_action_v1",
  "source_kind": "robot",
  "source_id": "ftp1/REPLACE_WITH_SOURCE",
  "dataset_info_sha256": "REPLACE_WITH_SHA256",
  "verification": {
    "status": "unverified",
    "evidence": []
  },
  "action_semantics": "recorded_command",
  "terminal_action_valid": false,
  "action": {
    "column": "action",
    "validity_column": "action.valid",
    "slots": ["34-43"],
    "representation": "openwam_canonical_components_v1",
    "coordinate_frame": "REPLACE_WITH_VERIFIED_FRAME",
    "units": ["m", "m", "m", "1", "1", "1", "1", "1", "1", "aperture_fraction"]
  },
  "proprio": {
    "column": "observation.state",
    "validity_column": "observation.state.valid",
    "slots": ["34-43"],
    "representation": "openwam_canonical_components_v1",
    "coordinate_frame": "REPLACE_WITH_VERIFIED_FRAME",
    "units": ["m", "m", "m", "1", "1", "1", "1", "1", "1", "aperture_fraction"]
  },
  "normalization": {
    "mode": "min-max",
    "population": "train",
    "file": "../normalization_stats.npy",
    "sha256": "REPLACE_WITH_SHA256",
    "action_key": "eef",
    "proprio_key": "eef_state"
  }
}
```

接受条件为 `verification.status=verified` 且有非空证据。reader 不对原始腕部六维位姿猜测欧拉角顺序，不对关节值猜 FK，也不自动将观测状态冒充记录的控制命令。

## 时间、缺测与归一化

- `recorded_command`：动作目标为参考时刻起 `0, 1/30, …, 31/30 s`（由 sampling_fps 和 num_frames 配置决定），低频命令因果保持，不从未来借命令；`terminal_action_valid=false` 排除末行常见的占位/重复命令。
- `future_state_target`：仅当源已验证允许以实现轨迹为监督时显式选择。动作与 proprio 必须引用同一已转换状态列、同一槽位。目标改为 `1/30, …, 32/30 s`，选择最近原始观测；必须严格晚于参考时刻。同一当前观测不会被标成未来动作。此类型在样本与训练合同中保持独立身份，不宣称是原始命令。
- 当前 proprio 取参考帧，使用独立 state 统计。动作和状态 mask 分开，每个窗口必须有真实动作监督和当前状态；否则报错，不能降级。
- 越过 episode 尾部的目标使用 false mask，不跨 episode 借标签。实际目标源时间通过 `action_label_time_offsets_s` 暴露，低频数据升采样不增加独立小时数。
- 未用槽及 rot6d 槽统计必须为恒等：min/q01=-1，max/q99=1，mean=0，std=1。其余槽使用已发布统计。训练与部署使用同一统计包和 mode，未用槽保持 0。
- 归一化是在 reader 中完成的；混合数据集 `normalization_stats=None`，防止第二次归一化。checkpoint 另行打包每个来源的统计文件与 SHA256。

## 验证

`scripts/preflight_touchwam.py` 会对每个来源检查声明的本体/模态，再解码抽样窗口并使用实际触觉 collator 检查。机器人动作合同缺失、模态错误、触觉发布缺失或 normalizer 不匹配均阻止启动。

新增测试 `tests/test_touchwam_robot_intensity.py` 包含真实 v3 视频与 episode offset、动作时间/尾帧、各类缺测、独立 state 统计、统计包部署往返，以及 reader→小型真实 Wan/ActionDiT→VAT loss/backward。它不证明某个尚未发布的 FTP 来源已经语义合格。
