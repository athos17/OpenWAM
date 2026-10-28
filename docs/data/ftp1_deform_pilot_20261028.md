# FTP-1 小样例：逐帧视触觉重建旁路

`reconstruct_pilot.py` 从**已完成的 LeRobot v3 task bucket**读取触觉视频，逐帧运行公开 GelSight Mini 法向网络并积分。第一版支持 `Unit`、`Unit_Bimanual`、`VLA_touch` 的原生 240×320 图像。每个来源先处理 5 条完整 episode；每帧都处理，QA 仅均匀显示 5 帧。

输出是 **未经校准的相对表面高度变化／形变代理**，不是压力，不是毫米深度，也不是已验收训练标签。所有 `train_valid` 为 false，不创建正式的 `tactile/manifest.json`。原始触觉 RGB、数值与元数据保持不变。

## 执行

权重为此前实验导出的数值数组目录 `numeric_weights/`。可复制到 CPU 作业的只读资产目录，不需要加载 `.pt` 或安装 PyTorch。每个 `.npy` 的 SHA256 与来源 checkpoint `091bb7ee7ac259d8f77e80ca61c0370c09af6a4659bc9a4222474d4eca0bd65e` 都会验证。公开权重的来源、仓库许可证与哈希随产物保留。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/data/ftp1/reconstruct_pilot.py \
  --bucket /transient/output/Unit/buckets/allenkey_insertion \
  --weights /readonly/assets/numeric_weights \
  --episodes 5 \
  --reference-policy episode-first-candidate \
  --channel-order RGB --xy-mode normalized_row_col \
  --preview-scale 20
```

`--bucket` 传实际任务目录，不是来源根目录。需要 NumPy、SciPy、OpenCV、PyAV、PyArrow、Pillow。解码器和 OpenCV 各使用一个线程，作业层再限制 BLAS 线程。全部产物放在该 bucket 的 `experimental_deform/mini_v1/`，可以随小样例整体上传 BOS。`--output` 可指定另一实验目录。

参考策略必须显式选择。当前只支持 `episode-first-candidate`：首帧仅为**候选参照图**，绝不假设其为空载；若首帧缺失或数值不可靠，整段没有可靠相对参照，重建全部无效。若首帧已经接触物体，结果只能表示相对于该接触状态的变化，静态接触会被减掉。

## 产物

每个 episode、每个公共区域分别存放：

| 文件 | 内容 |
|---|---|
| `signed_height_60x80.npy` | float32 有符号相对重建高度，保留负值，原图重建后 area resize；单位仅是重建相对单位 |
| `positive_preview_32x32.npy` | `area_resize(clip(signed_height / 20, 0, 1))`；所有帧统一固定尺度，仅用于预览 |
| `source_valid.npy` | 原始数据的逐帧有效性，缺块不会被解释为零接触 |
| `identity_valid.npy` | 逐帧 sensor/type/area 身份字段是否有效 |
| `reconstruction_valid.npy` | 在有效原图、身份、参考前提下，数值计算是否成功；不是标定通过 |
| `train_valid.npy` | 全 false |
| `frame_index.parquet` | v3 的索引、时间戳、原 task 帧号／episode 号、有效性及来源身份字段 |
| `reference_candidate.png` | 实际使用的候选参照图 |
| `frame_diagnostics.jsonl` | 每帧的有效性、数值范围、marker 比例、饱和和失败原因 |
| `qa.png` | 原 RGB、候选参考、有符号高度、固定尺度 32×32 图的并列对照 |
| `complete.json` | 公共 region ID、episode 来源、算法和输入 SHA256、参数、限制及全部产物校验和 |

缺失／无效的图保留 NaN，并用独立 mask 标记，绝不填成“观测到零接触”。公共区域直接沿用 67 区域定义；单臂右 A 面、双臂左右 A 面各自独立。逐帧原始 area、sensor、type 必须与映射定义相符。

过程按 episode＋区域落盘，可恢复已完成项。恢复会验证配置、输入视频／数据与全部产物哈希；改变算法或参数需换一个输出目录，不能悄悄混合不同定义。只有 root `experiment.json` 状态完成时，选定 episode 的所有表面都已处理。

## 本次进一步核实的颜色与坐标约定

固定审查版本仍为 [`gelsightinc/gsrobotics@321d6a22da64529138ff10237335038fd8c5189f`](https://github.com/gelsightinc/gsrobotics/tree/321d6a22da64529138ff10237335038fd8c5189f)。在此前的 `reconstruction.py` 之外，额外审查采集类与 3D demo，发现**同一仓库的实际调用链也存在不一致**：

1. [`utilities/gelsightmini.py` 第 274 行](https://github.com/gelsightinc/gsrobotics/blob/321d6a22da64529138ff10237335038fd8c5189f/utilities/gelsightmini.py#L274) 将 OpenCV 相机 BGR 转成 RGB，裁剪后返回 `self.current_frame`。这是该采集类返回 RGB 的直接证据。
2. [`demo_view3D.py` 第 138 行](https://github.com/gelsightinc/gsrobotics/blob/321d6a22da64529138ff10237335038fd8c5189f/demo_view3D.py#L138) 对 `cam_stream.update()` 返回值又调用 `COLOR_BGR2RGB`，再送进 `get_depthmap`。按上述调用链，实际再次交换 R/B，传给重建器的是 BGR 排列，而重建函数注释称输入 RGB。
3. [`reconstruction.py` 第 149～151 行](https://github.com/gelsightinc/gsrobotics/blob/321d6a22da64529138ff10237335038fd8c5189f/utilities/reconstruction.py#L149) 创建整数坐标数组，再把归一化浮点坐标写回，实际会截断成零。权重训练时究竟使用浮点 row/H、col/W，还是这个实际零坐标，依然没有直接训练代码证据。
4. 当前公开 FTP `data_processing/parse_data_module` 审查树 `8a835b75d0102914bf7c553a578c16515c15e4dc` 只有 Aether、MotionTrans、TouchInTheWild、UniVTAC 等解析入口，没有 Unit／VLA_touch 的具体转换脚本。不能由通用格式文档替代这几个来源原始通道、裁剪与标定版本的证明。

因此，本次固定使用 `RGB + normalized_row_col` 以复现此前实验的明确约定，并把两个选项写入产物；**这是一种可追溯的实验设置，不是对网络真实标定约定的确认**。CLI 允许另开输出目录运行 `BGR` 或 `legacy_zero` 对照，不能仅挑“看起来更像接触”的结果就宣布已标定。

额外证据文件保存在 Git 外的 `research/ftp1_reconstruction_experiment_20261028/assets/`：

- `utilities__gelsightmini.py`，SHA256 `7c05f20838f16e0d4e3f77a9dac9c2105a008e3ee9889daf3df9306c5b22540e`。
- `demo_view3D.py`，SHA256 `f0baa3b03ff2ad39e315d7fc749a1da9b3d17fe97282176e2043dfebe4a99853`。
- `convention_additional_evidence.json`，下载 blob URL、固定提交和内容哈希。

## 其他来源

当前 CLI 会为其他来源写出 `unsupported_optical_calibration` 与具体原因，不把 Mini 权重套到 MCTac、OpenLoongVTouch、AllTact、9DTact 或力向量。RDP 中的 Mini 已变为 224×224，原始裁剪／视野未核实，也不强行拉伸为 320×240。详见 [逐来源重建可行性](ftp1_reconstruction_feasibility.md)。

实现验证覆盖视频连续解码、原始帧号保留、缺块和缺参考的 mask、固定归一化尺度、公共区域身份、恢复校验及不生成正式训练 manifest。数值 backend 另有已知曲面的积分测试、参考零差、颜色／XY 和 marker 插值测试；这些检查不能替代真实传感器标定。

## 第二次小样例运行：有证据的静态区域默认值

第一轮真实 Unit 的 1,070 帧触觉图全部存在，但 `right_tactile_area_gripper.valid` 全 false，因此默认严格身份门控使重建全部无效。进一步核查**完整原归档索引**发现：该 area 数组为 Zarr v2 的整数 `[T,1]`，显式 `fill_value=0`，整个数组没有存储 chunk。Unit_Bimanual 左右 area 与本次 VLA `Mango_Touch` 也具有同样证据。sensor=`GelSightMini` 与 type=`image` 则存在真实有效字段。

这与触觉图缺失不同：area 是固定区域身份，来源配置已经明确“这个单传感器字段对应 A 面”。新增一个严格限定的显式选项，允许该静态区域依据来自已声明常量，而非假装原 area chunk 存在：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/data/ftp1/reconstruct_pilot.py \
  --bucket /small-transient/Unit/allenkey_insertion-cce1b8a3 \
  --weights /readonly/assets/numeric_weights \
  --episodes 5 --reference-policy episode-first-candidate \
  --channel-order RGB --xy-mode normalized_row_col --preview-scale 20 \
  --identity-policy declared_constant_area \
  --source-archive-index /small-transient/Unit/allenkey_insertion-cce1b8a3/meta/full_source_archive_index.json
```

默认仍为 `--identity-policy strict`。启用新选项后默认输出新目录 `experimental_deform/mini_v2`，已有 `mini_v1` 不覆盖。

新选项仅接受三个已注册 Mini 来源的明确单传感器字段→A 面映射；完整索引必须与 v3 索引的 archive fingerprint 和对应 array config 一致。**必须检查完整原始索引的 chunk map 为空，不能由已裁取 episode 的稀疏 index 推断整个原数组无 chunk。** 若完整原索引包含任何该 area 的 chunk、填充值不是整数 0、不是单传感器数组或映射配置不匹配，拒绝这一路径。

source RGB 的有效性、原始 area.valid 均保持不变。sensor/type 仍必须真实有效且值匹配；任何已观测的错误 area、sensor/type 都拒绝。`region_identity_from_declared_fill.npy` 单独指出哪些帧使用了该静态身份依据。完整原索引原样随 `mini_v2/full_source_archive_index.json` 发布，其 SHA256、fingerprint、配置及限定依据写入 plan 和每个表面的 provenance。全部 `train_valid` 仍为 false，这个修复没有解决光学标定与空载参考问题。

独立审计入口：

```bash
python scripts/data/ftp1/audit_deform_pilot.py \
  --root /dataset/TouchWAM/ftp1_lerobot_v3_pilot_20261028 \
  --output /small-audit-output --variant mini_v2 \
  --sources Unit Unit_Bimanual VLA_touch
```

审计按 16 帧小块检查所有低分辨率图，不解码原视频或读取原归档，核对哈希、原帧区间、来源身份、NaN/mask、固定尺度、参考自差为零及 `train_valid`。输出 JSON 和小型 QA montage；该审计通过仅表示格式、来源与数值合同通过，不表示传感器标定通过。

第二次真实数值验收发现 OpenCV `INTER_AREA` 在饱和区域会产生 `1.0000001192092896`（float32 一个 ULP）的边界舍入。实现增加 **post-resize boundary clamp**：归一化→area resize→再 `clip(0,1)`，保持所有已在范围内的值逐位不变。QA 继续要求严格 `[0,1]`；没有放宽边界或隐藏异常。回归测试包括全饱和原图与未饱和像素不变性。首次受影响输出未发布，修订代码从原 v3 触觉重新生成独立实验旁路。
