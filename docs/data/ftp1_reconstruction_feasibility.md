# FTP-1 非 Sharpa 触觉：形变图重建的可行性与接入方案

审查日期：2026-10-28。范围为当前 FTP-1 发布版 `d287c9c532514a08b2eb29bf1259997bb32f4f3b` 的 14 个非 Sharpa 来源。本报告只确认数据语义、公开重建资产与实现条件；没有把未验证的 RGB 伪标签加入训练，也没有声称完成了真实压力标定。

## 结论

可以从**部分**视触觉图恢复相对于无接触表面的形变／压入深度，再按固定规则归一化为 `[0,1]` 接触强度。不能把所有来源的 RGB 统一灰度化后称为压力图。

建议首先验证两条明确路线：

1. **GelSight Mini → 表面法向 → Poisson 积分 → 相对压入图**。官方公开了小型重建网络及权重，优先从保留 240×320 图像的 `Unit`、`Unit_Bimanual`、`VLA_touch` 取小样本；`RDP` 的 224×224 版本需要额外核实裁剪与像素几何。
2. **exUMI 的 9DTact 派生传感器 → 保留下来的当前灰度通道 → 匹配的灰度—深度标定 LUT → 压入图**。这里的数据已经是处理后的三通道表示，不是普通触觉 RGB。公开存在标定文件，但传感器身份、基线及裁剪存在未解决的匹配问题，不能直接全量套用。

另外，`RH20TCfg7Tactile` 的 uSkin 本来就是 **2 个 4×4×3 的空间税元阵列**，应走法向分量／基线校准路线，不需要 RGB 重建。

“压入深度图”与“压力分布图”是两种量。归一化去掉单位，并不会使它们物理等价。TouchWAM 可以把经验证的正向压入量作为接触强度代理，但必须记录 `reconstruction_output=indentation_depth`、传感器来源和校准版本。要得到压力，还需要材料、厚度、边界条件及力学标定／监督；不能只由图像亮度或总六维力唯一恢复。

## 实际来源与可用路线

以下尺寸为 FTP 样本字段的 **H×W**，不是原相机型号规格。字段形状及 sensor 字符串见仓库 `configs/data_sources/ftp1/*.json`、`regions/*.json`；本地原始 Zarr 抽查依据为 `sample-audit.json`。

| FTP 来源 | 已确认的实际传感器／表示 | 当前空间数据 | 可行路线与当前条件 |
|---|---|---|---|
| `Unit` | 归档字段为 GelSight Mini，带标记点；论文描述 GelSight 视触觉 | 1×240×320×3 | 优先试验 Mini 官方法向／深度网络；须匹配 RGB/BGR、相机视野裁剪、无接触参考及实例差异 |
| `Unit_Bimanual` | 两臂各一个 GelSight Mini | 左右各 1×240×320×3 | 同上；两实例独立基线和校准身份，不能共享一个任意参考图 |
| `VLA_touch` | GelSight Mini；原系统 Franka＋Robotiq 一侧装传感器 | 1×240×320×3 | 同上；优先验证实际空载帧是否保留，不能自动把轨迹第一帧视为空载 |
| `RDP` | GelSight Mini＋改进 MCTac，另有夹爪总力 | 每种各 1×224×224×3 | Mini 可在确定原裁剪／缩放后试验；MCTac 的标记运动代码可用，但尚无匹配的深度／压力重建资产证据 |
| `RDP_Bimanual` | 两臂的 GelSight Mini／MCTac，另有总力 | 每臂每种各 1×224×224×3 | 同上；还须保留缺失 chunk 的 mask，声明了数组却无实际块的数据不是零接触 |
| `FreeTacMan` | 归档名 `FreeTacMan`；原论文明确为基于 MCTac 的改进设计 | 2×224×224×3 | 不能冒用 Mini、DIGIT 或标准 MCTac 权重。需要该硬件版本的几何／光学标定或带压入真值的训练数据 |
| `exUMI` | 原论文明确为升级的 9DTact；归档 sensor 名为 `exUMI` | 1×460×680×3，实际拼接两面 | 当前灰度＋正／负亮度变化编码。可研究匹配 LUT 的压入重建，详见下文；不对三个通道取均值 |
| `ViTaMIn` | **AllTact Fin Ray**；不是 GelSight／DIGIT 平面凝胶传感器 | 2×224×224×3 | 内部相机同时看到手指整体弯曲与局部接触形变。图像像素不等于接触表面坐标，需要 AllTact 专用几何／形变标定或监督模型 |
| `VisuoTactile_D-WHEEL` | 归档名 **OpenLoongVTouch** | 两臂各 2×240×240×3 | 名称并未提供可复用的 Mini/DIGIT/DenseTact 标定身份证明。本次未获得匹配的官方重建模型／标定文件；先保留原图和身份 |
| `VisuoTactile_QINGLOONG` | 同为 OpenLoongVTouch | 两臂各 2×240×240×3 | 与上行同类，但仍需按实际设备／采集批次区分校准；不能仅凭外观认定相同光学模型 |
| `RH20TCfg7Tactile` | uSkin＋ATI Axia80M20 | uSkin 2×4×4×3；另有六维力矩 | 从已确认的法向通道减空载基线、纠正符号后形成 4×4 法向接触图；当前原值有较大偏置，不能直接除最大值或做三轴模长冒充法向压力 |
| `RH20TCfg5Franka` | FrankaTorque | 1×6 向量 | 没有空间接触图；不能从六维量恢复每个像素的分布 |
| `RH20TCfg6ATIAxia` | ATI Axia80M20 | 1×6 向量 | 同上 |
| `MotionTrans` | InspireHand 指关节／拇指力信号 | 4×1、1×2 数值 | 不是视触觉图，也不是已标定的接触面压力矩阵；不能复制成一块空间图 |

**没有在这 14 个来源中确认 DIGIT 或 DenseTact 数据。** 这些名称出现在论文的相关工作／参考文献里，不能据此认定 FTP 实际包含相应传感器。exUMI 的代码目录叫 `DTact`／`9DTact`，真实硬件是论文描述的 **9DTact 派生改版**，也不能把它直接等同于任意 DTact 实例。

## 路线 A：GelSight Mini 的可验证实现

官方 [gsrobotics](https://github.com/gelsightinc/gsrobotics/tree/321d6a22da64529138ff10237335038fd8c5189f) 提供：

- [`models/nnmini.pt`](https://github.com/gelsightinc/gsrobotics/blob/321d6a22da64529138ff10237335038fd8c5189f/models/nnmini.pt)：75,317 字节的小型网络权重。
- [`utilities/reconstruction.py`](https://github.com/gelsightinc/gsrobotics/blob/321d6a22da64529138ff10237335038fd8c5189f/utilities/reconstruction.py)：`RGB2NormNet` 接收 RGB 加像素位置，输出法向的两个分量，转换为梯度并做 Poisson 积分；支持标记点区域插值。
- [`default_config.json`](https://github.com/gelsightinc/gsrobotics/blob/321d6a22da64529138ff10237335038fd8c5189f/default_config.json)：默认 320×240、边界裁剪比例 0.15。
- 官方 README 说明发布的是一个参考校准模型；给出该 240×320 处理尺寸下约 0.0634 mm/pixel 的示例比例，同时说明 Mini 不是计量设备。不能把该比例直接套到任意 FTP 裁剪版本。

建议的离线管线：

```text
保存的 Mini 图像
  → 明确通道次序、原裁剪、镜头方向与有效 ROI
  → 法向网络，标记点区域检测／插值
  → 梯度积分成表面高度
  → 减去相同传感器的经确认空载参考高度
  → 验证压入正方向、接触边界及质量 mask
  → 固定尺度归一化成 [0,1]
  → 32×32 区域图 + region_id + sensor/profile_id + valid
```

不能直接把现版 live SDK 当离线数据清洗黑盒：

1. SDK 用前 50 帧做零点初始化，而 FTP 轨迹可能从接触中途开始。必须使用显式、经验证的参考帧，不能让每个 episode 的任意前 50 帧改变监督定义。
2. 当前审查版本的 `px_positions_normalized = np.vstack(np.where(...)).T` 初始为整数，再把除以图像尺寸的浮点结果写回同数组，存在坐标截断问题。离线移植应先转 float，并验证模型版本与训练时的坐标约定；不能仅修代码后便声称深度正确。
3. SDK 的相机输入还执行裁剪和 resize；FTP 恰好是 240×320，并不证明与模型的视野一致。224×224 方图更不能仅拉回 320×240 就恢复原像素几何。
4. RGB→法向→积分得到的是几何高度，不是压力。黑色 marker 的亮度不是压力低值，必须用标记掩码处理。

首轮试验建议每个来源抽取若干实际空载—接触—卸载片段，检查空载误差、压入方向、接触区域、漂移、饱和和跨设备差异。完成这些验证后再把 profile 标为 `validated` 并扩展到全量。只看一张“像形变图”的热力图不能验收。

## 路线 B：exUMI 9DTact 派生数据的解码与限制

原代码见 [exUMI 固定提交](https://github.com/silicx/exUMI/tree/3fba41ca40c22559b4aec91cf2092c380c479a4b)。`AR_03_align_trajectory.py` 调用 `raw_image_2_xy_repr`；生成时的 **BGR** 三通道为：

```text
B = 当前灰度 G(t)
G = 7 × max(G(t) − G(ref), 0)
R = 7 × max(G(ref) − G(t), 0)
```

视频以 RGB 解码进入 replay buffer 后，**RGB 的第 2 通道（蓝色）仍是当前灰度**。`AR_07_generate_replay_buffer.py` 将左传感器逆时针旋转 90°、右传感器顺时针旋转 90°，再水平拼接；当前 FTP 图为左右各 460×340。

这意味着可以优先保留蓝通道研究压入重建；但 R/G 并非可靠的单调形变强度：写入 uint8 时乘 7 可发生回绕。即使无损数组理论上可以做模逆，上游还经过视频编码，微小通道误差会使模逆不稳定。**不能使用模逆或把 R/G 当作线性压力标签。** FTP 当前 Zarr 的无损 Blosc 压缩也无法撤销上游视频损失。

公开仓库中确有左右传感器的 `Pixel_to_Depth.npy` 及参考图，但本次核验发现：

- 左 LUT 36 项、右 LUT 35 项，均单调；右 LUT 第 0 项约 `−0.5594`，第 2 项约 `0.0469`，不能把任意第 0 项当作正确零点。
- 原代码默认 `lighting_threshold=2`。避免 uint8 溢出的等价意图应明确为 `depth = LUT[max(reference_gray-current_gray, 2)] - LUT[2]`，并保留实际平滑／截断约定。
- 公开 `rectify_crop_ref.png` 为 **460×345**，而实际采集配置裁剪为 **460×340**，FTP 中还进行了两侧相反旋转。不能直接将公开参考图 resize 到 FTP ROI 后宣称完成匹配。
- 官方采集说明要求每个原视频空夹爪开始，代码在原视频 `i_frame==0` 更新参考。但 replay buffer 可以按任务区间截取，**FTP episode 首帧未必就是该原视频的零点帧**。须证明原视频／session 身份与保留下来的参考帧关系。
- 需要确认发布的 left/right LUT 属于 FTP 对应采集实例及光照／凝胶版本。仅存在同名文件不构成标定匹配证据。

因此首个可执行验证包应包含：一个能追溯原视频开始点的 episode、左右 ROI/旋转定义、实际空载灰度参考、对应 LUT 身份证据，以及恒定零点索引。先验证亮度差对受控压入量单调，再输出正向压入图。缺少匹配条件时，可以保留三通道数据用于后续独立研究，但不能把它启用成确定的压力监督。

## 其他光学来源为什么不能沿用这两个转换器

- **MCTac／FreeTacMan**：RDP 公开了 marker 检测、轨迹、PCA embedding 和 AR 形变显示代码，这证明可以提取标记运动，不证明已存在匹配的每像素压入深度／法向压力标定。FreeTacMan 又改动了硬件设计，不能继承一个无关传感器的 RGB—法向映射。
- **AllTact Fin Ray**：原论文明确内部相机同时看到整体手指弯曲与局部表面形变。要将图变成接触面压力矩阵，须先把图像区域关联到真实接触表面并解决整体弯曲影响；整张图灰度或光流模长不是法向压力。
- **OpenLoongVTouch**：归档证明传感器族名称、四个接触面及图像尺寸；尚未证明其光学模型与已公开的 Mini／9DTact 模型相同。应先获得对应设备的原始处理／标定说明。未知型号不补猜。

## 接到 TouchWAM 的定义

每一个可启用的传感器 profile 至少需要以下字段；它按硬件实例／光学设置区分，不只是按数据集名字区分：

| 定义 | 内容 |
|---|---|
| 输入语义 | 原始 RGB、BGR、处理后的 xy 编码，或者三轴 taxel；不得只写“image” |
| 空间变换 | 原始 ROI、校正、裁剪、旋转、镜像及原始到公共区域映射 |
| 重建身份 | 算法版本、代码哈希、权重／LUT 哈希、校准实例与来源证据 |
| 空载参考 | 参考图／参考值、所属传感器和 session、确认为无接触的依据；不从未来帧估计历史输入的参考 |
| 输出语义 | `indentation_depth`、已标定 normal pressure，或明确的其他量；不能暗中把总力称作空间压力 |
| 归一化 | 固定的零点、噪声阈值、上界或训练集标定尺度；不逐帧拉伸 |
| 有效性 | 传感器是否安装、帧是否实际存在、ROI／像素可用性、重建质量；缺失不能补成有效零图 |
| 验收 | 空载／接触／卸载片段、方向与单调性、漂移、压缩与 resize 敏感性、空间映射及时间戳检查 |

最终依然是统一的 `32×32` 单通道区域图，使用既定 region_id 和来源身份进入共用 token 编码器。统一发生在**经验证的接触强度表示**层；不同传感器的光学反演留在离线转换阶段。一个 profile 不通过，不应阻止该 episode 的有效视频／已验证动作参与相应监督，也不应伪造它的触觉标签。

实现接口已在 `scripts/data/ftp1/build_intensity.py` 中预留严格的 `depth_reconstruction` 插件、校准资产哈希、ROI、固定尺度、缺帧 mask 和独立 sidecar。`depth_plugins/dtact_lut.py` 是数值后端，已经显式要求 `reference_index`，不代表 exUMI 的真实标定条件已经验证。`intensity_examples/depth_reconstruction.json` 仍是 `unverified` 示例，不作为生产已验证 profile。

## 可复核来源

- [UniT 论文](https://arxiv.org/abs/2408.06481)；归档实际 `GelSightMini` 字段是本报告型号判断的直接依据。
- [VLA-Touch 论文](https://arxiv.org/abs/2507.17294)；对应归档字段 `GelSightMini` 与硬件章节一致。
- [RDP 论文](https://arxiv.org/abs/2503.02881)、[源码固定提交](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/tree/824c5e8de1fd1811106907a04b5f0186e0138c0b)、[marker/PCA 使用说明](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/824c5e8de1fd1811106907a04b5f0186e0138c0b/docs/tactile_embedding_guide.md)。
- [FreeTacMan 论文](https://arxiv.org/abs/2506.01941)、[官方源码](https://github.com/OpenDriveLab/FreeTacMan/tree/9285740a5d33385d3a9cf5ccdb185e3387b547bd)。
- [ViTaMIn 论文](https://arxiv.org/abs/2504.06156)、[AllTact Fin Ray 论文](https://arxiv.org/abs/2504.18064)。
- [exUMI 论文](https://arxiv.org/abs/2509.14688)、[采集说明](https://github.com/silicx/exUMI/blob/3fba41ca40c22559b4aec91cf2092c380c479a4b/DATA_COLLECTION.md)、[replay buffer 拼接代码](https://github.com/silicx/exUMI/blob/3fba41ca40c22559b4aec91cf2092c380c479a4b/scripts_slam_pipeline/AR_07_generate_replay_buffer.py)。
- [9DTact 公开形状标定代码和示例资产](https://github.com/linchangyi1/9DTact/tree/453bd85094efa5e91a6a809a7b9194c0b7ac6a01/shape_reconstruction)。这是标准 9DTact 的公开能力证据，不能替代 exUMI 某次采集的标定匹配。
- [MCTac 2.0 官方仓库](https://github.com/Tacxels/MCTac-2.0/tree/a2554947b2cc1f7661e183555d3165f1d8afab28)：本次版本主要为硬件说明和图片，未看到可直接用于上述 FTP 图像的深度权重。
