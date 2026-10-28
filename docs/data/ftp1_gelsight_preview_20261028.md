# FTP-1 GelSight Mini：真实小样本重建实验

日期：2026-10-28。**状态：`unverified_preview_only`，未启用训练。**

本实验确认公开 Mini 网络可以对部分 FTP 图像产生与可见接触结构对应的相对几何图；同时发现不同采集源的光学设置、参考帧和边界漂移仍会显著影响结果。现在不能把所有 Mini 图像直接批量转换成可信压力标签。

## 取了哪些真实数据

从 FTP-1 发布版 `d287c9c532514a08b2eb29bf1259997bb32f4f3b` 直接读取以下任务前两条 episode，每条均匀取 5 帧。只按索引解码对应 Zarr chunk，不加载整个源数组，不重复转换数据集。

| 来源与任务 | 轨迹 | 触觉面 | 抽取触觉帧 | 输入 |
|---|---:|---:|---:|---|
| Unit / allenkey_insertion | 2 | 右侧一路 | 10 | 240×320 RGB 形状 |
| Unit_Bimanual / chips_grasping | 2 | 左、右各一路 | 20 | 两份 240×320 RGB 形状 |
| VLA_touch / Wipe_Manipulation | 2 | 右侧一路 | 10 | 240×320 RGB 形状 |
| VLA_touch / Cup_Touch | 2 | 右侧一路 | 10 | 240×320 RGB 形状 |
| 合计 | 8 | — | 50 | 40 个采样时刻 |

这里的左、右是源数据的传感器字段名。没有在本实验中启用公共区域映射或训练监督。全部小样本约 15 MiB，原归档未修改；原始字段、source frame index、episode 边界、样本 SHA256 均保存在 `samples.json`。

## 实际结果

完整对照图每行依次为：场景相机、当前原触觉图、候选参考图、有符号高度差、固定尺度的正向预览、32×32 预览。所有图用**同一个固定 20 相对重建单位**作为显示上限，没有逐帧 min-max。这个 20 不是 20 N、20 mm 或真实标定上界。

| 来源／面 | 50 帧中的该组数量 | 有符号重建差范围 | 可观察结果 |
|---|---:|---:|---|
| Unit / right | 10 | −5.14 ～ 15.57 | 接触中段出现与扳手原图对应的竖条；episode 0 首尾变化在约 ±0.44 内，中段峰值约 13.85／14.86 |
| Unit_Bimanual / right | 10 | −7.72 ～ 29.78 | 接触薯片时出现局部斑块；两条 episode 的接触响应不同，固定显示尺度下最大饱和像素比例约 1.0% |
| Unit_Bimanual / left | 10 | −1.19 ～ 0.58 | 抽样图几乎没有局部接触变化；不能仅据此把左面判为始终无接触 |
| VLA / Wipe_Manipulation | 10 | −1.72 ～ 2.07 | 片段开始时场景中已持有海绵，触觉图变化小；首帧作参考会消除已有的静态接触分量 |
| VLA / Cup_Touch | 10 | −7.60 ～ 13.80 | 出现明显图像上边缘响应，离开物体后也有残余，尚不能确认是真实局部压入或边界／光照漂移 |

所有 50 帧的预测法向均没有出现 `nx²+ny² ≥ 1`。阈值标记点掩码占比约 11.4%～17.0%，进行了梯度插值，仍可见少量 marker 残纹。**数值有限和视觉上像接触形状，只是实验检查，不是几何／力学标定通过。**

以下为当前工作区内的真实产物；原始样本、权重和大图留在 Git 仓库外：

- [全部初始 8 张对照图](../../../research/ftp1_reconstruction_experiment_20261028/results/index.html)
- [Unit 扳手接触完整片段](../../../research/ftp1_reconstruction_experiment_20261028/results/Unit_right_episode_0.png)
- [双臂 Unit 右面片段](../../../research/ftp1_reconstruction_experiment_20261028/results/Unit_Bimanual_right_episode_0.png)
- [VLA 持物擦拭片段](../../../research/ftp1_reconstruction_experiment_20261028/results/VLA_touch_right_episode_0.png)
- [VLA Cup_Touch 补充 2 张对照图](../../../research/ftp1_reconstruction_experiment_20261028/results_touch/index.html)
- [初始逐帧数值与算法选项](../../../research/ftp1_reconstruction_experiment_20261028/results/experiment.json)
- [Cup_Touch 逐帧数值](../../../research/ftp1_reconstruction_experiment_20261028/results_touch/experiment.json)

## 网络、坐标和积分的具体约定

公开资产来自 [`gelsightinc/gsrobotics@321d6a22da64529138ff10237335038fd8c5189f`](https://github.com/gelsightinc/gsrobotics/tree/321d6a22da64529138ff10237335038fd8c5189f)：

- `nnmini.pt`：75,317 字节；SHA256 `091bb7ee7ac259d8f77e80ca61c0370c09af6a4659bc9a4222474d4eca0bd65e`。
- 仓库 `LICENSE` 为 GPL v3 文本；本次未发现独立的权重许可证。源码、权重、许可证原文与下载 URL／哈希均保存在外部 `assets/`。
- 读取 checkpoint 使用 `torch.load(..., weights_only=True)`，只将 8 组数值权重／偏置导出成 `.npy`。训练预处理插件仅读取显式哈希校验的数值数组。
- 网络为 `5→64→64→64→2`，输入是 3 个归一化颜色分量和 `row/H, column/W`，前三层后为 ReLU，推理无 dropout。输出两个法向分量，按 `nz=sqrt(1−nx²−ny²)` 转为梯度。
- 对真实 Unit 第 97 帧全部 76,800 像素，NumPy 后端和 PyTorch 线性层推理的法向最大绝对差为 **3.58×10⁻⁷**，见 `weight_numerical_check.json`。

没有直接复制 live SDK 的自动处理流程。当前后端明确做了以下改动，因此不能把它称为“已校准的官方原样输出”：

1. 使用浮点 `row/H, column/W`。所审查 SDK 把浮点归一化值写回整数坐标数组，实际会截断成 0。本实验另外计算 `legacy_zero` 对照；Unit 部分接触帧两种结果的逐像素平均高度差可达 1.42 相对单位。发布权重训练时究竟使用哪一种约定，仍需要来源确认。
2. 使用前向差分网格上的 Neumann 最小二乘 Poisson 解，正确保留零 DC 自由度，以均值为零固定积分常数。它通过已知解析高度场的梯度积分测试；官方所审查 solver 的频率索引与此不同。本实验不能由此推断物理深度更准确。
3. 黑 marker 用固定 RGB 灰度阈值 70 掩蔽，再用最近有效邻点梯度插值；不把 marker 黑色当作“压力低”。此处也与原 SDK 的插值函数不同。
4. 显式提供同一 episode 第一帧作为**未经验证的参考候选**，没有自动累计前 50 帧，也没有从未来帧估计基线。真实训练必须更换为已验证的对应传感器／session 空载参考。

当前主预览按储存数组通道顺序作为 RGB。另计算 BGR 交换对照，双臂 Unit 右面部分接触帧的平均高度差高达 **11.00** 相对单位。这说明颜色顺序不是可忽略的工程细节；仅凭数组末维为 3 或接口名带 RGB，仍不能替代原采集／转换代码与网络训练输入约定的核验。

## 这次能确认与下一步条件

已确认：实际源字段、样本与帧索引、权重身份、网络形状、数值推理一致性、部分片段的接触结构对应，以及当前方法对通道／参考的敏感性。

尚未确认：同型号不同传感器的光照与凝胶一致性，保存图与网络输入的裁剪视野是否相同，真正空载参考，法向符号／尺度及卸载漂移，非接触边界锚点，以及与实测压入或力的单调关系。均值为零的积分会带来正负两侧，直接取正不能自行解决接触区外偏移；生产定义可提供经过验证的非接触 `zero_anchor_mask`。

建议在决定 200 小时夹爪数据配额前，优先对 Unit／Unit_Bimanual 的具体传感器批次完成“空载—受控接触—卸载”标定验证。VLA 暂时保留原视触觉和示例，不因为同为 Mini 就启用深度监督。通过后再建立每个 sensor/session 的固定基线、光学约定、有效 ROI、归一化尺度和质量规则。输出应命名为**归一化形变／接触强度代理**，不能宣称真实压力。

代码入口：`scripts/data/ftp1/depth_plugins/gelsight_mini.py`、`scripts/data/ftp1/preview_gelsight.py`；数值 fixture 测试在 `tests/data/test_gelsight_mini.py`，6 项通过。没有新增已验证训练 profile，没有全量运行 RGB→形变转换。

```bash
# 在可读取原归档的机器上，只抽小样本；环境需 numcodecs、pyarrow。
python scripts/data/ftp1/preview_gelsight.py sample \
  --source-root /dataset/huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/main/d287c9c532514a08b2eb29bf1259997bb32f4f3b \
  --census configs/data_sources/ftp1/archive_census.json \
  --output /external/mini_experiment/samples

# assets 为上面固定提交下载的权重、许可证及 provenance.json，放在 Git 外。
OPENBLAS_NUM_THREADS=1 python scripts/data/ftp1/preview_gelsight.py render \
  --samples /external/mini_experiment/samples \
  --assets /external/mini_experiment/assets \
  --output /external/mini_experiment/results --preview-scale 20
```
