# FTP-1 LeRobot v3 与形变旁路小批次

本批次只处理 FTP-1 的 14 个非 Sharpa 来源示例，不启动全量转换或训练。
复用旧数据中的六个已验证 task bucket；其他来源只暂存所选完整 episode
所需的压缩块。既有 EgoDex、原生 SharpaDex、AgiBot 数据不参与本次转换。

## 固定选择

| 来源 | 完整 episode | 时间帧 | 本次形变旁路 |
|---|---:|---:|---|
| Unit | 5 | 1,070 | Mini 相对高度实验 |
| Unit_Bimanual | 5 | 699 | Mini 相对高度实验 |
| VLA_touch | 5 | 296 | Mini 相对高度实验 |
| FreeTacMan | 5 | 1,843 | 保留 RGB，待对应传感器标定 |
| MotionTrans | 5 | 404 | 保留力信号，不生成空间形变 |
| RDP | 5 | 2,319 | 保留 RGB，待原始裁剪与标定 |
| RDP_Bimanual | 5 | 2,572 | 保留 RGB，待原始裁剪与标定 |
| RH20TCfg5Franka | 2 | 710 | 保留六维力／力矩 |
| RH20TCfg6ATIAxia | 1 | 218 | 保留六维力／力矩 |
| RH20TCfg7Tactile | 5 | 712 | 保留空间三轴原始值，待零点标定 |
| ViTaMIn | 5 | 1,206 | 保留 RGB，待对应传感器标定 |
| VisuoTactile_D-WHEEL | 5 | 2,724 | 保留 RGB，待对应传感器标定 |
| VisuoTactile_QINGLOONG | 2 | 2,060 | 保留 RGB，待对应传感器标定 |
| exUMI | 5 | 1,795 | 保留原始 9DTact 编码，待匹配 LUT／参考 |
| **合计** | **60** | **18,628** | |

时间帧按同步轨迹统计，不按相机／触觉面重复累加。抽样用于检验读取、转换、发布
与重建流程，不能代表最终训练的数据配比或质量。QINGLOONG 为控制峰值暂存，选择
任务前、后三分之一中的两条较短完整轨迹（原 episode 14、118）。

## 存储与作业

- BOS 挂载路径：`/dataset/TouchWAM/ftp1_lerobot_v3_pilot_20261028`
- 服务器控制目录：`/mnt/pfs/v89atp/workspace/setup/touchwam_ftp1_bos_pilot_20261028`
- 独占临时工作目录：`/mnt/pfs/v89atp/workspace/dataset/touchwam_ftp1_pilot_20261028/work`
- CPU 作业：`job-5kr9ub4an5bn`，1 个副本、2 CPU、16 GiB 内存，无 GPU。
- 每次只处理一个来源；工作目录预算 10 GiB，超过预算停止该来源子进程。

CPU 节点只挂载 PFS。开发机协调器通过已有 BOS 挂载读取选定块，并把已完成的
v3 与实验形变整体发布到 BOS。worker 的 ready/computed/completed 与 published
回执分别记录交接、计算完成、子进程退出与 BOS 发布。所有回执绑定固定选择的
plan SHA256；重启不会因旧 ready 回执重新生成已经发布并清理的来源。

BOS 每个对象关闭后重新打开计算 SHA256，最后才写 `_BOS_PUBLICATION.json`。
只有完整校验通过，才删除本次专用 work 中对应的原始暂存块与转换结果。
旧数据和上轮转换结果不在清理范围。检查通过 POSIX BOS 挂载完成；未获得独立
BOS API 校验，不能声称绕过了挂载缓存。

每个来源下面保存一个完整 LeRobot v3 task bucket：`meta/info.json`、
`meta/episodes/*.parquet`、`data/*.parquet`、视频、原始触觉字段与来源映射。
原始 task 帧号和 episode 号保留，抽样后重新建立连续 v3 索引；缺块有显式
有效性字段。`publication.json` 的范围始终是 `selected_pilot_episodes`。
`_catalog/` 保存总目录和固定计划；`_provenance/` 保存选择清单、标定参考来源、
权重数值数组、许可证与代码哈希。

## 形变的含义

三个原生 240×320 Mini 来源用公开法向网络和 Poisson 积分，输出逐帧的
60×80 有符号相对高度及固定尺度 32×32 正向预览。参考为 episode 首帧候选，
未证明空载；首帧已有的静态接触会被抵消。颜色通道与 XY 训练约定仍有来源
证据缺口，因此不是已标定毫米深度，更不是压力标签。

首轮结果放在 `experimental_deform/mini_v1`，保存 source/identity/reconstruction
有效性与全 false 的 `train_valid`。实际 Mini 数据的 area 数组使用 Zarr 声明的
默认值 0，完整原始索引中没有实体 chunk；原始转换器保守保留 `.valid=false`，
所以首轮严格身份策略使这些形变全部无效。不能把已完成旁路等同于已有有效形变。

第二阶段在独立 CPU 作业中重新处理三个 Mini 来源的触觉帧，复用已发布的 v3。
仅当**完整原始索引**证明 area 数组为无实体 chunk 的单传感器常量、声明的
`fill_value` 与注册区域一致，且逐帧 sensor/type 真实有效时，才允许使用
`declared_constant_area` 身份依据。稀疏选择的索引不足以证明原数组为常量。
原始 `.valid` 保持不变，补充身份依据与完整索引哈希。新版实验放在 BOS
`_deform/mini_v2/{source}`，与源 v3 通过文件哈希和发布回执绑定；保留首版记录。

不创建正式 `tactile/manifest.json`，也不把其他传感器 RGB、颜色差或六维力复制成
形变图。其他来源明确记录待补的标定条件。
详细方法、限制与文件说明见 [形变旁路说明](ftp1_deform_pilot_20261028.md)。

## 运行入口

使用固定 `control/worker_plan.json`，不要在同一个发布目录内更换选择或算法。
多机的 PFS 可能通过不同祖先路径别名挂载：跨机回执保存统一逻辑绝对路径，
各机在本地做真实路径边界检查；专用 work 及其子树不允许符号链接。

```bash
# 在有 BOS 挂载的开发机运行协调器；只读原始数据。
python scripts/data/ftp1/coordinate_pilot.py \
  --control /path/to/control \
  --raw /path/to/FTP-1-Dataset/main/revision \
  --bos-output /dataset/TouchWAM/ftp1_lerobot_v3_pilot_20261028 \
  --max-work-bytes 10737418240

# CPU 作业节点：环境提供 RANK=0 WORLD_SIZE=1。
python scripts/data/ftp1/run_pilot_worker.py \
  --control /path/to/control --encoder-threads 2
```

运行完成以 `control/catalog.json`、各来源 BOS 发布回执与实际读取验收为准，
不能仅凭提交作业或存在输出目录判断完成。

## 真实执行与验收记录

主转换作业 `job-5kr9ub4an5bn` 已 Succeeded。14 来源、60 episode、18,628 帧
均已发布并完成原始暂存／生成视频的清理。主来源 payload 共 5,570,639,536 字节
（约 5.19 GiB，包含首版实验记录），专用 PFS work 仅剩 18,645,752 字节的索引、
manifest 与清理回执。

独立读取验收共检查 539 个文件大小、285 项小元数据 SHA256，全部 episode
边界与 180 行首／中／末帧映射；57 路视频实际解码 171 帧。
生产 `LeRobotVideoIntensityDataset` 实际读取 42 个窗口，全部通过，保持
`video_only` 且不伪造动作／触觉监督。此检查不等于模型 forward 或训练。

Mini 修订作业 `job-wr53xgpsftkr` 已 Succeeded，三个来源均通过独立的
来源、哈希、数值、缺失 mask 与固定尺度审计：

| 来源 | episode | 时间帧 | 触觉面展开帧 | 数值有效重建 |
|---|---:|---:|---:|---:|
| Unit | 5 | 1,070 | 1,070 | 1,070 |
| Unit_Bimanual | 5 | 699 | 1,398 | 1,398 |
| VLA_touch | 5 | 296 | 296 | 296 |
| 合计 | 15 | 2,065 | 2,764 | 2,764 |

三个 Mini 来源现已全部发布到 `_deform/mini_v2/{source}`，payload 合计
92,284,679 字节（约 88 MiB）；发布文件重开 SHA256 校验通过。
该目录下 `_catalog/` 保存发布回执、代码快照与历史尝试证据。两次 Mini 尝试的
临时输入和输出均已清理，work 目录各只保留一个 OWNER 记录。

首个 Mini 作业 `job-19enx5czuxik` 在验收中检测到 float32 一个 ULP 的越界而
停止，未发布其输出。修订作业使用独立快照和工作目录，复用验证后的 PFS
硬链接输入，没有再次转换 v3 或重新复制视频。两个 attempt 的来源、错误、代码
哈希、预检与运行记录保留。监控到的主转换与 Mini 暂存合计峰值约 3.35 GiB；
它是采样到的最大值，硬链接按逻辑字节重复计数，不是物理分配的精确峰值。

服务器最终相关代码回归 **90 passed**。证据位于控制目录的
`validation/code_pytest_final.txt`、`validation/bos_all14_readonly.json` 与
`deform_v2/attempt2/control/`。当前实验输出全部 `train_valid=false`，未经传感器
标定或接触基线确认，不把数值审计通过解释成压力监督可用。

最终图像抽查中，Unit 的条状接触、Unit_Bimanual 的左右局部接触在原 RGB 与
形变预览中有对应关系。VLA_touch/Mango 的响应主要是宽泛边缘或倾斜场，并有
明显 marker 格纹残影，尚不能确认可靠的局部压入形状。因此三个来源的结果均
继续作为实验形变保存，VLA_touch 尤其需要重新确认空载参考、传感器标定与
marker 去除质量。抽样可视化和独立审计随 BOS `_validation/pilot_20261028` 保存。
