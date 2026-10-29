# FTP-1 冻结训练子集：2026-10-29

训练目录：`dev-cpu:/mnt/pfs/v89atp/workspace/dataset/ftp1_frozen25h_v1_20261029`。

这份子集冻结的是 2026-10-29 19:49:32（北京时间）审计快照中的 **871 个已完成分片、5,196 个 episode、25.914875 小时**。后续转换出的分片不会自动加入。原始转换目录保持不变。

8547 个媒体和数据 parquet 文件采用硬链接，共享约 **559.29 GiB** 的已有负载，没有再次编码或复制视频。小元数据采用私有复制，初始约 125.25 MiB，另有归一化、验收和发布绑定记录。普通 `du` 对单独目录显示的大小包含共享文件，不能把它直接理解成新增 PFS 占用。

## 训练人口

| 来源 | 冻结小时 | 本轮监督 | 状态 |
|---|---:|---|---|
| FreeTacMan | 5.598185 | 视频、动作、RGB 触觉 | 可训练 |
| MotionTrans | 1.653403 | 视频、动作 | 可训练 |
| RDP_Bimanual | 0.072384 | 视频、动作、RGB 触觉 | 可训练 |
| RH20TCfg5Franka | 3.282083 | 视频、动作 | 可训练 |
| RH20TCfg6ATIAxia | 4.072944 | 视频、动作 | 可训练 |
| RH20TCfg7Tactile | 2.151528 | 视频、动作 | 可训练 |
| VLA_touch | 0.245972 | 视频、动作、RGB 触觉 | 可训练 |
| ViTaMIn | 0.274120 | 视频、动作、RGB 触觉 | 可训练 |
| exUMI | 0.584681 | 视频、动作、RGB 触觉 | 可训练 |
| Unit_Bimanual | 2.079278 | 保留原始数据 | 仅归档 |
| VisuoTactile_D-WHEEL | 3.777222 | 保留原始数据 | 仅归档 |
| VisuoTactile_QINGLOONG | 2.123074 | 保留原始数据 | 仅归档 |

**可训练人口为 533 个分片、17.935300925925933 小时**，其中 RGB 触觉来源约 6.775343 小时，其余约 11.159958 小时按 VA 使用。所有可训练来源的 train split 都覆盖其冻结 population。

其余 7.979574 小时的来源仅有关节数据，现有发布 profile 明确缺少已验证的机器人模型、关节约定和 FK 到公共 EEF 的定义。因此未伪造末端动作，也未将这些机器人来源改成 ego 训练。它们已经冻结保存，`reader_config.yaml` 明确 `enabled: false`。FTP 中的 Sharpa、尚未出现在冻结快照中的 RDP/Unit，以及之后新增的分片均未纳入。

## 数据与模型合同

- 使用已发布的真实 EEF pose／开度定义，生成按来源共享的 80 维 min-max 统计；统计只读取冻结 train 行中有效的原始数据，分位数为精确统计。
- 旋转 6D 和无效槽位保持恒等归一化。未知长度单位保留来源原生单位，并以来源统计归一化；不把它声明为已验证的米制数据。
- 每个分片的 `meta/action_contract.json` 绑定相邻来源根目录的 `normalization_stats.npy` 及其 SHA256。
- RGB 触觉历史输入为 224×224，未来标签为 128×128。RDP_Bimanual 根据已有区域元数据分别使用 GelSightMini/MCTac；其余来源也保留明确编码器身份。
- 数值力／力矩信号仍完整归档，当前 VA 来源没有把它们伪装成标量压力空间图。
- 元数据为私有副本。原始 `_PFS_PUBLICATION.json` 保存到根目录 `provenance/<Source>/<Bucket>/`，没有修改转换任务的原始发布。

## 验收与可重复执行

实际生产 reader 初始化并解码了全部 **533 个可训练分片**，总计 **587 个分片／边界窗口**；另有 **9 个来源级 registry 读取窗口**。场景图像尺寸 384×320，动作 `[32,80]`、proprio `[1,80]`，RGB 的区域、编码器和真实 collator 检查通过。

来源 `publication.json` 绑定冻结快照、原发布清单、原发布 receipt、归一化、验收结果以及全部分片 meta 文件（包括 info、episode 索引、动作合同、来源定义和 RGB 触觉定义）。训练入口检查这些哈希及准确的分片人口，元数据或人口改变会拒绝训练。

分片验收缓存另外绑定完整 meta 文件集合及解析后的触觉模型配置。恢复时，只要任一元数据或触觉配置改变，旧缓存就不能直接复用。本轮原有缓存的一次性身份补录验证了已经封存的完整发布哈希，并验证模型配置与验收时的 `cc581f293d3fb94930ec5af67181a4c59b78cfca` 基线完全一致；保留每个旧样本结果的摘要和迁移证据，没有重新解释原数据。普通恢复不自动推定缺失的历史缓存身份。

大文件完整性沿用已完成不可变发布中的 SHA256，本次另核对 receipt/info 哈希、payload 大小及硬链接 inode，**未重新全量读取 559 GiB 媒体来计算哈希**。实际视频解码采用上述逐分片抽样，不代表穷举所有视频帧。源级格式验收与完整模型分布式训练验收仍为不同步骤。

完整证据位于冻结目录：

```
snapshot.json
publication_inventory.json
freeze_complete.json
freeze_verification.json
training_catalog.json
<Source>/normalization_complete.json
<Source>/normalization_stats.npy
<Source>/reader_validation.json
<Source>/validation_cache_migration.json
<Source>/publication.json
<Source>/reader_config.yaml
```

在服务器上重复执行时，脚本只接受相同快照和自有输出；已完成验收可以复用，元数据更新的中断可以恢复。每个来源使用独立进程，释放 Arrow/PyTorch 缓存，适配 dev-cpu 的有限内存。不要并发运行两个发布器操作同一个输出目录。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/mnt/pfs/v89atp/workspace/envs/openwam/bin/python \
  /mnt/pfs/v89atp/workspace/OpenWAM-touchwam-training/scripts/data/ftp1/freeze_training_subset.py \
  --phase all \
  --snapshot /mnt/pfs/v89atp/workspace/setup/five_source_format_audit_20261029/ftp_snapshot.json \
  --output /mnt/pfs/v89atp/workspace/dataset/ftp1_frozen25h_v1_20261029
```

冻结根目录是包含归档来源的 catalog，不直接当作一个训练来源使用。训练配方只读取九个 `training_ready: true` 来源的独立发布目录。最终训练时长比例由外层混合配方控制，避免不同原生 FPS 按窗口数量放大采样权重。
