# SharpaDex 十指触觉接入

本轮只补充触觉缓存，复用上一轮的 LeRobot v3 视觉、动作、归一化统计和 train/val 划分。
训练数据直接放在 PFS；不需要 BOS 训练读取，也不重新转换 EgoDex 或 AgiBot。

## 路径与数据合同

| 内容 | dev-cpu / 资源池共享路径 |
|---|---|
| 原始处理数据 | `/mnt/pfs/v89atp/workspace/dataset/sharpadex_native_tcp_v1_20261026` |
| 原始 reader 入口 | 上述目录的 `buckets/` |
| 新增触觉缓存 | `/mnt/pfs/v89atp/workspace/dataset/touchwam_pretrain_v1_20261028/sharpadex` |
| 验收与资源池任务记录 | `/mnt/pfs/v89atp/workspace/setup/sharpa_tactile_ready_20261028` |
| 服务器训练代码 | `/mnt/pfs/v89atp/workspace/OpenWAM-touchwam` |

原库 495 个分区、28,993 条 episode、28,996,521 帧，合计约 268.49 h，包含验证集。
训练划分 26,112 条 episode、26,150,745 帧，约 242.14 h；不能把全库时长当成训练时长。
触觉图数组全库约 276.53 GiB，另有 mask、时间戳和清单。视觉与动作文件复用原文件。

`observation.images.tactile_deform` 是 480 高 × 1200 宽的拼图，包含十个 240×240 指尖面：

```
上排：左拇指尖  左食指尖  左中指尖  左无名指尖  左小指尖
下排：右拇指尖  右食指尖  右中指尖  右无名指尖  右小指尖
```

每面保留原图坐标方向，不翻转、不镜像，按面积缩小到 32×32。按视频声明的范围
转换亮度后存 uint8，reader 除以 255，得到 `tactile_intensity_v1` 的 0～1 相对形变
响应。该数值不是 N、Pa 或毫米，不能视为经过标定的压力。禁止逐帧 min-max。

已解码的全黑图是有效零响应；缺帧或解码失败才是 `valid=false`。
如果传感器掉线本身被原始视频编码成普通黑帧，现有原始字段无法进一步区分。

每个 episode 在 `tactile/` 中保存：

| 数组 | 形状 / 类型 | 含义 |
|---|---|---|
| `maps.npy` | `[T,10,32,32]` uint8 | 十指形变强度 |
| `valid.npy` | `[T,10]` bool | 每时刻每指观测是否存在 |
| `timestamps.npy` | `[T]` float64 | episode 内相对秒数 |
| `manifest.json` | JSON，分区级 | episode、区域、来源、转换版本和原视频偏移 |

区域身份按公共 registry 定义，无 `surface_id`。本数据没有掌面。
数据按十指拼图顺序存储；模型 collate 保留该顺序，把名称映射成公共区域 ID，
图、mask 和来源身份保持逐项对应，不要求所有来源使用相同的数组顺序。

## 训练读取

`SharpaNativeIntensityDataset` 在原生 Sharpa TCP reader 上增加外部触觉读取。
原 62 维动作／状态继续映射到 OpenWAM 80 槽位，其他槽位由 mask 排除。

以当前场景帧为 t=0：

- 输入触觉：`[-0.1, -2/30, -1/30, 0]` 秒，形状 `[4,10,1,32,32]`。
- 预测触觉：`[0.1,0.2,…,0.8]` 秒，形状 `[8,10,1,32,32]`。
- 轨迹开头缺少的历史、尾部不足的未来目标分别屏蔽；不跨 episode 取数据。
- 每指每时刻 4 个 patch token。十指历史共 160 个 token，未来共 320 个 token。
- 正常样本使用 V+A+T；尾帧没有未来触觉时仍可保留真实动作监督，不伪造未来触觉。

现有 `configs/dataloader/touchwam_pretrain.yaml` 已配置原始 `dataset_dir` 和新增
`tactile_root`，无需改动原数据。完整多源训练仍要求其他启用来源各自通过发布检查。
本次没有启动正式训练。

## 完成检查与复验

训练入口核对外部 catalog 的 hash、原数据发布 hash、分区集合、每分区 info、
episode 表和实际 reader 使用的触觉 manifest。相同帧数不能代替来源身份一致。

完整验收分三层：

1. 全 episode 校验 NPY 形状、类型、字节长度、时间戳、有效性、原视频偏移和
   episode 身份。文件变化后不能复用旧的成功验收；失败会撤销 `training_ready`。
2. 每分区取首／中／末 episode，各取首／中／末帧，从原视频独立解码、逐指裁切，
   与缓存逐像素比对。覆盖全部分区，但属于抽帧复验，不是全视频逐帧重放。
3. 用真实 train/val 样本检查 reader、生产 `prepare_inputs`、区域身份、动作 mask，
   再用缩小的真实 Wan／ActionDiT／触觉模块执行前向反向。VAE 与文本编码采用替代
   输出，不能将这项检查称为完整 Alpha 模型运行或多 GPU 性能验收。

在资源池 CPU 任务中运行以下检查；源码和缓存参数分别指上面的两个根目录：

```bash
python scripts/data/native/validate_sharpa_all.py \
  --source "$SHARPA_SOURCE" --cache "$SHARPA_CACHE" \
  --output "$SHARPA_AUDIT" --workers 2
python scripts/data/native/validate_sharpa_all.py \
  --source "$SHARPA_SOURCE" --cache "$SHARPA_CACHE" \
  --output "$SHARPA_AUDIT" --aggregate
python scripts/data/native/catalog_sharpa_tactile.py \
  --source "$SHARPA_SOURCE" --cache "$SHARPA_CACHE"
```

续跑只处理缺失分区；多个 CPU 节点使用固定且互不重叠的 rank 分片，不依赖跨节点
flock。主节点须等所有 rank 写出本次 job 的完成回执再退出，避免资源池提前停止
其他节点；平台任务成功不能代替全库验收。提取器源码已冻结，改变它会改变转换
签名，不应随意修改后强行复用旧缓存。
