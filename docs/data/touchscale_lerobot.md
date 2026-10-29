# TouchScale 人类视频与触觉

源版本 `2077AIDataFoundation/TouchScale@4bc32c64c48d3711e99e1483a2520bdceda50c85`，
BOS 挂载 `/dataset/datasets/raw/huggingface/2077AIDataFoundation/TouchScale/<revision>`。
当前元数据有 15,324 条轨迹、约 100 小时；转换全部 430 个 WebDataset tar 分片。
不创建机器人动作或本体监督，按 `source_kind=ego`、`video_tactile` 接入。

目标：`/mnt/pfs/v89atp/workspace/dataset/touchscale_lerobot_v3_full_v1_20261029/`。
训练配置 `train_touchwam_ftp1` 已声明该源；只有全量发布完成才允许训练。
用 `TOUCHWAM_TOUCHSCALE_ROOT` 覆盖其 `buckets/` 路径。

每个源 tar 对应一个 LeRobot v3 bucket，包含标准 `meta/info.json`、tasks/episodes Parquet、
逐帧数据 Parquet 与三路 H.264 视频。Parquet 保留 1,760 个原始法向力值和各区域有效性、
语言、视频来源帧号与物理时间。来源 H5 保存原始异步力/剪切分量、IMU 和标定；
头部深度视频留在原始 BOS，不作为本次模型输入。正式训练所需文件均驻留 PFS。

以 H5 共同区间为界，生成 30 Hz episode 相对时钟，视频始终选择不晚于目标时刻的帧。
最大相机样本年龄 100 ms，超龄/缺失使用显式有效性 mask；不伪造有效黑图。
连续源帧区间复用原始视频文件，非连续选择用 H.264 CRF 0 无损编码；所有源视频完整解码验证。
时钟网格可能舍去原始片段最后不足 1/30 秒的部分，实际发布小时数按输出帧统计。

触觉历史保持双手独立的原生采样时间，以 causal sample-and-hold 选择，最大年龄 50 ms。
未来标签选择目标时刻附近 25 ms 内的真实未来测量；不跨 episode 或 segment。
30 个区域对应 `configs/tactile_layouts/touchscale_full_hand.json` 的原生解剖映射。
矩阵不缩成 4×4，也不伪装成光学图像；每个原始 pad 放在 16×10 画布左上角，其余区域由 ROI mask 排除。
双手有效 taxel 共 1,760 个；历史 `[4,30,1,16,10]` 进入独立随机初始化的 `TouchScaleForce` Fourier/CNN。
未来统一接口为 `[8,30,1,32,32]`，仅对已测 ROI 监督。

原始供应商输出按 `N_assumed_vendor_calibrated` 记录。模型归一化为
`max(force,0)/(max(force,0)+1N)`：固定尺度、没有逐帧 min-max，也不会混同于 Sharpa deform 的标定。
负值偏置在模型输入中按零压缩接触处理；原始负值和剪切值保留可追溯。

转换入口 `scripts/data/touchscale/convert.py`：先 `plan`，再按互斥 shard 分工 `worker`，最后 `finalize`。
每条 episode 和每个 bucket 有完成回执，bucket 原子发布；中断后只恢复未完成内容。
在发布根创建 `STOP` 可请求停止，已有成品和回执保留。不同作业不得并发处理同一分片。
全量发布绑定转换计划、每个 bucket 元数据、触觉 manifest 和全文件 SHA256 验证清单。

数据原始授权 CC BY-NC 4.0 与引用保存在发布根 `provenance/README.md`。
