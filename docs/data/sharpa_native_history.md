# Sharpa 原始分辨率触觉历史

`train_touchwam_ftp1` 的 Sharpa 历史观测需要 224×224，未来标签仍为 32×32。
原始 deform 视频已有十个 240×240 指尖，组成 480×1200 两行五列拼图。
新 reader 按每条 episode 的原始视频偏移寻帧，灰度解码后逐指用 OpenCV
INTER_AREA 缩到 224×224，再除以 255；没有从低分辨率标签放大，也不重建压力。

## PFS 存储

默认新增发布目录：

```text
/mnt/pfs/v89atp/workspace/dataset/sharpadex_deform_history_v1_20261029/
├── publication.json
├── validation.json
├── README.md
└── partitions/<task>/<season>/observation.json
```

这是元数据发布，依赖两套已有 PFS 数据：

- `sharpadex_native_tcp_v1_20261026`：原始 deform 视频及原有动作、场景视频、统计、划分。
- `touchwam_pretrain_v1_20261028/sharpadex`：32×32 标签、时间戳和有效性。

不复制、重编码或修改这两套数据；训练期间必须保留。新目录不属于 FTP 转换的
临时工作目录，FTP 的发布后清理不会删除它。`TOUCHWAM_SHARPA_HISTORY_ROOT`
可覆盖新增目录位置，已有原始数据和标签的配置继续独立生效。

## 生成与验证

在训练仓库根目录，用同一 OpenWAM 环境执行：

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  scripts/data/native/publish_sharpa_history.py \
  --source /mnt/pfs/v89atp/workspace/dataset/sharpadex_native_tcp_v1_20261026 \
  --tactile /mnt/pfs/v89atp/workspace/dataset/touchwam_pretrain_v1_20261028/sharpadex \
  --output /mnt/pfs/v89atp/workspace/dataset/sharpadex_deform_history_v1_20261029
```

先验证现有发布身份，再核对全部 episode 的视频偏移、NPY 头、时钟、mask 和视频头。
每个分区从一条完整轨迹抽查开头、中间、末尾三个窗口。源目录保持只读；在相邻
`.building` 目录完成后原子发布。重复调用已完成目录仅核对发布，不覆盖。

高分辨率历史与旧标签共用物理时间选择和有效性规则。无效历史保持零且 mask=false，
真实的零接触仍然有效；不跨 episode，不用未来帧填历史。原始视频缺少被标为有效的
帧时直接报错。视频路径限制在原始 bucket 内，并核对发布时文件大小与 mtime。
这些检查不等于全视频 SHA256 或所有帧的视觉质检。

新增索引 SHA256 纳入数据合同和保存恢复校验。完整模型的训练/多 GPU 性能仍需
专门的短训验收；此次新增接口只负责提供正确的原始分辨率历史。
