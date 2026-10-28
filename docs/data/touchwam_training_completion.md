# TouchWAM 训练实现修复与使用边界

后续更新：光学触觉现采用 [原始 RGB 编码与预测分支](../touchwam/optical_rgb.md)，不再要求 FTP 光学来源先重建 deform。下文记录 `4b6b4fa` 时的单通道基线；当前 RGB 来源还需发布 [RGB 映射合同](tactile_rgb_contract.md) 和已验证动作合同。已有 Sharpa deform 路线保留。

本次从 `25401f8` 开始，在 `codex/touchwam-training-complete` 独立分支修复，并合入后续数据说明提交至 `1d2d7dd`；本次代码改动的比较基线为 `1d2d7dd`。服务器独立工作区为 `/mnt/pfs/v89atp/workspace/OpenWAM-touchwam-training`。原 `OpenWAM-touchwam` 数据转换工作区保持独立，不重转已有 EgoDex、SharpaDex、AgiBot，不实现 BOS 训练预取。

## 已补齐的代码

- 机器人动作 reader 及 source_kind/expected_modalities：FTP 必须 VA/VAT；触觉预训练选择 VAT，缺动作禁止降级。合同见 [robot_action_contract.md](robot_action_contract.md)。
- checkpoint 保留值统一为正数 K 或 -1；0 及其他负值拒绝。预训练默认保留最近 3 组权重/完整恢复状态，避免 PFS 无限增长。
- 周期保存延后到实际优化器更新边界。max_steps 仍计 microstep，但在累积中途达到预算时完成当前更新；epoch 尾也视为更新边界。
- 正常结束按 save_full_states_for_resume 保存最终完整状态，恢复读取保存的 opt_step，不再按 microstep 简单整除猜测；检查布局与累积边界。
- 每步写 `training_metrics.jsonl`，即使不开 W&B/debug 也保留来源、模态、有效触觉区域/帧、loss、学习率与速度。
- V/VA/VT/VAT 生成接口；ego VT 无需机器人动作/proprio。旧 VA/VAT 默认接口保持兼容，VAT checkpoint 可选择机器人来源 normalizer。
- 独立 held-out 触觉生成评估：有效 mask 下 MAE/RMSE、zero/hold-last 基线、按区域/预测时刻指标及 PNG/NPZ。见 [评估文档](../touchwam/tactile_generation_evaluation.md)。
- preflight 使用实际 collator，核对来源身份、区域/传感器、ROI、时间与模态；它明确不代替完整训练验收。

## 启动与恢复

在服务器训练工作区中激活原 OpenWAM 环境，使用既有完整 Alpha 参数和 PFS 数据：

```bash
cd /mnt/pfs/v89atp/workspace/OpenWAM-touchwam-training
export PATH=/mnt/pfs/v89atp/workspace/envs/openwam/bin:$PATH
export PYTHON_BIN=/mnt/pfs/v89atp/workspace/envs/openwam/bin/python
export TOUCHWAM_ALPHA_AUDIT=/path/to/passed/full_alpha_audit.json
bash scripts/train_touchwam_pretrain.sh check
# check 通过且正式采样/资源配方确定后：
bash scripts/train_touchwam_pretrain.sh train
```

继续同一训练运行（完整状态目录应位于其原 config/统计包目录中）：

```bash
export TOUCHWAM_RESUME_DIR=/path/to/training/run
bash scripts/train_touchwam_pretrain.sh resume training.max_steps=NEW_TOTAL_MICROSTEP_BUDGET
```

resume 包含完整优化器状态；仅有 safetensors 的旧训练只能 warm-start。`save_steps=0/null` 表示不保存 checkpoint，但常规指标仍会记录。恢复不是原子跨机器逐步复现实验的承诺：真实多 worker 随机数据增强的完全确定性仍需单独控制。

## 数据与验收边界

当前默认来源范围为旧 EgoDex 约300h、原 SharpaDex 约240h、AgiBot约140h、FTP候选夹爪触觉约80h；MotionTrans与RH20T Cfg5/6不作为夹爪空间触觉来源启用。最终有效FTP小时取决于动作和触觉合同验收，不能用原始RGB小时替代。

本次检查时，现有 FTP 示例的来源定义仍为 `action_enabled: false`，没有经过验证的 canonical 动作合同；实验形变的监督发布也尚未通过。因此，完成 reader 不等于这些样例已经可以训练 VAT。数据发布方需要提供 [动作合同](robot_action_contract.md) 所列的 canonical 列、有效性 mask、动作／状态统计与语义证据，以及有效触觉发布，再运行 preflight。不能将缺失动作的 FTP 当作 ego V/VT 使用。

混采代码支持明确权重，但默认仍是 proportional 基础配置，尚未代替用户决定最终比例、更新步数或全尺寸 batch。新模型触觉参数/完整Alpha权重保持审计不因本次读取与训练循环修复改变。

源码回归、小模型训练器/生成验证和完整 Alpha 驻留多GPU训练是不同验收范围。本次不会启动正式长训练；应在数据发布完毕后，用最终设备完成真实样本的完整模型短训练、显存/吞吐和多GPU NCCL验收。

## 已执行的代码验证

- 合并后的 CPU 回归：366 passed、1 skipped（CUDA 隐藏）。使用 PyTorch CPU 回退，避开本机 oneDNN 不支持的 BF16 backward；不修改训练精度。
- 单卡 RTX 4090：真实 `Accelerator + DeepSpeed ZeRO-2 + OpenWAMTrainer.train`，NCCL world size 为 1。使用 440,308 参数的 tiny Wan／ActionDiT／区域触觉模型，覆盖四种模态、accumulation=2、8 个 microstep／4 次更新、周期保存延后及最终完整状态保留。
- 连续训练与第 4 步中断恢复的模型、Adam 完整状态、scheduler SHA256 全部一致。该小模型峰值分配为 74.65 MiB，不能用于推算完整模型显存。
- 测试使用确定性小型 latent 替代 T5／VAE 和真实数据读取；完整 Alpha、真实多 worker 增强恢复、多 GPU 通信不在此次 GPU 验证范围。

单卡训练器复验命令（输出目录必须为空）：

```bash
PYTHONPATH=. python -m scripts.validation.touchwam_trainer_smoke \
  --phase-timeout-seconds 60 --output /tmp/touchwam-trainer-smoke-fresh
```
