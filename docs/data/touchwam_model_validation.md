# TouchWAM 多模态训练约束与验证

## 训练图与缺测

区域版支持 V、VA、VT、VAT 四条实际计算路径。V/VA 使用 `N=0`，不运行触觉编码器或触觉流；V/VT 不运行动作流，也不注入伪造 proprio。历史观测、未来查询、未来监督分别使用自己的 mask，未来标签缺失不改变查询是否存在。

正式多卡训练使用 `GlobalModalitySampler`：同一全局 microbatch 的所有 rank 必须使用同一种模态。不同 microbatch 可以切换模态，同一梯度累积窗口也可以包含 V 和 VAT。

T 模态要求每个 rank 保留该样本实际拥有的区域布局，区域数量可以不同。没有观测或未来标签时使用 false mask；没有触觉布局的数据使用 V/VA。进入模型前会共同检查模态和布局，防止一部分 rank 已进入反向通信、另一部分 rank 才报错。ZeRO-2 不能把跨 rank 的 `N=0`/`N>0` 图不一致交给 `ignore_unused_parameters` 自动处理。

## 无触觉步骤不更新触觉流

ZeRO-2 会把 unused gradients 补零并展平。普通 AdamW 仍可能对零梯度参数执行历史动量更新和 weight decay，因此仅“跳过触觉 forward”不足以保证触觉参数不变。

区域版使用 `ModalityAwareAdamW`，分成视频、动作、proprio、触觉四组。视频和动作组保留原有 LR override，proprio 和触觉仍使用 base LR。

- 每次 backward 前，根据全局模态激活本次使用的流。
- 梯度累积窗口中采用 OR：只要任何 microbatch 使用过某流，该流在窗口结束时正常更新。
- 整个窗口未使用的流，在底层 AdamW 更新时将整组 `grad` 设置为 `None`。参数、`exp_avg`、`exp_avg_sq`、Adam step 都保持不变。
- 每个优化器边界清空激活状态，包括 overflow 导致跳过更新的边界。
- 分组身份和激活状态属于 optimizer state dict，可以随本版 checkpoint 保存恢复。旧版单组优化器状态不能直接恢复到新分组，应从模型权重开始新训练。

当前配方要求 ZeRO-2、`offload_optimizer_device: none`。CPU offload 可能触发 Accelerate 替换优化器，不能悄悄丢弃上述语义。

## 已完成验证与边界

`scripts/validation/touchwam_multisource_gpu.py` 使用真实 Wan 和 ActionDiT 的小规模网络：440,308 参数、80 维动作、32×32 触觉图、4 帧历史、8 帧未来。它不是完整 Alpha 模型的显存或吞吐基准。

在 PyTorch 2.7.1+cu128、DeepSpeed 0.18.9、Accelerate 1.14.0 上完成：

- 单卡 BF16 CUDA forward/backward，V/VA/VT/VAT 切换。
- 一进程 ZeRO-2/NCCL。
- 两进程 DDP，以及两进程 ZeRO-2，使用单张 RTX 4090 上的 Gloo 通信。每步复核 rank 间参数一致。
- rank 0 没有未来标签；rank 0 的历史、查询、标签全部缺测；两 rank 区域数量为 1 与 2。
- 正式 ZeRO-2 路径共同拒绝跨 rank 的真实 `N=0`/`N>0` 不一致，没有伪造触觉布局。直接调用模型的 DDP 路径通过该情形。
- 累积窗口含 V+VAT、VAT+V，以及后续 V+V；逐位验证未激活流的 FP32 master weights 和所有 Adam 状态保持不变。
- ZeRO-2 模型与优化器 checkpoint 保存、扰动、恢复的逐位核验。

小模型测试峰值分配显存约 27 MiB。上述两进程共享一张 GPU 的 Gloo 测试不能替代多 GPU NCCL 或 64 卡验收。

复现生产约束、梯度累积与恢复测试：

```bash
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29740 \
  -m scripts.validation.touchwam_multisource_gpu \
  --backend zero2 --shared-device --trainer-call-style \
  --accumulation-steps 2 --resume-roundtrip \
  --output /tmp/touchwam_gpu_zero2_resume.json
```

真正的两 GPU NCCL 测试应去掉 `--shared-device`。脚本会检查空闲显存，且默认限制每进程最多分配 1 GiB；不会驱逐现有 GPU 进程。

## 完整 Alpha 初始化审计

`scripts/validation/touchwam_alpha_init.py` 检查完整 OpenWAM Alpha checkpoint 的 key、shape，并逐张量执行实际 `load_state_dict(assign=True)` 和源/目标字节哈希比对。每验证一个张量立即释放映射，不同时驻留整个约 23.11 GiB 权重。

已验证 2,089 / 2,089 个预训练张量逐位一致，只有 367 个触觉张量是新增项，共 30,538,508 参数。预训练部分含冻结预处理模块，总计 12,406,754,092 参数。完整 checkpoint SHA256：

```
180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d
```

审计报告明确 `full_resident_model_forward: false`。这证明权重映射与原参数保留，不证明完整模型已在 16 GiB 内存的 dev-cpu 上完成驻留 forward。正式训练仍需足够 CPU/GPU 内存的节点。发布门禁与 Alpha 初始化验证相互独立，检查模型不应先要求数据发布完成。
