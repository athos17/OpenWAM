# 公共触觉区域：touchwam_regions_67_v2

2026-10-28 更新。训练代码位于 dev-cpu 的 `/mnt/pfs/v89atp/workspace/OpenWAM-touchwam`。
**一个公共区域对应一份触觉图；不再使用 surface_id 或实例编码。**
同一区域的多个原生传感器由数据适配层按固定布局合成，再统一输入和预测。
权威定义：`openwam/tactile/registry.py`；当前完整表：[region_registry_v2.json](region_registry_v2.json)。
旧版 [region_registry_v1.json](region_registry_v1.json) 仅用于旧 checkpoint 审计，不是当前布局。

## 当前 67 类

| ID | 类别 | 含义 |
|---|---|---|
| 0–4 / 5–9 | 右 / 左五指 `.tip` | 保留原 ID |
| 10 / 11 | 右 / 左整掌 | 未细分掌面 |
| 12–34 / 35–57 | 右 / 左手、腕、前臂 | 保留原 ID |
| 58 / 59 / 60 | `right.gripper.jaw_a.surface` / `right.gripper.jaw_b.surface` / `right.gripper.base` | 右臂或单臂夹爪 |
| 61 / 62 / 63 | `generic.surface` / `generic.tool` / `generic.body` | 保留原通用 ID |
| 64 / 65 / 66 | `left.gripper.jaw_a.surface` / `left.gripper.jaw_b.surface` / `left.gripper.base` | 左臂夹爪 |

67 = 原手部及腕前臂 58 + 左右夹爪各 3 + 通用区域 3。
夹爪四个接触面分别为右 A=58、右 B=59、左 A=64、左 B=65；基座也按左右定义。
只为实际使用的区域生成 token，不为未使用 ID 创建空图。

## 单臂与双臂的映射规则

- 双臂：按真实连接的机械臂映射左右，各自 A/B 面使用稳定的原生面定义。
- 单臂：统一使用公共右臂夹爪 A/B（及基座）；这是跨数据集的规范映射。
- UniVTAC 的 `left_tactile` 是同一个夹爪的 A 面 → 58；`right_tactile` 是 B 面 → 59。
  它们不代表两条机械臂，不需要新增无侧别身份。
- 单臂映射右侧只规范区域身份，不自动镜像图像、改写力方向或重排动作。
  数据 reader 仍需说明坐标轴、A/B 方向和动作映射；动作侧别应与此约定一致。
- 同一样本的有效区域 ID 必须唯一；重复区域会在 layout、collator、forward、历史和 sidecar 处报错。
- `surface_id` / `tactile_surface_ids` 已从运行接口移除。旧字段不被静默忽略；旧数据需显式转换。

| 右侧 ID | 后缀 | 分组／边界 |
|---|---|---|
| 12 | `thumb.proximal` | 拇指近端段；不虚构拇指的第三节指骨 |
| 13 | `thumb.base_thenar` | 拇指基部／大鱼际复合区域，属于掌面开关组 |
| 14–16 | `index.middle / proximal / base` | 食指中段、近段、根部 |
| 17–19 | `middle.middle / proximal / base` | 中指中段、近段、根部 |
| 20–22 | `ring.middle / proximal / base` | 无名指中段、近段、根部 |
| 23–25 | `pinky.middle / proximal / base` | 小指中段、近段、根部 |
| 26–30 | `palm.thenar / hypothenar / central / distal / proximal` | 大鱼际、小鱼际、中央、远端、近端掌面 |
| 31 | `hand.dorsal` | 手背 |
| 32 | `hand.other` | 手部无法可靠细分的表面，必须说明映射理由 |
| 33、34 | `wrist / forearm` | 腕、前臂 |

`.tip` 是粗粒度末端／远端触觉区域，允许关联 Sharpa 指尖与手套 distal pad；
并不声称它们覆盖同一块皮肤或像素一一对应。保留 native_name、native_shape 和
传感器几何资料。掌面整体和掌面子区是覆盖粒度的选择，不要将同一测量复制成多个
监督目标。一个真实传感器跨越多个区域时，优先选择可信的复合／整体／generic
身份并保留覆盖说明；没有原始几何依据时不切割、不重复计数。

公共区域覆盖约定的粗粒度身份，不是穷举所有未来硬件。generic 入口使新拓扑无需
改表，但只提供通用身份；精细几何和新的物理表示仍需要实际适配和训练。
这张表不是每帧三维手模型，也不承担力、灰度、压入深度之间的物理标定。

## 同一区域的多个传感器合成

布局 schema 为 `touchwam_region_layout_v2`。每个 `surfaces` 条目是一份最终区域图，
身份只有 `region`；该字段的名称沿用 surface（物理表面），不表示实例编号。
一个区域可声明多个 `tiles`，记录原生 key、行列数、在合成图中的行列起点和旋转。
`native_shape` 是最终区域图的固定原生尺寸。无 tiles 时直接读取 native_key 对应的单图。

```json
{
  "schema": "touchwam_region_layout_v2",
  "registry": "touchwam_regions_67_v2",
  "name": "single_arm_composed_example",
  "surfaces": [{
    "native_key": "composed/jaw_a",
    "region": "right.gripper.jaw_a.surface",
    "sensor": "example_deform_sensor",
    "representation": "sharpa_deform_gray_v1",
    "units": "uint8_intensity",
    "native_shape": [240, 480],
    "tiles": [
      {"native_key": "jaw_a/pad_1", "native_shape": [240, 240], "origin": [0, 0]},
      {"native_key": "jaw_a/pad_2", "native_shape": [240, 240], "origin": [0, 240]}
    ]
  }]
}
```

此示例是合成布局，不是已采集的数据源；使用自定义来源前需注册模型 sensor_names。
A 与 B 面仍是两个区域；合成发生在同一面的多个传感器之间。

```python
from openwam.tactile.layout import SurfaceLayout
layout = SurfaceLayout.load('configs/tactile_layouts/single_arm_composed_deform_example.json')
frame = layout.compose(packets_by_native_key, observed=observed_by_key, dtype='uint8')
# frame['maps'][region]、frame['valid'][region]、frame['spatial_valid'][region]
# reader 再按 codec 统一尺寸和归一化，将 ROI 传给 tactile_spatial_valid。
```

已实现 `composition.py` 中的固定二维拼接、90° 整数旋转、空白 ROI、输入校验。
不重叠放置，不对两个测量直接求和、平均、逐帧 min-max；数值、负值和有效零值都保留。
同一原生 key 不能重复用于不同区域的监督。多个 tile 先由 reader 对齐时间与物理单位。

目前采用保守有效性：任一组成传感器缺包、过期或非有限值，该区域整帧 observed=false；
图置零仅用于存储，不能作为真实零标签。固定 ROI 在丢包时不变化，区域布局也不消失。
reader 缩放合成图时要同步 ROI，避免有效空白混淆；当前通用合成函数不负责源 codec、
实际几何标定、异步时钟、力值/深度归一化或跨帧插值。未知几何需明确固定布局，不能宣称物理配准。

## 现有数据布局与模型

- Sharpa：十个指尖，10 个区域，240×240 原图；当前 deform codec 32×32、patch16。
- TouchScale：30 个区域、1760 taxels；fingertips 为 10，without_palm 为 26。
- UniVTAC：单臂右夹爪 A/B 两个区域，240×320 毫米压入深度。
- 双臂夹爪：4 个接触区域；已提供 `dual_gripper_deform_example.json` 结构示例。

当前 Sharpa 十指仍为每帧 40 token、历史160、未来320。同一个区域合成多个传感器后，
其 token 数由最终图尺寸和 patch 大小决定；不会因原生传感器个数增加实例 token。
高分辨率合成图缩小到32×32会损失细节，需要实际训练比较，不声称无损。

token = 内容投影 + 区域 embedding + 来源 embedding + 局部 x/y/物理时间投影 + ROI 投影。
不再有实例编码。batch 仅补齐到最大实际区域数，padding仍占张量空间。
当前默认触觉模块参数为30,534,924；67类比64类多768个参数（BF16约1.5KiB）。

掌面组仍由 registry groups 决定，包括 thumb.base_thenar；palm_mode=off 移除掌面条件、
query和loss。夹爪不属于掌面组。

**能力边界：**公共定义、单区域身份、布局合成工具、deform前向/loss/生成/部署已实现。
TouchScale力值与UniVTAC深度的完整reader/codec、缺动作ego训练及完整无触觉路由仍待实现。
geometry_ref仅保存资料引用，不自动建立三维手或进行跨域标定。尚未训练本版模型。

## 版本与权重迁移

旧12类配置仍可加载原regional checkpoint。旧64类不作为当前配置接受；
显式warm-start可从12/64类迁移到67类，其余tactile_spec必须完全一致，旧64类需验证版本与哈希。

- 12→67：复用原12行及其余参数，初始化55行。
- 64→67：复用原64行及其余参数，新增左夹爪64–66行；原无侧别58–60行明确映射到右夹爪。
- 保留已训练gates，不恢复旧优化器。新参数由project.seed保证各rank一致。
- 使用过非零surface_id的旧模型不保证预测一致：实例编码已经移除，数据需重新合成并开启新微调run。
- 固定旧分支/提交659cbed与快照标签，保留旧注册表JSON，不静默覆盖旧checkpoint语义。

```text
training.initialize_touch_from_openwam=false
training.initialize_touch_from_legacy=false
training.initialize_touch_registry=true
training.finetune_ckpt_path=/path/to/old_regional/checkpoint.safetensors
training.resume_ckpt_path=null
```

原distance/contact模型迁移仍使用initialize_touch_from_legacy，与本迁移互斥。
