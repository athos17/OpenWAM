# 固定公共触觉区域：touchwam_regions_64_v1

训练代码位于 dev-cpu 的 `/mnt/pfs/v89atp/workspace/OpenWAM-touchwam`。
本次把公共区域定义固定为 **64 类**。新数据通过 layout 映射到这些身份，
同一身份内的多个原生 pad 使用独立表面实例，不再为每个数据集追加公共区域。
权威定义：`openwam/tactile/registry.py`；完整 ID、名称及分组：
[`region_registry_v1.json`](region_registry_v1.json)。

## 固定的 64 类

| ID | 类别 | 含义 |
|---|---|---|
| 0–4 | 右手五指 `.tip` | thumb / index / middle / ring / pinky |
| 5–9 | 左手五指 `.tip` | 同上，保持旧 ID |
| 10、11 | `right.palm`、`left.palm` | 原有整掌／未细分掌面 |
| 12–34 | 右手及腕前臂细分 | 见下面的单侧表 |
| 35–57 | 左手及腕前臂细分 | 与右侧逐项对应，ID 加 23 |
| 58、59 | `gripper.jaw_a.surface`、`gripper.jaw_b.surface` | 同一夹爪的两个独立接触面 |
| 60 | `gripper.base` | 夹爪基座接触面 |
| 61 | `generic.surface` | 无可靠解剖映射的接触面，包括其他夹爪拓扑 |
| 62 | `generic.tool` | 工具接触表面 |
| 63 | `generic.body` | 其他身体／机器人机体接触表面 |

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

64 类覆盖约定的粗粒度身份，不是穷举所有未来硬件。generic 入口使新拓扑无需
改表，但只提供通用身份；精细几何和新的物理表示仍需要实际适配和训练。
这张表不是每帧三维手模型，也不承担力、灰度、压入深度之间的物理标定。

## 公共区域与表面实例

模型输入的区域轴现在表示实际参与计算的 **表面**。每个表面由以下键确定：

```text
(region_id, surface_id)
```

- `region_id`：固定 0–63 的公共身份。
- `surface_id`：该区域内稳定的整数实例号，范围 0–65535，默认 0。
- 同一个样本中该二元组必须唯一；多个设备／pad 共用区域时分配不同实例号。
- 实例号从 layout 读取；跨 episode、帧、区域筛选和顺序变化保持不变，不能按
  当前可见 pad 重新编号。它是身份，不是强度、接触状态或时间步。
- `instance`、`native_key` 在 layout 中保留设备名称和原生字段；native_key 唯一。
  整数 surface_id 由 layout 作者明确分配，不会对字符串做不稳定的 hash。
- 表面实例使用固定正余弦编码加入历史和未来 token，无新增可训练实例 embedding
  表。实例 0 的编码严格为零，维持旧模型行为。不同布局编号没有解剖等价含义。

双夹爪示例：

| 物理表面 | region_id | surface_id | instance |
|---|---|---|---|
| 左臂夹爪 A 面 | 58 | 0 | left_gripper |
| 左臂夹爪 B 面 | 59 | 0 | left_gripper |
| 右臂夹爪 A 面 | 58 | 1 | right_gripper |
| 右臂夹爪 B 面 | 59 | 1 | right_gripper |

十个指尖仍然每帧只产生 40 个空间 token；固定 64 个区域不会自动分配 64 份图。
batch 只补齐到最大实际表面数，补齐部分仍占计算，尚未实现完整 packing。
同一表面内部仍保留多个空间 patch。新增 52 行区域 embedding 在 width=256 时
增加 13,312 个参数（BF16 约 26 KiB），不增加额外触觉 Transformer。

## 数据接入配置

提供三份布局定义，均可用 `SurfaceLayout.load(path)` 校验：

- `configs/tactile_layouts/sharpa_ten_fingertips.json`：10 个 240×240 灰度图。
- `configs/tactile_layouts/touchscale_full_hand.json`：30 个原生 pad、1760 taxels。
- `configs/tactile_layouts/univtac_isaac51_gripper.json`：两个 240×320 压入深度图。

TouchScale 保留所有实际存在的指节和掌面 pad，不补不存在的小指近端 pad。
原生 `thumb_middle` 映射到拇指近端粗粒度区域，名称保留为 native_name；这不是
断言其几何边界与其他设备吻合。pad 2 保留 `thumb.base_thenar` 复合语义，pad 18
映射 `palm.hypothenar`。`select('fingertips')` 返回 10 个 pad，
`select('without_palm')` 返回 26 个 pad；二者不是同一筛选。

```python
from openwam.tactile.layout import SurfaceLayout
layout = SurfaceLayout.load('configs/tactile_layouts/touchscale_full_hand.json')
identity = layout.identity_fields()  # region_names / surface_ids / sensor_names
```

`identity_fields()` 仅返回身份信息，不读取原始数据、不生成 mask、不转换数值。
`representation` 和 `units` 明确记录物理含义。TouchScale 的 N 单位仍沿用数据的
厂商标定假设，不据此重复标定或删除负值。UniVTAC 的 mm 压入深度也不是压力。
`geometry_ref` 可指向 reader 管理的原生局部坐标／轴／ROI 资料；此接口仅保存引用，
当前 deform 网络仍使用自身 patch 局部网格，不会自动读取这些坐标或做三维配准。

现有 deform sidecar manifest 可增加 `surface_ids` 数组；省略时全部为 0。
在线 `DeformHistory` 可指定 `surface_ids` 和唯一 `surface_keys`，后者用于接收
原生图像字典。输出携带整数实例号，部署返回同样的 `tactile_surface_ids`，
显示端据 layout 找回设备和原生 pad 名称。

新数据的接入步骤：

1. 选择可信的公共区域，确定每个原生表面的稳定实例号；无法映射时显式使用
   hand.other 或 generic，并填写 mapping_note，未知拼写不会自动落入兜底槽位。
2. 记录原生字段、形状、设备、物理量、单位、覆盖说明及可用几何资料。
3. reader 负责数值 codec、时间对齐、观测有效性、空间 ROI 和未来标签。
4. 把身份字段与实际观测一起送入模型；缺失传感器、丢包和真实零值仍分别处理。

**实现边界：** 公共定义、三个布局配置、重复表面身份在 deform 前向／loss／生成／
部署中的传递已实现。TouchScale 法向力 reader/codec、UniVTAC 深度 reader/codec、
异步时间对齐、无动作／完全无触觉数据路由尚未实现。布局文件不宣称这些数据已可
直接训练。模型仍拒绝将其他物理量伪装成 deform。

## 掌面开关

掌面组由注册表的 `groups` 显式定义，不再依赖字符串是否包含 `.palm`。
`thumb.base_thenar` 属于掌面组；`palm_mode=off` 与 palm dropout 同时移除它的
历史、未来请求及 loss。夹爪接触面不属于掌面组。

## 版本及旧权重

- 固定名字、ID、分组与版本 `touchwam_regions_64_v1`；JSON 和测试锁定内容哈希。
- 原有 12 区域配置仍可加载原 regional checkpoint，未强制改其权重。
- 新配置默认 64 类。旧 12 类权重不能直接当作 64 类严格加载或精确 resume。
- 显式迁移仅扩区域 embedding：原 12 行及其余权重完全复用，52 行新增权重由
  `project.seed` 确定。不会重置已训练的桥接 gates，不恢复旧优化器状态。
- 迁移要求其余 tactile_spec、权重键和形状一致；改变 codec、顺序、帧率等会拒绝。

从原 12 区域 checkpoint 开启新微调 run：

```text
training.initialize_touch_from_openwam=false
training.initialize_touch_from_legacy=false
training.initialize_touch_registry=true
training.finetune_ckpt_path=/path/to/old_regional/checkpoint.safetensors
training.resume_ckpt_path=null
```

其余模型设置需与原 checkpoint 一致；新 `touchwam_regions` 配置提供完整 64 类。
普通 64 类微调／resume 应把 initialize_touch_registry 设为 false。
不允许在本版本随意重排或改写 region_names；新传感器来源和新 codec 的引入仍需
独立版本与 checkpoint 管理，冻结公共区域不等于所有模型接口永远无需调整。
