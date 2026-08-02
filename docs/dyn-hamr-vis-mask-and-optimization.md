# Dyn-HaMR 帧有效性标记与三阶段优化详解

本文档详细解释 Dyn-HaMR 中四个紧密相关的核心机制：(1) `vis_mask` 的构建逻辑及其在优化中的掩码作用；(2) 2D 关键点的插值方法（坐标与置信度的独立处理）；(3) 三阶段 L-BFGS 优化的完整流程，包括 obs_data 字段清单、优化设计思路；(4) `run_opt` 输出产物的完整目录结构、文件约定和生产/消费链路。

---

## 第一部分：vis_mask 与 pred_valid

### 1. vis_mask 的构建逻辑

#### 1.1 数据源头

Dyn-HaMR 的 `vis_mask` 构建完全基于**磁盘文件的存在性**——具体是 HaMeR Pipeline 输出的 ViTPose 2D 关键点 JSON 文件。

上游 HaMeR（`third-party/hamer/run.py`）对每帧每手独立推理，输出保存到：
```
{tid:03d}/{frame_name}_mano.json       ← MANO 参数
{tid:03d}/{frame_name}_keypoints.json  ← ViTPose 2D 关键点
```

#### 1.2 阶段 A：原始 bool 数组（`MultiPeopleDataset.__init__`）

代码：`dyn-hamr/data/dataset.py:157-169`

```python
self.track_vis_masks = []
for pred_dir in self.track_dirs:
    kp_paths = [f"{pred_dir}/{x}_keypoints.json" for x in self.img_names]
    has_kp = [os.path.isfile(x) for x in kp_paths]
    vis_mask = np.array(has_kp)       # True = 文件存在, False = 不存在
    self.track_vis_masks.append(vis_mask)
```

这是全 shot 长度的 **bool 数组**，每个元素仅表示关键点 JSON 文件是否存在于磁盘。

#### 1.3 阶段 B：三值化（`get_ternary_mask()`）

代码：`dyn-hamr/data/dataset.py:402-411`（仅 10 行）

```python
def get_ternary_mask(vis_mask):
    vis_mask = torch.as_tensor(vis_mask)
    vis_idcs = torch.where(vis_mask)[0]                    # 所有 True 的索引
    track_s, track_e = min(vis_idcs), max(vis_idcs) + 1    # 轨道的 [首次出现, 最后出现)
    vis_mask = vis_mask.float()
    vis_mask[:track_s] = -1    # [0, track_s)   → 出镜
    vis_mask[track_e:] = -1    # [track_e, T)   → 出镜
    return vis_mask
    # 中间 [track_s, track_e) → 保持原值: True→1.0, False→0.0
```

**三值语义**：

| 值 | 含义 | 判定条件 | 优化行为 |
|----|------|---------|---------|
| **-1** | 手不在场景中（出镜） | `t < track_s` 或 `t ≥ track_e` | 完全排除，不参与任何 loss |
| **0** | 遮挡或检测失败 | `track_s ≤ t < track_e` 且文件不存在 | 参与 loss（与可见帧等同） |
| **1** | 可见（有检测） | `track_s ≤ t < track_e` 且文件存在 | 正常参与 loss |

**关键限制**：
- 依赖**纯文件系统检查**，无任何视觉语义验证
- **无法区分**物理遮挡和检测器漏检——两者合并为 vis_mask=0
- 核心假设：手一旦首次出现，直到最后一次出现之前，始终"应该"在画面中

#### 1.4 vis_mask = -1 帧的 MANO 参数状态

`vis_mask = -1` 的帧在数据加载时已被插值函数**排除**在插值范围外（`load_mano_preds()` 和 `load_keypoints_with_interp()` 仅插值 `[tmin, tmax]` 区间）。这些帧的 MANO 参数保持为**全零默认值**，但因为被后续掩码排除，不对优化产生任何影响。

#### 1.5 vis_mask = 0 帧的参数状态

`vis_mask = 0` 的帧（轨道跨度内的缺失帧）的 MANO 参数和 2D 关键点**已被 Slerp/线性插值填充**（`dyn-hamr/data/tools.py`）。这些插值结果作为该帧的"伪观测"参与优化。

#### 1.6 2D 关键点插值详解

代码：`dyn-hamr/data/tools.py:33-94` → `load_keypoints_with_interp()`，调用方：`dyn-hamr/data/dataset.py:230-240`

2D 关键点的插值在 `vis_mask` 三值化**之前**执行，且插值范围 `[tmin, tmax]` 直接决定了后续 `get_ternary_mask()` 中 `track_s` 和 `track_e` 的值。理解插值逻辑对理解 vis_mask 构建至关重要。

**完整数据流**：

```
每帧的 {frame}_keypoints.json
    │
    ▼
read_keypoints()  ← 文件不存在或 people=[] → 返回全零 (21, 3)
    │
    ▼
np.stack → (T, 21, 3)  joints2d_data
    │
    ▼
构建可见掩码:  文件存在 & 关键点不全为零 → True
    │  vis_idcs = 所有 True 的帧索引
    ▼
tmin = min(vis_idcs), tmax = max(vis_idcs) + 1
    │
    ▼
for j in range(21):          ← 每个关节独立处理
    ├─ x: interp1d(..., kind='linear') → 填充缺失帧的 x
    ├─ y: interp1d(..., kind='linear') → 填充缺失帧的 y
    └─ conf: 邻近可见帧同关节置信度均值 × 0.8
    │
    ▼
返回 joints2d_data (T, 21, 3)
    │
    ▼
dataset.py:235 → 所有帧所有关节的置信度强制设为 1.0
dataset.py:236-238 → 原始置信度 < MIN_KEYP_CONF(0.4) 的帧 → 整帧置零
```

##### 1.6.1 关节点坐标 (x, y)：逐关节独立线性插值

```python
for j in range(J):                    # J = 21 (OpenPose 手部关节数)
    x_interp = interp1d(vis_idcs,
        joints2d_data[vis_idcs, j, 0],   # 所有可见帧关节 j 的 x 坐标
        kind='linear', bounds_error=False)
    y_interp = interp1d(vis_idcs,
        joints2d_data[vis_idcs, j, 1],
        kind='linear', bounds_error=False)

    for t in times:                      # times = [tmin, tmax)
        if t not in vis_idcs:            # 仅对缺失帧插值
            joints2d_data[t, j, 0] = x_interp(t)
            joints2d_data[t, j, 1] = y_interp(t)
```

关键特点：

- **逐关节独立**：21 个关节各自有独立的 `interp1d` 插值器。关节 0（手腕）的轨迹和关节 15（拇指尖）的轨迹互不影响。这比使用统一的刚体运动模型更灵活——不同关节可能因局部遮挡有不同的可见模式。

- **`kind='linear'`**：纯线性插值。与 MANO 参数中旋转使用 **Slerp**（球面线性插值）形成对比——2D 关键点坐标在图像像素空间中，线性插值等价于匀速运动假设。

- **`bounds_error=False`**：对 `vis_idcs` 范围内的查询返回插值。超出边界的查询返回边界值（外推），但由于 `times = np.arange(tmin, tmax)` 严格限制在 `[min(vis_idcs), max(vis_idcs))` 范围内，**实际不会触发外推**。

- **与 MANO 插值一致的 `[tmin, tmax)` 范围**：轨道跨度之外的关键点保持全零，后续被 vis_mask=-1 排除。

##### 1.6.2 置信度 (confidence)：邻近均值 × 0.8 折扣

置信度的处理**不是真正的插值**——不使用 `interp1d` 的连续插值，而是离散的最近邻查询 + 打折扣：

```python
# 找到 t 之前的最近可见帧
prev_vis = vis_idcs[vis_idcs < t]
# 找到 t 之后的最近可见帧
next_vis = vis_idcs[vis_idcs > t]

if len(prev_vis) > 0 and len(next_vis) > 0:
    # 双侧包围：前后同关节置信度的平均 × 0.8
    conf_prev = joints2d_data[prev_vis[-1], j, 2]
    conf_next = joints2d_data[next_vis[0], j, 2]
    joints2d_data[t, j, 2] = (conf_prev + conf_next) / 2.0 * 0.8
elif len(prev_vis) > 0:
    # 只有前向（轨道尾部的缺失帧）：前邻 × 0.8
    joints2d_data[t, j, 2] = joints2d_data[prev_vis[-1], j, 2] * 0.8
elif len(next_vis) > 0:
    # 只有后向（轨道头部的缺失帧）：后邻 × 0.8
    joints2d_data[t, j, 2] = joints2d_data[next_vis[0], j, 2] * 0.8
```

关键特点：

- **逐关节计算**：每个关节的 conf 独立使用相邻帧**同关节**的置信度。例如遮挡帧的「食指第二关节」置信度使用前后可见帧中同一个「食指第二关节」的置信度。

- **0.8 折扣因子**：无论哪种情况，插值帧的置信度都被打了 8 折。这是一个启发式标记——「我们不太确定这个插值结果」。插值后的置信度范围在 `[0, 0.8]`（如果原始置信度最大为 1.0）。

- **单侧外推**：轨道首尾的缺失帧只能使用单侧邻帧的信息。

##### 1.6.3 插值后被覆写：dataset.py 的逻辑错误

`load_keypoints_with_interp()` 返回后，`dataset.py:235-240` 立即做了两步处理：

```python
# Step 1: 所有置信度强制设 1.0
joints2d_data[:, :, 2] = 1.0

# Step 2: 原始置信度 < 0.4 的帧 → 整帧关键点全部置零
joints2d_data[
    np.repeat(joints2d_data[:, :, [2]] < MIN_KEYP_CONF, 3, axis=2)
] = 0
```

**这是一个逻辑错误**：Step 1 先将所有帧的置信度覆写为 1.0，然后 Step 2 检查 `conf < 0.4`——但此时 conf 已经是 1.0，所以 `< 0.4` 的检查**始终为 False**，**没有任何帧会被过滤**。插值帧的 0.8× 折扣置信度也全部被覆写为 1.0。

预期行为（如果调换顺序）应是：
- 原始 ViTPose 置信度 < 0.4 的帧 → 整帧 (x, y, conf) 置零 → 被全零检测识别 → vis_mask 标记为不可见
- 其余帧 → 置信度统一为 1.0 → 优化中所有可见帧等权重

**实际行为**：所有帧（包括原始 ViTPose 低置信度的帧和插值的遮挡帧）的 conf 全部 = 1.0。这意味着 `Joints2DLoss` 中的 `joints2d_obs_conf` 项对所有帧恒为 1，**置信度加权机制完全失效**。

##### 1.6.4 与 MANO 参数插值的对比

| 维度 | 2D 关键点 (`load_keypoints_with_interp`) | MANO 参数 (`load_mano_preds`) |
|------|------------------------------------------|------------------------------|
| **坐标/平移** | `interp1d(kind='linear')` 逐关节线性 | `interp1d(kind='linear')` 线性 |
| **旋转** | 不适用 | `Slerp`（球面线性插值，保持旋转群结构） |
| **形状(betas)** | 不适用 | `interp1d(kind='linear')` 线性 |
| **插值粒度** | 逐关节独立（21 个独立插值器） | 全局方向 + 逐关节姿态（16 个 Slerp 插值器） |
| **插值范围** | `[tmin, tmax)` | `[tmin, tmax)` — 与 2D 关键点一致 |
| **缺失帧默认值** | 全零 `(21, 3)` | 全零 pose/rot/trans/betas |
| **可见帧判定** | 文件存在 + 非全零（双重检查） | 文件存在（单一检查） |
| **置信度/不确定性** | 邻近均值 × 0.8 折扣（被打折） | 无置信度概念 |
| **后处理** | conf 强制 1.0 + 低置信度清零（顺序有误） | 无后处理 |

##### 1.6.5 插值如何决定 vis_mask 的边界

`load_keypoints_with_interp()` 和 `load_mano_preds()` 的 `vis_idcs` 来自**独立**的文件存在性检查（各自读各自的 JSON 文件），但两者共享相同的文件命名约定和相同的 `[tmin, tmax)` 插值范围逻辑。`get_ternary_mask()` 使用**关键点**文件存在性（而非 MANO 文件）来确定 `track_s` 和 `track_e`：

```python
# dataset.py:157-169 — vis_mask 基于 keypoints.json 文件
kp_paths = [f"{pred_dir}/{x}_keypoints.json" for x in self.img_names]
has_kp = [os.path.isfile(x) for x in kp_paths]
```

理论上，MANO JSON 文件和关键点 JSON 文件是同时生成、同时存在的，所以两者的 `vis_idcs` 一致。但如果异常情况下一个文件存在而另一个不存在，`get_ternary_mask()` 的边界将由关键点文件决定，而 MANO 参数插值可能有不一致的锚点集合。

#### 1.7 检测框有效性与 2D 关键点的耦合关系

在 Dyn-HaMR 中，**检测框有效性和 2D 关键点获取不是两个独立概念**——它们被 HaMeR Pipeline 耦合在一起。

**上游耦合（`third-party/hamer/run.py` 的 3-pass 架构）**：

```
Pass 1: YOLO 检测
  │  YOLO 输出 bbox + handedness + confidence
  │
Pass 2: bbox 清理
  │  移除非主手、低置信度、超大 bbox、虚假短运动、手性错误的检测
  │  缺失帧通过 bbox 线性插值填补
  │
Pass 3: HaMeR 推理（在清理后的 bbox 上）
  │
  ├─► ViTPose 提取 2D 关键点 (21 个关节, OpenPose 格式)
  └─► HaMeR 回归 MANO 参数 (betas, body_pose, global_orient, cam_trans)
      │
      ▼
  同时写入磁盘（成对出现）:
    {tid:03d}/{frame}_keypoints.json   ← 2D 关键点
    {tid:03d}/{frame}_mano.json        ← MANO 参数
```

关键特征：

1. **成对生成**：MANO JSON 和关键点 JSON 由同一次 HaMeR 推理产生，**要么同时存在，要么同时不存在**。不存在「检测有效但关键点缺失」或「关键点有效但检测缺失」的中间状态。

2. **检测的有效性由 Pass 2 的清理结果隐式表达**：Pass 2 处理后，被移除的检测（低置信度、非主手、手性错误、短运动）不再进入 Pass 3，因此不会产生任何 JSON 文件。有效检测的帧则同时拥有两个 JSON 文件。

3. **没有独立的「检测框有效性」标志**：Dyn-HaMR 不存储类似 HaWoR `det=True/False` 的显式标记。检测框的有效性完全由 JSON 文件是否存在于磁盘来**隐式表达**。

4. **插值与检测框的隐式关系**：Pass 2 的缺失帧 bbox 插值可以「恢复」部分被跳过的帧——如果 bbox 被线性插值成功填补，该帧会在 Pass 3 重新运行 HaMeR，从而生成 JSON 文件。这意味着：

   - 短间隙（≤ 25 帧，`PATIENCE_FRAMES=25`）：bbox 插值 → HaMeR 重跑 → JSON 文件存在 → vis_mask=1
   - 长间隙（> 25 帧或 Pass 2 无法填补）：无 HaMeR 输出 → JSON 文件不存在 → vis_mask=0

#### 1.8 插值触发条件与判定逻辑总览

将上述所有机制串联，形成以下完整的判定链路：

```
帧 t 的手部数据状态:
  │
  ├─ {tid}/{frame_t}_keypoints.json 存在?
  │   │
  │   ├─ YES → 可见帧
  │   │   └─ vis_mask[t] = 1
  │   │   └─ MANO 参数 = 直接从 JSON 读取
  │   │   └─ 2D 关键点 = 直接从 JSON 读取
  │   │
  │   └─ NO → 缺失帧
  │       │
  │       ├─ t 在 [track_s, track_e) 范围内?
  │       │   │  track_s = 第一个有文件的帧索引
  │       │   │  track_e = 最后一个有文件的帧索引 + 1
  │       │   │
  │       │   ├─ YES → 遮挡/检测失败
  │       │   │   └─ vis_mask[t] = 0
  │       │   │   └─ MANO 参数: Slerp/线性插值 (load_mano_preds)
  │       │   │   └─ 2D 关键点: 线性插值 x/y + conf=邻近均值×0.8 (load_keypoints_with_interp)
  │       │   │   └─ 参与优化 loss (vis_mask >= 0 = True)
  │       │   │
  │       │   └─ NO → 出镜
  │       │       └─ vis_mask[t] = -1
  │       │       └─ MANO 参数: 全零默认值 (不被插值)
  │       │       └─ 2D 关键点: 全零默认值 (不被插值)
  │       │       └─ 不参与优化 loss (vis_mask >= 0 = False)
```

**插值仅发生在条件**：文件不存在 **且** `track_s ≤ t < track_e`。两个条件必须同时满足。

**vis_mask 三值化的判定时序**：

```
load_keypoints_with_interp()  ← 先执行，内部计算 vis_idcs
    │
    ▼
load_mano_preds()             ← 独立计算 vis_idcs（基于 _mano.json）
    │
    ▼
get_ternary_mask()            ← 基于 track_vis_masks（来自 keypoints.json 文件存在性）
    │  track_s = min(vis_idcs_from_keypoints)
    │  track_e = max(vis_idcs_from_keypoints) + 1
    │  vis_mask[t] = -1 if t outside [track_s, track_e) else (1 if file exists else 0)
    │
    ▼
最终 obs_data["vis_mask"]  (B, T) ∈ {-1.0, 0.0, 1.0}
```

#### 1.9 检测框存在但预测错误的四种情况

在 Dyn-HaMR 管线中，只要 YOLO 产生了有效的检测框（Pass 1），且 Pass 2 清理未将其移除，该 bbox 就会进入 Pass 3 的 HaMeR 推理，生成 `_mano.json` 和 `_keypoints.json`。此后 vis_mask = 1（可见）。然而，**以下四种情况会导致这些文件中的数据质量下降**，且管线中没有显式的语义级质量检查来拦截。

##### 情况一：YOLO 误检——bbox 正确但目标不是手

YOLO 可能将非手物体（类似肤色的区域、手套、甚至无关纹理）误分类为手部，输出 `conf ≥ 0.5` 的检测。该 bbox 通过 Pass 1 进入 Pass 3。

Pass 3 的 HaMeR 在这些 bbox 裁剪出的图像上执行前向推理——它**没有机制判断「这不是手」**，仍然输出一组数值合法的 MANO 参数和 2D 关节投影。下游 Dyn-HaMR 优化将这些帧的 2D 重投影 loss 计算为「异常值」——误差很大，但因 GMoF 鲁棒损失（σ=100）而软截断，不会产生梯度爆炸。

Pass 2 通过 oversized bbox 过滤（>50% 图像面积）、重叠去重（IoU > 0.7）、虚假短运动移除（<30 帧 + 前后各 ≥30 帧空白）间接过滤部分误检，但**没有专门的语义正确性检查**。

##### 情况二：部分遮挡导致 MANO 参数退化

当手部被严重遮挡时，HaMeR 的 ViT 骨干网络看到的裁剪图像中只有部分手指或手掌。Transformer 解码器仍然输出一组参数，但可能出现：

- **退化的关节位置**：MANO 3D 关节坍塌到原点附近 → `hand_scale` 极小（< 0.001m）
- **异常的 2D 重投影**：关节投影到图像外或高度集中的小区域

Dyn-HaMR 优化阶段的**被动检测**（`losses.py:856-861`）：

```python
if torch.isnan(hand_scale).any() or torch.isinf(hand_scale).any() or (hand_scale < 5.0).any():
    print(f"ERROR: Invalid hand_scale! min={hand_scale.min():.2f}")
    print(f"  This indicates MANO produced degenerate output (collapsed joints)")
    return torch.tensor(1e6, device=hand_scale.device, requires_grad=True)
```

当检测到退化的 hand_scale（< 5.0 像素，说明 3D 关节在图像上的投影范围异常小）时，返回巨大常量 loss（1e6），配合 L-BFGS 的 checkpoint 回滚机制（`optimizers.py:320-324`）防止优化发散。这是**被动补救**——不标记该帧为无效，仅阻止当次梯度爆炸。

##### 情况三：极端视角导致 2D 关键点偏离

HaMeR 的 `pred_keypoints_2d` 从 MANO 3D 关节通过透视投影计算。若预测的 MANO pose 在相机空间中产生了极端深度值（例如手腕被预测在相机后方），投影坐标会异常：

- `depth_constraint_loss`（权重 100）对「相机后方的关节」施加 100× 的 `relu(-depth)^2` 惩罚
- 2D 重投影点可能落在图像外，但 `Joints2DLoss` 对此**没有显式边界检查**——只计算 `pred - obs` 的误差。若 pred 在图像外（x=5000, y=-200），误差极大但被 GMoF 截断

##### 情况四：Pass 2 patience 插值产生的不精确 bbox

当 patience 插值（`run.py:1053-1177`）在间隙帧中线性内插 bbox 时，门控条件 `center_distance ≤ avg_width`（L1107）仅做**纯 bbox 层面的位置检查**，无法验证裁剪出的图像内容是否确实是完整的手。若手在消失期间实际发生了超出门控的运动幅度但侥幸未被检测到，插值 bbox 可能**不完全覆盖手部**→ 裁剪图像只包含部分手 → MANO 参数和 2D 关键点预测质量下降。

与情况一不同，这些帧的 bbox 是合理的位置插值（非误检），但图像内容不完整。

##### 质量保障机制总结

| 阶段 | 检查 | 方式 | 效果 |
|------|------|------|------|
| Pass 1 | YOLO conf ≥ 0.5 | 置信度阈值 | 过滤低置信度检测 |
| Pass 2 | oversized bbox >50% | 面积阈值 | 过滤异常大误检 |
| Pass 2 | IoU > 0.7 / containment > 0.7 | 重叠去重 | 过滤重复检测 |
| Pass 2 | 虚假短运动 < 30 帧 | 时序一致性 | 过滤孤立误检段 |
| Pass 2 | patience 距离门控 | 中心距离 ≤ bbox 宽度 | 防止跨大运动错误插值 |
| Pass 2→3 | keypoints shape 检查 | `(21, 3)` 断言 | 格式验证（非语义） |
| Dyn-HaMR loss | NaN/Inf 检测 | `torch.isnan/isinf` | 返回 1e6/1e8 + checkpoint 回滚 |
| Dyn-HaMR loss | degenerate hand_scale | < 5.0 像素检测 | 返回 1e6 + checkpoint 回滚 |
| Dyn-HaMR L-BFGS | NaN loss 回滚 | `np.isnan → load_checkpoint` | 恢复上一个有效参数快照 |

**关键空白**：HaMeR 推理后（Pass 3 输出）、写入 JSON 文件前，**没有任何语义级质量检查**——不会验证 MANO 关节位置是否物理合理、2D 投影是否在图像内、旋转矩阵是否正交、预测的手是否确实对应图像中的手。Dyn-HaMR 优化的 NaN/退化检测也只能在**优化运行时**被动发现，而不会在数据加载阶段预先标记问题帧并将其 vis_mask 设为 0。

---

### 2. vis_mask 在优化中的掩码作用

#### 2.1 二值化

进入优化器前，三值 `vis_mask` 被转换为二值掩码（`optim/optimizers.py:419, 474`）：

```python
vis_mask = obs_data["vis_mask"] >= 0   # -1→False, 0→True, 1→True
```

这意味着：**-1（出镜）被排除，0（遮挡）和 1（可见）都参与优化**——遮挡帧与可见帧在损失计算中完全等同。

#### 2.2 各 Loss 函数的掩码使用方式

| Loss | 文件:行号 | 掩码方式 | 效果 |
|------|----------|---------|------|
| `joints2d` (重投影) | `losses.py:834-835` | 索引过滤 `data[mask]` | 直接删除无效帧的样本 |
| `joints3d` | `losses.py:783-785` | `vis_mask & ~isinf(obs)` AND | 双重过滤：帧有效 + 观测非 inf |
| `verts3d` | `losses.py:800-801` | `vis_mask & ~isinf(obs)` AND | 同 joints3d |
| `joints3d_smooth` | `losses.py:1244-1246` | `mask[:,1:] & mask[:,:-1]` | 相邻两帧都有效才计算 delta |
| `pose_prior` | `losses.py:1190-1191` | `loss[mask.bool()]` bool 索引 | 仅对有效帧惩罚偏离初始值 |
| `bio_loss` | `bio_loss.py:280` | `joints[valid_mask]` bool 索引 | 仅对有效帧计算生物力学约束 |
| `penetration` | `losses.py:614-618` | **不使用掩码** | 对所有帧计算穿透损失 |
| `shape_prior` | `losses.py:595-596` | **不使用掩码** | 统一 L2 正则化 betas |
| `depth_constraint` | `losses.py:487-496` | 传递 `cam_R, cam_t` | 基于相机位姿的隐式逐帧约束 |

##### 2.2.1 joints2d 重投影损失的完整掩码链

2D 重投影损失是权重最高的 loss 项（10000），其掩码链涉及三个层级的过滤：

**层级 1 — 优化器层的 `vis_mask >= 0` 二值化**（`optimizers.py:419`）：

```python
vis_mask = obs_data["vis_mask"] >= 0   # (B, T) bool
# -1 (出镜) → False → 帧被排除
#  0 (遮挡) → True  → 帧参与 loss
#  1 (可见) → True  → 帧参与 loss
```

此掩码传递到 `RootLoss.forward()` / `SMPLLoss.forward()`，再传递给 `Joints2DLoss.forward()`。

**层级 2 — Joints2DLoss 内的索引过滤**（`losses.py:833-837`）：

```python
def forward(self, joints2d_obs, joints2d_pred, mask=None):
    if mask is not None:
        mask = mask.bool()
        joints2d_obs = joints2d_obs[mask]    # (N_valid, 21, 3)
        joints2d_pred = joints2d_pred[mask]  # (N_valid, 21, 2)
```

`mask.bool()` 将二值化后的 vis_mask 压平到 `(N_valid,)` ——只有 `vis_mask >= 0` 的帧的样本被保留。`N_valid ≤ B×T`（B=轨道数，T=序列长度）。

索引过滤将 `(B, T, ...)` 张量压缩为 `(N_valid, ...)`，**物理上删除了无效样本**。这比 bool 索引 `loss[mask]` 更激进——后者的输出形状不变，仅在无效位置为零；而索引过滤改变了张量的第一维大小。

**层级 3 — confidence 置信度加权**（`losses.py:839-842`）：

```python
joints2d_obs_conf = joints2d_obs[..., 2:3]   # (N_valid, 21, 1)
if self.ignore_op_joints is not None:
    joints2d_obs_conf[..., self.ignore_op_joints, :] = 0.0
```

由于 `dataset.py:235` 将所有置信度覆写为 1.0（见 §1.6.3），`joints2d_obs_conf` 在**所有有效帧 + 所有关节**上恒为 1：

```python
# dataset.py:235
joints2d_data[:, :, 2] = 1.0
```

因此 `joints2d_obs_conf**2 = 1.0`，置信度加权**实质上退化为了等权重**。

**最终计算**：

```python
error = joints2d_pred - joints2d_obs[..., :2]       # (N_valid, 21, 2)
error = error / hand_scale * 100.0                    # 归一化
robust_sqr_dist = gmof(error, sigma=100)              # GMoF 鲁棒化
reproj_err = (joints2d_obs_conf**2) * robust_sqr_dist # 加权
loss = torch.mean(reproj_err)                         # 均值
```

**完整掩码链路总结**：

```
obs_data["vis_mask"]  (B, T) float ∈ {-1, 0, 1}
    │
    ▼  optimizers.py:419
vis_mask >= 0  →  (B, T) bool  {-1→False, 0→True, 1→True}
    │
    ▼  losses.py:834-835
索引过滤 joints2d_obs[mask], joints2d_pred[mask]
    │  (B, T, 21, 3) → (N_valid, 21, 3)
    │  N_valid = vis_mask>=0 的帧数 ≤ B×T
    │
    ▼  losses.py:839
joints2d_obs_conf = joints2d_obs[..., 2:3]  → (N_valid, 21, 1)
    │  dataset.py:235 已将 conf 覆写为 1.0
    │  → conf^2 = 1.0, 置信度加权退化
    │
    ▼  losses.py:876-879
reproj_err = conf^2 × gmof(normalized_error, σ=100)
loss = mean(reproj_err)
```

**关键结论**：

- vis_mask=0 的遮挡帧（插值关键点）和 vis_mask=1 的可见帧（原始检测关键点）在 2D 重投影损失中**完全等同**——它们都通过 `>= 0` 检查，都参与 loss，且由于置信度被覆写为 1.0，权值也相同。
- vis_mask=-1 的出镜帧被 `>= 0` 的二值化**彻底排除**——它们的 2D 关键点为全零默认值，但根本不会进入 `Joints2DLoss`。
- 置信度加权机制因 `dataset.py:235` 的覆写而**名存实亡**——如果没有覆写，原始 ViTPose 高置信度帧和插值低置信度帧之间会有自然的权重差异。

#### 2.3 平滑损失的特殊处理

`joints3d_smooth_loss` 的掩码逻辑最为精细：它不仅要求当前帧有效，还需要**相邻帧对双方都有效**：

```python
if mask is not None:
    mask = mask.bool()
    mask = mask[:, 1:] & mask[:, :-1]   # t 和 t-1 都必须为 True
    loss = loss[mask]
```

这确保了平滑损失只在连续的可见段内计算，不会在出镜帧和可见帧之间产生虚假的平滑约束。

---

### 3. HaWoR 的 pred_valid（对比参考）

Dyn-HaMR **没有** `pred_valid` 字段。这是 HaWoR 独有的机制。

HaWoR 使用**两个独立 bool 数组**表达帧有效性（定义在 `scripts/.../batch_hawor_infiller.py` → `save_mano_data()`）：

```python
{
    'left_hand': {
        'mano_params': {...},
        'pred_valid': (T,) bool,          # HaWoR 模型推理是否成功
        'detection_failed': (T,) bool,    # 仅 allow_disappear=True
    }
}
```

- `pred_valid[i] = True`：HaWoR 模型成功推理（检测框存在 + MANO 参数有效）
- `pred_valid[i] = False`：无检测框或推理失败
- `detection_failed[i] = True`：检测器明确报告失败（赋予默认隐藏姿态）

| 维度 | Dyn-HaMR vis_mask | HaWoR pred_valid |
|------|------------------|-------------------|
| 数据结构 | (B, T) float，三值 {-1, 0, 1} | (T,) bool × 2（pred_valid + detection_failed） |
| 判定依据 | 关键点 JSON 文件是否存在 | HaWoR 模型推理是否成功 |
| 出镜判定 | 基于轨道首尾帧的时序启发式 | 基于检测框存在性 + allow_disappear 模式 |
| 遮挡/检测失败 | 合并为 vis_mask=0，无法区分 | 可区分（detection_failed 独立标记） |
| 对下游影响 | ≥0 二值掩码控制 loss | Infiller 注意力掩码 + 填充行为 |

---

## 第二部分：Dyn-HaMR 三阶段优化

### 4. 整体架构

优化入口：`dyn-hamr/run_opt.py` → `run_opt()`

```
N_STAGES = 3

stage_loss_weights[0] → RootOptimizer   (50 L-BFGS iterations)
stage_loss_weights[1] → SmoothOptimizer (300 L-BFGS iterations)
stage_loss_weights[2] → 未使用 (MotionOptimizer 已注释)
                        ↓
                  run_prior()  (HMP 精修, 可选)
```

两个优化器**共享同一个** `BaseSceneModel` 实例——参数值跨阶段保留，但优化器状态（L-BFGS 的 Hessian 近似）各自独立。

---

### 5. Stage 0: RootOptimizer（`root_fit`）

代码：`dyn-hamr/optim/optimizers.py:381-421`

#### 5.1 优化变量

```python
param_names = ["trans", "root_orient"]
```

| 参数 | 形状 | 含义 | 初始值来源 |
|------|------|------|-----------|
| `trans` | (B, T, 3) | 世界空间手腕平移 | HaMeR `cam_trans` → cam2world 转换 |
| `root_orient` | (B, T, 3) | 世界空间手腕旋转（轴角） | HaMeR `global_orient` → cam2world 转换 |

#### 5.2 冻结参数

通过 `set_require_grads()` 冻结的参数（`params.py:59-72`）：

```python
# set_require_grads 先冻结全部，再解冻指定参数
for name in self.param_names:
    self._set_param_grad(name, False)   # 全部冻结
for name in ["trans", "root_orient"]:
    self._set_param_grad(name, True)    # 仅解冻这两个
```

冻结的参数包括：
- `betas` (B, 10) — 保持 HaMeR 初始均值
- `latent_pose` (B, T, D) — 保持 HaMeR 初始值
- `world_scale` (1, 1) — 保持 1.0
- `cam_f`, `delta_cam_R` — 保持初始值

#### 5.3 前向计算

```python
def forward_pass(self, obs_data):
    pred_data = self.model.pred_params_mano(obs_data["is_right"])
    pred_data["cameras"] = self.model.params.get_cameras()
    vis_mask = obs_data["vis_mask"] >= 0
    loss, stats_dict = self.loss(obs_data, pred_data, vis_mask)
    return loss, stats_dict, pred_data
```

`pred_params_mano()` 调用链：`latent2pose(latent_pose)` → `pred_mano(trans, root_orient, body_pose, is_right, betas)` → MANO 前向 → `{joints3d, verts3d, points3d, joints3d_op, l_faces, r_faces, body_pose}`

#### 5.4 活跃损失函数（RootLoss）

权重来自 `confs/optim.yaml:loss_weights`，选取各列表的第 0 个元素（stage 0）：

| Loss | 权重 | GT 来源 | 预测来源 | 帧选取 | 归一化策略 |
|------|------|---------|---------|--------|-----------|
| `joints2d` | 10000 | `obs_data["joints2d"]` — ViTPose 2D 关键点 (B,T,21,3) | `cam_util.reproject(pred_data["joints3d_op"], *cameras)` | `vis_mask >= 0` 逐帧 | GMoF(σ=100) + hand_scale 归一化：`error / hand_scale * 100` |
| `joints3d_smooth` | 1000 | 无外部 GT（自监督平滑） | `pred_data["joints3d"]` 的帧间 delta | 邻帧对 mask: `mask[:,1:] & mask[:,:-1]` | hand_scale 归一化：`joints3d / hand_scale` |
| `depth_constraint` | 100 | 隐式（min_depth=0, max_depth=999） | `pred_data["joints3d"]` → 相机空间深度 | 逐帧 | 无归一化（深度单位为米） |

**GT 来源详解**：

- **joints2d**：`obs_data["joints2d"]` 来自 ViTPose 在 HaMeR Pipeline 中提取的 2D 关键点。数据加载时通过 `load_keypoints_with_interp()` 对缺失帧做了线性插值，且所有非零置信度被强制设为 1.0（`dataset.py:236`）。形状为 `(B, T, 21, 3)`，其中最后一维为 `(x, y, confidence)`。
- **joints3d_smooth**：无外部 GT。利用相邻帧间 3D 关节位置变化应平滑的先验，最小化帧间 delta 的 L2 范数。

**预测来源详解**：

- **joints3d_op**：MANO 模型输出的 21 个关节（与 ViTPose 的 21 个 OpenPose 手部关键点一一对应），通过 `cam_util.reproject()` 重投影到 2D。
- **joints3d**：MANO 模型输出的 16 个标准关节（1 手腕 + 15 手指），用于平滑损失。

**帧选取**：两个阶段都处理**完整序列**（B, T），不做窗口滑动或分块。`T = seq_len`（如 128 帧）。`vis_mask >= 0` 确保仅有效帧参与。

**归一化策略**：

**Joints2DLoss 的 hand_scale 归一化**（`losses.py:848-872`）：

```python
# 1. 从观测关键点估计手部尺度（bbox 对角线长度）
kp_min = valid_obs.min(dim=1, keepdim=True)[0]
kp_max = valid_obs.max(dim=1, keepdim=True)[0]
hand_scale = sqrt(((kp_max - kp_min)^2).sum(dim=-1))  # (N, 1, 1)
hand_scale = clamp(hand_scale, 5, 1000)

# 2. 归一化像素误差
error = (pred - obs) / hand_scale * 100.0  # 100x 放大避免数值过小

# 3. GMoF 鲁棒损失
robust_sqr_dist = gmof(error, sigma=100)  # sigma 同样在归一化空间
loss = mean(conf^2 * robust_sqr_dist)
```

**joints3d_smooth_loss 的 hand_scale 归一化**（`losses.py:1218-1251`）：

```python
# 1. 从 3D 关节估计手部尺度（3D bbox 对角线）
joints_min = joints3d.min(dim=2, keepdim=True)[0]
joints_max = joints3d.max(dim=2, keepdim=True)[0]
hand_scale = sqrt(((joints_max - joints_min)^2).sum(dim=-1))  # (B, T, 1, 1)
hand_scale = clamp(hand_scale, 0.001, 1.0)  # 米制单位

# 2. 归一化 3D 关节
joints3d_normalized = joints3d / hand_scale

# 3. 帧间 delta
delta = joints3d_normalized[:, 1:] - joints3d_normalized[:, :-1]
loss = 0.5 * sum(masked(delta^2))
```

**NaN/Inf 安全检查**：两个 loss 函数都包含显式的 NaN/Inf 检测——若发现则返回一个巨大的常量（1e6 或 1e8），配合 L-BFGS 的 checkpoint 回滚机制（`optimizers.py:320-324`）防止优化发散。

---

### 6. Stage 1: SmoothOptimizer（`smooth_fit`）

代码：`dyn-hamr/optim/optimizers.py:424-476`

#### 6.1 优化变量

```python
param_names = ["trans", "root_orient", "betas", "latent_pose"]
if model.opt_scale:     # 仅非静态相机
    param_names += ["world_scale"]
if model.opt_cams:      # 默认 False
    param_names += ["cam_f", "delta_cam_R"]
```

| 参数 | 形状 | 初始值来源 | 是否新加入 |
|------|------|-----------|-----------|
| `trans` | (B, T, 3) | Stage 0 优化结果（保留） | 否 |
| `root_orient` | (B, T, 3) | Stage 0 优化结果（保留） | 否 |
| `betas` | (B, 10) | HaMeR 初始均值（时间平均） | **是** |
| `latent_pose` | (B, T, D) | HaMeR 初始值 | **是** |
| `world_scale` | (1, 1) | 1.0 | **是**（条件） |

#### 6.2 参数传递机制

两个优化器**共享同一个 `base_model` 对象**（`run_opt.py:150-159`）：

```python
# Stage 0
optim = RootOptimizer(base_model, stage_loss_weights, **opts)
optim.run(...)   # 优化 trans, root_orient

# Stage 1 — 同一个 base_model
optim = SmoothOptimizer(base_model, stage_loss_weights, ...)
optim.run(...)   # 优化 trans, root_orient, betas, latent_pose
```

当 `SmoothOptimizer.__init__` 调用 `set_opt_vars()` 时：
1. 先冻结**全部**参数（包括 stage 0 优化过的 `trans` 和 `root_orient`）
2. 再解冻新的参数列表（包括 `trans`, `root_orient`, `betas`, `latent_pose`）

`trans` 和 `root_orient` 的值保留 stage 0 的优化结果，但 L-BFGS 的 Hessian 近似状态被重新初始化（因为 `SmoothOptimizer` 创建了新的 `torch.optim.LBFGS` 实例）。

**Checkpoint 机制**：每个优化器有独立的 checkpoint 文件：
```
{out_dir}/root_fit/root_fit_params.pth    ← Stage 0 参数
{out_dir}/root_fit/root_fit_optim.pth    ← Stage 0 优化器状态
{out_dir}/smooth_fit/smooth_fit_params.pth ← Stage 1 参数
{out_dir}/smooth_fit/smooth_fit_optim.pth ← Stage 1 优化器状态
```

这允许独立恢复任一阶段的优化。

#### 6.3 前向计算

相比 RootOptimizer，`SmoothOptimizer.forward_pass()` 额外传递了：
- `pred_data.update(self.model.params.get_vars())` — 将 `betas`, `latent_pose` 等所有参数暴露给 loss
- `pred_data["cam_R"], pred_data["cam_t"]` — 显式相机外参
- `nsteps = self.model.seq_len` — 用于缩放 `shape_prior`

#### 6.4 活跃损失函数（SMPLLoss，继承 RootLoss）

权重选取各列表的第 1 个元素（stage 1）。在 RootLoss 基础上**新增**：

| Loss | 权重 | GT 来源 | 预测来源 | 归一化策略 |
|------|------|---------|---------|-----------|
| `pose_prior` | 1 | `obs_data["init_latent_pose"]` — HaMeR 初始潜变量（在 `BaseSceneModel.initialize()` 中置入） | `pred_data["latent_pose"]` | `sum((pred - init)^2)` 无归一化 |
| `shape_prior` | 0.05 | 隐式（零均值高斯先验） | `pred_data["betas"]` | `sum(betas^2) * nsteps` |
| `penetration` | 0（禁用） | — | `pred_data["verts3d"]` + winding numbers | 穿透顶点间的最小距离平方 |

**权重变化**（相比 stage 0）：
- `joints3d_smooth`：1000 → **10000**（10× 增强时序平滑）
- `depth_constraint`：100 → **0**（stage 1 不再约束深度——手部已在正确的深度位置附近）

#### 6.5 pose_prior 的特殊性

这里的"pose prior"**不是**标准 VPoser 先验（KL 散度），而是对初始 HaMeR 预测的 L2 正则化：

```python
# losses.py:1177-1193
def pose_prior_loss(latent_pose_pred, latent_pose_init=None, mask=None):
    if latent_pose_init is not None:
        loss = (latent_pose_pred - latent_pose_init)**2    # L2 偏离初始值
    else:
        loss = latent_pose_pred**2                          # L2 偏离零
    if mask is not None:
        loss = loss[mask.bool()]
    return torch.sum(loss)
```

`init_latent_pose` 在 `BaseSceneModel.initialize()` 中被设置为 HaMeR 预测的 detach 副本（`base_scene.py:146`），且**不参与梯度**。这确保了优化后的手部姿态保持在 HaMeR 单帧预测的合理邻域内。

---

### 7. L-BFGS 优化器配置

代码：`dyn-hamr/optim/optimizers.py:52-54`

```python
self.optim = torch.optim.LBFGS(
    self.opt_params,
    max_iter=20,              # 每次 step 内部的最多函数评估次数
    lr=1.0,                   # 初始步长猜测
    line_search_fn="strong_wolfe"  # 强 Wolfe 条件线搜索
)
```

**与标准 SGD/Adam 的关键区别**：
- L-BFGS 是**全批次**二阶优化器——每次 `optim.step(closure)` 会多次调用 `closure()` 进行线搜索
- `max_iter=20` 控制每次 step 内线搜索的最大迭代次数，而非外循环
- `strong_wolfe` 线搜索确保足够的步长下降和曲率条件
- `loss_dicts` 记录了每次 `closure()` 调用的损失值（一个外循环迭代对应多个 loss 记录），最终以箱线图保存

**早停机制**（`optimizers.py:326-343`）：
- NaN 检测：`np.isnan(self.cur_loss)` → 从 checkpoint 回滚并 raise
- 平台检测：若连续 `max_chunk_steps=20` 步损失变化 < 20，提前终止
- 最后一块：`reached_max` 后再 `max_chunk_steps` 步终止

---

### 8. 配置速查

```yaml
# confs/optim.yaml (核心片段)
optim:
  options:
    lr: 1.0
    lbfgs_max_iter: 20
    save_every: 20
    max_chunk_steps: 20

  root:
    num_iters: 50         # Stage 0 外循环

  smooth:
    num_iters: 300        # Stage 1 外循环
    opt_scale: False      # world_scale 不优化

  loss_weights:           # [stage0, stage1, stage2]
    joints2d:           [10000,  10000,   10000]
    joints3d_smooth:    [1000,   10000.0, 0.0]
    pose_prior:         [1,      1,       1]
    shape_prior:        [0.05,   0.05,    0.05]
    depth_constraint:   [100.0,  100.0,   0.0]
    joints3d:           [0.0,    0.0,     0.0]    # 禁用
    bio:                [0.0,    0.0,     0.0]    # 禁用
    penetration:        [0,      0,       0]      # 禁用
```

---

### 9. 阶段间对比速查

| 维度 | Stage 0 (RootOptimizer) | Stage 1 (SmoothOptimizer) |
|------|------------------------|---------------------------|
| **优化变量** | trans, root_orient | trans, root_orient, betas, latent_pose |
| **冻结变量** | betas, latent_pose, world_scale, cam_* | world_scale, cam_* |
| **迭代次数** | 50 | 300 |
| **Loss 函数** | RootLoss | SMPLLoss (= RootLoss + priors) |
| **joints2d 权重** | 10000 | 10000 |
| **joints3d_smooth 权重** | 1000 | 10000 |
| **depth_constraint 权重** | 100 | 100 |
| **pose_prior** | 未激活（不在 loss 类中） | 1 (L2 向 HaMeR 初始值) |
| **shape_prior** | 未激活 | 0.05 × seq_len |
| **penetration** | 0（禁用） | 0（禁用） |
| **参数初始值来源** | HaMeR 预测 → cam2world 转换 | Stage 0 优化结果 |
| **优化器状态** | 新建 L-BFGS | 新建 L-BFGS（Hessian 近似重新初始化） |
| **checkpoint 文件** | `root_fit_params.pth` | `smooth_fit_params.pth` |
| **完整序列处理** | 是 (B, T 整段) | 是 (B, T 整段) |
| **窗口/分块** | 无 | 无 |

### 10. obs_data 字段完整清单与优化实例

#### 10.1 优化实例概览

三阶段优化共享**一个** `BaseSceneModel` 实例，该实例封装了：

- **`hand_model`**（`MANO` 实例，`batch_size=B*T`，`pose2rot=True`）—— 所有帧/轨道共享的单一 MANO 手部模型
- **`params`**（`CameraParams`，继承自 `Params`）—— 存储所有可优化和固定的 `nn.Parameter` 张量，通过 `requires_grad` 控制各阶段的优化范围
- 两个优化器（`RootOptimizer`、`SmoothOptimizer`）各自持有独立的 `torch.optim.LBFGS` 实例和 checkpoint 文件，但操作同一组参数

#### 10.2 obs_data 字段完整清单

`obs_data` 由 `MultiPeopleDataset` 加载、`BaseSceneModel.initialize()` 扩展，随后传递给每个优化器的 `forward_pass()`。

**来自数据集**（`dataset.py:__getitem__`，形状为单轨道的 `(1, T, ...)`）：

| 字段 | 形状 | 类型 | 来源 | 含义 |
|------|------|------|------|------|
| `joints2d` | (B, T, 21, 3) | float32 | ViTPose → `_keypoints.json` → `load_keypoints_with_interp()` → 线性插值 + conf=1.0 覆写 | 2D 关键点（x, y, confidence），OpenPose 手部格式 |
| `init_body_pose` | (B, T, 15, 3) | float32 | HaMeR → `_mano.json` → Slerp 插值 | 15 个手指关节轴角（不含手腕），作为初始值 |
| `init_body_shape` | (B, T, 10) | float32 | HaMeR → `_mano.json` → 线性插值 | MANO betas，时间平均后得 (B, 10) |
| `init_root_orient` | (B, T, 3) | float32 | HaMeR → `_mano.json` → Slerp 插值 | 手腕全局旋转轴角（相机空间） |
| `init_trans` | (B, T, 3) | float32 | HaMeR → `_mano.json` → 线性插值 | 手腕平移（相机空间） |
| `is_right` | (B, T) | float32 | HaMeR → `_mano.json` → 全轨道恒定 | 手性：1=右手, 0=左手；与 track_id 一致 |
| `vis_mask` | (B, T) | float32 | `get_ternary_mask()` | -1=出镜, 0=遮挡, 1=可见 |
| `seq_interval` | (2,) | int32 | 数据集内部的序列起止索引 | `[start_idx, end_idx)` |
| `track_interval` | (2,) | int32 | 数据集中该轨道首尾检测帧索引 | `[track_s, track_e)` |
| `track_id` | 标量 | int | 轨道 ID（tid） | 0=左手, 1=右手 |
| `seq_name` | 标量 | str | 配置中的序列名 | 用于日志和可视化命名 |

**来自相机**（`CameraData.as_dict()`，形状为单轨道 `(T, ...)`）：

| 字段 | 形状 | 类型 | 来源 | 含义 |
|------|------|------|------|------|
| `cam_R` | (T, 3, 3) | float32 | `cameras.npz` → `load_cameras_npz()` → 偏移后 | 世界到相机旋转矩阵，第一帧平移被置零 |
| `cam_t` | (T, 3) | float32 | 同上 | 世界到相机平移向量 |
| `intrins` | (T, 4) | float32 | 同上 | `[fx, fy, cx, cy]`，按图像尺寸缩放 |
| `static` | bool | — | 配置 + 文件检查 | True=静态相机，禁用 scale 优化 |

**由 BaseSceneModel.initialize() 扩展**：

| 字段 | 形状 | 来源 | 用途 |
|------|------|------|------|
| `init_latent_pose` | (B, T, D) | `self.pose2latent(init_body_pose).detach()` | SMPLLoss pose_prior 的 GT 目标——L2 正则化的锚点 |

**obs_data 的生命周期**：
```
MultiPeopleDataset.load_data()
  → data_out dict (含 joints2d, init_body_pose, init_body_shape, ...)
    → MultiPeopleDataset.__getitem__()  → 返回单轨道 obs_data
      → DataLoader collate → 批量为 (B, T, ...)
        → move_to(obs_data, device)
          → BaseSceneModel.initialize(obs_data, cam_data)  ← 消费 + 扩展
            → RootOptimizer.forward_pass(obs_data)
            → SmoothOptimizer.forward_pass(obs_data)
```

#### 10.3 预测数据（pred_data）

由 `BaseSceneModel.pred_params_mano()` → `pred_mano()` 产生，每次 L-BFGS closure 重新计算：

| 字段 | 形状 | 来源 |
|------|------|------|
| `joints3d` | (B, T, 16, 3) | MANO 前向：16 个标准关节（1 手腕 + 15 手指） |
| `joints3d_op` | (B, T, 16, 3) | 同上（无 OpenPose 重映射） |
| `verts3d` | (B, T, 778, 3) | MANO 前向：所有顶点 |
| `points3d` | (B, T, 778, 3) | 同 verts3d |
| `l_faces` | (F+14, 3) | 左手面片（反转绕组 + 额外水密面） |
| `r_faces` | (F+14, 3) | 右手面片（标准绕组 + 额外水密面） |
| `body_pose` | (B, T, 45) | MANO 输出的手部姿态轴角 |
| `is_right` | (B, T) | 手性 |

SmoothOptimizer 额外通过 `pred_data.update(self.model.params.get_vars())` 注入所有可优化参数的当前值（`latent_pose`, `betas`, `world_scale` 等），以及 `cam_R`, `cam_t`。

#### 10.4 优化设计思路

Dyn-HaMR 优化是一种**渐进式解冻**策略：每个阶段通过梯度冻结/解冻（`set_require_grads()`）控制优化范围，先锁定高杠杆参数（全局轨迹），再解锁全部参数精修（形状、手指姿态）。

**设计动机**：
- 平移和全局旋转是**最高杠杆**的参数（6 DOF/帧）——决定手在 3D 空间中的位置。若在此阶段同时优化手指姿态，手指关节的 2D 重投影误差会与手腕位置误差混淆，导致优化不稳定
- Stage 0 只用数据拟合项（joints2d, joints3d_smooth, depth_constraint），「完全信任观测」
- Stage 1 引入先验（pose_prior, shape_prior），在 HaMeR 初始值的邻域内精修，防止漂移
- 权重变化反映意图：joints3d_smooth 从 1000→10000（10× 增强），因为 Stage 1 会改变手指关节，需要更强的时序平滑约束

**损失权重作为阶段开关**：`optim.yaml` 中的 3 元素列表通过 `stage_loss_weights` 转置为各阶段独立的权重字典。权重为 0 的 loss 通过 `if weight > 0.0` 门控完全跳过——配置文件即阶段定义。

**参数共享而非复制**：两个优化器操作同一个 `BaseSceneModel.params` 张量存储。Stage 0 的 `trans`/`root_orient` 最优值被 Stage 1 直接继承（无需显式传递），只有 L-BFGS 的 Hessian 近似状态被重置。

**L-BFGS 优于 SGD**：全批次二阶梯度的准牛顿方法适合此类中规模优化（~10³ 参数），收敛速度远快于 SGD。`strong_wolfe` 线搜索保证了步长的充分下降和曲率条件。

---

## 第三部分：run_opt 输出产物总览

### 11. 输出目录结构

执行 `python run_opt.py data=video_vipe run_opt=True data.seq=demo1 is_static=False` 后，Hydra 在以下路径创建输出：

```
{log_root}/{data.type}-{data.split}/{YYYY-MM-DD}/{data.name}/
```

对于 demo 视频（`video-custom` 类型，seq=`demo1`，默认 track_ids=`all`，shot_idx=`0`），具体为：

```
../outputs/logs/video-custom/custom/2026-01-15/demo1-all-shot-0-0--1/
```

目录内部结构：

```
{out_dir}/
  │
  ├── .hydra/                           # Hydra 运行时配置快照
  │     config.yaml, overrides.yaml
  │
  ├── opt_log.txt                       # Logger 优化日志
  │
  ├── cameras.json                      # [生产] save_camera_json()
  │     └─ {rotation: (T,9), translation: (T,3), intrinsics: (T,4)}
  │     └─ 应用 Y-flip T=[[1,0,0],[0,-1,0],[0,0,-1]]
  │
  ├── track_info.json                   # [生产] save_track_info()
  │     └─ {tracks: {tid: {index, vis_mask}}, meta: {seq_interval, data_interval}}
  │
  ├── hamer/                            # [生产] save_input_poses() — HaMeR 原始预测
  │     └─ {seq}_000000_phalp_world_results.npz
  │         └─ {pose_body (B,T,15,3), trans (B,T,3), root_orient (B,T,3)}
  │
  ├── init/                             # [生产] save_initial_predictions() — 模型初始化后
  │     └─ {seq}_000000_init_world_results.npz
  │         └─ {pose_body, latent_pose, betas, trans, root_orient,
  │             is_right, init_body_pose, world_scale, cam_R, cam_t, intrins}
  │
  ├── root_fit/                         # [生产] Stage 0 RootOptimizer
  │     ├─ {seq}_000000_world_results.npz    ← 初始（含 HaMeR 输入）
  │     ├─ {seq}_000020_world_results.npz    ← iter 20
  │     ├─ {seq}_000050_world_results.npz    ← 最终 iter 50
  │     ├─ root_fit_params.pth               ← checkpoint（模型参数快照）
  │     ├─ root_fit_optim.pth                ← checkpoint（L-BFGS 状态）
  │     └─ {loss_name}.png                   ← 损失箱线图
  │
  ├── smooth_fit/                       # [生产] Stage 1 SmoothOptimizer
  │     ├─ {seq}_000000_world_results.npz    ← 初始
  │     ├─ {seq}_000020_world_results.npz
  │     ├─ ...
  │     ├─ {seq}_000300_world_results.npz    ← 最终 iter 300
  │     ├─ smooth_fit_params.pth
  │     ├─ smooth_fit_optim.pth
  │     └─ {loss_name}.png
  │
  ├── prior/                            # [生产] 仅 run_prior=True
  │     ├─ {seq}_000000_world_results.npz    ← HMP 精修后的世界空间结果
  │     └─ *.pkl                             ← HMP 内部中间产物
  │
  ├── {seq}_input.mp4                   # [消费产] save_input_frames() — 输入帧视频
  │
  ├── {seq}_{phase}_final_{iter}_src_cam.mp4  # [消费产] run_vis 渲染视频
  ├── {seq}_{phase}_final_{iter}_front.mp4
  ├── {seq}_{phase}_final_{iter}_above.mp4
  ├── {seq}_{phase}_final_{iter}_side.mp4
  ├── {seq}_{phase}_grid.mp4                  # [消费产] 2×2 网格视频
  │
  └── {phase}/                           # [消费产] OBJ 网格
        └─ {seq}_{iter}_meshes/
              ├─ 000000_0.obj            # 帧 t=0, 手 idx=0（左手）
              ├─ 000000_1.obj            # 帧 t=0, 手 idx=1（右手）
              ├─ 000001_0.obj
              └─ ...
```

### 12. 中间产物生产/消费链路

```
阶段                     生产者                             消费者
──────────────────────────────────────────────────────────────────────
[预处理]
images/{seq}/*.jpg        extract_frames.py                  HaMeR (Pass 1/3)
{seq}.pkl                 run.py (HaMeR Pass 3)              export_hamer.py
_mano.json                export_hamer.py                    dataset.py (load_mano_preds)
_keypoints.json           export_hamer.py                    dataset.py (load_keypoints_with_interp)
shot_idcs/{seq}.json      export_hamer.py                    dataset.py (shot 分割)
cameras.npz               VIPE / DROID-SLAM                  dataset.py (CameraData)

[数据加载]
obs_data dict             dataset.py  __getitem__            run_opt.py (optimizer)
cam_data dict             dataset.py  get_camera_data        run_opt.py (optimizer)

[优化阶段内]
cameras.json              save_camera_json()                 可视化 / 外部消费
track_info.json           save_track_info()                  可视化 / 调试
hamer/*.npz               save_input_poses()                 可视化对比
init/*.npz                save_initial_predictions()         可视化对比
root_fit/*.npz            RootOptimizer.save_results()       run_vis.py (可视化) / SmoothOptimizer (作为初始值——共享 params)
root_fit_params.pth       RootOptimizer.save_checkpoint()    RootOptimizer.load_checkpoint() (断点续跑)
smooth_fit/*.npz          SmoothOptimizer.save_results()     run_vis.py (可视化) / run_prior (HMP 输入)
smooth_fit_params.pth     SmoothOptimizer.save_checkpoint()  SmoothOptimizer.load_checkpoint()
prior/*.npz               run_prior() (HMP)                  run_vis.py (最终可视化)

[可视化]
{phase}_final_*.mp4       run_vis.py → animate_scene()      用户消费
{phase}/*_meshes/*.obj    run_vis.py → save_meshes_all()    外部 3D 工具 / Blender
{seq}_grid.mp4            run_vis.py → make_video_grid_2x2()用户消费
```

### 13. NPZ 文件内容约定

所有 `*_world_results.npz` 文件包含统一的字段集合：

| 键名 | 形状 | 出现阶段 | 含义 |
|------|------|---------|------|
| `trans` | (B, T, 3) | 全部 | **世界空间**手腕平移 |
| `root_orient` | (B, T, 3) | 全部 | **世界空间**手腕旋转（轴角） |
| `pose_body` | (B, T, 45) | 全部 | 手部姿态轴角（`latent2pose` 解码，15 关节 × 3） |
| `latent_pose` | (B, T, D) | init/root_fit/smooth_fit/prior | 优化潜变量（无 VPoser 时 D=45，与 pose_body 同值） |
| `betas` | (B, 10) | init/smooth_fit/prior | MANO 形状参数（RootOptimizer 阶段冻结，但包含在 npz 中） |
| `is_right` | (B, T) | 全部 | 手性（0=左手，1=右手） |
| `init_body_pose` | (B, T, 15, 3) | init | HaMeR 初始手指姿态（优化过程中保持，仅用于存档） |
| `world_scale` | (1, 1) | init/smooth_fit/prior（条件） | 全局尺度（仅 `opt_scale=True` 时存在） |
| `cam_R` | (B, T, 3, 3) | 全部 | 世界到相机旋转矩阵 |
| `cam_t` | (B, T, 3) | 全部 | 世界到相机平移向量 |
| `intrins` | (4,) | 全部 | `[fx, fy, cx, cy]`（标量，非逐帧——所有帧共享同一内参） |

**简化的 phalp（输入）NPZ**（`hamer/` 目录）仅含 `{pose_body, trans, root_orient}` 三项，用于可视化对比优化前后的差异。

**Prior 阶段额外字段**（HMP 输出）：
| 键名 | 形状 | 含义 |
|------|------|------|
| `decode_root` | (B, T, 3) | HMP 解码的根旋转 |
| `poses` | (B, T, 48) | HMP 局部旋转矩阵转轴角的完整姿态（48 = 16 关节 × 3，含根） |

### 14. 关键文件格式

| 文件 | 格式 | 读入代码 | 写入代码 |
|------|------|---------|---------|
| `*_world_results.npz` | `np.savez` | `load_result()` (`output.py`) | `save_results()` (`output.py`) |
| `*_params.pth` | `torch.save` | `StageOptimizer.load_checkpoint()` | `StageOptimizer.save_checkpoint()` |
| `*_optim.pth` | `torch.save` | 同上 | 同上 |
| `track_info.json` | `json.dump` | — | `save_track_info()` (`output.py`) |
| `cameras.json` | `json.dump` | — | `save_camera_json()` (`output.py`) |
| `opt_log.txt` | 纯文本 | — | `Logger` 类 |
| `{seq}_input.mp4` | MP4 视频 | — | `save_input_frames()` (`vis/output.py`) |
| `*_final_*.mp4` | MP4 视频 | — | `animate_scene()` (`vis/output.py`) |
| `*_meshes/*.obj` | Wavefront OBJ | —（外部消费） | `save_meshes_all()` / `vertices_to_trimesh()` |
