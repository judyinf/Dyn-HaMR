# Dyn-HaMR HMP 运动先验优化详解

本文档详细解析 HMP（Hand Motion Prior）的编码/解码架构、窗口化批处理策略和多阶段潜码优化流程。

---

## 1. 整体流程

HMP 在 `run_opt.py:166` 作为可选的后处理阶段调用：

```python
if cfg.run_prior and not os.path.exists(os.path.join(out_dir, 'prior')):
    run_prior(cfg, dataset, out_dir, device, ['smooth_fit'],
              obs_data, hand_model, cfg, cfg.data, os.path.join(out_dir, 'prior'))
```

内部流程：

```
fitting_prior()                          # fitting.py:1586
  │
  ├── 加载预训练 Architecture 模型        # model.load(optimal=True)
  ├── 创建 ForwardKinematicsLayer (FK)
  │
  ├── multi_stage_opt()                   # fitting.py:700+
  │   │
  │   ├── 加载 smooth_fit/*.npz 结果
  │   │
  │   ├── get_stage2_res()               # 运动学状态准备
  │   │     hand_pose(轴角) → rot6d + FK → pos/velocity/angular/global_xform
  │   │
  │   ├── 128帧窗口切分 / padding
  │   │
  │   ├── model.set_input(data)           # 将运动学数据喂入模型
  │   │
  │   ├── motion_reconstruction()         # fitting.py:531
  │   │   │
  │   │   ├── encode_local()   → z_l (1024)   ← 一次性编码
  │   │   ├── encode_global()  → z_g (256)
  │   │   │
  │   │   └── latent_optimization()       # fitting.py:948
  │   │       │
  │   │       ├── Stage 1: 优化 betas, trans, root_orient (z冻结)
  │   │       └── Stage 2: 额外优化 z_l (motion_prior λ=200)
  │   │
  │   └── 保存 prior/*.npz
```

---

## 2. 运动学状态准备：`get_stage2_res()`

代码：`HMP/fitting.py:184-263`

将 smooth_fit 输出（`pose_body (B,T,45)` 轴角 + `trans (B,T,3)`）转换为 HMP 编码器所需的完整运动学状态：

```
hand_pose (T, 16, 3) 轴角
    │
    ├── axis_angle_to_matrix(pose)         → rotmat (T, 16, 3, 3)
    │
    ├── matrix_to_rotation_6d(rotmat)      → rot6d (T, 16, 6)
    │     └── 6D连续旋转表示, 避免轴角的奇异性和旋转矩阵的冗余
    │
    ├── FK(rot6d)                          → pos (T, 16, 3)
    │     └── ForwardKinematicsLayer: 沿骨骼链累积局部变换
    │     └── global_xform (T, 16, 4, 4)  全局变换矩阵(取[:3,:3]→6D)
    │
    ├── estimate_angular_velocity(rotmat)  → angular (T, 16, 3)
    │     └── 数值差分: dR/dt → 角速度向量
    │
    ├── estimate_linear_velocity(pos)      → velocity (T, 16, 3)
    │     └── 数值差分: (p_{t+1} - p_{t-1}) / 2h  中心差分
    │
    ├── matrix_to_rotation_6d(root_orient) → root_orient (T, 6)
    │     └── rotmat[:, 0] → 6D: 手腕的全局方向
    │
    └── estimate_linear_velocity(trans)    → root_vel (T, 3)
          └── 手腕线速度
```

**`unified_orientation`**：在 `get_stage2_res` 中，`rotmat[:, 0]`（根旋转）被替换为单位矩阵，然后通过 FK 和 `root_rotation.repeat` 将根方向统一编码到 `global_pos` 中。这确保 HMP 学习的是手部关节的**相对运动**而非绝对方向——编码器看到的手部姿态是旋转归一化后的。

---

## 3. 编码器架构

### 3.1 `set_input()` — 数据注入

代码：`HMP/nemf/generative.py:136-140`

```python
def set_input(self, input):
    self.input_data = {
        k: v.float().to(self.device)
        for k, v in input.items()
        if k in ['pos', 'velocity', 'global_xform', 'angular',
                 'root_orient', 'root_vel', 'trans', 'contacts']
    }
    self.input_data['rotmat'] = rotation_6d_to_matrix(self.input_data['global_xform'])
```

### 3.2 `encode_local()` — 局部运动编码 → z_l

代码：`HMP/nemf/generative.py:147-177`

```
输入:
  pos          (B, T, 16, 3)   关节局部位置
  velocity     (B, T, 16, 3)   关节线速度
  global_xform (B, T, 16, 6)   关节全局方向 (6D)
  angular      (B, T, 16, 3)   关节角速度

处理:
  ├── 各分量分别归一化: (x - mean) / std (预计算训练集统计量)
  ├── 拼接: cat(pos, velocity, global_xform, angular) → (B, T, 16×(3+3+6+3)) = (B, T, 240)
  ├── reshape → (B, 240, T)        ← 通道×时间格式, 供1D卷积
  │
  └── LocalEncoder (prior.py:8)
        ├── 4层 SkeletonResidual/SkeletonConv
        │     channel_base=15 per joint
        │     每层 stride=2 时序下采样 (共16倍压缩)
        │
        └── nn.Linear → mu, logvar → z_l (1024)
              if lambda_kl != 0: z = reparameterize(mu, logvar)
              else: z = mu  ← 当前配置 lambda_kl=0, 取mu

输出: z_l (B, 1024)
```

`LocalEncoder` 的骨骼图卷积在关节维度上利用 MANO 的 kinematic tree 结构——相邻关节的特征通过 `SkeletonConv` 交换信息，`SkeletonPool` 沿骨骼链做层次化池化。

### 3.3 `encode_global()` — 全局轨迹编码 → z_g

代码：`HMP/nemf/generative.py:179-202`

```
输入:
  root_orient (B, T, 6)    手腕方向 (6D)
  root_vel    (B, T, 3)    手腕线速度 (可选, in_channels=9时拼接)

处理:
  ├── 归一化
  ├── reshape → (B, D, T)
  │
  └── GlobalEncoder (prior.py:85)
        ├── 4层 ResidualBlock (1D conv)
        │     128 → 256 → 512 → 512
        │     每层 stride=2 (16倍时序压缩)
        │
        └── nn.Linear → mu, logvar → z_g (256)

输出: z_g (B, 256)
```

### 3.4 编码的物理语义

| 分量 | 形状 | 编码信息 |
|------|------|---------|
| `pos` | (T,16,3) | 16个关节在规范坐标系中的3D位置 |
| `velocity` | (T,16,3) | 关节运动速度 (中心差分) |
| `global_xform` | (T,16,6) | 每个关节在世界系中的方向 (6D) |
| `angular` | (T,16,3) | 关节旋转角速度 (中心差分) |
| `root_orient` | (T,6) | 手腕全局方向 |
| `root_vel` | (T,3) | 手腕平移速度 |

**为什么编码运动学状态而非原始轴角**：轴角仅参数化旋转，缺乏显式的空间关系(pos)和时序动态(velocity/angular)。HMP 的图卷积编码器需要这些信号来学习骨骼运动的物理约束。

---

## 4. 解码器架构

### 4.1 `decode()` — 潜码 → 运动序列

代码：`HMP/nemf/generative.py:220-269`

```python
def decode(self, z_l, z_g, length, step=1):
    # 时间向量 t ∈ [-1, 1]
    t = torch.arange(0, length, step) / clip_length * 2 - 1  # (1, T)

    # NeMF 超网络: t + z_l + z_g → 运动序列
    local_motion, global_motion = self.field(t, z_l, z_g)
    # local_motion:  (B, T, 144)   144 = 16×6(rot6d) + 16×3(pos)
    # global_motion: (B, T, 6+)    6(root_orient) + 3(root_vel) + 1(height) + 8(contacts)

    # 提取旋转: 前 96 维 = 16关节 × 6D
    rot6d_recon = local_motion[:, :, :96] → (B, T, 16, 6)
    rotmat_recon = rotation_6d_to_matrix(rot6d_recon) → (B, T, 16, 3, 3)

    # FK 计算关节位置
    local_rotmat = FK.global_to_local(rotmat_recon)  # 全局旋转 → 局部旋转
    pos_recon, _ = FK(local_rotmat)                   # 局部旋转 → 关节3D坐标

    # 全局轨迹
    root_orient = global_motion[:, :, :6]  # 手腕方向 (6D)
    root_vel = global_motion[:, :, 6:9]    # 手腕速度
    root_height = global_motion[:, :, 9]   # 手腕高度
    trans = compute_trajectory(root_vel, root_height, origin, dt)
```

### 4.2 `NeuralMotionField` — 时间条件超网络

代码：`HMP/nemf/neural_motion.py`

```
输入:
  t        (B, T, 1)   时间 ∈ [-1, 1], positional encoding (bandwidth=7)
  z_l      (B, 1024)   局部潜码 (expand到 (B, T, 1024))
  z_g      (B, 256)    全局潜码 (注入全局层)

架构:
  11层 MLP with skip connections:
    前8层—局部层: 输入 = [t, z_l] + skip(原始[t,z_l])
    后3层—全局层: 注入 z_g
  FCBlock + LayerNorm (无Siren激活)

输出:
  local_output   (B, T, 144)   16×6(rot6d) + 16×3(pos)
  global_output  (B, T, 6+)    根方向 + 速度 + 高度 + 接触
```

**关键特性**：NeMF 是连续时间函数——`t` 可以是任意分辨率的（`step` 参数控制）。这允许 HMP 生成任意帧率的运动序列：只需改变 `length` 和 `step` 参数，无需重新训练。

---

## 5. 多阶段潜码优化

### 5.1 `latent_optimization()` — 优化流程

代码：`HMP/fitting.py:948`

```
输入:
  z_l (B, 1024)     ← encode_local() 的初始编码
  z_g (B, 256)      ← encode_global() 的初始编码
  target: 包含 cam_R, cam_t, cam_f, cam_center, is_right, joints2d

优化变量 (按阶段):

  Stage 1 (400 iters, lr=0.05):
    优化: betas, trans, root_orient
    冻结: z_l, z_g

  Stage 2 (400 iters, lr=0.05):
    优化: betas, trans, root_orient, z_l
    冻结: z_g

每次迭代:
  model.decode(z_l, z_g, length=128) → rotmat_recon, pos_recon, root_orient(6D)
    → rotation_6d_to_matrix(root_orient) → 手腕旋转矩阵
    → FK(local_rotmat) → joints3d
    → MANO(rotmat, betas, root_orient, trans) → joints3d, verts3d
    → cam_util.reproject(joints3d) → joints2d_pred
    →
    Loss =  reproj_loss(joints2d_pred, joints2d_obs)
          + orient_smooth_loss
          + trans_loss + trans_smooth_loss
          + bio_loss
          + [Stage 2] motion_prior_loss(z_l) * 200
```

### 5.2 运动先验损失

代码：`HMP/fitting.py:523-528`

```python
def motion_prior_loss(latent_motion_pred):
    loss = latent_motion_pred**2       # L2 → 0
    loss = torch.mean(loss)
    return loss
```

假设训练时 `z_l` 被正则化为标准正态分布 N(0, I)。优化时 L2 惩罚将 `z_l` 拉向原点（高概率区域），权重 λ=200（`hmp_config.yaml:stg2.lambda_motion_prior`）。

**效果**：`motion_prior_loss` 阻止 `z_l` 偏离训练分布太远——如果优化器试图将 `z_l` 推向分布外以过度拟合 2D 观测，200× 的正则化会将其拉回。这使得 HMP 在遮挡区域和有噪声的关键点场景下更鲁棒。

### 5.3 相比 L-BFGS 的差异

| 维度 | L-BFGS (RootOptimizer/SmoothOptimizer) | HMP (latent_optimization) |
|------|---------------------------------------|---------------------------|
| 优化器 | L-BFGS (strong_wolfe) | Adam (lr=0.05) |
| 优化变量 | MANO参数直接优化 (trans, root_orient, betas, latent_pose) | 潜码 z_l (1024维) + betas/trans/root_orient |
| 运动表示 | 逐帧独立的 latent_pose (45维/帧) | 128帧窗口共享一个 z_l (1024维) |
| 时序建模 | joints3d_smooth loss (局部平滑) | NeMF 连续时间函数 (全局时序一致性) |
| 先验 | pose_prior: L2→HaMeR初始值 | motion_prior: L2→单位高斯 (训练分布) |
| 迭代次数 | 50+300 | 400+400 (可配) |
| 批处理 | 单序列 (B, T) | 128帧窗口堆叠 (N_chunks, 128) |

---

## 6. 窗口化批处理

### 6.1 main 分支：chunk 堆叠

代码：`HMP/fitting.py:763-787`

```
full sequence (T帧)
  │
  ├── T > 128: torch.split(v, 128) → [chunk0, chunk1, ...]
  │     最后一个chunk < 128 → 重复末帧padding到128
  └── T < 128: padding到128
  │
  ▼
torch.stack([chunk0, chunk1, ...], dim=0) → (N_chunks, 128, ...)
  │  一次性 model.decode(z_l, z_g, length=128)
  │  padding帧参与所有loss (无掩码排除)
  │  batch_consistency loss: 相邻chunk边界位置的输出一致性
```

### 6.2 窗口数量对编码的影响

**每个 128 帧窗口独立编码**：`encode_local()` 和 `encode_global()` 接收的输入是 `(N_chunks, 128, ...)` 形状——所有窗口同时编码，产生 `z_l (N_chunks, 1024)` 和 `z_g (N_chunks, 256)`。

这意味着一个 500 帧序列（切为 4 个 128 帧 chunk）会产生 4 组独立的 `(z_l, z_g)`——HMP **不做跨窗口的编码一致性约束**（除 batch_consistency loss 外）。

---

## 7. 配置文件

`HMP/hmp_config.yaml` 的核心配置：

```yaml
data:
  fps: 30
  clip_length: 128
  normalize: [pos, velocity, global_xform, angular, root_orient, root_vel]
  root_transform: true
  up: y

nemf:
  local_output: 144          # 16×6(rot6d) + 16×3(pos)
  global_output: 6            # root_orient (6D)
  in_channels: 1455           # t(1) + z_l(1024) + z_g(256) + pos_enc(174) 等
  hidden_size: 512
  n_blocks: 11                # 8 local + 3 global

local_prior:
  z_dim: 1024

global_prior:
  z_dim: 256
  in_channels: 6

lambda_kl: 0                  # 无KL散度，潜码取mu

stg1:
  niters: 400
  lr: 0.05
  lambda_reproj: 2.0

stg2:
  niters: 400
  lr: 0.05
  lambda_motion_prior: 200    # z_l 正则化权重
  lambda_reproj: 2.0
```

---

## 8. 输出产物

`prior/` 目录下的 NPZ 文件在标准字段基础上增加了两个 HMP 专属字段：

| 键名 | 形状 | 含义 |
|------|------|------|
| `decode_root` | (B, T, 3) | HMP 解码的根旋转 (轴角) |
| `poses` | (B, T, 48) | HMP 局部旋转矩阵转轴角的完整姿态 (16关节×3, 含根) |
