# Dyn-HaMR Optimization and Losses

本文档说明 `run_pose3d_hand_stages.py` 当前三阶段 pipeline 的数据流、优化器差异、损失项实现位置、数学形式和主要超参数。这里的“三阶段”指外层 `root -> smooth -> prior`；`prior` 阶段内部还有 HMP 自己的 `stg1/stg2/stg3`。

## 1. 数据流

主入口是 `dyn-hamr/run_pose3d_hand_stages.py::run_from_cfg()`。默认配置 `data.stage=all` 时，执行顺序为：

```text
.pose3d_hand + track_info/keypoints
  -> Pose3DHandStageData
  -> root_fit
  -> smooth_fit
  -> HMP prior
  -> final .pose3d_hand
```

核心输入：

- `data.pose3d_hand`：左右手 MANO 参数、`pred_valid`、`relative_motion`、`slam_data`、`fps`。
- `data.track_info`：检测框、handedness、track 信息。
- `data.keypoints_npy`：2D hand keypoints；不存在且 `extract_keypoints=true` 时，由 ViTPose 从视频帧生成。

`Pose3DHandStageData.obs_data()` 将输入整理为优化观测：

- `joints2d`: `(B,T,J,3)`，2D 关键点 `(x,y,confidence)`。
- `vis_mask`: `(B,T)`，`>=0` 的帧参与优化。
- `is_right`: `(B,T)`，左手为 0，右手为 1。
- `init_body_pose`: `(B,T,15,3)`，MANO hand pose axis-angle。
- `init_body_shape`: MANO betas。
- `init_root_orient`: `(B,T,3)`，MANO global orientation。
- `init_trans`: `(B,T,3)`，MANO translation。
- `init_latent_pose`: 若不使用 VPoser，等价于 `init_body_pose.reshape(B,T,45)`。

`Pose3DHandStageData.camera_data()` 从 `slam_data` 构造：

- `cam_R`: world-to-camera rotation。
- `cam_t`: world-to-camera translation。
- `intrins = [fx, fy, cx, cy]`。
- `slam_scale`：传入 `world_scale` 初始化。

`prepare_cfg()` 会强制 `cfg.model.opt_cams=False`，所以当前主脚本不优化相机旋转、平移增量或焦距；`cfg.model.opt_scale=True` 时，`smooth` 阶段会优化 `world_scale`。

## 2. 优化器对比

### 2.1 LBFGS: root/smooth 阶段

实现位置：

- `dyn-hamr/optim/optimizers.py::StageOptimizer.__init__`
- `dyn-hamr/optim/optimizers.py::StageOptimizer.optim_step`
- `dyn-hamr/run_pose3d_hand_stages.py::run_root_or_smooth`

root 和 smooth 都使用 PyTorch `torch.optim.LBFGS`：

```python
torch.optim.LBFGS(
    opt_params,
    max_iter=lbfgs_max_iter,
    lr=lr,
    line_search_fn="strong_wolfe",
)
```

默认超参数来自 `dyn-hamr/confs/optim.yaml`：

- `lr = 1.0`
- `lbfgs_max_iter = 20`
- `line_search_fn = "strong_wolfe"`
- `root.num_iters = 50`
- `smooth.num_iters = 300`

LBFGS 是近似二阶优化方法。它不会只沿负梯度走固定一步，而是利用最近若干步的参数差和梯度差近似逆 Hessian，构造搜索方向 `p_k`。随后 strong Wolfe line search 在该方向上多次试探步长 `alpha`，寻找同时满足 sufficient decrease 和 curvature condition 的点：

$$
\mathbf{x}_{k+1}
= \mathbf{x}_{k}
+ \alpha_k \mathbf{p}_k,
\qquad
\mathbf{p}_k \approx -\mathbf{H}_k^{-1}\nabla L(\mathbf{x}_k).
$$

其中 $\mathbf{x}_k$ 是第 $k$ 次外层优化时的参数向量，$\nabla L(\mathbf{x}_k)$ 是 loss 对参数的梯度，$\mathbf{H}_k^{-1}$ 是 LBFGS 用历史梯度差近似出的逆 Hessian，$\mathbf{p}_k$ 是搜索方向，$\alpha_k$ 是 line search 找到的步长。

这里有两层迭代：

- 外层 `num_iters`：`run_root_or_smooth()` 中显式调用 `optimizer.optim_step()` 的次数。
- LBFGS 内层 `lbfgs_max_iter`：每次 `optim.step(closure)` 里，PyTorch LBFGS 最多执行的内部迭代/闭包评估次数。

因为 LBFGS 依赖 closure，同一个外层 step 内可能多次 forward/backward。项目里 loss 日志用字典存储同一外层迭代下的多次 loss 样本，也是因为 LBFGS 会多次计算 closure。

适用性：

- 优点：小到中等规模、batch 较小、确定性目标上通常收敛快；适合直接优化 MANO 参数这种非网络训练问题。
- 缺点：每一步更贵，line search 可能多次 forward/backward；对显存和 wall time 更敏感。

### 2.2 Adam + StepLR: HMP prior 阶段

实现位置：

- `dyn-hamr/HMP/fitting.py::latent_optimization`
- `dyn-hamr/HMP/fitting.py::optim_step`
- `dyn-hamr/HMP/hmp_config.yaml`

HMP prior 内部使用 Adam：

```python
optimizer = torch.optim.Adam(opt_params, lr=stg_conf.lr)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    args.scheduler.step_size,
    args.scheduler.gamma,
)
```

默认超参数：

- `scheduler.step_size = 200`
- `scheduler.gamma = 0.7`
- `stg1.lr = 0.05`, `stg1.niters = 400`
- `stg2.lr = 0.05`, `stg2.niters = 400`
- `stg3.lr = 0.0`, `stg3.niters = 0`

Adam 是一阶自适应优化器，维护梯度一阶矩和二阶矩：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \\
x_{t+1} &= x_t - \eta_t \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\end{aligned}
$$

其中 $x_t$ 是第 $t$ 次迭代的优化变量，$g_t=\nabla_x L(x_t)$ 是当前梯度，$m_t$ 是梯度一阶矩估计，$v_t$ 是梯度平方的二阶矩估计，$\hat m_t,\hat v_t$ 是 bias-corrected 版本，$\eta_t$ 是当前学习率，$\epsilon$ 用于避免除零。

StepLR 每次迭代后调用 `scheduler.step()`，每隔 `step_size` 次将学习率乘以 `gamma`：

$$
\eta_t = \eta_0 \gamma^{\left\lfloor t / s \right\rfloor}.
$$

其中 $\eta_0$ 是初始学习率，$\gamma$ 是衰减倍率，$s$ 是 `step_size`，$\lfloor\cdot\rfloor$ 表示向下取整。

对默认 HMP stage，学习率大致为：

- 第 0-199 次：`0.05`
- 第 200-399 次：`0.035`

适用性：

- 优点：每次迭代通常只需一次 forward/backward，计算可控；对 HMP latent `z_l` 这类高维变量更常用。
- 缺点：步长调度较粗，迭代次数多时可能继续追噪声；没有 line search，单步不会自动验证目标下降。

### 2.3 计算和时间复杂度

记一次完整 forward/backward 的代价为：

```text
C(B,T,J,V)
```

其中：

- `B`: hand/track 数，当前通常为 2；HMP window 内通常按单手处理，`B=1`。
- `T`: 帧数；HMP 每个 window 默认 `clip_length=128`。
- `J`: 关节数；MANO hand joints 约 16/21 个，取决于分支。
- `V`: MANO vertices 数，约 778。

一次 forward/backward 的主要成本来自：

- MANO 前向：约随 `B*T*V` 增长。
- 2D 投影和 joint loss：约 `O(B*T*J)`。
- 平滑 loss：约 `O(B*T*J)` 或 `O(B*T)`。
- penetration/contact 这类 mesh 交互项若启用，会显著更贵，可能接近顶点/面片两两关系的高阶成本。

root/smooth 的 LBFGS 总成本可粗略写成：

```text
O(N_outer * M_lbfgs * C)
```

其中 `N_outer` 是 `num_iters`，`M_lbfgs <= lbfgs_max_iter`，还会受 line search 闭包评估次数影响。默认上界粗略为：

- root：`50 * 20 * C`
- smooth：`300 * 20 * C`

HMP prior 的 Adam 总成本可粗略写成：

```text
O(N_hands * N_windows * (N_stg1 + N_stg2) * C_window)
```

默认每个 window：

```text
(400 + 400) * C(B=1,T=128)
```

如果启用 hand/window 并行，总计算量不变，但 wall time 会被 GPU/CPU 资源、进程启动、模型加载和显存竞争影响。

## 3. Root/Smooth 阶段损失项

root 和 smooth 的 optimizer 定义在 `dyn-hamr/optim/optimizers.py`：

- `RootOptimizer`: 优化 `trans`, `root_orient`。
- `SmoothOptimizer`: 优化 `trans`, `root_orient`, `betas`, `latent_pose`；若 `model.opt_scale=True` 还优化 `world_scale`。

loss 实现在 `dyn-hamr/optim/losses.py`：

- `RootLoss.forward()`
- `SMPLLoss.forward()`
- `Joints2DLoss.forward()`
- `joints3d_smooth_loss()`
- `depth_constraint_loss()`
- `pose_prior_loss()`
- `shape_prior_loss()`

权重来自 `dyn-hamr/confs/optim.yaml::optim.loss_weights`。列表第 0 列用于 root，第 1 列用于 smooth。第 2 列是旧 motion stage 风格权重，当前外层 HMP prior 不直接使用它作为主 loss。

### 3.1 2D 重投影损失

代码位置：

- `RootLoss.forward()` 中调用 `cam_util.reproject()` 和 `self.joints2d_loss(...)`
- `Joints2DLoss.forward()`

变量：

- `X_{b,t,j}`: MANO 预测 3D joint，来自 `pred_data["joints3d_op"]`。
- `R_t, t_t`: world-to-camera extrinsics。
- `f_t=(fx,fy)`, `c_t=(cx,cy)`: camera intrinsics。
- `u_{b,t,j}`: 观测 2D keypoint。
- `c_{b,t,j}`: keypoint confidence。
- `m_{b,t}`: 有效帧 mask。
- `s_{b,t}`: 2D hand scale，由观测 keypoints bbox 对角线估计。

投影：

$$
\mathbf{Y}_{b,t,j} = \mathbf{R}_t \mathbf{X}_{b,t,j} + \mathbf{t}_t,
$$

$$
\hat{\mathbf{u}}_{b,t,j}
= \mathbf{f}_t \odot
\frac{\mathbf{Y}_{b,t,j,xy}}{Y_{b,t,j,z}}
+ \mathbf{o}_t.
$$

其中 $\mathbf{X}_{b,t,j}\in\mathbb{R}^3$ 是 world space 的第 $b$ 条手轨迹、第 $t$ 帧、第 $j$ 个 MANO joint；$\mathbf{R}_t,\mathbf{t}_t$ 是 world-to-camera 外参；$\mathbf{Y}_{b,t,j}$ 是 camera space joint；$\mathbf{f}_t=(f_x,f_y)$ 是焦距；$\mathbf{o}_t=(c_x,c_y)$ 是主点；$\odot$ 表示逐元素乘法；$\hat{\mathbf{u}}_{b,t,j}\in\mathbb{R}^2$ 是投影到图像平面的预测 2D 点。

当前 root/smooth 的 `Joints2DLoss` 会先做尺度归一化：

$$
s_{b,t}
= \left\|
\max_j \mathbf{u}_{b,t,j}
- \min_j \mathbf{u}_{b,t,j}
\right\|_2,
\qquad
\mathbf{e}_{b,t,j}
= 100 \cdot \frac{\hat{\mathbf{u}}_{b,t,j}-\mathbf{u}_{b,t,j}}{\operatorname{clip}(s_{b,t},5,1000)}.
$$

其中 $\mathbf{u}_{b,t,j}$ 是观测 2D keypoint，$s_{b,t}$ 是由观测 keypoints 的 2D bbox 对角线估计的手部尺度，`clip` 对应代码中的尺度下限和上限保护。乘以 100 是把归一化后的误差拉回到较稳定的数值范围，使 `joints2d_sigma=100` 在归一化坐标里有意义。

再用 GMoF robust loss：

$$
\rho_\sigma(\mathbf{e})
= \frac{\sigma^2 \|\mathbf{e}\|^2}{\sigma^2+\|\mathbf{e}\|^2},
$$

$$
L_{\mathrm{2d}}
= \operatorname{mean}_{(b,t,j)\in\mathcal{M}}
\left[
q_{b,t,j}^2 \rho_{100}(\mathbf{e}_{b,t,j})
\right].
$$

其中 $\rho_\sigma$ 是 Geman-McClure robust penalty；$q_{b,t,j}$ 是 2D keypoint confidence；$\mathcal{M}$ 是由 `vis_mask >= 0` 选出的有效帧和 joint 集合；平方置信度会降低低置信度检测点对梯度的影响。

默认权重：

- root: `joints2d = 10000`
- smooth: `joints2d = 10000`

### 3.2 3D joint 平滑损失

代码位置：

- `RootLoss.forward()`
- `joints3d_smooth_loss()`

变量：

- `X_{b,t,j}`: 预测 3D joint。
- `s_{b,t}`: 3D hand scale，由预测 3D joints bbox 对角线估计。
- `m_{b,t}`: 有效帧 mask。

尺度归一化：

$$
r_{b,t}
= \left\|
\max_j \mathbf{X}_{b,t,j}
- \min_j \mathbf{X}_{b,t,j}
\right\|_2,
\qquad
\mathbf{X}'_{b,t,j}
= \frac{\mathbf{X}_{b,t,j}}{\operatorname{clip}(r_{b,t},0.001,1.0)}.
$$

其中 $r_{b,t}$ 是由预测 3D joints 的 3D bbox 对角线估计的手部尺度；$\mathbf{X}'$ 是尺度归一化后的 joint。归一化的目的不是改变几何输出，而是让不同尺度或 `world_scale` 下的平滑 loss 数值可比，避免大尺度序列天然产生更大的平滑惩罚。

平滑项：

$$
\mathcal{A}
= \left\{(b,t)\mid m_{b,t}=1,\ m_{b,t-1}=1,\ t>0\right\},
$$

$$
L_{\mathrm{smooth3d}}
= \frac{1}{2}
\sum_j
\operatorname{mean}_{(b,t)\in\mathcal{A}}
\left[
\left\|
\mathbf{X}'_{b,t,j}
- \mathbf{X}'_{b,t-1,j}
\right\|_2^2
\right].
$$

其中 $m_{b,t}$ 是有效帧 mask；$\mathcal{A}$ 只包含连续两帧都有效的相邻帧对；$\frac{1}{2}$ 是平方误差常用系数，使导数从 $2e$ 简化为 $e$ 的尺度。

默认权重：

- root: `joints3d_smooth = 1000`
- smooth: `joints3d_smooth = 10000`

### 3.3 Depth constraint

代码位置：

- `RootLoss.forward()`
- `depth_constraint_loss()`

变量：

- `Y_{b,t,j,z}`: joint 在 camera space 的深度。
- `min_depth=0.0`
- `max_depth=999`

公式：

$$
L_{\mathrm{depth}}
= 100 \sum_{b,t,j}\operatorname{ReLU}(-Y_{b,t,j,z})^2
+ \sum_{b,t,j}\operatorname{ReLU}(z_{\min}-Y_{b,t,j,z})^2
+ \sum_{b,t,j}\operatorname{ReLU}(Y_{b,t,j,z}-z_{\max})^2.
$$

其中 $Y_{b,t,j,z}$ 是 joint 的 camera-space 深度，$z_{\min}=0$，$z_{\max}=999$。$\operatorname{ReLU}(a)=\max(a,0)$，所以 $\operatorname{ReLU}(-Y_z)$ 只在 $Y_z<0$ 时非零，作用是惩罚落到相机后方的 joints。代码又给这一项乘以 100，使“相机后方”比普通过近/过远更严重。

默认权重：

- root: `depth_constraint = 100`
- smooth: `depth_constraint = 100`

这个项用于阻止手跑到相机后方。因为 `max_depth=999` 很大，实际主要约束通常是 `Y_z >= 0`。

### 3.4 Smooth 阶段 pose prior

代码位置：

- `SMPLLoss.forward()`
- `pose_prior_loss()`

变量：

- `z_{b,t}`: 当前优化的 `latent_pose`。
- `z^0_{b,t}`: 初始 `init_latent_pose`，来自输入 `.pose3d_hand` 的 MANO pose；不使用 VPoser 时就是 45 维 hand pose 展平。

公式：

$$
L_{\mathrm{pose}}
= \sum_{(b,t)\in\mathcal{M}}
\left\|
\mathbf{z}_{b,t}
- \mathbf{z}^{0}_{b,t}
\right\|_2^2.
$$

其中 $\mathbf{z}_{b,t}$ 是当前优化的 `latent_pose`，$\mathbf{z}^{0}_{b,t}$ 是从原始 MANO hand pose 得到的初始 latent pose，$\mathcal{M}$ 是有效帧集合。

默认权重：

- smooth: `pose_prior = 1`

注意：当前实现不是把 pose 拉向标准正态零均值，而是拉向初始 HaMeR/MANO 预测，目的是防止 2D loss 把手型拉崩。

### 3.5 Smooth 阶段 shape prior

代码位置：

- `SMPLLoss.forward()`
- `shape_prior_loss()`

变量：

- `beta_b`: MANO shape parameters。
- `nsteps`: 序列长度。

公式：

$$
L_{\mathrm{shape}}
= T \sum_b \left\|\boldsymbol{\beta}_b\right\|_2^2.
$$

其中 $\boldsymbol{\beta}_b$ 是第 $b$ 条手轨迹的 MANO shape 参数，$T$ 是序列长度 `nsteps`。MANO 的 shape space 通常以零向量作为平均形状，直接约束 $\|\beta\|_2^2$ 等价于高斯零均值先验，避免优化为了追 2D keypoints 把手掌比例、手指长度推到极端。

默认权重：

- smooth: `shape_prior = 0.05`

### 3.6 默认启用情况

按当前 `confs/optim.yaml`，root 默认主要启用：

```text
10000 * L_2d
+ 1000 * L_smooth3d
+ 100 * L_depth
```

smooth 默认主要启用：

```text
10000 * L_2d
+ 10000 * L_smooth3d
+ 100 * L_depth
+ 1 * L_pose
+ 0.05 * L_shape
```

`penetration`, `bio`, `joints3d`, `verts3d`, `points3d` 等配置存在，但默认权重为 0 或当前 `obs_data` 不提供对应观测，因此通常不贡献 loss。

## 4. HMP Prior 阶段损失项

外层入口：

- `run_pose3d_hand_stages.py::run_prior_stage`
- `HMP/fitting.py::fitting_prior`
- `HMP/fitting.py::multi_stage_opt`

HMP 内部窗口优化入口：

- `HMP/fitting.py::_optimize_hand_current_process`
- `HMP/fitting.py::_optimize_window_current_process`
- `HMP/fitting.py::motion_reconstruction`
- `HMP/fitting.py::latent_optimization`
- `HMP/fitting.py::optim_step`

默认 HMP 配置来自 `dyn-hamr/HMP/hmp_config.yaml`：

- `clip_length = 128`
- `overlap_len = 16`
- `scheduler.step_size = 200`
- `scheduler.gamma = 0.7`
- `stg1.niters = 400`
- `stg2.niters = 400`
- `stg3.niters = 0`

### 4.1 HMP stg1

默认优化变量：

```text
betas, trans, root_orient
```

默认主要权重：

```text
lambda_reproj = 0.05
lambda_bio = 1
lambda_trans = 10
lambda_trans_smooth = 10
lambda_orient_smooth = 1
betas_prior = 10
lambda_motion_prior = 0
```

### 4.2 HMP stg2

默认优化变量：

```text
betas, trans, root_orient, z_l
```

默认主要权重：

```text
lambda_reproj = 0.05
lambda_bio = 1
lambda_trans = 10
lambda_trans_smooth = 10
lambda_orient_smooth = 1
betas_prior = 10
lambda_motion_prior = 200
```

`z_l` 是 HMP local latent code。stg2 开启 motion prior，用来约束 latent 不要偏离 HMP 的运动先验分布。

### 4.3 HMP stg3

默认：

```text
niters = 0
lr = 0
```

因此当前配置下不会实际执行。

### 4.4 HMP 2D 重投影损失

代码位置：

- `HMP/fitting.py::optim_step`
- `HMP/fitting_utils.py::joints2d_loss`

变量：

- `X_{t,j}`: HMP decode 后经 MANO 得到的 3D joint。
- `R_t, t_t, f_t, c_t`: camera parameters。
- `u_{t,j}`: 观测 2D keypoint。
- `conf_{t,j}`: 2D keypoint confidence。

投影：

$$
\mathbf{Y}_{t,j} = \mathbf{R}_t\mathbf{X}_{t,j}+\mathbf{t}_t,
\qquad
\hat{\mathbf{u}}_{t,j}
= \mathbf{f}_t \odot
\frac{\mathbf{Y}_{t,j,xy}}{Y_{t,j,z}}
+ \mathbf{o}_t.
$$

符号含义与 root/smooth 相同，但这里通常是 HMP 单手 window 内的数据，$\mathbf{X}_{t,j}$ 来自 HMP decode 后再经 MANO 得到的 joints。

loss：

$$
L_{\mathrm{reproj}}^{\mathrm{HMP}}
= \operatorname{mean}_{t,j}
\left[
q_{t,j}^2
\rho_{40}(\hat{\mathbf{u}}_{t,j}-\mathbf{u}_{t,j})
\right].
$$

与 root/smooth 不同，HMP 这里直接在像素残差上计算 GMoF，不除以 hand scale；$\sigma=40$ 因此是像素尺度上的 robust transition 参数。

默认权重：

- stg1: `lambda_reproj = 0.05`
- stg2: `lambda_reproj = 0.05`

### 4.5 HMP bio loss

代码位置：

- `HMP/fitting.py::optim_step`
- `HMP/fitting_utils.py::BMCLoss`

公式上是三类手部生物力学约束加权和：

$$
L_{\mathrm{bio}}
= L_{\mathrm{bone\_length}}
+ L_{\mathrm{root\_bone}}
+ L_{\mathrm{joint\_angle}}.
$$

实现细节：

- `L_bone_length`: 骨长落在统计上下界之外时的 interval loss。
- `L_root_bone`: 根部骨骼曲率和角度范围约束。
- `L_joint_angle`: 关节角落在可行凸包/范围之外时的距离惩罚。

默认权重：

- stg1: `lambda_bio = 1`
- stg2: `lambda_bio = 1`

### 4.6 HMP trans loss

代码位置：

- `HMP/fitting.py::L_trans`
- `HMP/fitting.py::optim_step`

变量：

- `tau_t`: 优化中的 global translation。
- `tau^0_t`: prior 输入目标 translation，来自 smooth 结果。

默认 `l1_loss=true`，因此：

$$
L_{\mathrm{trans}}
= \operatorname{mean}_t
\left[
\left\|
\boldsymbol{\tau}_t-\boldsymbol{\tau}^{0}_t
\right\|_1
\right].
$$

其中 $\boldsymbol{\tau}_t$ 是优化中的 root/global translation，$\boldsymbol{\tau}^{0}_t$ 是 smooth 阶段输出传入 HMP 的目标 translation。

若配置改为 `l1_loss=false`，则使用 MSE。

默认权重：

- stg1: `lambda_trans = 10`
- stg2: `lambda_trans = 10`

### 4.7 HMP translation smoothness

代码位置：

- `HMP/fitting.py::optim_step`
- `HMP/nemf/losses.py::pos_smooth_loss`

变量：

- `tau_t`: global translation。

公式：

$$
\mathbf{v}_t
= 30\left(\boldsymbol{\tau}_t-\boldsymbol{\tau}_{t-1}\right),
\qquad
L_{\mathrm{trans\_sm}}
= \frac{1}{2}\operatorname{mean}_t
\left[
\left\|\mathbf{v}_t\right\|_2^2
\right].
$$

这里的 30 来自 `HMP/nemf/losses.py::pos_smooth_loss` 的固定 FPS 速度尺度：相邻帧位移乘以 30，近似转换为每秒速度。$\frac{1}{2}$ 是平方误差的常用缩放，主要让梯度尺度更自然。

默认权重：

- stg1: `lambda_trans_smooth = 10`
- stg2: `lambda_trans_smooth = 10`

### 4.8 HMP root orientation smoothness

代码位置：

- `HMP/fitting.py::optim_step`
- `HMP/nemf/losses.py::rot_smooth_loss`

变量：

- `R_t`: root orientation rotation matrix。
- `d_geo(R_t, R_{t-1})`: SO(3) geodesic distance。

公式：

$$
d_{\mathrm{geo}}(\mathbf{R}_t,\mathbf{R}_{t-1})
= \arccos\left(
\frac{\operatorname{tr}(\mathbf{R}_t\mathbf{R}_{t-1}^{\top})-1}{2}
\right),
$$

$$
\omega_t
= 30\,d_{\mathrm{geo}}(\mathbf{R}_t,\mathbf{R}_{t-1}),
\qquad
L_{\mathrm{orient\_sm}}
= \frac{1}{2}\operatorname{mean}_t
\left[
\omega_t^2
\right].
$$

其中 $d_{\mathrm{geo}}$ 是 SO(3) 旋转矩阵之间的测地距离，单位是弧度；乘以 30 后可理解为角速度尺度。

默认权重：

- stg1: `lambda_orient_smooth = 1`
- stg2: `lambda_orient_smooth = 1`

### 4.9 HMP motion prior

代码位置：

- `HMP/fitting.py::motion_prior_loss`
- `HMP/fitting.py::optim_step`

变量：

- `z_l`: HMP local latent code。

公式：

$$
L_{\mathrm{motion\_prior}}
= \operatorname{mean}
\left[
\left\|
\mathbf{z}^{l}
\right\|_2^2
\right].
$$

其中 $\mathbf{z}^{l}$ 是 HMP local latent code。HMP prior 将零附近视为更常见、更保守的 latent 区域，因此该项抑制异常运动 latent。

默认权重：

- stg1: `lambda_motion_prior = 0`
- stg2: `lambda_motion_prior = 200`

### 4.10 HMP betas prior

代码位置：

- `HMP/fitting.py::optim_step`

变量：

- `beta`: 当前优化 shape。
- `mean_beta`: smooth 输入中 betas 的均值。

公式：

$$
L_{\mathrm{betas\_prior}}^{\mathrm{HMP}}
= \operatorname{mean}
\left[
\left\|
\boldsymbol{\beta}
- \bar{\boldsymbol{\beta}}
\right\|_2^2
\right].
$$

其中 $\bar{\boldsymbol{\beta}}$ 是输入 smooth 结果里 betas 的均值。HMP 这里不是拉向零均值平均手，而是拉向 smooth 输入的平均 shape，目的是在 prior refinement 中保持个体手型一致。

默认权重：

- stg1: `betas_prior = 10`
- stg2: `betas_prior = 10`

## 5. Root/Smooth 与 HMP 的关键区别

### 5.1 2D 重投影损失区别

root/smooth：

- 代码：`optim/losses.py::Joints2DLoss`
- 使用观测 hand bbox 尺度归一化误差。
- `sigma=100`
- 外部权重大：默认 `joints2d=10000`
- 目标是强力把 MANO 手对齐到 2D keypoints，同时依赖 depth 和 smooth 项防止退化。

尺度归一化的具体目的：

- 消除图像中手部大小对 loss 数值的直接影响。近处的大手和远处的小手即使像素误差不同，也可以按“相对手宽/手长”比较。
- 减少焦距、裁剪尺度、slam/world scale 改变带来的 loss 尺度变化。
- 让 `sigma=100` 工作在归一化误差空间，而不是原始像素空间。代码中误差除以 hand scale 后乘 100，因此 $e=100$ 大致表示误差达到一个手部 bbox 尺度的量级。

HMP prior：

- 代码：`HMP/fitting_utils.py::joints2d_loss`
- 不做 hand scale 归一化，直接在像素误差上使用 GMoF。
- `sigma=40`
- 外部权重小：默认 `lambda_reproj=0.05`
- 目标是 refinement，不是完全重新拟合；同时受 translation、orientation smooth、bio、motion prior 共同约束。

直接像素误差的影响：

- 大手或近景手的同样相对误差会产生更大的像素残差，因此对 HMP reprojection loss 影响更大。
- `sigma=40` 直接表示像素尺度上的 robust transition；误差远大于 40 像素时，GMoF 梯度会逐渐饱和，异常 keypoint 的影响被压低。
- HMP 已经接收 smooth 结果作为初始化和目标，不需要像 root/smooth 一样强力用 2D loss 重新拉回全局位置，所以 reprojection 权重更小。

不同 sigma 的影响：

$$
\rho_\sigma(e)=\frac{\sigma^2 e^2}{\sigma^2+e^2}.
$$

当 $|e|\ll\sigma$ 时，$\rho_\sigma(e)\approx e^2$，近似 L2；当 $|e|\gg\sigma$ 时，$\rho_\sigma(e)\to\sigma^2$，异常残差被截断。更大的 $\sigma$ 让大误差仍保持较大梯度，更小的 $\sigma$ 更早进入饱和、更鲁棒但也更容易忽略远距离误差。root/smooth 的 `sigma=100` 配合尺度归一化，HMP 的 `sigma=40` 对应原始像素误差，两者不能只按数值大小直接比较。

### 5.2 平滑损失区别

root/smooth：

- 主要是 `joints3d_smooth_loss`。
- 对 3D joints 按每帧手尺度归一化后做相邻帧差分。
- 直接约束输出 MANO joints 的时间变化。
- 虽然 root 阶段只优化 `trans/root_orient`，loss 仍用所有 3D joints 计算。原因是所有 joints 都由同一个根平移和根旋转刚性带动；用整只手的 joints 衡量平滑，比只看 wrist/root translation 更能惩罚根旋转造成的指尖大幅跳动。

HMP prior：

- 主要是 `trans_smooth` 和 `orient_smooth`。
- `trans_smooth` 用 `30 * delta_trans` 表示速度。
- `orient_smooth` 用旋转矩阵 geodesic distance 表示角速度。
- HMP 默认没有开启 `lambda_j3d_smooth`，也没有使用 root/smooth 那种尺度归一化 joint smooth。
- HMP 的平滑直接作用在被优化的根节点 translation 和 orientation 上，而不是通过所有 joints 间接度量根运动。这样更贴合 HMP motion prior 的变量设计：姿态由 latent decoder 管，根轨迹连续性由 `trans_smooth/orient_smooth` 管。

HMP 中乘以 30 的原因是把相邻帧差分换算成近似速度：

$$
\mathbf{v}_t \approx \frac{\boldsymbol{\tau}_t-\boldsymbol{\tau}_{t-1}}{\Delta t},
\qquad \Delta t \approx \frac{1}{30}.
$$

所以代码写作 $30(\tau_t-\tau_{t-1})$。这假设或近似使用 30 FPS 的速度尺度。乘以 $\frac{1}{2}$ 是平方能量常用写法：

$$
\frac{\partial}{\partial e}\left(\frac{1}{2}e^2\right)=e,
$$

它不改变最优点，只改变梯度和 loss 的常数尺度。

### 5.3 HMP 阶段的本质区别

root/smooth 是直接优化 MANO 参数，使当前序列满足 2D 观测、深度和局部平滑。

HMP prior 是先把 smooth 结果转成 HMP 输入，再按窗口优化 HMP latent 和部分 MANO/global 参数。它的核心作用是：

- 利用 HMP decoder 给 hand pose motion 加运动先验。
- 用 `z_l` motion prior 抑制非自然运动。
- 用 window overlap blend 拼接长序列。
- 用 `trans/orient` smooth 改善全局轨迹连续性。

### 5.4 有效帧 mask 与 pred_valid

实现位置：

- `run_pose3d_hand_stages.py::_vis_mask_for_track`
- `run_pose3d_hand_stages.py::Pose3DHandStageData.obs_data`
- `optim/optimizers.py::RootOptimizer.forward_pass`
- `optim/optimizers.py::SmoothOptimizer.forward_pass`

`pred_valid` 来自 `.pose3d_hand` 中每只手的预测结果，表示该帧原始 MANO/HaMeR 风格预测是否有效。它会原样写回输出，用于标记最终 payload 里每只手的预测有效性，也用于 `relative_motion` 中决定相邻帧运动是否有效。

优化用的 `vis_mask` 不是简单等于 `pred_valid`。当前逻辑是：

$$
v^{\mathrm{opt}}_{b,t}
= v^{\mathrm{kpt}}_{b,t}
\lor
v^{\mathrm{pred}}_{b,t},
$$

其中 $v^{\mathrm{kpt}}_{b,t}$ 是 keypoints payload 中该手该帧是否有有效 2D 检测，$v^{\mathrm{pred}}_{b,t}$ 是 `.pose3d_hand` 的 `pred_valid`。代码用逻辑或，因此只要有 2D keypoint 或原始 MANO 预测有效，该帧就进入该 track 的优化区间。

随后代码找到第一个和最后一个有效帧：

$$
t_s=\min\{t\mid v^{\mathrm{opt}}_{b,t}=1\},
\qquad
t_e=1+\max\{t\mid v^{\mathrm{opt}}_{b,t}=1\}.
$$

并构造 ternary mask：

$$
m_{b,t}=
\begin{cases}
-1, & t<t_s\ \text{or}\ t\ge t_e,\\
1, & v^{\mathrm{opt}}_{b,t}=1,\\
0, & t_s\le t<t_e\ \text{and}\ v^{\mathrm{opt}}_{b,t}=0.
\end{cases}
$$

root/smooth 优化中实际使用的是：

$$
m^{\mathrm{loss}}_{b,t} = [m_{b,t}\ge 0].
$$

因此 `-1` 表示 track 出现前/消失后的帧，不参与 loss；`0` 和 `1` 都在 track 区间内，会参与当前 root/smooth loss。2D keypoint 本身仍通过 confidence 控制单个 joint 的贡献；如果某些帧 keypoint 是插值补齐，`vis_mask` 控制的是是否在 track 时间区间内，而不是每个 keypoint 的置信度。

简言之：

- `pred_valid`：输入/输出 payload 层面的原始预测有效性。
- `vis_mask`：优化层面的 track 时间区间 mask，由 `keypoint_valid OR pred_valid` 合成，并额外用 `-1` 标记区间外帧。

### 5.5 StepLR 的 step_size 与 LBFGS 设置的区别

HMP 的 `scheduler.step_size=200` 是学习率调度周期。它不表示一次优化内部要做 200 次 line search，也不表示一次 `optimizer.step()` 内部迭代 200 次。Adam 每个循环通常做一次 forward/backward/update，然后 `scheduler.step()` 把全局迭代计数加一；当计数达到 200、400、600 等边界时，学习率乘以 `gamma`。

LBFGS 的 `lbfgs_max_iter=20` 是一次 `optimizer.step(closure)` 内部最多执行多少次近似二阶更新/闭包评估。再加上 strong Wolfe line search，单个外层迭代可能多次调用 closure。它控制的是“单次外层 step 的内部求解强度”，不是学习率随时间衰减的周期。

两者对比：

- `StepLR.step_size`: Adam 的跨迭代学习率衰减周期，影响 $\eta_t$。
- `LBFGS.max_iter`: 单次 LBFGS step 内部最多尝试次数，影响每个外层 step 的计算成本和 line search 行为。
- Adam 的 `niters=400` 是实际循环次数；LBFGS 的 `num_iters=50/300` 只是外层循环次数，真实 forward/backward 次数还要乘以内部 closure 次数。

## 6. 迭代次数过多的风险

### 6.1 对 2D 噪声过拟合

2D keypoints 来自检测器，存在抖动、遮挡误检和置信度不稳定。迭代过多时，优化会继续追逐这些噪声：

- 手指可能局部扭曲以贴合错误 keypoint。
- depth/scale 可能用非真实方式补偿投影误差。
- 相邻帧出现高频抖动，尤其是 2D 权重远大于其他项时。

### 6.2 平滑项过强或迭代过久导致过度平滑

smooth 和 HMP 的平滑项会压制时间变化。迭代过多时可能出现：

- 快速手势被抹平。
- translation/root orientation 滞后。
- 真实加速度被当成噪声压掉。

### 6.3 HMP latent 被 prior 拉回均值

HMP stg2 的 `lambda_motion_prior=200` 会惩罚 `z_l^2`。迭代过多或权重过强时：

- motion latent 可能过度接近零。
- 输出更像平均运动，个性化细节减少。
- 2D 对齐和运动自然性之间出现偏差。

### 6.4 LBFGS 单步成本高，时间不可线性直觉估计

root/smooth 的 `num_iters` 不是实际 forward/backward 次数。每次外层迭代可能触发多次 closure，最多受 `lbfgs_max_iter=20` 和 line search 影响。因此把 `num_iters` 增大一倍，实际耗时可能接近或超过一倍，取决于 line search 的评估次数。

### 6.5 数值不稳定

长时间优化还可能放大数值问题：

- 投影深度接近 0 时，`x/z, y/z` 梯度会很大。
- MANO 参数可能进入不自然区域，产生 NaN/Inf。
- LBFGS 对非平滑 robust loss 和强非线性投影较敏感，某些阶段可能频繁闭包评估。
- HMP Adam 虽有梯度裁剪 `clip_grad_norm_(..., 5.0)`，但仍可能在错误观测上缓慢漂移。

因此实际调参时，迭代次数应和 loss 权重一起看：如果 2D loss 下降但可视化抖动增加，通常不是继续加迭代，而是提高先验/平滑约束、降低 2D 权重、过滤 keypoints 或提前停止。

## 7. 论文配置与当前仓库差异

本节对照 Dyn-HaMR 论文中的优化目标、损失系数和当前仓库实际配置。论文依据为 Sec. 3.2、Sec. 3.3、Sec. 4 Implementation details 与 Appendix A.2；仓库依据为 `confs/optim.yaml`、`HMP/hmp_config.yaml`、`optim/losses.py`、`HMP/fitting.py` 和 `HMP/fitting_utils.py`。

### 7.1 论文中的优化目标

论文 Stage II 是 4D global motion optimization，目标函数为：

$$
E_I(wq^h,\omega,R_t,\tau_t^c)
= \lambda_{2d}L_{2d}
+ \lambda_s L_{\mathrm{smooth}}
+ \lambda_{\mathrm{cam}}L_{\mathrm{cam}}
+ \lambda_J L_J
+ \lambda_\beta L_\beta.
$$

其中：

- `L2d`：将当前世界坐标手部 3D joints 通过相机外参、世界尺度和内参投影回 2D，并与初始化 2D keypoints 对齐；论文写作中使用 Geman-McClure robust function 和 joint visibility mask。
- `Lsmooth`：同时约束相邻帧 3D joints 位移与 local pose rotation 的 geodesic smoothness。
- `Lcam`：约束相机旋转和平移在时间上的平滑性。
- `LJ`：pose regularization，论文写成标准 pose prior/regularization。
- `Lbeta`：shape prior，即 MANO betas 的 L2 正则。

论文 Stage III 是 interacting motion prior optimization，目标函数为：

$$
E_{II}(wq^h,\omega,R_t,\tau_t^c)
= L_{\mathrm{prior}} + L_{\mathrm{pen}} + L_{\mathrm{bio}}
+ \lambda_{2d}L_{2d}
+ \lambda_s L_{\mathrm{smooth}}
+ \lambda_{\mathrm{cam}}L_{\mathrm{cam}}
+ \lambda_J L_J
+ \lambda_\beta L_\beta.
$$

其中：

- `Lprior = lambda_z Lz + lambda_phi Lphi + lambda_tau Ltau`。`Lz` 约束 HMP latent likelihood；`Lphi` 和 `Ltau` 约束 global orientation 与 translation 贴近前一阶段/初始化轨迹。
- `Lbio = lambda_ja Lja + lambda_bl Lbl + lambda_palm Lpalm`。三项分别约束 joint angle、bone length 和 palm region。
- `Lpen`：只在双手同时存在时使用，针对左右手相交顶点做双向最近点距离惩罚。

论文还说明 Stage II 先优化 root orientation 和 translation，再优化 local pose、shape、world scale 和 camera extrinsics；Stage III 先优化 root orientation 和 translation，随后加入 latent code、local pose 和 camera parameters。

### 7.2 论文报告的系数

论文 Sec. 4 报告的主要系数如下：

| 阶段 | 论文损失项 | 论文系数 |
| --- | --- | --- |
| Stage II | `L2d` | `lambda_2d = 0.001` |
| Stage II | `Lsmooth` | `lambda_smooth = 10` |
| Stage II | `Lcam` | `lambda_cam = 100` |
| Stage II | pose prior / `LJ` | `lambda_theta = 0.04` |
| Stage II | shape prior / `Lbeta` | `lambda_beta = 0.05` |
| Stage III | HMP latent prior / `Lz` | `lambda_z = 200` |
| Stage III | global orientation consistency / `Lphi` | `lambda_phi = 2` |
| Stage III | translation consistency / `Ltau` | `lambda_gamma` 或 `lambda_tau = 10` |
| Stage III | penetration / `Lpen` | `lambda_pen = 10` |
| Stage III | shape prior / `Lbeta` | `lambda_beta = 0.05` |
| Stage III | joint angle / `Lja` | `lambda_ja = 1` |
| Stage III | palm / `Lpalm` | `lambda_palm = 1` |
| Stage III | bone length / `Lbl` | `lambda_bl = 1` |

论文主文还写到三阶段优化使用 L-BFGS，学习率 `lr = 1`。Appendix A.2 进一步说明先以较低 `lambda_smooth = 1` 单独优化两只手，再联合优化双手；这和主文 Stage II 系数 `lambda_smooth = 10` 存在粒度差异，可理解为实现中的 staged schedule，而不是单一固定权重。

### 7.3 当前仓库实际启用的损失

当前 `run_pose3d_hand_stages.py` 的外层阶段是 `root -> smooth -> prior`。`root` 和 `smooth` 使用 `confs/optim.yaml::optim.loss_weights` 的三列权重，其中第 0 列用于 root，第 1 列用于 smooth，第 2 列保留了旧 motion-stage 风格权重；当前 HMP prior 主流程不直接使用 `optim.loss_weights` 的第 2 列作为 Stage III 总目标。

当前 root/smooth 的主要配置为：

| 配置项 | 当前值 | 影响 |
| --- | --- | --- |
| `joints2d` | `[10000, 10000, 10000]` | root/smooth 极强 2D 重投影约束 |
| `joints3d_smooth` | `[1000, 10000, 0]` | root/smooth 启用 3D joint temporal smooth |
| `cam_R_smooth` | `[0, 0, 0]` | 相机旋转平滑关闭 |
| `cam_t_smooth` | `[0, 0, 0]` | 相机平移平滑关闭 |
| `bg2d` | `[0, 0, 0]` | 背景/相机 reprojection 分支关闭 |
| `pose_prior` | `[1, 1, 1]` | smooth 中约束 latent pose 贴近初始化 |
| `shape_prior` | `[0.05, 0.05, 0.05]` | 与论文 `lambda_beta=0.05` 数值一致 |
| `depth_constraint` | `[100, 100, 0]` | 本地新增，防止 hand 在相机后方 |
| `penetration` | `[0, 0, 0]` | root/smooth 中禁用 interpenetration |
| `bio` | `[0, 0, 0]` | root/smooth 中禁用 biomechanical loss |

当前 HMP prior 的主要配置来自 `HMP/hmp_config.yaml`：

| HMP 阶段 | 当前优化器/迭代 | 主要启用项 |
| --- | --- | --- |
| `stg1` | Adam, `lr=0.05`, `niters=400` | `lambda_reproj=0.05`, `lambda_bio=1`, `lambda_orient_smooth=1`, `lambda_trans=10`, `lambda_trans_smooth=10` |
| `stg2` | Adam, `lr=0.05`, `niters=400` | stg1 项 + `lambda_motion_prior=200`，并优化 `z_l` |
| `stg3` | `lr=0`, `niters=0` | 实际关闭 |

HMP 使用 `StepLR(step_size=200, gamma=0.7)`。这与论文“Stage III 使用 L-BFGS，前 200 步 root/trans，后 200 步加入 latent/local/camera”的描述不同。

### 7.4 逐项差异表

| 损失项 | 论文计算/语义 | 论文系数 | 仓库实现位置 | 当前仓库系数 | 差异结论 |
| --- | --- | --- | --- | --- | --- |
| `L2d` / 2D reprojection | perspective projection 后与 2D keypoints 做 robust Geman-McClure，加 visibility mask | Stage II `0.001` | `optim/losses.py::Joints2DLoss`, `HMP/fitting_utils.py::joints2d_loss` | root/smooth: `[10000,10000,10000]`; HMP: `lambda_reproj=0.05` | 数值完全不可直接等价。root/smooth 版本还按 hand scale 归一化并乘 `100`，不是论文原始像素 robust error。HMP 版本用 `sigma=40` 的 `gmof` 和 confidence 加权。 |
| `Lsmooth` | 3D joints temporal smooth + local pose geodesic smooth | Stage II `10`，Appendix 提到局部阶段可用 `1` | `optim/losses.py::joints3d_smooth_loss`, `HMP/fitting.py` smooth terms | root/smooth: `joints3d_smooth=[1000,10000,0]`; HMP: `lambda_orient_smooth=1`, `lambda_trans_smooth=10`, `lambda_j3d_smooth=0` | root/smooth 只覆盖 3D joint smooth，且权重大幅高于论文；local pose geodesic smooth 在 root/smooth 中没有按论文形式完整启用。 |
| `Lcam` | 相邻相机旋转 geodesic + 相机平移 smooth/displacement | Stage II `100` | `optim/losses.py::rotation_smoothness_loss`, `translation_smoothness_loss` | `cam_R_smooth=[0,0,0]`, `cam_t_smooth=[0,0,0]`, `bg2d=[0,0,0]` | 主配置关闭相机平滑；对应代码分支目前还包含 `raise ValueError`，实际不能按论文启用。 |
| `LJ` / pose prior | 标准 pose prior/regularization | Stage II `lambda_theta=0.04` | `optim/losses.py::pose_prior_loss` | `pose_prior=[1,1,1]` | 当前实现是惩罚 optimized latent pose 偏离 `init_latent_pose`，更像 HaMeR 初始化保持项，不是论文中的标准 pose prior 语义。 |
| `Lbeta` / shape prior | MANO betas L2 prior | Stage II/III `0.05` | `optim/losses.py::shape_prior_loss`, HMP betas prior | root/smooth: `shape_prior=[0.05,0.05,0.05]`; HMP: `betas_prior=10` | root/smooth 数值与论文一致；HMP 内部还有独立 `betas_prior=10`，尺度与实现不同，不能直接和论文 `lambda_beta` 对齐。 |
| `Lz` / motion prior | HMP latent negative log-likelihood | Stage III `lambda_z=200` | `HMP/fitting.py::motion_prior_loss` | `stg2.lambda_motion_prior=200` | 数值一致，但当前实现为对 `z_l` 的平方惩罚；是否等价于论文 NLL 取决于 HMP prior 实现细节。 |
| `Lphi` | root/global orientation consistency | Stage III `lambda_phi=2` | `HMP/fitting.py::L_orient` | `lambda_orient=0`, `lambda_orient_smooth=1` | 当前没有启用 direct orientation-to-target consistency，只启用 orientation smooth，和论文 `Lphi` 不一致。 |
| `Ltau` / translation consistency | translation 贴近前阶段/初始化轨迹 | Stage III `lambda_gamma` 或 `lambda_tau=10` | `HMP/fitting.py::L_trans` | `lambda_trans=10` | 数值和论文一致，语义基本对应。另有 `lambda_trans_smooth=10`，论文中属于 smooth 类项。 |
| `Lbio` | joint angle、bone length、palm interval constraints | `lambda_ja=1`, `lambda_palm=1`, `lambda_bl=1` | `optim/bio_loss.py`, `HMP/fitting.py` | root/smooth: `bio=[0,0,0]`; HMP: `lambda_bio=1` | root/smooth 禁用；HMP prior 启用且内部 `BMCLoss(lambda_bl=1, lambda_rb=1, lambda_a=1)` 大体对应论文三项 biomechanical constraint。 |
| `Lpen` | 双手相交顶点的双向最近点距离 | Stage III `lambda_pen=10` | `optim/losses.py::GeneralContactLoss`, `SMPLLoss.forward` | `penetration=[0,0,0]` | 当前主配置禁用论文强调的 interpenetration loss；HMP prior 中也没有看到等价启用项。 |
| depth constraint | 论文无此项 | 无 | `optim/losses.py::depth_constraint_loss` | `depth_constraint=[100,100,0]` | 本地新增项，用于防止 joints 落在相机后方；会改变 Stage II/root-smooth 的优化行为。 |
| bone length / joint consistency 第 2/3 列 | 旧 motion stage rollout consistency | 论文中属于 Stage III bio 或 motion consistency 的一部分 | `optim/losses.py` 中旧 `MotionLoss` 注释块 | `bone_length=[0,2000,2000]`, `joint_consistency=[0,0,100]` | 当前旧 `MotionLoss` 主体被注释，外层 HMP prior 不直接使用这些 `optim.loss_weights` 第 2 列；配置值不代表论文 Stage III 已启用。 |

### 7.5 结论与风险

当前仓库不是论文配置的逐字复现，而是一个已经做过本地改造的实现分支。最核心差异有五点：

- root/smooth 的 `joints2d=10000` 和论文 `lambda_2d=0.001` 差异极大，同时 loss 内部尺度也不同，不能只看系数大小判断约束强弱。
- 论文 Stage II 的相机平滑/位移约束 `lambda_cam=100` 在当前主配置中关闭；相关代码分支启用后还会触发 `raise ValueError`，说明当前管线实际没有复现论文的 camera regularization。
- 论文 Stage III 的 penetration loss `lambda_pen=10` 在当前主配置中关闭，这会影响双手交互质量，尤其是论文 Fig. 3 和 Appendix ablation 强调的手部穿插场景。
- 当前 HMP prior 使用 Adam + StepLR，stg1/stg2 各 400 次；论文报告三阶段均使用 L-BFGS，Stage III 是 200+200 的 staged optimization。
- 当前新增 `depth_constraint`、latent-to-initial `pose_prior`、高权重 `joints3d_smooth` 等项，会让结果更偏向本地调参目标，而不是论文原始 objective。

如果目标是复现实验论文，需要重新核对官方 release 的配置、优化器和 staged schedule；如果目标是让当前仓库在本地数据上稳定运行，则应把这些差异视为有意的工程调参，并用可视化和 loss 曲线评估是否继续保留。
