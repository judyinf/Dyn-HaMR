# `run_opt.py` 逐行解析

本文档对 `dyn-hamr/run_opt.py` 的主入口逻辑进行逐行解释，覆盖 `set_seed`、`run_opt` 和 `main` 三个函数。

---

## 1. 导入与全局常量 (L1–52)

```python
import os, glob, json, subprocess, random       # 标准库
import numpy as np                               # 数值计算
import torch                                     # 深度学习
from torch.utils.data import DataLoader           # 单 batch 全轨道加载
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 日志
```

**项目内导入**：

```python
from data import get_dataset_from_cfg, expand_source_paths
```
L12 — 数据集工厂函数 + 路径通配符展开（见 §1.6）。

```python
from optim.base_scene import BaseSceneModel
```
L15 — 场景模型，持有 MANO + 参数容器。

```python
from optim.optimizers import RootOptimizer, SmoothOptimizer
```
L17–20 — Stage 0（根优化）和 Stage 1（平滑优化）的优化器。

```python
from optim.output import (
    save_track_info, save_camera_json,
    save_input_poses, save_initial_predictions,
)
```
L21–26 — 优化前中后的导出函数。

```python
from vis.viewer import init_viewer          # 实时可视化
from body_model import MANO                  # MANO 手部模型
from util.loaders import resolve_cfg_paths   # 路径解析
from util.logger import Logger               # 日志
from util.tensor import get_device, move_to, detach_all, to_torch  # 张量工具
from run_vis import run_vis                  # 可视化入口
```

**Hydra 配置**：

```python
import hydra
from omegaconf import DictConfig, OmegaConf
```
L40–41 — Hydra 配置框架：`@hydra.main` 装饰器 + OmegaConf 解析器注册。

```python
N_STAGES = 3
```
L44 — 损失权重列表固定为 3 阶段（Stage 0/1/2 对应 root/smpl/smooth，smpl 已禁用）。

**路径补丁与 HMP/VPoser**：

```python
import sys
sys.path.append('src/human_body_prior')   # 添加 human_body_prior 到搜索路径
sys.path.append('HMP/')                   # 添加 HMP 子模块
from HMP.fitting import run_prior         # 阶段 III: 运动先验精修
from human_body_prior.tools.model_loader import load_model   # VPoser 加载 (未使用)
from human_body_prior.models.vposer_model import VPoser      # VPoser 模型 (未使用)
```
L46–54 — VPoser 导入在 main 分支中被注释掉（L53–54 已注释），dev 分支进一步移除了导入行。

---

## 2. `set_seed(seed=42)` (L56–67)

```python
def set_seed(seed=42):
```
L56 — 默认种子 42，保证相同输入下可复现。

```python
    random.seed(seed)                             # Python 内置随机数
    np.random.seed(seed)                          # NumPy 随机数
    torch.manual_seed(seed)                       # PyTorch CPU 随机数
    torch.cuda.manual_seed(seed)                  # 当前 GPU 随机数
    torch.cuda.manual_seed_all(seed)              # 所有 GPU 随机数
```
L60–64 — 五层随机种子设置。

```python
    torch.backends.cudnn.benchmark = False         # 禁用 cuDNN 自动算法选择
    torch.backends.cudnn.deterministic = True      # 强制 cuDNN 确定性算法
```
L65–66 — 牺牲速度换取确定性。`benchmark=False` 要求 cuDNN 每次使用相同的卷积算法而非根据输入尺寸自适应选择；`deterministic=True` 禁用非确定性算法（如某些 `atomicAdd` 实现）。

```python
    os.environ['PYTHONHASHSEED'] = str(seed)       # 固定 Python hash 种子
```
L67 — 防止 `dict`/`set` 的迭代顺序在不同运行中变化。

---

## 3. `run_opt(cfg, dataset, out_dir, device)` (L69–169)

### 3.1 数据加载 (L70–80)

```python
    a = time.time()                                # 计时起点 (总优化)
    args = cfg.data                                # 数据配置简写
    B = len(dataset)                               # 轨道数 (= n_tracks)
    T = dataset.seq_len                            # 帧数 (= eidx - sidx)
    loader = DataLoader(dataset, batch_size=B, shuffle=False)
```
L70–74 — `batch_size=B` 将全部轨道堆叠为一个 batch；`shuffle=False` 保持轨道顺序。一次 `next(iter(loader))` 取出全部 B×T 数据。

```python
    obs_data = move_to(next(iter(loader)), device)     # 观测数据 → GPU
    cam_data = move_to(dataset.get_camera_data(), device)  # 相机数据 → GPU
```
L76–77 — `move_to` 递归地将 dict 中所有 `torch.Tensor`/`np.ndarray` 转移到 `device`。`obs_data` 形状为 `(B,T,...)`，`cam_data` 形状为 `(T,...)`。

```python
    print("Batch size (dataset_length), T (dataset.seq_len): ", B, len(dataset), T)
    print("OBS DATA", obs_data.keys())             # joints2d, vis_mask, init_body_pose, ...
    print("CAM DATA", cam_data.keys(), 'cam_R', cam_data['cam_R'])
```
L78–80 — 打印数据维度和相机旋转矩阵（首帧平移已置零）。

### 3.2 相机保存与静态检测 (L82–90)

```python
    cam_R, cam_t = dataset.cam_data.cam2world()     # w2c → c2w 转换
    intrins = dataset.cam_data.intrins              # (T, 4)
    save_camera_json(f"cameras.json", cam_R, cam_t, intrins)
```
L83–85 — 输出 `cameras.json`（施加 Y-flip 后），供可视化使用。保存到 Hydra 工作目录（`os.getcwd()` = `outputs/.../demo1-all-shot-0-0--1/`）。

```python
    cfg.model.opt_scale &= not dataset.cam_data.is_static
```
L89 — 静态相机（三脚架/固定机位）禁用世界尺度优化——SLAM 的轨迹已经是零均值的，若强行优化 scale 会产生数值不稳定。`&=` 确保即使 `cfg.model.opt_scale=True`，静态相机也覆盖为 `False`。

### 3.3 损失权重转置 (L92–98)

```python
    all_loss_weights = cfg.optim.loss_weights
    assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
```
L93–94 — `optim.yaml` 中每个 loss 权重为 `[stage0_wt, stage1_wt, stage2_wt]` 的三元素列表。断言所有 loss 的权重长度等于 `N_STAGES=3`。

```python
    stage_loss_weights = [
        {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
    ]
```
L95–97 — **列转行**：将 `{joints2d:[10000,10000,10000], ...}` 转置为 `[{joints2d:10000, ...}, {joints2d:10000, ...}, ...]`。每个 stage 得到一个独立的权重字典。

```python
    max_loss_weights = {k: max(wts) for k, wts in all_loss_weights.items()}
```
L98 — 各 loss 在所有阶段的最大权重，用于 TensorBoard 日志中归一化统计显示（被传入 `RootLoss/SMPLLoss.__init__`）。

### 3.4 模型加载 (L100–116)

```python
    cfg = resolve_cfg_paths(cfg)
    cfg.paths.base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../"))
```
L101–102 — `resolve_cfg_paths` 将 config 中的相对路径（如 `_DATA/data/`）解析为以项目根目录为基准的绝对路径。`base_dir` 被设为 `dyn-hamr/` 的父目录（即项目根目录）。

```python
    pose_prior = None
```
L107 — VPoser 未加载，`pose_prior = None`。后续 `BaseSceneModel.__init__` 检测到 `pose_prior is None` 后跳过 VPoser 相关逻辑，`latent2pose()`/`pose2latent()` 退化为恒等映射。

```python
    cfg = resolve_cfg_paths(cfg)       # 第二次调用 (L110, 与 L101 重复)
    mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
    hand_model = MANO(batch_size=B*T, pose2rot=True, **mano_cfg).to(device)
```
L114–116 — 将 MANO 配置键转为小写（`MODEL_PATH` → `model_path`）。`batch_size=B*T` 允许一次前向处理所有轨道的所有帧（`B` 个轨道 × `T` 帧 = `B*T` 个独立 batch 元素）。`pose2rot=True` 要求输入轴角格式的全局旋转和手部姿态，MANO 内部自动转为旋转矩阵。

### 3.5 场景模型初始化 (L121–131)

```python
    margs = cfg.model
    base_model = BaseSceneModel(B, T, hand_model, pose_prior, **margs)
```
L121–124 — `**margs` 展开为 `use_init=True, opt_cams=False, opt_scale=True`。

```python
    base_model.initialize(obs_data, cam_data)
```
L126 — cam2world 转换 + 参数注册。详见 §1.7.2。

```python
    base_model.to(device)
```
L127 — 将 `base_model` 及其所有子模块（`CameraParams` 包含的 `nn.Parameter`）迁移到 GPU。

```python
    save_input_poses(dataset, os.path.join(out_dir, "hamer"), args.seq)
```
L130 — 保存 HaMeR 原始预测的简化 NPZ（仅 `pose_body, trans, root_orient`）到 `hamer/` 目录。

```python
    save_initial_predictions(base_model, os.path.join(out_dir, "init"), args.seq)
```
L131 — 保存 cam2world 转换后、优化前的参数快照到 `init/` 目录。

### 3.6 可视化初始化 (L133–146)

```python
    opts = cfg.optim.options
    vis_scale = 0.25
    vis = None
    if opts.vis_every > 0:
        vis = init_viewer(
            dataset.img_size,
            cam_data["intrins"][0],
            vis_scale=vis_scale,
            bg_paths=dataset.sel_img_paths,
            fps=cfg.fps,
        )
```
L136–143 — `vis_every > 0` 时启用 pyrender 实时可视化（每 N 轮渲染一帧）。`vis_scale=0.25` 将渲染分辨率缩放到原图的 1/4。`bg_paths` 提供视频帧作为渲染背景。

```python
    writer = SummaryWriter(out_dir)
```
L146 — TensorBoard 日志写入 Hydra 输出目录。

### 3.7 Stage 0: RootOptimizer (L148–151)

```python
    a = time.time()
    optim = RootOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.root.num_iters, out_dir, vis, writer)
```
L149–151 — 创建 RootOptimizer → 执行 50 次 L-BFGS 迭代。`optim.run()` 内部调用 `set_opt_vars(["trans", "root_orient"])`（冻结 betas/latent_pose/world_scale），每 20 轮保存 checkpoint + NPZ。

### 3.8 Stage 1: SmoothOptimizer (L153–162)

```python
    args = cfg.optim.smooth
    b = time.time()
    print('root optimization time: ', b - a)
```
L153–156 — 打印 Stage 0 耗时。

```python
    optim = SmoothOptimizer(
        base_model, stage_loss_weights, opt_scale=args.opt_scale, **opts
    )
    optim.run(obs_data, args.num_iters, out_dir, vis, writer)
```
L157–160 — `SmoothOptimizer` 在 `RootOptimizer` 基础上增加 `betas`、`latent_pose`、可选 `world_scale` 的优化。`opt_scale` 从 `smooth.opt_scale` 配置读取（默认 False，即此阶段不优化 scale）。`num_iters=300`。

```python
    c = time.time()
    print('Smooth optimization time: ', c - b)
```
L161–162 — 打印 Stage 1 耗时。

### 3.9 Stage 2 (可选): HMP 运动先验 (L164–169)

```python
    if cfg.run_prior and not os.path.exists(os.path.join(out_dir, 'prior')):
        run_prior(cfg, dataset, out_dir, device, ['smooth_fit'],
                  obs_data, hand_model, cfg, cfg.data,
                  os.path.join(out_dir, 'prior'))
```
L165–167 — 只有 `run_prior=True` 且 `prior/` 目录不存在时才执行。`['smooth_fit']` 指定从 `smooth_fit/` 加载结果作为 HMP 输入。`run_prior()` 封装了 HMP 模型加载 + `multi_stage_opt()` 潜码优化 + 拼接回完整序列的完整流程。

```python
    d = time.time()
    print('prior optimization time: ', d-c)
```
L168–169 — 打印 HMP 耗时。

---

## 4. `main(cfg: DictConfig)` (L172–206)

```python
@hydra.main(version_base=None, config_path="confs", config_name="config.yaml")
def main(cfg: DictConfig):
```
L172–173 — `@hydra.main` 装饰器使 Hydra 在调用 `main` 前完成：加载 `confs/config.yaml` + 所有 `defaults` 覆盖 + OmegaConf 变量插值、将工作目录切换到 Hydra 输出路径（`chdir=True`）、注入解析后的 `cfg: DictConfig`。

```python
    OmegaConf.register_new_resolver("eval", eval)
```
L174 — 注册 `"eval"` 解析器，使配置中可以使用 `${eval:1+2}` 在配置解析时动态求值 Python 表达式。

```python
    print('run_opt.py: ', cfg)
    set_seed(cfg.get('seed', 42))
```
L175–178 — 打印完整配置（含所有 defaults 覆盖后的最终值），设置随机种子。

```python
    out_dir = os.getcwd()
    Logger.init(f"{out_dir}/opt_log.txt")
```
L180–182 — Hydra `chdir=True` 后 `os.getcwd()` 返回的是输出目录（如 `../outputs/logs/video-custom/2026-07-31/demo1-all-shot-0-0--1/`），后续所有 NPZ/JSON/pth 文件都写入此目录。

```python
    print("init SOURCES", cfg.data.sources)
    cfg.data.sources = expand_source_paths(cfg.data.sources)
    print("SOURCES", cfg.data.sources)
```
L185–187 — `expand_source_paths` 对 `sources` 中每个路径调用 `glob.glob` 展开通配符。若路径不包含通配符则直接返回原值。打印前后对比用于调试。

```python
    dataset = get_dataset_from_cfg(cfg)
    save_track_info(dataset, out_dir)
```
L189–190 — 构造 `MultiPeopleDataset`（含按需预处理：帧提取 + HaMeR + SLAM），保存轨道信息到 `track_info.json`。

```python
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu"))
    device_id = cfg.get("gpu")
```
L191–193 — 设置 PyTorch 可见 GPU（`config.yaml` 中的 `gpu` 字段）。注意 `CUDA_VISIBLE_DEVICES` 必须在任何 CUDA 初始化之前设置——这里在 `get_device()` 调用之前。

```python
    if cfg.run_opt:
        device = get_device(device_id)
        run_opt(cfg, dataset, out_dir, device)
```
L195–197 — `get_device(device_id)` 返回 `torch.device("cuda:0")` 或 `"cpu"`。`run_opt()` 完成两阶段 L-BFGS + 可选 HMP。

```python
    if cfg.run_vis:
        run_vis(cfg, dataset, out_dir, device_id, **cfg.get("vis", dict()))
```
L199–202 — 优化结束后渲染可视化视频。`cfg.vis` 指定了渲染的阶段（`phases: [smooth_fit, prior]`）和多视角（`render_views: [above, side, front, src_cam]`）。`run_opt` 和 `run_vis` 可以在同一次运行中顺序执行，也可以分离运行（`run_opt=False, run_vis=True`）。

```python
if __name__ == "__main__":
    main()
```
L205–206 — Hydra 在此处介入：Python 解释器执行到 `main()` 时 Hydra 装饰器拦截调用，完成配置加载和 chdir 后再进入函数体。

---

## 5. 完整执行时序

```
python run_opt.py data=video_vipe run_opt=True data.seq=demo1
    │
    ├── Hydra 装饰器  ──→  加载 config.yaml + video_vipe.yaml 覆盖
    │                       OmegaConf 解析 ${eval:...} 插值
    │                       chdir 到 ../outputs/logs/video-custom/<date>/<name>/
    │
    ├── main()
    │   ├── set_seed(42)         ← 五层随机种子
    │   ├── Logger.init()         ← 日志文件
    │   ├── expand_source_paths   ← 通配符展开
    │   ├── get_dataset_from_cfg  ← 按需预处理 + 构造 Dataset
    │   ├── save_track_info       ← track_info.json
    │   └── run_opt()
    │       ├── DataLoader → obs_data, cam_data → GPU
    │       ├── save_camera_json  ← cameras.json (Y-flip)
    │       ├── cfg.model.opt_scale &= not is_static
    │       ├── stage_loss_weights ← 列转行
    │       ├── MANO(batch_size=B*T) → hand_model
    │       ├── BaseSceneModel(B, T, hand_model, None, ...)
    │       ├── base_model.initialize(obs_data, cam_data)
    │       │   ├── CameraParams.set_cameras()
    │       │   ├── cam2world: trans, root_orient
    │       │   └── set_param: 6 个 MANO/相机参数
    │       ├── save_input_poses  ← hamer/*.npz
    │       ├── save_initial_predictions ← init/*.npz
    │       ├── RootOptimizer.run(50 iters)     ← Stage 0
    │       │   └── L-BFGS closure ×50:
    │       │       BaseSceneModel.pred_params_mano()
    │       │       → RootLoss.forward()
    │       │       → backward() → optim.step()
    │       ├── SmoothOptimizer.run(300 iters)  ← Stage 1
    │       │   └── L-BFGS closure ×300:
    │       │       BaseSceneModel.pred_params_mano()
    │       │       → SMPLLoss.forward()
    │       │       → backward() → optim.step()
    │       └── [HMP run_prior() — if cfg.run_prior]
    │
    └── [run_vis() — if cfg.run_vis]
          ├── 加载 smooth_fit/*.npz (和 prior/*.npz)
          ├── prep_result_vis() → MANO forward
          ├── pyrender 离屏渲染
          └── *_final_*.mp4 + *_meshes/*.obj
```
