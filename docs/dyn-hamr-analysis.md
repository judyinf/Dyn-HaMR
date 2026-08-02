# Dyn-HaMR 核心机制分析

本文档详细解析 Dyn-HaMR 项目的核心机制，涵盖数据流、三阶段优化、关键字段约定、插值方法、配置系统、相机坐标系、重要模型和类清单。

---

## 目录

1. [run_opt 数据流与输出产物](#1-run_opt-数据流与输出产物) — 整体数据流、处理单元与帧分类、输出目录树、NPZ 字段约定、生产/消费链路
2. [三阶段优化](#2-三阶段优化) — 设计思路、obs_data/pred_data 字段、Stage 0/1 详解、参数传递、L-BFGS 配置
3. [重要字段约定](#3-重要字段约定) — vis_mask 构建与掩码、bbox-关键点耦合、插值触发、预测错误场景
4. [插值方法](#4-插值方法) — 2D 关键点坐标/置信度插值、MANO 参数插值、对比总结
5. [配置文件系统设计](#5-配置文件系统设计) — Hydra/OmegaConf、配置层次、路径解析、关键配置项
6. [相机坐标系约定](#6-相机坐标系约定) — w2c 格式、世界原点、cam2world 转换、左乘/右乘、转换函数表
7. [重要模型说明](#7-重要模型说明) — MANO 前向、VPoser（已禁用）、HMP 运动先验（NeMF/FK/多阶段优化）
8. [重要类清单](#8-重要类清单) — 优化器/损失/参数/场景/数据/身体/HMP/可视化/工具类 + 生命周期总览

---

## 1. run_opt 数据流与输出产物

### 1.1 整体数据流

```
输入视频 (.mp4)
  │
  ├─[预处理] 帧提取 → images/{seq}/*.jpg
  │            HaMeR 跟踪 → _mano.json + _keypoints.json
  │            相机估计 → cameras.npz (VIPE/DROID-SLAM)
  │
  ▼
MultiPeopleDataset 加载
  │  输入: images + track_preds + shot_idcs + cameras.npz
  │  输出: obs_data (B, T, ...) + cam_data (T, ...)
  │
  ▼
BaseSceneModel.initialize()
  │  消费: obs_data + cam_data
  │  生产: params (trans, root_orient, betas, latent_pose, ...)
  │  扩展: obs_data["init_latent_pose"]
  │
  ▼
RootOptimizer.run()     [Stage 0, 50 L-BFGS iters]
  │  优化: trans, root_orient
  │  输出: root_fit/*.npz + root_fit_params.pth
  │
  ▼
SmoothOptimizer.run()   [Stage 1, 300 L-BFGS iters]
  │  优化: trans, root_orient, betas, latent_pose
  │  输出: smooth_fit/*.npz + smooth_fit_params.pth
  │
  ▼
run_prior()             [可选 HMP 精修]
  │  输出: prior/*.npz
  │
  ▼
run_vis()               [可选可视化]
  │  输出: *_final_*.mp4 + *_meshes/*.obj
```

### 1.2 输出目录结构

Hydra 输出路径（`config.yaml`）：`{log_root}/{data.type}-{data.split}/{YYYY-MM-DD}/{data.name}/`

对于 demo 视频（`video-custom` → `demo1`），典型结构：

```
../outputs/logs/video-custom/custom/2026-01-15/demo1-all-shot-0-0--1/
  │
  ├── .hydra/                           # Hydra 运行时配置快照
  │     config.yaml, overrides.yaml
  │
  ├── opt_log.txt                       # Logger 优化日志
  │
  ├── cameras.json                      # 相机外参/内参 (Y-flip 后)
  │     └─ {rotation: (T,9), translation: (T,3), intrinsics: (T,4)}
  │
  ├── track_info.json                   # 轨道可见性元数据
  │     └─ {tracks: {tid: {index, vis_mask}}, meta: {seq_interval, data_interval}}
  │
  ├── hamer/                            # HaMeR 原始预测 (仅可视化对比)
  │     └─ {seq}_000000_phalp_world_results.npz
  │         └─ {pose_body, trans, root_orient}
  │
  ├── init/                             # 模型初始化后的参数
  │     └─ {seq}_000000_init_world_results.npz
  │         └─ {pose_body, latent_pose, betas, trans, root_orient,
  │             is_right, init_body_pose, world_scale, cam_R, cam_t, intrins}
  │
  ├── root_fit/                         # Stage 0 (RootOptimizer)
  │     ├─ {seq}_000000_world_results.npz    ← iter 0
  │     ├─ {seq}_000020_world_results.npz    ← iter 20 (save_every=20)
  │     ├─ {seq}_000050_world_results.npz    ← 最终
  │     ├─ root_fit_params.pth
  │     ├─ root_fit_optim.pth
  │     └─ {loss_name}.png                   ← 损失箱线图
  │
  ├── smooth_fit/                       # Stage 1 (SmoothOptimizer)
  │     ├─ {seq}_000000_world_results.npz
  │     ├─ {seq}_000300_world_results.npz    ← 最终 (num_iters=300)
  │     ├─ smooth_fit_params.pth
  │     ├─ smooth_fit_optim.pth
  │     └─ {loss_name}.png
  │
  ├── prior/                            # HMP 精修 (仅 run_prior=True)
  │     ├─ {seq}_000000_world_results.npz
  │     └─ *.pkl
  │
  ├── {seq}_input.mp4                   # 输入帧视频
  ├── {seq}_{phase}_final_{iter}_src_cam.mp4
  ├── {seq}_{phase}_final_{iter}_above.mp4
  ├── {seq}_{phase}_grid.mp4            # 2×2 网格视频
  │
  └── {phase}/
        └─ {seq}_{iter}_meshes/         # 逐帧 OBJ
              ├─ 000000_0.obj
              ├─ 000000_1.obj
              └─ ...
```

### 1.3 NPZ 文件字段约定

所有 `*_world_results.npz` 文件包含统一字段：

| 键名 | 形状 | 含义 |
|------|------|------|
| `trans` | (B, T, 3) | **世界空间**手腕平移 |
| `root_orient` | (B, T, 3) | **世界空间**手腕旋转（轴角） |
| `pose_body` | (B, T, 45) | 手部姿态轴角（`latent2pose` 解码, 15关节×3） |
| `latent_pose` | (B, T, D) | 优化潜变量 (D=45, 无VPoser时与pose_body同值) |
| `betas` | (B, 10) | MANO 形状参数 |
| `is_right` | (B, T) | 手性 (1=右手, 0=左手) |
| `init_body_pose` | (B, T, 15, 3) | HaMeR 初始手指姿态 (仅 init 阶段, 用于存档) |
| `world_scale` | (1, 1) | 全局尺度 (仅 `opt_scale=True`) |
| `cam_R` | (B, T, 3, 3) | 世界到相机旋转矩阵 |
| `cam_t` | (B, T, 3) | 世界到相机平移向量 |
| `intrins` | (4,) | `[fx, fy, cx, cy]` (标量, 所有帧共享) |

Prior 阶段额外字段：`decode_root` (B,T,3), `poses` (B,T,48)。

简化 phalp NPZ（`hamer/` 目录）仅含 `{pose_body, trans, root_orient}`。

### 1.4 中间产物生产/消费链路

| 产物 | 阶段 | 格式 | 字段约定 | 生产 | 消费 | 生命周期 |
|------|------|------|---------|------|------|---------|
| `images/{seq}/*.jpg` | 预处理 | JPEG (6位零填充帧号) | — | `extract_frames.py:split_frame()` — `cv2.VideoCapture` 逐帧读 mp4 → JPEG | HaMeR `run.py` 读帧做检测+MANO推理 | 可保留作图像源，也可在 cameras.npz 生成后删除 |
| `{seq}.pkl` | 预处理 | joblib pickle | `{frame_name: {tid:[int], tracked_ids:[int], mano:[{betas(10), hand_pose(15,3,3), global_orient(3,3), is_right}], cam_trans(3), extra_data([x,y,c...]), shot:int}}` | HaMeR Pass 3 (`run_hamer_on_cleaned_bboxes`) — Pass 2清理后的bbox上重跑HaMeR推理 | `export_hamer.py:export_sequence_results()` 拆分到独立JSON | 拆分完成后仅备份 |
| `_mano.json` | 预处理 | JSON | `{betas:[10], body_pose:[15,3], global_orient:[3], cam_trans:[3], is_right:0\|1}` — 旋转量为轴角(Rodrigues转换) | `export_hamer.py:export_hamer_predictions()` → `unpack_frame()` 逐帧逐轨道写入 | `dataset.py:load_mano_preds()` → Slerp/线性插值 → `obs_data` 的 `init_body_pose/shape, init_root_orient, init_trans, is_right` | 数据集加载后仅作备份 |
| `_keypoints.json` | 预处理 | JSON (OpenPose) | `{"people":[{"pose_keypoints_2d":[x1,y1,c1,...,x21,y21,c21]}]}` — 21手部关节×3=63 float | `export_hamer.py:export_vitpose_keypoints()` 从pickle的extra_data提取写入 | `dataset.py:load_keypoints_with_interp()` → 线性插值+conf×0.8 → `obs_data["joints2d"]` | 同上 |
| `shot_idcs/{seq}.json` | 预处理 | JSON | `{frame_name: shot_index}` — 每帧→累加的shot索引 | `export_hamer.py:export_shot_changes()` 从pickle中PHALP的shot标记累加 | `dataset.py:get_shot_img_files()` 按shot_idx筛选帧列表(三层裁剪第1层) | 不参与优化 |
| `cameras.npz` | 预处理 | NPZ | `{height:float, width:float, focal:float, intrins:(N,4)[fx,fy,cx,cy], w2c:(N,4,4)}` | VIPE: `vidproc.py:run_vipe()`/`load_vipe_cameras()` → c2w→w2c→写入; DROID-SLAM: `run_slam.py:save_cameras()` 直接输出 | `CameraData.load_cameras_npz()` → 提取cam_R(T,3,3)+cam_t(T,3) → 偏移首帧→缩放intrins → `cam_data` | 加载后常驻内存 |
| `obs_data` | 数据加载 | Python dict (GPU tensor) | 含11数据集字段+1扩展字段, 形状(B,T,...) float32. 关键: `joints2d(B,T,21,3)`, `init_body_pose(B,T,15,3)`, `vis_mask(B,T)`, `is_right(B,T)`, `init_latent_pose(B,T,D)` (完整清单见§2.2) | `MultiPeopleDataset.__getitem__()` → `DataLoader(batch_size=B)` → `move_to(device)`; `BaseSceneModel.initialize()` 追加 `init_latent_pose` | RootOptimizer/SmoothOptimizer读GT+掩码; SMPLLoss读pose_prior锚点; run_prior读joints2d+vis_mask | `move_to()`→`run_opt()`返回, 全程GPU常驻 |
| `cam_data` | 数据加载 | Python dict | `{cam_R:(T,3,3), cam_t:(T,3), intrins:(T,4), static:bool}` — 世界到相机外参+内参 | `CameraData.as_dict()` 从cameras.npz加载后偏移/缩放 | `BaseSceneModel.initialize()` cam2world转换; `CameraParams.set_cameras()` 注册参数; `cam_util.reproject()` 2D重投影loss | 同obs_data, 全程GPU常驻 |
| `cameras.json` | 优化阶段内 | JSON | `{rotation:[[r00,...,r08],...](N,9), translation:[[tx,ty,tz],...](N,3), intrinsics:[[fx,fy,cx,cy],...](N,4)}` — 施加Y-flip `T=[[1,0,0],[0,-1,0],[0,0,-1]]` 后c2w展平 | `save_camera_json()`(`optim/output.py:168`) — `cam_data`→cam2world→Y-flip→写入 | 仅外部可视化/调试 | 不参与优化 |
| `track_info.json` | 优化阶段内 | JSON | `{tracks:{tid:{index:int, vis_mask:[...]}}, meta:{seq_interval:[s,e], data_interval:[s,e]}}` | `save_track_info()`(`optim/output.py`) — 记录轨道索引、vis_mask、区间 | 仅调试 | 不参与优化 |
| `hamer/*.npz` | 优化阶段内 | NPZ (简化) | `{pose_body:(B,T,15,3), trans:(B,T,3), root_orient:(B,T,3)}` — 轴角,相机空间, HaMeR原始预测 | `save_input_poses()`(`optim/output.py:66`) — 从`dataset.data_dict`提取写入 `{out_dir}/hamer/{seq}_000000_phalp_world_results.npz` | `run_vis.py` 渲染「优化前HaMeR输入」与优化后对比 | 不参与优化 |
| `init/*.npz` | 优化阶段内 | NPZ (14标准字段) | 见§1.3 — cam2world后、优化前的初始参数快照 | `save_initial_predictions()`(`optim/output.py:39`) — `base_model.get_optim_result()` → `{out_dir}/init/{seq}_000000_init_world_results.npz` | `run_vis.py` 渲染「初始化后世界空间姿态」与Stage 0/1对比 | 不参与优化 |
| `root_fit/*.npz` | 优化阶段内 | NPZ (14标准字段) | 见§1.3 — 每`save_every=20`轮保存 | `RootOptimizer.save_results()`(`optimizers.py:122`) — `get_optim_result()`→`detach().cpu()`→`np.savez()` → `{out_dir}/root_fit/{seq}_{iter:06d}_world_results.npz` | (1)`run_vis.py`渲染Stage 0; (2)Stage 1通过共享`BaseSceneModel.params`继承trans/root_orient(不直接读NPZ) | 持久化, 跨运行 |
| `root_fit_params.pth` | 优化阶段内 | PyTorch checkpoint | `BaseSceneModel.params` 全部 `nn.Parameter` 快照 (`get_dict()` 输出) | `RootOptimizer.save_checkpoint()`(`optimizers.py:103`) — 每次save_every+NaN回滚前调用 | `RootOptimizer.load_checkpoint()` Stage 0断点续跑; Stage 1不加载此文件(通过共享内存继承) | 持久化, 跨运行 |
| `smooth_fit/*.npz` | 优化阶段内 | NPZ (14标准字段) | 见§1.3 — Stage 1优化后的betas/latent_pose | `SmoothOptimizer.save_results()` — 同RootOptimizer, 写 `{out_dir}/smooth_fit/{seq}_{iter:06d}_world_results.npz` | (1)`run_vis.py`渲染最终结果; (2)`run_prior()`加载最高iter NPZ作为HMP输入 | 持久化, 跨运行 |
| `smooth_fit_params.pth` | 优化阶段内 | PyTorch checkpoint | 同root_fit_params | `SmoothOptimizer.save_checkpoint()` | `SmoothOptimizer.load_checkpoint()` Stage 1断点续跑 | 持久化, 跨运行 |
| `prior/*.npz` | 优化阶段内 | NPZ (14标准+2 HMP专属) | 标准字段+ `{decode_root:(B,T,3), poses:(B,T,48)}` — HMP解码根旋转+局部旋转矩阵转轴角 | `run_prior()`→`HMP/fitting.py:multi_stage_opt()` 潜码优化完成后保存世界空间精修结果 | `run_vis.py` 渲染HMP精修后最终结果 | 持久化, 跨运行 |
| `*_final_*.mp4` | 可视化 | MP4 (H.264) | 视频帧 — 支持多视角(src_cam, front, above, side)和多阶段 | `run_vis.py:animate_scene()`(`vis/output.py`) — 加载NPZ→`prep_result_vis()`→MANO前向→pyrender离屏渲染→`imageio`写视频 | 用户直接观看 | 按需生成 |
| `*_meshes/*.obj` | 可视化 | Wavefront OBJ | 每帧每手一个文件 `{t:06d}_{hand_idx}.obj` | `run_vis.py:save_meshes_all()`(`vis/output.py`) — `vertices_to_trimesh()` MANO顶点+面片→trimesh→导出 | Blender/Maya等外部3D软件离线渲染/动画 | 按需生成 |

### 1.5 run_opt 的处理单元与帧分类

`run_opt` 处理的最小单元**不是完整视频**，而是一个**子片段（sub-segment）**。该子片段由三层裁剪叠加确定。

#### 1.5.1 三层裁剪确定处理窗口

```
完整视频 (N 帧)
  │
  ├── 第 1 层: Shot 切分
  │     shot_idcs/{seq}.json → get_shot_img_files()
  │     筛选属于指定 shot_idx 的帧
  │     → shot_frames (N_shot 帧)
  │
  ├── 第 2 层: 用户指定的帧范围
  │     data.start_idx / data.end_idx (video_vipe.yaml)
  │     从 shot_frames 中切片 [start_idx, end_idx)
  │     end_idx=-1 表示到 shot 末尾
  │     → sel_frames (N_sel 帧)
  │
  └── 第 3 层: 轨道可见范围求并
        dataset.py:155-176
        遍历所有选中轨道的 track_vis_masks:
          对每个轨道找 [first_detection, last_detection]
          取所有轨道的最小 sidx 和最大 eidx
        → final_frames [sidx, eidx) (seq_len = eidx - sidx 帧)
```

代码位置：`dyn-hamr/data/dataset.py:106-177`。

**第 3 层的具体逻辑**（L155-176）：

```python
sidx = np.inf
eidx = -1
for pred_dir in self.track_dirs:
    has_kp = [os.path.isfile(f"{pred_dir}/{x}_keypoints.json") for x in self.img_names]
    idcs = np.where(has_kp)[0]
    if len(idcs) > 0:
        si, ei = min(idcs), max(idcs)
        sidx = min(sidx, si)    # 取所有轨道的最早首次出现
        eidx = max(eidx, ei)    # 取所有轨道的最晚最后出现

eidx = max(eidx + 1, 0)
sidx = min(sidx, eidx)
self.seq_len = eidx - sidx     # ← 最终优化窗口长度
```

**示例**：一个 500 帧的视频，shot_idx=0 包含全部 500 帧，`start_idx=100, end_idx=400`。轨道 0（左手）在帧 120-380 可见，轨道 1（右手）在帧 110-370 可见。则：
- `sidx = min(120, 110) = 110`，`eidx = max(380, 370) + 1 = 381`
- `seq_len = 381 - 110 = 271` 帧
- 帧 100-109（轨道首次出现前）和帧 381-399（轨道最后出现后）被裁剪掉

#### 1.5.2 帧的三类处理方式

优化窗口 `[sidx, eidx)` 内的每一帧，根据 vis_mask 值采用不同的处理策略：

| 帧状态 | vis_mask | 判定条件 | MANO 参数 | 2D 关键点 | 优化参与 |
|--------|----------|---------|-----------|----------|---------|
| **可见帧** | 1 | `t ∈ [track_s, track_e)` 且文件存在 | 从 JSON 直接读取（原始 HaMeR 预测） | 从 JSON 直接读取 | ✅ 全部 loss |
| **遮挡/检测失败** | 0 | `t ∈ [track_s, track_e)` 但文件不存在 | Slerp/线性插值 | 线性插值 + conf×0.8 | ✅ 全部 loss（与可见帧等同） |
| **出镜** | -1 | `t ∉ [track_s, track_e)` | 全零默认值（不被插值） | 全零默认值 | ❌ 排除（`vis_mask >= 0` 过滤） |

**注意**：`track_s` 和 `track_e` 是**每个轨道独立**的（`get_ternary_mask()` 逐轨道计算），而非全局统一。因此对于双向场景：
- 轨道 0（左手）可能在帧 120-380 有效 → `track_s=120, track_e=380`
- 轨道 1（右手）可能在帧 110-370 有效 → `track_s=110, track_e=370`
- 帧 115：轨道 0 无文件但 `115 ≥ 120` → wait，不对。让我重新思考。

实际上 `track_s` 和 `track_e` 是在 `load_data()` 中为每个轨道**独立**计算的（通过 `get_ternary_mask(vis_mask)`），而 `vis_mask` 是每个轨道独立从 `track_vis_masks[i][sidx:eidx]` 裁剪得到的。所以每个轨道有自己独立的 `[track_s, track_e)` 区间。

**完整的三层帧分类**：

```
帧在 [sidx, eidx) 范围内? → NO → 帧不在优化窗口内，不处理
                        → YES → 属于优化窗口
                                 │
                                 ├─ 对轨道 i，t ∈ [track_s_i, track_e_i)?
                                 │     ├─ YES + 文件存在 → vis_mask=1 (可见)
                                 │     └─ YES + 文件不存在 → vis_mask=0 (遮挡)
                                 │
                                 └─ t ∉ [track_s_i, track_e_i) → vis_mask=-1 (出镜)
```

#### 1.5.3 与配置命名的对应关系

输出目录名中的各字段直接反映了处理单元的范围（`config.yaml` 和 `video_vipe.yaml`）：

```
{data.name} = {seq}-{track_ids}-shot-{shot_idx}-{start_idx}-{end_idx}

例如: demo1-all-shot-0-100-400
       │     │    │    │   └─ 第 2 层: end_idx
       │     │    │    └─ 第 2 层: start_idx
       │     │    └─ 第 1 层: shot_idx
       │     └─ 轨道选择: "all" = 所有满足 MIN_TRACK_LEN 的轨道
       └─ 视频序列名
```

这解释了为什么每次 `run_opt` 调用处理的是一个精确限定的子片段——输出目录的唯一性依赖于这些参数。

#### 1.5.4 配置参数控制

| 配置项 | 文件 | 作用 |
|--------|------|------|
| `data.shot_idx` | `video_vipe.yaml` | 第 1 层：选择哪个 shot |
| `data.start_idx` / `data.end_idx` | `video_vipe.yaml` | 第 2 层：帧范围切片 |
| `data.track_ids` | `video_vipe.yaml` | 选择哪些轨道（`"all"` = 前 N 个满足长度要求的轨道） |
| `MIN_TRACK_LEN` | `dataset.py:33` | 轨道最小长度过滤（当前 = 60 帧） |
| `data.split_cameras` | `video_vipe.yaml` | 相机是否按 shot 独立处理 |

### 1.6 数据集加载：`get_dataset_from_cfg(cfg)`

代码：`run_opt.py:189` → `dataset.py:37-57`。

#### 1.6.1 调用入口与参数传递

```python
dataset = get_dataset_from_cfg(cfg)
```

`get_dataset_from_cfg` 从 Hydra 配置中提取数据参数，经过路径展开和预处理校验后，构造 `MultiPeopleDataset`：

```python
def get_dataset_from_cfg(cfg):
    args = cfg.data
    args.sources = expand_source_paths(args.sources)  # glob 展开通配符路径
    check_data_sources(args, cfg)  # 按需执行帧提取 + HaMeR + SLAM

    return MultiPeopleDataset(
        args.sources,                        # {images, cameras, tracks, shots} 路径字典
        args.seq,                            # 序列名 (如 "demo1")
        tid_spec=args.track_ids,             # 轨道选择 ("all" 或 "longest-N" 或 "000-001")
        shot_idx=args.shot_idx,              # 镜头索引 (0)
        start_idx=int(args.start_idx),       # 起始帧 (0)
        end_idx=int(args.end_idx),           # 结束帧 (-1=到末尾)
        is_static=cfg.is_static,             # 静态相机标志
        split_cameras=args.get("split_cameras", True),
    )
```

`check_data_sources` 在数据集构造前执行三层按需预处理（`dataset.py:74-84`）：
1. `preprocess_frames()` — 若图像目录为空，用 `ffmpeg` 从视频提取帧
2. `preprocess_tracks()` — 若 track 目录为空，运行 HaMeR 子进程
3. `preprocess_cameras()` — 若 camera 目录为空，运行 VIPE/DROID-SLAM

#### 1.6.2 `MultiPeopleDataset.__init__` 主要逻辑

代码：`dataset.py:87-184`。分为四个阶段：

**阶段 1 — Shot 与帧范围裁剪** (L106-120)：
```python
img_files, _ = get_shot_img_files(self.data_sources["shots"], shot_idx, pad_shot)
end_idx = end_idx if end_idx > 0 else len(img_files)
img_files = img_files[start_idx:end_idx]   # 用户指定的 [start, end) 范围
self.num_imgs = len(self.img_names)         # 选中帧数
```

**阶段 2 — 轨道发现与过滤** (L122-152)：
```python
track_ids = sorted(os.listdir(track_root))  # 枚举 {track_root}/ 下所有子目录
# 计算每个轨道在选中帧中的可见帧数
track_lens = [len(list(filter(os.path.isfile, paths))) for paths in track_paths]
# 过滤: 可见帧数 > MIN_TRACK_LEN (60帧)
track_ids = [tid for tid, length in zip(track_ids, track_lens) if length > 60]
track_ids = track_ids[:MAX_NUM_TRACKS]     # 最多 12 条轨道
```

**阶段 3 — 轨道可见范围求并** (L154-177)：
```python
sidx, eidx = np.inf, -1
for pred_dir in self.track_dirs:
    has_kp = [os.path.isfile(f"{pred_dir}/{x}_keypoints.json") for x in img_names]
    idcs = np.where(has_kp)[0]
    if len(idcs) > 0:
        sidx = min(sidx, min(idcs))   # 所有轨道最早的首次出现
        eidx = max(eidx, max(idcs))   # 所有轨道最晚的最后出现
self.seq_len = eidx - sidx + 1        # 最终优化窗口长度
```

**阶段 4 — 惰性数据缓存** (L182-184)：
```python
self.data_dict = {}    # 缓存 load_data() 的结果
self.cam_data = None   # 缓存相机数据
```

#### 1.6.3 数据加载：`load_data()` 与插值调用链

`load_data()` (L189-262) 在 `__getitem__` 首次被调用时执行，加载并插值所有数据。**MANO 参数和 2D 关键点的插值在此阶段完成。**

**相机加载** (L194)：
```python
self.load_camera_data()  → CameraData(cam_dir, seq_len, img_size, is_static, ...)
```

**2D 关键点加载 + 插值** (L225-240)：
```python
kp_paths = [f"{track_dirs[i]}/{x}_keypoints.json" for x in sel_img_names]
joints2d_data = load_keypoints_with_interp(kp_paths, interp=interp_input)
```

调用链: `load_keypoints_with_interp` (`data/tools.py:33-94`)
→ `read_keypoints()` 逐文件读 JSON → `np.stack` 得到 `(T, 21, 3)`
→ 构建 `vis_mask` (文件存在 + 非全零)
→ 逐关节独立 `interp1d(kind='linear')` 插值 x,y 坐标
→ 邻近均值×0.8 填充置信度
→ `dataset.py:235-239`: conf 强制设为 1.0 (覆写顺序错误)

**MANO 参数加载 + 插值** (L242-255)：
```python
pred_paths = [f"{track_dirs[i]}/{x}_mano.json" for x in sel_img_names]
pose_init, orient_init, trans_init, betas_init, is_right = load_mano_preds(
    pred_paths, tid=tid, interp=interp_input
)
```

调用链: `load_mano_preds` (`data/tools.py:137-168`)
→ `read_mano_preds()` 逐文件读 JSON → `np.stack` 得到 `(T, ...)`
→ `vis_idcs` = 文件存在的帧索引
→ 旋转: `Slerp(vis_idcs, Rotation.from_rotvec(orient[vis_idcs]))` — 球面线性
→ 平移: `interp1d(vis_idcs, trans[vis_idcs], axis=0)` — 线性
→ 形状: `interp1d(vis_idcs, betas[vis_idcs], axis=0)` — 线性
→ 手指关节: 逐关节独立 `Slerp` (15个关节各一个)
→ 范围: `[tmin, tmax)` = `[min(vis_idcs), max(vis_idcs)+1)`

**vis_mask 三值化** (L217-223)：
```python
vis_mask = self.track_vis_masks[i][sidx:eidx]
vis_mask = get_ternary_mask(vis_mask)  # -1=出镜, 0=遮挡, 1=可见
```

#### 1.6.4 数据流到优化器

`DataLoader(batch_size=B, shuffle=False)` 将所有 B 个轨道的数据堆叠为一个 batch (`run_opt.py:74-76`)：
```python
loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
obs_data = move_to(next(iter(loader)), device)  # (B, T, ...) → GPU
```

此时 `obs_data` 包含 11 个字段（见 §2.2），所有缺失帧已被插值填充，`vis_mask` 已三值化，数据全程驻留 GPU。

### 1.7 模型初始化：`BaseSceneModel(B, T, hand_model, pose_prior, **margs)`

代码：`run_opt.py:121-127` → `optim/base_scene.py:37-148`。

#### 1.7.1 构造参数

```python
margs = cfg.model                   # 来自 config.yaml 的 model 块
base_model = BaseSceneModel(
    B, T, hand_model, pose_prior, **margs
)
```

`cfg.model` (`config.yaml:6-10`) 展开为：

| 参数 | 值 | 含义 |
|------|----|------|
| `use_init` | True | 从 HaMeR 预测初始化 (非零起点) |
| `opt_cams` | False | 不优化相机外参 |
| `opt_scale` | True | 非静态相机优化 world_scale |
| `async_tracks` | True | 异步轨道 (未在 BaseSceneModel 中使用, MovingSceneModel 用) |

`__init__` 执行的核心操作 (`base_scene.py:37-70`)：

```python
def __init__(self, batch_size, seq_len, body_model, pose_prior,
             use_init=False, opt_cams=False, opt_scale=True, **kwargs):
    B, T = batch_size, seq_len
    self.body_model = body_model          # MANO 手部模型 (shared)
    self.hand_mean = body_model.hand_mean  # MANO pose_mean (仅 VPoser 加载时用)
    self.pose_prior = pose_prior           # None (VPoser 未加载)
    self.num_betas = body_model.num_betas  # 10
    self.use_init = use_init               # True
    self.opt_scale = opt_scale             # True (静态相机时被覆盖)
    self.opt_cams = opt_cams               # False
    self.params = CameraParams(batch_size) # 创建参数容器
```

**关键**：`self.params = CameraParams(batch_size)` 创建了一个空的 `nn.Module` 容器——此时尚未注册任何 `nn.Parameter`（所有参数在 `initialize()` 中注册）。

#### 1.7.2 `initialize(obs_data, cam_data)` — 参数注册与 cam2world 转换

代码：`base_scene.py:72-148`。分为相机注册、身体参数初始化和世界空间转换三步。

**步骤 1 — 相机参数注册** (L76-81)：
```python
self.params.set_cameras(cam_data, opt_scale=self.opt_scale,
    opt_cams=self.opt_cams, opt_focal=self.opt_cams)
```
→ `CameraParams.set_cameras()` (`params.py:80-127`)：
- 存储 `_cam_R(T,3,3)`, `_cam_t(T,3)` 为固定基础外参
- 从 `intrins(T,4)` 分离 `cam_center(T,2)` 和 `cam_f(T,2)`
- 若 `opt_scale=True`: 注册 `world_scale(1,1)=1.0` 为可优化参数
- 若 `opt_cams=True`: 注册 `delta_cam_R(T,3)=0`, `delta_cam_t(T,3)=0`

**步骤 2 — betas 初始化** (L86)：
```python
init_betas = torch.mean(obs_data["init_body_shape"], dim=1)  # (B,T,10) → (B,10)
```

**步骤 3 — latent_pose 初始化** (L88-90)：
```python
init_pose = obs_data["init_body_pose"][:, :, :15, :]           # (B,T,15,3)
init_pose_latent = self.pose2latent(init_pose)                  # 无VPoser→恒等, (B,T,45)
```

**步骤 4 — cam2world 转换 root_orient** (L96-106)：
```python
R_w2c, t_w2c = cam_data["cam_R"], cam_data["cam_t"]
R_c2w = R_w2c.transpose(-1, -2)                     # w2c 旋转求逆
t_c2w = -einsum("tij,tj->ti", R_c2w, t_w2c)         # w2c 平移转 c2w

init_rot_mat = angle_axis_to_rotation_matrix(init_rot)           # 轴角→矩阵
init_rot_mat = einsum("tij,btjk->btik", R_c2w, init_rot_mat)    # 左乘: 相机→世界
init_rot = rotation_matrix_to_angle_axis(init_rot_mat)           # 矩阵→轴角
```

**步骤 5 — cam2world 转换 trans** (L115-126)：
```python
# 先用初始 trans=0 跑一次 MANO 获取手腕关节位置
pred_data = self.pred_mano(zeros, init_rot, init_pose, is_right, init_betas)
root_loc = pred_data["joints3d"][..., 0, :]                     # 手腕关节 (B,T,3)

# 将 HaMeR 的相机空间平移转换到世界空间
init_trans = obs_data["init_trans"]                              # HaMeR cam_trans
init_trans = R_c2w @ (init_trans + root_loc) + t_c2w - root_loc
#                ↑ 先加手腕偏移得世界空间手腕位置
#                               ↑ 再减去世界空间手腕偏移得世界空间 root 平移
```

**步骤 6 — 注册所有参数** (L138-146)：
```python
self.params.set_param("init_body_pose", init_pose)               # (B,T,15,3) — 存档
self.params.set_param("latent_pose", init_pose_latent)            # (B,T,45) — 可优化
self.params.set_param("betas", init_betas)                        # (B,10) — 可优化
self.params.set_param("trans", init_trans)                        # (B,T,3) — 世界空间,可优化
self.params.set_param("root_orient", init_rot)                    # (B,T,3) — 世界空间,可优化
self.params.set_param("is_right", is_right, requires_grad=False)  # (B,T) — 固定
obs_data["init_latent_pose"] = init_pose_latent.detach()          # 存入 obs_data 供 pose_prior
```

#### 1.7.3 初始化后的参数状态

| 参数 | 形状 | 坐标系 | 可优化 | 初始值 |
|------|------|--------|--------|--------|
| `trans` | (B,T,3) | 世界空间 | ✅ | HaMeR cam_trans → cam2world |
| `root_orient` | (B,T,3) | 世界空间 | ✅ | HaMeR global_orient → cam2world |
| `betas` | (B,10) | — | ✅ | HaMeR init_body_shape 时间均值 |
| `latent_pose` | (B,T,45) | — | ✅ | HaMeR init_body_pose (恒等编码) |
| `is_right` | (B,T) | — | ❌ (固定) | HaMeR is_right |
| `init_body_pose` | (B,T,15,3) | — | ❌ (存档) | HaMeR 原始值 |
| `world_scale` | (1,1) | — | ✅ (条件) | 1.0 |
| `cam_f` | (T,2) | — | ❌ (条件) | cameras.npz intrins |

---

### 2.1 优化设计思路

Dyn-HaMR 采用**渐进式解冻**策略：通过 `Params.set_require_grads()` 控制梯度流，先优化高杠杆参数（全局轨迹），再解锁全部参数精修。

**动因**：平移和全局旋转是最高杠杆参数（6 DOF/帧）——决定手在 3D 空间中的位置。若同时优化手指姿态，关节的 2D 重投影误差会与手腕位置误差混淆，导致优化发散。

**损失权重作为阶段开关**：`confs/optim.yaml` 中所有 loss 权重为 3 元素列表（对应 stage 0/1/2），通过 `run_opt.py:93-98` 转置为每阶段独立权重字典：

```python
all_loss_weights = cfg.optim.loss_weights
assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
stage_loss_weights = [
    {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
]
```

权重为 0 的 loss 通过 `if weight > 0.0` 门控完全跳过。

**参数共享而非复制**：两个优化器操作同一个 `BaseSceneModel.params` 张量存储。Stage 0 的 `trans`/`root_orient` 最优值被 Stage 1 直接继承，仅 L-BFGS 的 Hessian 近似状态被重置（每个优化器独立的 checkpoint 文件）。

**L-BFGS 优于 SGD**：中规模优化（~10³ 参数）中全批次准牛顿方法收敛更快。`strong_wolfe` 线搜索保证充分下降。

### 2.2 obs_data 字段清单

由 `MultiPeopleDataset` 加载、`BaseSceneModel.initialize()` 扩展。

**来自数据集**（`dataset.py:__getitem__`）：

| 字段 | 形状 | 来源 | 含义 |
|------|------|------|------|
| `joints2d` | (B, T, 21, 3) | ViTPose → 线性插值 → conf=1.0 覆写 | 2D 关键点 (x,y,confidence), OpenPose 手部格式 |
| `init_body_pose` | (B, T, 15, 3) | HaMeR → Slerp 插值 | 15 个手指关节轴角(不含手腕) |
| `init_body_shape` | (B, T, 10) | HaMeR → 线性插值 | MANO betas, 时间平均得 (B,10) |
| `init_root_orient` | (B, T, 3) | HaMeR → Slerp 插值 | 手腕全局旋转轴角 (相机空间) |
| `init_trans` | (B, T, 3) | HaMeR → 线性插值 | 手腕平移 (相机空间) |
| `is_right` | (B, T) | HaMeR → 全轨道恒定 | 1=右手, 0=左手; 与 track_id 一致 |
| `vis_mask` | (B, T) | `get_ternary_mask()` | -1=出镜, 0=遮挡, 1=可见 |
| `seq_interval` | (2,) | 数据集内部 | `[start_idx, end_idx)` |
| `track_interval` | (2,) | 数据集中轨道首尾帧 | `[track_s, track_e)` |
| `track_id` | 标量 | 轨道 ID | 0=左手, 1=右手 |
| `seq_name` | 标量 | 配置 | 用于日志和可视化 |

**来自相机**（`CameraData.as_dict()`）：

| 字段 | 形状 | 含义 |
|------|------|------|
| `cam_R` | (T, 3, 3) | 世界到相机旋转 (第一帧平移被置零) |
| `cam_t` | (T, 3) | 世界到相机平移 |
| `intrins` | (T, 4) | `[fx, fy, cx, cy]` (按图像尺寸缩放) |
| `static` | bool | 静态相机标志 |

**由 `initialize()` 扩展**：

| 字段 | 形状 | 用途 |
|------|------|------|
| `init_latent_pose` | (B, T, D) | SMPLLoss pose_prior 的 GT 锚点 (`.detach()`) |

### 2.3 预测数据 (pred_data)

由 `BaseSceneModel.pred_params_mano()` → `pred_mano()` 产生，每次 L-BFGS closure 重新计算：

| 字段 | 形状 | 来源 |
|------|------|------|
| `joints3d` | (B, T, 16, 3) | MANO 前向: 16 标准关节 (1手腕+15手指) |
| `joints3d_op` | (B, T, 16, 3) | 同 joints3d (无 OpenPose 重映射) |
| `verts3d` | (B, T, 778, 3) | MANO 前向: 所有顶点 |
| `points3d` | (B, T, 778, 3) | 同 verts3d |
| `l_faces` | (F+14, 3) | 左手面片 (反转绕组 + 14 额外水密三角) |
| `r_faces` | (F+14, 3) | 右手面片 (标准绕组 + 14 额外水密三角) |
| `body_pose` | (B, T, 45) | MANO 输出的手部姿态轴角 |
| `is_right` | (B, T) | 手性 |

SmoothOptimizer 额外注入：`cam_R`, `cam_t` (from `get_extrinsics()`), 及所有当前参数值 (via `get_vars()`).

### 2.4 Stage 0: RootOptimizer (`root_fit`)

代码: `optim/optimizers.py:381-421`。实例化: `run_opt.py:150`。

```python
param_names = ["trans", "root_orient"]    # 仅优化 6 DOF/帧
self.loss = RootLoss(...)                  # 仅数据拟合项, 无先验
```

**优化变量详情**：

| 变量 | 形状 | 坐标系 | 含义 | 初始值来源 |
|------|------|--------|------|-----------|
| `trans` | (B, T, 3) | **世界空间** | 手腕关节在 3D 世界中的平移 | HaMeR `cam_trans` → cam2world 转换（`BaseSceneModel.initialize()` L115-126）：`R_c2w @ (init_trans + root_loc) + t_c2w - root_loc` |
| `root_orient` | (B, T, 3) | **世界空间** | 手腕关节的全局旋转（轴角, 3 DOF） | HaMeR `global_orient` → cam2world 旋转：`einsum("tij,btjk->btik", R_c2w, init_rot_mat)` → 转回轴角 |

**冻结变量**（`requires_grad=False`，保持 `initialize()` 写入的初始值）:

| 变量 | 形状 | 含义 |
|------|------|------|
| `betas` | (B, 10) | MANO 形状参数（HaMeR init_body_shape 时间均值, `initialize()` L86） |
| `latent_pose` | (B, T, D) | 手指关节潜变量（HaMeR init_body_pose 编码, D=45, `initialize()` L90） |
| `world_scale` | (1, 1) | 全局尺度因子（=1.0, `CameraParams.set_cameras()` L108-114） |
| `cam_f` | (T, 2) | 相机焦距（来自 `cameras.npz` intrins, `CameraParams.set_cameras()` L101-103） |
| `delta_cam_R` / `delta_cam_t` | (T, 3) / (T, 3) | 相机外参增量（=0, 仅 `opt_cams=True` 时注册, 当前默认 False） |

**活跃损失** (50 L-BFGS iterations):

| Loss | 权重 | GT 来源 | 预测来源 | 归一化 |
|------|------|---------|---------|--------|
| `joints2d` | 10000 | ViTPose 2D 关键点 | MANO 关节 → `cam_util.reproject()` | hand_scale 归一化 + GMoF(σ=100) |
| `joints3d_smooth` | 1000 | 自监督平滑 | 相邻帧 joints3d delta | hand_scale 归一化 |
| `depth_constraint` | 100 | 隐式 (min=0, max=999) | 相机空间深度 | 无 (米制) |

**前向计算**: `latent2pose()` → `pred_mano()` → `{joints3d, verts3d, ...}` → loss.

**冻结参数**: `betas`, `latent_pose`, `world_scale`, `cam_f`, `delta_cam_R` — 保持 HaMeR 初始值.

### 2.5 Stage 1: SmoothOptimizer (`smooth_fit`)

代码: `optim/optimizers.py:424-476`。实例化: `run_opt.py:157-160`。

```python
param_names = ["trans", "root_orient", "betas", "latent_pose"]
if model.opt_scale:  param_names += ["world_scale"]      # 非静态相机
if model.opt_cams:   param_names += ["cam_f", "delta_cam_R"]  # 默认 False
self.loss = SMPLLoss(...)  # RootLoss + 先验项
```

**优化变量详情**：

| 变量 | 形状 | 坐标系 | 含义 | 初始值来源 |
|------|------|--------|------|-----------|
| `trans` | (B, T, 3) | **世界空间** | 手腕关节在 3D 世界中的平移（与 Stage 0 相同） | Stage 0 优化结果（通过共享 `BaseSceneModel.params` 继承） |
| `root_orient` | (B, T, 3) | **世界空间** | 手腕关节的全局旋转（轴角, 3 DOF） | Stage 0 优化结果 |
| `betas` | (B, 10) | —（无坐标系, 形状参数） | MANO 形状系数, 控制手部胖瘦/比例 | HaMeR init_body_shape 时间均值（Stage 0 中冻结, 未被优化过） |
| `latent_pose` | (B, T, D) | —（无坐标系, 姿态参数） | 手指关节潜变量（D=45=15×3 轴角, 无 VPoser 时直接为轴角值） | HaMeR init_body_pose（Stage 0 中冻结, `pose2latent()` 恒等） |
| `world_scale` | (1, 1) | **世界空间缩放** | 全局尺度因子, 将 SLAM 的无量纲轨迹缩放到米制 | =1.0（`CameraParams.set_cameras()` L108）；仅 `opt_scale=True` 且非静态相机时优化 |
| `cam_f` | (T, 2) | —（像素单位） | 相机焦距 fx, fy | 来自 `cameras.npz` intrins；仅 `opt_cams=True` 时优化（默认 False） |
| `delta_cam_R` | (T, 3) | **相机自身坐标系** | 相机旋转的轴角增量（右乘 `cam_R @ dR`, 即相机系扰动） | =0；仅 `opt_cams=True` 时注册和优化（默认 False） |

**新增损失** (300 L-BFGS iterations):

| Loss | 权重 | GT 来源 | 归一化 |
|------|------|---------|--------|
| `pose_prior` | 1 | `obs_data["init_latent_pose"]` (HaMeR 初始值) | `sum((pred - init)^2)` |
| `shape_prior` | 0.05 | 零均值高斯先验 | `sum(betas^2) * nsteps` |
| `penetration` | 0 (禁用) | — | winding numbers 穿透检测 |

**权重变化**: `joints3d_smooth`: 1000 → **10000** (10×, Stage 1 会改变手指关节需要更强平滑).

**pose_prior 的特殊性**: 非 VPoser KL 散度——仅 L2 偏离 HaMeR 初始值 (`losses.py:1177-1193`).

### 2.6 优化流程抽象：从数据到梯度的闭环

Dyn-HaMR 的两个 L-BFGS 阶段共享一套统一的优化抽象，差异仅在于 `param_names`（控制优化变量范围）和 `stage_loss_weights[i]`（控制活跃损失项）。

#### 2.6.1 数据加载与常驻

数据加载在优化开始前**一次性完成**（`run_opt.py:76-77`），此后全阶段共享：

```python
obs_data = move_to(next(iter(loader)), device)   # (B, T, ...) → GPU
cam_data = move_to(dataset.get_camera_data(), device)
```

`DataLoader(batch_size=B, shuffle=False)` 将全部 B 个轨道的数据堆叠为一个 batch。`obs_data` 和 `cam_data` 在 GPU 显存中**常驻**整个优化生命周期——L-BFGS 的 closure 中直接读取这些张量，无磁盘 I/O。

#### 2.6.2 优化变量初始化

每阶段的 `set_opt_vars(param_names)` 执行两步操作（`optim/params.py:59-72`）：

1. **先冻结全部**：遍历 `self.param_names`（所有已注册的 `nn.Parameter`），设 `requires_grad = False`
2. **再解冻指定**：仅对 `param_names` 中的变量设 `requires_grad = True`

Stage 0 的参数初始值来自 `BaseSceneModel.initialize()`（HaMeR 预测 → cam2world）。Stage 1 的参数初始值来自 Stage 0 的优化结果——通过共享 `BaseSceneModel.params` 张量存储自动继承。

#### 2.6.3 损失函数计算：GT 和 pred 的获取链

每次 L-BFGS closure 中，优化器通过 `forward_pass(obs_data)` 计算损失。GT 和 pred 的获取路径：

**GT 来源**（观测数据，只读）：
```
obs_data["joints2d"]         ← ViTPose/HaMeR 2D 关键点, 经线性插值 + conf=1.0 覆写
obs_data["init_latent_pose"] ← BaseSceneModel.initialize() 存入的 HaMeR 初始值 (.detach())
obs_data["vis_mask"]         ← get_ternary_mask() 的三值掩码
```

**pred 来源**（MANO 前向链）：
```
pred_params_mano(is_right)                       # base_scene.py:240
  ├── latent2pose(self.params.latent_pose)       # 潜变量解码 (无VPoser时恒等)
  └── pred_mano(trans, root_orient, body_pose, is_right, betas)  # L198
        └── run_mano(body_model, ...)            # body_model/utils.py
              ├── 重塑 (B,T)→(B*T) → MANO(global_orient, hand_pose, betas, transl)
              ├── X 坐标镜像: joints[:,:,0] = (2*is_right-1) * joints[:,:,0]
              └── 返回 {joints3d:(B,T,16,3), verts3d:(B,T,778,3), points3d, ...}
```

**重投影**（仅 joints2d loss 需要）：
```
cam_util.reproject(pred_data["joints3d_op"], cam_R, cam_t, cam_f, cam_center)
  → p_c = R_w2c @ p_w + t_w2c    (世界→相机)
  → (x/z, y/z) * f + center       (透视投影 → 2D)
  → joints2d_pred (B, T, 21, 2)
```

#### 2.6.4 变量优化闭环

`StageOptimizer.optim_step()` 的 L-BFGS closure 形成完整闭环（`optim/optimizers.py:352-378`）：

```
closure():
  1. self.optim.zero_grad()                    # 清零梯度
  2. loss, stats, preds = self.forward_pass(obs_data)  # 前向 + 损失
     │
     ├── pred_data = model.pred_params_mano()   # MANO 前向
     └── loss = self.loss(obs_data, pred_data, vis_mask)  # 损失计算
  3. loss.backward()                            # 反向传播
     │
     └── 梯度从 scalar loss 流经:
           loss terms → pred_data → pred_mano → params (trans, root_orient, ...)
           仅 requires_grad=True 的参数得到 .grad
  4. return loss                                 # L-BFGS 使用此标量做线搜索

self.optim.step(closure)                        # L-BFGS: 内部多次调用 closure()
```

**关键**: L-BFGS 的 `step()` 内部会多次调用 `closure()`（每次线搜索迭代），因此 MANO 前向和损失计算在**每个 L-BFGS step 内被执行多次**（最多 `max_iter=20` 次函数评估）。

#### 2.6.5 有效帧掩码选取

Stage 0 和 Stage 1 **均处理完整序列**（B, T），不做窗口切分。帧的有效性仅通过 `vis_mask` 控制：

```python
# optimizers.py:419, 474 — 两阶段统一模式
vis_mask = obs_data["vis_mask"] >= 0   # -1(出镜)→False, 0(遮挡)→True, 1(可见)→True
loss, stats_dict = self.loss(obs_data, pred_data, vis_mask)
```

不同 loss 将掩码用于不同的过滤策略（详见 §3.2）：

- **`joints2d`**: `data[mask]` 索引过滤 → 物理删除无效样本，`(B,T,21,3)` → `(N_valid,21,3)`
- **`joints3d_smooth`**: `mask[:,1:] & mask[:,:-1]` → 仅相邻帧对双方都有效时计算 delta
- **`pose_prior`**: `loss[mask.bool()]` → 保留形状，仅对有效位置计算

**无窗口 padding**: Stage 0/1 不使用分块策略，因此不存在 padding 帧。vis_mask=-1 的自然边界起到了隐式的「窗口边界」作用——平滑 loss 在边界处自动切断（邻帧对掩码为 False）。

#### 2.6.6 Chunk 边界输出一致性

Dyn-HaMR 的 L-BFGS 阶段**不使用分块**——整个 `(B, T)` 序列在一个优化问题中联合求解，因此不存在 chunk 边界一致性问题。

对于可选的 HMP 阶段（`run_prior=True`），main 分支使用 `(N_chunks, 128)` 的固定 chunk 堆叠策略（`HMP/fitting.py:763-787`）：

```python
# HMP/fitting.py:763-787 — chunk 切分与堆叠
if v.shape[0] > 128:
    data_split = list(torch.split(v, 128))           # 按128帧切分
    if data_split[-1].shape[0] < 128:
        data_split[-1] = pad_to_128(...)              # 最后一段padding到128
    data[k] = torch.stack(data_split, dim=0)          # (N, 128, ...)
```

**跨 chunk 一致性通过 `lambda_batch_cs` 损失实现**（`HMP/fitting.py:1238-1251`）。在主优化循环的 closure 中：

```python
# 若 lambda_batch_cs > 0, 对相邻chunk边界位置施加一致性约束
if args.lambda_batch_cs > 0 and poses.shape[0] > 1:
    # chunk i 的最后一帧 与 chunk i+1 的第一帧 在重叠位置的输出应一致
    cs_loss = batch_consistency_loss(poses, ...)
    loss += args.lambda_batch_cs * cs_loss
```

`batch_consistency_loss` 惩罚相邻 chunk 在边界处的关节位置差异（chunk `i` 的末帧输出与 chunk `i+1` 的首帧输出应相等，因为它们描述同一物理帧的手部姿态）。这等价于在 chunk 边界施加硬性的连续性约束。

**注意**: 当前配置中 `hmp_config.yaml` 未显式设置 `lambda_batch_cs`（默认为 0 或未激活），因此 main 分支的 HMP 实际上**不施加** batch consistency loss——各 chunk 独立优化，padding 帧（当 `T < 128` 时重复末帧）被包含在 loss 中。

dev 分支通过滑动窗口 + overlap blending（`_stitch_hand_windows`）来解决此问题，替代了 batch_consistency 机制。

### 2.7 阶段间参数传递

两个优化器共享 `base_model` 对象（`run_opt.py:122, 150, 157`）：

```
Stage 0: RootOptimizer(base_model)
  → set_opt_vars(["trans", "root_orient"])
  → trans, root_orient 被优化, 其余冻结
  → checkpoint: root_fit_params.pth

Stage 1: SmoothOptimizer(base_model)    ← 同一个 base_model
  → set_opt_vars(["trans", "root_orient", "betas", "latent_pose"])
  → trans, root_orient 继承 Stage 0 的最优值
  → betas, latent_pose 从 HaMeR 初始值开始优化
  → checkpoint: smooth_fit_params.pth   ← 独立优化器状态
```

L-BFGS Hessian 近似在 Stage 切换时重新初始化（新建 `torch.optim.LBFGS`），但参数值保留。

### 2.8 L-BFGS 配置与早停

`optim/optimizers.py:52-54`:

```python
self.optim = torch.optim.LBFGS(
    self.opt_params, max_iter=20, lr=1.0, line_search_fn="strong_wolfe"
)
```

**早停机制** (`optimizers.py:319-343`):
- NaN 回滚: `np.isnan(cur_loss)` → 从 checkpoint 恢复并 raise
- 平台检测: 连续 `max_chunk_steps=20` 步 loss_change < 20，提前终止
- 零变化: `loss_change == 0` 直接终止

### 2.9 断点续跑与阶段分离

三阶段优化**支持分开运行和从中断处恢复**，依赖 `StageOptimizer` 的 checkpoint 机制。

#### 2.8.1 跳过已完成的阶段

`StageOptimizer.run()` 在进入优化循环前检查 checkpoint 进度（`optimizers.py:289-292`）：

```python
if self.cur_step >= num_iters:
    Logger.log(f"Checkpoint at {self.cur_step} >= {num_iters}, skipping")
    return
```

如果 checkpoint 记录的迭代计数已达到目标，整个阶段直接跳过。因此可以：
- 第一次运行完成 Stage 0 → `root_fit_params.pth` 记录 `cur_step=50`
- 第二次重新运行 → Stage 0 检测到 `cur_step≥50`，跳过；直接进入 Stage 1

#### 2.8.2 参数值跨阶段继承

两个优化器操作**同一个** `BaseSceneModel.params` 中的张量。`set_opt_vars()` 只改变 `requires_grad` 标志，不影响张量值。因此 Stage 0 优化后的 `trans`/`root_orient` 值被 Stage 1 自然继承。

Checkpoint 的加载逻辑（`optimizers.py:86-101`）：

```python
def load_checkpoint(self, out_dir, device=None):
    param_path = os.path.join(out_dir, f"{self.name}_params.pth")
    if os.path.isfile(param_path):
        param_dict = torch.load(param_path, map_location=device)
        self.model.params.load_dict(param_dict)   # 写入共享的 params

    optim_path = os.path.join(out_dir, f"{self.name}_optim.pth")
    if os.path.isfile(optim_path):
        optim_dict = torch.load(optim_path)
        self.optim.load_state_dict(optim_dict["optim"])   # 恢复 L-BFGS 状态
        self.cur_step = optim_dict["cur_step"]             # 恢复迭代计数
```

**关键细节**：`load_dict()` 写入的是**共享的** `BaseSceneModel.params`。由于 SmoothOptimizer 的 `smooth_fit_params.pth` 中也包含 `trans` 和 `root_orient`（它们在 Stage 1 同样被优化），如果该文件存在且加载，会覆写 Stage 0 在内存中的结果——但这等价于从 Stage 1 的 checkpoint 恢复，参数值是一致的。

#### 2.8.3 断点续跑场景

| 场景 | 操作 | 行为 |
|------|------|------|
| Stage 0 中断于 iter 30 | 重新运行（保留 `root_fit_params.pth`） | Stage 0: `load_checkpoint()` → 从 iter 30 恢复，参数和 L-BFGS 状态都还原 |
| Stage 0 中断于 iter 30 | 重新运行（删除 `root_fit_params.pth`） | Stage 0: checkpoint 不存在，跳过加载 → 从头（iter 0）开始 |
| Stage 0 完成, Stage 1 中断于 iter 150 | 重新运行 | Stage 0: `cur_step=50≥50`，跳过; Stage 1: 从 iter 150 恢复 |
| 只想重跑 Stage 1 | 保留 `root_fit_params.pth`，删除 `smooth_fit_params.pth` | Stage 0 跳过; Stage 1 `load_checkpoint()` 找不到文件 → 从头开始，`trans`/`root_orient` 继承 Stage 0 最优值 |
| 修改了 `optim.yaml` 权重 | 删除对应阶段的 `*_params.pth` 和 `*_optim.pth` | 该阶段从头优化，使用新权重 |
| Stage 0 完成, Stage 1 完成 | 重新运行（不删任何文件） | 两个阶段都跳过（`cur_step ≥ num_iters`），直接进入 `run_vis()` |

#### 2.8.4 不可跳过的前提

不能跳过 Stage 0 直接运行 Stage 1 **且不提供 Stage 0 结果**。因为 Stage 1 的初始 `trans`/`root_orient` 来自 Stage 0 优化后的值（或从 `root_fit_params.pth` 加载）。如果既未运行 Stage 0 也无 checkpoint 文件，`trans`/`root_orient` 保持 `BaseSceneModel.initialize()` 中的 HaMeR 初始值——此时 Stage 1 会从原始（未经 Stage 0 优化的）参数开始，2D 重投影误差较大，收敛速度受影响。

### 2.10 优化中参数与模型的内存/磁盘访问

优化过程中，参数和模型数据在 GPU 显存、CPU 内存和磁盘之间按需搬移。以下是各阶段的完整访问路径。

#### 2.10.1 数据加载 → GPU

`run_opt.py:76-78`:

```python
obs_data = move_to(next(iter(loader)), device)   # CPU → GPU
cam_data = move_to(dataset.get_camera_data(), device)
```

`DataLoader` 以 `batch_size=B`（轨道数）加载数据。`move_to()` 将 `obs_data` 中所有 tensor 和 `cam_data` 迁移到 GPU。此后整个优化过程中，`obs_data` 和 `cam_data` **常驻 GPU 显存**，不再搬移。

#### 2.10.2 模型初始化 → GPU

`run_opt.py:116, 127`:

```python
hand_model = MANO(batch_size=B*T, ...).to(device)    # 模型权重 → GPU
base_model.initialize(obs_data, cam_data)              # 参数初始化 → GPU
base_model.to(device)                                  # 所有 nn.Parameter → GPU
```

`BaseSceneModel.to(device)` 将 `self.params` 中的所有 `nn.Parameter`（`trans`, `root_orient`, `betas`, `latent_pose`, `world_scale` 等）迁移到 GPU。此后整个优化过程中，**参数常驻 GPU 显存**。

#### 2.10.3 优化循环中的内存访问

每次 L-BFGS closure 的调用链全部在 GPU 上完成：

```
GPU 显存中的常驻数据:
  obs_data (joints2d, init_body_pose, vis_mask, ...)  ← 只读
  cam_data (cam_R, cam_t, intrins)                     ← 只读
  self.params (trans, root_orient, betas, ...)          ← 读写 (梯度更新)

每次 closure:
  pred_params_mano() → MANO 前向 (GPU)
    → cam_util.reproject() → 2D 投影 (GPU)
    → RootLoss/SMPLLoss.forward() → loss 计算 (GPU)
    → loss.backward() → 梯度 (GPU)
    → L-BFGS 内部线搜索 → 参数更新 (GPU)

  无 CPU↔GPU 搬移
  无磁盘 I/O
```

#### 2.10.4 阶段切换时的磁盘 I/O

阶段切换（Stage 0 → Stage 1）不涉及额外的内存搬移——两者共享同一个 `base_model`，参数已经在 GPU 上。仅发生：

1. `RootOptimizer` 析构（L-BFGS 状态释放）
2. `SmoothOptimizer` 构造（新建 L-BFGS，`set_opt_vars()` 改变 `requires_grad` 标志）
3. 参数值保持不变（仍在 GPU 上）

磁盘写入仅在 **checkpoint 保存** 时触发（`save_every=20` 轮一次）：

```
save_checkpoint() → GPU 参数 → .detach().cpu() → torch.save() → 磁盘 .pth
save_results()    → GPU 参数 → .detach().cpu() → np.savez()   → 磁盘 .npz
```

磁盘读取仅在 **checkpoint 加载** 时触发（每个 `run()` 开始时）：

```
load_checkpoint() → torch.load() 磁盘 .pth → CPU tensor → .to(device) → GPU
```

#### 2.10.5 HMP 阶段的独立性

HMP（`run_prior()`）**不访问 GPU 上的 `BaseSceneModel`**。它从磁盘 `smooth_fit/*.npz` 加载 Stage 1 结果到 CPU，在自己的 `Architecture` 模型中进行优化：

```
磁盘 smooth_fit/*.npz → np.load() → CPU tensor → HMP model (GPU)
```

HMP 使用 Adam 优化器（而非 L-BFGS），优化潜码 `z_l` 和 `z_g`，而非直接优化 MANO 参数。HMP 结果写入 `prior/*.npz` 后，`run_vis()` 读取该文件做可视化。

#### 2.10.6 数据驻留总结

| 数据 | 位置 | 生命周期 |
|------|------|---------|
| `obs_data` | GPU | 从 `move_to()` 到 `run_opt()` 返回 |
| `cam_data` | GPU | 同上 |
| `hand_model` (MANO) | GPU | 同上 |
| `BaseSceneModel.params` | GPU | 同上 |
| L-BFGS 内部状态 (Hessian) | CPU/GPU (torch 管理) | 阶段内 |
| checkpoint `.pth` | 磁盘 | 持久化，跨运行 |
| 结果 `.npz` | 磁盘 | 持久化，供可视化/HMP 消费 |

### 2.11 阶段对比速查

| 维度 | Stage 0 (RootOptimizer) | Stage 1 (SmoothOptimizer) |
|------|------------------------|---------------------------|
| 优化变量 | trans, root_orient | trans, root_orient, betas, latent_pose |
| 冻结变量 | betas, latent_pose, world_scale, cam_* | world_scale, cam_* |
| 迭代次数 | 50 | 300 |
| Loss 函数 | RootLoss | SMPLLoss (= RootLoss + priors) |
| joints2d 权重 | 10000 | 10000 |
| joints3d_smooth | 1000 | 10000 |
| depth_constraint | 100 | 100 |
| pose_prior | 未激活 | 1 (L2 → HaMeR init) |
| shape_prior | 未激活 | 0.05 × seq_len |
| 参数初始值 | HaMeR → cam2world | Stage 0 结果 |
| 优化器状态 | 新建 L-BFGS | 新建 L-BFGS |
| 全序列处理 | 是 (B,T 整段) | 是 (B,T 整段) |

---

## 3. 重要字段约定

### 3.1 vis_mask 构建逻辑

`vis_mask` 基于**磁盘文件存在性**（`dyn-hamr/data/dataset.py:157-169, 402-411`），无视觉语义。

**阶段 A** — 文件存在检查: `track_vis_masks` bool 数组 (True = `{tid}/{frame}_keypoints.json` 存在).

**阶段 B** — 三值化 (`get_ternary_mask()`, L402-411, 仅 10 行):

```python
vis_idcs = torch.where(vis_mask)[0]
track_s, track_e = min(vis_idcs), max(vis_idcs) + 1
vis_mask[:track_s] = -1    # 出镜
vis_mask[track_e:] = -1    # 出镜
# [track_s, track_e) → True→1.0(可见), False→0.0(遮挡)
```

**三值语义**: -1=出镜 (排除), 0=遮挡/检测失败 (参与优化), 1=可见 (参与优化).

**无法区分**物理遮挡和检测器漏检 (合并为 vis_mask=0).

### 3.2 vis_mask 在优化中的掩码

优化器入口二值化 (`optimizers.py:419, 474`): `vis_mask >= 0` → {-1→False, 0→True, 1→True}.

各 Loss 掩码方式:

| Loss | 掩码方式 | 效果 |
|------|---------|------|
| `joints2d` | 索引过滤 `data[mask]` | 物理删除无效样本, (B,T,21,3)→(N_valid,21,3) |
| `joints3d_smooth` | `mask[:,1:] & mask[:,:-1]` | 邻帧对双方有效才计算 delta |
| `pose_prior` | `loss[mask.bool()]` bool索引 | 仅有效帧惩罚偏离 |
| `bio_loss` | `joints[valid_mask]` | 仅有效帧计算 |
| `penetration` | 不使用掩码 | 所有帧 |
| `shape_prior` | 不使用掩码 | betas 是全局参数, 无需逐帧掩码 |
| `depth_constraint` | 不使用掩码 | 全零默认值帧不会触发惩罚 |

**joints2d 重投影损失的完整掩码链**:
```
vis_mask (B,T) ∈ {-1,0,1} → >=0 → (B,T) bool
  → Joints2DLoss.forward → 索引过滤 data[mask] → (N_valid, 21, 3)
  → conf^2 加权 (conf=1.0, 退化为等权重)
  → GMoF(normalized_error, σ=100)
```

### 3.3 检测框与 2D 关键点的耦合

HaMeR 的 3-pass 架构中 Pass 3 同时生成 `_mano.json` 和 `_keypoints.json`——**成对出现，要么同时存在要么同时不存在**。不存在独立的「检测框有效性」标志——完全由 JSON 文件系统存在性隐式表达。

在 YOLO 版 `run.py` 中，ViTPose 实际**未被使用**——2D 关键点由 HaMeR 自身 MANO 3D→2D 投影产生 (conf 强制 = 1.0)。

### 3.4 插值触发条件

```
帧 t 的手部数据状态:
文件存在? → YES: vis_mask=1, 直接读取
         → NO:  t 在 [track_s, track_e) 内?
                    → YES: vis_mask=0, Slerp/线性插值, 参与优化
                    → NO:  vis_mask=-1, 全零默认值, 排除
```

插值仅发生在: 文件不存在 **且** `track_s ≤ t < track_e`.

### 3.5 检测框存在但预测错误

四种情况导致 vis_mask=1 但数据质量差:
1. **YOLO 误检** (非手物体) → HaMeR 无「非手」判断 → 数值合法但语义错误的参数
2. **部分遮挡** → MANO 关节坍塌 (hand_scale < 5px) → Dyn-HaMR loss 返回 1e6 + checkpoint 回滚 (被动补救)
3. **极端视角** → 手腕在相机后方 → depth_constraint 惩罚 + GMoF 截断
4. **patience 插值 bbox 不精确** → 裁剪图像不完整 → 预测退化

---

## 4. 插值方法

### 4.1 2D 关键点坐标插值

代码: `dyn-hamr/data/tools.py:33-94` → `load_keypoints_with_interp()`.

**x, y 坐标**: 逐关节独立 `interp1d(kind='linear', bounds_error=False)`. 21 个关节各有独立插值器. 插值范围 `[tmin, tmax)` (轨道跨度内). `bounds_error=False` 保证不触发外推 (因为 `times` 严格在 `[min, max)` 内).

**与 MANO 旋转 Slerp 的差异**: 2D 关键点在欧氏空间中, 线性插值等价于匀速运动假设.

### 4.2 2D 关键点置信度插值

**非真正插值**——离散最近邻查询 + 折扣:

```python
# 双侧: (conf_prev + conf_next) / 2 * 0.8
# 单侧: conf_neighbor * 0.8
```

逐关节独立. 0.8 折扣标记「插值结果不如原始检测可靠」.

**dataset.py 覆写逻辑错误** (L235-239):
```python
joints2d_data[:, :, 2] = 1.0                          # 先覆写为 1.0
joints2d_data[conf < MIN_KEYP_CONF] = 0               # 后检查 (conf已=1.0, 永远为False)
```
顺序错误导致置信度加权机制完全失效——所有帧 conf=1.0.

### 4.3 MANO 参数插值

代码: `dyn-hamr/data/tools.py:112-168` → `load_mano_preds()`.

- **旋转** (global_orient, body_pose): `scipy.spatial.transform.Slerp` (球面线性, 保持 SO(3))
- **平移** (cam_trans): `interp1d(kind='linear')` (线性)
- **形状** (betas): `interp1d(kind='linear')` (线性)
- **范围**: `[tmin, tmax)` (仅轨道跨度内)
- **粒度**: 全局方向 1 个 Slerp + 每关节 (15) 独立的 Slerp

### 4.4 插值对比

| 维度 | 2D 关键点坐标 | 2D 关键点置信度 | MANO 旋转 | MANO 平移/形状 |
|------|-------------|---------------|-----------|---------------|
| 方法 | `interp1d(linear)` | 邻近均值×0.8 | `Slerp` | `interp1d(linear)` |
| 粒度 | 逐关节独立 (21) | 逐关节独立 (21) | 全局+逐关节 (16) | 全局 |
| 范围 | `[tmin,tmax)` | `[tmin,tmax)` | `[tmin,tmax)` | `[tmin,tmax)` |
| 默认值 | 全零 (21,3) | 全零 | 全零 | 全零 |
| 后处理 | conf强制1.0 (顺序错误) | 被覆写 | 无 | 无 |

---

## 5. 配置文件系统设计

### 5.1 Hydra/OmegaConf 架构

入口: `run_opt.py:172`: `@hydra.main(version_base=None, config_path="confs", config_name="config.yaml")`.

Hydra 行为:
1. 加载 `confs/config.yaml` → 处理 `defaults` (data: HOT3D, optim) + OmegaConf 插值 (`${...}`)
2. 切换到 Hydra 输出目录 (`chdir: True`)
3. 将解析后的 `cfg: DictConfig` 传给 `main()`

OmegaConf resolver: `OmegaConf.register_new_resolver("eval", eval)` 允许 `${eval:...}` 动态求值.

### 5.2 配置层次

```
config.yaml                  ← 顶层, 定义 defaults + model + paths + hydra
  ├── data/{type}.yaml       ← 数据配置 (video_vipe, video_driod, HOT3D, ...)
  │     └── sources: 路径 + 范围
  ├── optim.yaml              ← 优化器配置 + 3 阶段 loss 权重
  │     ├── options: L-BFGS 参数
  │     ├── root/smooth/motion_chunks: 迭代次数
  │     └── loss_weights: [stage0, stage1, stage2] 3 元素列表
  └── init.yaml               ← 单独的初始化管线配置
```

### 5.3 路径解析 (`util/loaders.py`)

`resolve_cfg_paths(cfg)`: 遍历 `cfg.paths` 项, 相对路径前加 `ROOT_DIR` (项目根目录的绝对路径). 不能以 `/` 开头.

### 5.4 关键配置项

**`config.yaml`**:
- `model.use_init: True` — 从 HaMeR 预测初始化 (非零起点)
- `model.opt_cams: False` — 不优化相机外参
- `model.opt_scale: True` — 非静态相机优化 world_scale
- `run_prior: False` — HMP 默认禁用

**`optim.yaml`**:
- `options.lr: 1.0, lbfgs_max_iter: 20` — L-BFGS 配置
- `root.num_iters: 50, smooth.num_iters: 300`
- `loss_weights` 核心项: `joints2d: [10000,10000,10000]`, `joints3d_smooth: [1000,10000,0]`, `pose_prior: [1,1,1]`

**数据配置**: `data.name = ${data.seq}-${data.track_ids}-shot-${data.shot_idx}-${data.start_idx}-${data.end_idx}` — 自动组装唯一输出目录名.

---

## 6. 相机坐标系约定

### 6.1 存储格式

**w2c (world-to-camera)** 存储在 `cameras.npz` → `CameraData` → `cam_R (T,3,3)`, `cam_t (T,3)`.

```python
p_camera = R_w2c @ p_world + t_w2c    # 左乘约定
```

### 6.2 世界原点

Dyn-HaMR 主动将第一帧平移置零 (`dataset.py:366`):
```python
t0 = -cam_t[sidx:sidx+1] + torch.randn(3) * 0.1
self.cam_t = cam_t[sidx:eidx] - t0
```
第一帧相机平移 ≈ 零 (加微小随机偏移). 旋转**不归一化** (保持 SLAM 输出的原始方向).

### 6.3 cam2world 转换

`BaseSceneModel.initialize()` (`optim/base_scene.py:97-126`):
```python
R_c2w = R_w2c.transpose(-1, -2)           # 旋转: 转置即求逆
t_c2w = -einsum("tij,tj->ti", R_c2w, t_w2c)  # 平移: -R^T @ t

# 手部参数从相机空间转换到世界空间:
R_world = einsum("tij,btjk->btik", R_c2w, R_camera)
t_world = R_c2w @ (t_camera + root_loc) + t_c2w - root_loc
```

### 6.4 左乘/右乘约定

**统一左乘** (`R @ x`). 唯一例外: `CameraParams.get_extrinsics()` (`params.py:141`) 中 `cam_R @ dR` (右乘)——delta 旋转在相机自身坐标系中的扰动 (仅 `opt_cams=True` 时激活).

### 6.5 关键转换函数

| 函数 | 文件 | 乘法模式 | 功能 |
|------|------|---------|------|
| `reproject()` | `geometry/camera.py` | 左乘 `einsum("btij,btnj->btni",...)` | 3D→2D 批量重投影 |
| `invert_camera()` | `geometry/camera.py` | 左乘 `-R.T @ t` | w2c ↔ c2w |
| `compose_cameras()` | `geometry/camera.py` | `R1 @ R2`, `t1 + R1 @ t2` | 外参复合 |
| `CameraData.cam2world()` | `data/dataset.py` | `R.T, -R.T @ t` | 存储的 w2c → c2w |
| `BaseSceneModel.initialize()` | `optim/base_scene.py` | `R_c2w @ R_cam` + offset | 相机→世界 手部参数 |
| `CameraParams.get_extrinsics()` | `optim/params.py` | **右乘** `R @ dR` | 相机外参 delta 扰动 |

### 6.6 VIPE 输入处理

VIPE 输出 c2w → `np.linalg.inv(c2w)` 转 w2c → 保存为 cameras.npz (`data/vidproc.py:111`).

---

## 7. 重要模型说明

### 7.1 MANO 前向

**初始化** (`run_opt.py:116`): `MANO(batch_size=B*T, pose2rot=True, **mano_cfg)`.

关键参数:
- `batch_size=B*T` — 每 (track, frame) 独立批量元素
- `pose2rot=True` — 输入轴角自动转旋转矩阵
- `GENDER: neutral`, `NUM_HAND_JOINTS: 15`

**左手处理**: 共用同一个 `MANO_RIGHT.pkl`. X 坐标镜像 (`run_mano()` 中 `joints[:,:,:,0] = (2*is_right-1)*joints[:,:,:,0]`). 面片绕组反转 (`l_faces = r_faces[:, [0,2,1]]`).

**额外面片**: 14 个额外三角形封闭手腕截面 (水密, 用于穿透损失).

**前向流程**: `latent2pose(latent_pose)` → 重塑 `(B,T,45)→(B*T,45)` + expand betas → `body_model(hand_pose, betas, global_orient, transl)` → 输出 joints `(B*T,21,3)` + verts `(B*T,778,3)` → 重塑回 `(B,T,...)` + X 坐标镜像.

### 7.2 VPoser (当前禁用)

`run_opt.py:53-54` 导入了 VPoser 的加载函数和模型类：

```python
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
```

但在 `run_opt.py:105-107`，**加载调用被显式注释掉**，代之以 `None`：

```python
# pose_prior, _ = load_model(paths.vposer, model_code=VPoser,
#     remove_words_in_model_weights='vp_model.', disable_grad=True)
# pose_prior = pose_prior.to(device)
pose_prior = None
```

配置中指定的路径 `paths.vposer` 指向 `VPoser/pretrained/Vposer_right_mirrored`，通过 `resolve_cfg_paths()` 解析为项目根目录下的绝对路径。加载代码本身是完整且可用的——`load_model()` 会从该路径加载最优 snapshot、实例化 VPoser 模型、加载权重并设为 eval 模式——只是被主动禁用了。

**禁用原因推测**：VPoser 是针对**全身 SMPL 人体姿态**训练的（基于 AMASS 数据集），其潜空间（latentD）编码的是 21 个身体关节 + 2×15 个手部关节的联合分布。对于 Dyn-HaMR 的**纯手部场景**（仅 15 个手指关节），VPoser 的手部先验分量在统计上不匹配，且额外引入了不必要的潜变量维度。HMP（针对手部运动训练的 NeMF 先验）完全替代了 VPoser 的功能。当前的 pose_prior 退化为对 HaMeR 初始值的简单 L2 正则化。

影响:
- `latent2pose()` 退化为恒等: `latent_pose` 即 body_pose 轴角 (D=45)
- `latent_pose_dim` 未设置 (从 VPoser 的 latentD)
- `pose_prior` loss 权重在 config 中存在但无效果 (loss 中的 VPoser KL 路径未激活)

### 7.3 HMP 运动先验

入口: `run_prior()` (`HMP/fitting.py:1637`), 在 `run_opt.py:166` 可选调用.

**核心组件** (`HMP/nemf/`):

| 组件 | 文件 | 功能 |
|------|------|------|
| `Architecture` | `generative.py:54` | VAE 主模型, 封装 encoder + NeMF + (可选) GMP |
| `LocalEncoder` | `prior.py:8` | 图卷积骨骼编码器 → 局部潜码 z_l (1024维) |
| `GlobalEncoder` | `prior.py:85` | 1D 残差块编码器 → 全局潜码 z_g (256维) |
| `NeuralMotionField` | `neural_motion.py` | 时间条件超网络 MLP (11层), 生成运动序列 |
| `ForwardKinematicsLayer` | `fk.py` | 可微 FK, 基于 MANO_RIGHT.pkl 骨骼结构 |

**流程**: 加载 smooth_fit 结果 → `fitting_prior()` → `multi_stage_opt()` → 逐轨道优化:
1. 数据准备: MANO 结果 → 6D 旋转 + FK → pos/velocity/angular
2. 编码: `encode_local()` + `encode_global()` → `(z_l, z_g)`
3. 多阶段潜码优化 (Adam, 运动先验正则化 `lambda=200`):

```python
def motion_prior_loss(latent_motion_pred):
    return torch.mean(latent_motion_pred**2)  # L2 → 单位高斯先验
```

4. 解码: `NeuralMotionField(t, z_l, z_g)` → rot6d → 旋转矩阵 → FK → MANO forward → reprojection loss

`NeuralMotionField` 架构: 时间 t (positional encoding) + z_l (1024) + z_g (256) → 11 层带跳跃连接的 MLP → local_output (144=16×6+16×3) + global_output (6).

---

## 8. 重要类清单

六个核心类在优化中各司其职，形成一条清晰的职责链：

```
MultiPeopleDataset                              ← 数据供应者: 磁盘→插值→GPU张量
    │  obs_data + cam_data
    ▼
CameraData                                      ← 相机容器: cameras.npz→w2c+intrins
    │  cam_data
    ▼
BaseSceneModel                                  ← 前向执行器: initialize()注册参数+pred_mano()
    │  self.params (CameraParams)                ← 参数存储: nn.Parameter张量+梯度控制
    │  pred_data
    ▼
StageLoss (RootLoss / SMPLLoss)                ← 损失计算: pred+obs→标量loss
    │  loss
    ▼
StageOptimizer (RootOptimizer / SmoothOptimizer) ← 优化控制器: L-BFGS+closure+checkpoint
```

| 类 | 角色 | 做什么 | 不做什么 |
|----|------|--------|---------|
| **`MultiPeopleDataset`** | 数据供应者 | 从磁盘加载 JSON/NPZ → 插值缺失帧 → 构建 `vis_mask` → `DataLoader` 堆叠为 GPU batch | 不参与优化循环，不感知 MANO 和 L-BFGS |
| **`CameraData`** | 相机容器 | 从 `cameras.npz` 加载 w2c+intrins → 首帧平移置零 → 缩放内参匹配图像尺寸 | 不参与梯度计算，不持有优化状态 |
| **`Params` / `CameraParams`** | 参数存储与梯度控制 | 持有所有 `nn.Parameter` 张量；`set_require_grads()` 先冻结全部再解冻指定参数；封装 world_scale 缩放和 delta 相机扰动 | 不执行前向计算，不持有优化器 |
| **`BaseSceneModel`** | 前向执行器 | 持有 MANO 模型和 CameraParams；`initialize()` 完成 cam2world 转换和参数注册；`pred_params_mano()` 执行 MANO 前向输出 `pred_data` | 不控制梯度，不执行 `backward()` 或 `optim.step()` |
| **`StageLoss` / `RootLoss` / `SMPLLoss`** | 损失计算 | 接收 `pred_data`+`obs_data`，计算各项损失；通过权重门控决定活跃损失项 | 不持有优化器，不执行 `backward()` |
| **`StageOptimizer` / `RootOptimizer` / `SmoothOptimizer`** | 优化控制器 | 持有 L-BFGS 实例；`set_opt_vars()` 控制优化范围；closure 串联 forward_pass→loss→backward 闭环；管理 checkpoint 和 NPZ 导出 | 不持有参数（持有对 CameraParams 的引用），不执行 MANO 前向（委托给 BaseSceneModel） |

一次 L-BFGS closure 的完整调用链：

```
StageOptimizer.optim_step()
  └─ closure()
       ├─ StageOptimizer.forward_pass(obs_data)
       │    ├─ BaseSceneModel.pred_params_mano(is_right)
       │    │    ├─ latent2pose(CameraParams.latent_pose)          [Params 读取]
       │    │    └─ pred_mano(trans, root_orient, body_pose, ...)  [MANO 前向]
       │    └─ StageLoss.forward(obs_data, pred_data, vis_mask)    [Loss 计算]
       ├─ loss.backward()                                           [梯度→Params]
       └─ return loss → L-BFGS 线搜索 → self.optim.step()          [参数更新→Params]
```

**核心设计原则**：每个组件仅做一件事。`Params` 不管前向，`BaseSceneModel` 不管梯度，`StageLoss` 不管优化器，`StageOptimizer` 不管 MANO——通过共享 `CameraParams` 实例来协同。这种设计使得在 Stage 0 和 Stage 1 之间切换时，只需改变 `StageOptimizer` 的 `param_names` 列表和 `StageLoss` 的权重字典，其他组件完全不变。

### 8.0 分层架构设计

Dyn-HaMR 的优化架构是一种**六层单向分层抽象**，每层只依赖下一层的接口，不跨层访问内部状态。

#### 8.0.1 分层模型

```
┌──────────────────────────────────────────────────────────────┐
│ StageOptimizer              ← 优化控制层                      │
│ 「优化什么参数、用哪种优化器、何时保存 checkpoint」             │
│ 依赖: BaseSceneModel (前向) + StageLoss (损失)                │
│       + CameraParams (通过 set_require_grads 控制梯度)         │
├──────────────────────────────────────────────────────────────┤
│ StageLoss                   ← 损失计算层                      │
│ 「给定 pred 和 obs, 计算多少惩罚」                             │
│ 依赖: obs_data (只读) + pred_data (从 BaseSceneModel 传入)     │
├──────────────────────────────────────────────────────────────┤
│ BaseSceneModel              ← 前向执行层                      │
│ 「给定参数值, 运行 MANO 得到 joints/verts」                    │
│ 依赖: CameraParams (参数源) + body_model (MANO 模型)           │
├──────────────────────────────────────────────────────────────┤
│ CameraParams                ← 参数存储与梯度控制层             │
│ 「哪些参数存在、哪些参数当前被优化」                             │
│ 依赖: 无 (纯数据容器, 继承 nn.Module)                          │
├──────────────────────────────────────────────────────────────┤
│ MultiPeopleDataset          ← 数据供应层                      │
│ + CameraData                                                │
│ 「从磁盘加载什么、如何插值缺失帧、如何掩码」                     │
│ 依赖: 磁盘文件 (JSON/NPZ)                                      │
├──────────────────────────────────────────────────────────────┤
│ obs_data / cam_data         ← 数据层 (Python dict)            │
│ 「观测数据是什么形状、什么值」                                   │
│ 依赖: 无 (纯数据)                                              │
└──────────────────────────────────────────────────────────────┘
```

#### 8.0.2 分层动机

**`MultiPeopleDataset` 与 `CameraData` 的分离**：相机参数是轨道无关的——同一视频片段中左右手共享同一组相机位姿。MANO 参数和 2D 关键点是逐轨道独立的——每只手有各自的 `vis_mask`、`init_body_pose`、`is_right`。两者分离后 `cam_data` 不需要在 `DataLoader` 中被冗余复制 B 份，也不需要混入 `__getitem__` 的返回内容中。

**为什么 MANO 前向放在 `BaseSceneModel`**：`pred_mano()` 需要同时访问 `CameraParams`（参数源）和 `body_model`（MANO 模型），而 `BaseSceneModel` 是唯一同时持有这两者的类。`latent2pose()` 需要 `self.pose_prior` 和 `self.hand_mean`——这两者也绑定在 `BaseSceneModel` 上。这种设计使得 MANO 前向对所有优化阶段都是相同的——无论 Stage 0 还是 Stage 1，`pred_params_mano()` 的行为完全一致，差异仅在于 `CameraParams` 中哪些参数有 `requires_grad`。

#### 8.0.3 设计原则

**单向依赖，无循环**：`StageOptimizer` → `BaseSceneModel` → `CameraParams`。数据从下层流向上层（`obs_data` → `StageLoss`），控制从上层的 `set_require_grads()` 流向下层（改变了 `CameraParams` 中张量的一个标志位）。

**每层接口极窄**：

| 层 | 对外接口 | 接口类型 |
|----|---------|---------|
| `MultiPeopleDataset` | `__getitem__` + `__len__` | PyTorch Dataset 协议 |
| `CameraParams` | `set_require_grads` + `get_cameras` + `get_dict/load_dict` | 梯度控制 + 参数序列化 |
| `BaseSceneModel` | `pred_params_mano` + `get_optim_result` | 前向计算 + 快照导出 |
| `StageLoss` | `forward(obs_data, pred_data, vis_mask)` → `scalar loss` | 标量损失 |
| `StageOptimizer` | `set_opt_vars` + `optim_step` + `save/load_checkpoint` | 优化控制 |

**共享状态的最小化**：整个架构中唯一的共享状态是 `CameraParams` 中的 `nn.Parameter` 张量——`BaseSceneModel` 读取它们做前向，`StageOptimizer` 通过 L-BFGS 更新它们。其余数据流（`obs_data`, `pred_data`, `loss`）都是单向传递，不产生副作用。

**可替换性**：因为每层只依赖接口而非实现，可以独立替换：

| 替换内容 | 影响范围 | 示例 |
|---------|---------|------|
| 换损失函数 | 只改 `StageLoss` 子类 | `RootLoss` → `SMPLLoss` |
| 换优化器 | 只改 `StageOptimizer` 子类 | 注释中的 `MotionOptimizer` |
| 换参数策略 | 只改 `CameraParams` | 固定外参 → 可优化 `delta_cam_R` |
| 换数据来源 | 只改 `MultiPeopleDataset` | JSON 文件 → 数据库 |
| 换身体模型 | 只改 `BaseSceneModel.body_model` | MANO → 其他手部模型 |

这种分层使得 `run_opt.py` 中 Stage 0 和 Stage 1 的差异被压缩到两行配置：

```python
# Stage 0
param_names = ["trans", "root_orient"]
self.loss = RootLoss(stage_loss_weights[0], ...)

# Stage 1
param_names = ["trans", "root_orient", "betas", "latent_pose"]
self.loss = SMPLLoss(stage_loss_weights[1], ...)
```

其余 200+ 行优化基础设施代码（`forward_pass`, `closure`, checkpoint, NPZ 导出）完全共享。

### 8.1 优化器类

#### `StageOptimizer` (`optim/optimizers.py:32`)

基础优化器. 持有 L-BFGS 实例和损失记录.

- **构造**: `__init__(name, model, param_names, lr=1.0, lbfgs_max_iter=20, save_every=10, ...)`
- **关键方法**: `set_opt_vars()` (设置梯度范围), `load_checkpoint()`/`save_checkpoint()` (断点续跑), `save_results()` (保存 .npz), `vis_result()` (渲染), `log_losses()` (记录损失), `plot_losses()` (箱线图), `run()` (主循环: 迭代+早停+NaN回滚+定期保存), `optim_step()` (L-BFGS closure + `optim.step`)
- **实例化**: 不直接实例化 (子类 `RootOptimizer`, `SmoothOptimizer` 继承)
- **生命周期**: 每阶段创建一次, `run()` 调用一次

#### `RootOptimizer` (`optim/optimizers.py:381`)

Stage 0. 优化 `trans`, `root_orient`.

- **构造**: `__init__(model, all_loss_weights, ...)` — `param_names = ["trans", "root_orient"]`, `stage = 0`, 创建 `RootLoss`
- **forward_pass**: `pred_params_mano()` → 添加 cameras → `RootLoss.forward(obs_data, pred_data, vis_mask>=0)`
- **实例化**: `run_opt.py:150` — `RootOptimizer(base_model, stage_loss_weights, **opts)`
- **生命周期**: 每序列一次

#### `SmoothOptimizer` (`optim/optimizers.py:424`)

Stage 1. 扩展优化变量到 `trans, root_orient, betas, latent_pose`.

- **构造**: `__init__(model, all_loss_weights, ...)` — `stage = 1`, 条件添加 `world_scale`, `cam_f`, `delta_cam_R`
- **forward_pass**: 同 RootOptimizer + 注入 `get_vars()` 和 `get_extrinsics()`, 传递 `nsteps` 给 loss
- **实例化**: `run_opt.py:157` — `SmoothOptimizer(base_model, stage_loss_weights, **opts)`
- **生命周期**: 每序列一次

### 8.2 损失函数类

#### `StageLoss` (`optim/losses.py:342`)

损失基类. 继承 `nn.Module`.
- **构造**: `__init__(loss_weights)` — 存储权重, 调用 `setup_losses()` (抽象)
- **不直接实例化**

#### `RootLoss` (`optim/losses.py:359`)

Stage 0 损失, 仅数据拟合项.
- **子损失模块**: `Joints2DLoss`, `Points3DLoss`, `GeneralContactLoss`, `BMCLoss`
- **forward**: 遍历 joints3d/bio/verts3d/points3d/joints2d(joints3d_op+cameras→reproject)/joints3d_smooth/depth_constraint, 每项通过权重门控
- **实例化**: `RootOptimizer.__init__` (`optimizers.py:398`)

#### `SMPLLoss` (`optim/losses.py:566`)

Stage 1 损失, 继承 RootLoss + 先验.
- **forward**: `super().forward()` + `pose_prior` (L2→HaMeR init) + `shape_prior` (L2→zero, ×nsteps) + `penetration` (winding numbers, 权重=0禁用)
- **实例化**: `SmoothOptimizer.__init__` (`optimizers.py:446`)

#### `Joints2DLoss` (`optim/losses.py:814`)

2D 重投影损失.
- **构造**: `__init__(ignore_op_joints, joints2d_sigma=100, normalize_by_scale=True)`
- **forward**: NaN/Inf 检测 → 掩码索引过滤 → hand_scale 归一化 `error/hand_scale*100` → GMoF 鲁棒化 → conf^2 加权 → mean
- **实例化**: `RootLoss.setup_losses()` (`losses.py:370`)

#### `GeneralContactLoss` (`optim/losses.py:995`)

穿透损失.
- **构造**: `__init__(faces)` — 添加 14 个水密三角, 注册 `l_faces/r_faces`
- **init_loss**: 返回内部函数, 使用 **winding numbers** 检测内部顶点 + 逐点对距离
- **实例化**: `RootLoss.setup_losses()` (`losses.py:372`)
- 当前权重=0 (禁用)

#### `BMCLoss` (`optim/bio_loss.py:232`)

生物力学约束. 独立类 (非 `nn.Module`).
- **构造**: `__init__(lambda_bl, lambda_rb, lambda_a)` — 从 `_DATA/BMC/` 加载预计算的骨长/曲率/关节角限制
- **compute_loss**: 骨长区间损失 + 根骨曲率 + 关节角 (PIP/DIP/TIP 屈曲/外展的凸包约束)
- **实例化**: `RootLoss.setup_losses()` (`losses.py:373`) + `HMP/fitting.py:38` (全局实例)

### 8.3 参数管理类

#### `Params` (`optim/params.py:16`)

梯度控制容器. 继承 `nn.Module`.
- **关键方法**: `set_param(name, val, requires_grad)` — 注册为 `nn.Parameter`; `set_require_grads(names)` — 先冻结全部, 再解冻指定; `get_dict()`/`get_vars()` — 获取参数快照
- **实例化**: 不直接; `CameraParams` 继承

#### `CameraParams` (`optim/params.py:75`)

相机参数容器. 继承 `Params`.
- **构造**: 继承 `Params(batch_size)`
- **set_cameras(cam_data, opt_scale, opt_cams, opt_focal)**: 存储 `_cam_R`, `_cam_t`, `cam_center`, `cam_f`; 可选注册 `world_scale`, `delta_cam_R`, `delta_cam_t`
- **get_extrinsics()**: 返回 `cam_R, cam_t * world_scale`; 若 `opt_cams` 施加 `cam_R @ dR` (右乘, 相机系扰动)
- **get_cameras(idcs)**: 返回广播到 `(B,T,3,3)` 的相机参数, 供 loss 使用
- **实例化**: `BaseSceneModel.__init__` (`base_scene.py:70`); `set_cameras` 在 `initialize()` (`base_scene.py:76`) 调用

### 8.4 场景模型类

#### `BaseSceneModel` (`optim/base_scene.py:23`)

手部场景模型. 实例化: `run_opt.py:122`.

- **构造**: `__init__(batch_size, seq_len, body_model, pose_prior, use_init, opt_cams, opt_scale)` — 创建 `CameraParams`, 存储 body_model/hand_mean
- **initialize(obs_data, cam_data)**: 设置相机 → 平均 betas → latent2pose 编码初始姿态 → cam2world 转换 root_orient 和 trans → 注册 params + 在 obs_data 中存储 `init_latent_pose`
- **pred_mano(trans, root_orient, body_pose, is_right, betas)**: MANO 前向 → joints3d/verts3d/points3d/faces
- **pred_params_mano(is_right)**: `latent2pose(latent_pose)` → `pred_mano()`
- **get_optim_result()**: 收集所有参数 + 解码 pose_body + 相机 → `{"world": dict}`
- **latent2pose()/pose2latent()**: VPoser 编码/解码 (当前 VPoser=None, 恒等)
- **生命周期**: 每序列创建一次, 跨阶段共享

### 8.5 数据类

#### `MultiPeopleDataset` (`data/dataset.py:87`)

手部跟踪数据集. 继承 `torch.utils.data.Dataset`.
- **构造**: `__init__(data_sources, seq_name, tid_spec, shot_idx, start_idx, end_idx, is_static, ...)` — 扩展路径, 加载 shot 图像, 发现手部轨道, 提取 `track_vis_masks`
- **load_data(interp_input)**: 加载相机 + 每轨道加载 joints2d (插值) + MANO 预测 (pose/orient/trans/betas/is_right)
- **__getitem__(idx)**: 返回 `obs_data` dict (11 个字段)
- **load_camera_data()**: 创建 `CameraData`
- **实例化**: `run_opt.py:189` 通过 `get_dataset_from_cfg()`; **生命周期**: 每序列一次

#### `CameraData` (`data/dataset.py:325`)

相机数据容器. 独立类.
- **构造**: `__init__(cam_dir, seq_len, img_size, is_static, ...)` — 计算帧索引, 调用 `load_data()`
- **load_data()**: 从 `cameras.npz` 加载 w2c + intrins → 偏移首帧平移, 缩放 intrins
- **cam2world()**: w2c → c2w 求逆
- **as_dict()**: 返回 `{cam_R, cam_t, intrins, static}`
- **实例化**: `MultiPeopleDataset.load_camera_data()` (`dataset.py:315`)

### 8.6 身体模型类

`body_model/` 目录包含**两层** MANO 实现，通过继承链和 `__init__.py` 的 import 顺序组装。

**文件关系**：

```
body_model/__init__.py          ← 对外暴露的统一接口
  from .body_model import *     ← 先导入 smplx 的 MANO/MANOLayer
  from .mano_wrapper import *   ← 再导入 wrapper 的 MANO（覆盖上面的 MANO）
```

由于 `mano_wrapper.py` 中的 `MANO(MANOLayer)` 在 import 顺序上后于 `body_model.py` 中的 `MANO(SMPL)`，当调用 `from body_model import MANO` 时，实际导入的是**wrapper 版本的 MANO**（`mano_wrapper.py:11`），而非 smplx 原版。

**继承链**：

```
SMPL (body_model.py:43)                     ← LBS + 通用 SMPL 模型
  └── MANO (body_model.py:1491)             ← MANO 手部模型 (加载 .pkl, PCA 组件, hand_mean)
        └── MANOLayer (body_model.py:1703)   ← 无注册参数版本 (create_* = False, 用作纯前向层)
              └── MANO (mano_wrapper.py:11)  ← [项目实际使用] 添加额外关节 + joint_map
```

#### `MANO` (wrapper, `body_model/mano_wrapper.py:11`)

项目实际使用的手部网格模型。继承 `smplx.MANOLayer`。

- **构造**: `__init__(*args, joint_regressor_extra=None, pose2rot, ...)`
  - 调用 `super().__init__(*args, **kwargs)` → 初始化 smplx MANOLayer
  - 注册 `extra_joints_idxs`：从 `vertex_ids['mano']` 获取指尖顶点索引（5 个指尖）
  - 注册 `joint_map`：`[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]` — 将 MANO 16 标准关节 + 5 指尖 → **21 关节 OpenPose 手部顺序**
  - 可选：若提供 `joint_regressor_extra`，加载额外关节回归器（从顶点回归更多关节）
- **forward**: `super().forward()`（MANOLayer → 16 标准关节 + 778 顶点）→ 指尖顶点 `index_select` → 拼接为 21 关节 → `joint_map` 重排序 → 可选额外回归器追加
- **实例化**: `run_opt.py:116` — `MANO(batch_size=B*T, pose2rot=True, **mano_cfg)`
- **生命周期**: 每序列一次, 跨阶段共享

#### `MANO` / `MANOLayer` (smplx, `body_model/body_model.py`)

上游 SMPL-X 库代码。项目中**不直接实例化**——通过 wrapper 的 `super().__init__` 链调用。

- **`MANO(SMPL)`** (L1491): 手部模型。加载 `MANO_RIGHT.pkl`，注册 `hand_mean`、`hand_components`（PCA）、`pose_mean`。`forward()` 执行 LBS 并返回 `MANOOutput`
- **`MANOLayer(MANO)`** (L1703): 无注册参数版本（`create_* = False`），作为纯前向层使用——所有参数从外部传入而非从 `self.*` 成员读取。wrapper 继承此类以避免与 `Params` 容器中的 `nn.Parameter` 冲突

**为什么需要 wrapper**：
1. smplx 的 MANO 仅输出 16 个标准关节；Dyn-HaMR 需要 **21 个 OpenPose 格式关节**（含指尖）以匹配 ViTPose 的 2D 关键点
2. 关节索引顺序需要重映射（MANO 原始顺序 ≠ OpenPose 手部顺序）
3. 可选的外部关节回归器用于与下游数据集兼容

### 8.7 HMP 相关类

#### `Architecture` (`HMP/nemf/generative.py:54`)

HMP VAE 主模型. 继承 `BaseModel`.
- **构造**: 创建 `ForwardKinematicsLayer` + `LocalEncoder` + `GlobalEncoder` + `NeuralMotionField` + 可选 `GlobalMotionPredictor`
- **encode_local()**: 拼接 pos/velocity/global_xform/angular → 归一化 → `LocalEncoder` → z_l (1024)
- **encode_global()**: root_orient(6D) + root_vel → 归一化 → `GlobalEncoder` → z_g (256)
- **decode(z_l, z_g, length, step)**: t∈[-1,1] + z_l + z_g → `NeuralMotionField` → rot6d → FK → joints
- **实例化**: `HMP/fitting.py:1629` — `Architecture(args, ngpu)`; **生命周期**: 每次 `run_prior()` 调用

#### `LocalEncoder` (`HMP/nemf/prior.py:8`)

骨骼图卷积编码器. 继承 `nn.Module`.
- **构造**: 多层 `SkeletonResidual`/`SkeletonConv` + `SkeletonPool` (stride=2) → `nn.Linear` → (mu, logvar)
- **输入**: `(B, J*D, T)`; **输出**: z_l (1024)
- **实例化**: `Architecture.__init__` (`generative.py:70`)

#### `GlobalEncoder` (`HMP/nemf/prior.py:85`)

全局轨迹编码器. 继承 `nn.Module`.
- **构造**: 4 层 `ResidualBlock` (1D conv, 128→256→512→512) → `nn.Linear` → (mu, logvar)
- **输入**: `(B, D, T)`; **输出**: z_g (256)
- **实例化**: `Architecture.__init__` (`generative.py:71`)

#### `NeuralMotionField` (`HMP/nemf/neural_motion.py`)

时间条件超网络. 继承 `nn.Module`.
- **构造**: 11 层 MLP (8 局部 + 3 全局), skip connections, positional encoding (bandwidth=7), `FCBlock` + LayerNorm
- **forward(t, z_l, z_g)**: t + z_l (局部层) + z_g (全局层注入) → local_output (144) + global_output (6)
- **实例化**: `Architecture.__init__` (`generative.py:73`)

#### `ForwardKinematicsLayer` (`HMP/nemf/fk.py`)

可微正向运动学. 继承 `nn.Module`.
- **构造**: 从 `MANO_RIGHT.pkl` 加载 parents + rest pose offsets
- **forward(rotations, positions)**: 沿 kinematic chain 累积 4×4 变换 → joints 坐标 (B,J,3)

### 8.8 可视化类

#### `OffscreenAnimation` (`vis/viewer.py:258`)

离屏渲染器. 继承 `AnimationBase`.
- **构造**: `__init__(img_size, intrins, fps, ext)` — pyrender 场景 + `OffscreenRenderer` + 光照 + bg_seq/keypoints_seq
- **关键方法**: `set_bg_seq()`, `render()`, `animate()` (写视频/图片序列), `render_mesh_layers()` (分层渲染)
- **实例化**: `init_viewer()` (`vis/viewer.py:40`) — 当 `PYOPENGL_PLATFORM=egl`; `run_opt.py:137`; **生命周期**: 每序列一次 (若 `vis_every>0`)

#### `AnimationViewer` (`vis/viewer.py:498`)

交互式查看器. 继承 `AnimationBase`.
- **实例化**: `init_viewer()` (`vis/viewer.py:33`) — 当 `PYOPENGL_PLATFORM=pyglet`

### 8.9 工具类

#### `Logger` (`util/logger.py:4`)

静态日志类.
- **类方法**: `init(log_path)`, `log(write_str)`
- **实例化**: 无 (纯静态). `run_opt.py:182` 初始化

### 8.10 类层次与生命周期总览

```
run_opt.py (main)
  │
  ├── MultiPeopleDataset (每序列一次)
  │     └── CameraData
  │
  ├── MANO hand_model (每序列一次, 跨阶段共享)
  │
  ├── BaseSceneModel (每序列一次, 跨阶段共享)
  │     └── CameraParams (self.params)
  │
  ├── RootOptimizer (Stage 0, 每序列一次)
  │     └── RootLoss
  │           ├── Joints2DLoss
  │           ├── Points3DLoss
  │           ├── GeneralContactLoss
  │           └── BMCLoss
  │
  ├── SmoothOptimizer (Stage 1, 每序列一次)
  │     └── SMPLLoss (继承 RootLoss 子模块)
  │
  ├── [HMP] run_prior → fitting_prior (Stage 2, 可选)
  │     └── Architecture
  │           ├── LocalEncoder
  │           ├── GlobalEncoder
  │           ├── NeuralMotionField
  │           └── ForwardKinematicsLayer
  │
  └── [可视化] OffscreenAnimation (每序列一次, 若 vis_every>0)
```

**注意**: `BMCLoss` 有两个实例——一个在 `RootLoss` 内 (被 `SMPLLoss` 继承), 一个在 `HMP/fitting.py:38` 作为全局变量供 HMP 优化使用.

### 8.11 生命周期术语说明

文中「每序列一次」指的是 `run_opt()` 函数**每次被调用处理一个视频片段**的完整生命周期，即从 `run_opt.py:main()` 进入到 `run_opt()` 返回之间的时间段。

具体来说，一次 `run_opt()` 调用：
1. 创建一个 `MultiPeopleDataset` 实例，加载一个视频段（单个 shot、固定的 `[start_idx, end_idx)` 范围）
2. 创建一个 `MANO` 手部模型和一个 `BaseSceneModel` 场景模型
3. 依次运行 `RootOptimizer.run()`（50 轮）和 `SmoothOptimizer.run()`（300 轮），两者**共享**同一个 `BaseSceneModel`
4. 可选运行 `run_prior()` 和 `run_vis()`

各实例的生命周期边界：

```
run_opt() 进入
  │
  ├── MultiPeopleDataset ────────────────┐
  │     └── CameraData                    │
  ├── MANO hand_model                     │  构造一次
  ├── BaseSceneModel                      │  复用至 run_opt 返回
  │     └── CameraParams                  │
  ├── RootOptimizer ──────────────────┐   │
  │     └── RootLoss                  │   │
  │         ├── Joints2DLoss          │   │
  │         ├── Points3DLoss          │   │
  │         ├── GeneralContactLoss    │   │
  │         └── BMCLoss               │   │  Stage 0
  │                                   │   │
  ├── SmoothOptimizer ────────────────┤   │
  │     └── SMPLLoss                  │   │  Stage 1
  │         (继承 RootLoss 子模块)     │   │
  │                                   │   │
  ├── [HMP] Architecture ─────────────┘   │  Stage 2 (新建/析构)
  │     ├── LocalEncoder                  │
  │     ├── GlobalEncoder                 │
  │     ├── NeuralMotionField             │
  │     └── ForwardKinematicsLayer        │
  │                                       │
  └── [Vis] OffscreenAnimation ───────────┘  可视化 (新建/析构)

run_opt() 返回 → 所有实例析构
```

**跨阶段共享** vs **新建/析构**：
- `MultiPeopleDataset`、`MANO`、`BaseSceneModel`：在 `run_opt()` 开始时构造，贯穿 Stage 0 和 Stage 1，在函数返回时析构。它们的内部状态（参数值、数据缓存）跨阶段保留
- `RootOptimizer`、`SmoothOptimizer`：各自的优化器实例和 L-BFGS 状态在对应阶段开始时新建，阶段结束时析构。它们操作的是同一个 `BaseSceneModel.params` 中的参数张量，但各自持有独立的 Hessian 近似
- `Architecture`（HMP）：仅在 `run_prior=True` 时构造，`run_prior()` 返回后即析构。通过从磁盘加载 `smooth_fit/*.npz` 获得 Stage 1 的结果，不直接访问内存中的 `BaseSceneModel`
- `OffscreenAnimation`：仅在 `vis_every > 0` 时构造，用于可视化渲染，不参与优化

### 8.12 BaseSceneModel、Params 与优化器之间的参数管理关系

三类组件在优化中各司其职，通过共享同一个 `CameraParams` 实例协同工作：

```
                    ┌─────────────────────────┐
                    │     CameraParams         │  ← 继承 nn.Module
                    │  (optim/params.py:75)    │
                    │                         │
                    │  nn.Parameter 张量:      │
                    │   trans      (B,T,3)     │
                    │   root_orient (B,T,3)    │
                    │   betas      (B,10)      │
                    │   latent_pose (B,T,D)    │
                    │   world_scale (1,1)      │
                    │   ...                    │
                    │                         │
                    │  核心方法:               │
                    │   set_require_grads()    │← 控制哪些参数有梯度
                    │   get_dict()             │→ 导出所有参数快照
                    │   load_dict()            │← 从 checkpoint 恢复
                    └──────┬──────────────────┘
                           │ 同一个实例 (self.params)
              ┌────────────┼────────────┐
              │            │            │
    ┌─────────▼──┐  ┌─────▼──────┐  ┌─▼───────────┐
    │BaseSceneModel│  │RootOptimizer│  │SmoothOptimizer│
    │(持有者)      │  │(Stage 0)   │  │(Stage 1)     │
    │             │  │             │  │              │
    │ 职责:       │  │ 职责:       │  │ 职责:         │
    │ • 构造 params│  │ • 调用       │  │ • 调用         │
    │ • initialize │  │ set_opt_vars│  │ set_opt_vars  │
    │   写入初始值  │  │   (["trans", │  │   (["trans",   │
    │ • pred_mano  │  │   "root_    │  │   "root_orient"│
    │   读取参数做  │  │   orient"]) │  │   ,"betas",    │
    │   MANO 前向  │  │ • 创建       │  │   "latent_pose"│
    │              │  │   L-BFGS     │  │   ])           │
    │              │  │ • forward_   │  │ • 创建 L-BFGS  │
    │              │  │   pass() →   │  │ • forward_pass │
    │              │  │   pred_params│  │   () →         │
    │              │  │   _mano()    │  │   pred_params  │
    │              │  │ • optim.step │  │   _mano()      │
    │              │  │   (closure)  │  │ • optim.step   │
    └──────────────┘  └─────────────┘  └───────────────┘
```

**关系总结**：

| 角色 | 做什么 | 不做什么 |
|------|--------|---------|
| **`CameraParams`** (参数存储) | 持有所有 `nn.Parameter` 张量；通过 `requires_grad` 控制哪些参数参与当前阶段的梯度更新；提供 `get_dict()`/`load_dict()` 做磁盘序列化 | 不执行前向计算；不知道 MANO 模型的存在；不感知优化器 |
| **`BaseSceneModel`** (前向计算) | 在 `initialize()` 中写入参数初始值；在 `pred_params_mano()` 中读取参数并执行 MANO 前向，产生 `pred_data` | 不控制梯度；不持有优化器；不执行 `backward()` 或 `optim.step()` |
| **`StageOptimizer`** (优化控制) | 通过 `set_opt_vars()` 设置各阶段的梯度范围；创建 L-BFGS 实例；在 `forward_pass()` 中调用 `BaseSceneModel.pred_params_mano()` 获取预测，计算 loss，调用 `loss.backward()` 和 `optim.step()` | 不持有参数（持有的是对 `CameraParams` 中张量的**引用**）；不执行 MANO 前向（委托给 `BaseSceneModel`） |

**一次 L-BFGS closure 的完整调用链**：

```
SmoothOptimizer.optim_step()
  │
  └─► closure()
        │
        ├─► SmoothOptimizer.forward_pass(obs_data)
        │     │
        │     ├─► BaseSceneModel.pred_params_mano(is_right)
        │     │     │
        │     │     ├─► BaseSceneModel.latent2pose(self.params.latent_pose)
        │     │     │      ← 从 CameraParams 读取 latent_pose (requires_grad=True)
        │     │     │
        │     │     └─► BaseSceneModel.pred_mano(trans, root_orient, body_pose, ...)
        │     │            ← 从 CameraParams 读取 trans, root_orient, betas
        │     │            → 返回 {joints3d, verts3d, ...}
        │     │
        │     ├─► CameraParams.get_cameras()  → cam_R, cam_t, cam_f, cam_center
        │     ├─► CameraParams.get_extrinsics() → cam_R, cam_t
        │     └─► SMPLLoss.forward(obs_data, pred_data, ...) → (loss, stats)
        │
        ├─► loss.backward()
        │     ← 梯度流过 pred_data → pred_mano → CameraParams 中的 nn.Parameter
        │     ← set_require_grads 控制的参数得到梯度，其余为 None
        │
        └─► return loss
              │
              └─► L-BFGS 内部: 使用 self.opt_params 的 .grad 进行线搜索和参数更新
```
