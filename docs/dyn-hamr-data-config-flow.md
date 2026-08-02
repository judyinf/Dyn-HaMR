# Dyn-HaMR 数据格式、配置体系与帧加载流程

本文档系统化梳理 Dyn-HaMR 的核心数据约定、配置体系、视频帧加载全流程、ARCTIC benchmark 数据格式对比，以及 HMP 优化中 vis_mask 的角色与批处理方式对比。

---

## 目录

- [一、数据格式与字段约定完整清单](#一数据格式与字段约定完整清单)
- [二、配置体系拆解](#二配置体系拆解)
- [三、视频帧加载与vis_mask全流程](#三视频帧加载与vis_mask全流程)
- [四、ARCTIC benchmark 数据格式与 Dyn-HaMR 对比](#四arctic-benchmark-数据格式与-dyn-hamr-对比)
- [五、vis_mask在HMP优化中的角色与批处理方式对比](#五vis_mask在hmp优化中的角色与批处理方式对比)

---

## 一、数据格式与字段约定完整清单

### 1.1 输入文件

| 文件名 | 格式 | 生成者 | 路径约定 |
|--------|------|--------|---------|
| `{frame}_mano.json` | JSON | `export_hamer.py` | `{root}/dynhamr/track_preds/{seq}/{tid:03d}/` |
| `{frame}_keypoints.json` | JSON | `export_hamer.py` | 同上 |
| `cameras.npz` | NPZ | VIPE / DROID-SLAM | `{root}/dynhamr/cameras/{seq}/shot-{idx}/` |
| `{seq}.json` | JSON | `export_hamer.py` | `{root}/dynhamr/shot_idcs/` |

#### `_mano.json`

```json
{
  "betas": [10],
  "body_pose": [15, 3],
  "global_orient": [3],
  "cam_trans": [3],
  "is_right": 0 或 1
}
```

字段含义：`betas`=MANO形状(10)，`body_pose`=15个手指关节轴角(相机空间)，`global_orient`=手腕全局旋转轴角(相机空间)，`cam_trans`=手腕平移(相机空间)，`is_right`=手性(0=左,1=右,与tid一致)。

> HaMeR 内部以旋转矩阵格式存储，`export_hamer.py:unpack_frame()` 通过 `cv2.Rodrigues` 转为轴角。

#### `_keypoints.json`

```json
{
  "people": [{
    "pose_keypoints_2d": [x1,y1,c1, ..., x21,y21,c21]
  }]
}
```

21个OpenPose手部格式关键点(63个float)。在当前YOLO版`run.py`中由HaMeR自身MANO 3D→2D投影产生(非外部ViTPose)，conf强制=1.0。

#### `cameras.npz`

```python
{
    'height': float,       'width': float,
    'focal': float,        'intrins': (N,4),    # [fx,fy,cx,cy]
    'w2c': (N,4,4),        # 世界到相机矩阵
}
```

VIPE输出c2w格式，由`vidproc.py:load_vipe_cameras()`通过`np.linalg.inv(c2w)`转为w2c后保存。

#### `shot_idcs/{seq}.json`

```json
{"frame_000001.jpg": 0, "frame_000002.jpg": 0, ..., "frame_000100.jpg": 1}
```

每帧映射到shot索引。由PHALP跟踪系统在上游生成。

### 1.2 核心参数容器

#### obs_data — 数据集供给优化器的观测数据

由 `MultiPeopleDataset.__getitem__()` 返回，经 `DataLoader(batch_size=B)` 批处理为 `(B, T, ...)`。

**数据集字段** (`dataset.py:264-307`)：

| 字段 | 形状 | 来源 | 含义 |
|------|------|------|------|
| `joints2d` | (B,T,21,3) | `load_keypoints_with_interp()` + conf=1.0 | 2D关键点GT (x,y,conf) |
| `init_body_pose` | (B,T,15,3) | `load_mano_preds()` Slerp插值 | 手指关节轴角(相机空间) |
| `init_body_shape` | (B,T,10) | `load_mano_preds()` 线性插值 | MANO betas(时间平均得(B,10)) |
| `init_root_orient` | (B,T,3) | `load_mano_preds()` Slerp插值 | 手腕旋转轴角(相机空间) |
| `init_trans` | (B,T,3) | `load_mano_preds()` 线性插值 | 手腕平移(相机空间) |
| `is_right` | (B,T) | 全轨道恒定(float32) | 1=右手,0=左手 |
| `vis_mask` | (B,T) | `get_ternary_mask()` | -1=出镜,0=遮挡,1=可见 |
| `track_interval` | (2,) | 轨道内首尾可见帧索引 | `[track_s, track_e)` |
| `seq_interval` | (2,) | 数据集内部帧范围 | `[start_idx, end_idx)` |
| `track_id` | scalar | 轨道ID(int) | 0=左手,1=右手 |
| `seq_name` | scalar | 配置(str) | 日志/可视化命名 |

**由 `BaseSceneModel.initialize()` 扩展** (`base_scene.py:146`)：

| 字段 | 形状 | 含义 |
|------|------|------|
| `init_latent_pose` | (B,T,D) | HaMeR初始潜变量(detach), SMPLLoss pose_prior的GT锚点 |

#### cam_data — 相机数据

由 `CameraData.as_dict()` 返回 (`dataset.py:393-399`)：

| 字段 | 形状 | 含义 |
|------|------|------|
| `cam_R` | (T,3,3) | 世界到相机旋转矩阵(首帧平移被置零) |
| `cam_t` | (T,3) | 世界到相机平移向量 |
| `intrins` | (T,4) | `[fx,fy,cx,cy]`(按图像尺寸缩放) |
| `static` | bool | 静态相机标志(True→禁用scale优化) |

#### pred_data — 每次L-BFGS closure的预测输出

由 `BaseSceneModel.pred_params_mano()` → `pred_mano()` 产生：

| 字段 | 形状 | 来源 |
|------|------|------|
| `joints3d` | (B,T,16,3) | MANO前向: 16标准关节 |
| `joints3d_op` | (B,T,16,3) | 同joints3d(无OpenPose重映射) |
| `verts3d` | (B,T,778,3) | MANO前向: 所有顶点 |
| `points3d` | (B,T,778,3) | 同verts3d |
| `l_faces` | (F+14,3) | 左手面片(反转绕组+14个水密三角) |
| `r_faces` | (F+14,3) | 右手面片(标准绕组+14个水密三角) |
| `body_pose` | (B,T,45) | MANO输出的手部姿态轴角 |
| `is_right` | (B,T) | 手性 |

SmoothOptimizer通过`pred_data.update(self.model.params.get_vars())`额外注入所有可优化参数(含`latent_pose`, `betas`, `world_scale`)，以及显式的`cam_R`, `cam_t`。

### 1.3 输出文件

所有`*_world_results.npz`的完整字段约定：

| 键名 | 形状 | 出现阶段 | 含义 |
|------|------|---------|------|
| `trans` | (B,T,3) | 全部 | **世界空间**手腕平移 |
| `root_orient` | (B,T,3) | 全部 | **世界空间**手腕旋转(轴角) |
| `pose_body` | (B,T,45) | 全部 | 手部姿态轴角(latent2pose解码) |
| `latent_pose` | (B,T,D) | init/root_fit/smooth_fit/prior | 优化潜变量(D=45,无VPoser时=pose_body) |
| `betas` | (B,10) | init/smooth_fit/prior | MANO形状参数 |
| `is_right` | (B,T) | 全部 | 手性(0/1) |
| `init_body_pose` | (B,T,15,3) | init | HaMeR原始手指姿态(存档用) |
| `world_scale` | (1,1) | init/smooth_fit/prior | 全局尺度(仅opt_scale=True) |
| `cam_R` | (B,T,3,3) | 全部 | 世界到相机旋转 |
| `cam_t` | (B,T,3) | 全部 | 世界到相机平移 |
| `intrins` | (4,) | 全部 | `[fx,fy,cx,cy]`(标量,全帧共享) |

Prior阶段额外字段: `decode_root`(B,T,3), `poses`(B,T,48)。

简化phalp NPZ(`hamer/`目录)仅含`{pose_body(B,T,15,3), trans(B,T,3), root_orient(B,T,3)}`——仅用于可视化优化前后对比。

---

## 二、配置体系拆解

### 2.1 Hydra入口

`run_opt.py:173`: `@hydra.main(version_base=None, config_path="confs", config_name="config.yaml")`

Hydra行为: 加载`confs/config.yaml`→处理`defaults`+OmegaConf插值→`chdir:True`切换工作目录→将解析后的`cfg:DictConfig`传给`main()`。

OmegaConf resolver: `OmegaConf.register_new_resolver("eval", eval)` 允许配置中使用`${eval:...}`动态求值。

### 2.2 config.yaml — 顶层配置

**模型控制**:
| 字段 | 默认值 | 含义 |
|------|--------|------|
| `model.use_init` | True | 从HaMeR预测初始化(非零起点) |
| `model.opt_cams` | False | 不优化相机外参(固定w2c) |
| `model.opt_scale` | True | 非静态相机优化world_scale |
| `model.async_tracks` | True | 支持异步轨道(不同时出现的手) |

**运行控制**:
| 字段 | 默认值 | 含义 |
|------|--------|------|
| `run_opt` | True | 是否执行优化 |
| `run_vis` | True | 优化后是否可视化 |
| `run_prior` | False | 是否执行HMP运动先验精修 |
| `overwrite` | False | 是否覆盖已有输出 |

**MANO模型配置**:
```yaml
MANO:
  MODEL_PATH: ${paths.DATA_DIR}/mano    # MANO_RIGHT.pkl所在目录
  GENDER: neutral
  NUM_HAND_JOINTS: 15
  MEAN_PARAMS: ${paths.DATA_DIR}/mano_mean_params.npz
  CREATE_BODY_POSE: FALSE
```

**Paths**:
```yaml
paths:
  base_dir: None                                    # 由resolve_cfg_paths自动填充
  DATA_DIR: _DATA/data/                             # 所有模型文件根目录
  vposer: VPoser/pretrained/Vposer_right_mirrored   # VPoser路径(当前未使用)
```

**HMP**:
```yaml
HMP:
  config: hmp_config.yaml    # HMP自身配置文件
  use_hposer: False          # 不使用HPoser变体
  vid_path: ${data.root}/video/${data.seq}.mp4
```

**Hydra输出路径**:
```yaml
fps: 30
log_root: ../outputs/logs
log_dir: ${log_root}/${data.type}-${data.split}
exp_name: ${now:%Y-%m-%d}
hydra:
  job:
    chdir: True              # 运行时切换到输出目录
  run:
    dir: ${log_dir}/${exp_name}/${data.name}
```

最终输出路径: `../outputs/logs/video-custom/2026-07-31/demo1-all-shot-0-0--1/`

**可视化**:
```yaml
vis:
  phases: [smooth_fit, prior]        # 渲染哪些阶段
  render_views: [above, side, front, src_cam]
  make_grid: True                    # 是否生成2x2网格视频
  overwrite: True
```

### 2.3 optim.yaml — 优化配置

**优化器选项**:
| 字段 | 值 | 含义 |
|------|----|------|
| `options.lr` | 1.0 | L-BFGS初始步长 |
| `options.lbfgs_max_iter` | 20 | 每次step内最大函数评估数 |
| `options.save_every` | 20 | 每N轮保存一次checkpoint+NPZ |
| `options.vis_every` | -1 | 可视化频率(-1=禁用) |
| `options.max_chunk_steps` | 20 | 早停: 连续N步loss变化<20则终止 |

**阶段迭代次数**:
```yaml
root:
  num_iters: 50        # Stage 0: 根优化
smpl:
  num_iters: 0         # SLAHMR的SMPL阶段(已禁用)
smooth:
  num_iters: 300       # Stage 1: 平滑优化
  opt_scale: False     # 不在Stage 1优化world_scale
```

**loss_weights**: 每个loss为3元素列表`[stage0, stage1, stage2]`。权重=0的loss通过`if weight > 0.0`门控跳过。

| Loss | 值 | Stage 0 | Stage 1 | 活跃与否 |
|------|----|---------|---------|---------|
| `joints2d` | [10000,10000,10000] | ✅ | ✅ | 主导loss |
| `joints3d_smooth` | [1000,10000,0] | ✅ | ✅(10x增强) | 时序平滑 |
| `depth_constraint` | [100,100,0] | ✅ | ✅ | 防相机后穿透 |
| `pose_prior` | [1,1,1] | ❌(在RootLoss中不计算) | ✅ | L2→HaMeR初始值 |
| `shape_prior` | [0.05,0.05,0.05] | ❌ | ✅ | L2→0(betas正则) |
| `joints3d` | [0,0,0] | ❌ | ❌ | 禁用 |
| `bio` | [0,0,0] | ❌ | ❌ | 禁用 |
| `penetration` | [0,0,0] | ❌ | ❌ | 禁用 |
| `verts3d/points3d` | [0,0,0] | ❌ | ❌ | 禁用 |

### 2.4 data/{type}.yaml — 数据配置

**video_vipe.yaml**(VIPE相机,推荐):
```yaml
type: video
split: custom
root: /data/home/.../Dyn-HaMR/test    # 视频根目录
video_dir: videos
seq: prod1                            # 视频名(不含扩展名)
ext: mp4
src_path: ${data.root}/${data.video_dir}/${data.seq}.${data.ext}
use_cams: True
track_ids: "all"                      # 轨道选择("all"=最长N个)
shot_idx: 0                           # 镜头索引
start_idx: 0                          # 起始帧(0=从shot首帧开始)
end_idx: -1                           # 结束帧(-1=到shot末帧)
split_cameras: True

sources:                              # 数据路径(由expand_source_paths展开)
  images: ${data.root}/images/${data.seq}
  cameras: ${data.root}/dynhamr/cameras/${data.seq}/shot-${data.shot_idx}
  tracks: ${data.root}/dynhamr/track_preds/${data.seq}
  shots: ${data.root}/dynhamr/shot_idcs/${data.seq}.json

use_vipe: True                        # 使用VIPE相机
vipe_dir: .../third-party/vipe/vipe_results

name: ${data.seq}-${data.track_ids}-shot-${data.shot_idx}-${data.start_idx}-${data.end_idx}
```

**video_driod.yaml**: 同结构但`use_vipe: False`(隐式), 使用DROID-SLAM相机。

### 2.5 路径解析

`util/loaders.py:resolve_cfg_paths(cfg)`: 遍历`cfg.paths`, 相对路径(不以`/`开头)前加`ROOT_DIR`(项目根目录绝对路径)。

`data/dataset.py:expand_source_paths(data_sources)`: 对`sources`中的每个路径做`glob.glob()`展开(处理通配符, 若不存在返回原值)。

---

## 三、视频帧加载与vis_mask全流程

### 3.1 帧加载逻辑

以demo1.mp4为例, 配置`shot_idx=0, start_idx=0, end_idx=-1`:

```
完整视频 (500帧)
  │
  ├──[第1层: Shot切分] shot_idcs/demo1.json → get_shot_img_files()
  │    筛选shot_idx=0的帧 → 假设全部500帧属于shot 0
  │
  ├──[第2层: 帧范围] data_start=0, data_end=-1→500
  │    img_files = img_files[0:500]  → 全部500帧
  │    self.num_imgs = 500
  │
  ├──[第3层: 轨道裁剪] 遍历所有轨道的track_vis_masks
  │    轨道0(左手,tid=0): 首可见=120, 末可见=380
  │    轨道1(右手,tid=1): 首可见=110, 末可见=370
  │    sidx = min(120,110) = 110
  │    eidx = max(380,370)+1 = 381
  │    seq_len = 381 - 110 = 271帧  ← 最终优化窗口
  │
  └──[DataLoader] batch_size=B=2, shuffle=False
        obs_data: (2, 271, ...) 一次加载到GPU
```

代码: `dataset.py:106-177`

### 3.2 vis_mask构建

`get_ternary_mask()` (`dataset.py:402-411`): 对每个轨道独立计算:

```
track_vis_masks[i] (bool, 271帧)
  │  True: {tid}/{frame}_keypoints.json 存在
  │  False: 文件不存在
  ▼
vis_idcs = where(True)        → 所有可见帧索引
track_s = min(vis_idcs)       → 该轨道首次出现
track_e = max(vis_idcs) + 1   → 该轨道最后出现+1
  │
  ▼
[0, track_s)     → -1 (出镜: 手还没出现)
[track_s, track_e) + True  → 1  (可见)
[track_s, track_e) + False → 0  (遮挡/检测失败)
[track_e, T)     → -1 (出镜: 手已离开)
```

### 3.3 vis_mask在BaseSceneModel.initialize()中

初始化阶段 (`base_scene.py:72-148`): vis_mask**不直接参与**相机空间→世界空间的参数转换。所有帧(包括vis=-1)的`trans`和`root_orient`都被cam2world转换——但vis=-1帧的参数值为全零默认值, 转换后仍在世界原点附近。

`init_latent_pose`被存入`obs_data`作为pose_prior的GT锚点(L146)。

### 3.4 vis_mask在三阶段优化中的作用

进入优化器后二值化 (`optimizers.py:419, 474`):

```python
vis_mask = obs_data["vis_mask"] >= 0   # -1→False, 0→True, 1→True
```

**Stage 0 (RootOptimizer)**和**Stage 1 (SmoothOptimizer)**中, 此二值掩码传入各loss:

| Loss | 掩码方式 | vis=-1 | vis=0(遮挡) | vis=1(可见) |
|------|---------|--------|------------|------------|
| `joints2d` | `data[mask]`索引过滤,物理删除样本 | ❌ | ✅(插值关键点) | ✅(原始关键点) |
| `joints3d_smooth` | `mask[:,1:] & mask[:,:-1]`邻帧对 | ❌(跨越-1边界切断) | ✅(连续遮挡段内平滑) | ✅ |
| `pose_prior` | `loss[mask.bool()]` | ❌ | ✅ | ✅ |
| `bio_loss` | `joints[valid_mask]` | ❌ | ✅ | ✅ |
| `shape_prior` | 无掩码 | ○(betas全局共享) | ○ | ○ |
| `penetration` | 无掩码 | ✅(权重0禁用) | ✅ | ✅ |
| `depth_constraint` | 无掩码 | ○(全零参数不会触发) | ○ | ○ |

**vis=-1帧的参数不会被优化**: 虽然它们的`nn.Parameter.requires_grad=True`, 但所有loss对这些帧的贡献为零→`loss.backward()`时梯度不流经这些帧的MANO前向路径→`.grad`恒为零→参数保持全零初始值。

### 3.5 为什么保持张量形状统一而不动态裁剪有效帧范围?

Dyn-HaMR选择`(B, T)`固定形状张量而非仅保留`vis_mask>=0`帧, 原因有三:

**1. L-BFGS全批次优化要求固定参数张量形状**

L-BFGS在优化器内部维护固定维度的Hessian近似矩阵。若每帧独立决定是否参与优化(动态改变参数张量的第一维), 则每个L-BFGS step的张量形状可能不同——Hessian近似矩阵的维度会频繁变化, 无法收敛。PyTorch的`torch.optim.LBFGS`也不支持动态参数集。

**2. MANO前向的批量并行需求**

`MANO(batch_size=B*T)`在初始化时固定了批量大小。`pred_mano()`将`(B,T,...)`重塑为`(B*T,...)`后一次性前向。若动态排除vis=-1帧, 需要每次closure重新构造变长batch——或在被排除位置填充dummy值(当前的全零策略实质上就是这样), 两者等价。

**3. 掩码loss提供了等效的排除语义**

通过`vis_mask>=0` + 各loss的掩码过滤, vis=-1帧的loss贡献为零, 梯度为零, 参数不被更新——等效于这些帧「不在优化中」。唯一的代价是MANO前向和loss计算中对这些帧的浮点运算(约占20%的总计算量), 但这部分开销远小于动态batch重组的复杂性。

**总结**: Dyn-HaMR用少量冗余计算(对vis=-1帧做无梯度前向)换取张量形状的稳定性、L-BFGS兼容性和代码简洁性。vis_mask在loss层面提供了等效的帧排除——不是「排除帧不计算」, 而是「计算但不产生梯度」。

---

## 四、ARCTIC benchmark 数据格式与 Dyn-HaMR 对比

`/cpfs/yinghua/data/benchmark_v1/arctic/` 下包含 25 个序列的标准化评测数据。本节描述其格式并与 Dyn-HaMR 对应字段对比。

### 4.1 目录结构

```
arctic/
  ├── eval_list.txt                    # 评测帧清单 (序列名 + 6位帧号)
  ├── pose3d/<seq>.pose3d_hand        # 25 个 torch.save 文件
  ├── tracks/<seq>_tracks.npy         # 25 个 pickled dict
  └── videos/<seq>.mp4                # h264, 2800x2000, 30fps
```

### 4.2 pose3d_hand 格式

`torch.load(path, map_location='cpu', weights_only=False)` 加载，顶层 dict 含 5 个键：

```
{
  'left_hand':  { 'mano_params': {...}, 'pred_valid': (T,) bool },
  'right_hand': { 'mano_params': {...}, 'pred_valid': (T,) bool },
  'slam_data':  { 'traj': (T,7) f32, 'scale': f64, 'img_focal': f64, 'img_center': (2,) f64 },
  'fps':  float,
  'meta': { 'source_dataset', 'mano_hand_pose_source/export', 'mano_transl_export', 'mano_model_dir' }
}
```

**mano_params**（双手结构相同）：

| 字段 | 形状 | 类型 | 含义 |
|------|------|------|------|
| `transl` | (T, 3) | f32 | 世界空间手腕平移 |
| `global_orient` | (T, 3) | f32 | 世界空间手腕旋转（轴角） |
| `hand_pose` | (T, 45) | f32 | 15 关节手指姿态（轴角） |
| `betas` | (T, 10) | f32 | MANO 形状参数（逐帧，非时间平均） |

**pred_valid**: (T,) bool — 所有文件全部为 `True`。

**slam_data**: `traj`(T,7) 的列语义为 `[tx, ty, tz, qx, qy, qz, qw]`（c2w，四元数 x,y,z,w 顺序）。`scale` 恒为 1.0。`img_focal` ~2409-2414。`img_center` ~[1328, 982]（2800×2000 的视频中心）。

**meta**: 记录源数据集、MANO 导出约定、模型路径等来源信息。

### 4.3 tracks.npy 格式

`np.load(path, allow_pickle=True).item()` 加载，pickled dict 结构：

```
{
  0: [{'frame': int, 'det': bool, 'det_box': (1,5) f32, 'det_handedness': (1,) f32}, ...],  # 左手
  1: [{'frame': int, 'det': bool, 'det_box': (1,5) f32, 'det_handedness': (1,) f32}, ...]   # 右手
}
```

- 两个键为整数 `0`（左手）和 `1`（右手），各映射到一个长度为 T 的 list
- `det_box`: xyxy 格式 `[x1, y1, x2, y2, score]`，`det=False` 时为 `[-1,-1,1,1,0]`，`det=True` 时 score 恒为 1.0
- `det_handedness`: (1,) f32，每侧恒定（0 侧 = 0.0，1 侧 = 1.0）
- T 与 pose3d_hand 的帧数严格一致

### 4.4 与 Dyn-HaMR 格式的关键差异

| 维度 | ARCTIC benchmark | Dyn-HaMR (main) |
|------|-----------------|-----------------|
| **pose3d_hand 顶层键** | `left_hand`, `right_hand`, `slam_data`, `fps`, `meta` | 同结构，但**无 `meta`** |
| **hand 子字段** | `mano_params` + `pred_valid` | `mano_params` + `pred_valid` + `detection_failed` + `relative_motion` |
| **mano_params 字段顺序** | `transl, global_orient, hand_pose, betas` | `global_orient, hand_pose, betas, transl` |
| **mano_params 类型** | 全为 `np.ndarray` (f32) | Dyn-HaMR export 中为 `torch.Tensor`/`np.ndarray` |
| **betas 形状** | (T, 10) per-frame | (B, 10) 时间平均（优化阶段内） |
| **pred_valid** | 全部 True（无帧有效性标记） | 反映 HaWoR Infiller 填充结果 |
| **detection_failed** | 不存在 | 仅 `allow_disappear=True` 时存在 |
| **relative_motion** | 不存在 | `rel_rot_aa/rel_rot_mat/rel_trans/pair_valid` |
| **slam_data.traj** | (T,7) c2w `[tx,ty,tz,qx,qy,qz,qw]` | 同格式，但含 `tstamp`、chunk 计数等额外字段 |
| **slam_data.scale** | 恒为 1.0 | Metric3D 估计值 |
| **tracks 文件命名** | `<seq>_tracks.npy` | Dyn-HaMR demo 中使用 `<name>_track_info.npy` |
| **det_box 形状** | (1,5) f32 | (5,) f32 或 (1,5)（有标准化函数） |
| **tracks 键类型** | int (0/1) | 同（track_id） |
| **vis_mask 等价物** | 无（`det` bool 隐式表达，frames dict 覆盖全部帧） | `get_ternary_mask()` 三值 |
| **帧缺失表达** | `det=False`（det_box为sentinel值）+ 轨道始终 T 长度 | 文件不存在 → vis_mask=0(遮挡)或-1(出镜) |

**互相加载兼容性**: Dyn-HaMR的`dyn-hamr/data/pose_io.py:load_pose3d_payload`对所有字段做了标准化(mano_params→f32 ndarrays, pred_valid→bool, slam ints→int)，对未知键会保留不报错。因此ARCTIC benchmark的pose3d_hand可以被Dyn-HaMR代码加载(仅缺失字段为None)。benchmark的`betas`是逐帧(T,10)的——若被Dyn-HaMR加载，`BaseSceneModel.initialize()`仅使用`init_body_shape`的时间均值(L86)，不会冲突。

---

## 五、vis_mask在HMP优化中的角色与批处理方式对比

### 5.1 vis_mask在HMP (main)中的作用

**结论：vis_mask在HMP优化中完全没有使用。**

代码追踪(`HMP/fitting.py`):

```
L204:  vis_mask = npz_init_dict['vis_mask']       ← 从NPZ读入
L288:  'vis_mask': torch.tensor(vis_mask).to(device) ← 存入data dict
L740:  'vis_mask': obs_data['vis_mask'][idx]       ← 从L-BFGS阶段传入
L808:  target['vis_mask'] = data['vis_mask']       ← 复制到target
L963:  vis_mask = torch.tensor(target["vis_mask"]) ← 取出
L1043/1090/1342:  传入optim_step(..., vis_mask=vis_mask) ← 接收但从未在closure中引用
L1312-1317: mask_data(data, mask)                 ← 死代码(仅在注释行中引用)
```

HMP的损失函数(`optim_step`/`optim_step_new`的closure)通过**关键点置信度**`joints2d[..., 2]`做加权而非vis_mask。旋转/重投影/方向损失使用`mano_joint_conf`(从2D关键点置信度推导)，不区分出镜/遮挡/可见。

### 5.2 main分支HMP批处理：固定128帧chunk堆叠

main分支(`/cpfs/yinghua/Dyn-HaMR-main/dyn-hamr/HMP/fitting.py`):

```
full sequence (T帧)
  │
  ├── T > 128?  → 切分为 [128, 256, ...], 最后一段 <128 则重复末帧padding到128
  └── T < 128?  → padding到128
  │
  ▼
全部chunk堆叠为 (N_chunks, 128) → 一次性输入 model.decode()
  │  model.decode(z_l, z_g, length=128, step=1) — 固定解码128帧
  │  所有chunk的loss求和, 含batch_consistency跨chunk边界损失
  │  padding帧被包含在loss中(无掩码排除)
  │
  ▼
固定形状 (N_chunks, 128) — 等价于 L-BFGS 的 (B, T) 固定形状策略
```

### 5.3 dev分支HMP批处理：滑动窗口 (1, 128) + 有效帧索引

dev分支(`/cpfs/yinghua/Dyn-HaMR/dyn-hamr/HMP/fitting.py`，+1164行改动):

```
full sequence (T帧)
  │
  ├── compute_seq_intervals(seq_len, 128, overlap_len=16)
  │     产生重叠滑动窗口: [0,128), [112,240), ...
  │     最后一个窗口可能 <128 帧: [448, 565) → valid_len=117
  │
  ▼
每个窗口独立处理:
  │  _build_window_data(start, end, clip_len=128)     ← fitting.py:852-881
  │    切片 data[start:end] → 尾部重复末帧padding到128
  │    对padding帧: joints2d[:, :, 2] = 0 (置信度置零)
  │                vis_mask = -1
  │    valid_len = end - start (实际有效帧数)
  │
  ▼
  _optimize_window_current_process()                  ← fitting.py:956-1002
  │  model.set_input(window_data)                    → (1, 128)
  │  target = _build_window_target(data, start, end)
  │  T = torch.arange(valid_len)                     ← 仅有效帧
  │  motion_reconstruction(..., T=T, ...)
  │    │
  │    └── latent_optimization(..., T=T, ...)
  │          所有loss中按 T 索引: joints2d_obs[:, T], joints2d_pred[:, T]
  │                               local_rotmat[:, T], root_orient[:, T]
  │                              body_pose[:, T], joints3d[..., T]
  │          padding帧(valid_len..127)被排除在全部loss之外
  │
  ▼
_stitch_hand_windows(): 将所有窗口结果拼回完整T帧      ← fitting.py:1057-1075
  │  旋转: quaternion sign-aligned averaging (Slerp等价)
  │  平移: cosine overlap-add blending
  │  betas: 窗口均值
  │  可选 OneEuro 时序滤波
```

**关键改进**:
- padding帧通过 `vis_mask=-1` 标记(用于下游读取)，但在loss中通过 `T=torch.arange(valid_len)` 索引排除——vis_mask仍然不直接参与loss
- 每个窗口的 `model.decode(z_l, z_g, length=128)` 仍解码固定128帧(NeMF是时间条件超网络，需要固定输入维度)，但数据项仅评估前`valid_len`帧
- 运动先验loss(`motion_prior_loss = ||z_l||^2`)作用于整个窗口(包括padding帧的潜码部分)
- 窗口可并行: `ProcessPoolExecutor` (spawn context), 每worker独立初始化HMP模型

### 5.4 三种批处理方式对比

| 维度 | L-BFGS (main) | HMP (main) | HMP (dev) |
|------|--------------|------------|-----------|
| 处理单元 | 全序列 (B, T) | 128帧chunk堆叠 (N, 128) | 滑动窗口 (1, 128) × N |
| 固定形状 | 是 (B, T固定) | 是 (N×128 padding后堆叠) | 是 (每窗口(1,128),decode固定128) |
| padding帧排除 | vis_mask>=0 掩码 | 无排除(padding帧参与loss) | `T=torch.arange(valid_len)` 索引排除 |
| 窗口重叠 | 不适用 | 无重叠 | 16帧 overlap |
| 拼接方式 | 不需要 | 不需要 | cosine+quaternion blending |
| 并行 | 序列化(L-BFGS) | 序列化 | ProcessPoolExecutor 并行窗口 |
| vis_mask角色 | 功能性的(loss掩码) | 传入但未使用 | 传入但未使用(padding标记,非loss) |

**核心差异**: Dyn-HaMR的三条优化路径都保持**固定张量形状**(L-BFGS: (B,T); main HMP: (N,128); dev HMP: (1,128))，但对无效帧的排除方式不同——L-BFGS用vis_mask二值掩码、main HMP不排除、dev HMP用`T`有效帧索引。
