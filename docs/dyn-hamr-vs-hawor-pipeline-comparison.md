# Dyn-HaMR vs HaWoR：优化阶段前流水线对比分析

本文档详细对比 Dyn-HaMR 和 HaWoR 两个项目在**进入优化阶段之前**的数据流水线，包括视频预处理、手部检测与跟踪、HaMeR 预测、相机参数估计、坐标系约定、MANO 模型使用、中间产物格式等关键差异。

---

## 1. 整体流水线概览

### Dyn-HaMR 流水线

```
输入视频 (.mp4)
  │
  ├─[1] 帧提取 (cv2.VideoCapture)
  │   └─ 代码: dyn-hamr/preproc/extract_frames.py (split_frame)
  │   └─► images/{seq}/*.jpg
  │
  ├─[2] HaMeR 手部跟踪
  │   └─ 代码: dyn-hamr/preproc/launch_hamer.py → third-party/hamer/run.py
  │   ├─ 输出: {track_preds}/{seq}/results/{seq}.pkl (joblib)
  │   └─► 解包: dyn-hamr/preproc/export_hamer.py (export_sequence_results)
  │       ├─ {tid:03d}/{frame}_mano.json     ← MANO 参数 (相机空间)
  │       ├─ {tid:03d}/{frame}_keypoints.json ← ViTPose 2D 关键点
  │       └─ shot_idcs/{seq}.json            ← 镜头边界
  │
  ├─[3] 相机估计 (VIPE 推荐 / DROID-SLAM 备选)
  │   └─ 代码: dyn-hamr/data/vidproc.py (preprocess_cameras, load_vipe_cameras, run_vipe)
  │   │        dyn-hamr/preproc/run_slam.py (DROID-SLAM 内部)
  │   │        dyn-hamr/preproc/launch_slam.py (SLAM 调度)
  │   ├─ VIPE: 子进程调用 conda + vipe infer → pose/{seq}.npz (c2w) + intrinsics/{seq}.npz
  │   └─ DROID-SLAM: 内部运行
  │   └─► cameras/{seq}/shot-{idx}/cameras.npz
  │       {height, width, focal, intrins: (N,4), w2c: (N,4,4)}
  │
  └─[4] 优化入口
      └─ 代码: dyn-hamr/run_opt.py (main), dyn-hamr/data/dataset.py (MultiPeopleDataset)
      └─► MultiPeopleDataset 加载上述所有中间产物
```

### HaWoR 流水线

```
输入视频 (.mp4) + track_info.npy (外部检测)
  │
  ├─[1] 帧提取 (detect_track_video.py, ffmpeg)
  │   └─ 代码: scripts/scripts_test_video/detect_track_video.py (detect_track_video, extract_frames)
  │   └─► extracted_images/%04d.jpg
  │
  ├─[2] 检测后处理
  │   └─ 代码: scripts/scripts_test_video/detect_track_video.py (detect_track_video, postprocess_track_info_dict)
  │   │        track_info_post_process/track_info_postprocess.py (postprocess_track_info_dict)
  │   │        track_info_post_process/handedness_codec.py (normalize_deimv2_det_handedness)
  │   ├─ 输入: track_info.npy (DEIMv2 检测结果)
  │   ├─ 处理: 规范化 handedness → 置信度过滤 → 时序平滑 → 遮挡插值 → 去重
  │   └─► tracks_{start}_{end}/model_tracks.npy
  │       {track_id: [{frame, det, det_box(1,5), det_handedness}, ...]}
  │
  ├─[3] HaWoR 手部姿态估计
  │   └─ 代码: scripts/scripts_test_video/hawor_video.py (hawor_motion_estimation)
  │   │        lib/models/hawor.py (HAWOR 模型)
  │   │        lib/models/mano_wrapper.py (MANO 封装)
  │   │        hawor/utils/process.py (run_mano, run_mano_left)
  │   ├─ 模型: ViT + Transformer Decoder (自研，非标准 HaMeR)
  │   ├─ 左手: 图像水平翻转 → 推理 → y/z 旋转取反
  │   ├─ 渲染手部 mask → model_masks_packed.npz (供 SLAM 使用)
  │   └─► cam_space/{0,1}/{start}_{end}.json
  │       {init_root_orient, init_hand_pose, init_trans, init_betas}
  │
  ├─[4] 相机估计
  │   └─ 代码: scripts/scripts_test_video/hawor_slam.py (hawor_slam)
  │   │        lib/pipeline/masked_droid_slam.py (run_slam, 带手部 mask)
  │   │        lib/pipeline/est_scale.py (est_scale_hybrid, Metric3D 尺度)
  │   ├─ DROID-SLAM (带手部 mask 排除手部区域)
  │   ├─ Metric3D 度量尺度估计 (中值比 + BFGS 鲁棒优化)
  │   └─► SLAM/hawor_slam_w_scale_{start}_{end}.npz
  │       {tstamp, traj(相机到世界), scale(度量), mono_conf, ...}
  │
  ├─[5] 相机空间→世界空间转换
  │   └─ 代码: lib/eval_utils/custom_utils.py (cam2world_convert)
  │   └─► world_space_res.pth (joblib)
  │
  ├─[6] Infiller 缺失帧填充
  │   └─ 代码: infiller/lib/model/network.py (TransformerModel)
  │   │        lib/eval_utils/filling_utils.py (filling_preprocess, filling_postprocess)
  │   └─ 8 层 Transformer, 120 帧 horizon, 掩码注意力
  │
  └─[7] 最终输出
      └─ 代码: scripts/scripts_test_video/batch_hawor_infiller.py (save_mano_data)
      │        lib/utils/hand_relative_motion.py (augment_pose3d_hand)
      └─► mano/{clip_id}.pose3d_hand (torch.save)
          {left_hand, right_hand, slam_data, fps}
```

---

## 2. 视频预处理

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **提取方式** | `cv2.VideoCapture` 逐帧读取 (`split_frame()`) | `ffmpeg` 命令行 (`video_to_frames()`) |
| **帧率** | 配置 fps 参数指定 (默认 25) | 自动检测视频原始 fps |
| **起始帧索引** | 从 `1` 开始 (`%06d` 格式) | 从 `1` 开始 (`%04d` 格式) |
| **输出目录** | `{root}/images/{seq}/*.jpg` | `{seq_folder}/extracted_images/%04d.jpg` |
| **MOV 特殊处理** | `.MOV` 文件会上下翻转 (`frame[::-1, ::-1]`) | 无 |
| **代码位置** | `dyn-hamr/preproc/extract_frames.py` (`split_frame`) | `scripts/scripts_test_video/detect_track_video.py` (`extract_frames`) |

**关键差异**：Dyn-HaMR 使用 OpenCV 逐帧读取，HaWoR 使用 ffmpeg。两者的帧索引都从 1 开始，但 Dyn-HaMR 使用 6 位补零，HaWoR 使用 4 位。

> **HaWoR 项目根目录**：`/cpfs/yinghua/EgoDataPipeline/HaWoR`。下文所有 HaWoR 文件路径均相对于此根目录。
> **Dyn-HaMR 项目根目录**：`/cpfs/yinghua/Dyn-HaMR-main`。下文所有 Dyn-HaMR 文件路径均相对于此根目录。

---

## 3. 检测框的来源与后处理

### Dyn-HaMR：无显式检测框系统

Dyn-HaMR **不维护独立的检测框数据**。手部检测和跟踪完全由 HaMeR（基于 PHALP 的跟踪系统）内部完成。

- HaMeR (PHALP) 直接在整个图像上运行，输出每帧的跟踪结果
- 跟踪 ID (`tid`) 由 HaMeR 内部分配，直接对应手部手性：`tid=0` → 左手 (is_right=0)，`tid=1` → 右手 (is_right=1)
- 没有独立的检测框后处理流程
- 没有 handedness 纠正或时序平滑

### HaWoR：外部检测 + 深度后处理

HaWoR **不在内部运行检测器**，依赖外部系统（DEIMv2）提供的 `track_info.npy`。

**检测框来源**：DEIMv2 检测器，输出 5 类 handedness 标签：
```
ego_left=0, ego_right=1, likely_left=2, likely_right=3, other=4
```

**后处理流水线**（代码：`track_info_post_process/track_info_postprocess.py` → `postprocess_track_info_dict()`；`track_info_post_process/handedness_codec.py` → `normalize_deimv2_det_handedness()`）：

1. **规范化** (`normalize_track_info_det_handedness`)：5 类标签归一化为 3 类 (0=ego_left, 1=ego_right, 2=other)
2. **置信度过滤** (`filter_tracks_by_confidence`)：`det_box[4] < threshold` 的检测被设为 `det=False`（默认阈值 0.45）
3. **非主手过滤** (`filter_tracks_non_ego_handedness`)：handedness ≠ 0 且 ≠ 1 的检测被设为 `det=False`
4. **时序 handedness 平滑**：3 帧多数投票过滤异常值
5. **轨迹 handedness 锁定**：整条轨迹的 handedness 按加权多数投票锁定
6. **遮挡填补**（两类）：
   - 边缘遮挡：检测框缩小且靠近图像边缘 → 线性插值
   - 区域缩小遮挡：检测框缩小但不要求靠近边缘 → 线性插值
7. **边界框插值**：短间隙（≤ ~0.25s）的线性插值
8. **重复框去重**：同帧同侧 IoU > 0.35 或互相包含的重复检测去重

**Track ID 管理**（关键差异）：
- 外部输入的 track_id 可能不是 0/1
- HaWoR 在手部姿态估计阶段**重新索引**：所有左手数据 → tid=0，所有右手数据 → tid=1
- 输出目录使用手部索引命名：`cam_space/0/`（左手）、`cam_space/1/`（右手）

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **检测来源** | HaMeR 内部 (PHALP 跟踪) | 外部 DEIMv2 检测器 |
| **handedness 编码** | `is_right`: 0=左手, 1=右手 | `det_handedness`: 0=左手, 1=右手, 2=其他 |
| **后处理** | 无 | 8 步深度后处理流水线 |
| **track_id 语义** | tid == is_right (0=左, 1=右) | 重新索引为 {0: 左手, 1: 右手} |
| **检测框格式** | 不存储 | `det_box`: [x1, y1, x2, y2, confidence] |
| **track_info 存储** | 不需要 | `model_tracks.npy` (dict 格式) |

---

## 4. 手部姿态估计

### Dyn-HaMR：标准 HaMeR（PHALP 版本）

**模型**：`third-party/hamer/` 中的标准 HaMeR 实现（参见 `third-party/hamer/run.py`）

**配置文件**（HaMeR 使用）：视频数据配置在 `dyn-hamr/confs/data/video_vipe.yaml` 或 `dyn-hamr/confs/data/video_driod.yaml`；优化配置在 `dyn-hamr/confs/optim.yaml`；模型路径在 `dyn-hamr/confs/config.yaml`。

**运行方式**：
- 入口：`dyn-hamr/preproc/launch_hamer.py` → `process_seq()` → 子进程调用 `third-party/hamer/run.py`
- 输入：整个图像目录
- 输出：单个 `.pkl` 文件 (joblib 格式)，包含所有帧和跟踪结果
- 解包：`dyn-hamr/preproc/export_hamer.py` → `export_sequence_results()` → `export_hamer_predictions()` / `export_vitpose_keypoints()` / `export_shot_changes()` 解包为每帧每轨道的 JSON

**MANO 参数格式**（JSON 中）：
```json
{
  "betas": [10],           // MANO 形状参数
  "body_pose": [15, 3],    // 15 个关节的轴角 (从旋转矩阵转换)
  "global_orient": [3],    // 手腕全局旋转轴角 (从旋转矩阵转换)
  "cam_trans": [3],        // 相机空间中的手腕平移 (预异序！)
  "is_right": 0 或 1      // 手性
}
```

> **注意**：HaMeR 内部以**旋转矩阵**格式存储 `global_orient` 和 `hand_pose`（shape [3,3] 和 [15,3,3]），`dyn-hamr/preproc/export_hamer.py` 的 `unpack_frame()` 中通过 `cv2.Rodrigues` 转换为**轴角**。

**2D 关键点**：通过 ViTPose 提取，存储在 `{tid:03d}/{frame}_keypoints.json` 中：
```json
{
  "people": [{
    "pose_keypoints_2d": [21 * 3]  // 21 个关键点 (x, y, confidence)，OpenPose 手部格式
  }]
}
```

**左右手处理**：
- 使用同一 MANO 右手模型 (`MANO_RIGHT.pkl`)
- `is_right` 字段标记手性
- 优化阶段通过 `dyn-hamr/HMP/fitting.py` 的 `run_mano()` 函数处理：左手使用 `faces[:, [0,2,1]]`（反转面片绕组）

### HaWoR：自研 HaWoR 模型

**模型**：`lib/models/hawor.py` — `HAWOR` 类 (继承 `pl.LightningModule`)，ViT 主干 + Transformer 解码器（**非标准 HaMeR**）；配置定义在 `weights/hawor/model_config.yaml`，通过 `hawor/configs/__init__.py` 加载。

**架构**：
- ViT 主干（1280 维特征，192×256 裁剪）
- 可选时空 Transformer 模块（6 层，512 隐藏维）
- 可选运动 Transformer 模块（6 层，384 隐藏维）
- `MANOTransformerDecoderHead`（6 层，8 头，1024 MLP 维）

**运行方式**：
- 内部推理（PyTorch 或 TensorRT）
- 检测框裁剪手部图像 → 模型推理 → 回归 MANO 参数 + 3D 关节
- **不使用 ViTPose 或其他外部 2D 关键点检测器**
- 2D 关键点通过将 3D MANO 关节透视投影获得（仅用于训练监督）

**分块处理**：
- 固定序列长度 `SEQ_LEN = 16` 帧
- 轨迹被分割为连续块（最小长度 16 帧）
- 每个块独立推理

**左手特殊处理**（关键差异）：
1. 左手图像**水平翻转**后再输入模型（`do_flip=True`）
2. 推理后对旋转分量取反：
   ```python
   init_root[..., 1] *= -1    # y 轴
   init_root[..., 2] *= -1    # z 轴
   init_hand_pose[..., 1] *= -1
   init_hand_pose[..., 2] *= -1
   ```

**输出格式**（相机空间，`cam_space/{hand_idx}/{start}_{end}.json`）：
```json
{
  "init_root_orient": [T, 3, 3],   // 旋转矩阵格式
  "init_hand_pose": [T, 15, 3, 3], // 旋转矩阵格式
  "init_trans": [T, 3],             // 相机空间平移
  "init_betas": [T, 10]             // 形状参数
}
```

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **模型** | 标准 HaMeR (PHALP) | 自研 HaWoR (ViT+Transformer) |
| **2D 关键点** | ViTPose 外部提取 | 内部 3D→2D 投影 |
| **MANO 参数存储格式** | 轴角 (aa) | 旋转矩阵 (rotmat) |
| **批量大小** | 不固定（整段序列） | 固定 16 帧/块 |
| **左手处理** | 同右手模型 + 面片反转 | 专用左手模型 + 图像翻转 + 旋转取反 |
| **输出中间文件** | `{tid}/{frame}_mano.json` (JSON, 轴角) | `cam_space/{0,1}/{start}_{end}.json` (JSON, 旋转矩阵) |

### 两项目手部姿态模型的深度对比

#### 前向推理架构

**Dyn-HaMR 所用 HaMeR**（代码：`third-party/hamer/hamer/models/hamer.py` → `HAMER` 类，继承 `pl.LightningModule`）：

```
输入: 手部裁剪图 (B, 3, 256, 192)
  │
  ├─ 骨干网络: ViT-H (ViTPose 预训练)
  │   └─ img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32, num_heads=16
  │   └─ 输入裁剪: x[:,:,:,32:-32] — 去掉左右各 32 像素 (实际输入 256×128)
  │   └─ 输出: (B, 1280, Hp, Wp) — 空间特征图
  │
  ├─ MANO Head: TransformerDecoderHead
  │   └─ 特征重塑为 (B, Hp*Wp, 1280) — token 序列
  │   └─ Transformer Decoder: depth=6, heads=8, dim_head=64, mlp_dim=1024
  │   └─ 单查询 token (num_tokens=1, 零初始化)，交叉注意力以图像 token 为条件
  │   └─ 迭代误差反馈 (IEF): 默认 IEF_ITERS=1 轮
  │   └─ 三个线性投影头:
  │       ├─ decpose: Linear(1024, 96) — 16 个关节 × 6D 旋转表示
  │       ├─ decshape: Linear(1024, 10) — MANO betas
  │       └─ deccam: Linear(1024, 3) — 弱透视相机 [s, tx, ty]
  │
  └─ MANO 层: MANOLayer (pose2rot=False, 输入旋转矩阵)
      └─ 输出: joints (B, 21, 3), vertices (B, 778, 3)
      └─ 2D 关键点通过 perspective_projection(3D joints, cam_t, focal_length) 获得
```

**HaWoR 模型**（代码：`lib/models/hawor.py` → `HAWOR` 类，继承 `pl.LightningModule`）：

```
输入: 手部裁剪图 (B, 3, 256, 192) — 来自检测框
  │
  ├─ 骨干网络: ViT (timm vit_base 或类似)
  │   └─ embed_dim=1280, 输出 1280 维特征
  │   └─ 输入裁剪: x[:,:,:,32:-32] — 与 HaMeR 相同
  │
  ├─ 可选时空 Transformer 模块
  │   └─ depth=6, hidden_dim=512
  │   └─ 在时序块内聚合相邻帧特征 (SEQ_LEN=16)
  │
  ├─ 可选运动 Transformer 模块
  │   └─ depth=6, hidden_dim=384
  │   └─ 处理时序姿态序列
  │
  ├─ MANOTransformerDecoderHead
  │   └─ depth=6, heads=8, mlp_dim=1024
  │   └─ 直接回归:
  │       ├─ pred_rotmat: (T, 16, 3, 3) — 旋转矩阵格式 (非 6D)
  │       ├─ pred_trans: (T, 3) — 相机空间平移
  │       └─ pred_shape: (T, 10) — MANO betas
  │
  └─ MANO 层: MANOLayer (pose2rot=False, 输入旋转矩阵)
      └─ 输出: joints, vertices
      └─ 不使用外部 2D 关键点检测器
```

**关键架构差异**：

| 维度 | Dyn-HaMR 的 HaMeR | HaWoR |
|------|-------------------|-------|
| **骨干网络** | ViT-H (32 层, 1280 维) | ViT (1280 维) |
| **旋转表示** | 6D 旋转 (通过 rot6d_to_rotmat 转换) | 旋转矩阵 (直接输出) |
| **相机模型** | 弱透视 [s, tx, ty]，转换为 tz = 2f/(img_size*s) | 直接回归 (T, 3) 平移 |
| **时序建模** | 无 — 逐帧独立推理 | 可选时空/运动 Transformer 模块 (块大小 16) |
| **迭代精修** | IEF 循环 (默认 1 轮) | 无 IEF |
| **2D 关键点** | 由 ViTPose 独立提取 (133 全身关键点 → 21 手部) | 内部 3D→2D 投影 (仅训练监督) |
| **检测器** | YOLO 手部检测器 (run.py 中的 3 遍架构) | 外部 DEIMv2 (不在模型内部) |

#### 分块处理

**Dyn-HaMR 的 HaMeR**：
- **无时序分块**：HaMeR 逐帧独立推理。
- 批量处理：每帧检测到的手被组成批次（`batch_size=48`），但批次内不同条目来自不同帧/手，彼此独立。
- 时间一致性由外部 3 遍后处理保证（第 1 遍 YOLO 检测，第 2 遍清理/插值，第 3 遍 HaMeR 重跑）。
- 代码位置：`third-party/hamer/run.py` → `run_hamer_on_cleaned_bboxes()` → DataLoader(batch_size=48)。

**HaWoR**：
- **固定时序块**：`SEQ_LEN = 16`（定义在 `lib/models/hawor.py` 第 42 行）。
- 轨迹被 `parse_chunks()` 分割为连续块（`scripts/scripts_test_video/hawor_video.py`），最小块长度 16 帧。
- 每个块独立输入时空 Transformer 模块，进行时序特征聚合。
- 块间结果在输出时拼接。

#### 左右手特殊处理

**Dyn-HaMR 的 HaMeR**（代码：`third-party/hamer/hamer/datasets/vitdet_dataset.py` → `ViTDetDataset.__getitem__`）：

1. **输入图像翻转**：左手图像水平翻转（`flip = right == 0` → `img = img[:, ::-1]`），使模型只需学习右手几何。
2. **相机参数补偿**：`pred_cam[:, 1] *= (2*right - 1)` — 左手 x 平移取反。
3. **顶点镜像**：渲染时 `verts[:, 0] = (2*is_right - 1) * verts[:, 0]`。
4. **MANO 模型**：左右手共用同一个 `MANO_RIGHT.pkl`。左手通过面片绕组反转 `[:, [0,2,1]]` 实现。
5. **Track ID**：`is_right == tid`（从 PHALP 输出继承），tid=0 为左手，tid=1 为右手。

**HaWoR**（代码：`scripts/scripts_test_video/hawor_video.py` → `hawor_motion_estimation()`；`hawor/utils/process.py` → `run_mano_left()`）：

1. **输入图像翻转**：左手图像水平翻转（`do_flip=True`），与 HaMeR 相同。
2. **旋转分量取反**（推理后）：
   ```python
   init_root[..., 1] *= -1    # y 轴取反
   init_root[..., 2] *= -1    # z 轴取反
   init_hand_pose[..., 1] *= -1
   init_hand_pose[..., 2] *= -1
   ```
   这与 HaMeR 仅翻转 x 平移不同——HaWoR 对旋转的 y 和 z 分量取反。
3. **MANO 模型**：使用两个独立模型文件——`MANO_RIGHT.pkl` 和 `MANO_LEFT.pkl`。左手模型有 `shapedirs[:,0,:] *= -1` 修正。
4. **Track ID**：外部 track_id 被重新索引为 `{0: 左手, 1: 右手}`。

**左右手处理差异总结**：

| 处理步骤 | Dyn-HaMR 的 HaMeR | HaWoR |
|----------|-------------------|-------|
| 图像翻转 | 左手水平翻转 | 左手水平翻转（相同） |
| 相机平移补偿 | x 分量取反 (`pred_cam[:,1] *= -1`) | y, z 旋转分量取反 |
| MANO 模型 | 单一 `MANO_RIGHT.pkl` + 面片反转 | 双模型 `MANO_RIGHT.pkl` + `MANO_LEFT.pkl` |
| shapedirs 修正 | 不需要 | `shapedirs[:,0,:] *= -1` |
| 推理时旋转取反 | 不需要 (6D 旋转不含 y/z 奇偶性) | 需要 (直接输出旋转矩阵) |

---

## 5. 2D 关键点提取

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **方法** | ViTPose (外部 2D 姿态估计器) | 内部 3D MANO 关节透视投影 |
| **关节数量** | 21 (OpenPose 手部格式) | 21 (MANO 映射到 OpenPose) |
| **输出格式** | `{tid}/{frame}_keypoints.json` | 不单独保存 2D 关键点 |
| **在优化中的作用** | 作为 2D 重投影损失的真值 | 不在优化中使用（HaWoR 直接输出世界空间结果） |
| **置信度处理** | 保留 ViTPose 输出的置信度 | N/A |

---

## 6. 相机参数估计

### Dyn-HaMR

**后端选择**：
- **VIPE**（推荐）：`data=video_vipe` 配置
- **DROID-SLAM**（备选）：`data=video` 配置

**VIPE 流程**（代码：`dyn-hamr/data/vidproc.py` → `preprocess_cameras()` → `run_vipe()` / `load_vipe_cameras()` / `save_vipe_cameras_as_droid()`）：
1. 激活 conda 环境 `vipe`
2. 运行 `vipe infer {video_path}`
3. 读取 `pose/{seq}.npz`（**相机到世界** c2w 格式）和 `intrinsics/{seq}.npz`
4. **转换 c2w → w2c**：`w2c = np.linalg.inv(c2w)`
5. 保存为 DROID-SLAM 兼容格式：`cameras.npz` → `{w2c, intrins, height, width, focal}`

**DROID-SLAM 流程**（代码：`dyn-hamr/preproc/run_slam.py` → `main()` → `get_frame_cameras()` → `save_cameras()`；调度入口：`dyn-hamr/preproc/launch_slam.py` → `get_command()`）：
1. 图像缩放到约 384×512（8 的倍数）
2. 运行 DROID-SLAM 前端+后端优化
3. `w2c` (world-to-camera) 输出为 `(N, 4, 4)` 矩阵
4. 保存为 `cameras.npz`

**关键特征**：
- 无手部 mask 剔除（SLAM 在全图上运行）
- 无度量尺度估计（尺度由后续优化阶段恢复）
- 内参默认：`focal = 0.5 * (H + W)`，光心 = `(W/2, H/2)`

### HaWoR

**后端选择**：
- **droid**（默认）：DROID-SLAM + Metric3D 尺度估计
- **vo_grpc**：VGGT + DPVO + PGO（通过 gRPC 调用 Docker 容器）

**DROID-SLAM 流程**（代码：`scripts/scripts_test_video/hawor_slam.py` → `hawor_slam()`；`lib/pipeline/masked_droid_slam.py` → `run_slam()`——带手部 mask 的 DROID-SLAM）：

1. **手部 mask 渲染**（关键差异）：
   - 从 HaWoR 推理结果渲染 MANO 网格 → 二值 mask
   - mask 区域被清零（SLAM 忽略手部区域）
   - mask 保存为 `model_masks_packed.npz`

2. **DROID-SLAM**：
   - 图像缩放到 384×512，确保尺寸为 8 的倍数
   - 默认步长 2（可配置 `DROID_SLAM_STRIDE`）
   - 滑动窗口：关键帧阈值 5.0，前端阈值 16.0
   - 输出轨迹：(T, 7) = `[tx, ty, tz, qx, qy, qz, qw]`（**相机到世界** c2w 格式）

3. **Metric3D 尺度估计**（代码：`lib/pipeline/est_scale.py` → `est_scale_hybrid()`）：
   - Metric3D（`thirdparty/Metric3D`）预测每帧度量深度
   - 在 SLAM 关键帧上批量运行
   - `est_scale_hybrid()`：中值比 + BFGS Geman-McClure 鲁棒优化
   - 手部区域排除在尺度拟合之外
   - 鲁棒阈值：`near_thresh=0.4`, `far_thresh=0.7`

4. **分块 SLAM**：
   - `max_slam_frames` (默认 1800) 和 `slam_overlap_frames` (默认 15)
   - 使用重叠区域 SE3 对齐拼接轨迹

**保存格式**：`hawor_slam_w_scale_{start}_{end}.npz`
```python
{
    'tstamp': 关键帧索引,
    'traj': (T, 7) [tx,ty,tz, qx,qy,qz,qw],  # c2w, 四元数 x,y,z,w 顺序
    'img_focal': float,         # 平均焦距
    'img_center': [cx, cy],
    'scale': float,             # Metric3D 度量尺度
    'mono_conf_min/mean/std',   # Metric3D 置信度
    'dba_errors/residual_mean/final',
    'slam_weight_mean/min',
}
```

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **SLAM 后端** | VIPE (推荐) / DROID-SLAM | DROID-SLAM / vo_grpc |
| **手部 mask** | 不掩码 | 渲染 MANO → mask 排除手部 |
| **度量尺度** | 无 (优化阶段恢复) | Metric3D 估计 |
| **尺度估计方法** | N/A | 中值比 + BFGS 鲁棒优化 |
| **分块策略** | 按 shot 分块 | 按帧数均匀分块 (max 1800) |
| **轨迹格式** | w2c 4×4 矩阵 | c2w [tx,ty,tz,qx,qy,qz,qw] |
| **内参来源** | 默认 focal=0.5*(H+W) | 优先 K 矩阵文件，其次 img_focal |

### 相机内参来源对比

两个项目获取相机内参的方式存在本质差异：Dyn-HaMR 以内参作为 SLAM 的**输入**，HaWoR 则同时支持文件输入和基于图像的自动估计。

#### Dyn-HaMR 内参来源

**来源优先级**（代码：`dyn-hamr/data/vidproc.py` → `preprocess_cameras()`；`dyn-hamr/preproc/run_slam.py` → `load_intrins()`）：

1. **VIPE 路径**（`data=video_vipe`）：VIPE 直接输出内参 `intrinsics/{seq}.npz`，格式 `(N, 4)` [fx, fy, cx, cy]。VIPE 内部自行估计内参，Dyn-HaMR 无需额外处理。
2. **DROID-SLAM 路径**（`data=video`）：
   - a) **显式 .txt 文件**：若配置中指定了 `sources.intrins` 路径，`load_intrins()` 从文本文件读取 `(N, 4)` 或 `(1, 4)` 格式的 [fx, fy, cx, cy]。
   - b) **默认估计**：若未提供文件，`get_hwf()` 从第一帧图像读取 H、W，计算 `focal = 0.5 * (H + W)`，光心 = (W/2, H/2)。
3. **静态相机后备**（代码：`dyn-hamr/data/dataset.py` → `CameraData.load_data()`）：若 `cameras.npz` 不存在且 `is_static=True`，使用与 b) 相同的默认公式。

**内参格式演变**：
```
.txt 文件（如有）     → [fx, fy, cx, cy] 每行，或单行广播到所有帧
load_intrins() 输出   → (N, 6) [fx, fy, cx, cy, W, H]
image_stream() 输入   → (4,) [fx*sx, fy*sy, cx*sx, cy*sy]（按缩放比调整）
cameras.npz 保存      → intrins: (N, 4) [fx, fy, cx, cy]; focal: 标量 mean(fx,fy)
CameraData 加载后     → self.intrins: (T, 4)，按 img_w/orig_width 缩放
```

**内参传递链**：配置 → `load_intrins()` → `image_stream()` → DROID-SLAM → `cameras.npz` → `CameraData.as_dict()` → 优化阶段

#### HaWoR 内参来源

**来源优先级**（代码：`scripts/scripts_test_video/hawor_slam.py` → `resolve_slam_calib()`；`hawor_video.py` → `resolve_motion_calib()`）：

1. **显式内参文件**（`--intrinsics_file`）：
   - a) **VGGT JSON 格式**（`.json`）：包含 `median_native: {fx, fy, cx, cy}` 或 `per_frame: [{K_native}]`。代码：`hawor_video.py` → `_load_intrinsics_from_vggt_json()`。
   - b) **3×3 K 矩阵 .txt**（`.txt`）：9 个逗号分隔值 `fx,0,cx,0,fy,cy,0,0,1`。代码：`lib/pipeline/tools.py` → `load_intrinsics_k_from_txt()` → `calib_from_intrinsics_k()` 转换为 `[fx, fy, cx, cy]`。
2. **命令行焦距**（`--img_focal`）：标量焦距值（像素），光心从图像尺寸推断 (W/2, H/2)。
3. **持久化缓存**（`est_focal.txt`）：若以上均未提供，从 `{seq_folder}/est_focal.txt` 读取之前缓存的焦距值。若文件不存在，计算 `focal = W / 2.0` 并写入文件（代码：`hawor_slam.py` 第 363-369 行；`hawor_video.py` 第 115-125 行）。
4. **默认估计**（`est_calib()`）：代码：`lib/pipeline/masked_droid_slam.py` → `est_calib()`。`focal = max(H, W)`，`cx = W/2`，`cy = H/2`。

**内参格式演变**：
```
TXT 文件               → (3, 3) K 矩阵 → calib_from_intrinsics_k() → [fx, fy, cx, cy]
VGGT JSON              → {fx, fy, cx, cy} → img_focal=(fx+fy)/2, img_center=[cx, cy]
resolve_slam_calib()   → (4,) [focal, focal, cx, cy]
image_stream() 输入    → (4,) [fx*sx, fy*sy, cx*sx, cy*sy]（按缩放比调整）
hawor_slam NPZ 保存    → img_focal: 标量; img_center: (2,)
```

**SLAM 与运动估计使用不同的分辨率路径**：
- SLAM 从 `resolve_slam_calib()` 获取 `(4,)` 格式内参
- 运动估计从 `resolve_motion_calib()` 获取 `(img_focal, img_center)` 对

#### 内参来源差异总结

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **3×3 K 矩阵支持** | 不支持（仅 [fx,fy,cx,cy] txt） | 支持（txt K 矩阵 + VGGT JSON） |
| **VGGT 内参** | 不支持 | 支持（JSON 格式，含 per_frame 聚合） |
| **命令行焦距覆盖** | 不支持 | 支持（`--img_focal`） |
| **焦距缓存** | 不持久化（每次重新估计或从 cameras.npz 加载） | 持久化到 `est_focal.txt`（跨运行复用） |
| **默认焦距公式** | `0.5 * (H + W)` | 运动: `W / 2.0`；SLAM `est_calib`: `max(H, W)` |
| **VIPE 内参** | VIPE 自行估计，Dyn-HaMR 直接使用 | 不适用（HaWoR 不使用 VIPE） |
| **内参存储位置** | `cameras.npz` 中 `intrins (N,4)` | `hawor_slam_*.npz` 中 `img_focal`(标量) + `img_center`(2,) |
| **图像尺寸变化处理** | `CameraData` 按 img_w/orig_w 缩放内参 | `image_stream()` 按缩放比实时调整 |

---

## 7. 相机坐标系约定

### 坐标轴方向

两个项目在内部使用标准 OpenCV 相机坐标系（x 向右，y 向下，z 向前），但在存储和转换时有显著差异。

### Dyn-HaMR 约定

```
存储格式:     w2c (world-to-camera)
cam_R:        (T, 3, 3) — 世界到相机旋转矩阵
cam_t:        (T, 3) — 世界到相机平移向量
cam2world():  R.T 和 -R.T @ t
```

**初始化**（代码：`dyn-hamr/data/dataset.py` → `CameraData.load_data()`）：第一帧相机平移置为接近零（加小随机偏移 `torch.randn(3) * 0.1`），使世界坐标系原点接近第一帧相机位置。

**从相机空间到世界空间的转换**（代码：`dyn-hamr/optim/base_scene.py` → `BaseSceneModel.initialize()`）：
```python
R_c2w = R_w2c.transpose(-1, -2)
t_c2w = -torch.einsum("tij,tj->ti", R_c2w, t_w2c)

# 旋转: R_world = R_c2w @ R_cam
init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w, init_rot_mat)

# 平移: t_world = R_c2w @ root_loc + t_c2w + (t_cam - root_loc)
init_trans = R_c2w @ (init_trans + root_loc) + t_c2w - root_loc
```

**VIPE 输入**：VIPE 以 c2w 格式输出，Dyn-HaMR 在保存前转换为 w2c：
```python
w2c = np.linalg.inv(c2w)  # c2w → w2c
```

### HaWoR 约定

```
存储格式:     c2w (camera-to-world) 在 SLAM 输出中
traj:         (T, 7) [tx, ty, tz, qx, qy, qz, qw]
              平移 × scale 后为米制单位
```

**四元数顺序**：SLAM 输出为 `[x, y, z, w]`，在 `lib/eval_utils/custom_utils.py` → `load_slam_cam()` 中重排为 `[w, x, y, z]` 用于转换为旋转矩阵：
```python
pred_camq = torch.tensor(pred_traj[:, 3:], dtype=torch.float32)
R_c2w_sla = quaternion_to_matrix(pred_camq[:, [3,0,1,2]])  # [x,y,z,w] → [w,x,y,z]
R_w2c_sla = R_c2w_sla.transpose(-1, -2)
t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)
```

**从相机空间到世界空间的转换**（代码：`lib/eval_utils/custom_utils.py` → `cam2world_convert()`）：
```python
# 旋转: R_world = R_c2w @ R_cam
init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w_sla, init_rot_mat)

# 平移: t_world = R_c2w @ root_loc + t_c2w + (t_cam - root_loc)
root_loc = outputs["joints"][..., 0, :]  # MANO 手腕关节
offset = init_trans - root_loc            # 常量偏移
init_trans = R_c2w @ root_loc + t_c2w + offset
```

> **注意**：两者的 cam-to-world 转换公式**基本一致**，区别在于 Dyn-HaMR 从 w2c 推导 c2w，而 HaWoR 直接存储 c2w。

**可视化坐标系转换**（代码：`demo.py`）：
HaWoR 在渲染时应用 `R_x = [[1,0,0],[0,-1,0],[0,0,-1]]` 来对齐渲染坐标系：
```python
R_x = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]], dtype=torch.float32)
R_c2w_sla_vis = R_x @ R_c2w_sla
```

### 坐标系约定对比

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **存储格式** | w2c (world-to-camera) | c2w (camera-to-world) |
| **旋转存储** | 3×3 矩阵 | 四元数 `[tx,ty,tz, qx,qy,qz,qw]` |
| **度量尺度** | 无尺度因子 | `scale` 乘平移 |
| **世界原点** | 第一帧相机位置附近 | SLAM 定义的初始相机位置 |
| **四元数顺序** | N/A (使用矩阵) | 存储: [x,y,z,w]; 矩阵转换: [w,x,y,z] |
| **内参格式** | `[fx, fy, cx, cy]` | `[fx, fy, cx, cy]` 或 3×3 K 矩阵 |

### 左乘与右乘约定

两个项目在相机坐标系变换中**统一使用左乘**（`R @ x`，即旋转矩阵左乘列向量），但有一个重要的例外。

**左乘的含义**：若 `p_w` 是世界坐标系中的 3D 点（列向量），`R_w2c` 是世界到相机的旋转矩阵，则变换为：
```
p_c = R_w2c @ p_w + t_w2c    (左乘)
```

两个项目的所有 `einsum` 都采用 `"...ij,...j->...i"` 或 `"bij,bkj->bki"` 模式，等价于 `output[k] = R @ input[k]`，即左乘。

**唯一的右乘例外**：Dyn-HaMR 的 `dyn-hamr/optim/params.py` → `CameraParams.get_extrinsics()` 中：
```python
# 相机位姿扰动
cam_R = torch.matmul(cam_R, dR)  # cam_R @ dR — 右乘
```
此处 `dR`（delta 旋转，轴角格式通过 `batch_rodrigues` 转矩阵）被**右乘**到 `cam_R` 上。这表示 `dR` 是在**相机自身坐标系**中的扰动（外参增量），等价于 `R_new = R_old @ dR`。平移增量直接相加：`cam_t = cam_t + delta_cam_t`。此代码路径仅在 `opt_cams=True` 时激活（Dyn-HaMR 当前默认不优化相机）。

HaWoR 中**没有右乘**的使用。

### 两项目转换函数清单

#### Dyn-HaMR 转换函数

| 函数 | 文件 | 乘法模式 | 功能 |
|------|------|----------|------|
| `perspective_projection()` | `dyn-hamr/geometry/camera.py` | 左乘 `R @ x` | 3D 点透视投影到 2D |
| `reproject()` | `dyn-hamr/geometry/camera.py` | 左乘 `R @ x` | 批量时序 3D→2D 重投影 |
| `invert_camera()` | `dyn-hamr/geometry/camera.py` | 左乘 `-R.T @ t` | w2c ↔ c2w 互转 |
| `compose_cameras()` | `dyn-hamr/geometry/camera.py` | `R1 @ R2`, `t1 + R1 @ t2` | 两组外参的复合 |
| `relative_pose_c2w()` | `dyn-hamr/geometry/camera.py` | `Rc2w @ Rwc1` | c2w 位姿间的相对位姿 |
| `relative_pose_w2c()` | `dyn-hamr/geometry/camera.py` | 同上模式 | w2c 位姿间的相对位姿 |
| `lookat_matrix()` | `dyn-hamr/geometry/camera.py` | `R = [right, up, back]` | 构建 c2w 矩阵（列为轴） |
| `convert_yup()` | `dyn-hamr/geometry/camera.py` | `[x, -y, -z]` | 翻转 y 和 z 轴方向 |
| `CameraData.cam2world()` | `dyn-hamr/data/dataset.py` | `R.T, -R.T @ t` | w2c → c2w 转换 |
| `BaseSceneModel.initialize()` | `dyn-hamr/optim/base_scene.py` | `R_c2w @ R_cam`, `R_c2w @ (t+root) + t_c2w - root` | 相机空间→世界空间手部参数 |
| `CameraParams.get_extrinsics()` | `dyn-hamr/optim/params.py` | **右乘** `R @ dR` | 相机外参 delta 扰动（唯一右乘） |
| `load_vipe_cameras()` | `dyn-hamr/data/vidproc.py` | `inv(c2w)` | VIPE c2w → w2c 转换（4×4 矩阵求逆） |
| `run_mano()` | `dyn-hamr/HMP/fitting.py` | `(2*is_right-1) * joints[:,:,:,0]` | 左右手 x 坐标镜像 |

#### HaWoR 转换函数

| 函数 | 文件 | 乘法模式 | 功能 |
|------|------|----------|------|
| `load_slam_cam()` | `lib/eval_utils/custom_utils.py` | `R_c2w.T`, `-R_c2w.T @ t` | SLAM 轨迹 (c2w 四元数) → w2c + c2w |
| `quaternion_to_matrix()` | `lib/eval_utils/custom_utils.py` | 标准公式 | 四元数 (WXYZ) → 旋转矩阵 |
| `cam2world_convert()` | `lib/eval_utils/custom_utils.py` | `R_c2w @ R_cam`, `R_c2w @ root + t_c2w + offset` | 相机空间→世界空间手部参数 |
| `perspective_projection()` | `lib/utils/geometry.py` | 左乘 `R @ x`, `K @ x` | 3D 点透视投影（含畸变） |
| `world2canonical_convert()` | `lib/eval_utils/filling_utils.py` | `R_c2w @ R_cam` | 世界→规范坐标系转换 |
| `_compute_wrist_relative()` | `lib/utils/hand_relative_motion.py` | `R_{i-1}.T @ R_i`, `R_{i-1}.T @ dt` | 帧间相对运动计算 |
| `R_x` 可视化转换 | `demo.py` | `R_x @ R_c2w`, `R_x @ t_c2w` | Y-up→Z-up 坐标系对齐 |
| `batch_rodrigues()` | `hawor/utils/rotation.py` | `K @ K` | 轴角→旋转矩阵 |

> **关键一致性**：两个项目的 `cam2world` 转换函数（Dyn-HaMR 的 `BaseSceneModel.initialize()` 和 HaWoR 的 `cam2world_convert()`）使用**完全相同的左乘公式**：
> ```python
> # 旋转: R_world = R_c2w @ R_cam
> # 平移: t_world = R_c2w @ root_loc + t_c2w + offset  (offset = t_cam - root_loc)
> ```

### 世界坐标系原点的确定

**Dyn-HaMR**（代码：`dyn-hamr/data/dataset.py` → `CameraData.load_data()`）：
- 世界原点 = **第一帧相机位置附近**。
- 仅平移被重新居中——旋转不调整（被注释掉的 `R0` compose 代码在 `dataset.py` 第 362-364 行）。
- 具体做法：所有相机的平移减去 `t0 = -cam_t[0] + randn(3)*0.1`，使第一帧相机平移接近零（加微小随机偏移以防止数值退化）。
- 第一帧相机的旋转保持为 SLAM 输出的原始值（通常接近单位矩阵，但不强制为单位矩阵）。
- 这意味着世界坐标系的坐标轴方向由 SLAM 的第一帧相机朝向决定。

**HaWoR**：
- 世界原点 = **SLAM 定义的初始相机位置**。
- SLAM 输出 c2w 轨迹：`traj[i] = [tx_i, ty_i, tz_i, qx_i, qy_i, qz_i, qw_i]`。
- 第一帧（i=0）的 c2w 定义了世界坐标系。如果 DROID-SLAM 将第一帧设为参考帧，则 `t_c2w[0] ≈ (0,0,0)`、`q_c2w[0] ≈ (0,0,0,1)`（恒等四元数）。
- 平移需要乘以 `scale`（Metric3D 度量尺度）以获得米制单位。
- 由于 DROID-SLAM 内部可能不完全将第一帧设为原点，实际上世界原点在初始相机位置附近但可能有小偏移。

**关键差异**：Dyn-HaMR 强制将第一帧平移置为零（主动重新居中），而 HaWoR 依赖 SLAM 自然提供的参考帧。两个项目都没有对第一帧进行**旋转归一化**（即不强制第一帧相机旋转为单位矩阵）。

---

## 8. MANO 模型

### Dyn-HaMR

**模型文件**：仅使用 `MANO_RIGHT.pkl`

**初始化**（代码：`dyn-hamr/run_opt.py` → `run_opt()`）：
```python
mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
hand_model = MANO(batch_size=B*T, pose2rot=True, **mano_cfg).to(device)
```

- 单一 MANO 实例，`batch_size = B*T`（B=轨道数, T=帧数）
- `is_rhand=True`（默认）
- `pose2rot=True`：输入轴角自动转换为旋转矩阵
- 中性性别 (`GENDER: neutral`)
- 15 个手部关节 (`NUM_HAND_JOINTS: 15`)
- `MEAN_PARAMS: _DATA/data/mano_mean_params.npz`

**左手处理**（代码：`dyn-hamr/body_model/mano_wrapper.py` → `MANO` 类，继承自 `smplx.MANOLayer`；实际左右手面片切换在 `dyn-hamr/HMP/fitting.py` 的 `run_mano()` 中）：
- 使用同一个右手 MANO 模型
- 左手面片绕组反转：`l_faces = r_faces[:, [0, 2, 1]]`
- 不需要 `fix_shapedirs` 修正

**MANO 关节定义**（`dyn-hamr/body_model/specs.py`）：
```python
MANO_JOINTS = {
    'wrist': 0, 'index1': 1, 'index2': 2, 'index3': 3,
    'middle1': 4, 'middle2': 5, 'middle3': 6,
    'pinky1': 7, 'pinky2': 8, 'pinky3': 9,
    'ring1': 10, 'ring2': 11, 'ring3': 12,
    'thumb1': 13, 'thumb2': 14, 'thumb3': 15,
}
```

**额外面片**：在 `dyn-hamr/HMP/fitting.py` 的 `run_mano()` 中添加 14 个额外三角形以封闭手腕截面（使网格水密，用于穿透损失计算）。

### HaWoR

**模型文件**：使用两个独立的 MANO 模型
- `_DATA/data/mano/MANO_RIGHT.pkl` — 右手
- `_DATA/data_left/mano_left/MANO_LEFT.pkl` — 左手

**右手 MANO**（代码：`hawor/utils/process.py` → `run_mano()`）：
- 标准 `MANOLayer`，`is_rhand=True`

**左手 MANO**（代码：`hawor/utils/process.py` → `run_mano_left()`）：
- 专用左手模型，`is_rhand=False`，模型文件 `_DATA/data_left/mano_left/MANO_LEFT.pkl`
- **关键修正**：`mano.shapedirs[:, 0, :] *= -1`（修复 smplx 问题 #48 中左手 shapedirs 的符号错误）
- 左手面片：`faces_left = faces_right[:, [0, 2, 1]]`

**关节映射**（代码：`lib/models/mano_wrapper.py` → `MANO` 类）：
```python
mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
```
将 MANO 16 关节 + 指尖关节映射为 21 个 OpenPose 格式关节。

**额外面片**：与 Dyn-HaMR 相同，添加 14 个三角形封闭手腕截面。

### MANO 模型对比

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **右手模型** | `MANO_RIGHT.pkl` | `MANO_RIGHT.pkl` |
| **左手模型** | 同右手 + 面片反转 | `MANO_LEFT.pkl` + shapedirs 修正 |
| **batch_size** | B×T (整段序列) | 16 (固定块大小) |
| **pose2rot** | True (轴角→旋转矩阵) | False (HaWoR 直接输出旋转矩阵) |
| **wrap 方式** | 继承 MANOLayer，添加额外关节 | 继承 MANOLayer，添加额外关节 + joint_map |
| **关节输出** | 16 关节 | 21 关节 (MANO→OpenPose 映射) |
| **左手预处理** | 无需（同模型） | 图像水平翻转 + y/z 旋转取反 |

---

## 9. 中间产物完整清单

### Dyn-HaMR 中间产物

| 文件 | 格式 | 生成阶段 | 内容描述 |
|------|------|----------|----------|
| `images/{seq}/*.jpg` | JPG | 帧提取 | 提取的视频帧 (从 1 开始编号) |
| `dynhamr/track_preds/{seq}/results/{seq}.pkl` | joblib | HaMeR | 所有帧和轨道的完整 HaMeR 输出 |
| `dynhamr/track_preds/{seq}/{tid:03d}/{frame}_mano.json` | JSON | HaMeR 导出 | `{betas[10], body_pose[15,3], global_orient[3], cam_trans[3], is_right}` |
| `dynhamr/track_preds/{seq}/{tid:03d}/{frame}_keypoints.json` | JSON | HaMeR 导出 | `{people: [{pose_keypoints_2d: [21*3]}]}` |
| `dynhamr/shot_idcs/{seq}.json` | JSON | HaMeR 导出 | `{frame_name: shot_index}` |
| `dynhamr/cameras/{seq}/shot-{idx}/cameras.npz` | NPZ | 相机估计 | `{height, width, focal, intrins(N,4), w2c(N,4,4)}` |

### HaWoR 中间产物

| 文件 | 格式 | 生成阶段 | 内容描述 |
|------|------|----------|----------|
| `extracted_images/%04d.jpg` | JPG | 帧提取 | 提取的视频帧 |
| `tracks_{s}_{e}/model_tracks.npy` | NPY | 检测后处理 | `{track_id: [{frame, det, det_box[5], det_handedness}]}` |
| `tracks_{s}_{e}/model_boxes.npy` | NPY | 检测后处理 | 仅检测框（备用格式） |
| `tracks_{s}_{e}/model_masks_packed.npz` | NPZ | 手部姿态 | `packbits` 压缩的二值手部 mask (T,H,W) |
| `tracks_{s}_{e}/frame_chunks_all.npy` | joblib | 手部姿态 | `{0: [chunks], 1: [chunks]}` 手部时序块信息 |
| `cam_space/{0,1}/{start}_{end}.json` | JSON | 手部姿态 | `{init_root_orient[T,3,3], init_hand_pose[T,15,3,3], init_trans[T,3], init_betas[T,10]}` — 相机空间 |
| `SLAM/hawor_slam_w_scale_{s}_{e}.npz` | NPZ | 相机估计 | `{tstamp, traj(T,7), scale, mono_conf, dba_errors, ...}` |
| `world_space_res.pth` | joblib | 空间转换 | cam2world 转换后的世界空间手部参数 |
| `mano/{clip_id}.pose3d_hand` | torch.save | 最终输出 | 完整的世界空间 MANO 参数 + SLAM 数据 + 相对运动 |
| `mano/{clip_id}_confidence.npy` | NPY | 最终输出 | 检测/掩码/深度置信度统计 |

---

## 10. 插值与缺失帧处理

### Dyn-HaMR：传统插值方法

**MANO 参数插值**（代码：`dyn-hamr/data/tools.py` → `load_mano_preds()`）：
- **旋转** (global_orient, body_pose)：使用 `scipy.spatial.transform.Slerp`（球面线性插值）
- **平移** (cam_trans)：使用 `scipy.interpolate.interp1d` 线性插值
- **形状** (betas)：使用 `scipy.interpolate.interp1d` 线性插值
- **范围**：仅在可见帧的最小和最大索引之间插值 (`tmin` 到 `tmax`)

**2D 关键点插值**（代码：`dyn-hamr/data/tools.py` → `load_keypoints_with_interp()`）：
- 使用 `scipy.interpolate.interp1d` 线性插值
- 插值帧的置信度设为相邻可见帧的 80%

### HaWoR：Transformer Infiller 模型

**方法**：深度学习 Transformer 模型（代码：`infiller/lib/model/network.py` → `TransformerModel`）

**模型架构**：
- 8 层 Transformer，`d_model=384`，`nhead=8`，`d_hid=2048`
- Horizon = 120 帧（前后各 60 帧上下文）
- 掩码注意力：有效帧可以关注所有帧（包括无效帧），无效帧只能关注有效帧

**输入表示**（双手）：
- 手腕位置 (3) + 形状参数 (10) + 旋转 6D (15 关节 × 6 + 手腕 6 = 96)
- 双手总计：`2 * (3 + 10 + 96) = 218` 维

**预处理**（代码：`lib/eval_utils/filling_utils.py` → `filling_preprocess()`）：
- 将序列转换到规范坐标系（以第一个有效帧为参考）
- 对平移和形状进行 `linear_interpolation_nd()`
- 对旋转进行 `slerp_interpolation_aa()`
- 有效帧外推到序列边界

**后处理**（代码：`lib/eval_utils/filling_utils.py` → `filling_postprocess()`）：
- 应用 Transformer 输出（残差）
- 转换回世界坐标

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **旋转插值** | Slerp (球面线性) | Slerp 初始化 + Transformer 精修 |
| **平移插值** | 线性 | 线性初始化 + Transformer 精修 |
| **上下文范围** | 仅 track 范围内的可见帧 | ±60 帧 horizon (120 帧窗口) |
| **模型** | 无（纯数学插值） | 8 层 Transformer, ~38M 参数 |
| **处理对象** | 单个轨迹的 MANO 参数 | 双手同时填充 |

---

## 11. 帧有效性标记与在优化中的作用

两个项目使用**不同的机制**标记每帧手部数据的有效性，且这些标记直接影响后续优化或填充的行为。

### Dyn-HaMR：三值 `vis_mask`

#### 构建逻辑

`vis_mask` 的构建分两个阶段，核心代码在 `dyn-hamr/data/dataset.py`。

**阶段 A — 文件存在性检查**（`MultiPeopleDataset.__init__`，L157–169）：对每个手部轨道 (`tid`)，逐帧检查 ViTPose 2D 关键点 JSON 文件是否存在于磁盘上，生成 bool 数组 `track_vis_masks`：

```python
kp_paths = [f"{pred_dir}/{x}_keypoints.json" for x in self.img_names]
has_kp = [os.path.isfile(x) for x in kp_paths]
self.track_vis_masks.append(np.array(has_kp))  # True=文件存在, False=不存在
```

这是一个**纯文件系统级**的检查——不涉及任何视觉内容或语义信息。

**阶段 B — 三值化**（`get_ternary_mask()`，L402–411）：

```python
def get_ternary_mask(vis_mask):
    vis_mask = torch.as_tensor(vis_mask)
    vis_idcs = torch.where(vis_mask)[0]           # 所有有文件的帧索引
    track_s, track_e = min(vis_idcs), max(vis_idcs) + 1  # 轨道的时间跨度
    vis_mask = vis_mask.float()
    vis_mask[:track_s] = -1       # 首次出现之前 → 出镜
    vis_mask[track_e:] = -1       # 最后出现之后 → 出镜
    return vis_mask
    # 中间区域保持原值: True→1.0(可见), False→0.0(遮挡/检测失败)
```

#### 三值语义

| 值 | 含义 | 判定条件 | 代码位置 |
|----|------|---------|---------|
| **-1** | 手**不在场景中**（出镜） | `t < track_s` 或 `t ≥ track_e` | `dataset.py:408-409` |
| **0** | 遮挡或检测失败 | `track_s ≤ t < track_e` 且文件不存在 | `dataset.py:410` (保持原 False) |
| **1** | 可见（有检测） | `track_s ≤ t < track_e` 且文件存在 | `dataset.py:410` (保持原 True) |

**关键限制**：系统**无法区分**「物理遮挡」和「检测器漏检」。只要在轨道时间跨度内缺失关键点文件，一律标记为 0。运动模糊、极端光照导致的检测失败，与真正的遮挡被同等对待。区分出镜（-1）和遮挡（0）的**唯一依据**是帧是否在 `[track_s, track_e)` 区间内——这是一个纯时序启发式规则，不依赖任何视觉语义。

#### 与插值的关系

`vis_mask` 的构建依赖插值函数 `load_mano_preds()` 和 `load_keypoints_with_interp()`（`dyn-hamr/data/tools.py`），两者内部有**独立的** bool mask 计算（同样基于文件存在性）：

- 插值**仅**在轨道跨度内进行（`[tmin, tmax]`，对应 vis_mask 值为 0 或 1 的区域）
- 轨道跨度**之外**（对应 vis_mask = -1）的帧保持为零/默认值，不被插值
- 进入优化阶段时，vis_mask = -1 的帧被掩码完全排除，其 MANO 参数值（全零）不产生任何影响

#### 在优化中的使用

进入优化器前，`vis_mask` 被转换为**二值掩码**：`vis_mask >= 0`（`optim/optimizers.py:419,474`）。这意味着 **-1（出镜）被排除，0（遮挡）和 1（可见）都参与优化**——遮挡帧与可见帧在损失计算中完全等同。

各个 loss 函数使用掩码的方式：

| Loss | 使用方式 | 文件:行号 |
|------|---------|----------|
| `joints2d` (重投影) | 索引过滤：`joints2d_obs[mask], joints2d_pred[mask]` | `optim/losses.py:834-835` |
| `joints3d_smooth` (平滑) | 邻帧对过滤：`mask[:,1:] & mask[:,:-1]` — 相邻两帧都有效才计算 delta | `optim/losses.py:1245-1246` |
| `pose_prior` (姿态先验) | bool 索引：`loss[mask.bool()]` | `optim/losses.py:1190-1191` |
| `bio_loss` (生物力学) | bool 索引：`joints = ori_joints[valid_mask]` | `optim/bio_loss.py:280` |
| `penetration` (穿透) | **不使用掩码** — 对所有帧计算 | `optim/losses.py:614-618` |
| `shape_prior` (形状先验) | **不使用掩码** — 统一应用 | `optim/losses.py:595-596` |
| `joints3d` / `verts3d` | `vis_mask & ~isinf(data)` — 仅在观测数据存在且帧有效时 | `optim/losses.py:783-785` |

#### 优化中的完整数据流

```
HaMeR 检测 (run.py)
  │
  ├─► {tid:03d}/{frame}_keypoints.json  ← 文件是否存在？
  │
  ▼
MultiPeopleDataset.__init__
  │  track_vis_masks: bool[]  (文件存在性，全 shot 长度)
  ▼
MultiPeopleDataset.load_data()
  │  ├─ load_mano_preds()           Slerp/线性插值填充缺失帧 [tmin, tmax]
  │  ├─ load_keypoints_with_interp() 线性插值填充缺失帧 [tmin, tmax]
  │  └─ get_ternary_mask()          bool → {-1, 0, 1} 三值化
  │
  ▼
obs_data["vis_mask"]  (B, T) float tensor, 值 ∈ {-1.0, 0.0, 1.0}
  │
  ├─► 优化器: vis_mask >= 0 → 二值掩码 (排除-1, 保留0和1)
  │     ├─ RootOptimizer.forward_pass()   → RootLoss
  │     └─ SmoothOptimizer.forward_pass() → SMPLLoss
  │           ├─ joints2d_loss      ← mask (索引过滤)
  │           ├─ joints3d_smooth    ← mask (邻帧对)
  │           ├─ pose_prior         ← mask (bool索引)
  │           ├─ bio_loss           ← mask (bool索引)
  │           ├─ penetration         ← 无 mask
  │           └─ shape_prior         ← 无 mask
  │
  └─► 可视化: filter_visible_meshes(vis_mask >= 0)
        ├─ -1: 完全不显示该帧手部
        ├─  0: 半透明显示或正常显示 (vis_opacity)
        └─  1: 正常显示
```

### HaWoR：双 bool 数组 `pred_valid` + `detection_failed`

HaWoR 使用**两个独立的 bool 数组**表达帧有效性，定义在 `scripts/scripts_test_video/batch_hawor_infiller.py` → `save_mano_data()` 中，写入到 `.pose3d_hand` 的每只手结构里：

```python
{
    'left_hand': {
        'mano_params': {...},
        'pred_valid': (T,) bool,          # 每帧是否有有效预测
        'detection_failed': (T,) bool,    # 仅 allow_disappear=True 时存在
    },
    'right_hand': {...}
}
```

**判定逻辑**（代码：`scripts/scripts_test_video/hawor_video.py` → `hawor_motion_estimation()`；`scripts/scripts_test_video/batch_hawor_infiller.py`）：

- `pred_valid[i] = True`：HaWoR 模型在该帧成功推理（检测框存在且模型输出了有效 MANO 参数）
- `pred_valid[i] = False`：无检测框或推理失败
- `detection_failed[i] = True`（仅 `allow_disappear=True` 模式）：检测器明确报告失败，该帧被赋予默认隐藏平移 `[0.0, -0.4, -1.0]`
- `pred_valid[i] = False` 且 `detection_failed[i] = False`：该帧位于轨道跨度之外（手不在画面中）

**与 Dyn-HaMR 的关键区别**：HaWoR 可以区分「检测器明确报告失败」（detection_failed=True）和「手不在画面中」（pred_valid=False, detection_failed=False），而 Dyn-HaMR 将这两种情况合并为 vis_mask=0。

**在 Infiller 中的作用**：
- `pred_valid` 直接作为 Infiller Transformer 的注意力掩码：有效帧可以关注所有帧，无效帧只能关注有效帧
- `detection_failed` 决定是否使用默认的「手隐藏」姿态作为填充种子

### 帧有效性机制对比

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **数据结构** | `vis_mask`: (B, T) float，三值 {-1, 0, 1} | `pred_valid`: (T,) bool + `detection_failed`: (T,) bool |
| **定义位置** | `dyn-hamr/data/dataset.py` → `get_ternary_mask()` | `scripts/.../batch_hawor_infiller.py` → `save_mano_data()` |
| **出镜判定** | 基于轨道首尾检测帧的时序位置（纯启发式） | 基于检测器是否输出检测框（含 allow_disappear 模式） |
| **遮挡/检测失败** | 合并为 vis_mask=0，无法区分 | 可区分（detection_failed 独立标记） |
| **判定依据** | 关键点 JSON 文件是否存在（文件系统级） | HaWoR 模型推理是否成功（模型级） |
| **对下游影响** | `>= 0` 转换后控制 loss 掩码 | 控制 Infiller 注意力掩码和填充行为 |
| **代码位置** | `dataset.py:157-169, 402-411`；`optimizers.py:419,474`；`losses.py:783-1246` | `hawor_video.py`；`batch_hawor_infiller.py`；`infiller/lib/model/network.py` |

---

## 12. 检测、跟踪、后处理及遮挡/出镜判定全流程对比

本节从 Pipeline 视角完整对比两个项目的检测→跟踪→后处理→遮挡/出镜判定链路。

### Dyn-HaMR：3-Pass 流水线

Dyn-HaMR 的检测/跟踪/后处理**全部集中在 `third-party/hamer/run.py`**（1681 行）的 3-pass 架构中。

#### Pass 1: YOLO 检测 + 最高置信度筛选

```
逐帧 YOLO 手部检测器
  ├─ 输出: left_detections, right_detections (每类多个候选)
  └─ 每类取最高置信度: max(detections, key=lambda x: x['conf'])
      └─ 代码: run.py:185-238 (extract_raw_bboxes)
```

- 检测器：YOLO（替换了原始 HaMeR 的 Detectron2 + ViTPose 链）
- 同类多检处理：只保留最高置信度的那个
- 无传统 NMS——仅做简单的 argmax

#### Pass 2: bbox 清理（`clean_bbox_sequences`，5 个步骤）

```
Step 1: 移除超大 bbox (>50% 图像面积, MAX_BBOX_AREA_RATIO=0.5)
    ↓
Step 2: 手性纠正 (4 种启发式方法)
    ├─ fix_handedness_swaps_by_trajectory()  位置跳变检测 (200px阈值)
    ├─ fix_handedness_swaps()                邻居投票 (10帧窗口, 2:1比例)
    ├─ fix_handedness_swaps_frame_to_frame() IoU匹配到对侧轨迹 (0.6-0.7)
    └─ fix_handedness_inconsistencies()      时空一致性 (IoU>0.5, 1.5x比率)
    ↓
Step 3: 虚假短运动移除 (<30帧的孤立检测段, MIN_MOTION_DURATION=30)
    └─ 前后各≥30帧无检测的孤立段被完全删除 (bbox/keypoints/conf置None/0)
    ↓
Step 4: 缺失帧 bbox 插值
    ├─ 使用 PATIENCE_FRAMES=25：前方25帧内若有检测则插值
    ├─ 重叠去重: detect_overlapping_bboxes() — IoU>0.7或containment>0.7
    │   └─ 保留置信度高者, 被移除者记录 *_removed_due_to_overlap_with
    │   └─ 在 Step 2 和 Step 4 各调用一次 (插值可能产生新重叠)
    └─ bbox跳变检测: detect_bbox_jumps() — 自适应IoU阈值替换异常跳变帧
    ↓
Step 5: 恢复被误删的手 (如果 Step 4 中"胜出者"被 handness 检查标记为无效)
```

**代码位置**：`third-party/hamer/run.py:1006-1297`

#### Pass 3: HaMeR 在清理后的 bbox 上推理

- 对每帧每手独立运行 HaMeR（ViT-H + Transformer Decoder, `IEF_ITERS=1`）
- 左手图像水平翻转 (`flip = right == 0`)
- 相机 x 平移补偿：`pred_cam[:,1] *= (2*right - 1)`
- 输出保存至 `{tid:03d}/{frame}_mano.json` + `{tid:03d}/{frame}_keypoints.json`
- **无时序分块**：逐帧独立推理, 时序一致性由 Pass 2 保证

#### 出镜 vs 遮挡判定

```
判定流程 (全部在 Dyn-HaMR 优化阶段, 非 HaMeR 内部):
  track_vis_masks (bool, 全shot长度)
    │  基于: {tid}/{frame}_keypoints.json 文件是否存在?
    │  代码: dataset.py:157-169
    ▼
  get_ternary_mask()
    │  track_s = 第一个有文件的帧索引
    │  track_e = 最后一个有文件的帧索引 + 1
    │  代码: dataset.py:402-411 (仅 10 行)
    ▼
  结果:
    [0, track_s)   → -1 (出镜)
    [track_s, track_e) + 文件存在 → 1 (可见)
    [track_s, track_e) + 文件不存在 → 0 (遮挡/检测失败)
    [track_e, T)   → -1 (出镜)
```

**核心假设**：手一旦首次出现，直到最后一次出现之前，始终"应该"在画面中。**无任何视觉语义**验证手是否真的在画面中——这纯粹是基于时序边界和文件存在性的启发式规则。

**无法处理的情况**：
- 手真正离开画面后又重新进入（依赖 PHALP 分配新 tid 来切断轨道）
- 轨道首尾的误检拉长 `[track_s, track_e]` 区间
- 长段漏检被错误标记为遮挡（实际可能已出镜）
- 无法区分物理遮挡和检测器漏检——两者合并为 vis_mask=0

---

### HaWoR：外部检测 + 深度后处理 Pipeline

HaWoR 的检测**不在内部运行**，而是深度加工上游 DEIMv2 的输出。

#### 阶段 A: 加载外部检测 + 初始过滤

```
加载 track_info.npy (DEIMv2 输出)
  │  格式: {track_id: [{frame, det, det_box(1,5), det_handedness(5类)}]}
  │  代码: detect_track_video.py:301-335
  ▼
Step A1: 规范化 handedness (handedness_codec.py:26-62)
  │  DEIMv2 5类 → HaWoR 3类
  │  ego_left=0, ego_right=1, likely_left→2, likely_right→2, other→2
  ▼
Step A2: 置信度过滤 (detect_track_video.py:88-112)
  │  det_box[4] < threshold → det=False
  │  默认阈值: 0.45 (DEIMv2), 0.6 (YOLO)
  ▼
Step A3: 非主手过滤 (detect_track_video.py:115-139)
  │  handedness ≠ 0 且 ≠ 1 → det=False
  ▼
Step A4: 帧范围映射 (detect_track_video.py:32-50)
  │  model_tracks_filter_remap(): 过滤到 [start, end] 范围, 映射到本地帧号
  └─► tracks_{start}_{end}/model_tracks.npy
```

#### 阶段 B: 8 步深度后处理 (`track_info_postprocess.py:989-1193`)

```
Step B1: 手性单帧尖峰修正 (_fix_single_frame_handedness_spikes)
  └─ f-1和f+1一致但与f不同 → 修正f, 仅影响0/1标签
  └─ 代码: L624-676

Step B2: 时序手性平滑 (_temporal_smooth_handedness)
  └─ 加权投票(left/right=1, other=0.25) + 前向后向填充 + 3帧多数滤波
  └─ 代码: L678-751

Step B3: 短手性片段修正 (_fix_short_handedness_runs)
  └─ ≤5帧的孤立手性片段, 若两端一致则修正
  └─ 代码: L815-889

Step B4: 轨迹手性锁定 (_lock_track_handedness)
  └─ 加权多数投票锁定整条track的handedness
  └─ 若weighted_other > max(left,right): 整条锁为other, 跳过后续所有处理
  └─ 代码: L754-812

Step B5: 边缘遮挡填补 (_fill_edge_occlusion_gaps_for_track)
  └─ 条件: bbox面积缩小(ratio≤0.82) + 靠近图像边缘
  │        + 此前连续稳定帧 + 场景中双手俱全
  └─ 插值条件: 当前帧另一只手也存在
  └─ 代码: L422-527

Step B6: 区域缩小遮挡填补 (_fill_area_shrink_occlusion_gaps_for_track)
  └─ 同上但不要求靠近图像边缘 (纯面积缩小检测)
  └─ 代码: L530-621

Step B7: 短间隙bbox线性插值
  └─ 间隙≤自动估计的max_gap(基于fps, 上限6帧)
  └─ 漂移门控: 中心位移≤1.5×scale, 面积比≤4 (防止跨大运动/外观变化的插值)
  └─ 代码: L219-240, 1122-1168

Step B8: 同帧同侧去重 (_suppress_same_frame_same_side_duplicates)
  └─ IoU≥0.35或互相包含 → 保留最高置信度
  └─ 代码: L171-216
```

#### 阶段 C: 时序分块 + HAWOR 推理 (`hawor_video.py`)

```
Step C1: handedness 分流 (L198-220)
  │  ego_hand_side_index() 将det_handedness映射为0(左)/1(右)
  │  非主手检测被跳过
  │  final_tracks = {0: left_trk, 1: right_trk}
  ▼
Step C2: bbox线性插值 (L261-270)
  │  interpolate_bboxes(): 零值bbox线性填充 (custom_utils.py:101-117)
  ▼
Step C3: parse_chunks() 切分连续块 (L278)
  │  检测帧号不连续处 (frame[i+1]-frame[i] != 1) → 切分
  │  每chunk最小16帧 (SEQ_LEN=16)
  └─► frame_chunks_all.npy
  ▼
Step C4: HAWOR 模型分块推理 (L300-309)
  │  左手: do_flip=True (图像水平翻转), 推理后 y/z 旋转取反
  │  右手: do_flip=False
  └─► cam_space/{0,1}/{start}_{end}.json (旋转矩阵格式)
  ▼
Step C5: 手部mask渲染 (供SLAM使用)
  └─► model_masks_packed.npz
```

#### 阶段 D: Infiller 填充 + 最终输出

```
Step D1: cam2world_convert() — 相机空间→世界空间
  │  代码: custom_utils.py:10-40
  ▼
Step D2: 识别缺失帧
  │  pred_valid=0 的帧被分组为连续缺失chunk
  ▼
Step D3: Transformer Infiller 填充 (可选, skip_filling_missing_frames=True时跳过)
  │  输入: 双手 120帧窗口, 218维 (3+10+96)×2, 掩码注意力
  │  前处理: filling_preprocess() — Slerp/线性初始化 + 规范坐标系变换
  │  后处理: filling_postprocess() — 逆变换回世界坐标
  └─► pred_valid=1 (填充后的帧)
  ▼
Step D4: save_mano_data() 写入 .pose3d_hand
  │  allow_disappear=True: 整个视频无有效chunk时
  │  → 所有帧 detection_failed=True, 使用 DEFAULT_HIDDEN_TRANSL [0,-0.4,-1.0]
  └─► mano/{clip_id}.pose3d_hand
```

#### 出镜 vs 遮挡判定：多层隐式推断

HaWoR **不做显式的二值区分**，而是通过多层信号叠加隐式推断：

```
层级 1: 检测器级
  det=True/False  ← 检测框是否存在 + 置信度是否足够

层级 2: 后处理级 (隐式区分遮挡和出镜)
  边缘遮挡填补:
    条件: bbox面积缩小(≤0.82) + 靠近图像边缘 + 场景中有双手
    暗示: 手正在走出画面 (出镜)
  区域缩小填补:
    条件: bbox面积缩小但不靠边缘 + 场景中有双手
    暗示: 手被物体遮挡 (遮挡)
  短间隙插值:
    条件: ≤6帧的间隙, 漂移门控过滤大幅运动
    暗示: 短暂检测失败或瞬时遮挡

层级 3: 时序块级
  parse_chunks() 在帧号不连续处切分
  块间间隙(≥2帧) → 可以是出镜或长遮挡
  若间隙后出现新chunk → 暗示手重新进入画面 (消失→再现)

层级 4: Infiller级
  pred_valid=False 的帧 → Transformer填充
  若整个轨道无chunk + allow_disappear=True
    → detection_failed=True (手从未出现/全程检测失败)
```

**HaWoR 的关键优势**：利用 bbox 的**动态特征**（面积变化趋势、边缘位置）和**场景上下文**（另一只手是否存在），对帧缺失的原因做更细粒度的推断。

---

### 全流程对比总结

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **检测器** | YOLO (内置, 3-pass架构) | DEIMv2 (外部, track_info.npy 加载) |
| **同类多检处理** | 取最高置信度 (argmax) | 取最高置信度 + IoU去重 (≥0.35) |
| **跟踪器** | PHALP (HaMeR内置 3D跟踪) | 无内置 (track_id由上游预分配) |
| **后处理复杂度** | 5步 bbox清理 (run.py ~600行) | 8步深度后处理 (track_info_postprocess.py ~1200行) |
| **手性纠正** | 4种启发式 (位置跳变/邻居投票/IoU匹配/时空一致性) | 3阶段 (单帧修正/时序平滑/轨迹锁定) |
| **遮挡处理** | 无专门遮挡检测 (合并入vis_mask=0) | bbox面积缩小 + 边缘位置 + 双手上下文 |
| **缺失帧填补** | Slerp/线性插值 (区间内全部填补) | 短间隙线性插值 + 长间隙Transformer Infiller |
| **分块策略** | 无 (逐帧独立推理) | parse_chunks() 按帧号连续性切分, 最小16帧 |
| **出镜判定** | 纯时序启发式 (首尾检测帧区间边界, 10行代码) | 多层隐式推断 (动态特征 + 场景上下文 + chunk间隙) |
| **判定依据** | 关键点JSON文件是否存在 (文件系统级) | 检测框置信度 + bbox动态 + 模型推理结果 |
| **能否区分遮挡/检测失败** | 不能 (合并为 vis_mask=0) | 能 (detection_failed 独立标记) |
| **核心代码量** | `get_ternary_mask()` 10行 + `run.py` ~600行后处理 | `track_info_postprocess.py` ~1200行 + `hawor_video.py` ~600行 |

---

## 13. 最终输出格式

### Dyn-HaMR：优化阶段输出

Dyn-HaMR 的"最终输出"产生于**优化阶段内部**（非预处理阶段）。预处理仅为优化提供输入。

每个优化阶段保存为 `.npz` 文件：
```
{out_dir}/root_fit/{seq_name}_{iter:06d}_world_results.npz
{out_dir}/smooth_fit/{seq_name}_{iter:06d}_world_results.npz
{out_dir}/prior/{seq_name}_{iter:06d}_world_results.npz
```

NPZ 内容（代码：`dyn-hamr/optim/output.py` → `save_results()`；`dyn-hamr/optim/optimizers.py` → `StageOptimizer.save_results()`）：
```python
{
    'trans': (B, T, 3),          # 世界空间平移
    'root_orient': (B, T, 3),    # 世界空间全局旋转 (轴角)
    'pose_body': (B, T, 45),     # 手部姿态 (轴角)
    'betas': (B, 10),            # 形状参数
    'latent_pose': (B, T, D),    # VPoser 潜变量
    'cam_R': (B, T, 3, 3),      # 相机旋转
    'cam_t': (B, T, 3),         # 相机平移
    'intrins': (4,),             # 相机内参
    'is_right': (B, T),          # 手性
}
```

### HaWoR：`.pose3d_hand` 格式

预处理流水线的最终输出（优化前）。代码：`scripts/scripts_test_video/batch_hawor_infiller.py` → `save_mano_data()`；相对运动由 `lib/utils/hand_relative_motion.py` → `augment_pose3d_hand()` 添加：

```python
{
    'left_hand': {
        'mano_params': {
            'global_orient': (T, 3),     # 轴角，世界空间
            'hand_pose': (T, 45),        # 轴角，世界空间
            'betas': (T, 10),
            'transl': (T, 3),            # 世界空间手腕位置
        },
        'pred_valid': (T,) bool,          # 每帧是否有效
        'detection_failed': (T,) bool,    # (仅 allow_disappear=True)
        'relative_motion': {              # (通过 augment_pose3d_hand 添加)
            'rel_rot_aa': (T, 3),         # 帧间相对手腕旋转 (轴角)
            'rel_rot_mat': (T, 3, 3),     # 帧间相对手腕旋转 (矩阵)
            'rel_trans': (T, 3),          # 前一帧局部坐标系下的相对平移
            'pair_valid': (T,),           # 帧对是否都有效
        }
    },
    'right_hand': { 同上 },
    'slam_data': {
        'tstamp': (N,),
        'traj': (N, 7),                  # [tx,ty,tz,qx,qy,qz,qw] c2w
        'img_focal': float,
        'img_center': [cx, cy],
        'scale': float,                   # 度量尺度
        'slam_n_chunks': int,
        'max_slam_frames': int,
        'slam_overlap_frames': int,
    },
    'fps': float,
}
```

**存储格式**：`torch.save(mano_data, path)` — PyTorch 序列化格式，扩展名 `.pose3d_hand`

### 输出对比

| 维度 | Dyn-HaMR | HaWoR |
|------|----------|-------|
| **输出时机** | 优化阶段内部 | 预处理阶段末尾 |
| **世界空间结果** | 是 | 是 |
| **文件格式** | `.npz` (numpy) | `.pose3d_hand` (torch.save) |
| **相机参数** | 内嵌在 results 中 | 内嵌在 `slam_data` 中 |
| **度量尺度** | 未知（在优化中恢复） | 已知（Metric3D 估计） |
| **帧间相对运动** | 不预计算 | 预计算并内嵌 |
| **每帧有效性** | 通过 vis_mask 表达 | 通过 pred_valid + detection_failed 表达 |

---

## 14. 关键差异总结

### 哲学差异

- **Dyn-HaMR**：预处理提供**初始化和观测数据**（HaMeR 预测 + ViTPose 关键点 + 相机位姿），最优参数由下游多阶段 L-BFGS 优化确定。
- **HaWoR**：预处理提供**最终世界空间结果**（HaWoR + SLAM + Infiller），输出已经是可用的 3D 手部运动数据，优化在模型内部完成。

### 架构差异

1. **检测框系统**：Dyn-HaMR 无独立检测框系统（HaMeR 内嵌跟踪）；HaWoR 有完整的外部检测 + 8 步后处理流水线
2. **手部姿态模型**：Dyn-HaMR 用标准 HaMeR；HaWoR 用自研 ViT+Transformer 模型
3. **2D 关键点**：Dyn-HaMR 用 ViTPose；HaWoR 用内部 3D→2D 投影
4. **相机后端**：Dyn-HaMR 推荐 VIPE；HaWoR 使用 DROID-SLAM + Metric3D 或 vo_grpc
5. **尺度恢复**：Dyn-HaMR 在优化阶段恢复；HaWoR 预处理阶段通过 Metric3D 估计
6. **坐标存储**：Dyn-HaMR 用 w2c；HaWoR 用 c2w
7. **MANO 左手**：Dyn-HaMR 同右手模型 + 面片反转；HaWoR 专用左手模型 + shapedirs 修正 + 图像翻转
8. **插值策略**：Dyn-HaMR 用 Slerp/线性；HaWoR 用 Transformer 深度学习模型
9. **输出格式**：Dyn-HaMR 用 NPZ（优化阶段内）；HaWoR 用 `.pose3d_hand`（预处理末尾）
10. **帧有效性标记**：Dyn-HaMR 用三值 `vis_mask (-1/0/1)` 基于文件存在性 + 时序启发式；HaWoR 用双 bool `pred_valid + detection_failed` 基于模型推理结果
