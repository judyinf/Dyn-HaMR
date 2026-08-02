# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dyn-HaMR (CVPR 2025 Highlight) recovers 4D interacting hand motion from monocular videos captured by **dynamic (moving) cameras**. It disentangles camera motion from hand motion in a shared global reference frame.

**Pipeline**: video → frame extraction → HaMeR hand tracking (MANO params) + VIPE/DROID-SLAM camera estimation → multi-stage L-BFGS optimization → visualization with rendered meshes.

## Common Commands

### Environment setup
```bash
source scripts/install_pip.sh        # pip-based virtualenv
source scripts/install_conda.sh      # conda-based (alternative)
source scripts/prepare.sh            # download model checkpoints to _DATA/
```

### Run optimization on a video
```bash
cd dyn-hamr

# VIPE camera (recommended, better quality):
python run_opt.py data=video_vipe run_opt=True data.seq=demo1 is_static=False

# DROID-SLAM camera (fallback):
python run_opt.py data=video run_opt=True data.seq=demo1 is_static=False

# With motion prior (stage III; requires chunk_size=128):
python run_opt.py data=video_vipe run_opt=True run_prior=True is_static=False

# One-shot: optimize + visualize:
python run_opt.py data=video_vipe run_opt=True run_vis=True is_static=False
```

### Visualize results
```bash
cd dyn-hamr
# Render all log directories found under log_root:
python run_vis.py --log_root <LOG_ROOT> --gpus 0

# Render specific views:
python run_vis.py --log_root <LOG_ROOT> -rv src_cam above side front -g

# Render with 2D keypoint overlays:
python run_vis.py --log_root <LOG_ROOT> -kp
```

### Configuration
- Set GPU: edit `dyn-hamr/confs/config.yaml` → `gpu`
- Set video path: edit `dyn-hamr/confs/data/video_vipe.yaml` → `root`, `video_dir`, `seq`
- Set frame interval: edit `data.start_idx` / `data.end_idx` in the data YAML
- Optimization weights: edit `dyn-hamr/confs/optim.yaml` → `optim.loss_weights`
- `is_static=True` for tripod/static camera (disables scale optimization)

## Architecture

### Entry Points
- **`dyn-hamr/run_opt.py`**: Hydra-based main script. Creates dataset → initializes MANO model → runs `BaseSceneModel.initialize()` with HaMeR observations → runs multi-stage optimization (RootOptimizer → SmoothOptimizer → optional HMP prior).
- **`dyn-hamr/run_vis.py`**: Visualization script. Can be used standalone (`--log_root`) or called from `run_opt.py` (`run_vis=True`). Loads results, runs MANO forward pass, renders multi-view videos and exports OBJ meshes.

### Optimization Pipeline (3 stages)

1. **RootOptimizer** (`optim/optimizers.py:RootOptimizer`): Optimizes only `trans` and `root_orient` per frame. Uses `RootLoss` (2D reprojection, 3D smoothness, biomechanical constraints). ~50 L-BFGS iterations.

2. **SmoothOptimizer** (`optim/optimizers.py:SmoothOptimizer`): Adds `betas`, `latent_pose`, optionally `world_scale` and camera params. Uses `SMPLLoss` which extends `RootLoss` with pose prior (deviation from HaMeR init), shape prior, and inter-hand penetration loss. ~300 iterations.

3. **HMP Prior** (optional, `run_prior=True`): Refines with learned hand motion prior from `dyn-hamr/HMP/`. Requires `chunk_size=128`.

### Key Modules

| Module | Purpose |
|--------|---------|
| `dyn-hamr/data/` | `MultiPeopleDataset` — loads per-track 2D keypoints (ViTPose), HaMeR MANO predictions (pose, shape, trans), camera parameters. Handles frame intervals, track visibility masks, keypoint interpolation. |
| `dyn-hamr/body_model/` | `MANO` class (wraps SMPL-X `MANOLayer`) — hand mesh model with 15 joints + wrist, 778 vertices. Factory: `run_mano()` handles left/right hand batching with separate face winding. |
| `dyn-hamr/optim/base_scene.py` | `BaseSceneModel` — holds MANO params (`trans`, `root_orient`, `latent_pose`, `betas`), cameras (`CameraParams`), and does MANO forward pass + camera-to-world transform of initial HaMeR predictions. |
| `dyn-hamr/optim/moving_scene.py` | `MovingSceneModel` — extends base with motion prior support, floor estimation, async track handling, world↔prior frame transforms. (Mostly commented out in current code — the active path uses `BaseSceneModel` directly.) |
| `dyn-hamr/optim/params.py` | `CameraParams` — manages camera extrinsics/intrinsics with optional optimization of `world_scale`, `cam_f`, `delta_cam_R`. |
| `dyn-hamr/optim/losses.py` | Loss functions: `RootLoss` (2D reprojection via GMoF, 3D joint/vertex losses, biomechanical, smoothness, depth constraint), `SMPLLoss` (adds pose prior, shape prior, penetration via winding numbers). |
| `dyn-hamr/optim/bio_loss.py` | `BMCLoss` — biomechanical constraints (bone length, RoM, anatomical validity). |
| `dyn-hamr/geometry/camera.py` | Camera math: `reproject()`, `perspective_projection()`, `invert_camera()`, `compose_cameras()`. |
| `dyn-hamr/geometry/rotation.py` | Rotation conversions: axis-angle ↔ rotation matrix, batch_rodrigues. |
| `dyn-hamr/vis/viewer.py` | Pyrender-based offscreen renderer (`OffscreenAnimation`) with multi-view support. |
| `dyn-hamr/vis/output.py` | `prep_result_vis()` — converts optimization results to mesh geometry; `animate_scene()` — renders videos. |
| `dyn-hamr/preproc/` | Preprocessing scripts: `extract_frames.py` (ffmpeg), `launch_hamer.py` (runs HaMeR on extracted frames), `launch_slam.py` / `run_slam.py` (camera estimation via VIPE or DROID-SLAM). |
| `dyn-hamr/HMP/` | Hand Motion Prior (HMP/NEMF): neural motion field for realistic hand motion sequences. `fitting.py` has `run_prior()`. |

### Configuration System

Uses [Hydra](https://hydra.cc/) with OmegaConf. Config files live in `dyn-hamr/confs/`:
- **`config.yaml`**: Top-level — sets defaults (`data: HOT3D`, `optim`), model flags (`opt_cams`, `opt_scale`, `async_tracks`), paths (`_DATA/data/`), MANO cfg, GPU, HMP config.
- **`optim.yaml`**: Loss weights (3-element lists for [stage1, stage2, stage3]), optimizer settings (L-BFGS params, iteration counts per stage).
- **`data/video_vipe.yaml`**: Video input config with VIPE camera estimation.
- **`data/video_driod.yaml`**: Video input config with DROID-SLAM camera.
- **`init.yaml`**: Initialization-specific overrides.

Hydra changes working directory to the output log dir at runtime (`hydra.job.chdir: True`).

### Data Flow

1. **Input**: Video → frames extracted to `<root>/images/<seq>/`
2. **HaMeR** processes frames → per-track MANO predictions at `<root>/dynhamr/track_preds/<seq>/<tid>/<frame>_mano.json`
3. **Camera**: VIPE or DROID-SLAM estimates camera poses → `<root>/dynhamr/cameras/<seq>/shot-0/cameras.npz`
4. **Shot detection**: `<root>/dynhamr/shot_idcs/<seq>.json` maps frame names to shot indices
5. **Optimization**: `MultiPeopleDataset` loads all of the above → `BaseSceneModel.initialize()` transforms HaMeR preds to world frame → multi-stage L-BFGS → saves `.npz` per stage to Hydra output dir
6. **Visualization**: `run_vis.py` loads `.npz` results → MANO forward pass → pyrender multi-view rendering → `.mp4` videos and `.obj` meshes

### Camera Coordinate Convention
- Cameras stored as **world-to-cam** (`w2c`): `cam_R` (T, 3, 3), `cam_t` (T, 3)
- `cam2world()`: `R_w2c.T`, `-R_w2c.T @ t_w2c`
- Camera at origin: first frame translation set near zero (with small random offset)

### Hand Model (MANO)
- 16 joints (1 wrist + 15 finger joints). `MANO_JOINTS` dict in `body_model/specs.py`.
- 778 vertices. Face winding differs for left vs right hand.
- Left hand: faces are right-hand faces with reversed winding `[:, [0,2,1]]`.
- MANO parameters: `global_orient` (3), `hand_pose` (45 = 15×3 axis-angle), `betas` (10 shape), `transl` (3).
- `pose2rot=True` in the MANO wrapper: axis-angle inputs are converted to rotation matrices internally.

### Important Conventions
- `B` = batch size (number of hand tracks, typically 1 or 2)
- `T` = sequence length (number of frames)
- Track IDs match hand chirality: `tid=0` → left hand (`is_right=0`), `tid=1` → right hand (`is_right=1`)
- Loss weights are 3-element lists in `optim.yaml` corresponding to the 3 optimization stages
- The pose prior is a simple L2 penalty toward the initial HaMeR prediction (not VPoser KL divergence)
- `vis_mask` is ternary: `-1` = out of scene, `0` = occluded, `1` = visible
