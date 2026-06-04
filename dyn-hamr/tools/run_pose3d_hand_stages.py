#!/usr/bin/env python3
"""Run Dyn-HaMR optimization stages with .pose3d_hand as the stage interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

_SCRIPT_DIR = Path(__file__).resolve().parent
_DYN_HAMR_ROOT = _SCRIPT_DIR.parent
if str(_DYN_HAMR_ROOT) not in sys.path:
    sys.path.insert(0, str(_DYN_HAMR_ROOT))

sys.path.append(str(_DYN_HAMR_ROOT / "src/human_body_prior"))
sys.path.append(str(_DYN_HAMR_ROOT / "HMP"))

from body_model import MANO, OP_NUM_JOINTS
from util.loaders import resolve_cfg_paths
from util.logger import Logger
from util.tensor import move_to

from export_pose3d_hand import _compute_relative_motion


STAGE_ORDER = ("root", "smooth", "prior")
STAGE_DIR = {"root": "root_fit", "smooth": "smooth_fit", "prior": "prior"}


def _as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return arr.astype(dtype) if dtype is not None else arr


def _as_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    return Rotation.from_quat(quat).as_matrix().astype(np.float32)


def matrix_to_quat_xyzw(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(-1, 3, 3)
    return Rotation.from_matrix(mat).as_quat().astype(np.float32)


def traj_to_camera(traj: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32).reshape(-1, 7)
    R = quat_xyzw_to_matrix(traj[:, 3:7])
    t = traj[:, :3]
    if convention == "c2w":
        cam_R = np.swapaxes(R, -1, -2)
        cam_t = -np.einsum("tij,tj->ti", cam_R, t)
    elif convention == "w2c":
        cam_R, cam_t = R, t
    else:
        raise ValueError(f"Unsupported traj convention: {convention}")
    return cam_R.astype(np.float32), cam_t.astype(np.float32)


def camera_to_traj(cam_R: np.ndarray, cam_t: np.ndarray, convention: str) -> np.ndarray:
    cam_R = np.asarray(cam_R, dtype=np.float32).reshape(-1, 3, 3)
    cam_t = np.asarray(cam_t, dtype=np.float32).reshape(-1, 3)
    if convention == "c2w":
        R = np.swapaxes(cam_R, -1, -2)
        t = -np.einsum("tij,tj->ti", R, cam_t)
    elif convention == "w2c":
        R, t = cam_R, cam_t
    else:
        raise ValueError(f"Unsupported traj convention: {convention}")
    quat = matrix_to_quat_xyzw(R)
    return np.concatenate([t.astype(np.float32), quat], axis=-1).astype(np.float32)


def infer_seq_name(path: Path) -> str:
    name = path.name
    if name.endswith(".pose3d_hand"):
        name = name[: -len(".pose3d_hand")]
    for suffix in ("_export", "_root_fit", "_smooth_fit", "_prior"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def read_keypoints(path: Path) -> np.ndarray:
    empty = np.zeros((OP_NUM_JOINTS, 3), dtype=np.float32)
    if not path.is_file():
        return empty
    with path.open() as f:
        data = json.load(f)
    people = data.get("people", [])
    if not people:
        return empty
    keypoints = np.asarray(people[0].get("pose_keypoints_2d", empty), dtype=np.float32)
    return keypoints.reshape(-1, 3)


def write_keypoints(path: Path, keypoints: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"people": [{"pose_keypoints_2d": np.asarray(keypoints).reshape(-1, 3).tolist()}]}
    with path.open("w") as f:
        json.dump(payload, f)


def interpolate_keypoints(keypoints: np.ndarray, valid: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool)
    valid_idx = np.where(valid & ~np.all(keypoints == 0, axis=(1, 2)))[0]
    if valid_idx.size <= 1:
        return keypoints
    times = np.arange(valid_idx[0], valid_idx[-1] + 1)
    for joint_idx in range(keypoints.shape[1]):
        for coord in (0, 1, 2):
            f = interp1d(valid_idx, keypoints[valid_idx, joint_idx, coord], bounds_error=False)
            missing = times[~np.isin(times, valid_idx)]
            if missing.size:
                keypoints[missing, joint_idx, coord] = f(missing)
    return keypoints


def run_vitpose_for_box(
    image_path: Path,
    det_box: np.ndarray,
    config: Path | None,
    checkpoint: Path | None,
    device: str,
) -> np.ndarray:
    if config is None or checkpoint is None:
        raise RuntimeError(
            "Missing --vitpose-config/--vitpose-checkpoint and no cached keypoints found"
        )
    try:
        from mmpose.apis import inference_topdown, init_model
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("mmpose/ViTPose is not available in this environment") from exc

    model = run_vitpose_for_box._model_cache.get((str(config), str(checkpoint), device))
    if model is None:
        model = init_model(str(config), str(checkpoint), device=device)
        run_vitpose_for_box._model_cache[(str(config), str(checkpoint), device)] = model

    bbox = np.asarray(det_box[:4], dtype=np.float32)
    result = inference_topdown(model, str(image_path), bboxes=bbox[None])
    if not result:
        return np.zeros((OP_NUM_JOINTS, 3), dtype=np.float32)
    pred = result[0].pred_instances
    kpts = np.asarray(pred.keypoints[0], dtype=np.float32)
    scores = np.asarray(pred.keypoint_scores[0], dtype=np.float32)
    return np.concatenate([kpts, scores[:, None]], axis=-1).astype(np.float32)


run_vitpose_for_box._model_cache = {}


def load_track_info_npy(path: Path) -> dict[int, list[dict[str, Any]]]:
    raw = np.load(path, allow_pickle=True).item()
    return {int(float(k)): list(v) for k, v in raw.items()}


def pick_tracks(track_info: dict[int, list[dict[str, Any]]]) -> list[int]:
    by_hand: dict[int, tuple[int, int]] = {}
    for tid, entries in track_info.items():
        valid = [e for e in entries if bool(e.get("det", False))]
        if not valid:
            continue
        handed = int(round(float(valid[0].get("det_handedness", tid))))
        count = len(valid)
        old = by_hand.get(handed)
        if old is None or count > old[1]:
            by_hand[handed] = (tid, count)
    return [by_hand[h][0] for h in sorted(by_hand)]


def keypoint_path(root: Path, seq_name: str, track_dir_id: int, frame: int) -> Path:
    return root / seq_name / f"{track_dir_id:03d}" / f"{frame:06d}_keypoints.json"


@dataclass
class Pose3DHandStageData:
    pose_path: Path
    track_info_path: Path
    keypoints_root: Path
    image_root: Path | None
    seq_name: str
    traj_convention: str
    vitpose_config: Path | None = None
    vitpose_checkpoint: Path | None = None
    vitpose_device: str = "cuda:0"
    reuse_keypoints: bool = True

    def __post_init__(self) -> None:
        self.payload = torch.load(self.pose_path, map_location="cpu", weights_only=False)
        self.track_info = load_track_info_npy(self.track_info_path)
        self.track_ids = pick_tracks(self.track_info)
        if len(self.track_ids) == 0:
            self.track_ids = [0, 1]
        self.T = len(self.payload["left_hand"]["pred_valid"])
        self.seq_interval = (0, self.T)
        self._obs_cache: dict[str, Any] | None = None

    @property
    def n_tracks(self) -> int:
        return len(self.track_ids)

    def _side_for_track(self, track_id: int) -> str:
        return "right_hand" if self._handedness_for_track(track_id) == 1 else "left_hand"

    def _handedness_for_track(self, track_id: int) -> int:
        entries = self.track_info.get(track_id, [])
        handed = None
        for entry in entries:
            if bool(entry.get("det", False)):
                handed = int(round(float(entry.get("det_handedness", track_id))))
                break
        if handed is None:
            handed = int(track_id)
        return handed

    def _vis_mask_for_track(self, track_id: int, side: str) -> np.ndarray:
        mask = np.zeros(self.T, dtype=bool)
        for entry in self.track_info.get(track_id, []):
            frame = int(entry["frame"])
            if 0 <= frame < self.T and bool(entry.get("det", False)):
                mask[frame] = True
        pred_valid = np.asarray(self.payload[side]["pred_valid"], dtype=bool)
        return mask | pred_valid

    def _load_or_extract_keypoints(self, track_id: int) -> np.ndarray:
        keypoints = np.zeros((self.T, OP_NUM_JOINTS, 3), dtype=np.float32)
        valid = np.zeros(self.T, dtype=bool)
        # Existing Dyn-HaMR keypoint caches use 000/001 hand folders, not raw
        # track_info.npy ids such as 2 or 10000.
        track_dir_id = self._handedness_for_track(track_id)
        for entry in self.track_info.get(track_id, []):
            frame = int(entry["frame"])
            if frame < 0 or frame >= self.T or not bool(entry.get("det", False)):
                continue
            out_path = keypoint_path(self.keypoints_root, self.seq_name, track_dir_id, frame)
            if self.reuse_keypoints and out_path.is_file():
                kpts = read_keypoints(out_path)
            else:
                if self.image_root is None:
                    kpts = np.zeros((OP_NUM_JOINTS, 3), dtype=np.float32)
                else:
                    image_path = self.image_root / f"{frame:06d}.jpg"
                    if not image_path.is_file():
                        image_path = self.image_root / f"{frame:06d}.png"
                    kpts = run_vitpose_for_box(
                        image_path,
                        np.asarray(entry["det_box"], dtype=np.float32),
                        self.vitpose_config,
                        self.vitpose_checkpoint,
                        self.vitpose_device,
                    )
                    write_keypoints(out_path, kpts)
            keypoints[frame] = kpts
            valid[frame] = not np.all(kpts == 0)
        return interpolate_keypoints(keypoints, valid)

    def obs_data(self) -> dict[str, torch.Tensor | list[str]]:
        if self._obs_cache is not None:
            return self._obs_cache

        obs: dict[str, Any] = {
            "joints2d": [],
            "vis_mask": [],
            "is_right": [],
            "track_id": [],
            "seq_interval": [],
            "track_interval": [],
            "init_body_pose": [],
            "init_body_shape": [],
            "init_root_orient": [],
            "init_trans": [],
            "seq_name": [self.seq_name],
        }
        for track_id in self.track_ids:
            side = self._side_for_track(track_id)
            mano = self.payload[side]["mano_params"]
            is_right = 1.0 if side == "right_hand" else 0.0
            vis_mask = self._vis_mask_for_track(track_id, side)
            valid_idx = np.where(vis_mask)[0]
            track_s = int(valid_idx[0]) if valid_idx.size else 0
            track_e = int(valid_idx[-1] + 1) if valid_idx.size else self.T
            ternary = vis_mask.astype(np.float32)
            ternary[:track_s] = -1
            ternary[track_e:] = -1

            obs["joints2d"].append(self._load_or_extract_keypoints(track_id))
            obs["vis_mask"].append(ternary)
            obs["is_right"].append(np.full(self.T, is_right, dtype=np.float32))
            obs["track_id"].append(track_id)
            obs["seq_interval"].append(np.array(self.seq_interval, dtype=np.int32))
            obs["track_interval"].append(np.array([track_s, track_e], dtype=np.int32))
            obs["init_body_pose"].append(np.asarray(mano["hand_pose"], dtype=np.float32).reshape(self.T, 15, 3))
            obs["init_body_shape"].append(np.asarray(mano["betas"], dtype=np.float32))
            obs["init_root_orient"].append(np.asarray(mano["global_orient"], dtype=np.float32))
            obs["init_trans"].append(np.asarray(mano["transl"], dtype=np.float32))

        tensor_obs: dict[str, Any] = {}
        for key, value in obs.items():
            if key == "seq_name":
                tensor_obs[key] = value
            elif key == "track_id":
                tensor_obs[key] = torch.as_tensor(value, dtype=torch.long)
            elif key in {"seq_interval", "track_interval"}:
                tensor_obs[key] = torch.as_tensor(np.stack(value), dtype=torch.int32)
            else:
                tensor_obs[key] = torch.as_tensor(np.stack(value), dtype=torch.float32)
        self._obs_cache = tensor_obs
        return tensor_obs

    def camera_data(self) -> dict[str, torch.Tensor | bool]:
        slam = self.payload["slam_data"]
        cam_R, cam_t = traj_to_camera(np.asarray(slam["traj"]), self.traj_convention)
        focal = _as_float(slam["img_focal"])
        center = np.asarray(slam["img_center"], dtype=np.float32).reshape(2)
        intrins = np.tile(np.array([focal, focal, center[0], center[1]], dtype=np.float32), (self.T, 1))
        return {
            "cam_R": torch.from_numpy(cam_R),
            "cam_t": torch.from_numpy(cam_t),
            "intrins": torch.from_numpy(intrins),
            "static": False,
        }

    def initial_params(self) -> dict[str, torch.Tensor]:
        obs = self.obs_data()
        betas = torch.mean(obs["init_body_shape"], dim=1)
        return {
            "init_body_pose": obs["init_body_pose"],
            "latent_pose": obs["init_body_pose"].reshape(self.n_tracks, self.T, 45),
            "betas": betas,
            "trans": obs["init_trans"],
            "root_orient": obs["init_root_orient"],
            "is_right": obs["is_right"],
        }


def make_model(
    cfg: Any,
    body_model: MANO,
    stage_data: Pose3DHandStageData,
    device: torch.device,
) -> Any:
    from optim.base_scene import BaseSceneModel

    model = BaseSceneModel(
        stage_data.n_tracks,
        stage_data.T,
        body_model,
        pose_prior=None,
        use_init=False,
        opt_cams=bool(cfg.model.opt_cams),
        opt_scale=bool(cfg.model.opt_scale),
    )
    model.params.set_cameras(
        move_to(stage_data.camera_data(), device),
        opt_scale=bool(cfg.model.opt_scale),
        opt_cams=bool(cfg.model.opt_cams),
        opt_focal=bool(cfg.model.opt_cams),
    )
    for name, value in move_to(stage_data.initial_params(), device).items():
        model.params.set_param(name, value, requires_grad=False)
    return model.to(device)


def hand_slot_from_arrays(
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    betas: np.ndarray,
    transl: np.ndarray,
    pred_valid: np.ndarray,
) -> dict[str, Any]:
    global_orient = np.asarray(global_orient, dtype=np.float32).reshape(-1, 3)
    hand_pose = np.asarray(hand_pose, dtype=np.float32).reshape(len(global_orient), 45)
    transl = np.asarray(transl, dtype=np.float32).reshape(len(global_orient), 3)
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = np.tile(betas[None], (len(global_orient), 1))
    pred_valid = np.asarray(pred_valid, dtype=bool).reshape(len(global_orient))
    return {
        "mano_params": {
            "global_orient": global_orient,
            "hand_pose": hand_pose,
            "betas": betas.astype(np.float32),
            "transl": transl,
        },
        "pred_valid": pred_valid,
        "detection_failed": ~pred_valid,
        "relative_motion": _compute_relative_motion(
            torch.from_numpy(global_orient),
            torch.from_numpy(transl),
            torch.from_numpy(pred_valid),
        ),
    }


def save_pose3d_payload(payload: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def payload_from_model(
    base_payload: dict[str, Any],
    model: Any,
    traj_convention: str,
) -> dict[str, Any]:
    with torch.no_grad():
        params = model.params.get_dict()
        body_pose = model.latent2pose(model.params.latent_pose).detach().cpu().numpy()
        cam_R, cam_t, _, _ = model.params.get_cameras()
    trans = params["trans"].detach().cpu().numpy()
    root_orient = params["root_orient"].detach().cpu().numpy()
    betas = params["betas"].detach().cpu().numpy()
    is_right = params["is_right"].detach().cpu().numpy()

    payload = {
        "left_hand": base_payload["left_hand"],
        "right_hand": base_payload["right_hand"],
        "slam_data": dict(base_payload["slam_data"]),
        "fps": float(base_payload.get("fps", 30.0)),
    }
    for b in range(trans.shape[0]):
        side = "right_hand" if int(round(float(is_right[b, 0]))) == 1 else "left_hand"
        pred_valid = np.asarray(base_payload[side]["pred_valid"], dtype=bool)
        payload[side] = hand_slot_from_arrays(
            root_orient[b],
            body_pose[b],
            betas[b],
            trans[b],
            pred_valid,
        )
    payload["slam_data"]["traj"] = camera_to_traj(
        cam_R[0].detach().cpu().numpy(),
        cam_t[0].detach().cpu().numpy(),
        traj_convention,
    )
    return payload


def res_dict_from_payload(
    payload: dict[str, Any],
    track_ids: list[int],
    track_info: dict[int, list[dict[str, Any]]],
    traj_convention: str,
) -> dict[str, torch.Tensor]:
    root_orient = []
    pose_body = []
    trans = []
    betas = []
    is_right = []
    for track_id in track_ids:
        entries = track_info.get(track_id, [])
        handed = int(round(float(entries[0].get("det_handedness", track_id)))) if entries else int(track_id)
        side = "right_hand" if handed == 1 else "left_hand"
        mano = payload[side]["mano_params"]
        root_orient.append(np.asarray(mano["global_orient"], dtype=np.float32))
        pose_body.append(np.asarray(mano["hand_pose"], dtype=np.float32))
        trans.append(np.asarray(mano["transl"], dtype=np.float32))
        beta = np.asarray(mano["betas"], dtype=np.float32)
        if beta.ndim == 2:
            valid = np.asarray(payload[side]["pred_valid"], dtype=bool)
            beta = beta[valid].mean(axis=0) if valid.any() else beta.mean(axis=0)
        betas.append(beta.reshape(10))
        is_right.append(np.full(len(root_orient[-1]), float(handed), dtype=np.float32))
    slam = payload["slam_data"]
    cam_R, cam_t = traj_to_camera(np.asarray(slam["traj"]), traj_convention)
    focal = _as_float(slam["img_focal"])
    center = np.asarray(slam["img_center"], dtype=np.float32).reshape(2)
    intrins = np.array([focal, focal, center[0], center[1]], dtype=np.float32)
    B = len(track_ids)
    return {
        "root_orient": torch.from_numpy(np.stack(root_orient)),
        "pose_body": torch.from_numpy(np.stack(pose_body)),
        "trans": torch.from_numpy(np.stack(trans)),
        "betas": torch.from_numpy(np.stack(betas)),
        "is_right": torch.from_numpy(np.stack(is_right)),
        "cam_R": torch.from_numpy(np.tile(cam_R[None], (B, 1, 1, 1))),
        "cam_t": torch.from_numpy(np.tile(cam_t[None], (B, 1, 1))),
        "intrins": torch.from_numpy(intrins),
    }


def payload_from_prior_result(base_payload: dict[str, Any], result: dict[str, np.ndarray], track_ids: list[int], track_info: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    payload = {
        "left_hand": base_payload["left_hand"],
        "right_hand": base_payload["right_hand"],
        "slam_data": dict(base_payload["slam_data"]),
        "fps": float(base_payload.get("fps", 30.0)),
    }
    for b, track_id in enumerate(track_ids):
        entries = track_info.get(track_id, [])
        handed = int(round(float(entries[0].get("det_handedness", track_id)))) if entries else int(track_id)
        side = "right_hand" if handed == 1 else "left_hand"
        pred_valid = np.asarray(base_payload[side]["pred_valid"], dtype=bool)
        payload[side] = hand_slot_from_arrays(
            result["root_orient"][b],
            result["pose_body"][b],
            result["betas"][b],
            result["trans"][b],
            pred_valid,
        )
    return payload


def latest_pose3d(stage_dir: Path) -> Path:
    matches = sorted(stage_dir.glob("*.pose3d_hand"))
    if not matches:
        raise FileNotFoundError(f"No .pose3d_hand found under {stage_dir}")
    return matches[-1]


def stage_input(args: argparse.Namespace, stage: str) -> Path:
    if stage == "root" or not args.resume:
        return args.input_pose3d_hand
    previous = STAGE_ORDER[STAGE_ORDER.index(stage) - 1]
    return latest_pose3d(args.work_dir / STAGE_DIR[previous])


def build_cfg(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.load(_DYN_HAMR_ROOT / "confs" / "config.yaml")
    optim_cfg = OmegaConf.load(_DYN_HAMR_ROOT / "confs" / "optim.yaml")
    data_cfg = OmegaConf.load(_DYN_HAMR_ROOT / "confs" / "data" / f"{args.config_name}.yaml")
    cfg = OmegaConf.merge(cfg, optim_cfg)
    cfg.data = data_cfg
    cfg.data.seq = args.seq_name
    cfg.data.pose3d_hand = str(args.input_pose3d_hand)
    cfg.data.track_info = str(args.track_info)
    cfg.data.sources.keypoints = str(args.keypoints_root)
    cfg.model.opt_cams = False
    cfg.model.opt_scale = False
    cfg.run_vis = False
    cfg.run_opt = False
    cfg.paths.base_dir = str(_DYN_HAMR_ROOT.parent.resolve())
    return resolve_cfg_paths(cfg)


def make_body_model(cfg: Any, batch_size: int, device: torch.device) -> MANO:
    mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
    return MANO(batch_size=batch_size, pose2rot=True, **mano_cfg).to(device)


def stage_loss_weights(cfg: Any) -> list[dict[str, float]]:
    weights = cfg.optim.loss_weights
    return [{k: weights[k][i] for k in weights.keys()} for i in range(3)]


def run_root_or_smooth(args: argparse.Namespace, stage: str, cfg: Any, device: torch.device) -> Path:
    from optim.optimizers import RootOptimizer, SmoothOptimizer

    input_path = stage_input(args, stage)
    data = Pose3DHandStageData(
        pose_path=input_path,
        track_info_path=args.track_info,
        keypoints_root=args.keypoints_root,
        image_root=args.image_root,
        seq_name=args.seq_name,
        traj_convention=args.traj_convention,
        vitpose_config=args.vitpose_config,
        vitpose_checkpoint=args.vitpose_checkpoint,
        vitpose_device=args.device,
        reuse_keypoints=args.reuse_keypoints,
    )
    obs_data = move_to(data.obs_data(), device)
    body_model = make_body_model(cfg, data.n_tracks * data.T, device)
    model = make_model(cfg, body_model, data, device)
    all_loss_weights = stage_loss_weights(cfg)
    opt_kwargs = dict(cfg.optim.options)
    optimizer_cls = RootOptimizer if stage == "root" else SmoothOptimizer
    optimizer = optimizer_cls(model, all_loss_weights, **opt_kwargs)
    num_iters = int(cfg.optim.root.num_iters if stage == "root" else cfg.optim.smooth.num_iters)
    Logger.log(f"Running {stage} for {num_iters} iterations")
    for i in range(num_iters):
        optimizer.cur_step = i
        optimizer.loss.cur_step = i
        optimizer.optim_step(obs_data, i)
    optimizer.cur_step = num_iters
    payload = payload_from_model(data.payload, model, args.traj_convention)
    output = args.work_dir / STAGE_DIR[stage] / f"{args.seq_name}_{num_iters:06d}.pose3d_hand"
    return save_pose3d_payload(payload, output)


def run_prior_stage(args: argparse.Namespace, cfg: Any, device: torch.device) -> Path:
    from HMP.fitting import fitting_prior

    input_path = stage_input(args, "prior")
    data = Pose3DHandStageData(
        pose_path=input_path,
        track_info_path=args.track_info,
        keypoints_root=args.keypoints_root,
        image_root=args.image_root,
        seq_name=args.seq_name,
        traj_convention=args.traj_convention,
        vitpose_config=args.vitpose_config,
        vitpose_checkpoint=args.vitpose_checkpoint,
        vitpose_device=args.device,
        reuse_keypoints=args.reuse_keypoints,
    )
    obs_data = move_to(data.obs_data(), device)
    body_model = make_body_model(cfg, data.n_tracks * data.T, device)
    prior_dir = args.work_dir / STAGE_DIR["prior"]
    prior_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.base_dir = str(_DYN_HAMR_ROOT.parent.resolve())
    cfg.HMP.exp_name = args.seq_name
    result = fitting_prior(
        obs_data,
        (
            move_to(
                res_dict_from_payload(
                    data.payload,
                    data.track_ids,
                    data.track_info,
                    args.traj_convention,
                ),
                device,
            ),
        ),
        body_model,
        cfg,
        cfg.data,
        str(prior_dir),
        device,
    )
    if isinstance(result, tuple):
        result_dict = result[0]
    else:
        result_dict = result
    if result_dict is None:
        npz_path = prior_dir / f"{args.seq_name}_000000_world_results.npz"
        with np.load(npz_path, allow_pickle=False) as npz:
            result_dict = {k: npz[k] for k in npz.files}
    result_np = {k: _as_numpy(v) for k, v in result_dict.items()}
    payload = payload_from_prior_result(data.payload, result_np, data.track_ids, data.track_info)
    output = prior_dir / f"{args.seq_name}_000000.pose3d_hand"
    return save_pose3d_payload(payload, output)


def copy_final(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RootOptimizer, SmoothOptimizer, and HMP prior using .pose3d_hand "
            "as the stage input/output format.\n\n"
            "Optimization variables:\n"
            "  root: trans, root_orient\n"
            "  smooth: trans, root_orient, betas, latent_pose; optional world_scale, cam_f, delta_cam_R\n"
            "  prior: stg1 betas/trans/root_orient; stg2 betas/trans/root_orient/z_l; optional pose"
        ),
        epilog=(
            "Configuration files:\n"
            "  dyn-hamr/confs/data/video_new.yaml: pose3d_hand, track_info.npy, keypoint/image sources\n"
            "  dyn-hamr/confs/optim.yaml: Root/Smooth iteration counts, optimizer options, loss weights\n"
            "  dyn-hamr/confs/config.yaml: model.opt_cams/model.opt_scale defaults\n"
            "  dyn-hamr/HMP/hmp_config.yaml: HMP prior stages stg1/stg2/stg3 and opt_params\n\n"
            "pose3d_hand field mapping:\n"
            "  trans -> mano_params.transl\n"
            "  root_orient -> mano_params.global_orient\n"
            "  betas -> mano_params.betas\n"
            "  latent_pose/z_l decoded pose -> mano_params.hand_pose\n"
            "  slam_data.traj is kept as the stage camera trajectory and converted to cam_R/cam_t only internally"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-pose3d-hand", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("root", "smooth", "prior", "all"), default="all")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seq-name", default=None)
    parser.add_argument("--track-info", type=Path, required=True)
    parser.add_argument("--keypoints-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--vitpose-config", type=Path, default=None)
    parser.add_argument("--vitpose-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-keypoints", action="store_true")
    parser.add_argument("--traj-convention", choices=("c2w", "w2c"), default="c2w")
    parser.add_argument("--config-name", default="video_new")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_pose3d_hand = args.input_pose3d_hand.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.track_info = args.track_info.expanduser().resolve()
    args.keypoints_root = args.keypoints_root.expanduser().resolve()
    args.image_root = args.image_root.expanduser().resolve() if args.image_root else None
    args.vitpose_config = args.vitpose_config.expanduser().resolve() if args.vitpose_config else None
    args.vitpose_checkpoint = (
        args.vitpose_checkpoint.expanduser().resolve() if args.vitpose_checkpoint else None
    )
    args.seq_name = args.seq_name or infer_seq_name(args.input_pose3d_hand)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    Logger.init(str(args.work_dir / "pose3d_hand_stages.log"))
    cfg = build_cfg(args)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    stages = STAGE_ORDER if args.stage == "all" else (args.stage,)
    last_output = args.input_pose3d_hand
    for stage in stages:
        if stage in {"root", "smooth"}:
            last_output = run_root_or_smooth(args, stage, cfg, device)
        else:
            last_output = run_prior_stage(args, cfg, device)
        args.input_pose3d_hand = last_output
        args.resume = False

    copy_final(last_output, args.output)
    print(f"Saved final pose3d_hand to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
