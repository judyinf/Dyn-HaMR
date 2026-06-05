#!/usr/bin/env python3
"""Run Dyn-HaMR stages with .pose3d_hand and Hydra configuration."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

_DYN_HAMR_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _DYN_HAMR_ROOT.parent
if str(_DYN_HAMR_ROOT) not in sys.path:
    sys.path.insert(0, str(_DYN_HAMR_ROOT))

sys.path.append(str(_DYN_HAMR_ROOT / "src/human_body_prior"))
sys.path.append(str(_DYN_HAMR_ROOT / "HMP"))
sys.path.append(str(_REPO_ROOT / "third-party/hamer/third-party/ViTPose"))

from body_model import MANO, OP_NUM_JOINTS
from preproc.extract_frames import video_to_frames
from util.loaders import resolve_cfg_paths
from util.logger import Logger
from util.tensor import move_to


STAGE_ORDER = ("root", "smooth", "prior")
STAGE_DIR = {"root": "root_fit", "smooth": "smooth_fit", "prior": "prior"}


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_DYN_HAMR_ROOT / path).resolve()


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


def _slam_scale_value(slam: dict) -> float:
    return _as_float(slam.get("scale", 1.0))


def slam_traj_to_camera_unscaled(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    traj layout: [:3] unscaled camera center t_c2w, [3:7] R_w2c quaternion (xyzw).
    Returns unscaled (R_w2c, cam_t) with cam_t = -R_w2c @ t_c2w.
    """
    traj = np.asarray(traj, dtype=np.float32).reshape(-1, 7)
    cam_R = quat_xyzw_to_matrix(traj[:, 3:7])
    t_c2w = traj[:, :3]
    cam_t = -np.einsum("tij,tj->ti", cam_R, t_c2w)
    return cam_R.astype(np.float32), cam_t.astype(np.float32)


def camera_to_slam_traj_unscaled(cam_R: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
    """Inverse of slam_traj_to_camera_unscaled; does not apply scale."""
    cam_R = np.asarray(cam_R, dtype=np.float32).reshape(-1, 3, 3)
    cam_t = np.asarray(cam_t, dtype=np.float32).reshape(-1, 3)
    t_c2w = -np.einsum("tij,tj->ti", np.swapaxes(cam_R, -1, -2), cam_t)
    quat = matrix_to_quat_xyzw(cam_R)
    return np.concatenate([t_c2w.astype(np.float32), quat], axis=-1).astype(np.float32)


def slam_traj_to_camera_scaled(traj: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply scale to translation only: cam_t_scaled = cam_t_unscaled * scale."""
    cam_R, cam_t = slam_traj_to_camera_unscaled(traj)
    return cam_R, (cam_t * np.float32(scale)).astype(np.float32)


def _compute_relative_motion(
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    pred_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    T = global_orient.shape[0]
    rel_trans = torch.zeros(T, 3, dtype=torch.float32)
    rel_rot_mat = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(T, 1, 1)
    rel_rot_aa = torch.zeros(T, 3, dtype=torch.float32)
    pair_valid = torch.zeros(T, dtype=torch.bool)

    rot_mats_np = Rotation.from_rotvec(global_orient.detach().cpu().numpy()).as_matrix()
    rot_mats = torch.from_numpy(rot_mats_np.astype(np.float32))
    for t in range(1, T):
        if not (bool(pred_valid[t]) and bool(pred_valid[t - 1])):
            continue
        pair_valid[t] = True
        rel_trans[t] = transl[t] - transl[t - 1]
        delta = rot_mats[t] @ rot_mats[t - 1].transpose(-1, -2)
        rel_rot_mat[t] = delta
        rel_rot_aa[t] = torch.from_numpy(
            Rotation.from_matrix(delta.numpy()).as_rotvec().astype(np.float32)
        )

    return {
        "rel_trans": rel_trans,
        "rel_rot_mat": rel_rot_mat,
        "rel_rot_aa": rel_rot_aa,
        "pair_valid": pair_valid,
    }


def infer_seq_name(path: Path) -> str:
    name = path.name
    if name.endswith(".pose3d_hand"):
        name = name[: -len(".pose3d_hand")]
    for suffix in ("_export", "_root_fit", "_smooth_fit", "_prior"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def interpolate_keypoints(keypoints: np.ndarray, valid: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool)
    valid_idx = np.where(valid & ~np.all(keypoints == 0, axis=(1, 2)))[0]
    if valid_idx.size <= 1:
        return keypoints
    times = np.arange(valid_idx[0], valid_idx[-1] + 1)
    missing = times[~np.isin(times, valid_idx)]
    if missing.size == 0:
        return keypoints
    for joint_idx in range(keypoints.shape[1]):
        for coord in (0, 1, 2):
            f = interp1d(valid_idx, keypoints[valid_idx, joint_idx, coord], bounds_error=False)
            keypoints[missing, joint_idx, coord] = f(missing)
    return keypoints


def run_vitpose_for_box(
    image_path: Path,
    det_box: np.ndarray,
    config: Path | None,
    checkpoint: Path | None,
    device: str,
    handedness: int | None = None,
) -> np.ndarray:
    if config is None or checkpoint is None:
        raise RuntimeError(
            "Missing vitpose_config/vitpose_checkpoint and keypoints_npy does not exist"
        )
    bbox = np.asarray(det_box[:4], dtype=np.float32)
    try:
        from mmpose.apis import inference_topdown, init_model

        api = "mmpose_v1"
    except Exception:
        try:
            from mmpose.apis import inference_top_down_pose_model, init_pose_model

            api = "mmpose_v0"
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("mmpose/ViTPose is not available in this environment") from exc

    cache_key = (api, str(config), str(checkpoint), device)
    model = run_vitpose_for_box._model_cache.get(cache_key)
    if model is None:
        if api == "mmpose_v1":
            model = init_model(str(config), str(checkpoint), device=device)
        else:
            model = init_pose_model(str(config), str(checkpoint), device=device)
        run_vitpose_for_box._model_cache[cache_key] = model

    if api == "mmpose_v1":
        result = inference_topdown(model, str(image_path), bboxes=bbox[None])
        if not result:
            return np.zeros((OP_NUM_JOINTS, 3), dtype=np.float32)
        pred = result[0].pred_instances
        kpts = np.asarray(pred.keypoints[0], dtype=np.float32)
        scores = np.asarray(pred.keypoint_scores[0], dtype=np.float32)
        keypoints = np.concatenate([kpts, scores[:, None]], axis=-1)
    else:
        pose_results, _ = inference_top_down_pose_model(
            model,
            str(image_path),
            person_results=[{"bbox": bbox}],
            bbox_thr=None,
            format="xyxy",
        )
        if not pose_results:
            return np.zeros((OP_NUM_JOINTS, 3), dtype=np.float32)
        keypoints = np.asarray(pose_results[0]["keypoints"], dtype=np.float32)

    if keypoints.shape[0] > OP_NUM_JOINTS:
        if handedness == 0:
            keypoints = keypoints[91:112]
        elif handedness == 1:
            keypoints = keypoints[112:133]
    if keypoints.shape != (OP_NUM_JOINTS, 3):
        raise ValueError(f"ViTPose returned keypoints with shape {keypoints.shape}; expected {(OP_NUM_JOINTS, 3)}")
    return keypoints.astype(np.float32)


run_vitpose_for_box._model_cache = {}


def _normalize_track_entry(entry: dict[str, Any]) -> dict[str, Any]:
    handedness = np.asarray(entry["det_handedness"], dtype=np.float32)
    return {
        "frame": int(entry["frame"]),
        "det": bool(entry["det"]),
        "det_box": np.asarray(entry["det_box"], dtype=np.float32),
        "det_handedness": np.float32(handedness.item()) if handedness.shape == () else handedness,
    }


def _coerce_track_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return _normalize_track_entry(entry)


def _normalize_track_info_on_load(raw: dict[Any, Any]) -> dict[int, list[dict[str, Any]]]:
    return {
        int(float(k)): [_normalize_track_entry(entry) for entry in list(v)]
        for k, v in raw.items()
    }


def _coerce_track_info_on_save(track_info: dict[int, list[dict[str, Any]]]) -> dict[int, list[dict[str, Any]]]:
    return {
        tid: [_coerce_track_entry(entry) for entry in entries]
        for tid, entries in track_info.items()
    }


def load_track_info_npy(path: Path) -> dict[int, list[dict[str, Any]]]:
    raw = np.load(path, allow_pickle=True).item()
    return _normalize_track_info_on_load(raw)


def save_track_info_npy(path: Path, track_info: dict[int, list[dict[str, Any]]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _coerce_track_info_on_save(track_info)
    np.save(path, payload, allow_pickle=True)
    return path


def group_tracks_by_handedness(track_info: dict[int, list[dict[str, Any]]]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {0: [], 1: []}
    for tid, entries in track_info.items():
        handedness = None
        for entry in entries:
            if bool(entry.get("det", False)):
                handedness = int(round(float(entry.get("det_handedness", tid))))
                break
        if handedness in grouped:
            grouped[handedness].append(tid)
    return grouped


def select_candidates_by_frame(
    entries_by_track: dict[int, list[dict[str, Any]]],
    T: int,
) -> tuple[list[dict[str, Any] | None], np.ndarray, np.ndarray]:
    selected: list[dict[str, Any] | None] = [None] * T
    source_track = np.full(T, -1, dtype=np.int64)
    score = np.zeros(T, dtype=np.float32)
    prev_center: np.ndarray | None = None

    for frame in range(T):
        candidates = []
        for tid, entries in entries_by_track.items():
            for entry in entries:
                if int(entry["frame"]) == frame and bool(entry.get("det", False)):
                    box = np.asarray(entry["det_box"], dtype=np.float32)
                    candidates.append((tid, entry, box))
        if not candidates:
            continue
        candidates.sort(key=lambda item: float(item[2][4]) if item[2].size >= 5 else 0.0, reverse=True)
        best_tid, best_entry, best_box = candidates[0]
        if prev_center is not None and len(candidates) > 1:
            best_score = float(best_box[4]) if best_box.size >= 5 else 0.0
            close = [
                item for item in candidates
                if best_score - (float(item[2][4]) if item[2].size >= 5 else 0.0) < 0.05
            ]
            if len(close) > 1:
                best_tid, best_entry, best_box = min(
                    close,
                    key=lambda item: float(
                        np.linalg.norm(((item[2][:2] + item[2][2:4]) / 2.0) - prev_center)
                    ),
                )
        selected[frame] = best_entry
        source_track[frame] = best_tid
        score[frame] = float(best_box[4]) if best_box.size >= 5 else 0.0
        prev_center = (best_box[:2] + best_box[2:4]) / 2.0
    return selected, source_track, score


def extract_keypoints_from_track_info(
    track_info: dict[int, list[dict[str, Any]]],
    T: int,
    seq_name: str,
    image_root: Path,
    vitpose_config: Path,
    vitpose_checkpoint: Path,
    device: str,
) -> dict[str, dict[int, dict[str, Any]]]:
    grouped = group_tracks_by_handedness(track_info)
    tracks: dict[int, dict[str, Any]] = {}
    for handedness, tids in grouped.items():
        keypoints = np.zeros((T, OP_NUM_JOINTS, 3), dtype=np.float32)
        valid = np.zeros(T, dtype=bool)
        entries_by_track = {tid: track_info[tid] for tid in tids}
        selected, source_track, score = select_candidates_by_frame(entries_by_track, T)
        for frame, entry in enumerate(selected):
            if entry is None:
                continue
            image_path = image_root / f"{frame:06d}.jpg"
            if not image_path.is_file():
                image_path = image_root / f"{frame:06d}.png"
            keypoints[frame] = run_vitpose_for_box(
                image_path,
                np.asarray(entry["det_box"], dtype=np.float32),
                vitpose_config,
                vitpose_checkpoint,
                device,
                handedness=handedness,
            )
            valid[frame] = not np.all(keypoints[frame] == 0)
        keypoints = interpolate_keypoints(keypoints, valid)
        tracks[handedness] = {
            "is_right": handedness,
            "source_track_ids": np.asarray(tids, dtype=np.int64),
            "frames": np.arange(T, dtype=np.int64),
            "keypoints": keypoints,
            "valid": valid,
            "source_track_per_frame": source_track,
            "score": score,
        }
    return {"tracks": tracks}


def load_keypoints_npy(path: Path, T: int) -> dict[int, dict[str, Any]]:
    raw = np.load(path, allow_pickle=True).item()
    tracks = raw["tracks"] if "tracks" in raw else raw
    out: dict[int, dict[str, Any]] = {}
    for key, value in tracks.items():
        handedness = int(key)
        kpts = np.asarray(value["keypoints"], dtype=np.float32)
        if kpts.shape != (T, OP_NUM_JOINTS, 3):
            raise ValueError(f"keypoints[{key}] shape {kpts.shape} does not match {(T, OP_NUM_JOINTS, 3)}")
        out[handedness] = {
            "is_right": int(value.get("is_right", handedness)),
            "source_track_ids": np.asarray(value.get("source_track_ids", []), dtype=np.int64),
            "frames": np.asarray(value.get("frames", np.arange(T)), dtype=np.int64),
            "keypoints": kpts,
            "valid": np.asarray(value.get("valid", ~np.all(kpts == 0, axis=(1, 2))), dtype=bool),
            "source_track_per_frame": np.asarray(
                value.get("source_track_per_frame", np.full(T, -1)), dtype=np.int64
            ),
            "score": np.asarray(value.get("score", np.zeros(T)), dtype=np.float32),
        }
    return out


def save_keypoints_npy(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, payload)


def _to_pose3d_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.as_tensor(value)


def _normalize_hand_value(key: str, value: Any) -> Any:
    if key == "relative_motion":
        if not isinstance(value, dict):
            raise TypeError(f"relative_motion must be a dict, got {type(value)}")
        return {k: _to_pose3d_tensor(v) for k, v in value.items()}
    if key == "mano_params":
        if not isinstance(value, dict):
            raise TypeError(f"mano_params must be a dict, got {type(value)}")
        return {k: np.asarray(v, dtype=np.float32) for k, v in value.items()}
    if key in {"pred_valid", "detection_failed"}:
        return np.asarray(value, dtype=bool)
    return value


def _normalize_slam_value(key: str, value: Any) -> Any:
    if key in {"slam_n_chunks", "max_slam_frames", "slam_overlap_frames"}:
        return int(value)
    if isinstance(value, np.generic):
        return np.asarray(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, (float, int, bool)) and key in {"img_focal", "scale"}:
        return np.asarray(value, dtype=np.float32)
    return value


def _normalize_pose3d_on_load(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore pose3d_hand layout after load: ndarray fields + Tensor relative_motion."""
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"left_hand", "right_hand"}:
            if not isinstance(value, dict):
                raise TypeError(f"{key} must be a dict, got {type(value)}")
            out[key] = {k: _normalize_hand_value(k, v) for k, v in value.items()}
        elif key == "slam_data":
            if not isinstance(value, dict):
                raise TypeError("slam_data must be a dict")
            out[key] = {k: _normalize_slam_value(k, v) for k, v in value.items()}
        elif key == "fps":
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _coerce_hand_value(key: str, value: Any) -> Any:
    if key == "relative_motion":
        if not isinstance(value, dict):
            raise TypeError(f"relative_motion must be a dict, got {type(value)}")
        return {k: _to_pose3d_tensor(v) for k, v in value.items()}
    if key == "mano_params":
        if not isinstance(value, dict):
            raise TypeError(f"mano_params must be a dict, got {type(value)}")
        return {k: np.asarray(v, dtype=np.float32) for k, v in value.items()}
    if key in {"pred_valid", "detection_failed"}:
        return np.asarray(value, dtype=bool)
    return value


def _coerce_slam_value(key: str, value: Any) -> Any:
    if key in {"slam_n_chunks", "max_slam_frames", "slam_overlap_frames"}:
        return int(value)
    return np.asarray(value)


def _coerce_pose3d_on_save(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure exported pose3d_hand matches the canonical on-disk layout."""
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"left_hand", "right_hand"}:
            out[key] = {k: _coerce_hand_value(k, v) for k, v in value.items()}
        elif key == "slam_data":
            out[key] = {k: _coerce_slam_value(k, v) for k, v in value.items()}
        elif key == "fps":
            out[key] = float(value)
        else:
            out[key] = value
    return out


def load_pose3d_payload(pose_path: Path) -> dict[str, Any]:
    payload = torch.load(pose_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {pose_path}, got {type(payload)}")
    return _normalize_pose3d_on_load(payload)


def init_body_pose_from_pose_path(pose_path: Path) -> torch.Tensor:
    """Load left/right hand_pose as (n_tracks, T, 15, 3) axis-angle."""
    payload = load_pose3d_payload(pose_path)
    poses = []
    for side in ("left_hand", "right_hand"):
        hp = np.asarray(payload[side]["mano_params"]["hand_pose"], dtype=np.float32)
        if hp.ndim == 2:
            hp = hp.reshape(hp.shape[0], 15, 3)
        elif hp.ndim == 3 and hp.shape[-1] != 3:
            hp = hp.reshape(hp.shape[0], 15, 3)
        poses.append(torch.from_numpy(hp))
    return torch.stack(poses, dim=0)


def init_latent_pose_from_body_pose(
    body_pose: torch.Tensor,
    pose_prior: Any | None,
    hand_mean: torch.Tensor,
) -> torch.Tensor:
    """Encode body pose to latent space; identity reshape when pose_prior is None."""
    if pose_prior is None:
        return body_pose.reshape(body_pose.shape[0], body_pose.shape[1], -1)
    b, t = body_pose.shape[:2]
    flat = body_pose.reshape(b * t, -1) - hand_mean.reshape(1, -1)
    with torch.no_grad():
        latent = pose_prior.encode(flat).mean.reshape(b, t, pose_prior.latentD)
    return latent


def init_latent_pose_from_pose_path(
    pose_path: Path,
    pose_prior: Any | None = None,
    hand_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    body_pose = init_body_pose_from_pose_path(pose_path)
    if pose_prior is None:
        return body_pose.reshape(body_pose.shape[0], body_pose.shape[1], -1)
    if hand_mean is None:
        raise ValueError("hand_mean is required when encoding init_latent_pose with VPoser")
    return init_latent_pose_from_body_pose(body_pose, pose_prior, hand_mean)


def load_pose_prior(cfg: DictConfig, device: torch.device) -> Any | None:
    if not bool(cfg.model.get("use_vposer", False)):
        return None
    from util.loaders import load_vposer

    expr_dir = resolve_path(cfg.paths.vposer)
    Logger.log(f"Loading VPoser pose prior from {expr_dir}")
    pose_prior, _ = load_vposer(str(expr_dir), vp_model="snapshot")
    return pose_prior.to(device).eval()


def build_frozen_init_latent_pose(
    cfg: DictConfig,
    device: torch.device,
    original_pose_path: Path,
    pose_prior: Any | None,
) -> torch.Tensor:
    body_pose = init_body_pose_from_pose_path(original_pose_path)
    if pose_prior is None:
        return body_pose.reshape(body_pose.shape[0], body_pose.shape[1], -1)
    b, t = body_pose.shape[:2]
    hand_model = make_body_model(cfg, b * t, device)
    return init_latent_pose_from_body_pose(body_pose, pose_prior, hand_model.hand_mean)


@dataclass
class Pose3DHandStageData:
    pose_path: Path
    track_info_path: Path
    keypoints_npy: Path
    image_root: Path | None
    seq_name: str
    extract_keypoints: bool = False
    vitpose_config: Path | None = None
    vitpose_checkpoint: Path | None = None
    vitpose_device: str = "cuda:0"
    frozen_init_latent_pose: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.payload = load_pose3d_payload(self.pose_path)
        self.track_info = load_track_info_npy(self.track_info_path)
        self.T = len(self.payload["left_hand"]["pred_valid"])
        self.seq_interval = (0, self.T)
        self.track_ids = [0, 1]
        self.keypoints_payload = self._load_or_create_keypoints()
        self._obs_cache: dict[str, Any] | None = None

    @property
    def n_tracks(self) -> int:
        return len(self.track_ids)

    def _load_or_create_keypoints(self) -> dict[int, dict[str, Any]]:
        if self.keypoints_npy.is_file():
            return load_keypoints_npy(self.keypoints_npy, self.T)
        if not self.extract_keypoints:
            raise FileNotFoundError(
                f"keypoints_npy not found: {self.keypoints_npy}. Set data.extract_keypoints=true to generate it."
            )
        if self.image_root is None or self.vitpose_config is None or self.vitpose_checkpoint is None:
            raise ValueError("image_root, vitpose_config, and vitpose_checkpoint are required to extract keypoints")
        payload = extract_keypoints_from_track_info(
            self.track_info,
            self.T,
            self.seq_name,
            self.image_root,
            self.vitpose_config,
            self.vitpose_checkpoint,
            self.vitpose_device,
        )
        save_keypoints_npy(self.keypoints_npy, payload)
        return load_keypoints_npy(self.keypoints_npy, self.T)

    def _side_for_track(self, track_id: int) -> str:
        return "right_hand" if track_id == 1 else "left_hand"

    def _vis_mask_for_track(self, track_id: int, side: str) -> np.ndarray:
        keypoint_valid = self.keypoints_payload.get(track_id, {}).get("valid", np.zeros(self.T, dtype=bool))
        pred_valid = np.asarray(self.payload[side]["pred_valid"], dtype=bool)
        return np.asarray(keypoint_valid, dtype=bool) | pred_valid

    def _load_keypoints(self, track_id: int) -> np.ndarray:
        if track_id not in self.keypoints_payload:
            return np.zeros((self.T, OP_NUM_JOINTS, 3), dtype=np.float32)
        keypoints = np.asarray(self.keypoints_payload[track_id]["keypoints"], dtype=np.float32)
        valid = np.asarray(self.keypoints_payload[track_id]["valid"], dtype=bool)
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

            obs["joints2d"].append(self._load_keypoints(track_id))
            obs["vis_mask"].append(ternary)
            obs["is_right"].append(np.full(self.T, is_right, dtype=np.float32))
            obs["track_id"].append(track_id)
            obs["seq_interval"].append(np.array(self.seq_interval, dtype=np.int32))
            obs["track_interval"].append(np.array([track_s, track_e], dtype=np.int32))
            obs["init_body_pose"].append(np.asarray(mano["hand_pose"], dtype=np.float32).reshape(self.T, 15, 3))
            obs["init_body_shape"].append(np.asarray(mano["betas"], dtype=np.float32))
            # MANO global_orient/transl are world-frame (^w phi, ^w tau).
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
        if self.frozen_init_latent_pose is not None:
            tensor_obs["init_latent_pose"] = self.frozen_init_latent_pose
        else:
            tensor_obs["init_latent_pose"] = tensor_obs["init_body_pose"].reshape(
                self.n_tracks, self.T, 45
            )
        self._obs_cache = tensor_obs
        return tensor_obs

    def camera_data(self) -> dict[str, torch.Tensor | bool | float]:
        slam = self.payload["slam_data"]
        cam_R, cam_t = slam_traj_to_camera_unscaled(np.asarray(slam["traj"]))
        focal = _as_float(slam["img_focal"])
        center = np.asarray(slam["img_center"], dtype=np.float32).reshape(2)
        intrins = np.tile(np.array([focal, focal, center[0], center[1]], dtype=np.float32), (self.T, 1))
        return {
            "cam_R": torch.from_numpy(cam_R),
            "cam_t": torch.from_numpy(cam_t),
            "intrins": torch.from_numpy(intrins),
            "slam_scale": _slam_scale_value(slam),
            "static": False,
        }

    def initial_params(self, model: Any) -> dict[str, torch.Tensor]:
        obs = self.obs_data()
        init_pose = obs["init_body_pose"]
        betas = torch.mean(obs["init_body_shape"], dim=1)
        latent_pose = model.pose2latent(init_pose)
        if model.pose_prior is None:
            latent_pose = latent_pose.reshape(self.n_tracks, self.T, -1)
        return {
            "init_body_pose": init_pose,
            "latent_pose": latent_pose,
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
    pose_prior: Any | None = None,
) -> Any:
    from optim.base_scene import BaseSceneModel

    model = BaseSceneModel(
        stage_data.n_tracks,
        stage_data.T,
        body_model,
        pose_prior=pose_prior,
        use_init=False,
        opt_cams=bool(cfg.model.opt_cams),
        opt_scale=bool(cfg.model.opt_scale),
    )
    cam_payload = stage_data.camera_data()
    slam_scale = float(cam_payload.pop("slam_scale"))
    world_scale = torch.tensor([[slam_scale]], dtype=torch.float32, device=device)
    model.params.set_cameras(
        move_to(cam_payload, device),
        opt_scale=bool(cfg.model.opt_scale),
        opt_cams=bool(cfg.model.opt_cams),
        opt_focal=bool(cfg.model.opt_cams),
        world_scale=world_scale,
    )
    for name, value in move_to(stage_data.initial_params(model), device).items():
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
    torch.save(_coerce_pose3d_on_save(payload), output)
    return output


def _model_world_scale(model: Any) -> float:
    world_scale = model.params.world_scale
    if isinstance(world_scale, torch.Tensor):
        return float(world_scale.detach().cpu().reshape(-1)[0])
    return float(world_scale)


def _export_slam_data(
    slam: dict[str, Any],
    *,
    scale: float | None = None,
    cam_R: np.ndarray | None = None,
    cam_t: np.ndarray | None = None,
) -> dict[str, Any]:
    exported = dict(slam)
    if cam_R is not None and cam_t is not None:
        exported["traj"] = camera_to_slam_traj_unscaled(cam_R, cam_t)
    elif "traj" in exported:
        exported["traj"] = np.asarray(exported["traj"], dtype=np.float32)
    if scale is not None:
        exported["scale"] = np.array(scale, dtype=np.float64)
    elif "scale" in exported:
        exported["scale"] = np.asarray(exported["scale"], dtype=np.float64)
    return exported


def latest_slam_data_for_prior(cfg: DictConfig, fallback: dict[str, Any]) -> dict[str, Any]:
    smooth_dir = resolve_path(cfg.data.work_dir) / STAGE_DIR["smooth"]
    try:
        smooth_path = latest_pose3d(smooth_dir)
        smooth_payload = load_pose3d_payload(smooth_path)
        return dict(smooth_payload["slam_data"])
    except FileNotFoundError:
        return dict(fallback)


def payload_from_model(base_payload: dict[str, Any], model: Any) -> dict[str, Any]:
    with torch.no_grad():
        params = model.params.get_dict()
        body_pose = model.latent2pose(model.params.latent_pose).detach().cpu().numpy()
        cam_R = model.params._cam_R.detach().cpu().numpy()
        cam_t = model.params._cam_t.detach().cpu().numpy()
    trans = params["trans"].detach().cpu().numpy()
    root_orient = params["root_orient"].detach().cpu().numpy()
    betas = params["betas"].detach().cpu().numpy()
    is_right = params["is_right"].detach().cpu().numpy()
    scale = _model_world_scale(model)

    payload = {
        "left_hand": base_payload["left_hand"],
        "right_hand": base_payload["right_hand"],
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
    payload["slam_data"] = _export_slam_data(
        base_payload["slam_data"],
        scale=scale,
        cam_R=cam_R,
        cam_t=cam_t,
    )
    return payload


def res_dict_from_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    root_orient = []
    pose_body = []
    trans = []
    betas = []
    is_right = []
    for handed in (0, 1):
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
    scale = _slam_scale_value(slam)
    cam_R, cam_t = slam_traj_to_camera_scaled(np.asarray(slam["traj"]), scale)
    focal = _as_float(slam["img_focal"])
    center = np.asarray(slam["img_center"], dtype=np.float32).reshape(2)
    intrins = np.array([focal, focal, center[0], center[1]], dtype=np.float32)
    B = 2
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


def payload_from_prior_result(
    base_payload: dict[str, Any],
    result: dict[str, np.ndarray],
    slam_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_slam = dict(slam_data if slam_data is not None else base_payload["slam_data"])
    payload = {
        "left_hand": base_payload["left_hand"],
        "right_hand": base_payload["right_hand"],
        "fps": float(base_payload.get("fps", 30.0)),
    }
    for b, handed in enumerate((0, 1)):
        side = "right_hand" if handed == 1 else "left_hand"
        pred_valid = np.asarray(base_payload[side]["pred_valid"], dtype=bool)
        payload[side] = hand_slot_from_arrays(
            result["root_orient"][b],
            result["pose_body"][b],
            result["betas"][b],
            result["trans"][b],
            pred_valid,
        )
    payload["slam_data"] = _export_slam_data(source_slam, scale=_slam_scale_value(source_slam))
    return payload


def latest_pose3d(stage_dir: Path) -> Path:
    matches = sorted(stage_dir.glob("*.pose3d_hand"))
    if not matches:
        raise FileNotFoundError(f"No .pose3d_hand found under {stage_dir}")
    return matches[-1]


def stage_output_path(cfg: DictConfig, stage: str) -> Path:
    if stage == "prior":
        return resolve_path(cfg.data.work_dir) / STAGE_DIR["prior"] / f"{cfg.data.seq}_000000.pose3d_hand"
    num_iters = int(cfg.optim.root.num_iters if stage == "root" else cfg.optim.smooth.num_iters)
    return resolve_path(cfg.data.work_dir) / STAGE_DIR[stage] / f"{cfg.data.seq}_{num_iters:06d}.pose3d_hand"


def try_latest_stage_output(cfg: DictConfig, stage: str) -> Path | None:
    stage_dir = resolve_path(cfg.data.work_dir) / STAGE_DIR[stage]
    try:
        return latest_pose3d(stage_dir)
    except FileNotFoundError:
        return None


def save_stage_loss_plots(optimizer: Any, stage_dir: Path, cfg: DictConfig) -> None:
    if not bool(cfg.data.save_loss_plots):
        return
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        optimizer.plot_losses(str(stage_dir))
        Logger.log(f"Saved loss plots to {stage_dir}")
    except Exception as exc:
        Logger.log(f"WARNING: failed to save loss plots to {stage_dir}: {exc}")


def stage_input(cfg: DictConfig, stage: str) -> Path:
    if stage == "root" or not bool(cfg.data.resume):
        return resolve_path(cfg.data.pose3d_hand)
    previous = STAGE_ORDER[STAGE_ORDER.index(stage) - 1]
    return latest_pose3d(resolve_path(cfg.data.work_dir) / STAGE_DIR[previous])


def make_stage_data(
    cfg: DictConfig,
    stage: str,
    frozen_init_latent_pose: torch.Tensor | None = None,
) -> Pose3DHandStageData:
    vitpose_device = cfg.data.get("device_override", None)
    if vitpose_device is None:
        vitpose_device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    return Pose3DHandStageData(
        pose_path=stage_input(cfg, stage),
        track_info_path=resolve_path(cfg.data.track_info),
        keypoints_npy=resolve_path(cfg.data.keypoints_npy),
        image_root=resolve_path(cfg.data.image_root),
        seq_name=str(cfg.data.seq),
        extract_keypoints=bool(cfg.data.extract_keypoints),
        vitpose_config=resolve_path(cfg.data.vitpose_config),
        vitpose_checkpoint=resolve_path(cfg.data.vitpose_checkpoint),
        vitpose_device=str(vitpose_device),
        frozen_init_latent_pose=frozen_init_latent_pose,
    )


def prepare_cfg(cfg: DictConfig) -> DictConfig:
    cfg.model.opt_cams = False
    cfg.run_vis = False
    cfg.run_opt = False
    cfg.paths.base_dir = str(_REPO_ROOT.resolve())
    cfg = resolve_cfg_paths(cfg)
    return cfg
<<<<<<< HEAD


def snapshot_work_dir_config(cfg: DictConfig, work_dir: Path) -> None:
    if not bool(cfg.get("snapshot_config", True)):
        return
    hydra_dir = work_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")
    for name in ("overrides.yaml", "hydra.yaml"):
        src = Path.cwd() / ".hydra" / name
        if src.is_file():
            shutil.copy2(src, hydra_dir / name)
    Logger.log(f"Saved config snapshot to {hydra_dir}")


def collect_prior_loss_plots(prior_dir: Path, cfg: DictConfig) -> None:
    if not bool(cfg.data.save_loss_plots):
        return
    summary_candidates = sorted(prior_dir.rglob("all_stages_loss.jpg"))
    if summary_candidates:
        dst = prior_dir / "all_stages_loss.jpg"
        src = summary_candidates[-1]
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        Logger.log(f"Collected prior loss summary to {dst}")
    else:
        Logger.log(f"WARNING: no all_stages_loss.jpg found under {prior_dir}")

    for src in sorted(prior_dir.rglob("stage_*_loss.jpg")):
        if src.parent.resolve() == prior_dir.resolve():
            continue
        rel = src.relative_to(prior_dir)
        dst = prior_dir / "_".join(rel.parts)
        if dst.resolve() == src.resolve() or dst.is_file():
            continue
        shutil.copy2(src, dst)
        Logger.log(f"Collected prior stage loss plot to {dst}")


def has_image_frames(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.jpg")) or path.is_dir() and any(path.glob("*.png"))


def read_video_fps(path: Path) -> float:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV is required to read video FPS when frame_opts.fps=auto") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Could not read a valid FPS from {path}")
    return fps


def ensure_keypoint_frames(cfg: DictConfig) -> None:
    if not bool(cfg.data.get("extract_keypoints", False)):
        return
    keypoints_path = resolve_path(cfg.data.keypoints_npy)
    if keypoints_path.is_file() and bool(cfg.data.get("reuse_keypoints_npy", True)):
        return

    image_root = resolve_path(cfg.data.image_root)
    if image_root is None:
        raise ValueError("data.image_root is required when extracting keypoints")
    if has_image_frames(image_root):
        return

    src_path = resolve_path(cfg.data.get("src_path", None))
    if src_path is None or not src_path.is_file():
        raise FileNotFoundError(f"Cannot extract frames; data.src_path is not a file: {src_path}")

    frame_opts = OmegaConf.to_container(cfg.data.get("frame_opts", {}), resolve=True)
    frame_opts = dict(frame_opts) if frame_opts is not None else {}
    fps = frame_opts.get("fps", 30)
    if fps is None or str(fps).lower() == "auto":
        fps = read_video_fps(src_path)
        frame_opts["fps"] = fps

    image_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting keypoint frames from {src_path} to {image_root} at {fps} fps")
    out = video_to_frames(src_path, image_root, **frame_opts)
    if out != 0:
        raise RuntimeError(f"Failed to extract frames from {src_path} to {image_root}")
=======
>>>>>>> af015a6 (Refactor pose3d_hand processing and enhance camera trajectory functions)


def snapshot_work_dir_config(cfg: DictConfig, work_dir: Path) -> None:
    if not bool(cfg.get("snapshot_config", True)):
        return
    hydra_dir = work_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")
    for name in ("overrides.yaml", "hydra.yaml"):
        src = Path.cwd() / ".hydra" / name
        if src.is_file():
            shutil.copy2(src, hydra_dir / name)
    Logger.log(f"Saved config snapshot to {hydra_dir}")


def collect_prior_loss_plots(prior_dir: Path, cfg: DictConfig) -> None:
    if not bool(cfg.data.save_loss_plots):
        return
    summary_candidates = sorted(prior_dir.rglob("all_stages_loss.jpg"))
    if summary_candidates:
        dst = prior_dir / "all_stages_loss.jpg"
        src = summary_candidates[-1]
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        Logger.log(f"Collected prior loss summary to {dst}")
    else:
        Logger.log(f"WARNING: no all_stages_loss.jpg found under {prior_dir}")

    for src in sorted(prior_dir.rglob("stage_*_loss.jpg")):
        if src.parent.resolve() == prior_dir.resolve():
            continue
        rel = src.relative_to(prior_dir)
        dst = prior_dir / "_".join(rel.parts)
        if dst.resolve() == src.resolve() or dst.is_file():
            continue
        shutil.copy2(src, dst)
        Logger.log(f"Collected prior stage loss plot to {dst}")


def has_image_frames(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.jpg")) or path.is_dir() and any(path.glob("*.png"))


def read_video_fps(path: Path) -> float:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV is required to read video FPS when frame_opts.fps=auto") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Could not read a valid FPS from {path}")
    return fps


def ensure_keypoint_frames(cfg: DictConfig) -> None:
    if not bool(cfg.data.get("extract_keypoints", False)):
        return
    keypoints_path = resolve_path(cfg.data.keypoints_npy)
    if keypoints_path.is_file() and bool(cfg.data.get("reuse_keypoints_npy", True)):
        return

    image_root = resolve_path(cfg.data.image_root)
    if image_root is None:
        raise ValueError("data.image_root is required when extracting keypoints")
    if has_image_frames(image_root):
        return

    src_path = resolve_path(cfg.data.get("src_path", None))
    if src_path is None or not src_path.is_file():
        raise FileNotFoundError(f"Cannot extract frames; data.src_path is not a file: {src_path}")

    frame_opts = OmegaConf.to_container(cfg.data.get("frame_opts", {}), resolve=True)
    frame_opts = dict(frame_opts) if frame_opts is not None else {}
    fps = frame_opts.get("fps", 30)
    if fps is None or str(fps).lower() == "auto":
        fps = read_video_fps(src_path)
        frame_opts["fps"] = fps

    image_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting keypoint frames from {src_path} to {image_root} at {fps} fps")
    out = video_to_frames(src_path, image_root, **frame_opts)
    if out != 0:
        raise RuntimeError(f"Failed to extract frames from {src_path} to {image_root}")


def make_body_model(cfg: DictConfig, batch_size: int, device: torch.device) -> MANO:
    mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
    return MANO(batch_size=batch_size, pose2rot=True, **mano_cfg).to(device)


def stage_loss_weights(cfg: DictConfig) -> list[dict[str, float]]:
    weights = cfg.optim.loss_weights
    return [{k: weights[k][i] for k in weights.keys()} for i in range(3)]


def run_root_or_smooth(
    cfg: DictConfig,
    stage: str,
    device: torch.device,
    frozen_init_latent_pose: torch.Tensor | None = None,
    pose_prior: Any | None = None,
) -> Path:
    from optim.optimizers import RootOptimizer, SmoothOptimizer

    data = make_stage_data(cfg, stage, frozen_init_latent_pose=frozen_init_latent_pose)
    num_iters = int(cfg.optim.root.num_iters if stage == "root" else cfg.optim.smooth.num_iters)
    output = stage_output_path(cfg, stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    if num_iters == 0:
        Logger.log(f"Skipping {stage} optimization because num_iters=0")
        return save_pose3d_payload(data.payload, output)

    obs_data = move_to(data.obs_data(), device)
    body_model = make_body_model(cfg, data.n_tracks * data.T, device)
    model = make_model(cfg, body_model, data, device, pose_prior=pose_prior)
    all_loss_weights = stage_loss_weights(cfg)
    opt_kwargs = dict(cfg.optim.options)
    optimizer_cls = RootOptimizer if stage == "root" else SmoothOptimizer
    optimizer = optimizer_cls(model, all_loss_weights, **opt_kwargs)
    Logger.log(f"Running {stage} for {num_iters} iterations")
    for i in range(num_iters):
        optimizer.cur_step = i
        optimizer.loss.cur_step = i
        optimizer.optim_step(obs_data, i)
    optimizer.cur_step = num_iters
    payload = payload_from_model(data.payload, model)
    save_pose3d_payload(payload, output)
    save_stage_loss_plots(optimizer, output.parent, cfg)
    return output


def run_prior_stage(
    cfg: DictConfig,
    device: torch.device,
    frozen_init_latent_pose: torch.Tensor | None = None,
) -> Path:
    from HMP.fitting import fitting_prior

    data = make_stage_data(cfg, "prior", frozen_init_latent_pose=frozen_init_latent_pose)
    obs_data = move_to(data.obs_data(), device)
    body_model = make_body_model(cfg, data.n_tracks * data.T, device)
    prior_dir = resolve_path(cfg.data.work_dir) / STAGE_DIR["prior"]
    prior_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.base_dir = str(_REPO_ROOT.resolve())
    cfg.HMP.exp_name = str(cfg.data.seq)
    result = fitting_prior(
        obs_data,
        (move_to(res_dict_from_payload(data.payload), device),),
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
        npz_path = prior_dir / f"{cfg.data.seq}_000000_world_results.npz"
        with np.load(npz_path, allow_pickle=False) as npz:
            result_dict = {k: npz[k] for k in npz.files}
    result_np = {k: _as_numpy(v) for k, v in result_dict.items()}
    slam_data = latest_slam_data_for_prior(cfg, data.payload["slam_data"])
    payload = payload_from_prior_result(data.payload, result_np, slam_data=slam_data)
    output = stage_output_path(cfg, "prior")
    save_pose3d_payload(payload, output)
    collect_prior_loss_plots(prior_dir, cfg)
    return output


def copy_final(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def select_device(cfg: DictConfig) -> torch.device:
    override = cfg.data.get("device_override", None)
    if override is not None:
        device_name = str(override)
    else:
        device_name = f"cuda:{cfg.gpu}"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    return torch.device(device_name)


def run_from_cfg(cfg: DictConfig) -> Path:
    cfg = prepare_cfg(cfg)
    ensure_keypoint_frames(cfg)
    work_dir = resolve_path(cfg.data.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    snapshot_work_dir_config(cfg, work_dir)
    Logger.init(str(work_dir / "pose3d_hand_stages.log"))
    device = select_device(cfg)
    pose_prior = load_pose_prior(cfg, device)
    original_pose_path = resolve_path(cfg.data.pose3d_hand)
    frozen_init_latent_pose = build_frozen_init_latent_pose(
        cfg, device, original_pose_path, pose_prior
    )
    stages = STAGE_ORDER if str(cfg.data.stage) == "all" else (str(cfg.data.stage),)
    keep_resume = bool(cfg.data.resume)
    last_output = original_pose_path
    for stage in stages:
        existing = try_latest_stage_output(cfg, stage) if keep_resume else None
        if existing is not None:
            Logger.log(f"resume: skip {stage}, use {existing}")
            last_output = existing
        elif stage in {"root", "smooth"}:
            last_output = run_root_or_smooth(
                cfg,
                stage,
                device,
                frozen_init_latent_pose=frozen_init_latent_pose,
                pose_prior=pose_prior,
            )
        elif stage == "prior":
            last_output = run_prior_stage(
                cfg, device, frozen_init_latent_pose=frozen_init_latent_pose
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
        cfg.data.pose3d_hand = str(last_output)
        if not keep_resume:
            cfg.data.resume = False
    final_output = copy_final(last_output, resolve_path(cfg.data.output))
    print(f"Saved final pose3d_hand to {final_output}")
    return final_output


@hydra.main(version_base=None, config_path="confs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    run_from_cfg(cfg)


if __name__ == "__main__":
    main()
