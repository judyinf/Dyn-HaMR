#!/usr/bin/env python3
"""Project .pose3d_hand MANO joints to video-space 2D keypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

_SCRIPT_DIR = Path(__file__).resolve().parent
_DYN_HAMR_ROOT = _SCRIPT_DIR.parent
if str(_DYN_HAMR_ROOT) not in sys.path:
    sys.path.insert(0, str(_DYN_HAMR_ROOT))

from body_model import MANO


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def load_pose3d_hand(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def video_info(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frame_count, width, height, fps


def make_mano_model(model_path: Path, batch_size: int, device: torch.device) -> MANO:
    return MANO(
        model_path=str(model_path),
        gender="neutral",
        num_hand_joints=15,
        mean_params=str(model_path.parent / "mano_mean_params.npz"),
        create_body_pose=False,
        batch_size=batch_size,
        pose2rot=True,
    ).to(device)


def traj_to_camera(traj: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert supported pose3d_hand traj variants to world-to-camera extrinsics."""
    traj = np.asarray(traj, dtype=np.float32).reshape(-1, 7)
    t = traj[:, :3]
    R = Rotation.from_quat(traj[:, 3:7]).as_matrix().astype(np.float32)
    if convention == "center-rw2c":
        R_w2c = R
        t_w2c = -np.einsum("tij,tj->ti", R_w2c, t).astype(np.float32)
    elif convention == "c2w":
        R_w2c = np.swapaxes(R, -1, -2)
        t_w2c = -np.einsum("tij,tj->ti", R_w2c, t).astype(np.float32)
    elif convention == "w2c":
        R_w2c = R
        t_w2c = t.astype(np.float32)
    else:
        raise ValueError(f"Unsupported traj convention: {convention}")
    return R_w2c, t_w2c


def choose_traj_convention(payload: dict, traj: np.ndarray) -> str:
    candidates = ("center-rw2c", "c2w", "w2c")
    best_name = candidates[0]
    best_score = (-1.0, -np.inf)
    for name in candidates:
        R_w2c, t_w2c = traj_to_camera(traj, name)
        positive = []
        medians = []
        for side in ("left_hand", "right_hand"):
            hand = payload[side]
            valid = np.asarray(hand["pred_valid"], dtype=bool)
            transl = np.asarray(hand["mano_params"]["transl"], dtype=np.float32)
            if not valid.any():
                continue
            root_cam = np.einsum("tij,tj->ti", R_w2c, transl) + t_w2c
            z = root_cam[valid, 2]
            positive.append(float(np.mean(z > 0)))
            medians.append(float(np.median(z)))
        score = (float(np.mean(positive)) if positive else 0.0, float(np.mean(medians)) if medians else -np.inf)
        if score > best_score:
            best_name = name
            best_score = score
    return best_name


def project_points(
    joints3d: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    focal: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    joints_cam = np.einsum("tij,tkj->tki", R_w2c, joints3d) + t_w2c[:, None, :]
    z = joints_cam[..., 2]
    denom = np.where(np.abs(z[..., None]) < 1e-8, np.nan, z[..., None])
    xy = joints_cam[..., :2] / denom
    joints2d = focal[None, None, :] * xy + center[None, None, :]
    return joints2d.astype(np.float32), z.astype(np.float32)


def mano_joints_for_hand(
    hand_payload: dict,
    is_right: bool,
    model_path: Path,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    params = hand_payload["mano_params"]
    global_orient = np.asarray(params["global_orient"], dtype=np.float32)
    hand_pose = np.asarray(params["hand_pose"], dtype=np.float32)
    transl = np.asarray(params["transl"], dtype=np.float32)
    betas = np.asarray(params["betas"], dtype=np.float32)

    T = global_orient.shape[0]
    joints_chunks: list[np.ndarray] = []
    sign = 1.0 if is_right else -1.0

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        n = end - start
        model = make_mano_model(model_path, n, device)
        with torch.no_grad():
            out = model(
                global_orient=torch.from_numpy(global_orient[start:end]).to(device),
                hand_pose=torch.from_numpy(hand_pose[start:end]).to(device),
                betas=torch.from_numpy(betas[start:end]).to(device),
                transl=torch.from_numpy(transl[start:end]).to(device),
            )
            joints = out.joints.detach().cpu().numpy().astype(np.float32)
        joints[..., 0] *= sign
        joints_chunks.append(joints)

    return np.concatenate(joints_chunks, axis=0)


def export_keypoints(
    pose3d_hand: Path,
    video: Path,
    output: Path,
    mano_model_path: Path,
    device: torch.device,
    chunk_size: int,
    traj_convention: str,
) -> dict[str, np.ndarray]:
    payload = load_pose3d_hand(pose3d_hand)
    frame_count, width, height, video_fps = video_info(video)

    traj = np.asarray(payload["slam_data"]["traj"], dtype=np.float32)
    if traj.shape[0] != frame_count:
        raise ValueError(
            f"traj length {traj.shape[0]} does not match video frame count {frame_count}"
        )
    scale = float(np.asarray(payload["slam_data"].get("scale", 1.0), dtype=np.float64).reshape(-1)[0])
    traj_for_camera = traj.copy()
    traj_for_camera[:, :3] *= np.float32(scale)
    if traj_convention == "auto":
        traj_convention = choose_traj_convention(payload, traj_for_camera)
        print(f"Using traj convention: {traj_convention}")

    focal_scalar = float(np.asarray(payload["slam_data"]["img_focal"]).reshape(-1)[0])
    focal = np.array([focal_scalar, focal_scalar], dtype=np.float32)
    center = np.asarray(payload["slam_data"]["img_center"], dtype=np.float32).reshape(2)
    R_w2c, t_w2c = traj_to_camera(traj_for_camera, traj_convention)

    result: dict[str, np.ndarray] = {
        "cam_R_w2c": R_w2c,
        "cam_t_w2c": t_w2c,
        "img_focal": focal,
        "img_center": center,
        "video_size": np.array([width, height], dtype=np.int32),
        "video_fps": np.array(video_fps, dtype=np.float32),
        "traj_convention": np.array(traj_convention),
        "scale": np.array(scale, dtype=np.float64),
        "traj_translation_scaled": np.array(True),
    }

    for side, is_right in (("left_hand", False), ("right_hand", True)):
        hand = payload[side]
        joints3d = mano_joints_for_hand(hand, is_right, mano_model_path, device, chunk_size)
        joints2d, depth = project_points(joints3d, R_w2c, t_w2c, focal, center)
        prefix = "left" if side == "left_hand" else "right"
        result[f"{prefix}_joints3d"] = joints3d
        result[f"{prefix}_joints2d"] = joints2d
        result[f"{prefix}_depth"] = depth
        result[f"{prefix}_pred_valid"] = np.asarray(hand["pred_valid"], dtype=bool)
        result[f"{prefix}_detection_failed"] = np.asarray(hand["detection_failed"], dtype=bool)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return result


def draw_overlay(video: Path, output: Path, keypoints: dict[str, np.ndarray]) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open overlay writer: {output}")

    colors = {
        "left": (255, 80, 80),
        "right": (80, 220, 255),
    }
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for side in ("left", "right"):
            if not bool(keypoints[f"{side}_pred_valid"][t]):
                continue
            pts = keypoints[f"{side}_joints2d"][t]
            depth = keypoints[f"{side}_depth"][t]
            finite = np.isfinite(pts).all(axis=1) & (depth > 0)
            color = colors[side]
            for a, b in HAND_EDGES:
                if finite[a] and finite[b]:
                    pa = tuple(np.round(pts[a]).astype(int).tolist())
                    pb = tuple(np.round(pts[b]).astype(int).tolist())
                    cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                if finite[i]:
                    p = tuple(np.round(pt).astype(int).tolist())
                    cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)
        writer.write(frame)
        t += 1

    cap.release()
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project .pose3d_hand MANO joints to video 2D keypoints."
    )
    parser.add_argument("--pose3d-hand", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--overlay", type=Path, default=None, help="Optional overlay .mp4 path")
    parser.add_argument(
        "--mano-model-path",
        type=Path,
        default=Path("_DATA/data/mano"),
        help="MANO model directory",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--traj-convention",
        choices=("auto", "center-rw2c", "c2w", "w2c"),
        default="auto",
        help=(
            "Camera traj interpretation. center-rw2c means traj stores "
            "[camera center in world, quat_xyzw(R_w2c)]."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    keypoints = export_keypoints(
        args.pose3d_hand,
        args.video,
        args.output,
        args.mano_model_path,
        device,
        args.chunk_size,
        args.traj_convention,
    )
    print(f"Saved 2D keypoints to {args.output}")
    if args.overlay is not None:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        draw_overlay(args.video, args.overlay, keypoints)
        print(f"Saved overlay video to {args.overlay}")


if __name__ == "__main__":
    main()
