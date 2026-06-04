#!/usr/bin/env python3
"""Export Dyn-HaMR world_results.npz to .pose3d_hand (pose_estimation-compatible)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_DYN_HAMR_ROOT = _SCRIPT_DIR.parent
if str(_DYN_HAMR_ROOT) not in sys.path:
    sys.path.insert(0, str(_DYN_HAMR_ROOT))

from HMP.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion
from optim.output import load_track_info


def result_sort_key(path: Path):
    parts = path.name.split("_")
    if len(parts) < 3 or parts[-2] not in {"world", "prior"}:
        raise ValueError(f"Unexpected result filename: {path.name}")
    iteration = parts[-3]
    return (int(iteration) if iteration.isdigit() else -1, path.name)


def find_latest_result(log_dir: Path, phase: str) -> Path | None:
    phase_dir = log_dir / phase
    if not phase_dir.is_dir():
        return None
    results = sorted(phase_dir.glob("*_results.npz"), key=result_sort_key)
    return results[-1] if results else None


def load_world_results(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _as_batch_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[None]
    return arr


def _pred_valid_from_vis_mask(vis_mask: np.ndarray, seq_len: int) -> np.ndarray:
    mask = np.asarray(vis_mask, dtype=np.float32).reshape(-1)
    seq_start, seq_end = 0, seq_len
    if mask.shape[0] > seq_len:
        # track_info vis_mask covers the optimized subsequence interval
        mask = mask[:seq_len]
    elif mask.shape[0] < seq_len:
        padded = np.full(seq_len, -1, dtype=np.float32)
        padded[: mask.shape[0]] = mask
        mask = padded
    return (mask >= 0).astype(bool)


def _betas_per_frame(betas: np.ndarray, seq_len: int) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        return np.repeat(betas[None], seq_len, axis=0)
    if betas.ndim == 2 and betas.shape[0] == seq_len:
        return betas
    if betas.ndim == 2 and betas.shape[0] == 1:
        return np.repeat(betas, seq_len, axis=0)
    raise ValueError(f"Unsupported betas shape {betas.shape} for T={seq_len}")


def _compute_relative_motion(
    global_orient: torch.Tensor,
    transl: torch.Tensor,
    pred_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """global_orient (T,3), transl (T,3), pred_valid (T,) bool."""
    T = global_orient.shape[0]
    rel_trans = torch.zeros(T, 3, dtype=torch.float32)
    rel_rot_mat = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(T, 1, 1)
    rel_rot_aa = torch.zeros(T, 3, dtype=torch.float32)
    pair_valid = torch.zeros(T, dtype=torch.bool)

    rot_mats = axis_angle_to_matrix(global_orient)
    for t in range(1, T):
        if not (pred_valid[t] and pred_valid[t - 1]):
            continue
        pair_valid[t] = True
        rel_trans[t] = transl[t] - transl[t - 1]
        delta = rot_mats[t] @ rot_mats[t - 1].transpose(-1, -2)
        rel_rot_mat[t] = delta
        rel_rot_aa[t] = matrix_to_axis_angle(delta.unsqueeze(0)).squeeze(0)

    return {
        "rel_trans": rel_trans,
        "rel_rot_mat": rel_rot_mat,
        "rel_rot_aa": rel_rot_aa,
        "pair_valid": pair_valid,
    }


def _empty_hand(seq_len: int) -> dict:
    return {
        "mano_params": {
            "global_orient": torch.zeros(seq_len, 3, dtype=torch.float32),
            "hand_pose": torch.zeros(seq_len, 45, dtype=torch.float32),
            "betas": torch.zeros(seq_len, 10, dtype=torch.float32),
            "transl": torch.zeros(seq_len, 3, dtype=torch.float32),
        },
        "relative_motion": {
            "rel_rot_aa": torch.zeros(seq_len, 3, dtype=torch.float32),
            "rel_rot_mat": torch.eye(3, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(seq_len, 1, 1),
            "rel_trans": torch.zeros(seq_len, 3, dtype=torch.float32),
            "pair_valid": torch.zeros(seq_len, dtype=torch.bool),
        },
        "pred_valid": torch.zeros(seq_len, dtype=torch.bool),
    }


def build_hand_slot(
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    pred_valid: np.ndarray,
) -> dict:
    T = trans.shape[0]
    global_orient = torch.from_numpy(np.asarray(root_orient, dtype=np.float32)).reshape(T, 3)
    hand_pose = torch.from_numpy(np.asarray(pose_body, dtype=np.float32)).reshape(T, 45)
    transl = torch.from_numpy(np.asarray(trans, dtype=np.float32)).reshape(T, 3)
    betas_t = torch.from_numpy(_betas_per_frame(betas, T))
    pred_valid_t = torch.from_numpy(np.asarray(pred_valid, dtype=bool))

    return {
        "mano_params": {
            "global_orient": global_orient,
            "hand_pose": hand_pose,
            "betas": betas_t,
            "transl": transl,
        },
        "relative_motion": _compute_relative_motion(global_orient, transl, pred_valid_t),
        "pred_valid": pred_valid_t,
    }


def _intrinsics_from_npz(data: dict[str, np.ndarray], batch_idx: int = 0) -> tuple[float, np.ndarray]:
    intrins = data["intrins"]
    if intrins.ndim >= 2:
        row = intrins[batch_idx] if intrins.shape[0] > batch_idx else intrins[0]
    else:
        row = intrins
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size < 4:
        raise ValueError(f"intrins must have 4 elements, got shape {intrins.shape}")
    focal = float((row[0] + row[1]) / 2.0)
    center = row[2:4].astype(np.float64)
    return focal, center


def _camera_track_array(
    arr: np.ndarray,
    batch_idx: int,
    seq_len: int,
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    unbatched_shape = (seq_len, *trailing_shape)
    batched_shape = (-1, seq_len, *trailing_shape)
    if arr.shape == unbatched_shape:
        return arr
    try:
        return arr.reshape(batched_shape)[batch_idx]
    except ValueError as exc:
        raise ValueError(
            f"Expected camera array shape (T, {trailing_shape}) or "
            f"(B, T, {trailing_shape}), got {arr.shape}"
        ) from exc


def _camera_traj(data: dict[str, np.ndarray], batch_idx: int, seq_len: int) -> torch.Tensor:
    cam_t = torch.from_numpy(_camera_track_array(data["cam_t"], batch_idx, seq_len, (3,)))
    cam_R = torch.from_numpy(_camera_track_array(data["cam_R"], batch_idx, seq_len, (3, 3)))
    quat_wxyz = matrix_to_quaternion(cam_R)
    quat_xyzw = torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)
    return torch.cat([cam_t, quat_xyzw], dim=-1).to(dtype=torch.float32)


def _slam_tstamp(seq_len: int, max_keyframes: int) -> torch.Tensor:
    if seq_len <= 0:
        return torch.zeros(0, dtype=torch.int32)
    if seq_len <= max_keyframes:
        return torch.arange(seq_len, dtype=torch.int32)
    if max_keyframes <= 1:
        return torch.tensor([0], dtype=torch.int32)
    step = max(1, seq_len // (max_keyframes - 1))
    tstamp = torch.arange(0, seq_len, step, dtype=torch.int32)[:max_keyframes]
    if tstamp.numel() < max_keyframes:
        tail = torch.tensor([seq_len - 1], dtype=torch.int32)
        tstamp = torch.unique(torch.cat([tstamp, tail]), sorted=True)
    return tstamp[:max_keyframes]


def build_pose3d_hand(
    data: dict[str, np.ndarray],
    track_vis_masks: dict[int, np.ndarray] | None,
    track_ids: list[int] | None,
    fps: float,
    prefer_track_index: int | None,
    disp_size: int,
    max_slam_keyframes: int,
) -> dict:
    trans = _as_batch_array(data["trans"])
    B, seq_len = trans.shape[0], trans.shape[1]

    left_hand = _empty_hand(seq_len)
    right_hand = _empty_hand(seq_len)

    by_side: dict[str, list[int]] = {"left": [], "right": []}
    for b in range(B):
        is_right = bool(
            np.round(float(np.asarray(_as_batch_array(data["is_right"])[b]).reshape(-1)[0]))
        )
        by_side["right" if is_right else "left"].append(b)

    for side, indices in by_side.items():
        if len(indices) > 1 and prefer_track_index is None:
            raise ValueError(
                f"Multiple tracks map to {side} hand: batch indices {indices}. "
                "Pass --prefer-track-index to select one."
            )

    if prefer_track_index is not None:
        if prefer_track_index < 0 or prefer_track_index >= B:
            raise ValueError(f"--prefer-track-index {prefer_track_index} out of range [0, {B})")
        export_indices = [prefer_track_index]
    else:
        export_indices = []
        for indices in by_side.values():
            if indices:
                export_indices.append(indices[0])

    for b in export_indices:
        is_right = bool(
            np.round(float(np.asarray(_as_batch_array(data["is_right"])[b]).reshape(-1)[0]))
        )

        if track_vis_masks is not None and track_ids is not None:
            tid = int(track_ids[b]) if b < len(track_ids) else sorted(track_vis_masks)[b]
            vis = track_vis_masks.get(tid)
            if vis is None:
                raise KeyError(f"track_info has no vis_mask for track id {tid}")
            pred_valid = _pred_valid_from_vis_mask(vis, seq_len)
        else:
            pred_valid = np.ones(seq_len, dtype=bool)

        slot = build_hand_slot(
            _as_batch_array(data["root_orient"])[b],
            _as_batch_array(data["pose_body"])[b],
            _as_batch_array(data["trans"])[b],
            _as_batch_array(data["betas"])[b],
            pred_valid,
        )
        if is_right:
            right_hand = slot
        else:
            left_hand = slot

    traj_batch = export_indices[0] if export_indices else 0
    focal, center = _intrinsics_from_npz(data, traj_batch)
    slam_data = {
        "tstamp": _slam_tstamp(seq_len, max_slam_keyframes),
        "disps": torch.zeros(1, disp_size, disp_size, dtype=torch.float32),
        "traj": _camera_traj(data, traj_batch, seq_len),
        "img_focal": torch.tensor(focal, dtype=torch.float64),
        "img_center": torch.from_numpy(center),
        "scale": torch.tensor(1.0, dtype=torch.float64),
    }

    return {
        "fps": float(fps),
        "left_hand": left_hand,
        "right_hand": right_hand,
        "slam_data": slam_data,
    }


def export_pose3d_hand(
    log_dir: Path,
    phase: str,
    output: Path,
    fps: float,
    prefer_track_index: int | None,
    disp_size: int,
    max_slam_keyframes: int,
    result_path: Path | None = None,
) -> Path:
    npz_path = result_path or find_latest_result(log_dir, phase)
    if npz_path is None:
        raise FileNotFoundError(f"No *_world_results.npz under {log_dir / phase}")

    data = load_world_results(npz_path)
    required = {
        "trans",
        "root_orient",
        "pose_body",
        "betas",
        "is_right",
        "cam_R",
        "cam_t",
        "intrins",
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{npz_path} missing keys: {sorted(missing)}")

    track_vis_masks = None
    track_ids_list = None
    track_info_path = log_dir / "track_info.json"
    if track_info_path.is_file():
        tids, vis_masks, _, seq_interval = load_track_info(str(track_info_path))
        seq_start, seq_end = map(int, seq_interval)
        track_ids_list = [int(t) for t in tids.tolist()]
        track_vis_masks = {}
        for tid, mask in zip(track_ids_list, vis_masks.tolist()):
            m = np.asarray(mask, dtype=np.float32)
            if m.shape[0] >= seq_end:
                m = m[seq_start:seq_end]
            track_vis_masks[tid] = m

    payload = build_pose3d_hand(
        data,
        track_vis_masks,
        track_ids_list,
        fps=fps,
        prefer_track_index=prefer_track_index,
        disp_size=disp_size,
        max_slam_keyframes=max_slam_keyframes,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Saved {output} from {npz_path} (T={data['trans'].shape[1]}, B={data['trans'].shape[0]})")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Dyn-HaMR world_results.npz to .pose3d_hand format"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Optimization log directory (contains prior/ or smooth_fit/)",
    )
    parser.add_argument(
        "--phase",
        default="prior",
        choices=("prior", "smooth_fit", "root_fit", "init"),
        help="Which phase folder to read (default: prior)",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help="Explicit path to *_world_results.npz (overrides --phase lookup)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pose3d_hand path",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate for top-level fps")
    parser.add_argument(
        "--prefer-track-index",
        type=int,
        default=None,
        help="Batch index when multiple tracks share the same handedness",
    )
    parser.add_argument(
        "--disp-size",
        type=int,
        default=512,
        help="Placeholder disps spatial size (default 512)",
    )
    parser.add_argument(
        "--max-slam-keyframes",
        type=int,
        default=21,
        help="Maximum sparse slam_data.tstamp entries to emit (default 21)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        print(f"error: log-dir not found: {log_dir}", file=sys.stderr)
        return 1

    try:
        export_pose3d_hand(
            log_dir=log_dir,
            phase=args.phase,
            output=args.output.expanduser().resolve(),
            fps=args.fps,
            prefer_track_index=args.prefer_track_index,
            disp_size=args.disp_size,
            max_slam_keyframes=args.max_slam_keyframes,
            result_path=args.result.expanduser().resolve() if args.result else None,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
