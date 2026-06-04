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


class Timeline:
    def __init__(
        self,
        data_interval: tuple[int, int],
        seq_interval: tuple[int, int],
    ) -> None:
        data_start, data_end = data_interval
        seq_start, seq_end = seq_interval
        if data_end <= data_start:
            raise ValueError(f"Invalid data_interval {data_interval}")
        if seq_start < data_start or seq_end > data_end or seq_end <= seq_start:
            raise ValueError(
                f"Invalid seq_interval {seq_interval} for data_interval {data_interval}"
            )
        self.data_start = data_start
        self.data_end = data_end
        self.seq_start = seq_start
        self.seq_end = seq_end

    @property
    def full_len(self) -> int:
        return self.data_end - self.data_start

    @property
    def seq_len(self) -> int:
        return self.seq_end - self.seq_start

    @property
    def insert_start(self) -> int:
        return self.seq_start - self.data_start

    @property
    def insert_end(self) -> int:
        return self.seq_end - self.data_start


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


def _as_float_scalar(value: np.ndarray | float | int) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot read scalar from an empty array")
    return float(arr[0])


def _pred_valid_from_vis_mask(vis_mask: np.ndarray, seq_len: int) -> np.ndarray:
    raw = np.asarray(vis_mask).reshape(-1)
    if raw.dtype == np.bool_:
        mask = raw.astype(bool)
        if mask.shape[0] > seq_len:
            mask = mask[:seq_len]
        elif mask.shape[0] < seq_len:
            padded = np.zeros(seq_len, dtype=bool)
            padded[: mask.shape[0]] = mask
            mask = padded
        return mask

    mask = raw.astype(np.float32)
    seq_start, seq_end = 0, seq_len
    if mask.shape[0] > seq_len:
        # track_info vis_mask covers the optimized subsequence interval
        mask = mask[:seq_len]
    elif mask.shape[0] < seq_len:
        padded = np.full(seq_len, -1, dtype=np.float32)
        padded[: mask.shape[0]] = mask
        mask = padded
    return (mask >= 0).astype(bool)


def _vis_mask_for_timeline(vis_mask: np.ndarray, timeline: Timeline) -> np.ndarray:
    mask = np.asarray(vis_mask).reshape(-1)
    is_bool = mask.dtype == np.bool_
    if mask.shape[0] >= timeline.data_end:
        mask = mask[timeline.data_start : timeline.data_end]
    elif mask.shape[0] > timeline.full_len:
        mask = mask[: timeline.full_len]
    elif mask.shape[0] < timeline.full_len:
        fill_value = False if is_bool else -1
        padded = np.full(timeline.full_len, fill_value, dtype=mask.dtype)
        padded[: mask.shape[0]] = mask
        mask = padded
    return mask


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
    pred_valid = np.zeros(seq_len, dtype=bool)
    return {
        "mano_params": {
            "global_orient": np.zeros((seq_len, 3), dtype=np.float32),
            "hand_pose": np.zeros((seq_len, 45), dtype=np.float32),
            "betas": np.zeros((seq_len, 10), dtype=np.float32),
            "transl": np.zeros((seq_len, 3), dtype=np.float32),
        },
        "relative_motion": {
            "rel_rot_aa": torch.zeros(seq_len, 3, dtype=torch.float32),
            "rel_rot_mat": torch.eye(3, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(seq_len, 1, 1),
            "rel_trans": torch.zeros(seq_len, 3, dtype=torch.float32),
            "pair_valid": torch.zeros(seq_len, dtype=torch.bool),
        },
        "pred_valid": pred_valid,
        "detection_failed": ~pred_valid,
    }


def build_hand_slot(
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    pred_valid: np.ndarray,
) -> dict:
    T = trans.shape[0]
    global_orient = np.asarray(root_orient, dtype=np.float32).reshape(T, 3)
    hand_pose = np.asarray(pose_body, dtype=np.float32).reshape(T, 45)
    transl = np.asarray(trans, dtype=np.float32).reshape(T, 3)
    betas_arr = _betas_per_frame(betas, T)
    pred_valid_arr = np.asarray(pred_valid, dtype=bool)

    global_orient_t = torch.from_numpy(global_orient)
    transl_t = torch.from_numpy(transl)
    pred_valid_t = torch.from_numpy(pred_valid_arr)

    return {
        "mano_params": {
            "global_orient": global_orient,
            "hand_pose": hand_pose,
            "betas": betas_arr,
            "transl": transl,
        },
        "relative_motion": _compute_relative_motion(global_orient_t, transl_t, pred_valid_t),
        "pred_valid": pred_valid_arr,
        "detection_failed": ~pred_valid_arr,
    }


def _insert_hand_slot(full_hand: dict, slot: dict, timeline: Timeline) -> dict:
    start, end = timeline.insert_start, timeline.insert_end
    for key, value in slot["mano_params"].items():
        full_hand["mano_params"][key][start:end] = value
    for key, value in slot["relative_motion"].items():
        full_hand["relative_motion"][key][start:end] = value
    full_hand["pred_valid"][start:end] = slot["pred_valid"]
    full_hand["detection_failed"] = ~full_hand["pred_valid"]
    return full_hand


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


def _camera_traj_from_arrays(cam_t_arr: np.ndarray, cam_R_arr: np.ndarray) -> np.ndarray:
    cam_t = torch.from_numpy(np.asarray(cam_t_arr, dtype=np.float32).reshape(-1, 3))
    cam_R = torch.from_numpy(np.asarray(cam_R_arr, dtype=np.float32).reshape(-1, 3, 3))
    quat_wxyz = matrix_to_quaternion(cam_R)
    quat_xyzw = torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)
    traj = torch.cat([cam_t, quat_xyzw], dim=-1).to(dtype=torch.float32)
    return traj.numpy()


def _camera_traj(data: dict[str, np.ndarray], batch_idx: int, seq_len: int) -> np.ndarray:
    cam_t = _camera_track_array(data["cam_t"], batch_idx, seq_len, (3,))
    cam_R = _camera_track_array(data["cam_R"], batch_idx, seq_len, (3, 3))
    return _camera_traj_from_arrays(cam_t, cam_R)


def _camera_traj_for_timeline(
    data: dict[str, np.ndarray],
    batch_idx: int,
    timeline: Timeline,
) -> np.ndarray:
    cam_t_arr = np.asarray(data["cam_t"], dtype=np.float32)
    cam_R_arr = np.asarray(data["cam_R"], dtype=np.float32)
    full_shape_t = (timeline.full_len, 3)
    full_shape_r = (timeline.full_len, 3, 3)
    if cam_t_arr.shape == full_shape_t and cam_R_arr.shape == full_shape_r:
        cam_t = cam_t_arr
        cam_R = cam_R_arr
    elif (
        cam_t_arr.ndim >= 2
        and cam_R_arr.ndim >= 3
        and cam_t_arr.shape[-2:] == full_shape_t
        and cam_R_arr.shape[-3:] == full_shape_r
    ):
        cam_t = cam_t_arr.reshape(-1, timeline.full_len, 3)[batch_idx]
        cam_R = cam_R_arr.reshape(-1, timeline.full_len, 3, 3)[batch_idx]
    else:
        cam_t = np.zeros(full_shape_t, dtype=np.float32)
        cam_R = np.eye(3, dtype=np.float32)[None].repeat(timeline.full_len, axis=0)
        seq_cam_t = _camera_track_array(data["cam_t"], batch_idx, timeline.seq_len, (3,))
        seq_cam_R = _camera_track_array(data["cam_R"], batch_idx, timeline.seq_len, (3, 3))
        cam_t[timeline.insert_start : timeline.insert_end] = seq_cam_t
        cam_R[timeline.insert_start : timeline.insert_end] = seq_cam_R
    return _camera_traj_from_arrays(cam_t, cam_R)


def _slam_tstamp(seq_len: int, max_keyframes: int) -> np.ndarray:
    if seq_len <= 0:
        return np.zeros(0, dtype=np.int32)
    if seq_len <= max_keyframes:
        return np.arange(seq_len, dtype=np.int32)
    if max_keyframes <= 1:
        return np.array([0], dtype=np.int32)
    step = max(1, seq_len // (max_keyframes - 1))
    tstamp = np.arange(0, seq_len, step, dtype=np.int32)[:max_keyframes]
    if tstamp.size < max_keyframes:
        tstamp = np.unique(np.concatenate([tstamp, np.array([seq_len - 1], dtype=np.int32)]))
    return tstamp[:max_keyframes]


def _slam_n_chunks(seq_len: int, max_slam_frames: int, slam_overlap_frames: int) -> int:
    if max_slam_frames <= 0:
        raise ValueError("--max-slam-frames must be positive")
    if slam_overlap_frames < 0:
        raise ValueError("--slam-overlap-frames must be non-negative")
    if slam_overlap_frames >= max_slam_frames:
        raise ValueError("--slam-overlap-frames must be smaller than --max-slam-frames")
    if seq_len <= max_slam_frames:
        return 1
    stride = max_slam_frames - slam_overlap_frames
    return int(np.ceil((seq_len - slam_overlap_frames) / stride))


def _optional_scalar(
    data: dict[str, np.ndarray],
    keys: tuple[str, ...],
    default: float,
    batch_idx: int = 0,
) -> float:
    for key in keys:
        if key not in data:
            continue
        value = np.asarray(data[key], dtype=np.float64)
        if value.ndim > 0 and value.shape[0] > batch_idx and value.size > 1:
            value = value[batch_idx]
        return _as_float_scalar(value)
    return float(default)


def build_pose3d_hand(
    data: dict[str, np.ndarray],
    track_vis_masks: dict[int, np.ndarray] | None,
    track_ids: list[int] | None,
    fps: float,
    prefer_track_index: int | None,
    max_slam_keyframes: int,
    max_slam_frames: int,
    slam_overlap_frames: int,
    timeline: Timeline | None,
) -> dict:
    trans = _as_batch_array(data["trans"])
    B, seq_len = trans.shape[0], trans.shape[1]
    if timeline is not None and timeline.seq_len != seq_len:
        raise ValueError(
            f"world_results T={seq_len} does not match seq_interval length "
            f"{timeline.seq_len} ({timeline.seq_start}, {timeline.seq_end})"
        )
    out_len = timeline.full_len if timeline is not None else seq_len

    left_hand = _empty_hand(out_len)
    right_hand = _empty_hand(out_len)

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
            if timeline is not None:
                vis = _vis_mask_for_timeline(vis, timeline)[
                    timeline.insert_start : timeline.insert_end
                ]
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
        if timeline is not None:
            slot = _insert_hand_slot(_empty_hand(out_len), slot, timeline)
        if is_right:
            right_hand = slot
        else:
            left_hand = slot

    traj_batch = export_indices[0] if export_indices else 0
    focal, center = _intrinsics_from_npz(data, traj_batch)
    fps_value = _optional_scalar(data, ("fps",), fps, traj_batch)
    scale_value = _optional_scalar(data, ("scale", "world_scale"), 1.0, traj_batch)
    traj = (
        _camera_traj_for_timeline(data, traj_batch, timeline)
        if timeline is not None
        else _camera_traj(data, traj_batch, seq_len)
    )
    slam_data = {
        "tstamp": _slam_tstamp(out_len, max_slam_keyframes),
        "traj": traj,
        "img_focal": np.array(focal, dtype=np.float64),
        "img_center": center.astype(np.float64),
        "scale": np.array(scale_value, dtype=np.float64),
        "slam_n_chunks": _slam_n_chunks(out_len, max_slam_frames, slam_overlap_frames),
        "max_slam_frames": int(max_slam_frames),
        "slam_overlap_frames": int(slam_overlap_frames),
    }

    return {
        "fps": float(fps_value),
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
    max_slam_keyframes: int,
    max_slam_frames: int,
    slam_overlap_frames: int,
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
    timeline = None
    track_info_path = log_dir / "track_info.json"
    if track_info_path.is_file():
        tids, vis_masks, data_interval, seq_interval = load_track_info(str(track_info_path))
        data_start, data_end = map(int, data_interval)
        seq_start, seq_end = map(int, seq_interval)
        timeline = Timeline((data_start, data_end), (seq_start, seq_end))
        track_ids_list = [int(t) for t in tids.tolist()]
        track_vis_masks = {}
        for tid, mask in zip(track_ids_list, vis_masks.tolist()):
            track_vis_masks[tid] = np.asarray(mask)

    payload = build_pose3d_hand(
        data,
        track_vis_masks,
        track_ids_list,
        fps=fps,
        prefer_track_index=prefer_track_index,
        max_slam_keyframes=max_slam_keyframes,
        max_slam_frames=max_slam_frames,
        slam_overlap_frames=slam_overlap_frames,
        timeline=timeline,
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
        help="Deprecated no-op; disps is not emitted by default",
    )
    parser.add_argument(
        "--max-slam-keyframes",
        type=int,
        default=21,
        help="Maximum sparse slam_data.tstamp entries to emit (default 21)",
    )
    parser.add_argument(
        "--max-slam-frames",
        type=int,
        default=900,
        help="Maximum frames per SLAM chunk metadata field (default 900)",
    )
    parser.add_argument(
        "--slam-overlap-frames",
        type=int,
        default=30,
        help="Overlap frames between SLAM chunks metadata field (default 30)",
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
            max_slam_keyframes=args.max_slam_keyframes,
            max_slam_frames=args.max_slam_frames,
            slam_overlap_frames=args.slam_overlap_frames,
            result_path=args.result.expanduser().resolve() if args.result else None,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
