#!/usr/bin/env python3
"""Export a Dyn-HaMR log directory to .pose3d_hand format.

Reads:
  - {root_fit,smooth_fit,prior}/*_world_results.npz  (MANO + cameras + world_scale)
  - track_info.json                                   (per-hand visibility masks)

Writes the same layout as demo/*.pose3d_hand and run_pose3d_hand_stages.py:
  left_hand / right_hand / slam_data / fps

Stage selection
---------------
Use --stage root|smooth|prior to pick the latest *_world_results.npz under the
matching subdirectory (sorted by iteration number in the filename).

Camera conversion (from npz)
----------------------------
npz cam_R / cam_t come from params.get_cameras() -> get_extrinsics(), where
cam_t is already scaled: cam_t_npz = _cam_t * world_scale.

This script un-scales cam_t, then writes slam_data as:
  traj[:3]   unscaled camera center t_c2w
  traj[3:7]  R_w2c quaternion (xyzw)
  scale      npz world_scale (re-applied to cam_t at load time)
  img_focal / img_center from npz intrins [fx, fy, cx, cy]

Track filling and timestamps
----------------------------
Export length T is authoritative from track_info.meta.seq_interval [start, end):
  T = end - start, tstamp = [0, T-1].

pred_valid comes from vis_mask[start:start+T] (padded with False if short).
Invalid frames keep MANO numeric values; only pred_valid/detection_failed flag them.
If a track is missing from track_info and npz, an empty hand slot is emitted.

Usage
-----
  python dyn-hamr/tools/export_pose3d_hand_from_log.py \\
    --log-dir outputs/logs/video-custom/segment_000_ch1_undistort/segment_000_ch1_undistort-all-shot-0-0--1 \\
    --stage prior \\
    --output demo/segment_000_ch1_undistort_from_log.pose3d_hand \\
    --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_DYN_HAMR_ROOT = Path(__file__).resolve().parents[1]
if str(_DYN_HAMR_ROOT) not in sys.path:
    sys.path.insert(0, str(_DYN_HAMR_ROOT))

from run_pose3d_hand_stages import (  # noqa: E402
    _coerce_pose3d_on_save,
    camera_to_slam_traj_unscaled,
    hand_slot_from_arrays,
    load_pose3d_payload,
    slam_traj_to_camera_scaled,
    slam_traj_to_camera_unscaled,
)

STAGE_SUBDIRS = {
    "root": "root_fit",
    "smooth": "smooth_fit",
    "prior": "prior",
}
STAGE_CHOICES = tuple(STAGE_SUBDIRS.keys())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _world_results_sort_key(path: Path) -> tuple[int, str]:
    parts = path.name.split("_")
    iteration = parts[-3]
    return (int(iteration) if iteration.isdigit() else -1, path.name)


def latest_world_results_npz(log_dir: Path, stage: str) -> Path:
    if stage not in STAGE_SUBDIRS:
        raise ValueError(f"unknown stage {stage!r}, expected one of {STAGE_CHOICES}")
    subdir = log_dir / STAGE_SUBDIRS[stage]
    candidates = sorted(subdir.glob("*_world_results.npz"), key=_world_results_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No *_world_results.npz under {subdir}")
    return candidates[-1]


def _align_time_series(arr: np.ndarray, T: int, axis: int = 0) -> np.ndarray:
    arr = np.asarray(arr)
    current = int(arr.shape[axis])
    if current == T:
        return arr
    if current > T:
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(0, T)
        return arr[tuple(sl)]
    if current == 0:
        return np.zeros(
            tuple(T if i == axis else arr.shape[i] for i in range(arr.ndim)),
            dtype=arr.dtype,
        )
    edge_slice = [slice(None)] * arr.ndim
    edge_slice[axis] = slice(current - 1, current)
    edge_value = arr[tuple(edge_slice)]
    return np.concatenate(
        [arr] + [np.repeat(edge_value, T - current, axis=axis)],
        axis=axis,
    )


def _track_entry(tracks: dict[str, Any], track_id: int | str) -> dict[str, Any] | None:
    for key in (str(track_id), int(track_id)):
        if key in tracks:
            return tracks[key]
    return None


def pred_valid_from_track_info(
    track_info: dict[str, Any],
    *,
    track_id: int | str,
    start: int,
    length: int,
) -> np.ndarray:
    tracks = track_info.get("tracks", track_info)
    entry = _track_entry(tracks, track_id)
    if entry is None:
        return np.zeros(length, dtype=bool)
    vis_mask = np.asarray(entry["vis_mask"], dtype=bool)
    sliced = vis_mask[start : start + length]
    if sliced.shape[0] < length:
        sliced = np.pad(sliced, (0, length - sliced.shape[0]), constant_values=False)
    return sliced.astype(bool)


def _hand_batch_index(npz: dict[str, np.ndarray], handedness: int) -> int | None:
    if "is_right" not in npz:
        return None
    is_right = np.asarray(npz["is_right"], dtype=np.float32)
    for b in range(is_right.shape[0]):
        if int(round(float(is_right[b, 0]))) == handedness:
            return b
    return None


def empty_hand_slot(T: int) -> dict[str, Any]:
    return hand_slot_from_arrays(
        np.zeros((T, 3), dtype=np.float32),
        np.zeros((T, 45), dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.zeros((T, 3), dtype=np.float32),
        np.zeros(T, dtype=bool),
    )


def npz_to_slam_data(
    npz: dict[str, np.ndarray],
    T: int,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    Derive pose3d_hand slam_data camera fields from npz.

    npz cam_t is scaled (see optim.params.CameraParams.get_extrinsics); divide by
    world_scale before camera_to_slam_traj_unscaled.
    """
    if "cam_R" not in npz or "cam_t" not in npz:
        raise KeyError("npz must contain cam_R and cam_t for slam export")
    scale = float(np.asarray(npz["world_scale"], dtype=np.float64).reshape(-1)[0])
    cam_R = _align_time_series(np.asarray(npz["cam_R"][0], dtype=np.float32), T, axis=0)
    cam_t_scaled = _align_time_series(np.asarray(npz["cam_t"][0], dtype=np.float32), T, axis=0)
    cam_t_unscaled = (cam_t_scaled / np.float32(scale)).astype(np.float32)
    traj = camera_to_slam_traj_unscaled(cam_R, cam_t_unscaled)

    intrins = np.asarray(npz["intrins"], dtype=np.float32).reshape(-1)
    if intrins.size != 4:
        raise ValueError(f"expected npz intrins with 4 values, got shape {intrins.shape}")
    img_focal = float(np.mean(intrins[:2]))
    img_center = intrins[2:4].astype(np.float32)
    return traj, scale, img_focal, img_center


def _hand_slot_from_npz(
    npz: dict[str, np.ndarray],
    batch_idx: int,
    T: int,
    pred_valid: np.ndarray,
) -> dict[str, Any]:
    pose_body = _align_time_series(
        np.asarray(npz["pose_body"][batch_idx], dtype=np.float32),
        T,
        axis=0,
    ).reshape(T, 15, 3)
    hand_pose = pose_body.reshape(T, 45)
    betas = np.asarray(npz["betas"][batch_idx], dtype=np.float32).reshape(-1)
    if betas.size == 10:
        betas_out = betas
    else:
        betas_out = betas.reshape(-1)[:10]
    return hand_slot_from_arrays(
        _align_time_series(np.asarray(npz["root_orient"][batch_idx], dtype=np.float32), T, axis=0),
        hand_pose,
        betas_out,
        _align_time_series(np.asarray(npz["trans"][batch_idx], dtype=np.float32), T, axis=0),
        pred_valid,
    )


def build_hand_slot(
    npz: dict[str, np.ndarray],
    track_info: dict[str, Any],
    *,
    handedness: int,
    start: int,
    T: int,
) -> dict[str, Any]:
    track_id = handedness
    pred_valid = pred_valid_from_track_info(track_info, track_id=track_id, start=start, length=T)
    batch_idx = _hand_batch_index(npz, handedness)
    tracks = track_info.get("tracks", track_info)
    has_track = _track_entry(tracks, track_id) is not None

    if batch_idx is None:
        return empty_hand_slot(T)
    if not has_track:
        return _hand_slot_from_npz(npz, batch_idx, T, np.zeros(T, dtype=bool))
    return _hand_slot_from_npz(npz, batch_idx, T, pred_valid)


def build_pose3d_payload(
    npz_path: Path,
    track_info_path: Path,
    *,
    fps: float = 30.0,
) -> dict[str, Any]:
    npz = _load_npz(npz_path)
    track_info = _load_json(track_info_path)
    meta = track_info.get("meta", {})
    seq_interval = meta.get("seq_interval", [0, npz["trans"].shape[1]])
    start, end = int(seq_interval[0]), int(seq_interval[1])

    T = end - start
    if T <= 0:
        raise ValueError(f"Invalid export length T={T} from seq_interval {seq_interval}")

    traj, scale, img_focal, img_center = npz_to_slam_data(npz, T)

    return {
        "left_hand": build_hand_slot(npz, track_info, handedness=0, start=start, T=T),
        "right_hand": build_hand_slot(npz, track_info, handedness=1, start=start, T=T),
        "fps": float(fps),
        "slam_data": {
            "tstamp": np.asarray([0.0, float(max(T - 1, 0))], dtype=np.float32),
            "traj": traj,
            "img_focal": np.asarray(img_focal, dtype=np.float32),
            "img_center": img_center,
            "scale": np.asarray(scale, dtype=np.float64),
            "slam_n_chunks": np.asarray(1, dtype=np.int64),
            "max_slam_frames": np.asarray(max(900, T), dtype=np.int64),
            "slam_overlap_frames": np.asarray(0, dtype=np.int64),
        },
    }


def verify_camera_consistency(payload: dict[str, Any], npz_path: Path) -> dict[str, float]:
    npz = _load_npz(npz_path)
    slam = payload["slam_data"]
    scale = float(np.asarray(slam["scale"], dtype=np.float64).reshape(-1)[0])
    traj = np.asarray(slam["traj"], dtype=np.float32)
    cam_R, cam_t = slam_traj_to_camera_scaled(traj, scale)

    T = cam_R.shape[0]
    npz_cam_R = _align_time_series(np.asarray(npz["cam_R"][0], dtype=np.float32), T, axis=0)
    npz_cam_t = _align_time_series(np.asarray(npz["cam_t"][0], dtype=np.float32), T, axis=0)
    return {
        "cam_R_max_diff": float(np.max(np.abs(cam_R - npz_cam_R))),
        "cam_t_max_diff": float(np.max(np.abs(cam_t - npz_cam_t))),
        "frames": float(T),
    }


def save_pose3d_payload(payload: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_coerce_pose3d_on_save(payload), output)
    return output


def _default_log_paths(log_dir: Path, stage: str) -> tuple[Path, Path]:
    npz = latest_world_results_npz(log_dir, stage)
    track_info = log_dir / "track_info.json"
    return npz, track_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Dyn-HaMR log directory (root_fit/, smooth_fit/, prior/, track_info.json)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .pose3d_hand path")
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="prior",
        help="Optimization stage subdirectory to read (default: prior)",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Override auto-discovered *_world_results.npz",
    )
    parser.add_argument("--track-info", type=Path, default=None, help="Override track_info.json")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check exported traj against source npz cam_R/cam_t",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    stage = str(args.stage)
    npz_default, track_info_default = _default_log_paths(log_dir, stage)
    npz_path = (args.npz or npz_default).resolve()
    track_info_path = (args.track_info or track_info_default).resolve()
    output = args.output.expanduser().resolve()
    if output.suffix != ".pose3d_hand":
        output = output.with_suffix(".pose3d_hand")

    for path in (npz_path, track_info_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = build_pose3d_payload(
        npz_path,
        track_info_path,
        fps=float(args.fps),
    )
    save_pose3d_payload(payload, output)

    T = int(np.asarray(payload["slam_data"]["traj"]).shape[0])
    scale = float(np.asarray(payload["slam_data"]["scale"]).reshape(-1)[0])
    tstamp = np.asarray(payload["slam_data"]["tstamp"], dtype=np.float32).reshape(-1)
    print(f"Wrote {output}")
    print(f"  stage={stage}, npz={npz_path}")
    print(f"  frames={T}, scale={scale:.6f}, tstamp={tstamp.tolist()}")
    print(f"  left valid={int(payload['left_hand']['pred_valid'].sum())}/{T}")
    print(f"  right valid={int(payload['right_hand']['pred_valid'].sum())}/{T}")

    if args.verify:
        stats = verify_camera_consistency(payload, npz_path)
        print(
            "  verify:",
            f"cam_R max diff={stats['cam_R_max_diff']:.3e},",
            f"cam_t max diff={stats['cam_t_max_diff']:.3e}",
        )
        loaded = load_pose3d_payload(output)
        assert set(loaded.keys()) >= {"left_hand", "right_hand", "slam_data", "fps"}
        print("  load_pose3d_payload: OK")

    traj = np.asarray(payload["slam_data"]["traj"], dtype=np.float32)
    cam_R, cam_t_unscaled = slam_traj_to_camera_unscaled(traj)
    back = camera_to_slam_traj_unscaled(cam_R, cam_t_unscaled)
    rt_err = float(np.max(np.abs(back - traj)))
    print(f"  traj roundtrip max diff={rt_err:.3e}")


if __name__ == "__main__":
    main()
