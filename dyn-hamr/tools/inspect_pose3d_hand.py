#!/usr/bin/env python3
"""Print a concise preview of .pose3d_hand field shapes and first N frames.

Usage
-----
  python dyn-hamr/tools/inspect_pose3d_hand.py demo/foo.pose3d_hand
  python dyn-hamr/tools/inspect_pose3d_hand.py demo/foo.pose3d_hand --n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_DYN_HAMR_ROOT = Path(__file__).resolve().parents[1]
if str(_DYN_HAMR_ROOT) not in sys.path:
    sys.path.insert(0, str(_DYN_HAMR_ROOT))

from run_pose3d_hand_stages import load_pose3d_payload  # noqa: E402


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _indent_str(indent: int) -> str:
    return "  " * indent


def _print_lines(indent: int, lines: list[str]) -> None:
    prefix = _indent_str(indent)
    for line in lines:
        print(f"{prefix}{line}")


def preview_array(
    name: str,
    arr: np.ndarray,
    *,
    n: int = 3,
    indent: int = 0,
    time_axis: int | None = 0,
    extra_note: str = "",
) -> None:
    note = f"  {extra_note}" if extra_note else ""
    header = f"{name}: shape={arr.shape} dtype={arr.dtype}{note}"
    _print_lines(indent, [header])

    if arr.size == 0:
        _print_lines(indent + 1, ["(empty)"])
        return

    if arr.ndim == 0 or (arr.ndim == 1 and arr.shape[0] <= 10):
        _print_lines(indent + 1, [f"value: {np.array2string(arr, precision=4, suppress_small=True)}"])
        return

    if time_axis is not None and arr.ndim > time_axis and arr.shape[time_axis] > 1:
        end = min(n, arr.shape[time_axis])
        for i in range(end):
            item = np.take(arr, i, axis=time_axis)
            text = np.array2string(item, precision=4, suppress_small=True)
            _print_lines(indent + 1, [f"[{i}]: {text}"])
        if arr.shape[time_axis] > end:
            _print_lines(indent + 1, [f"... ({arr.shape[time_axis] - end} more frames)"])
        return

    flat_preview = arr.reshape(-1)[:n]
    _print_lines(
        indent + 1,
        [f"preview: {np.array2string(flat_preview, precision=4, suppress_small=True)}"],
    )


def preview_bool_mask(name: str, mask: np.ndarray, *, n: int = 3, indent: int = 0) -> None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    valid_count = int(mask.sum())
    header = f"{name}: shape={mask.shape} dtype=bool  valid={valid_count}/{len(mask)}"
    _print_lines(indent, [header])
    end = min(n, len(mask))
    _print_lines(indent + 1, [f"[:{end}]: {mask[:end].tolist()}"])


def preview_hand_pose(name: str, hand_pose: np.ndarray, *, n: int = 3, indent: int = 0) -> None:
    hp = np.asarray(hand_pose, dtype=np.float32).reshape(-1, 45)
    preview_array(name, hp, n=n, indent=indent, time_axis=0, extra_note="(T, 45)")
    end = min(n, hp.shape[0])
    aa = hp[:end].reshape(end, 15, 3)
    _print_lines(indent + 1, [f"as axis-angle (15, 3) for frames [0:{end}):"])
    for i in range(end):
        text = np.array2string(aa[i], precision=4, suppress_small=True)
        _print_lines(indent + 2, [f"[{i}]: {text}"])


def preview_betas(name: str, betas: np.ndarray, *, n: int = 3, indent: int = 0) -> None:
    b = np.asarray(betas, dtype=np.float32)
    if b.ndim == 1:
        preview_array(name, b, n=n, indent=indent, time_axis=None)
        return
    preview_array(name, b, n=n, indent=indent, time_axis=0)


def preview_traj(name: str, traj: np.ndarray, *, n: int = 3, indent: int = 0) -> None:
    t = np.asarray(traj, dtype=np.float32).reshape(-1, 7)
    preview_array(
        name,
        t,
        n=n,
        indent=indent,
        time_axis=0,
        extra_note="(t_c2w[:3] + quat_xyzw[3:7])",
    )


def preview_value(name: str, value: Any, *, n: int = 3, indent: int = 0) -> None:
    """Recursively print a pose3d_hand field preview."""
    if value is None:
        _print_lines(indent, [f"{name}: None"])
        return

    if isinstance(value, (float, int, bool, np.integer, np.floating, np.bool_)):
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            value = value.item()
        _print_lines(indent, [f"{name}: {value}"])
        return

    if isinstance(value, dict):
        _print_lines(indent, [f"{name}:"])
        for key, sub in value.items():
            preview_value(str(key), sub, n=n, indent=indent + 1)
        return

    if isinstance(value, (np.ndarray, torch.Tensor)):
        arr = _as_numpy(value)
        if name.endswith("hand_pose") or name == "hand_pose":
            preview_hand_pose(name, arr, n=n, indent=indent)
        elif name.endswith("betas") or name == "betas":
            preview_betas(name, arr, n=n, indent=indent)
        elif name == "traj":
            preview_traj(name, arr, n=n, indent=indent)
        elif name in {"pred_valid", "detection_failed", "pair_valid"}:
            preview_bool_mask(name, arr, n=n, indent=indent)
        elif arr.dtype == bool:
            preview_bool_mask(name, arr, n=n, indent=indent)
        else:
            preview_array(name, arr, n=n, indent=indent, time_axis=0)
        return

    _print_lines(indent, [f"{name}: {type(value).__name__} = {value!r}"])


def print_pose3d_hand_preview(
    path: str | Path,
    *,
    n_frames: int = 3,
) -> dict[str, Any]:
    """Load .pose3d_hand and print field summary; return payload for further use."""
    pose_path = Path(path).expanduser().resolve()
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    payload = load_pose3d_payload(pose_path)

    print(f"=== pose3d_hand: {pose_path} ===")
    if "fps" in payload:
        preview_value("fps", payload["fps"], n=n_frames, indent=0)

    for hand_key in ("left_hand", "right_hand"):
        if hand_key in payload:
            print()
            print(f"--- {hand_key} ---")
            hand = payload[hand_key]
            if "mano_params" in hand:
                print("  mano_params:")
                mano = hand["mano_params"]
                for mano_key in ("global_orient", "hand_pose", "betas", "transl"):
                    if mano_key in mano:
                        preview_value(
                            f"mano_params.{mano_key}",
                            mano[mano_key],
                            n=n_frames,
                            indent=1,
                        )
            for extra_key in ("pred_valid", "detection_failed", "relative_motion"):
                if extra_key in hand:
                    preview_value(extra_key, hand[extra_key], n=n_frames, indent=1)

    if "slam_data" in payload:
        print()
        print("--- slam_data ---")
        slam = payload["slam_data"]
        preferred = (
            "tstamp",
            "traj",
            "img_focal",
            "img_center",
            "scale",
            "slam_n_chunks",
            "max_slam_frames",
            "slam_overlap_frames",
        )
        seen = set()
        for key in preferred:
            if key in slam:
                preview_value(key, slam[key], n=n_frames, indent=1)
                seen.add(key)
        for key, value in slam.items():
            if key not in seen:
                preview_value(key, value, n=n_frames, indent=1)

    for key, value in payload.items():
        if key not in {"fps", "left_hand", "right_hand", "slam_data"}:
            preview_value(key, value, n=n_frames, indent=0)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to .pose3d_hand file")
    parser.add_argument(
        "-n",
        "--n-frames",
        type=int,
        default=3,
        dest="n_frames",
        help="Number of leading frames/elements to preview (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_pose3d_hand_preview(args.path, n_frames=max(1, int(args.n_frames)))


if __name__ == "__main__":
    main()
