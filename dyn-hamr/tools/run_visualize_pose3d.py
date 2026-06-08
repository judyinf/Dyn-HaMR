#!/usr/bin/env python3
"""
overlay MANO hand meshes on raw videos using .pose3d_hand predictions.

Layout mirrors pose3d outputs under [visualization] visual_result_root:
  <visual_result_root>/<canonical_dataset>/<algorithm>/<stem>.mp4
"""
from __future__ import annotations

import argparse
import configparser
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

_EVAL_PIPELINE_ROOT = Path(__file__).resolve().parent
if str(_EVAL_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_PIPELINE_ROOT))
from pipeline_paths import apply_run_tag_to_dir, normalize_run_tag

import imageio.v2 as imageio
import numpy as np
import torch
from manopth.manolayer import ManoLayer
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
)
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.camera_conversions import _cameras_from_opencv_projection
from pytorch3d.structures import Meshes
from tqdm import tqdm


def create_worker_render_runtime(
    model_dir: Path,
    device: str,
) -> Dict[str, Any]:
    """Initialize reusable heavy render resources once per worker."""
    device_t = select_device(device)
    left_model = build_mano_model(model_dir, "left", device_t)
    right_model = build_mano_model(model_dir, "right", device_t)
    return {
        "device_t": device_t,
        "left_model": left_model,
        "right_model": right_model,
        "faces_left": left_model.th_faces.to(device_t).long(),
        "faces_right": right_model.th_faces.to(device_t).long(),
        "left_color": torch.tensor([0.2, 0.8, 0.2], device=device_t).view(1, 3),
        "right_color": torch.tensor([0.8, 0.2, 0.2], device=device_t).view(1, 3),
        "lights": PointLights(device=device_t, location=[[0.0, 0.0, -2.0]]),
        "renderer_cache": {},
    }


def get_cached_renderer(
    runtime: Dict[str, Any],
    image_size: Tuple[int, int],
    bin_size: Optional[int],
    max_faces_per_bin: Optional[int],
) -> MeshRenderer:
    renderer_cache: Dict[Tuple[int, int, int, int], MeshRenderer] = runtime["renderer_cache"]
    bin_tag = -1 if bin_size is None else int(bin_size)
    max_faces_tag = -1 if max_faces_per_bin is None else int(max_faces_per_bin)
    key = (int(image_size[0]), int(image_size[1]), bin_tag, max_faces_tag)
    if key not in renderer_cache:
        renderer_cache[key] = build_renderer(
            image_size=image_size,
            device=runtime["device_t"],
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin,
        )
    return renderer_cache[key]


def _resize_frame_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        return cv2.resize(frame, (width, height))
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(frame).resize((width, height)), dtype=np.uint8)


def _draw_bbox_on_frame(
    frame: np.ndarray, box_xyxy: np.ndarray, color: Tuple[int, int, int], thickness: int
) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy.tolist()]
    x1 = max(0, min(x1, frame.shape[1] - 1))
    y1 = max(0, min(y1, frame.shape[0] - 1))
    x2 = max(0, min(x2, frame.shape[1] - 1))
    y2 = max(0, min(y2, frame.shape[0] - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return frame

    try:
        import cv2  # type: ignore

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thickness))
        return frame
    except ImportError:
        from PIL import Image, ImageDraw

        rgb = frame[..., ::-1]
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], outline=(color[2], color[1], color[0]), width=max(1, thickness))
        out_rgb = np.asarray(img, dtype=np.uint8)
        frame[:] = out_rgb[..., ::-1]
        return frame


def _looks_like_xyxy(arr: np.ndarray) -> bool:
    valid = np.isfinite(arr).all(axis=-1)
    if valid.size == 0 or not np.any(valid):
        return True
    rows = arr[valid]
    return bool(np.mean((rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])) >= 0.6)


def _sanitize_xyxy(box: np.ndarray, frame_w: int, frame_h: int) -> Optional[np.ndarray]:
    if box.shape[0] < 4 or not np.isfinite(box[:4]).all():
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(max(frame_w - 1, 0)))
    y1 = min(max(y1, 0.0), float(max(frame_h - 1, 0)))
    x2 = min(max(x2, 0.0), float(max(frame_w - 1, 0)))
    y2 = min(max(y2, 0.0), float(max(frame_h - 1, 0)))
    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _extract_numeric_box_candidates(obj: Any) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for item in obj.flat:
                out.extend(_extract_numeric_box_candidates(item))
            return out
        if np.issubdtype(obj.dtype, np.number) and obj.ndim >= 2 and obj.shape[-1] >= 4:
            out.append(obj)
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_l = str(k).lower()
            if ("box" in key_l) or ("bbox" in key_l) or ("track" in key_l):
                out.extend(_extract_numeric_box_candidates(v))
        if not out:
            for v in obj.values():
                out.extend(_extract_numeric_box_candidates(v))
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_extract_numeric_box_candidates(v))
    return out


def _candidate_to_frame_boxes(arr: np.ndarray, frame_w: int, frame_h: int) -> List[Optional[np.ndarray]]:
    coord = arr[..., :4].astype(np.float64, copy=False)
    if coord.ndim == 2:
        rows = coord
    else:
        rows = coord.reshape(coord.shape[0], -1, 4)
    is_xyxy = _looks_like_xyxy(coord.reshape(-1, 4))
    boxes: List[Optional[np.ndarray]] = []

    if coord.ndim == 2:
        for i in range(rows.shape[0]):
            row = rows[i]
            raw = row if is_xyxy else np.asarray([row[0], row[1], row[0] + row[2], row[1] + row[3]])
            boxes.append(_sanitize_xyxy(raw, frame_w, frame_h))
        return boxes

    for i in range(rows.shape[0]):
        frame_boxes = rows[i]
        chosen: Optional[np.ndarray] = None
        for j in range(frame_boxes.shape[0]):
            row = frame_boxes[j]
            raw = row if is_xyxy else np.asarray([row[0], row[1], row[0] + row[2], row[1] + row[3]])
            chosen = _sanitize_xyxy(raw, frame_w, frame_h)
            if chosen is not None:
                break
        boxes.append(chosen)
    return boxes


def _parse_box_from_frame_record(record: Any, frame_w: int, frame_h: int) -> Optional[np.ndarray]:
    if not isinstance(record, dict):
        return None
    det_flag = record.get("det", True)
    if isinstance(det_flag, (bool, np.bool_)) and not bool(det_flag):
        return None
    if "det_box" not in record:
        return None
    det_box = record["det_box"]
    try:
        arr = np.asarray(det_box, dtype=np.float64)
    except Exception:
        return None
    if arr.size < 4:
        return None
    if arr.ndim == 1:
        row = arr[:4]
    else:
        row = arr.reshape(-1, arr.shape[-1])[0][:4]
    # Tracks produced by preprocess are xyxy(+score), consistent with preprocess_dataset.
    return _sanitize_xyxy(row, frame_w, frame_h)


def _load_bbox_from_track_dict(
    data: Any, frame_w: int, frame_h: int
) -> Optional[List[Dict[int, np.ndarray]]]:
    # Expected structure:
    #   {track_id: [ {"frame": i, "det": bool, "det_box": (1,5), ...}, ... ], ...}
    if not isinstance(data, dict) or not data:
        return None
    list_items = [(k, v) for k, v in data.items() if isinstance(v, list) and v]
    if not list_items:
        return None

    max_frame_idx = -1
    per_track_records: List[Tuple[int, List[Tuple[int, Optional[np.ndarray]]]]] = []
    for fallback_track_id, (track_id_raw, records) in enumerate(list_items):
        parsed: List[Tuple[int, Optional[np.ndarray]]] = []
        try:
            track_id = int(track_id_raw)
        except Exception:
            track_id = int(fallback_track_id)
        for rec_idx, rec in enumerate(records):
            frame_idx = rec_idx
            if isinstance(rec, dict):
                frame_raw = rec.get("frame", rec_idx)
                if isinstance(frame_raw, (int, np.integer)):
                    frame_idx = int(frame_raw)
            box = _parse_box_from_frame_record(rec, frame_w, frame_h)
            parsed.append((frame_idx, box))
            if frame_idx > max_frame_idx:
                max_frame_idx = frame_idx
        per_track_records.append((track_id, parsed))

    if max_frame_idx < 0:
        return None

    boxes_per_frame: List[Dict[int, np.ndarray]] = [{} for _ in range(max_frame_idx + 1)]
    for track_id, parsed in per_track_records:
        for frame_idx, box in parsed:
            if box is None or frame_idx < 0 or frame_idx >= len(boxes_per_frame):
                continue
            boxes_per_frame[frame_idx][track_id] = box

    if not any(len(v) > 0 for v in boxes_per_frame):
        return None
    return boxes_per_frame


def load_bbox_sequence(track_npy_path: Path, frame_w: int, frame_h: int) -> List[Dict[int, np.ndarray]]:
    try:
        data = np.load(track_npy_path, allow_pickle=True)
    except Exception:
        return []

    # First try the known track-map structure used in *_tracks.npy.
    obj = data.item() if isinstance(data, np.ndarray) and data.dtype == object and data.shape == () else data
    from_track_dict = _load_bbox_from_track_dict(obj, frame_w, frame_h)
    if from_track_dict is not None:
        return from_track_dict

    candidates = _extract_numeric_box_candidates(data)
    if not candidates:
        return []
    best = max(candidates, key=lambda a: int(a.shape[0]) if a.ndim >= 2 else -1)
    try:
        one_box_seq = _candidate_to_frame_boxes(best, frame_w, frame_h)
        # Generic fallback has no per-hand id; use key=-1 as "unknown hand".
        return [{-1: b} if b is not None else {} for b in one_box_seq]
    except Exception:
        return []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4: render hand mesh overlays from pose3d + raw video.")
    p.add_argument("--dataset", required=True, help="Dataset key, e.g. h2o / h2o-oc")
    p.add_argument("--algorithm", required=True, help="Algorithm key matching [algorithm.<name>]")
    p.add_argument("--config", required=True, help="Path to pipeline_config.ini")
    p.add_argument(
        "--dataset_override",
        action="append",
        default=[],
        help="Override dataset.<name> config in-memory, format: key=value (repeatable).",
    )
    p.add_argument(
        "--run-tag",
        default="",
        dest="run_tag",
        help="Optional run id: must match Step2/3; adds <run-tag>/ under pred and visualization roots.",
    )
    return p.parse_args()


def require_section(cfg: configparser.ConfigParser, name: str) -> configparser.SectionProxy:
    if name not in cfg:
        raise KeyError(f"Missing section in config: [{name}]")
    return cfg[name]


def apply_dataset_overrides(
    cfg: configparser.ConfigParser, dataset_name: str, overrides: list[str]
) -> None:
    if not overrides:
        return
    sec_name = f"dataset.{dataset_name}"
    if sec_name not in cfg:
        raise KeyError(f"Missing section in config: [{sec_name}]")
    sec = cfg[sec_name]
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid --dataset_override '{raw}', expected key=value")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --dataset_override '{raw}', empty key")
        sec[key] = value.strip()


def resolve_dataset_section(cfg: configparser.ConfigParser, dataset_name: str) -> configparser.SectionProxy:
    return require_section(cfg, f"dataset.{dataset_name}")


def resolve_algorithm_section(cfg: configparser.ConfigParser, algorithm_name: str) -> configparser.SectionProxy:
    return require_section(cfg, f"algorithm.{algorithm_name}")


def normalize_dataset_name(dataset_name: str) -> str:
    aliases = {"h2o-half_lf": "h2o-half_lr"}
    return aliases.get(dataset_name, dataset_name)


def canonicalize_dataset_for_pose_output(dataset_name: str) -> str:
    if dataset_name in {"h2o-oc", "h2o-noc"}:
        return "h2o"
    return dataset_name


def _sanitize_result_tag(tag: str) -> str:
    return (
        tag.strip()
        .replace(" ", "")
        .replace(".", "p")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def resolve_dataset_output_name(cfg: configparser.ConfigParser, dataset_name: str) -> str:
    base = canonicalize_dataset_for_pose_output(dataset_name)
    sec_name = f"dataset.{dataset_name}"
    if sec_name not in cfg:
        return base
    sec = cfg[sec_name]
    raw_tag = sec.get("result_tag", "").strip()
    if not raw_tag:
        return base
    if raw_tag.lower() == "auto":
        if "scale_min" in sec and "scale_max" in sec:
            raw_tag = f"s{sec['scale_min']}_{sec['scale_max']}"
        else:
            return base
    tag = _sanitize_result_tag(raw_tag)
    if not tag:
        return base
    return f"{base}__{tag}"


def resolve_pred_dir(base_pred_dir: Path, dataset_name: str, algorithm_name: str) -> Path:
    return base_pred_dir / dataset_name / algorithm_name


def _sanitize_config_scope(tag: str) -> str:
    return (
        tag.strip()
        .replace(" ", "")
        .replace(".", "p")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _is_config_scope_enabled(cfg: configparser.ConfigParser) -> bool:
    if not cfg.has_section("common"):
        return False
    return cfg.getboolean("common", "config_scope_enabled", fallback=False)


def _get_config_scope(cfg: configparser.ConfigParser) -> str:
    if cfg.has_section("common"):
        raw = cfg["common"].get("config_stem", "").strip()
        if raw:
            tag = _sanitize_config_scope(raw)
            if tag:
                return tag
    return "config"


def _apply_config_scope_to_dir(cfg: configparser.ConfigParser, base_dir: Path) -> Path:
    if not _is_config_scope_enabled(cfg):
        return base_dir
    return base_dir / _get_config_scope(cfg)


def ensure_exists(path_str: str, desc: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"{desc} not found: {p}")
    return p


def collect_videos(video_dir: Path, video_glob: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for video_path in sorted(video_dir.glob(video_glob)):
        if not video_path.is_file():
            continue
        stem = video_path.stem
        out.append((stem, video_path))
    return out


def _commit_video_tmp_file(tmp_path: Path, final_path: Path) -> None:
    """Finalize temp video file with robust fallback for network/mounted FS."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(str(tmp_path), str(final_path))
        return
    except OSError:
        # Some mounted/network filesystems may reject/flake on rename semantics.
        pass

    copied = False
    try:
        with open(tmp_path, "rb") as src, open(final_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        copied = True
    finally:
        if copied:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _make_temp_output_path(output_video: Path) -> Path:
    """
    Prefer local temp filesystem for ffmpeg streaming writes.
    Mounted/network paths can break pipe during trailer/finalization.
    """
    candidates: List[Path] = []
    env_tmp = os.environ.get("TMPDIR", "").strip()
    if env_tmp:
        candidates.append(Path(env_tmp))
    candidates.extend([Path("/tmp"), Path("/var/tmp"), output_video.parent])

    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        try:
            fd, tmp_out_str = tempfile.mkstemp(
                suffix=".tmp.mp4",
                prefix=f".{output_video.stem}.",
                dir=str(root),
            )
            os.close(fd)
            return Path(tmp_out_str)
        except Exception:
            continue

    raise RuntimeError(f"Failed to create temp video file for {output_video}")


def _normalize_fps(fps_value: Any, fallback: float = 30.0) -> float:
    try:
        fps = float(fps_value)
    except Exception:
        return float(fallback)
    if not np.isfinite(fps) or fps <= 0:
        return float(fallback)
    return fps


def _resolve_writer_fps(source_fps: Any, stride: int, fallback: float = 30.0) -> float:
    base_fps = _normalize_fps(source_fps, fallback=fallback)
    stride_safe = max(1, int(stride))
    out_fps = base_fps / float(stride_safe)
    # Keep writer fps strictly positive for unusual metadata or very large stride.
    return max(out_fps, 1e-6)


def _safe_scalar_float(value: Any, fallback: float) -> float:
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() <= 0:
                return float(fallback)
            value = value.detach().cpu().reshape(-1)[0].item()
        elif isinstance(value, np.ndarray):
            if value.size <= 0:
                return float(fallback)
            value = value.reshape(-1)[0].item()
        elif isinstance(value, (list, tuple)):
            if len(value) <= 0:
                return float(fallback)
            value = value[0]
        return float(value)
    except Exception:
        return float(fallback)


def _extract_pose_fps_from_clip(clip: Any, fallback: float = 30.0) -> float:
    if not isinstance(clip, dict):
        return float(fallback)
    return _normalize_fps(_safe_scalar_float(clip.get("fps", fallback), fallback), fallback=fallback)


def _build_time_aligned_index_map(
    source_len: int,
    source_fps: Any,
    target_len: int,
    target_fps: Any,
) -> np.ndarray:
    src_len = max(0, int(source_len))
    tgt_len = max(0, int(target_len))
    if src_len <= 0 or tgt_len <= 0:
        return np.zeros((0,), dtype=np.int64)

    src_fps = _normalize_fps(source_fps, fallback=30.0)
    tgt_fps = _normalize_fps(target_fps, fallback=30.0)
    ratio = src_fps / max(tgt_fps, 1e-9)
    target_idx = np.arange(tgt_len, dtype=np.float64)
    mapped = np.rint(target_idx * ratio).astype(np.int64)
    return np.clip(mapped, 0, src_len - 1)


def resolve_prediction_dir(
    cfg: configparser.ConfigParser,
    dataset_name: str,
    algorithm_name: str,
    run_tag: str = "",
) -> Tuple[Path, Path]:
    """Return (pred_dir_used_for_glob, pred_scoped_dir) following run_evaluation layout rules."""
    algo_sec = resolve_algorithm_section(cfg, algorithm_name)
    pred_base_dir = ensure_exists(algo_sec["pose3d_result"], "pose3d_result")
    pred_dataset = resolve_dataset_output_name(cfg, dataset_name)
    rt = normalize_run_tag(run_tag)
    pred_leaf = apply_run_tag_to_dir(
        resolve_pred_dir(pred_base_dir, pred_dataset, algorithm_name),
        rt,
    )
    pred_scoped_dir = _apply_config_scope_to_dir(cfg, pred_leaf)
    scoped_files = list(pred_scoped_dir.glob("*.pose3d_hand")) if pred_scoped_dir.exists() else []
    legacy_files = list(pred_base_dir.glob("*.pose3d_hand"))
    if (not _is_config_scope_enabled(cfg)) and ((not pred_scoped_dir.exists()) or (not scoped_files and legacy_files)):
        print(
            f"[WARN] Scoped pred dir unavailable/empty: {pred_scoped_dir}. "
            f"Using legacy pred dir: {pred_base_dir}"
        )
        return pred_base_dir, pred_scoped_dir
    return pred_scoped_dir, pred_scoped_dir


def _load_pose_clip_raw(data_path: Path) -> Dict[str, Any]:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "left_hand" in data:
        return data
    if isinstance(data, dict) and len(data) == 1:
        only_v = next(iter(data.values()))
        if isinstance(only_v, dict):
            return only_v
    if isinstance(data, dict):
        return data
    raise TypeError(f"Unsupported pose file format: {data_path}")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_traj_xyz_quat(pose_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    clip = _load_pose_clip_raw(pose_path)
    clip_fps = _extract_pose_fps_from_clip(clip, fallback=30.0)
    slam = clip.get("slam_data", {})
    if not isinstance(slam, dict):
        raise KeyError("Missing slam_data")
    traj_v = slam.get("traj", None)
    if traj_v is None:
        raise KeyError("Missing slam_data.traj")
    traj = _to_numpy(traj_v).astype(np.float64, copy=False)
    if traj.ndim != 2 or traj.shape[1] < 7:
        raise ValueError(f"Invalid traj shape: {traj.shape}")
    traj = traj[:, :7].copy()
    scale_v = slam.get("scale", 1.0)
    scale_arr = _to_numpy(scale_v).astype(np.float64, copy=False).reshape(-1)
    if scale_arr.size == 0:
        scale_arr = np.ones((traj.shape[0],), dtype=np.float64)
    elif scale_arr.size == 1:
        scale_arr = np.full((traj.shape[0],), float(scale_arr[0]), dtype=np.float64)
    elif scale_arr.size != traj.shape[0]:
        scale_arr = np.full((traj.shape[0],), float(scale_arr[0]), dtype=np.float64)
    xyz = traj[:, :3] * scale_arr[:, None]
    quat = traj[:, 3:7].astype(np.float64, copy=False)
    # Normalize quaternions for robust direction decoding.
    n = np.linalg.norm(quat, axis=1, keepdims=True)
    n = np.where(n > 1e-12, n, 1.0)
    quat = quat / n
    return xyz, quat, clip_fps


def _quat_to_rotmat_np(quat_xyzw: np.ndarray) -> np.ndarray:
    q = quat_xyzw.astype(np.float64, copy=False)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x**2 + y**2),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    return r


def _umeyama_similarity_np(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source_xyz.shape != target_xyz.shape:
        raise ValueError("source/target shape mismatch")
    n = source_xyz.shape[0]
    if n < 3:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64)
    src_mean = source_xyz.mean(axis=0)
    tgt_mean = target_xyz.mean(axis=0)
    src_c = source_xyz - src_mean
    tgt_c = target_xyz - tgt_mean
    cov = (tgt_c.T @ src_c) / float(n)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1.0
    r = u @ s @ vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = 1.0 if var_src <= 1e-12 else float(np.trace(np.diag(d) @ s) / var_src)
    t = tgt_mean - scale * (r @ src_mean)
    return scale, r, t


def _translation_only_align_np(source_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """Align source to target using translation only (no rotation, no scaling)."""
    if source_xyz.shape != target_xyz.shape:
        raise ValueError("source/target shape mismatch")
    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("source/target must be [N,3]")
    if source_xyz.shape[0] == 0:
        return source_xyz.copy()
    shift = target_xyz.mean(axis=0) - source_xyz.mean(axis=0)
    return source_xyz + shift


def _build_world_to_view_matrix(yaw_deg: float = -35.0, pitch_deg: float = 25.0) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pit = math.radians(float(pitch_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pit), math.sin(pit)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    return rx @ ry


def _project_world_points(points_xyz: np.ndarray, world_to_view: np.ndarray) -> np.ndarray:
    view = (world_to_view @ points_xyz.T).T
    return view[:, :2]


def _draw_line_np(frame: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int], thickness: int) -> None:
    try:
        import cv2  # type: ignore

        cv2.line(frame, p0, p1, color, max(1, thickness), lineType=cv2.LINE_AA)
        return
    except ImportError:
        pass
    h, w = frame.shape[:2]
    x0, y0 = p0
    x1, y1 = p1
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, n + 1).astype(np.int32)
    ys = np.linspace(y0, y1, n + 1).astype(np.int32)
    for x, y in zip(xs, ys):
        if 0 <= x < w and 0 <= y < h:
            frame[y, x] = np.asarray(color, dtype=np.uint8)


def _draw_text_np(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    try:
        import cv2  # type: ignore

        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
        return
    except ImportError:
        pass


def _select_plane(points_xyz: np.ndarray, plane: str) -> np.ndarray:
    if plane == "xoy":
        return points_xyz[:, [0, 1]]
    if plane == "xoz":
        return points_xyz[:, [0, 2]]
    if plane == "yoz":
        return points_xyz[:, [1, 2]]
    raise ValueError(f"Unsupported plane: {plane}")


def _compute_plane_bounds(points_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    center = (mins + maxs) * 0.5
    span = np.maximum(maxs - mins, 1e-6)
    span = span * 1.15
    mins = center - span * 0.5
    maxs = center + span * 0.5
    return mins, maxs


def _world2px(
    p2: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    width: int,
    height: int,
    pad: int,
) -> tuple[int, int]:
    span = np.maximum(maxs - mins, 1e-6)
    sx = (width - 2 * pad) / float(span[0])
    sy = (height - 2 * pad) / float(span[1])
    s = min(sx, sy)
    content_w = span[0] * s
    content_h = span[1] * s
    ox = pad + (width - 2 * pad - content_w) * 0.5
    oy = pad + (height - 2 * pad - content_h) * 0.5
    x = ox + (float(p2[0]) - float(mins[0])) * s
    y = oy + (float(p2[1]) - float(mins[1])) * s
    return int(round(x)), int(round(height - 1 - y))


def _draw_slam_plane_panel(
    *,
    panel_h: int,
    panel_w: int,
    gt_hist_2d: np.ndarray,
    pred_hist_2d: np.ndarray,
    gt_dir_2d: np.ndarray,
    pred_dir_2d: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    title: str,
    axis_label: str,
) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    pad = 28

    # Light grid for readability.
    grid_color = (45, 45, 45)
    for r in range(1, 4):
        y = int(round(panel_h * r / 4))
        _draw_line_np(panel, (0, y), (panel_w - 1, y), grid_color, 1)
    for c in range(1, 4):
        x = int(round(panel_w * c / 4))
        _draw_line_np(panel, (x, 0), (x, panel_h - 1), grid_color, 1)

    gt_px = [_world2px(p, mins, maxs, panel_w, panel_h, pad) for p in gt_hist_2d]
    pred_px = [_world2px(p, mins, maxs, panel_w, panel_h, pad) for p in pred_hist_2d]
    for i in range(1, len(gt_px)):
        _draw_line_np(panel, gt_px[i - 1], gt_px[i], (0, 255, 255), 2)
    for i in range(1, len(pred_px)):
        _draw_line_np(panel, pred_px[i - 1], pred_px[i], (255, 255, 0), 2)

    # Draw fixed local axes on this projection from world origin.
    axis_world_len = float(max(maxs - mins) * 0.22)
    o = _world2px(np.asarray([0.0, 0.0], dtype=np.float64), mins, maxs, panel_w, panel_h, pad)
    x_end = _world2px(np.asarray([axis_world_len, 0.0], dtype=np.float64), mins, maxs, panel_w, panel_h, pad)
    y_end = _world2px(np.asarray([0.0, axis_world_len], dtype=np.float64), mins, maxs, panel_w, panel_h, pad)
    _draw_line_np(panel, o, x_end, (0, 0, 255), 2)
    _draw_line_np(panel, o, y_end, (0, 255, 0), 2)

    # Pose arrows from current camera position.
    def _arrow_tip(cur_hist: np.ndarray, cur_dir: np.ndarray) -> tuple[int, int]:
        d_norm = float(np.linalg.norm(cur_dir))
        if d_norm <= 1e-9:
            return _world2px(cur_hist[-1], mins, maxs, panel_w, panel_h, pad)
        d = cur_dir / d_norm
        tip_world = cur_hist[-1] + d * (max(maxs - mins) * 0.08)
        return _world2px(tip_world, mins, maxs, panel_w, panel_h, pad)

    gt_cur = gt_px[-1]
    pred_cur = pred_px[-1]
    gt_tip = _arrow_tip(gt_hist_2d, gt_dir_2d)
    pred_tip = _arrow_tip(pred_hist_2d, pred_dir_2d)
    _draw_line_np(panel, gt_cur, gt_tip, (0, 255, 255), 3)
    _draw_line_np(panel, pred_cur, pred_tip, (255, 255, 0), 3)

    _draw_text_np(panel, title, (12, 24), (225, 225, 225))
    _draw_text_np(panel, "GT(cyan) / Pred(yellow)", (12, panel_h - 28), (190, 190, 190))
    _draw_text_np(panel, axis_label, (12, panel_h - 10), (160, 220, 160))
    return panel


def _build_slam_plane_panel_base(
    *,
    panel_h: int,
    panel_w: int,
    mins: np.ndarray,
    maxs: np.ndarray,
    title: str,
    axis_label: str,
) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    pad = 28
    grid_color = (45, 45, 45)
    for r in range(1, 4):
        y = int(round(panel_h * r / 4))
        _draw_line_np(panel, (0, y), (panel_w - 1, y), grid_color, 1)
    for c in range(1, 4):
        x = int(round(panel_w * c / 4))
        _draw_line_np(panel, (x, 0), (x, panel_h - 1), grid_color, 1)

    axis_world_len = float(max(maxs - mins) * 0.22)
    o = _world2px(np.asarray([0.0, 0.0], dtype=np.float64), mins, maxs, panel_w, panel_h, pad)
    x_end = _world2px(np.asarray([axis_world_len, 0.0], dtype=np.float64), mins, maxs, panel_w, panel_h, pad)
    y_end = _world2px(np.asarray([0.0, axis_world_len], dtype=np.float64), mins, maxs, panel_w, panel_h, pad)
    _draw_line_np(panel, o, x_end, (0, 0, 255), 2)
    _draw_line_np(panel, o, y_end, (0, 255, 0), 2)
    _draw_text_np(panel, title, (12, 24), (225, 225, 225))
    _draw_text_np(panel, "GT(cyan) / Pred(yellow)", (12, panel_h - 28), (190, 190, 190))
    _draw_text_np(panel, axis_label, (12, panel_h - 10), (160, 220, 160))
    return panel


def _arrow_tip_px(
    cur_hist_2d: np.ndarray,
    cur_dir_2d: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    panel_w: int,
    panel_h: int,
    pad: int = 28,
) -> tuple[int, int]:
    d_norm = float(np.linalg.norm(cur_dir_2d))
    if d_norm <= 1e-9:
        return _world2px(cur_hist_2d[-1], mins, maxs, panel_w, panel_h, pad)
    d = cur_dir_2d / d_norm
    tip_world = cur_hist_2d[-1] + d * (max(maxs - mins) * 0.08)
    return _world2px(tip_world, mins, maxs, panel_w, panel_h, pad)


def render_slam_camera_pose_video(
    *,
    video_path: Path,
    pred_pose_path: Path,
    gt_pose_path: Path,
    output_video: Path,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> None:
    pred_xyz_raw, pred_q_raw, pred_pose_fps = _extract_traj_xyz_quat(pred_pose_path)
    gt_xyz_raw, gt_q, gt_pose_fps = _extract_traj_xyz_quat(gt_pose_path)
    pred_len = min(pred_xyz_raw.shape[0], pred_q_raw.shape[0])
    gt_len = min(gt_xyz_raw.shape[0], gt_q.shape[0])
    if pred_len <= 1 or gt_len <= 1:
        raise RuntimeError(f"Not enough SLAM frames for {video_path.name}")
    pred_xyz_raw = pred_xyz_raw[:pred_len]
    gt_xyz = gt_xyz_raw[:gt_len]
    pred_q = pred_q_raw[:pred_len]
    gt_q = gt_q[:gt_len]

    # Display-only recentering: make each trajectory start from origin.
    gt_xyz_vis = gt_xyz - gt_xyz[0:1]
    pred_xyz_vis = pred_xyz_raw - pred_xyz_raw[0:1]

    pred_r_w2c = _quat_to_rotmat_np(pred_q)
    gt_r_w2c = _quat_to_rotmat_np(gt_q)
    pred_r_c2w = np.transpose(pred_r_w2c, (0, 2, 1))
    gt_r_c2w = np.transpose(gt_r_w2c, (0, 2, 1))
    # Full pose alignment (rotation part): find one constant world rotation that
    # makes first-frame predicted camera orientation match GT first frame.
    # With recentering above, this yields first-frame pose match at origin.
    pred_to_gt_r = gt_r_c2w[0] @ pred_r_c2w[0].T
    pred_xyz_vis = (pred_to_gt_r @ pred_xyz_vis.T).T
    pred_r_c2w = np.einsum("ij,njk->nik", pred_to_gt_r, pred_r_c2w)
    f_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pred_fwd = np.einsum("nij,j->ni", pred_r_c2w, f_cam)
    gt_fwd = np.einsum("nij,j->ni", gt_r_c2w, f_cam)
    pred_fwd_norm = np.linalg.norm(pred_fwd, axis=1, keepdims=True)
    pred_fwd = pred_fwd / np.where(pred_fwd_norm > 1e-12, pred_fwd_norm, 1.0)
    gt_fwd_norm = np.linalg.norm(gt_fwd, axis=1, keepdims=True)
    gt_fwd = gt_fwd / np.where(gt_fwd_norm > 1e-12, gt_fwd_norm, 1.0)

    # Plane-projection visualization principle:
    # XOY / XOZ / YOZ are rendered with fixed whole-trajectory bounds.
    gt_xoy_all = _select_plane(gt_xyz_vis, "xoy")
    pred_xoy_all = _select_plane(pred_xyz_vis, "xoy")
    gt_xoz_all = _select_plane(gt_xyz_vis, "xoz")
    pred_xoz_all = _select_plane(pred_xyz_vis, "xoz")
    gt_yoz_all = _select_plane(gt_xyz_vis, "yoz")
    pred_yoz_all = _select_plane(pred_xyz_vis, "yoz")
    xoy_bounds = _compute_plane_bounds(np.concatenate([gt_xoy_all, pred_xoy_all], axis=0))
    xoz_bounds = _compute_plane_bounds(np.concatenate([gt_xoz_all, pred_xoz_all], axis=0))
    yoz_bounds = _compute_plane_bounds(np.concatenate([gt_yoz_all, pred_yoz_all], axis=0))

    reader = None
    cap = None
    fps = 30.0
    left_w = 0
    left_h = 0
    video_len = None
    use_cv2_reader = False
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap is not None and cap.isOpened():
            use_cv2_reader = True
            fps = _normalize_fps(cap.get(cv2.CAP_PROP_FPS), fallback=fps)
            left_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            left_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                video_len = frame_count
    except Exception:
        use_cv2_reader = False

    if not use_cv2_reader:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
        reader = imageio.get_reader(str(video_path))
        meta = reader.get_meta_data()
        fps = _normalize_fps(meta.get("fps", 30), fallback=fps)
        video_size = meta.get("size", None)
        if video_size is None:
            first = reader.get_data(0)
            left_h, left_w = first.shape[:2]
        else:
            left_w, left_h = int(video_size[0]), int(video_size[1])

    num_frames = min(pred_len, gt_len)
    if video_len is None and reader is not None:
        try:
            vid_len = reader.get_length()
            if isinstance(vid_len, (int, float, np.integer, np.floating)) and np.isfinite(vid_len) and vid_len > 0:
                video_len = int(vid_len)
        except Exception:
            video_len = None
    if isinstance(video_len, (int, float, np.integer, np.floating)) and np.isfinite(video_len) and video_len > 0:
        num_frames = int(video_len)
    if max_frames is not None and max_frames > 0:
        num_frames = min(num_frames, int(max_frames))
    stride = max(1, int(stride))
    if num_frames <= 0:
        raise RuntimeError(f"No frames available for {video_path.name}")

    pred_idx_map = _build_time_aligned_index_map(pred_len, pred_pose_fps, num_frames, fps)
    gt_idx_map = _build_time_aligned_index_map(gt_len, gt_pose_fps, num_frames, fps)
    if abs(float(pred_pose_fps) - float(fps)) > 1e-3 or abs(float(gt_pose_fps) - float(fps)) > 1e-3:
        print(
            f"[INFO] SLAM time-aligned sampling: video_fps={fps:.4f}, "
            f"pred_pose_fps={pred_pose_fps:.4f}, gt_pose_fps={gt_pose_fps:.4f}"
        )

    out_h, out_w = left_h, left_w
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_video = _make_temp_output_path(output_video)
    writer = imageio.get_writer(
        str(tmp_output_video),
        fps=_resolve_writer_fps(fps, stride),
        codec="libx264",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-movflags", "frag_keyframe+empty_moov+default_base_moof"],
    )

    top_h = left_h // 2
    bot_h = left_h - top_h
    left_w_half = left_w // 2
    right_w = left_w - left_w_half
    xoy_gt_px_all = np.asarray(
        [_world2px(p, xoy_bounds[0], xoy_bounds[1], right_w, top_h, 28) for p in gt_xoy_all],
        dtype=np.int32,
    )
    xoy_pred_px_all = np.asarray(
        [_world2px(p, xoy_bounds[0], xoy_bounds[1], right_w, top_h, 28) for p in pred_xoy_all],
        dtype=np.int32,
    )
    xoz_gt_px_all = np.asarray(
        [_world2px(p, xoz_bounds[0], xoz_bounds[1], left_w_half, bot_h, 28) for p in gt_xoz_all],
        dtype=np.int32,
    )
    xoz_pred_px_all = np.asarray(
        [_world2px(p, xoz_bounds[0], xoz_bounds[1], left_w_half, bot_h, 28) for p in pred_xoz_all],
        dtype=np.int32,
    )
    yoz_gt_px_all = np.asarray(
        [_world2px(p, yoz_bounds[0], yoz_bounds[1], right_w, bot_h, 28) for p in gt_yoz_all],
        dtype=np.int32,
    )
    yoz_pred_px_all = np.asarray(
        [_world2px(p, yoz_bounds[0], yoz_bounds[1], right_w, bot_h, 28) for p in pred_yoz_all],
        dtype=np.int32,
    )
    xoy_gt_px_by_video = xoy_gt_px_all[gt_idx_map]
    xoy_pred_px_by_video = xoy_pred_px_all[pred_idx_map]
    xoz_gt_px_by_video = xoz_gt_px_all[gt_idx_map]
    xoz_pred_px_by_video = xoz_pred_px_all[pred_idx_map]
    yoz_gt_px_by_video = yoz_gt_px_all[gt_idx_map]
    yoz_pred_px_by_video = yoz_pred_px_all[pred_idx_map]
    gt_xoy_by_video = gt_xoy_all[gt_idx_map]
    pred_xoy_by_video = pred_xoy_all[pred_idx_map]
    gt_xoz_by_video = gt_xoz_all[gt_idx_map]
    pred_xoz_by_video = pred_xoz_all[pred_idx_map]
    gt_yoz_by_video = gt_yoz_all[gt_idx_map]
    pred_yoz_by_video = pred_yoz_all[pred_idx_map]
    gt_fwd_by_video = gt_fwd[gt_idx_map]
    pred_fwd_by_video = pred_fwd[pred_idx_map]

    panel_xoy_base = _build_slam_plane_panel_base(
        panel_h=top_h,
        panel_w=right_w,
        mins=xoy_bounds[0],
        maxs=xoy_bounds[1],
        title="SLAM XOY",
        axis_label="Horizontal: X, Vertical: Y",
    )
    panel_xoz_base = _build_slam_plane_panel_base(
        panel_h=bot_h,
        panel_w=left_w_half,
        mins=xoz_bounds[0],
        maxs=xoz_bounds[1],
        title="SLAM XOZ",
        axis_label="Horizontal: X, Vertical: Z",
    )
    panel_yoz_base = _build_slam_plane_panel_base(
        panel_h=bot_h,
        panel_w=right_w,
        mins=yoz_bounds[0],
        maxs=yoz_bounds[1],
        title="SLAM YOZ",
        axis_label="Horizontal: Y, Vertical: Z",
    )
    panel_xoy_trail = panel_xoy_base.copy()
    panel_xoz_trail = panel_xoz_base.copy()
    panel_yoz_trail = panel_yoz_base.copy()

    render_ok = False
    try:
        if use_cv2_reader and cap is not None:
            import cv2  # type: ignore

            frame_idx = -1
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx >= num_frames:
                    break
                if frame_idx % stride != 0:
                    continue
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame.shape[0] != left_h or frame.shape[1] != left_w:
                    frame = _resize_frame_bgr(frame, left_w, left_h)
                if frame_idx > 0:
                    _draw_line_np(
                        panel_xoy_trail,
                        tuple(xoy_gt_px_by_video[frame_idx - 1]),
                        tuple(xoy_gt_px_by_video[frame_idx]),
                        (0, 255, 255),
                        2,
                    )
                    _draw_line_np(
                        panel_xoy_trail,
                        tuple(xoy_pred_px_by_video[frame_idx - 1]),
                        tuple(xoy_pred_px_by_video[frame_idx]),
                        (255, 255, 0),
                        2,
                    )
                    _draw_line_np(
                        panel_xoz_trail,
                        tuple(xoz_gt_px_by_video[frame_idx - 1]),
                        tuple(xoz_gt_px_by_video[frame_idx]),
                        (0, 255, 255),
                        2,
                    )
                    _draw_line_np(
                        panel_xoz_trail,
                        tuple(xoz_pred_px_by_video[frame_idx - 1]),
                        tuple(xoz_pred_px_by_video[frame_idx]),
                        (255, 255, 0),
                        2,
                    )
                    _draw_line_np(
                        panel_yoz_trail,
                        tuple(yoz_gt_px_by_video[frame_idx - 1]),
                        tuple(yoz_gt_px_by_video[frame_idx]),
                        (0, 255, 255),
                        2,
                    )
                    _draw_line_np(
                        panel_yoz_trail,
                        tuple(yoz_pred_px_by_video[frame_idx - 1]),
                        tuple(yoz_pred_px_by_video[frame_idx]),
                        (255, 255, 0),
                        2,
                    )

                gt_dir_xoy = gt_fwd_by_video[frame_idx, [0, 1]]
                pred_dir_xoy = pred_fwd_by_video[frame_idx, [0, 1]]
                gt_dir_xoz = gt_fwd_by_video[frame_idx, [0, 2]]
                pred_dir_xoz = pred_fwd_by_video[frame_idx, [0, 2]]
                gt_dir_yoz = gt_fwd_by_video[frame_idx, [1, 2]]
                pred_dir_yoz = pred_fwd_by_video[frame_idx, [1, 2]]
                panel_xoy = panel_xoy_trail.copy()
                panel_xoz = panel_xoz_trail.copy()
                panel_yoz = panel_yoz_trail.copy()
                _draw_line_np(
                    panel_xoy,
                    tuple(xoy_gt_px_by_video[frame_idx]),
                    _arrow_tip_px(gt_xoy_by_video[: frame_idx + 1], gt_dir_xoy, xoy_bounds[0], xoy_bounds[1], right_w, top_h),
                    (0, 255, 255),
                    3,
                )
                _draw_line_np(
                    panel_xoy,
                    tuple(xoy_pred_px_by_video[frame_idx]),
                    _arrow_tip_px(pred_xoy_by_video[: frame_idx + 1], pred_dir_xoy, xoy_bounds[0], xoy_bounds[1], right_w, top_h),
                    (255, 255, 0),
                    3,
                )
                _draw_line_np(
                    panel_xoz,
                    tuple(xoz_gt_px_by_video[frame_idx]),
                    _arrow_tip_px(gt_xoz_by_video[: frame_idx + 1], gt_dir_xoz, xoz_bounds[0], xoz_bounds[1], left_w_half, bot_h),
                    (0, 255, 255),
                    3,
                )
                _draw_line_np(
                    panel_xoz,
                    tuple(xoz_pred_px_by_video[frame_idx]),
                    _arrow_tip_px(pred_xoz_by_video[: frame_idx + 1], pred_dir_xoz, xoz_bounds[0], xoz_bounds[1], left_w_half, bot_h),
                    (255, 255, 0),
                    3,
                )
                _draw_line_np(
                    panel_yoz,
                    tuple(yoz_gt_px_by_video[frame_idx]),
                    _arrow_tip_px(gt_yoz_by_video[: frame_idx + 1], gt_dir_yoz, yoz_bounds[0], yoz_bounds[1], right_w, bot_h),
                    (0, 255, 255),
                    3,
                )
                _draw_line_np(
                    panel_yoz,
                    tuple(yoz_pred_px_by_video[frame_idx]),
                    _arrow_tip_px(pred_yoz_by_video[: frame_idx + 1], pred_dir_yoz, yoz_bounds[0], yoz_bounds[1], right_w, bot_h),
                    (255, 255, 0),
                    3,
                )

                video_panel = _resize_frame_bgr(frame, left_w_half, top_h)
                _draw_text_np(video_panel, f"Frame: {frame_idx + 1}/{num_frames}", (12, 24), (40, 230, 40))

                composed = np.zeros((left_h, left_w, 3), dtype=np.uint8)
                composed[:top_h, :left_w_half, :] = video_panel
                composed[:top_h, left_w_half:, :] = panel_xoy
                composed[top_h:, :left_w_half, :] = panel_xoz
                composed[top_h:, left_w_half:, :] = panel_yoz
                if composed.shape[0] != out_h or composed.shape[1] != out_w:
                    composed = _resize_frame_bgr(composed, out_w, out_h)
                writer.append_data(composed)
        else:
            assert reader is not None
            for frame_idx, frame in enumerate(reader):
                if frame_idx >= num_frames:
                    break
                if frame_idx % stride != 0:
                    continue
                if frame.shape[0] != left_h or frame.shape[1] != left_w:
                    frame = _resize_frame_bgr(frame, left_w, left_h)
                if frame_idx > 0:
                    _draw_line_np(
                        panel_xoy_trail,
                        tuple(xoy_gt_px_by_video[frame_idx - 1]),
                        tuple(xoy_gt_px_by_video[frame_idx]),
                        (0, 255, 255),
                        2,
                    )
                    _draw_line_np(
                        panel_xoy_trail,
                        tuple(xoy_pred_px_by_video[frame_idx - 1]),
                        tuple(xoy_pred_px_by_video[frame_idx]),
                        (255, 255, 0),
                        2,
                    )
                    _draw_line_np(
                        panel_xoz_trail,
                        tuple(xoz_gt_px_by_video[frame_idx - 1]),
                        tuple(xoz_gt_px_by_video[frame_idx]),
                        (0, 255, 255),
                        2,
                    )
                    _draw_line_np(
                        panel_xoz_trail,
                        tuple(xoz_pred_px_by_video[frame_idx - 1]),
                        tuple(xoz_pred_px_by_video[frame_idx]),
                        (255, 255, 0),
                        2,
                    )
                    _draw_line_np(
                        panel_yoz_trail,
                        tuple(yoz_gt_px_by_video[frame_idx - 1]),
                        tuple(yoz_gt_px_by_video[frame_idx]),
                        (0, 255, 255),
                        2,
                    )
                    _draw_line_np(
                        panel_yoz_trail,
                        tuple(yoz_pred_px_by_video[frame_idx - 1]),
                        tuple(yoz_pred_px_by_video[frame_idx]),
                        (255, 255, 0),
                        2,
                    )

                gt_dir_xoy = gt_fwd_by_video[frame_idx, [0, 1]]
                pred_dir_xoy = pred_fwd_by_video[frame_idx, [0, 1]]
                gt_dir_xoz = gt_fwd_by_video[frame_idx, [0, 2]]
                pred_dir_xoz = pred_fwd_by_video[frame_idx, [0, 2]]
                gt_dir_yoz = gt_fwd_by_video[frame_idx, [1, 2]]
                pred_dir_yoz = pred_fwd_by_video[frame_idx, [1, 2]]
                panel_xoy = panel_xoy_trail.copy()
                panel_xoz = panel_xoz_trail.copy()
                panel_yoz = panel_yoz_trail.copy()
                _draw_line_np(
                    panel_xoy,
                    tuple(xoy_gt_px_by_video[frame_idx]),
                    _arrow_tip_px(gt_xoy_by_video[: frame_idx + 1], gt_dir_xoy, xoy_bounds[0], xoy_bounds[1], right_w, top_h),
                    (0, 255, 255),
                    3,
                )
                _draw_line_np(
                    panel_xoy,
                    tuple(xoy_pred_px_by_video[frame_idx]),
                    _arrow_tip_px(pred_xoy_by_video[: frame_idx + 1], pred_dir_xoy, xoy_bounds[0], xoy_bounds[1], right_w, top_h),
                    (255, 255, 0),
                    3,
                )
                _draw_line_np(
                    panel_xoz,
                    tuple(xoz_gt_px_by_video[frame_idx]),
                    _arrow_tip_px(gt_xoz_by_video[: frame_idx + 1], gt_dir_xoz, xoz_bounds[0], xoz_bounds[1], left_w_half, bot_h),
                    (0, 255, 255),
                    3,
                )
                _draw_line_np(
                    panel_xoz,
                    tuple(xoz_pred_px_by_video[frame_idx]),
                    _arrow_tip_px(pred_xoz_by_video[: frame_idx + 1], pred_dir_xoz, xoz_bounds[0], xoz_bounds[1], left_w_half, bot_h),
                    (255, 255, 0),
                    3,
                )
                _draw_line_np(
                    panel_yoz,
                    tuple(yoz_gt_px_by_video[frame_idx]),
                    _arrow_tip_px(gt_yoz_by_video[: frame_idx + 1], gt_dir_yoz, yoz_bounds[0], yoz_bounds[1], right_w, bot_h),
                    (0, 255, 255),
                    3,
                )
                _draw_line_np(
                    panel_yoz,
                    tuple(yoz_pred_px_by_video[frame_idx]),
                    _arrow_tip_px(pred_yoz_by_video[: frame_idx + 1], pred_dir_yoz, yoz_bounds[0], yoz_bounds[1], right_w, bot_h),
                    (255, 255, 0),
                    3,
                )

                video_panel = _resize_frame_bgr(frame, left_w_half, top_h)
                _draw_text_np(video_panel, f"Frame: {frame_idx + 1}/{num_frames}", (12, 24), (40, 230, 40))

                composed = np.zeros((left_h, left_w, 3), dtype=np.uint8)
                composed[:top_h, :left_w_half, :] = video_panel
                composed[:top_h, left_w_half:, :] = panel_xoy
                composed[top_h:, :left_w_half, :] = panel_xoz
                composed[top_h:, left_w_half:, :] = panel_yoz
                if composed.shape[0] != out_h or composed.shape[1] != out_w:
                    composed = _resize_frame_bgr(composed, out_w, out_h)
                writer.append_data(composed)
        render_ok = True
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        try:
            if reader is not None:
                reader.close()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
        if not render_ok:
            try:
                tmp_output_video.unlink()
            except OSError:
                pass
    _commit_video_tmp_file(tmp_output_video, output_video)


def select_device(device: Optional[str]) -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "mps":
        print("PyTorch3D does not support MPS. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def load_pose_data(data_path: Path, device: torch.device) -> Tuple[Dict, float]:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "left_hand" in data:
        clip = data
    elif isinstance(data, dict) and len(data) == 1:
        clip = next(iter(data.values()))
    else:
        clip = data
    clip_fps = _extract_pose_fps_from_clip(clip, fallback=30.0)
    return move_to_device(clip, device), clip_fps


def move_to_device(data: Dict, device: torch.device) -> Dict:
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
        elif isinstance(value, dict):
            data[key] = move_to_device(value, device)
        elif isinstance(value, np.ndarray):
            if device.type == "mps" and value.dtype == np.float64:
                value = value.astype(np.float32)
            data[key] = torch.from_numpy(value).to(device)
        elif isinstance(value, (int, float, list, tuple)):
            data[key] = torch.tensor(value, device=device)
    return data


def build_mano_model(
    model_dir: Path,
    side: str,
    device: torch.device,
    flat_hand_mean: bool = True,
    fix_shapedirs: bool = True,
) -> ManoLayer:
    model_dir = Path(model_dir)
    pkl_name = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"
    pkl_path = model_dir / pkl_name
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"MANO model not found: {pkl_path}\n"
            "Place MANO_LEFT.pkl and MANO_RIGHT.pkl in this directory."
        )
    model = ManoLayer(
        mano_root=str(model_dir),
        use_pca=False,
        flat_hand_mean=flat_hand_mean,
        side=side,
    ).to(device)
    if side == "left" and fix_shapedirs and hasattr(model, "th_shapedirs"):
        with torch.no_grad():
            model.th_shapedirs[:, 0, :] *= -1
    return model


def forward_mano(model: ManoLayer, params: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    n = params["transl"].shape[0]
    betas = params["betas"].view(n, 10)
    global_orient = params["global_orient"].view(n, 3)
    hand_pose = params["hand_pose"].view(n, 45)
    transl = params["transl"].view(n, 3)
    th_pose = torch.cat([global_orient, hand_pose], dim=1)
    verts, joints = model(th_pose, th_betas=betas, th_trans=transl)
    return verts / 1000.0, joints / 1000.0


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = torch.stack(
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qx * qw),
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx**2 + qy**2),
        ],
        dim=1,
    ).view(-1, 3, 3)
    return r


def world_to_camera(points: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
    t = traj[:, :3]
    q = traj[:, 3:]
    r = quat_to_rotmat(q)
    return torch.bmm(points - t.unsqueeze(1), r)


def build_intrinsics(img_focal: torch.Tensor, img_center: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [[img_focal, 0, img_center[0]], [0, img_focal, img_center[1]], [0, 0, 1]],
        device=img_focal.device,
        dtype=torch.float32,
    )


def scale_intrinsics(
    k: torch.Tensor, img_center: torch.Tensor, video_w: int, video_h: int
) -> torch.Tensor:
    base_w = float(img_center[0] * 2)
    base_h = float(img_center[1] * 2)
    sx = video_w / base_w if base_w > 1e-6 else 1.0
    sy = video_h / base_h if base_h > 1e-6 else 1.0
    k = k.clone()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] *= sx
    k[1, 2] *= sy
    return k.to(dtype=torch.float32)


def build_renderer(
    image_size: Tuple[int, int],
    device: torch.device,
    bin_size: Optional[int],
    max_faces_per_bin: Optional[int],
) -> MeshRenderer:
    if device.type != "cuda":
        bin_size = None
        max_faces_per_bin = None
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=1e-5,
        faces_per_pixel=1,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
    )
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    shader = SoftPhongShader(device=device)
    return MeshRenderer(rasterizer=rasterizer, shader=shader)


def create_cameras(
    k: torch.Tensor, image_size: Tuple[int, int], batch_size: int, device: torch.device
):
    r = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    t = torch.zeros((batch_size, 3), device=device)
    k_batch = k.unsqueeze(0).repeat(batch_size, 1, 1)
    image_size_t = torch.tensor([[image_size[0], image_size[1]]], device=device).repeat(batch_size, 1)
    return _cameras_from_opencv_projection(r, t, k_batch, image_size_t)


def build_mesh_batch(
    left_vertices: torch.Tensor,
    right_vertices: torch.Tensor,
    left_valid: torch.Tensor,
    right_valid: torch.Tensor,
    faces_left: torch.Tensor,
    faces_right: torch.Tensor,
    left_color: torch.Tensor,
    right_color: torch.Tensor,
):
    verts_list: List[torch.Tensor] = []
    faces_list: List[torch.Tensor] = []
    colors_list: List[torch.Tensor] = []

    for i in range(left_vertices.shape[0]):
        has_left = bool(left_valid[i].item())
        has_right = bool(right_valid[i].item())
        if (not has_left) and (not has_right):
            continue

        verts_parts: List[torch.Tensor] = []
        faces_parts: List[torch.Tensor] = []
        colors_parts: List[torch.Tensor] = []
        vert_offset = 0

        if has_left:
            v_left = left_vertices[i]
            verts_parts.append(v_left)
            faces_parts.append(faces_left + vert_offset)
            colors_parts.append(left_color.expand(v_left.shape[0], 3))
            vert_offset += int(v_left.shape[0])

        if has_right:
            v_right = right_vertices[i]
            verts_parts.append(v_right)
            faces_parts.append(faces_right + vert_offset)
            colors_parts.append(right_color.expand(v_right.shape[0], 3))

        v = torch.cat(verts_parts, dim=0)
        f = torch.cat(faces_parts, dim=0)
        c = torch.cat(colors_parts, dim=0)
        verts_list.append(v)
        faces_list.append(f)
        colors_list.append(c)

    textures = TexturesVertex(verts_features=colors_list)
    return Meshes(verts=verts_list, faces=faces_list, textures=textures)


def render_mesh_overlay_video(
    pose_path: Path,
    video_path: Path,
    output_video: Path,
    model_dir: Path,
    *,
    device: str = "auto",
    hand_space: str = "world",
    batch_size: int = 1,
    bin_size: Optional[int] = 128,
    max_faces_per_bin: Optional[int] = 20000,
    overlay_alpha: float = 0.75,
    max_frames: Optional[int] = None,
    stride: int = 1,
    downscale: float = 1.0,
    bbox_seq: Optional[List[Dict[int, np.ndarray]]] = None,
    draw_bbox: bool = True,
    bbox_color: Tuple[int, int, int] = (0, 255, 255),
    bbox_thickness: int = 2,
    render_runtime: Optional[Dict[str, Any]] = None,
) -> None:
    """Single-view mesh overlay (adapted from render_ego_mano_demo_v3.render_video)."""
    if render_runtime is None:
        render_runtime = create_worker_render_runtime(model_dir=model_dir, device=device)
    device_t = render_runtime["device_t"]
    clip, pose_fps = load_pose_data(pose_path, device_t)

    left_params = clip["left_hand"]["mano_params"]
    right_params = clip["right_hand"]["mano_params"]
    left_pred_valid_raw = clip.get("left_hand", {}).get("pred_valid", None)
    right_pred_valid_raw = clip.get("right_hand", {}).get("pred_valid", None)

    left_model: ManoLayer = render_runtime["left_model"]
    right_model: ManoLayer = render_runtime["right_model"]

    with torch.no_grad():
        left_verts, _ = forward_mano(left_model, left_params)
        right_verts, _ = forward_mano(right_model, right_params)

    cam_data = clip.get("slam_data", {})
    traj = cam_data.get("traj", None)
    img_focal = cam_data.get("img_focal", None)
    img_center = cam_data.get("img_center", None)
    scale = cam_data.get("scale", torch.tensor(1.0, device=device_t))

    if traj is None or img_focal is None or img_center is None:
        raise ValueError("Missing slam_data (traj / img_focal / img_center) in pose file.")

    traj = traj.clone().to(dtype=torch.float32)
    traj[:, :3] = traj[:, :3] * scale

    left_verts = left_verts.to(dtype=torch.float32)
    right_verts = right_verts.to(dtype=torch.float32)

    if hand_space == "world":
        left_verts = world_to_camera(left_verts, traj)
        right_verts = world_to_camera(right_verts, traj)
    elif hand_space != "camera":
        raise ValueError(f"Unknown hand_space: {hand_space}")

    reader = None
    cap = None
    fps = 30.0
    video_w = 0
    video_h = 0
    video_len = None
    use_cv2_reader = False
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap is not None and cap.isOpened():
            use_cv2_reader = True
            fps = _normalize_fps(cap.get(cv2.CAP_PROP_FPS), fallback=fps)
            video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                video_len = frame_count
    except Exception:
        use_cv2_reader = False

    if not use_cv2_reader:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
        reader = imageio.get_reader(str(video_path))
        meta = reader.get_meta_data()
        fps = _normalize_fps(meta.get("fps", 30), fallback=fps)
        video_size = meta.get("size", None)
        if video_size is None:
            first = reader.get_data(0)
            video_h, video_w = first.shape[:2]
        else:
            video_w, video_h = video_size

    if downscale <= 0:
        raise ValueError("downscale must be > 0")
    render_w = max(int(round(video_w / downscale)), 1)
    render_h = max(int(round(video_h / downscale)), 1)
    bbox_sx = float(render_w) / float(video_w) if video_w > 0 else 1.0
    bbox_sy = float(render_h) / float(video_h) if video_h > 0 else 1.0

    k = build_intrinsics(img_focal, img_center)
    k = scale_intrinsics(k, img_center, int(render_w), int(render_h))
    k = k.to(dtype=torch.float32)

    renderer = get_cached_renderer(
        runtime=render_runtime,
        image_size=(int(render_h), int(render_w)),
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
    )
    lights: PointLights = render_runtime["lights"]
    faces_left: torch.Tensor = render_runtime["faces_left"]
    faces_right: torch.Tensor = render_runtime["faces_right"]
    left_color: torch.Tensor = render_runtime["left_color"]
    right_color: torch.Tensor = render_runtime["right_color"]

    num_pose_frames = min(left_verts.shape[0], right_verts.shape[0])
    num_frames = num_pose_frames
    if video_len is None and reader is not None:
        try:
            video_len = reader.get_length()
        except Exception:
            video_len = None
    if isinstance(video_len, (int, float, np.integer, np.floating)) and np.isfinite(video_len) and video_len > 0:
        num_frames = int(video_len)

    if max_frames is not None and max_frames > 0:
        num_frames = min(num_frames, int(max_frames))
    if num_frames <= 0:
        raise RuntimeError(f"No frames available for {video_path.name}")
    pose_idx_map = _build_time_aligned_index_map(num_pose_frames, pose_fps, num_frames, fps)
    if abs(float(pose_fps) - float(fps)) > 1e-3:
        print(
            f"[INFO] Mesh time-aligned sampling: video_fps={fps:.4f}, pose_fps={pose_fps:.4f}, "
            f"video_frames={num_frames}, pose_frames={num_pose_frames}"
        )

    def _build_hand_valid_mask(raw_valid: Any, n: int) -> torch.Tensor:
        if raw_valid is None:
            return torch.ones((n,), dtype=torch.bool, device=device_t)
        if isinstance(raw_valid, torch.Tensor):
            t = raw_valid.to(device=device_t)
        elif isinstance(raw_valid, np.ndarray):
            t = torch.from_numpy(raw_valid).to(device=device_t)
        elif isinstance(raw_valid, (list, tuple)):
            t = torch.tensor(raw_valid, device=device_t)
        elif isinstance(raw_valid, (bool, np.bool_)):
            return torch.full((n,), bool(raw_valid), dtype=torch.bool, device=device_t)
        elif isinstance(raw_valid, (int, float, np.integer, np.floating)):
            return torch.full((n,), bool(raw_valid), dtype=torch.bool, device=device_t)
        else:
            return torch.ones((n,), dtype=torch.bool, device=device_t)

        t = t.reshape(-1)
        if t.numel() == 0:
            return torch.ones((n,), dtype=torch.bool, device=device_t)
        if t.dtype != torch.bool:
            t = t.to(dtype=torch.float32) > 0.5
        if t.numel() == 1:
            t = t.repeat(n)
        if t.numel() < n:
            pad = t.new_full((n - t.numel(),), bool(t[-1].item()))
            t = torch.cat([t, pad], dim=0)
        return t[:n].to(dtype=torch.bool)

    left_valid_mask = _build_hand_valid_mask(left_pred_valid_raw, num_pose_frames)
    right_valid_mask = _build_hand_valid_mask(right_pred_valid_raw, num_pose_frames)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    total_to_render = (num_frames + stride - 1) // stride

    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_video = _make_temp_output_path(output_video)
    writer = imageio.get_writer(
        str(tmp_output_video),
        fps=_resolve_writer_fps(fps, stride),
        codec="libx264",
        macro_block_size=1,
        ffmpeg_log_level="error",
        # Fragmented MP4 is less sensitive to trailer seek/finalize issues on
        # mounted/network filesystems and interrupted runs.
        output_params=["-movflags", "frag_keyframe+empty_moov+default_base_moof"],
    )

    frame_batch: List[np.ndarray] = []
    index_batch: List[int] = []

    def flush_batch() -> int:
        nonlocal frame_batch, index_batch
        if not frame_batch:
            return 0
        frames_np = np.stack(frame_batch, axis=0)
        blended = frames_np.copy()

        pose_indices_np = pose_idx_map[np.asarray(index_batch, dtype=np.int64)]
        idx = torch.from_numpy(pose_indices_np.astype(np.int64)).to(device=device_t)
        left_valid_batch = left_valid_mask.index_select(0, idx)
        right_valid_batch = right_valid_mask.index_select(0, idx)
        valid_any = left_valid_batch | right_valid_batch
        valid_row_idx = torch.nonzero(valid_any, as_tuple=False).reshape(-1)

        if valid_row_idx.numel() > 0 and overlay_alpha > 0:
            src_idx = idx.index_select(0, valid_row_idx)
            left_batch = left_verts.index_select(0, src_idx)
            right_batch = right_verts.index_select(0, src_idx)
            left_valid_sel = left_valid_mask.index_select(0, src_idx)
            right_valid_sel = right_valid_mask.index_select(0, src_idx)
            cameras = create_cameras(k, (int(render_h), int(render_w)), int(src_idx.numel()), device_t)
            with torch.no_grad():
                meshes = build_mesh_batch(
                    left_batch,
                    right_batch,
                    left_valid_sel,
                    right_valid_sel,
                    faces_left,
                    faces_right,
                    left_color,
                    right_color,
                )
                fragments = renderer.rasterizer(meshes, cameras=cameras)
                rendered = renderer.shader(fragments, meshes, cameras=cameras, lights=lights)

            render_rgb = (rendered[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            mask = (fragments.pix_to_face[..., 0] >= 0).cpu().numpy()[..., None]
            alpha = float(overlay_alpha)
            valid_rows_np = valid_row_idx.detach().cpu().numpy().astype(np.int64)
            for rr, row_idx in enumerate(valid_rows_np):
                blended[row_idx] = np.where(
                    mask[rr],
                    (frames_np[row_idx] * (1 - alpha) + render_rgb[rr] * alpha).astype(np.uint8),
                    frames_np[row_idx],
                )

        n = int(blended.shape[0])
        if draw_bbox and bbox_seq:
            for row_idx, video_idx in enumerate(index_batch):
                if video_idx < len(bbox_seq):
                    frame_boxes = bbox_seq[video_idx]
                    for hand_id, box in frame_boxes.items():
                        box_scaled = np.asarray(
                            [
                                box[0] * bbox_sx,
                                box[1] * bbox_sy,
                                box[2] * bbox_sx,
                                box[3] * bbox_sy,
                            ],
                            dtype=np.float32,
                        )
                        _draw_bbox_on_frame(
                            blended[row_idx],
                            box_scaled,
                            color=bbox_color,
                            thickness=max(1, bbox_thickness),
                        )
        for row in blended:
            writer.append_data(row)
        frame_batch = []
        index_batch = []
        return n

    render_ok = False
    try:
        with tqdm(total=total_to_render, desc=f"overlay {video_path.name}", unit="frame", disable=True) as pbar:
            if use_cv2_reader and cap is not None:
                import cv2  # type: ignore

                frame_idx = -1
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if frame_idx >= num_frames:
                        break
                    if frame_idx % stride != 0:
                        continue
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if (frame.shape[1] != render_w) or (frame.shape[0] != render_h):
                        frame = _resize_frame_bgr(frame, render_w, render_h)
                    frame_batch.append(frame)
                    index_batch.append(frame_idx)

                    should_flush = len(frame_batch) >= batch_size or (frame_idx + stride) >= num_frames
                    if should_flush:
                        pbar.update(flush_batch())
            else:
                assert reader is not None
                for frame_idx, frame in enumerate(reader):
                    if frame_idx >= num_frames:
                        break
                    if frame_idx % stride != 0:
                        continue
                    if (frame.shape[1] != render_w) or (frame.shape[0] != render_h):
                        frame = _resize_frame_bgr(frame, render_w, render_h)
                    frame_batch.append(frame)
                    index_batch.append(frame_idx)

                    should_flush = len(frame_batch) >= batch_size or (frame_idx + stride) >= num_frames
                    if should_flush:
                        pbar.update(flush_batch())

            pbar.update(flush_batch())
        render_ok = True
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        try:
            if reader is not None:
                reader.close()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
        if not render_ok:
            try:
                tmp_output_video.unlink()
            except OSError:
                pass

    _commit_video_tmp_file(tmp_output_video, output_video)


def _query_idle_gpu_ids(max_count: int = 4) -> List[str]:
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_ids=auto requires CUDA, but no CUDA device is available.")
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("gpu_ids=auto requires nvidia-smi, but it is not found in PATH.")
    cmd = [
        nvidia_smi,
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to query GPUs via nvidia-smi: {proc.stderr.strip() or proc.stdout.strip()}")

    idle: List[str] = []
    mem_free_threshold_mb = 30 * 1024
    util_threshold_pct = 50
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            continue
        idx_s, mem_free_s, util_s = parts[0], parts[1], parts[2]
        try:
            mem_free_i = int(float(mem_free_s))
            util_i = int(float(util_s))
        except ValueError:
            continue
        if util_i < util_threshold_pct and mem_free_i > mem_free_threshold_mb:
            idle.append(idx_s)

    idle = idle[: max(1, int(max_count))]
    if not idle:
        raise RuntimeError(
            "gpu_ids=auto found no idle GPU. "
            "Please free at least one GPU, or set explicit gpu_ids manually."
        )
    return idle


def resolve_gpu_ids(gpu_ids_arg: str) -> List[str]:
    normalized = ",".join([x.strip() for x in gpu_ids_arg.split(",") if x.strip()])
    if not normalized:
        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
        return []
    if normalized.lower() == "auto":
        return _query_idle_gpu_ids(max_count=4)
    return [x.strip() for x in normalized.split(",") if x.strip()]


def _render_one_video(
    *,
    stem: str,
    video_path: Path,
    tracks_dir: Optional[Path],
    pred_dir: Path,
    out_root: Path,
    mano_model_dir: Path,
    device: str,
    hand_space: str,
    batch_size: int,
    bin_size: int,
    max_faces_per_bin: int,
    overlay_alpha: float,
    max_frames_opt: Optional[int],
    stride: int,
    downscale: float,
    draw_bbox: bool,
    bbox_color: Tuple[int, int, int],
    bbox_thickness: int,
    force_rerun: bool,
    render_runtime: Dict[str, Any],
) -> None:
    pose_path = pred_dir / f"{stem}.pose3d_hand"
    out_path = out_root / f"{stem}.mp4"
    if not pose_path.is_file():
        return
    if pose_path.stat().st_size == 0:
        return
    if (not force_rerun) and out_path.is_file() and out_path.stat().st_size > 0:
        return

    bbox_seq: List[Dict[int, np.ndarray]] = []
    tracks_file = (tracks_dir / f"{stem}_tracks.npy") if tracks_dir is not None else None
    if draw_bbox and tracks_file is not None and tracks_file.is_file():
        try:
            src_w = 0
            src_h = 0
            try:
                import cv2  # type: ignore

                tmp_cap = cv2.VideoCapture(str(video_path))
                if tmp_cap is not None and tmp_cap.isOpened():
                    src_w = int(tmp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    src_h = int(tmp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if tmp_cap is not None:
                    tmp_cap.release()
            except Exception:
                src_w = 0
                src_h = 0
            if src_w <= 0 or src_h <= 0:
                tmp_reader = imageio.get_reader(str(video_path))
                tmp_meta = tmp_reader.get_meta_data()
                video_size = tmp_meta.get("size", None)
                if video_size is None:
                    first_frame = tmp_reader.get_data(0)
                    src_h, src_w = first_frame.shape[:2]
                else:
                    src_w, src_h = int(video_size[0]), int(video_size[1])
                tmp_reader.close()
            bbox_seq = load_bbox_sequence(tracks_file, src_w, src_h)
        except Exception:
            bbox_seq = []

    render_mesh_overlay_video(
        pose_path,
        video_path,
        out_path,
        mano_model_dir,
        device=device,
        hand_space=hand_space,
        batch_size=max(1, batch_size),
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
        overlay_alpha=overlay_alpha,
        max_frames=max_frames_opt,
        stride=max(1, stride),
        downscale=downscale,
        bbox_seq=bbox_seq,
        draw_bbox=draw_bbox,
        bbox_color=bbox_color,
        bbox_thickness=bbox_thickness,
        render_runtime=render_runtime,
    )
    print(f"[OK] Saved overlay video: {out_path}", flush=True)


def _render_camera_pose_one_video(
    *,
    stem: str,
    video_path: Path,
    pred_dir: Path,
    gt_dir: Path,
    out_root: Path,
    force_rerun: bool,
    max_frames_opt: Optional[int],
    stride: int,
) -> None:
    pred_pose = pred_dir / f"{stem}.pose3d_hand"
    gt_pose = gt_dir / f"{stem}.pose3d_hand"
    out_path = out_root / f"{stem}.mp4"
    if not pred_pose.is_file() or not gt_pose.is_file():
        return
    if pred_pose.stat().st_size <= 0 or gt_pose.stat().st_size <= 0:
        return
    if (not force_rerun) and out_path.is_file() and out_path.stat().st_size > 0:
        return
    render_slam_camera_pose_video(
        video_path=video_path,
        pred_pose_path=pred_pose,
        gt_pose_path=gt_pose,
        output_video=out_path,
        max_frames=max_frames_opt,
        stride=max(1, stride),
    )
    print(f"[OK] Saved camera-pose video: {out_path}", flush=True)


def _camera_pose_worker_main(
    worker_videos: List[Tuple[str, str]],
    shared: Dict[str, Any],
) -> None:
    warnings.filterwarnings("ignore")
    pred_dir = Path(shared["pred_dir"])
    gt_dir = Path(shared["gt_dir"])
    out_root = Path(shared["out_root"])
    force_rerun = bool(shared["force_rerun"])
    max_frames_opt = shared["max_frames_opt"]
    stride = int(shared["stride"])
    for stem, video_path_str in worker_videos:
        try:
            _render_camera_pose_one_video(
                stem=stem,
                video_path=Path(video_path_str),
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                out_root=out_root,
                force_rerun=force_rerun,
                max_frames_opt=max_frames_opt,
                stride=max(1, stride),
            )
        except Exception as exc:
            out_path = out_root / f"{stem}.mp4"
            print(f"[ERROR]{out_path} camera pose video error:{exc}", flush=True)


def _worker_main(
    worker_rank: int,
    gpu_id: Optional[str],
    worker_videos: List[Tuple[str, str]],
    shared: Dict[str, Any],
) -> None:
    warnings.filterwarnings("ignore")
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        device = f"cuda:{int(gpu_id)}"
    else:
        device = shared["device_fallback"]

    pred_dir = Path(shared["pred_dir"])
    tracks_dir_raw = str(shared.get("tracks_dir", "")).strip()
    tracks_dir = Path(tracks_dir_raw) if tracks_dir_raw else None
    out_root = Path(shared["out_root"])
    mano_model_dir = Path(shared["mano_model_dir"])
    render_runtime = create_worker_render_runtime(model_dir=mano_model_dir, device=device)
    for stem, video_path_str in worker_videos:
        video_path = Path(video_path_str)
        out_path = out_root / f"{stem}.mp4"
        try:
            _render_one_video(
                stem=stem,
                video_path=video_path,
                tracks_dir=tracks_dir,
                pred_dir=pred_dir,
                out_root=out_root,
                mano_model_dir=mano_model_dir,
                device=device,
                hand_space=shared["hand_space"],
                batch_size=int(shared["batch_size"]),
                bin_size=int(shared["bin_size"]),
                max_faces_per_bin=int(shared["max_faces"]),
                overlay_alpha=float(shared["overlay_alpha"]),
                max_frames_opt=shared["max_frames_opt"],
                stride=int(shared["stride"]),
                downscale=float(shared["downscale"]),
                draw_bbox=bool(shared["draw_bbox"]),
                bbox_color=tuple(shared["bbox_color"]),
                bbox_thickness=int(shared["bbox_thickness"]),
                force_rerun=bool(shared["force_rerun"]),
                render_runtime=render_runtime,
            )
        except Exception as exc:
            print(f"[ERROR]{out_path} overlay video error:{exc}", flush=True)


def main() -> int:
    warnings.filterwarnings("ignore")
    args = parse_args()
    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding="utf-8")
    if "common" not in cfg:
        cfg["common"] = {}
    cfg["common"]["config_name"] = Path(args.config).name
    cfg["common"]["config_stem"] = Path(args.config).stem
    rt = normalize_run_tag(getattr(args, "run_tag", ""))
    cfg["common"]["cli_run_tag"] = rt
    dataset_name = normalize_dataset_name(args.dataset.lower().strip())
    algorithm_name = args.algorithm.strip()
    apply_dataset_overrides(cfg, dataset_name, args.dataset_override)

    if not cfg.has_section("visualization"):
        return 0

    s4 = cfg["visualization"]
    mesh_enabled = s4.getboolean("enabled", fallback=False)
    # Independent SLAM camera-pose visualization switch; keep compatibility
    # with user's existing naming.
    camera_pose_enabled = s4.getboolean("visualize_camera_pose", fallback=False)
    if (not mesh_enabled) and (not camera_pose_enabled):
        return 0

    visual_root = Path(s4.get("visual_result_root", "/home/xiaoxuan/Data/visual_result").strip())
    force_rerun_mesh = s4.getboolean("force_rerun_mesh", fallback=False)
    force_rerun_camera_pose = s4.getboolean("force_rerun_camera_pose", fallback=False)
    device = s4.get("device", "auto").strip() or "auto"
    overlay_alpha = s4.getfloat("overlay_alpha", fallback=0.75)
    downscale = s4.getfloat("downscale", fallback=1.0)
    batch_size = s4.getint("batch_size", fallback=1)
    hand_space = s4.get("hand_space", "world").strip() or "world"
    stride = s4.getint("stride", fallback=1)
    max_frames = s4.getint("max_frames", fallback=-1)
    max_frames_opt: Optional[int] = None if max_frames <= 0 else max_frames

    bin_size = s4.getint("bin_size", fallback=128)
    max_faces = s4.getint("max_faces_per_bin", fallback=20000)
    draw_bbox = s4.getboolean("draw_bbox", fallback=True)
    bbox_thickness = max(1, s4.getint("bbox_thickness", fallback=2))
    bbox_color_raw = s4.get("bbox_color_bgr", "0,255,255").strip()
    try:
        c = [int(x.strip()) for x in bbox_color_raw.split(",")]
        if len(c) != 3 or any((v < 0 or v > 255) for v in c):
            raise ValueError("bbox_color_bgr must have 3 ints in [0,255]")
        bbox_color: Tuple[int, int, int] = (c[0], c[1], c[2])
    except Exception as exc:
        raise ValueError(f"Invalid [visualization] bbox_color_bgr='{bbox_color_raw}': {exc}") from exc

    mano_raw = s4.get("mano_model_dir", "").strip()
    if not mano_raw:
        mano_raw = require_section(cfg, "evaluation")["mano_model_dir"].strip()
    mano_model_dir = ensure_exists(mano_raw, "mano_model_dir")

    dataset_sec = resolve_dataset_section(cfg, dataset_name)
    video_dir = ensure_exists(dataset_sec["video_dir"], "video_dir")
    tracks_dir = ensure_exists(dataset_sec["tracks_dir"], "tracks_dir")
    video_glob = s4.get("video_glob", "*.mp4").strip() or "*.mp4"

    pred_dir, _pred_scoped = resolve_prediction_dir(cfg, dataset_name, algorithm_name, run_tag=rt)
    pred_dataset = resolve_dataset_output_name(cfg, dataset_name)
    gt_dir: Optional[Path] = None
    if camera_pose_enabled:
        gt_dir_raw = dataset_sec.get("gt_pose3d_dir", "").strip()
        if gt_dir_raw:
            gt_dir = ensure_exists(gt_dir_raw, "gt_pose3d_dir")
        else:
            print("[WARN] [dataset.*] gt_pose3d_dir is empty; disable camera-pose visualization.")
            camera_pose_enabled = False

    videos = collect_videos(video_dir, video_glob)
    if not videos:
        return 0

    out_root = _apply_config_scope_to_dir(
        cfg,
        apply_run_tag_to_dir(visual_root / pred_dataset / algorithm_name, rt),
    )
    if mesh_enabled:
        out_root.mkdir(parents=True, exist_ok=True)

    camera_out_root_cfg = s4.get("camera_pose_output_root", "").strip()
    if camera_out_root_cfg:
        camera_root_base = Path(camera_out_root_cfg)
    else:
        camera_root_base = visual_root
    # Keep camera-pose output layout consistent with mesh visualization:
    # <root>/<canonical_dataset>/<algorithm>/<stem>.mp4
    camera_out_root = _apply_config_scope_to_dir(
        cfg,
        apply_run_tag_to_dir(camera_root_base / pred_dataset / algorithm_name, rt),
    )
    if camera_pose_enabled:
        camera_out_root.mkdir(parents=True, exist_ok=True)

    # Shared worker settings for mesh/camera branches.
    requested_gpu_ids = resolve_gpu_ids(s4.get("gpu_ids", ""))
    cfg_workers = max(1, s4.getint("num_workers", fallback=1))

    if mesh_enabled:
        if requested_gpu_ids and torch.cuda.is_available():
            num_workers = min(cfg_workers, len(requested_gpu_ids))
            gpu_ids: List[Optional[str]] = requested_gpu_ids[:num_workers]
        else:
            num_workers = 1 if cfg_workers <= 1 else cfg_workers
            gpu_ids = [None] * num_workers

        video_shards: List[List[Tuple[str, str]]] = []
        as_pairs = [(stem, str(vp)) for stem, vp in videos]
        for i in range(num_workers):
            video_shards.append(as_pairs[i::num_workers])

        shared: Dict[str, Any] = {
            "pred_dir": str(pred_dir),
            "tracks_dir": str(tracks_dir) if tracks_dir is not None else "",
            "out_root": str(out_root),
            "mano_model_dir": str(mano_model_dir),
            "device_fallback": device,
            "hand_space": hand_space,
            "batch_size": max(1, batch_size),
            "bin_size": bin_size,
            "max_faces": max_faces,
            "overlay_alpha": overlay_alpha,
            "max_frames_opt": max_frames_opt,
            "stride": max(1, stride),
            "downscale": downscale,
            "draw_bbox": draw_bbox,
            "bbox_color": bbox_color,
            "bbox_thickness": bbox_thickness,
            "force_rerun": force_rerun_mesh,
        }

        if num_workers == 1:
            _worker_main(0, gpu_ids[0], video_shards[0], shared)
        else:
            mp.set_start_method("spawn", force=True)
            procs: List[mp.Process] = []
            for i in range(num_workers):
                p = mp.Process(target=_worker_main, args=(i, gpu_ids[i], video_shards[i], shared))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

    if camera_pose_enabled and gt_dir is not None:
        if requested_gpu_ids and torch.cuda.is_available():
            camera_workers = min(cfg_workers, len(requested_gpu_ids))
        else:
            camera_workers = max(1, cfg_workers)
        camera_pairs = [(stem, str(video_path)) for stem, video_path in videos]
        camera_shards: List[List[Tuple[str, str]]] = [camera_pairs[i::camera_workers] for i in range(camera_workers)]
        camera_shared: Dict[str, Any] = {
            "pred_dir": str(pred_dir),
            "gt_dir": str(gt_dir),
            "out_root": str(camera_out_root),
            "force_rerun": force_rerun_camera_pose,
            "max_frames_opt": max_frames_opt,
            "stride": max(1, stride),
        }
        if camera_workers == 1:
            _camera_pose_worker_main(camera_shards[0], camera_shared)
        else:
            mp.set_start_method("spawn", force=True)
            procs: List[mp.Process] = []
            for i in range(camera_workers):
                p = mp.Process(target=_camera_pose_worker_main, args=(camera_shards[i], camera_shared))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())