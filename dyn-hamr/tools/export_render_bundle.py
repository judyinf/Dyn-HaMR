#!/usr/bin/env python3
"""Export the minimum processed data needed to render a Dyn-HaMR log locally."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


DEFAULT_PHASES = ("init", "root_fit", "smooth_fit", "prior")
PRIOR_REQUIRED_KEYS = {
    "pose_body",
    "trans",
    "root_orient",
    "betas",
    "is_right",
    "cam_R",
    "cam_t",
    "intrins",
}


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def copy_file(src: Path, dst: Path):
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def result_sort_key(path: Path):
    parts = path.name.split("_")
    if len(parts) < 3 or parts[-2] not in {"world", "prior"}:
        raise ValueError(f"Unexpected result filename: {path.name}")
    iteration = parts[-3]
    return (int(iteration) if iteration.isdigit() else -1, path.name)


def find_latest_result(log_dir: Path, phase: str):
    phase_dir = log_dir / phase
    results = sorted(phase_dir.glob("*_results.npz"), key=result_sort_key)
    return results[-1] if results else None


def inspect_result(path: Path):
    with np.load(path) as data:
        if "trans" not in data.files or data["trans"].ndim < 2:
            raise ValueError(f"{path} does not contain a valid trans trajectory")
        return set(data.files), int(data["trans"].shape[1])


def resolve_source(cfg, name: str):
    return Path(str(cfg.data.sources[name])).expanduser().resolve()


def get_selected_frames(cfg, track_info):
    shots_path = resolve_source(cfg, "shots")
    shots = load_json(shots_path)
    shot_idx = int(cfg.data.shot_idx)
    shot_frames = sorted(name for name, value in shots.items() if int(value) == shot_idx)

    data_start, data_end = map(int, track_info["meta"]["data_interval"])
    seq_start, seq_end = map(int, track_info["meta"]["seq_interval"])
    data_end = len(shot_frames) if data_end < 0 else data_end
    if not 0 <= data_start <= data_end <= len(shot_frames):
        raise ValueError(f"Invalid data interval [{data_start}, {data_end}) for {len(shot_frames)} shot frames")
    selected_frames = shot_frames[data_start:data_end][seq_start:seq_end]
    if not selected_frames:
        raise ValueError("The selected frame interval is empty")
    return selected_frames, (data_start, data_end), (seq_start, seq_end)


def copy_selected_tracks(cfg, output: Path, frames, track_info):
    src_tracks = resolve_source(cfg, "tracks")
    dst_tracks = output / "data" / "dynhamr" / "track_preds" / str(cfg.data.seq)
    selected_tracks = sorted(track_info["tracks"], key=int)
    local_tracks = {}

    for track_id in selected_tracks:
        src_track = src_tracks / f"{int(track_id):03d}"
        dst_track = dst_tracks / f"{int(track_id):03d}"
        src_mask = track_info["tracks"][track_id]["vis_mask"]
        seq_start, seq_end = map(int, track_info["meta"]["seq_interval"])
        vis_mask = src_mask[seq_start:seq_end]
        visible_count = 0

        if len(vis_mask) != len(frames):
            raise ValueError(f"Track {track_id} visibility mask length does not match frame count")

        for frame, visible in zip(frames, vis_mask):
            stem = Path(frame).stem
            keypoints = src_track / f"{stem}_keypoints.json"
            mano = src_track / f"{stem}_mano.json"
            if keypoints.exists() != mano.exists():
                raise ValueError(f"Track {track_id}, frame {frame}: keypoints and MANO files are incomplete")
            if visible and not keypoints.exists():
                raise ValueError(f"Track {track_id}, frame {frame}: visible track data is missing")
            if keypoints.exists():
                copy_file(keypoints, dst_track / keypoints.name)
                copy_file(mano, dst_track / mano.name)
                visible_count += 1

        if visible_count != sum(bool(value) for value in vis_mask):
            raise ValueError(f"Track {track_id}: copied file count does not match visibility mask")
        local_tracks[track_id] = {
            "index": int(track_info["tracks"][track_id]["index"]),
            "vis_mask": [bool(value) for value in vis_mask],
        }

    return local_tracks


def copy_selected_cameras(cfg, output: Path, frame_count: int, data_interval, seq_interval):
    src = resolve_source(cfg, "cameras") / "cameras.npz"
    if not src.is_file():
        raise FileNotFoundError(src)

    data_start, _ = data_interval
    seq_start, seq_end = seq_interval
    if bool(cfg.data.get("split_cameras", True)):
        start, end = data_start + seq_start, data_start + seq_end
    else:
        start, end = seq_start, seq_end

    dst = output / "data" / "dynhamr" / "cameras" / str(cfg.data.seq) / f"shot-{cfg.data.shot_idx}" / "cameras.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with np.load(src) as cameras:
        if "w2c" not in cameras.files:
            raise ValueError(f"{src} does not contain w2c")
        if not 0 <= start <= end <= len(cameras["w2c"]):
            raise ValueError(f"Camera interval [{start}, {end}) exceeds {len(cameras['w2c'])} poses")
        cropped = {}
        for key in cameras.files:
            value = cameras[key]
            cropped[key] = value[start:end] if value.ndim > 0 and len(value) == len(cameras["w2c"]) else value
        if len(cropped["w2c"]) != frame_count:
            raise ValueError("Cropped camera count does not match frame count")
        np.savez(dst, **cropped)
    return dst


def copy_diagnostics(log_dir: Path, output: Path):
    src = log_dir / "prior" / "pkls"
    dst = output / "diagnostics" / "prior" / "pkls"
    copied = []
    if not src.is_dir():
        return copied
    patterns = ("stg_*.pkl", "stage_*_loss.pkl", "stage_*_loss.jpg", "all_stages_loss.jpg")
    for pattern in patterns:
        for path in sorted(src.glob(pattern)):
            copy_file(path, dst / path.name)
            copied.append(str((dst / path.name).relative_to(output)))
    return copied


def export_bundle(log_dir: Path, output: Path, phases):
    log_dir = log_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    hydra_dir = log_dir / ".hydra"
    config_path = hydra_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    shutil.copytree(hydra_dir, output / ".hydra", dirs_exist_ok=True)

    cfg = OmegaConf.load(config_path)
    track_info_path = log_dir / "track_info.json"
    track_info = load_json(track_info_path)
    frames, data_interval, seq_interval = get_selected_frames(cfg, track_info)

    exported_phases = {}
    frame_count = None
    for phase in phases:
        result = find_latest_result(log_dir, phase)
        if result is None:
            print(f"Skipping missing phase: {phase}")
            continue
        keys, result_frames = inspect_result(result)
        if phase == "prior" and not PRIOR_REQUIRED_KEYS.issubset(keys):
            missing = sorted(PRIOR_REQUIRED_KEYS - keys)
            raise ValueError(f"{result} is missing prior render keys: {missing}")
        if frame_count is None:
            frame_count = result_frames
        elif result_frames != frame_count:
            raise ValueError(f"{result} has {result_frames} frames, expected {frame_count}")
        dst = output / phase / result.name
        copy_file(result, dst)
        exported_phases[phase] = str(dst.relative_to(output))

    if not exported_phases:
        raise ValueError("No optimization result files were found")
    if frame_count != len(frames):
        raise ValueError(f"Results contain {frame_count} frames, but track metadata selects {len(frames)}")

    src_images = resolve_source(cfg, "images")
    dst_images = output / "data" / "images" / str(cfg.data.seq)
    for frame in frames:
        copy_file(src_images / frame, dst_images / frame)

    local_tracks = copy_selected_tracks(cfg, output, frames, track_info)
    cameras = copy_selected_cameras(cfg, output, frame_count, data_interval, seq_interval)
    local_shots = output / "data" / "dynhamr" / "shot_idcs" / f"{cfg.data.seq}.json"
    local_shots.parent.mkdir(parents=True, exist_ok=True)
    with local_shots.open("w") as f:
        json.dump({frame: int(cfg.data.shot_idx) for frame in frames}, f, indent=2)

    local_track_info = {
        "tracks": local_tracks,
        "meta": {"seq_interval": [0, frame_count], "data_interval": [0, frame_count]},
    }
    with (output / "track_info.json").open("w") as f:
        json.dump(local_track_info, f, indent=2)

    manifest = {
        "format_version": 1,
        "sequence": str(cfg.data.seq),
        "shot_idx": int(cfg.data.shot_idx),
        "frame_count": frame_count,
        "track_ids": [int(track_id) for track_id in sorted(local_tracks, key=int)],
        "phases": exported_phases,
        "data": {
            "images": str(dst_images.relative_to(output)),
            "tracks": str((output / "data" / "dynhamr" / "track_preds" / str(cfg.data.seq)).relative_to(output)),
            "shots": str(local_shots.relative_to(output)),
            "cameras": str(cameras.parent.relative_to(output)),
        },
        "diagnostics": copy_diagnostics(log_dir, output),
    }
    with (output / "render_bundle.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported {frame_count} frames and {len(local_tracks)} tracks to {output}")
    print(f"Phases: {', '.join(exported_phases)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phases", nargs="+", default=DEFAULT_PHASES)
    args = parser.parse_args()
    export_bundle(args.log_dir, args.output, args.phases)


if __name__ == "__main__":
    main()
