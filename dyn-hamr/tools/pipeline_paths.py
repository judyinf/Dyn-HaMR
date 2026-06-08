from __future__ import annotations

from pathlib import Path


def normalize_run_tag(tag: str) -> str:
    return (tag or "").strip()


def apply_run_tag_to_dir(base_dir: Path, run_tag: str) -> Path:
    rt = normalize_run_tag(run_tag)
    return base_dir if not rt else base_dir / rt
