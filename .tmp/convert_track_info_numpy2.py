from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import pickle

import numpy as np


SRC = Path("demo/segment_000_ch1_undistort_track_info.npy")
DST = Path(".tmp/segment_000_ch1_undistort_track_info.sanitized.pkl")


def sanitize(obj: Any, counts: Counter[str]) -> Any:
    if isinstance(obj, np.ndarray):
        counts["ndarray"] += 1
        return obj.tolist()
    if isinstance(obj, np.generic):
        counts["numpy_scalar"] += 1
        return obj.item()
    if isinstance(obj, dict):
        counts["dict"] += 1
        return {sanitize(k, counts): sanitize(v, counts) for k, v in obj.items()}
    if isinstance(obj, list):
        counts["list"] += 1
        return [sanitize(v, counts) for v in obj]
    if isinstance(obj, tuple):
        counts["tuple"] += 1
        return tuple(sanitize(v, counts) for v in obj)
    counts[type(obj).__name__] += 1
    return obj


def main() -> None:
    if not SRC.is_file():
        raise FileNotFoundError(SRC)
    print("numpy", np.__version__)
    raw = np.load(SRC, allow_pickle=True).item()
    counts: Counter[str] = Counter()
    clean = sanitize(raw, counts)
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("wb") as f:
        pickle.dump(clean, f, protocol=4)
    print("wrote", DST)
    print("converted objects", dict(counts))


if __name__ == "__main__":
    main()
