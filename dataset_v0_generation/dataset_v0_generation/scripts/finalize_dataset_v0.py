#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_ARRAYS = (
    "coarse_pr_dbm.npy",
    "gt_pr_dbm.npy",
    "terrain_loss_lite_db.npy",
    "terrain_loss_hf_db.npy",
    "weather_loss_db.npy",
    "residual_dhf_to_e_db.npy",
)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and finalize Dataset-v0.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=17280)
    parser.add_argument("--check-arrays", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    root = args.output_root.expanduser().resolve()
    base = root / "base_maps"
    if not base.is_dir():
        raise FileNotFoundError(base)

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    frame_dirs = sorted(path for path in base.iterdir() if path.is_dir() and not path.name.endswith(".tmp"))

    for index, frame_dir in enumerate(frame_dirs, start=1):
        meta_path = frame_dir / "meta.json"
        if not meta_path.is_file():
            errors.append(f"missing meta.json: {frame_dir}")
            continue
        missing = [name for name in REQUIRED_ARRAYS if not (frame_dir / name).is_file()]
        if missing:
            errors.append(f"missing {missing}: {frame_dir}")
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid meta.json: {meta_path}: {exc}")
            continue

        if args.check_arrays:
            shapes = set()
            for name in REQUIRED_ARRAYS:
                try:
                    arr = np.load(frame_dir / name, mmap_mode="r")
                    shapes.add(tuple(arr.shape))
                    if arr.dtype != np.float32:
                        errors.append(f"dtype {arr.dtype} != float32: {frame_dir / name}")
                except Exception as exc:
                    errors.append(f"cannot load {frame_dir / name}: {exc}")
            if len(shapes) != 1:
                errors.append(f"shape mismatch {sorted(shapes)}: {frame_dir}")

        records.append(meta)
        if index % 1000 == 0:
            print(f"validated {index}/{len(frame_dirs)} frame directories")

    records.sort(key=lambda item: (str(item.get("region", "")), int(item.get("frame_id", -1))))
    complete = len(records)
    print(f"Complete frames: {complete}")
    print(f"Expected frames: {args.expected_frames}")
    print(f"Errors: {len(errors)}")

    if errors:
        error_path = root / "dataset_v0_validation_errors.txt"
        atomic_write_text(error_path, "\n".join(errors) + "\n")
        print(f"Validation errors written to: {error_path}", file=sys.stderr)

    if (complete != args.expected_frames or errors) and not args.allow_incomplete:
        raise RuntimeError(
            f"Dataset is incomplete: complete={complete}, expected={args.expected_frames}, errors={len(errors)}"
        )

    atomic_write_text(
        root / "dataset_v0_manifest.json",
        json.dumps(records, ensure_ascii=False, indent=2),
    )
    csv_text = pd.DataFrame(records).to_csv(index=False)
    atomic_write_text(root / "dataset_v0_manifest.csv", csv_text)

    summary = {
        "output_root": str(root),
        "expected_frames": args.expected_frames,
        "complete_frames": complete,
        "validation_errors": len(errors),
        "required_per_frame_arrays": list(REQUIRED_ARRAYS),
        "finalized_at_unix": time.time(),
    }
    atomic_write_text(
        root / "dataset_v0_finalization_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    print(f"Manifest written: {root / 'dataset_v0_manifest.csv'}")
    print("Dataset finalization PASSED")


if __name__ == "__main__":
    main()
