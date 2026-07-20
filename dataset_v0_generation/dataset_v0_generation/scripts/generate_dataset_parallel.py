#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def human_seconds(value: float) -> str:
    hours, rem = divmod(int(round(value)), 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely generate Dataset-v0 with multiple CPU workers.")
    parser.add_argument("--root", type=Path, default=Path("/home/liuzhongkai/DEM_pre"))
    parser.add_argument("--schedule-csv", type=Path, default=None)
    parser.add_argument("--tle-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--expected-frames", type=int, default=17280)
    parser.add_argument("--min-free-gib", type=float, default=300.0)
    parser.add_argument("--check-arrays", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    schedule_path = (args.schedule_csv or root / "results/dataset_v0/service_schedule_20s.csv").expanduser().resolve()
    tle_path = (args.tle_path or root / "data/l1_space/2025-01-01_leo_payload.tle").expanduser().resolve()
    output_root = (args.output_root or root / "results/dataset_v0").expanduser().resolve()
    generator = root / "src/generate_dataset_minimal_direct.py"
    finalizer = root / "scripts/finalize_dataset_v0.py"

    for required in (schedule_path, tle_path, generator, finalizer):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.workers < 1 or args.threads_per_worker < 1:
        raise ValueError("workers and threads-per-worker must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    free_gib = usage.free / (1024**3)
    print(f"Free space at output filesystem: {free_gib:.1f} GiB")
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"Insufficient free space: {free_gib:.1f} GiB < required {args.min_free_gib:.1f} GiB"
        )

    schedule = pd.read_csv(schedule_path)
    if "no_service_flag" in schedule.columns:
        schedule = schedule[schedule["no_service_flag"] == False].copy()  # noqa: E712
    schedule = schedule.sort_values(["region", "frame_id"]).reset_index(drop=True)
    if len(schedule) != args.expected_frames:
        raise RuntimeError(
            f"Schedule rows={len(schedule)}, expected={args.expected_frames}. Refusing to start an unintended run."
        )

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_root = output_root / "parallel_runs" / run_tag
    shard_root = run_root / "shards"
    log_root = run_root / "logs"
    shard_root.mkdir(parents=True)
    log_root.mkdir(parents=True)

    # Round-robin shards balance terrain/elevation cost while keeping every frame unique.
    shards: list[Path] = []
    for worker_id in range(args.workers):
        shard = schedule.iloc[worker_id::args.workers].copy()
        shard_path = shard_root / f"worker_{worker_id:02d}.csv"
        shard.to_csv(shard_path, index=False)
        shards.append(shard_path)

    run_config = {
        "run_tag": run_tag,
        "root": str(root),
        "schedule_csv": str(schedule_path),
        "tle_path": str(tle_path),
        "output_root": str(output_root),
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "schedule_frames": len(schedule),
        "expected_frames": args.expected_frames,
        "free_space_gib_at_start": free_gib,
        "start_unix": time.time(),
        "generator": str(generator),
        "geometry_backend": "cpu_ecef_enu_numba",
        "per_frame_files": 7,
    }
    atomic_json(run_root / "run_config.json", run_config)

    env_base = os.environ.copy()
    env_base.update(
        {
            "NUMBA_NUM_THREADS": str(args.threads_per_worker),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    processes: list[tuple[int, subprocess.Popen, object, Path]] = []
    start = time.perf_counter()
    print("=" * 78)
    print("Parallel Dataset-v0 generation")
    print(f"Frames: {len(schedule)}")
    print(f"Workers: {args.workers}")
    print(f"Numba threads per worker: {args.threads_per_worker}")
    print(f"Output: {output_root}")
    print(f"Logs: {log_root}")
    print("=" * 78)

    try:
        for worker_id, shard_path in enumerate(shards):
            log_path = log_root / f"worker_{worker_id:02d}.log"
            log_handle = log_path.open("w", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                str(generator),
                "--root", str(root),
                "--schedule-csv", str(shard_path),
                "--tle-path", str(tle_path),
                "--output-root", str(output_root),
                "--core-size-km", "76.8",
                "--buffer-km", "40",
                "--resolution-m", "100",
                "--freq-ghz", "14.5",
                "--numba-threads", str(args.threads_per_worker),
                "--skip-existing",
                "--worker-mode",
            ]
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env_base,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((worker_id, process, log_handle, log_path))
            print(f"started worker {worker_id:02d}, pid={process.pid}, rows={sum(1 for _ in open(shard_path, encoding='utf-8')) - 1}")

        failed: list[tuple[int, int, Path]] = []
        while True:
            alive = 0
            completed = 0
            for worker_id, process, _handle, log_path in processes:
                code = process.poll()
                if code is None:
                    alive += 1
                else:
                    completed += 1
                    if code != 0 and not any(item[0] == worker_id for item in failed):
                        failed.append((worker_id, code, log_path))
            elapsed = time.perf_counter() - start
            frame_count = sum(1 for _ in (output_root / "base_maps").glob("*/gt_pr_dbm.npy"))
            print(
                f"[monitor] elapsed={human_seconds(elapsed)} alive={alive} "
                f"completed_workers={completed}/{args.workers} complete_frames={frame_count}/{args.expected_frames}",
                flush=True,
            )
            if alive == 0:
                break
            time.sleep(30)

        for _worker_id, _process, handle, _log_path in processes:
            handle.close()

        if failed:
            details = ", ".join(f"worker={wid} exit={code} log={path}" for wid, code, path in failed)
            raise RuntimeError(f"Parallel generation failed: {details}")

    except KeyboardInterrupt:
        print("Interrupted: terminating all workers...", file=sys.stderr)
        for _worker_id, process, _handle, _log_path in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise
    finally:
        for _worker_id, process, handle, _log_path in processes:
            if process.poll() is None:
                process.terminate()
            try:
                handle.close()
            except Exception:
                pass

    generation_wall = time.perf_counter() - start
    finalize_start = time.perf_counter()
    finalize_command = [
        sys.executable,
        str(finalizer),
        "--output-root", str(output_root),
        "--expected-frames", str(args.expected_frames),
    ]
    if args.check_arrays:
        finalize_command.append("--check-arrays")
    subprocess.run(finalize_command, cwd=root, check=True)
    finalize_wall = time.perf_counter() - finalize_start

    final_count = sum(1 for _ in (output_root / "base_maps").glob("*/gt_pr_dbm.npy"))
    total_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    total_wall = time.perf_counter() - start
    throughput = final_count * 3600.0 / generation_wall if generation_wall else 0.0
    summary = {
        **run_config,
        "end_unix": time.time(),
        "generation_wall_seconds": generation_wall,
        "finalization_wall_seconds": finalize_wall,
        "total_wall_seconds": total_wall,
        "complete_frames": final_count,
        "throughput_frames_per_hour": throughput,
        "dataset_size_bytes": total_bytes,
        "dataset_size_gib": total_bytes / (1024**3),
        "average_mib_per_frame": total_bytes / max(final_count, 1) / (1024**2),
        "status": "success",
    }
    atomic_json(run_root / "parallel_run_summary.json", summary)
    atomic_json(output_root / "latest_parallel_run_summary.json", summary)

    print("=" * 78)
    print("ONE-DAY DATASET GENERATION PASSED")
    print(f"Complete frames: {final_count}")
    print(f"Generation wall time: {human_seconds(generation_wall)}")
    print(f"Throughput: {throughput:.1f} frames/hour")
    print(f"Dataset size: {summary['dataset_size_gib']:.2f} GiB")
    print(f"Run summary: {run_root / 'parallel_run_summary.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
