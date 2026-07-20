#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile CPU, memory, and throughput for Dataset-v0 direct generation.

The tool benchmarks two independent scaling dimensions:

1. Thread scaling: one generator process with different Numba thread counts.
2. Process scaling: several independent generator processes running concurrently.

For each configuration it records wall time, CPU seconds, effective CPU cores,
peak memory, per-stage timings, frames/hour, and estimated time/core-hours for
producing the complete service schedule.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


FRAME_RE = re.compile(
    r"\[(?P<index>\d+)/(?P<count>\d+)\]\s+\S+\s+"
    r"total=(?P<total>[0-9.]+)s\s+"
    r"geom\+A\+B=(?P<geometry>[0-9.]+)s"
    r"(?:\s+\(SGP4=(?P<sgp4>[0-9.]+)s,\s+CPU-fused=(?P<cpu_fused>[0-9.]+)s\))?\s+"
    r"D-lite=(?P<lite>[0-9.]+)s\s+"
    r"D-HF=(?P<hf>[0-9.]+)s\s+"
    r"E=(?P<model_e>[0-9.]+)s"
    r"(?:\s+save=(?P<save>[0-9.]+)s)?"
)

TIME_PATTERNS = {
    "user_seconds": re.compile(r"^\s*User time \(seconds\):\s*([0-9.]+)\s*$"),
    "system_seconds": re.compile(r"^\s*System time \(seconds\):\s*([0-9.]+)\s*$"),
    "cpu_percent": re.compile(r"^\s*Percent of CPU this job got:\s*([0-9.]+)%\s*$"),
    "max_rss_kb": re.compile(r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$"),
    "fs_inputs": re.compile(r"^\s*File system inputs:\s*(\d+)\s*$"),
    "fs_outputs": re.compile(r"^\s*File system outputs:\s*(\d+)\s*$"),
    "voluntary_cs": re.compile(r"^\s*Voluntary context switches:\s*(\d+)\s*$"),
    "involuntary_cs": re.compile(r"^\s*Involuntary context switches:\s*(\d+)\s*$"),
}


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("Values must be positive integers")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return sorted(set(values))


def false_like(value: str) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", "n", ""}


def human_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "N/A"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    if days >= 1:
        return f"{int(days)}d {int(hours):02d}h {int(minutes):02d}m"
    if hours >= 1:
        return f"{int(hours)}h {int(minutes):02d}m {sec:04.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {sec:04.1f}s"
    return f"{sec:.2f}s"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimeMetrics:
    user_seconds: float = 0.0
    system_seconds: float = 0.0
    cpu_percent: float = 0.0
    max_rss_kb: int = 0
    fs_inputs: int = 0
    fs_outputs: int = 0
    voluntary_cs: int = 0
    involuntary_cs: int = 0

    @property
    def cpu_seconds(self) -> float:
        return self.user_seconds + self.system_seconds


@dataclass
class RunResult:
    mode: str
    processes: int
    numba_threads: int
    repeat: int
    frames_requested: int
    frames_completed: int
    wall_seconds: float
    cpu_seconds: float
    effective_cores: float
    host_cpu_percent: float
    peak_rss_gib: float
    peak_sampled_rss_gib: float
    max_threads_observed: int
    frame_total_mean_s: float
    frame_total_median_s: float
    frame_total_p95_s: float
    geometry_mean_s: float
    sgp4_mean_s: float
    cpu_fused_mean_s: float
    terrain_lite_mean_s: float
    terrain_hf_mean_s: float
    model_e_mean_s: float
    save_mean_s: float
    unaccounted_mean_s: float
    geometry_fraction: float
    frames_per_hour: float
    estimated_full_hours: float
    estimated_cpu_core_hours: float
    recommended_ram_gib: float
    success: bool
    output_dir: str
    error: str = ""


@dataclass
class Worker:
    index: int
    process: subprocess.Popen[Any]
    log_handle: Any
    log_path: Path
    time_path: Path
    output_root: Path
    schedule_path: Path


def parse_time_file(path: Path) -> TimeMetrics:
    metrics = TimeMetrics()
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for field, pattern in TIME_PATTERNS.items():
            match = pattern.match(line)
            if not match:
                continue
            value = match.group(1)
            if field in {"user_seconds", "system_seconds", "cpu_percent"}:
                setattr(metrics, field, float(value))
            else:
                setattr(metrics, field, int(value))
    return metrics


def parse_stage_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in FRAME_RE.finditer(text):
        rows.append(
            {
                "total": float(match.group("total")),
                "geometry": float(match.group("geometry")),
                "sgp4": float(match.group("sgp4") or 0.0),
                "cpu_fused": float(match.group("cpu_fused") or 0.0),
                "lite": float(match.group("lite")),
                "hf": float(match.group("hf")),
                "model_e": float(match.group("model_e")),
                "save": float(match.group("save") or 0.0),
            }
        )
    return rows


def get_hardware_info() -> dict[str, Any]:
    logical = os.cpu_count() or 1
    physical = None
    memory_total = None
    if psutil is not None:
        physical = psutil.cpu_count(logical=False)
        memory_total = psutil.virtual_memory().total

    info: dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": logical,
        "physical_cpus": physical,
        "memory_total_bytes": memory_total,
    }

    if shutil.which("lscpu"):
        completed = subprocess.run(
            ["lscpu", "-J"],
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        if completed.returncode == 0:
            try:
                info["lscpu"] = json.loads(completed.stdout)
            except json.JSONDecodeError:
                info["lscpu_text"] = completed.stdout
    return info


def load_schedule_rows(path: Path, region: str) -> tuple[list[str], list[dict[str, str]], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Schedule has no header: {path}")
        all_rows = list(reader)
        fieldnames = list(reader.fieldnames)

    valid_rows: list[dict[str, str]] = []
    for row in all_rows:
        if "no_service_flag" in row and not false_like(row["no_service_flag"]):
            continue
        if region and row.get("region", "") != region:
            continue
        valid_rows.append(row)

    total_valid_all_regions = 0
    for row in all_rows:
        if "no_service_flag" in row and not false_like(row["no_service_flag"]):
            continue
        total_valid_all_regions += 1

    return fieldnames, valid_rows, total_valid_all_regions


def write_schedule(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_process_group(workers: list[Worker]) -> tuple[float, int]:
    """Return aggregate RSS in bytes and number of threads for all live trees."""
    if psutil is None:
        return 0.0, 0
    seen: set[int] = set()
    rss = 0
    threads = 0
    for worker in workers:
        try:
            root = psutil.Process(worker.process.pid)
            processes = [root] + root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        for process in processes:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                rss += process.memory_info().rss
                threads += process.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return float(rss), threads


def run_configuration(
    *,
    root: Path,
    generator: Path,
    tle_path: Path,
    fieldnames: list[str],
    selected_rows: list[dict[str, str]],
    mode: str,
    processes: int,
    numba_threads: int,
    frames_per_worker: int,
    repeat: int,
    run_dir: Path,
    total_schedule_frames: int,
    logical_cpus: int,
    extra_generator_args: list[str],
) -> RunResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    workers: list[Worker] = []
    required_rows = processes * frames_per_worker
    if len(selected_rows) < required_rows:
        raise ValueError(
            f"Need {required_rows} schedule rows for {processes} workers, "
            f"but only {len(selected_rows)} are available"
        )

    for worker_index in range(processes):
        start = worker_index * frames_per_worker
        end = start + frames_per_worker
        subset = selected_rows[start:end]
        worker_dir = run_dir / f"worker_{worker_index:02d}"
        output_root = worker_dir / "output"
        schedule_path = worker_dir / "schedule.csv"
        log_path = worker_dir / "stdout.log"
        time_path = worker_dir / "time_verbose.txt"
        worker_dir.mkdir(parents=True, exist_ok=True)
        write_schedule(schedule_path, fieldnames, subset)

        command = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(time_path),
            sys.executable,
            str(generator),
            "--root",
            str(root),
            "--schedule-csv",
            str(schedule_path),
            "--tle-path",
            str(tle_path),
            "--output-root",
            str(output_root),
            "--max-frames",
            str(frames_per_worker),
            "--numba-threads",
            str(numba_threads),
            *extra_generator_args,
        ]
        environment = {
            **os.environ,
            "LC_ALL": "C",
            "PYTHONUNBUFFERED": "1",
            "NUMBA_NUM_THREADS": str(numba_threads),
            # Avoid hidden BLAS oversubscription during process scaling.
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
        }
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        workers.append(
            Worker(
                index=worker_index,
                process=process,
                log_handle=log_handle,
                log_path=log_path,
                time_path=time_path,
                output_root=output_root,
                schedule_path=schedule_path,
            )
        )

    wall_start = time.perf_counter()
    peak_sampled_rss = 0.0
    max_threads = 0
    while any(worker.process.poll() is None for worker in workers):
        rss, threads = sample_process_group(workers)
        peak_sampled_rss = max(peak_sampled_rss, rss)
        max_threads = max(max_threads, threads)
        time.sleep(0.25)
    wall_seconds = time.perf_counter() - wall_start

    return_codes: list[int] = []
    for worker in workers:
        return_codes.append(worker.process.wait())
        worker.log_handle.close()

    time_metrics = [parse_time_file(worker.time_path) for worker in workers]
    stage_rows: list[dict[str, float]] = []
    for worker in workers:
        stage_rows.extend(parse_stage_log(worker.log_path))

    cpu_seconds = sum(metric.cpu_seconds for metric in time_metrics)
    effective_cores = cpu_seconds / wall_seconds if wall_seconds > 0 else float("nan")
    host_cpu_percent = 100.0 * effective_cores / logical_cpus if logical_cpus else float("nan")
    # Sum of per-worker maxima is conservative. Sampled peak is closer to real aggregate RSS.
    summed_peak_rss_gib = sum(metric.max_rss_kb for metric in time_metrics) / (1024.0**2)
    sampled_peak_rss_gib = peak_sampled_rss / (1024.0**3)
    peak_rss_gib = max(summed_peak_rss_gib, sampled_peak_rss_gib)

    totals = [row["total"] for row in stage_rows]
    geometries = [row["geometry"] for row in stage_rows]
    sgp4s = [row["sgp4"] for row in stage_rows]
    cpu_fused = [row["cpu_fused"] for row in stage_rows]
    lites = [row["lite"] for row in stage_rows]
    hfs = [row["hf"] for row in stage_rows]
    model_es = [row["model_e"] for row in stage_rows]
    saves = [row["save"] for row in stage_rows]
    unaccounted = [
        max(0.0, row["total"] - row["geometry"] - row["lite"] - row["hf"] - row["model_e"] - row["save"])
        for row in stage_rows
    ]
    frames_completed = len(stage_rows)
    frames_per_hour = 3600.0 * frames_completed / wall_seconds if wall_seconds > 0 else 0.0
    estimated_full_hours = (
        total_schedule_frames / frames_per_hour if frames_per_hour > 0 else float("inf")
    )
    estimated_cpu_core_hours = estimated_full_hours * effective_cores
    recommended_ram_gib = peak_rss_gib * 1.25
    geometry_mean = statistics.fmean(geometries) if geometries else float("nan")
    frame_mean = statistics.fmean(totals) if totals else float("nan")
    geometry_fraction = geometry_mean / frame_mean if frame_mean > 0 else float("nan")
    success = all(code == 0 for code in return_codes) and frames_completed == required_rows
    error = ""
    if not success:
        error = f"return_codes={return_codes}, frames={frames_completed}/{required_rows}"

    return RunResult(
        mode=mode,
        processes=processes,
        numba_threads=numba_threads,
        repeat=repeat,
        frames_requested=required_rows,
        frames_completed=frames_completed,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        effective_cores=effective_cores,
        host_cpu_percent=host_cpu_percent,
        peak_rss_gib=peak_rss_gib,
        peak_sampled_rss_gib=sampled_peak_rss_gib,
        max_threads_observed=max_threads,
        frame_total_mean_s=frame_mean,
        frame_total_median_s=statistics.median(totals) if totals else float("nan"),
        frame_total_p95_s=percentile(totals, 0.95),
        geometry_mean_s=geometry_mean,
        sgp4_mean_s=statistics.fmean(sgp4s) if sgp4s else float("nan"),
        cpu_fused_mean_s=statistics.fmean(cpu_fused) if cpu_fused else float("nan"),
        terrain_lite_mean_s=statistics.fmean(lites) if lites else float("nan"),
        terrain_hf_mean_s=statistics.fmean(hfs) if hfs else float("nan"),
        model_e_mean_s=statistics.fmean(model_es) if model_es else float("nan"),
        save_mean_s=statistics.fmean(saves) if saves else float("nan"),
        unaccounted_mean_s=statistics.fmean(unaccounted) if unaccounted else float("nan"),
        geometry_fraction=geometry_fraction,
        frames_per_hour=frames_per_hour,
        estimated_full_hours=estimated_full_hours,
        estimated_cpu_core_hours=estimated_cpu_core_hours,
        recommended_ram_gib=recommended_ram_gib,
        success=success,
        output_dir=str(run_dir),
        error=error,
    )


def write_results_csv(path: Path, results: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_configs(results: list[RunResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[RunResult]] = {}
    for result in results:
        if not result.success:
            continue
        groups.setdefault((result.mode, result.processes, result.numba_threads), []).append(result)

    summary: list[dict[str, Any]] = []
    for (mode, processes, threads), group in sorted(groups.items()):
        def med(field: str) -> float:
            return statistics.median(float(getattr(item, field)) for item in group)

        summary.append(
            {
                "mode": mode,
                "processes": processes,
                "numba_threads": threads,
                "repeats": len(group),
                "frames_per_hour": med("frames_per_hour"),
                "effective_cores": med("effective_cores"),
                "host_cpu_percent": med("host_cpu_percent"),
                "peak_rss_gib": max(item.peak_rss_gib for item in group),
                "recommended_ram_gib": max(item.recommended_ram_gib for item in group),
                "frame_total_median_s": med("frame_total_median_s"),
                "geometry_mean_s": med("geometry_mean_s"),
                "sgp4_mean_s": med("sgp4_mean_s"),
                "cpu_fused_mean_s": med("cpu_fused_mean_s"),
                "terrain_lite_mean_s": med("terrain_lite_mean_s"),
                "terrain_hf_mean_s": med("terrain_hf_mean_s"),
                "model_e_mean_s": med("model_e_mean_s"),
                "save_mean_s": med("save_mean_s"),
                "unaccounted_mean_s": med("unaccounted_mean_s"),
                "geometry_fraction": med("geometry_fraction"),
                "estimated_full_hours": med("estimated_full_hours"),
                "estimated_cpu_core_hours": med("estimated_cpu_core_hours"),
            }
        )
    return summary


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 模式 | 进程 | 线程/进程 | 帧/小时 | 有效CPU核 | 峰值内存 | 单帧 | SGP4 | CPU融合 | D-lite | D-HF | 保存 | 全天预计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {processes} | {numba_threads} | {frames_per_hour:.1f} | "
            "{effective_cores:.2f} | {peak_rss_gib:.1f} GiB | "
            "{frame_total_median_s:.3f}s | {sgp4_mean_s:.3f}s | "
            "{cpu_fused_mean_s:.3f}s | {terrain_lite_mean_s:.3f}s | "
            "{terrain_hf_mean_s:.3f}s | {save_mean_s:.3f}s | {full} |".format(
                **row,
                full=human_seconds(row["estimated_full_hours"] * 3600.0),
            )
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    hardware: dict[str, Any],
    summary: list[dict[str, Any]],
    total_frames: int,
    region: str,
    target_hours: float,
) -> None:
    logical = hardware.get("logical_cpus") or 1
    successful = [row for row in summary if row["frames_per_hour"] > 0]
    best = max(successful, key=lambda row: row["frames_per_hour"]) if successful else None
    efficient = max(
        successful,
        key=lambda row: row["frames_per_hour"] / max(row["effective_cores"], 1e-9),
    ) if successful else None

    lines = [
        "# Dataset-v0 直接生成 CPU 性能分析",
        "",
        f"- 测试区域：`{region}`",
        f"- 全天估算帧数：`{total_frames}`",
        f"- 主机逻辑CPU：`{logical}`",
        f"- 主机物理CPU：`{hardware.get('physical_cpus')}`",
        "",
        "## 配置结果",
        "",
        markdown_table(summary) if summary else "没有成功的测试结果。",
        "",
        "## 自动结论",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- 当前最高吞吐配置：**{best['processes']}进程 × "
                f"{best['numba_threads']} Numba线程/进程**。",
                f"- 实测吞吐：**{best['frames_per_hour']:.1f} 帧/小时**。",
                f"- 实际平均CPU需求：**{best['effective_cores']:.2f} 个满负载逻辑核**，"
                f"约占本机 {best['host_cpu_percent']:.1f}%。",
                f"- 峰值内存约 **{best['peak_rss_gib']:.1f} GiB**，"
                f"建议准备至少 **{best['recommended_ram_gib']:.1f} GiB**。",
                f"- 生成全部 {total_frames} 帧预计需要 **"
                f"{human_seconds(best['estimated_full_hours'] * 3600.0)}**。",
                f"- 预计总CPU成本约 **{best['estimated_cpu_core_hours']:.1f} 核时**。",
            ]
        )
        required_rate = total_frames / target_hours
        node_count = max(1, math.ceil(required_rate / best["frames_per_hour"]))
        lines.extend(
            [
                f"- 若希望在 **{target_hours:g}小时** 内完成，需要平均 **{required_rate:.1f}帧/小时**。",
                f"- 按当前最佳实测节点配置估算，需要约 **{node_count} 个同等节点**；"
                f"总有效CPU约 **{node_count * best['effective_cores']:.1f}核**，"
                f"每节点内存建议不低于 **{best['recommended_ram_gib']:.1f} GiB**。",
            ]
        )
    if efficient:
        efficiency = efficient["frames_per_hour"] / max(efficient["effective_cores"], 1e-9)
        lines.append(
            f"- 单核效率最高配置：**{efficient['processes']}进程 × "
            f"{efficient['numba_threads']}线程**，约 **{efficiency:.1f} 帧/(核·小时)**。"
        )

    lines.extend(
        [
            "",
            "## 资源采购/申请时应重点看",
            "",
            "1. `effective_cores`：代码实际持续占用的CPU核数，而不是配置的线程数。",
            "2. `frames_per_hour`：决定大规模数据集的实际交付时间。",
            "3. `peak_rss_gib`：多进程时内存基本按进程数增长，必须留出20%～30%余量。",
            "4. `CPU-fused`、`D-lite`、`D-HF`：比较线程增长后各阶段是否继续下降，判断增加核心是否有效。",
            "5. 多进程测试用于评估整机吞吐；线程测试用于确定单进程最合适的Numba线程数。",
            "",
            "## 解释",
            "",
            "- `有效CPU核 = 所有进程的(user+system CPU秒) / 墙钟秒`。",
            "- `CPU核时 = 有效CPU核 × 全天预计小时数`，适合估算集群配额。",
            "- 进程扩展测试采用每个进程相同帧数的弱扩展方式，反映节点吞吐能力。",
            "- 首次Numba编译会污染结果，报告应以预热后的重复测试中位数为准。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_warmup(
    root: Path,
    generator: Path,
    tle_path: Path,
    fieldnames: list[str],
    selected_rows: list[dict[str, str]],
    out_dir: Path,
    threads: int,
    extra_args: list[str],
) -> None:
    print(f"[warmup] Compiling/warming Numba with {threads} threads...")
    result = run_configuration(
        root=root,
        generator=generator,
        tle_path=tle_path,
        fieldnames=fieldnames,
        selected_rows=selected_rows,
        mode="warmup",
        processes=1,
        numba_threads=threads,
        frames_per_worker=1,
        repeat=0,
        run_dir=out_dir / "warmup",
        total_schedule_frames=1,
        logical_cpus=os.cpu_count() or 1,
        extra_generator_args=extra_args,
    )
    if not result.success:
        raise RuntimeError(f"Warmup failed: {result.error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU and memory needs for direct Dataset-v0 generation."
    )
    parser.add_argument("--root", type=Path, default=Path("/home/liuzhongkai/DEM_pre"))
    parser.add_argument(
        "--generator", type=Path, default=Path("src/generate_dataset_minimal_direct.py")
    )
    parser.add_argument(
        "--schedule", type=Path, default=Path("results/dataset_v0/service_schedule_20s.csv")
    )
    parser.add_argument(
        "--tle", type=Path, default=Path("data/l1_space/2025-01-01_leo_payload.tle")
    )
    parser.add_argument("--region", default="qinling")
    parser.add_argument(
        "--mode", choices=["threads", "processes", "all"], default="all"
    )
    parser.add_argument("--thread-list", type=parse_int_list, default=parse_int_list("1,2,4,8,16"))
    parser.add_argument("--process-list", type=parse_int_list, default=parse_int_list("1,2,4"))
    parser.add_argument(
        "--process-numba-threads",
        type=int,
        default=2,
        help="Numba threads per worker during process scaling tests.",
    )
    parser.add_argument("--frames-per-worker", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--keep-generated-data", action="store_true")
    parser.add_argument("--estimate-frames", type=int, default=0)
    parser.add_argument("--target-hours", type=float, default=24.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--extra-generator-arg",
        action="append",
        default=[],
        help="Pass one extra argument token to the generator; may be repeated.",
    )
    args = parser.parse_args()

    if args.frames_per_worker <= 0 or args.repeats <= 0 or args.target_hours <= 0:
        parser.error("--frames-per-worker and --repeats must be positive")
    if args.process_numba_threads <= 0:
        parser.error("--process-numba-threads must be positive")
    if not Path("/usr/bin/time").exists():
        parser.error("GNU time is required at /usr/bin/time")

    root = args.root.expanduser().resolve()
    generator = (root / args.generator).resolve() if not args.generator.is_absolute() else args.generator.resolve()
    schedule = (root / args.schedule).resolve() if not args.schedule.is_absolute() else args.schedule.resolve()
    tle_path = (root / args.tle).resolve() if not args.tle.is_absolute() else args.tle.resolve()
    for required in [generator, schedule, tle_path]:
        if not required.exists():
            parser.error(f"Required path does not exist: {required}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "results" / "cpu_profile" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    hardware = get_hardware_info()
    (output_dir / "hardware.json").write_text(
        json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fieldnames, region_rows, schedule_total = load_schedule_rows(schedule, args.region)
    total_schedule_frames = args.estimate_frames or schedule_total
    max_processes = max(args.process_list) if args.mode in {"processes", "all"} else 1
    needed_rows = max_processes * args.frames_per_worker
    if len(region_rows) < needed_rows:
        parser.error(
            f"Region {args.region!r} has {len(region_rows)} valid rows, "
            f"but benchmarks require at least {needed_rows}"
        )

    print("=" * 72)
    print("Dataset CPU profiler")
    print(f"Root: {root}")
    print(f"Generator: {generator}")
    print(f"Schedule: {schedule}")
    print(f"Region: {args.region}, available rows: {len(region_rows)}")
    print(f"Full-dataset estimate frames: {total_schedule_frames}")
    print(f"Logical CPUs: {hardware.get('logical_cpus')}")
    print(f"Output: {output_dir}")
    print("=" * 72)

    if not args.no_warmup:
        warmup_threads = max(max(args.thread_list), args.process_numba_threads)
        run_warmup(
            root,
            generator,
            tle_path,
            fieldnames,
            region_rows,
            output_dir,
            warmup_threads,
            args.extra_generator_arg,
        )

    results: list[RunResult] = []
    logical_cpus = int(hardware.get("logical_cpus") or 1)

    configs: list[tuple[str, int, int]] = []
    if args.mode in {"threads", "all"}:
        configs.extend(("threads", 1, threads) for threads in args.thread_list)
    if args.mode in {"processes", "all"}:
        configs.extend(
            ("processes", processes, args.process_numba_threads)
            for processes in args.process_list
        )

    for mode, processes, threads in configs:
        for repeat in range(1, args.repeats + 1):
            name = f"{mode}_p{processes}_t{threads}_r{repeat}"
            run_dir = output_dir / "runs" / name
            print(
                f"\n[run] mode={mode}, processes={processes}, "
                f"threads/process={threads}, repeat={repeat}"
            )
            result = run_configuration(
                root=root,
                generator=generator,
                tle_path=tle_path,
                fieldnames=fieldnames,
                selected_rows=region_rows,
                mode=mode,
                processes=processes,
                numba_threads=threads,
                frames_per_worker=args.frames_per_worker,
                repeat=repeat,
                run_dir=run_dir,
                total_schedule_frames=total_schedule_frames,
                logical_cpus=logical_cpus,
                extra_generator_args=args.extra_generator_arg,
            )
            results.append(result)
            print(
                f"  success={result.success}, wall={human_seconds(result.wall_seconds)}, "
                f"throughput={result.frames_per_hour:.1f} frames/h, "
                f"effective_cores={result.effective_cores:.2f}, "
                f"peak_ram={result.peak_rss_gib:.1f} GiB, "
                f"full_estimate={human_seconds(result.estimated_full_hours * 3600.0)}"
            )
            write_results_csv(output_dir / "profile_results.csv", results)

            if result.success and not args.keep_generated_data:
                for worker_output in run_dir.glob("worker_*/output"):
                    shutil.rmtree(worker_output, ignore_errors=True)

    summary = aggregate_configs(results)
    with (output_dir / "profile_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        if summary:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    write_report(
        output_dir / "CPU_PERFORMANCE_REPORT.md",
        hardware,
        summary,
        total_schedule_frames,
        args.region,
        args.target_hours,
    )

    print("\n" + "=" * 72)
    print("Profiling finished")
    print(f"Raw results: {output_dir / 'profile_results.csv'}")
    print(f"Summary: {output_dir / 'profile_summary.csv'}")
    print(f"Report: {output_dir / 'CPU_PERFORMANCE_REPORT.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
