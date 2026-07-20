#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def hours_text(hours: float) -> str:
    total_minutes = int(round(hours * 60))
    return f"{total_minutes // 60}小时{total_minutes % 60:02d}分钟"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable compute-requirement report.")
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--effective-cores", type=float, default=18.12)
    parser.add_argument("--days", type=int, default=1)
    args = parser.parse_args()

    summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    frames = int(summary["complete_frames"])
    wall_hours = float(summary["generation_wall_seconds"]) / 3600.0
    throughput = float(summary["throughput_frames_per_hour"])
    size_gib = float(summary["dataset_size_gib"])
    core_hours = wall_hours * args.effective_cores
    target_frames = frames * args.days
    target_size = size_gib * args.days
    target_wall = wall_hours * args.days
    target_core_hours = core_hours * args.days

    report = f"""# Dataset-v0 CPU算力需求方案

## 1. 实测基线

- 数据范围：2025-01-01全天，4个区域，20秒/帧。
- 总帧数：{frames:,}帧。
- 并行方式：{summary['workers']}个进程 × 每进程{summary['threads_per_worker']}个Numba线程。
- 实测生成时间：{hours_text(wall_hours)}。
- 实测吞吐：{throughput:,.1f}帧/小时。
- 数据集实际大小：{size_gib:.2f} GiB。
- 实测有效CPU核参考值：{args.effective_cores:.2f}核。
- 估算CPU核时：{core_hours:.1f}核时/天。

## 2. 对外算力申请规格

### 推荐节点

- CPU：x86_64，至少20个物理核心；推荐24个物理核心或以上。
- 逻辑CPU：至少36线程；推荐48线程或以上。
- 内存：最低16 GB；推荐32 GB。
- 临时本地存储：NVMe SSD，可用空间不少于{max(350.0, math.ceil(size_gib * 1.35 / 50) * 50):.0f} GiB/天。
- GPU：不需要。
- 操作系统：Linux。
- 软件环境：Python 3.10、NumPy、Pandas、Numba、Skyfield、PyYAML。
- 调度方式：{summary['workers']}个独立生成进程，每进程{summary['threads_per_worker']}个Numba线程；OMP/MKL/OpenBLAS均限制为1线程。

### 性能验收指标

硬件型号不同，不能只按“核数”验收，建议同时给出实际吞吐要求：

- 最低可接受吞吐：8,000帧/小时。
- 推荐吞吐：9,500帧/小时或以上。
- 目标：一天{frames:,}帧在2.5小时内完成。
- 生成过程中不得出现缺帧、重复帧或NaN/Inf数组。

## 3. 扩展到{args.days}天的资源量

- 总帧数：{target_frames:,}帧。
- 预计墙钟时间：{hours_text(target_wall)}（单个同等节点顺序生成）。
- 预计CPU配额：约{target_core_hours:.1f}核时。
- 预计存储：约{target_size:.1f} GiB。
- 建议申请CPU配额：{math.ceil(target_core_hours * 1.2):d}核时，含20%运行余量。
- 建议申请存储：{math.ceil(target_size * 1.25 / 100) * 100:d} GiB，含校验、日志和临时文件余量。

## 4. 运行约束

1. 数据生成必须使用本地NVMe作为工作目录，不建议直接在机械盘或高延迟网络盘并发写入。
2. 每帧保存6张float32地图和1个meta.json；完成后统一重建manifest并校验总帧数。
3. 支持断点续跑，已存在且含gt_pr_dbm.npy的帧跳过。
4. 正式运行前先执行4场景样本测试与数值一致性验证。
"""
    output = args.output or args.run_summary.with_name("COMPUTE_REQUIREMENT_REPORT.md")
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
