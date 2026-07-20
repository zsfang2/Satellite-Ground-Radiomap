#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/liuzhongkai/DEM_pre}
TLE=${TLE:-$ROOT/data/l1_space/2025-01-01_leo_payload.tle}
REGION=${REGION:-qinling}
NUMBA_THREADS=${NUMBA_THREADS:-16}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/results/direct_cpu_benchmark}

cd "$ROOT"
rm -rf "$OUTPUT_ROOT"

# Keep BLAS helper libraries from silently oversubscribing CPU cores.
export NUMBA_NUM_THREADS="$NUMBA_THREADS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

/usr/bin/time -f '\nElapsed: %E\nCPU: %P\nMax memory: %M KB' \
  python src/generate_dataset_minimal_direct.py \
    --root "$ROOT" \
    --schedule-csv "$ROOT/results/dataset_v0/service_schedule_20s.csv" \
    --tle-path "$TLE" \
    --output-root "$OUTPUT_ROOT" \
    --core-size-km 76.8 \
    --buffer-km 40 \
    --resolution-m 100 \
    --freq-ghz 14.5 \
    --numba-threads "$NUMBA_THREADS" \
    --regions "$REGION" \
    --max-frames 1 \
    "$@"
