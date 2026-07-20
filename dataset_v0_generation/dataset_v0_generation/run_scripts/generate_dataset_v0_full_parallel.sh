#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/liuzhongkai/DEM_pre}
WORKERS=${WORKERS:-12}
THREADS_PER_WORKER=${THREADS_PER_WORKER:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/results/dataset_v0}
EXPECTED_FRAMES=${EXPECTED_FRAMES:-17280}
MIN_FREE_GIB=${MIN_FREE_GIB:-300}

cd "$ROOT"

exec python scripts/generate_dataset_parallel.py \
  --root "$ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  --threads-per-worker "$THREADS_PER_WORKER" \
  --expected-frames "$EXPECTED_FRAMES" \
  --min-free-gib "$MIN_FREE_GIB" \
  "$@"
