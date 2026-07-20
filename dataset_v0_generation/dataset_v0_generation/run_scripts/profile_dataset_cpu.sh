#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/liuzhongkai/DEM_pre}
THREAD_LIST=${THREAD_LIST:-1,2,4,8,16}
PROCESS_LIST=${PROCESS_LIST:-1,2,4}
PROCESS_NUMBA_THREADS=${PROCESS_NUMBA_THREADS:-2}
FRAMES_PER_WORKER=${FRAMES_PER_WORKER:-2}
REPEATS=${REPEATS:-2}
REGION=${REGION:-qinling}
TARGET_HOURS=${TARGET_HOURS:-24}

cd "$ROOT"

python scripts/profile_dataset_cpu.py \
  --root "$ROOT" \
  --mode all \
  --thread-list "$THREAD_LIST" \
  --process-list "$PROCESS_LIST" \
  --process-numba-threads "$PROCESS_NUMBA_THREADS" \
  --frames-per-worker "$FRAMES_PER_WORKER" \
  --repeats "$REPEATS" \
  --region "$REGION" \
  --target-hours "$TARGET_HOURS" \
  "$@"
