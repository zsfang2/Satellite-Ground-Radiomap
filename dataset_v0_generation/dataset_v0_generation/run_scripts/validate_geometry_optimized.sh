#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/liuzhongkai/DEM_pre}
cd "$ROOT"
python src/validate_geometry_ecef.py \
  --root "$ROOT" \
  --schedule-csv "$ROOT/results/dataset_v0/service_schedule_20s.csv" \
  --tle-path "$ROOT/data/l1_space/2025-01-01_leo_payload.tle" \
  --region "${REGION:-qinling}" \
  --samples-per-axis "${SAMPLES_PER_AXIS:-5}"
