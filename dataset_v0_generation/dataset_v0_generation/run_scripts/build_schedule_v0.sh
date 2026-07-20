#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/liuzhongkai/DEM_pre}
TLE=${TLE:-$ROOT/data/l1_space/2025-01-01_leo_payload.tle}
OUTPUT=${OUTPUT:-$ROOT/results/dataset_v0/service_schedule_20s.csv}

cd "$ROOT"

# The frame interval is fixed at 20 seconds inside the Python script.
# Time range options are forwarded from this shell script, for example:
#   bash run_scripts/build_schedule_v0.sh \
#     --start-time 2025-01-01T00:00:00Z \
#     --duration-hours 24
# or:
#   bash run_scripts/build_schedule_v0.sh \
#     --start-time 2025-01-01T00:00:00Z \
#     --end-time 2025-01-02T00:00:00Z
python src/build_service_schedule_20s.py \
  --root "$ROOT" \
  --regions-json "$ROOT/config/regions_dataset_v0.json" \
  --tle-path "$TLE" \
  --min-elev-deg 25 \
  --output "$OUTPUT" \
  "$@"
