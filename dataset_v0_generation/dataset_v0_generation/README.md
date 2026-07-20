# Dataset-v0 CPU Generation Pipeline

This directory is a clean export of the finalized Dataset-v0 generation code.

## Final per-frame files

- `coarse_pr_dbm.npy`
- `gt_pr_dbm.npy`
- `terrain_loss_lite_db.npy`
- `terrain_loss_hf_db.npy`
- `weather_loss_db.npy`
- `residual_dhf_to_e_db.npy`
- `meta.json`

## Main features

- CPU-only ECEF/ENU geometry acceleration
- one SGP4 propagation per frame
- fused Geometry + Model-A + Model-B Numba kernel
- parallel D-Lite and D-HF terrain computation
- resumable multi-process generation
- atomic frame writes
- final integrity checking and manifest generation
- CPU performance profiling
- NORAD leading-zero normalization

## Validated result

- 8,640 frames
- 12 processes x 2 Numba threads
- 41 min 06 s
- 12,612 frames/hour
- 113.97 GiB
- 0 finalization errors

Large input/output data are intentionally excluded.
