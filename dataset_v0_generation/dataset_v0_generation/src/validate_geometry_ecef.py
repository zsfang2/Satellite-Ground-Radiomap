#!/usr/bin/env python3
"""Validate optimized ECEF/ENU geometry against the legacy Skyfield path.

The legacy path is evaluated only at a small set of pixels, so validation is
fast while still checking elevation, azimuth, slant range, and Model-B power.
"""
from __future__ import annotations

import argparse
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from skyfield.api import load, wgs84

from generate_dataset_minimal_direct import (
    build_tle_catalog,
    compute_geometry_and_model_b,
    parse_utc,
    prepare_region_state,
    select_satellite,
)


def circular_difference_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def make_generation_args(root: Path) -> Namespace:
    return Namespace(
        root=root,
        core_size_km=76.8,
        buffer_km=40.0,
        resolution_m=100.0,
        freq_ghz=14.5,
        tx_power_dbw=10.0,
        rx_gain_dbi=0.0,
        array_rows=64,
        array_cols=64,
        element_gain_dbi=5.0,
        efficiency_loss_db=3.0,
        max_atten_db=35.0,
        scan_loss_q=1.3,
        disable_scan_loss=False,
        gas_zenith_db=0.08,
        cloud_zenith_db=0.10,
        atm_min_elev_deg=5.0,
        model_e_seed=20250101,
        rain_max_loss_db=8.0,
        clutter_max_loss_db=2.0,
    )


def legacy_sample_model_b(
    state,
    satellite,
    scalar_time,
    timescale,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    args: Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = state.lat_grid[row_indices, col_indices]
    lon = state.lon_grid[row_indices, col_indices]
    height = state.core_dem[row_indices, col_indices].astype(np.float64)
    observers = wgs84.latlon(lat, lon, elevation_m=height)
    vector_time = timescale.tt_jd(
        np.full(row_indices.size, scalar_time.tt, dtype=np.float64)
    )
    topocentric = (satellite - observers).at(vector_time)
    altitude, azimuth, distance = topocentric.altaz()
    elevation = altitude.degrees.astype(np.float64)
    azimuth_deg = azimuth.degrees.astype(np.float64)
    slant_km = distance.km.astype(np.float64)

    center_row = state.core_dem.shape[0] // 2
    center_col = state.core_dem.shape[1] // 2
    center_observer = wgs84.latlon(
        float(state.lat_grid[center_row, center_col]),
        float(state.lon_grid[center_row, center_col]),
        elevation_m=float(state.core_dem[center_row, center_col]),
    )
    center_topocentric = (satellite - center_observer).at(scalar_time)
    center_vector = -center_topocentric.position.km
    center_vector /= np.linalg.norm(center_vector)

    sat_xyz = satellite.at(scalar_time).position.km
    nadir_vector = -sat_xyz / np.linalg.norm(sat_xyz)
    cos_scan = float(np.clip(nadir_vector @ center_vector, 1.0e-6, 1.0))
    scan_loss_db = max(
        0.0, -10.0 * args.scan_loss_q * math.log10(cos_scan)
    )

    observer_from_satellite = -topocentric.position.km
    observer_from_satellite /= np.linalg.norm(
        observer_from_satellite, axis=0, keepdims=True
    )
    cos_theta = np.clip(center_vector @ observer_from_satellite, -1.0, 1.0)
    off_axis_deg = np.degrees(np.arccos(cos_theta))

    freq_mhz = args.freq_ghz * 1000.0
    fspl_db = (
        32.45
        + 20.0 * np.log10(freq_mhz)
        + 20.0 * np.log10(slant_km)
    )
    ideal_peak_gain = args.element_gain_dbi + 10.0 * np.log10(
        args.array_rows * args.array_cols
    )
    effective_peak_gain = (
        ideal_peak_gain - args.efficiency_loss_db - scan_loss_db
    )
    theta_3db_deg = (101.4 / max(args.array_rows, args.array_cols)) / 2.0
    attenuation = np.minimum(
        3.0 * (off_axis_deg / theta_3db_deg) ** 2,
        args.max_atten_db,
    )
    model_a = (
        args.tx_power_dbw
        + effective_peak_gain
        - attenuation
        + args.rx_gain_dbi
        - fspl_db
        + 30.0
    )
    used_elevation = np.maximum(elevation, args.atm_min_elev_deg)
    atmosphere = (
        args.gas_zenith_db + args.cloud_zenith_db
    ) / np.sin(np.deg2rad(used_elevation))
    return elevation, azimuth_deg, slant_km, model_a - atmosphere


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--schedule-csv",
        type=Path,
        default=Path("results/dataset_v0/service_schedule_20s.csv"),
    )
    parser.add_argument(
        "--tle-path",
        type=Path,
        default=Path("data/l1_space/2025-01-01_leo_payload.tle"),
    )
    parser.add_argument("--region", default="qinling")
    parser.add_argument("--samples-per-axis", type=int, default=5)
    args_cli = parser.parse_args()

    root = args_cli.root.expanduser().resolve()
    schedule_path = (
        args_cli.schedule_csv
        if args_cli.schedule_csv.is_absolute()
        else root / args_cli.schedule_csv
    )
    tle_path = (
        args_cli.tle_path
        if args_cli.tle_path.is_absolute()
        else root / args_cli.tle_path
    )

    schedule = pd.read_csv(schedule_path)
    schedule = schedule[
        (schedule["region"] == args_cli.region)
        & (schedule["no_service_flag"] == False)  # noqa: E712
    ].sort_values("frame_id")
    if schedule.empty:
        raise RuntimeError(f"No service frame for {args_cli.region}")
    row = schedule.iloc[0]

    ts = load.timescale()
    catalog = build_tle_catalog(tle_path, ts)
    target_time = parse_utc(str(row["time_utc"]))
    scalar_time = ts.from_datetime(target_time)
    satellite = select_satellite(
        catalog, str(int(row["service_sat_id"])), float(scalar_time.tt)
    )

    generation_args = make_generation_args(root)
    validation_output = root / "results" / "geometry_validation"
    state = prepare_region_state(
        root, validation_output, args_cli.region, generation_args
    )
    elevation, azimuth, slant_km, model_b, timings = (
        compute_geometry_and_model_b(
            state, satellite, target_time, ts, generation_args
        )
    )

    rows, cols = state.core_dem.shape
    row_values = np.linspace(
        0, rows - 1, args_cli.samples_per_axis, dtype=int
    )
    col_values = np.linspace(
        0, cols - 1, args_cli.samples_per_axis, dtype=int
    )
    rr, cc = np.meshgrid(row_values, col_values, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()

    old_e, old_a, old_s, old_b = legacy_sample_model_b(
        state, satellite, scalar_time, ts, rr, cc, generation_args
    )
    new_e = elevation[rr, cc].astype(np.float64)
    new_a = azimuth[rr, cc].astype(np.float64)
    new_s = slant_km[rr, cc].astype(np.float64)
    new_b = model_b[rr, cc].astype(np.float64)

    metrics = {
        "elevation_max_abs_deg": float(np.max(np.abs(new_e - old_e))),
        "azimuth_max_circular_deg": float(
            np.max(circular_difference_deg(new_a, old_a))
        ),
        "slant_max_abs_m": float(np.max(np.abs(new_s - old_s)) * 1000.0),
        "model_b_max_abs_db": float(np.max(np.abs(new_b - old_b))),
    }
    print("Optimized timings:", timings)
    for key, value in metrics.items():
        print(f"{key}: {value:.9g}")

    # Tolerances are deliberately stricter than what matters for 100 m pixels.
    passed = (
        metrics["elevation_max_abs_deg"] < 1.0e-3
        and metrics["azimuth_max_circular_deg"] < 1.0e-3
        and metrics["slant_max_abs_m"] < 2.0
        and metrics["model_b_max_abs_db"] < 1.0e-3
    )
    if not passed:
        raise SystemExit("Validation FAILED")
    print("Validation PASSED")


if __name__ == "__main__":
    main()
