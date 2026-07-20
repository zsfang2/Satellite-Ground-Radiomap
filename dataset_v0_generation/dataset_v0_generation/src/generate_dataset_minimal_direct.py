#!/usr/bin/env python3
"""Direct, low-I/O generator for the Minimal-3 Dataset-v0 profile.

Unlike the legacy orchestrator, this program does not launch six Python
subprocesses and does not write intermediate Geometry/Model-A/B/D/E folders.
It keeps intermediate arrays in memory and writes only the final dataset:

Per frame:
  coarse_pr_dbm.npy
  gt_pr_dbm.npy
  terrain_loss_lite_db.npy
  terrain_loss_hf_db.npy
  weather_loss_db.npy
  residual_dhf_to_e_db.npy
  meta.json

Per scene:
  scenes/<region>/dem_m.npy
  scenes/<region>/clutter_loss_db.npy
  scenes/<region>/grid_meta.json

The terrain calculation uses Numba parallel loops when Numba is installed.
Geometry uses one Skyfield SGP4/ITRS propagation per frame and a CPU-only
WGS-84 ECEF/ENU calculation fused with Model-A/B in a Numba parallel kernel.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from skyfield.api import EarthSatellite, load
from skyfield.framelib import itrs

try:
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for environments without Numba
    NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

    def set_num_threads(_count: int) -> None:
        return None



EARTH_RADIUS_M = 6_371_000.0
WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass
class RegionState:
    region: str
    metadata_path: Path
    metadata: dict[str, Any]
    center_lon: float
    center_lat: float
    resolution_m: float
    core_dem: np.ndarray
    expanded_dem: np.ndarray
    core_slice: tuple[int, int, int, int]
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    ground_ecef_m: np.ndarray
    east_unit: np.ndarray
    north_unit: np.ndarray
    up_unit: np.ndarray
    weather_x_km: np.ndarray
    weather_y_km: np.ndarray
    rain_cells: np.ndarray
    rain_texture: np.ndarray
    clutter_loss: np.ndarray


def parse_utc(value: str) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def safe_time_tag(time_utc: str) -> str:
    return str(time_utc).replace("-", "").replace(":", "")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def load_meta(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def center_from_meta(meta: dict[str, Any]) -> tuple[str, float, float, float]:
    region = str(meta.get("region_name", meta.get("region", "region")))
    lon = float(meta.get("center_lon", meta.get("center_lon_deg")))
    lat = float(meta.get("center_lat", meta.get("center_lat_deg")))
    resolution = float(meta.get("resolution_m"))
    return region, lon, lat, resolution


def metadata_path_for_region(
    root: Path,
    region: str,
    core_size_km: float,
    buffer_km: float,
    resolution_m: float,
) -> Path:
    dem_dir = root / "data" / "l2_topo" / "DEM_data_process" / "dem"
    patterns = (
        f"{region}_lon*-lat*_core{core_size_km:.1f}km_buf{buffer_km:.1f}km_ext*_{int(round(resolution_m))}m_metadata.json",
        f"{region}_lon*-lat*_core{core_size_km:g}km_buf{buffer_km:g}km_ext*_{int(round(resolution_m))}m_metadata.json",
        f"{region}_*_metadata.json",
    )
    for pattern in patterns:
        matches = sorted(dem_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find DEM metadata for region={region} in {dem_dir}")


def resolve_dem_paths(
    meta: dict[str, Any],
    metadata_path: Path,
    core_size_km: float,
    buffer_km: float,
    resolution_m: float,
) -> tuple[Path, Path, tuple[int, int, int, int]]:
    files = meta.get("files", {}) if isinstance(meta.get("files", {}), dict) else {}
    if "expanded_dem_npy" in files and "core_dem_npy" in files:
        expanded = Path(files["expanded_dem_npy"])
        core = Path(files["core_dem_npy"])
        if not expanded.is_absolute():
            expanded = (metadata_path.parent / expanded).resolve()
        if not core.is_absolute():
            core = (metadata_path.parent / core).resolve()
    else:
        region, lon, lat, _ = center_from_meta(meta)
        extent_km = core_size_km + 2.0 * buffer_km
        tag = (
            f"{region}_lon{lon:.2f}-lat{lat:.2f}_core{core_size_km:.1f}km_"
            f"buf{buffer_km:.1f}km_ext{extent_km:.1f}km_{int(round(resolution_m))}m"
        )
        expanded = metadata_path.parent / f"{tag}_expanded.npy"
        core = metadata_path.parent / f"{tag}_core.npy"

    for path in (expanded, core):
        if not path.is_file():
            raise FileNotFoundError(path)

    core_shape = tuple(int(x) for x in meta.get("core_shape_yx", []))
    if len(core_shape) != 2:
        n = int(round(core_size_km * 1000.0 / resolution_m))
        core_shape = (n, n)

    core_slice_meta = meta.get("core_slice_yx")
    if core_slice_meta:
        y0 = int(core_slice_meta["y_start"])
        y1 = int(core_slice_meta["y_stop"])
        x0 = int(core_slice_meta["x_start"])
        x1 = int(core_slice_meta["x_stop"])
    else:
        buffer_pixels = int(round(buffer_km * 1000.0 / resolution_m))
        y0, x0 = buffer_pixels, buffer_pixels
        y1, x1 = y0 + core_shape[0], x0 + core_shape[1]

    return expanded, core, (y0, y1, x0, x1)


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256((text + str(base_seed)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def correlated_field(
    rows: int,
    cols: int,
    pixel_size_m: float,
    corr_len_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=(rows, cols))
    fy = np.fft.fftfreq(rows, d=pixel_size_m)
    fx = np.fft.fftfreq(cols, d=pixel_size_m)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    filt = np.exp(
        -0.5
        * (2.0 * np.pi * corr_len_m) ** 2
        * (fx_grid * fx_grid + fy_grid * fy_grid)
    )
    field = np.fft.ifft2(np.fft.fft2(white) * filt).real
    field -= np.mean(field)
    field /= np.std(field) + 1e-12
    return field.astype(np.float32)



def geodetic_grid_to_ecef_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    height_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute WGS-84 ECEF coordinates and local ENU unit vectors.

    These arrays are fixed for a scene and are reused by every time frame.
    Using them avoids constructing hundreds of thousands of Skyfield observer
    objects and repeating the same Earth-fixed coordinate transforms.
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    height = np.asarray(height_m, dtype=np.float64)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    prime_vertical = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (prime_vertical + height) * cos_lat * cos_lon
    y = (prime_vertical + height) * cos_lat * sin_lon
    z = (prime_vertical * (1.0 - WGS84_E2) + height) * sin_lat
    ground_ecef = np.stack((x, y, z), axis=0)

    zeros = np.zeros_like(lat)
    east = np.stack((-sin_lon, cos_lon, zeros), axis=0)
    north = np.stack(
        (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), axis=0
    )
    up = np.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), axis=0)

    return (
        np.ascontiguousarray(ground_ecef, dtype=np.float64),
        np.ascontiguousarray(east, dtype=np.float64),
        np.ascontiguousarray(north, dtype=np.float64),
        np.ascontiguousarray(up, dtype=np.float64),
    )


def prepare_weather_state(
    region: str,
    rows: int,
    cols: int,
    resolution_m: float,
    seed: int,
    rain_max_loss_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute values that the legacy Model-E recalculated every frame."""
    rng = np.random.default_rng(stable_seed(region + "_rain", seed))
    x_km = (np.arange(cols) - (cols - 1) / 2.0) * resolution_m / 1000.0
    y_km = ((rows - 1) / 2.0 - np.arange(rows)) * resolution_m / 1000.0
    x_grid, y_grid = np.meshgrid(x_km, y_km)

    cell_rows: list[list[float]] = []
    n_cells = int(rng.integers(2, 5))
    width_km = cols * resolution_m / 1000.0
    height_km = rows * resolution_m / 1000.0
    for _ in range(n_cells):
        cell_rows.append(
            [
                float(rng.uniform(-0.45, 0.45) * width_km),
                float(rng.uniform(-0.45, 0.45) * height_km),
                float(rng.uniform(-8.0, 8.0) / 3600.0),
                float(rng.uniform(-8.0, 8.0) / 3600.0),
                float(rng.uniform(5.0, 16.0)),
                float(rng.uniform(5.0, 16.0)),
                float(rng.uniform(1.5, rain_max_loss_db)),
            ]
        )

    # Same RNG state/order as the legacy dynamic_rain_loss implementation.
    texture = correlated_field(rows, cols, resolution_m, 4500.0, rng)
    return (
        x_grid.astype(np.float32),
        y_grid.astype(np.float32),
        np.asarray(cell_rows, dtype=np.float64),
        texture,
    )


def evaluate_weather_loss(
    x_grid_km: np.ndarray,
    y_grid_km: np.ndarray,
    cells: np.ndarray,
    texture: np.ndarray,
    elapsed_s: float,
    max_loss_db: float,
) -> np.ndarray:
    rain = np.zeros_like(x_grid_km, dtype=np.float32)
    for x0, y0, vx, vy, sx, sy, amplitude in cells:
        x = x0 + vx * elapsed_s
        y = y0 + vy * elapsed_s
        cell = amplitude * np.exp(
            -0.5
            * (
                ((x_grid_km - x) / sx) ** 2
                + ((y_grid_km - y) / sy) ** 2
            )
        )
        rain += cell.astype(np.float32)
    rain *= 1.0 + 0.08 * texture
    return np.clip(rain, 0.0, max_loss_db).astype(np.float32)


def static_clutter_loss(
    region: str,
    rows: int,
    cols: int,
    resolution_m: float,
    seed: int,
    max_loss_db: float,
) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(region + "_clutter", seed))
    field = correlated_field(rows, cols, resolution_m, 1800.0, rng)
    threshold = np.quantile(field, 0.72)
    loss = np.maximum(field - threshold, 0.0)
    loss = loss / (np.max(loss) + 1e-12) * max_loss_db
    return loss.astype(np.float32)


@njit(cache=True)
def _knife_edge_loss_scalar(v: float) -> float:
    if v <= -0.78:
        return 0.0
    value = 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return value if value > 0.0 else 0.0


@njit(cache=True)
def _boundary_distance(
    origin_row: float,
    origin_col: float,
    az_rad: float,
    rows: int,
    cols: int,
    resolution_m: float,
) -> float:
    dr = -math.cos(az_rad) / resolution_m
    dc = math.sin(az_rad) / resolution_m
    best = 1.0e30
    if dr < -1.0e-12:
        value = origin_row / (-dr)
        if value > 0.0 and value < best:
            best = value
    elif dr > 1.0e-12:
        value = (rows - 1 - origin_row) / dr
        if value > 0.0 and value < best:
            best = value
    if dc < -1.0e-12:
        value = origin_col / (-dc)
        if value > 0.0 and value < best:
            best = value
    elif dc > 1.0e-12:
        value = (cols - 1 - origin_col) / dc
        if value > 0.0 and value < best:
            best = value
    return 0.0 if best == 1.0e30 else best


@njit(parallel=True, cache=True)
def compute_terrain_loss_fast(
    expanded_dem: np.ndarray,
    core_dem: np.ndarray,
    elevation_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    slant_range_m: np.ndarray,
    y0: int,
    x0: int,
    resolution_m: float,
    freq_ghz: float,
    ray_step_m: float,
    ray_start_m: float,
    min_elevation_deg: float,
    clearance_m: float,
    diffraction_cap_db: float,
    k_factor: float,
    curvature_enabled: bool,
) -> np.ndarray:
    rows, cols = core_dem.shape
    exp_rows, exp_cols = expanded_dem.shape
    output = np.zeros((rows, cols), dtype=np.float32)
    wavelength_m = 2.998e8 / (freq_ghz * 1.0e9)
    effective_radius_m = k_factor * EARTH_RADIUS_M

    for i in prange(rows):
        for j in range(cols):
            elevation = float(elevation_deg[i, j])
            if not math.isfinite(elevation) or elevation <= min_elevation_deg:
                continue

            az_rad = math.radians(float(azimuth_deg[i, j]))
            el_rad = math.radians(elevation)
            cos_az = math.cos(az_rad)
            sin_az = math.sin(az_rad)
            tan_el = math.tan(el_rad)
            cos_el = max(math.cos(el_rad), 1.0e-6)
            origin_i = i + y0
            origin_j = j + x0
            exit_distance = _boundary_distance(
                float(origin_i),
                float(origin_j),
                az_rad,
                exp_rows,
                exp_cols,
                resolution_m,
            )
            if exit_distance < ray_start_m:
                continue

            source_height = float(core_dem[i, j])
            total_distance = max(float(slant_range_m[i, j]), 1.0)
            max_loss = 0.0
            distance = ray_start_m
            limit = exit_distance + ray_step_m * 0.25

            while distance <= limit:
                rr = int(np.rint(origin_i - (distance / resolution_m) * cos_az))
                cc = int(np.rint(origin_j + (distance / resolution_m) * sin_az))
                if 0 <= rr < exp_rows and 0 <= cc < exp_cols:
                    terrain_height = float(expanded_dem[rr, cc])
                    if math.isfinite(terrain_height):
                        ray_height = source_height + distance * tan_el
                        if curvature_enabled:
                            ray_height += (distance * distance) / (2.0 * effective_radius_m)
                        height_excess = terrain_height - ray_height
                        d1 = max(distance / cos_el, 1.0)
                        d2 = max(total_distance - d1, 1.0)
                        v = height_excess * math.sqrt(
                            (2.0 / wavelength_m) * (1.0 / d1 + 1.0 / d2)
                        )
                        loss = _knife_edge_loss_scalar(v)
                        if loss > max_loss:
                            max_loss = loss
                distance += ray_step_m

            if max_loss > diffraction_cap_db:
                max_loss = diffraction_cap_db
            output[i, j] = max_loss

    return output


def build_tle_catalog(tle_path: Path, timescale: Any) -> dict[str, tuple[list[float], list[EarthSatellite]]]:
    lines = [
        line.strip()
        for line in tle_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    raw: dict[str, list[EarthSatellite]] = {}
    for index in range(0, len(lines) - 1, 2):
        line1, line2 = lines[index], lines[index + 1]
        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            continue
        # NORAD编号统一转换为不带前导零的十进制字符串。
        # 例如TLE中的"04387"与CSV读取后的4387统一表示为"4387"。
        sat_id_field = line1[2:7].strip()
        sat_id = str(int(sat_id_field))
        satellite = EarthSatellite(line1, line2, sat_id, timescale)
        raw.setdefault(sat_id, []).append(satellite)

    catalog: dict[str, tuple[list[float], list[EarthSatellite]]] = {}
    for sat_id, satellites in raw.items():
        satellites.sort(key=lambda sat: sat.epoch.tt)
        catalog[sat_id] = ([float(sat.epoch.tt) for sat in satellites], satellites)
    return catalog


def select_satellite(
    catalog: dict[str, tuple[list[float], list[EarthSatellite]]],
    sat_id: str,
    target_tt: float,
) -> EarthSatellite:
    # 查询前再次标准化，兼容4387、04387、数值型4387等形式。
    sat_id = str(int(str(sat_id).strip()))

    if sat_id not in catalog:
        raise KeyError(f"No TLE found for NORAD {sat_id}")
    epochs, satellites = catalog[sat_id]
    index = bisect.bisect_right(epochs, target_tt) - 1
    if index < 0:
        index = 0
    return satellites[index]


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write a NumPy array atomically so parallel workers cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy_file(source: Path, target: Path) -> None:
    """Copy a scene metadata file atomically for safe multi-process startup."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_region_state(
    root: Path,
    out_root: Path,
    region: str,
    args: argparse.Namespace,
) -> RegionState:
    metadata_path = metadata_path_for_region(
        root,
        region,
        args.core_size_km,
        args.buffer_km,
        args.resolution_m,
    )
    metadata = load_meta(metadata_path)
    meta_region, center_lon, center_lat, meta_resolution = center_from_meta(metadata)
    if meta_region != region:
        raise ValueError(f"Region mismatch: schedule={region}, metadata={meta_region}")
    if not np.isclose(meta_resolution, args.resolution_m):
        raise ValueError(
            f"Resolution mismatch: metadata={meta_resolution}, argument={args.resolution_m}"
        )

    expanded_path, core_path, core_slice = resolve_dem_paths(
        metadata,
        metadata_path,
        args.core_size_km,
        args.buffer_km,
        args.resolution_m,
    )
    expanded_dem = np.load(expanded_path).astype(np.float32)
    core_dem = np.load(core_path).astype(np.float32)
    y0, y1, x0, x1 = core_slice
    if not np.allclose(
        expanded_dem[y0:y1, x0:x1], core_dem, atol=1e-4, rtol=0.0, equal_nan=True
    ):
        raise ValueError(f"Core DEM does not match expanded DEM slice for {region}")

    rows, cols = core_dem.shape
    row_offsets_m = (np.arange(rows) - (rows - 1) / 2.0) * args.resolution_m
    col_offsets_m = (np.arange(cols) - (cols - 1) / 2.0) * args.resolution_m
    north_offsets_m = -row_offsets_m[:, None] * np.ones((1, cols))
    east_offsets_m = np.ones((rows, 1)) * col_offsets_m[None, :]
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(center_lat))
    lat_grid = (center_lat + north_offsets_m / meters_per_deg_lat).astype(np.float64)
    lon_grid = (center_lon + east_offsets_m / meters_per_deg_lon).astype(np.float64)

    # Scene-static geometry: compute once, reuse for all 4,320 frames.
    ground_ecef, east_unit, north_unit, up_unit = geodetic_grid_to_ecef_enu(
        lat_grid, lon_grid, core_dem
    )


    weather_x, weather_y, rain_cells, rain_texture = prepare_weather_state(
        region,
        rows,
        cols,
        args.resolution_m,
        args.model_e_seed,
        args.rain_max_loss_db,
    )
    clutter = static_clutter_loss(
        region,
        rows,
        cols,
        args.resolution_m,
        args.model_e_seed,
        args.clutter_max_loss_db,
    )

    scene_dir = out_root / "scenes" / region
    scene_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(scene_dir / "dem_m.npy", core_dem)
    atomic_save_npy(scene_dir / "clutter_loss_db.npy", clutter)
    atomic_copy_file(metadata_path, scene_dir / "grid_meta.json")

    return RegionState(
        region=region,
        metadata_path=metadata_path,
        metadata=metadata,
        center_lon=center_lon,
        center_lat=center_lat,
        resolution_m=args.resolution_m,
        core_dem=core_dem,
        expanded_dem=expanded_dem,
        core_slice=core_slice,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        ground_ecef_m=ground_ecef,
        east_unit=east_unit,
        north_unit=north_unit,
        up_unit=up_unit,
        weather_x_km=weather_x,
        weather_y_km=weather_y,
        rain_cells=rain_cells,
        rain_texture=rain_texture,
        clutter_loss=clutter,
    )



@njit(parallel=True, cache=True)
def compute_geometry_model_b_numba(
    sat_x_m: float,
    sat_y_m: float,
    sat_z_m: float,
    ground_ecef_m: np.ndarray,
    east_unit: np.ndarray,
    north_unit: np.ndarray,
    up_unit: np.ndarray,
    center_x: float,
    center_y: float,
    center_z: float,
    freq_mhz: float,
    tx_power_dbw: float,
    rx_gain_dbi: float,
    effective_peak_gain_dbi: float,
    theta_3db_deg: float,
    max_atten_db: float,
    atmosphere_zenith_db: float,
    atm_min_elev_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fused CPU kernel for Geometry + Model-A + Model-B.

    The scene-static ECEF and ENU arrays are flattened by Numba automatically.
    Each pixel is independent, so prange can use the configured Numba threads.
    """
    rows = ground_ecef_m.shape[1]
    cols = ground_ecef_m.shape[2]
    count = rows * cols
    elevation = np.empty(count, dtype=np.float32)
    azimuth = np.empty(count, dtype=np.float32)
    slant_km = np.empty(count, dtype=np.float32)
    model_b = np.empty(count, dtype=np.float32)
    rad_to_deg = 180.0 / math.pi

    ground_x = ground_ecef_m[0].ravel()
    ground_y = ground_ecef_m[1].ravel()
    ground_z = ground_ecef_m[2].ravel()
    east_x = east_unit[0].ravel()
    east_y = east_unit[1].ravel()
    east_z = east_unit[2].ravel()
    north_x = north_unit[0].ravel()
    north_y = north_unit[1].ravel()
    north_z = north_unit[2].ravel()
    up_x = up_unit[0].ravel()
    up_y = up_unit[1].ravel()
    up_z = up_unit[2].ravel()

    for index in prange(count):
        dx = sat_x_m - ground_x[index]
        dy = sat_y_m - ground_y[index]
        dz = sat_z_m - ground_z[index]
        distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)
        inv_distance = 1.0 / distance_m

        east_component = dx * east_x[index] + dy * east_y[index] + dz * east_z[index]
        north_component = (
            dx * north_x[index] + dy * north_y[index] + dz * north_z[index]
        )
        up_component = dx * up_x[index] + dy * up_y[index] + dz * up_z[index]

        sin_elevation = up_component * inv_distance
        if sin_elevation < -1.0:
            sin_elevation = -1.0
        elif sin_elevation > 1.0:
            sin_elevation = 1.0
        elevation_deg = math.asin(sin_elevation) * rad_to_deg
        azimuth_deg = math.atan2(east_component, north_component) * rad_to_deg
        if azimuth_deg < 0.0:
            azimuth_deg += 360.0

        # observer_from_satellite = -line_of_sight
        cos_theta = (
            center_x * (-dx) + center_y * (-dy) + center_z * (-dz)
        ) * inv_distance
        if cos_theta < -1.0:
            cos_theta = -1.0
        elif cos_theta > 1.0:
            cos_theta = 1.0
        off_axis_deg = math.acos(cos_theta) * rad_to_deg

        distance_km = distance_m / 1000.0
        fspl_db = 32.45 + 20.0 * math.log10(freq_mhz) + 20.0 * math.log10(distance_km)
        beam_attenuation_db = 3.0 * (off_axis_deg / theta_3db_deg) ** 2
        if beam_attenuation_db > max_atten_db:
            beam_attenuation_db = max_atten_db
        beam_gain_dbi = effective_peak_gain_dbi - beam_attenuation_db
        model_a_dbm = tx_power_dbw + beam_gain_dbi + rx_gain_dbi - fspl_db + 30.0

        used_elevation_deg = elevation_deg
        if used_elevation_deg < atm_min_elev_deg:
            used_elevation_deg = atm_min_elev_deg
        atmosphere_loss_db = atmosphere_zenith_db / math.sin(
            used_elevation_deg / rad_to_deg
        )

        elevation[index] = elevation_deg
        azimuth[index] = azimuth_deg
        slant_km[index] = distance_km
        model_b[index] = model_a_dbm - atmosphere_loss_db

    return (
        elevation.reshape((rows, cols)),
        azimuth.reshape((rows, cols)),
        slant_km.reshape((rows, cols)),
        model_b.reshape((rows, cols)),
    )

def compute_geometry_model_b_numpy(
    sat_ecef_m: np.ndarray,
    ground_ecef_m: np.ndarray,
    east_unit: np.ndarray,
    north_unit: np.ndarray,
    up_unit: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy fallback used only when Numba is unavailable."""
    line_of_sight = sat_ecef_m[:, None, None] - ground_ecef_m
    slant_m = np.sqrt(np.sum(line_of_sight * line_of_sight, axis=0))
    east_component = np.sum(line_of_sight * east_unit, axis=0)
    north_component = np.sum(line_of_sight * north_unit, axis=0)
    up_component = np.sum(line_of_sight * up_unit, axis=0)
    elevation_deg = np.rad2deg(
        np.arcsin(np.clip(up_component / slant_m, -1.0, 1.0))
    )
    azimuth_deg = np.mod(
        np.rad2deg(np.arctan2(east_component, north_component)) + 360.0,
        360.0,
    )

    rows, cols = slant_m.shape
    center_row, center_col = rows // 2, cols // 2
    observer_from_satellite = -line_of_sight
    center_vector = observer_from_satellite[:, center_row, center_col]
    center_vector = center_vector / np.linalg.norm(center_vector)
    nadir_vector = -sat_ecef_m / np.linalg.norm(sat_ecef_m)
    cos_scan = float(np.clip(nadir_vector @ center_vector, 1.0e-6, 1.0))
    scan_loss_db = (
        0.0
        if args.disable_scan_loss
        else float(max(0.0, -10.0 * args.scan_loss_q * math.log10(cos_scan)))
    )

    cos_theta = np.clip(
        np.sum(center_vector[:, None, None] * observer_from_satellite, axis=0)
        / slant_m,
        -1.0,
        1.0,
    )
    off_axis_deg = np.rad2deg(np.arccos(cos_theta))
    slant_range_km = slant_m / 1000.0
    freq_mhz = args.freq_ghz * 1000.0
    fspl_db = (
        32.45
        + 20.0 * math.log10(freq_mhz)
        + 20.0 * np.log10(slant_range_km)
    )
    element_count = args.array_rows * args.array_cols
    ideal_peak_gain_dbi = args.element_gain_dbi + 10.0 * math.log10(element_count)
    effective_peak_gain_dbi = (
        ideal_peak_gain_dbi - args.efficiency_loss_db - scan_loss_db
    )
    theta_3db_deg = (101.4 / max(args.array_rows, args.array_cols)) / 2.0
    beam_attenuation_db = np.minimum(
        3.0 * (off_axis_deg / theta_3db_deg) ** 2,
        args.max_atten_db,
    )
    model_a_dbm = (
        args.tx_power_dbw
        + effective_peak_gain_dbi
        - beam_attenuation_db
        + args.rx_gain_dbi
        - fspl_db
        + 30.0
    )
    used_elevation = np.maximum(elevation_deg, args.atm_min_elev_deg)
    atmosphere_loss_db = (
        args.gas_zenith_db + args.cloud_zenith_db
    ) / np.sin(np.deg2rad(used_elevation))
    model_b_dbm = model_a_dbm - atmosphere_loss_db

    return (
        elevation_deg.astype(np.float32),
        azimuth_deg.astype(np.float32),
        slant_range_km.astype(np.float32),
        model_b_dbm.astype(np.float32),
    )


def compute_geometry_and_model_b(
    state: RegionState,
    satellite: EarthSatellite,
    target_time: datetime,
    timescale: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """CPU-only Geometry + Model-A/B.

    Skyfield propagates one satellite once per frame. The 768x768 ground grid
    is then evaluated with scene-static ECEF/ENU arrays. With Numba installed,
    Geometry, Model-A, and Model-B are fused into one parallel per-pixel kernel.
    """
    timings: dict[str, float] = {}
    scalar_time = timescale.from_datetime(target_time)

    t0 = time.perf_counter()
    sat_ecef_m = np.asarray(
        satellite.at(scalar_time).frame_xyz(itrs).m,
        dtype=np.float64,
    )
    timings["satellite_sgp4_itrs"] = time.perf_counter() - t0

    center_row = state.core_dem.shape[0] // 2
    center_col = state.core_dem.shape[1] // 2
    center_observer_from_sat = (
        state.ground_ecef_m[:, center_row, center_col] - sat_ecef_m
    )
    center_observer_from_sat /= np.linalg.norm(center_observer_from_sat)
    nadir_vector = -sat_ecef_m / np.linalg.norm(sat_ecef_m)
    cos_scan_value = float(
        np.clip(nadir_vector @ center_observer_from_sat, 1.0e-6, 1.0)
    )
    scan_loss_db = (
        0.0
        if args.disable_scan_loss
        else float(
            max(
                0.0,
                -10.0 * args.scan_loss_q * math.log10(cos_scan_value),
            )
        )
    )
    element_count = args.array_rows * args.array_cols
    ideal_peak_gain_dbi = args.element_gain_dbi + 10.0 * math.log10(
        element_count
    )
    effective_peak_gain_dbi = (
        ideal_peak_gain_dbi - args.efficiency_loss_db - scan_loss_db
    )
    theta_3db_deg = (101.4 / max(args.array_rows, args.array_cols)) / 2.0

    t0 = time.perf_counter()
    if NUMBA_AVAILABLE:
        elevation_np, azimuth_np, slant_np, model_b_np = (
            compute_geometry_model_b_numba(
                float(sat_ecef_m[0]),
                float(sat_ecef_m[1]),
                float(sat_ecef_m[2]),
                state.ground_ecef_m,
                state.east_unit,
                state.north_unit,
                state.up_unit,
                float(center_observer_from_sat[0]),
                float(center_observer_from_sat[1]),
                float(center_observer_from_sat[2]),
                float(args.freq_ghz * 1000.0),
                float(args.tx_power_dbw),
                float(args.rx_gain_dbi),
                float(effective_peak_gain_dbi),
                float(theta_3db_deg),
                float(args.max_atten_db),
                float(args.gas_zenith_db + args.cloud_zenith_db),
                float(args.atm_min_elev_deg),
            )
        )
        timings["geometry_model_ab_fused_numba"] = time.perf_counter() - t0
    else:
        elevation_np, azimuth_np, slant_np, model_b_np = (
            compute_geometry_model_b_numpy(
                sat_ecef_m,
                state.ground_ecef_m,
                state.east_unit,
                state.north_unit,
                state.up_unit,
                args,
            )
        )
        timings["geometry_model_ab_numpy"] = time.perf_counter() - t0

    timings["geometry_model_ab_total"] = float(sum(timings.values()))
    return elevation_np, azimuth_np, slant_np, model_b_np, timings


def atomic_write_frame(
    frame_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    temporary_dir = frame_dir.with_name(frame_dir.name + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=False)

    try:
        reference_shape: tuple[int, ...] | None = None
        for name, array in arrays.items():
            array = np.asarray(array, dtype=np.float32)
            if array.ndim != 2:
                raise ValueError(f"{name} must be a 2-D array, got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or Inf")
            if reference_shape is None:
                reference_shape = array.shape
            elif array.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch: {name}={array.shape}, expected={reference_shape}"
                )
            np.save(temporary_dir / name, array)

        (temporary_dir / "meta.json").write_text(
            json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        temporary_dir.rename(frame_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def write_dataset_config(out_root: Path, args: argparse.Namespace) -> None:
    config = {
        "dataset_name": "Dataset-v0-Minimal3-Direct",
        "generator": "generate_dataset_minimal_direct.py",
        "generation_mode": "single_process_in_memory",
        "terrain_engine": "numba_parallel" if NUMBA_AVAILABLE else "python_fallback",
        "per_frame_arrays": [
            "coarse_pr_dbm.npy",
            "gt_pr_dbm.npy",
            "terrain_loss_lite_db.npy",
            "terrain_loss_hf_db.npy",
            "weather_loss_db.npy",
            "residual_dhf_to_e_db.npy",
        ],
        "per_scene_arrays": ["dem_m.npy", "clutter_loss_db.npy"],
        "minimal_model_inputs": [
            "coarse_pr_dbm",
            "sparse_residual_generated_during_training",
            "sampling_mask_generated_during_training",
        ],
        "parameters": vars(args),
    }
    (out_root / "dataset_config.json").write_text(
        json.dumps(to_jsonable(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Minimal-3 dataset directly without intermediate files."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--schedule-csv", type=Path, required=True)
    parser.add_argument("--tle-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--core-size-km", type=float, default=76.8)
    parser.add_argument("--buffer-km", type=float, default=40.0)
    parser.add_argument("--resolution-m", type=float, default=100.0)
    parser.add_argument("--freq-ghz", type=float, default=14.5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--regions", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--worker-mode",
        action="store_true",
        help=(
            "Parallel-worker mode: do not write dataset_config or shared manifest files. "
            "A parent launcher must finalize the dataset after every worker exits."
        ),
    )
    parser.add_argument("--numba-threads", type=int, default=0)

    # Model-A parameters.
    parser.add_argument("--tx-power-dbw", type=float, default=10.0)
    parser.add_argument("--rx-gain-dbi", type=float, default=0.0)
    parser.add_argument("--array-rows", type=int, default=64)
    parser.add_argument("--array-cols", type=int, default=64)
    parser.add_argument("--element-gain-dbi", type=float, default=5.0)
    parser.add_argument("--efficiency-loss-db", type=float, default=3.0)
    parser.add_argument("--max-atten-db", type=float, default=35.0)
    parser.add_argument("--scan-loss-q", type=float, default=1.3)
    parser.add_argument("--disable-scan-loss", action="store_true")

    # Model-B parameters.
    parser.add_argument("--gas-zenith-db", type=float, default=0.08)
    parser.add_argument("--cloud-zenith-db", type=float, default=0.10)
    parser.add_argument("--atm-min-elev-deg", type=float, default=5.0)

    # Model-D parameters.
    parser.add_argument("--lite-ray-step-m", type=float, default=1000.0)
    parser.add_argument("--lite-ray-start-m", type=float, default=500.0)
    parser.add_argument("--lite-cap-db", type=float, default=25.0)
    parser.add_argument("--hf-ray-step-m", type=float, default=100.0)
    parser.add_argument("--hf-ray-start-m", type=float, default=150.0)
    parser.add_argument("--hf-cap-db", type=float, default=40.0)
    parser.add_argument("--min-elevation-deg", type=float, default=3.0)
    parser.add_argument("--terrain-clearance-m", type=float, default=5.0)
    parser.add_argument("--k-factor", type=float, default=4.0 / 3.0)
    parser.add_argument("--disable-earth-curvature", action="store_true")

    # Model-E parameters.
    parser.add_argument("--model-e-seed", type=int, default=20250101)
    parser.add_argument("--rain-max-loss-db", type=float, default=8.0)
    parser.add_argument("--clutter-max-loss-db", type=float, default=2.0)
    parser.add_argument("--system-bias-loss-db", type=float, default=0.4)
    parser.add_argument("--measurement-noise-std-db", type=float, default=0.15)

    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    args.schedule_csv = args.schedule_csv.expanduser().resolve()
    args.tle_path = args.tle_path.expanduser().resolve()
    out_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else root / "results" / "dataset_v0"
    )
    base_root = out_root / "base_maps"
    base_root.mkdir(parents=True, exist_ok=True)

    print("Geometry backend: CPU-only optimized ECEF/ENU + Numba fusion")

    if not NUMBA_AVAILABLE:
        print(
            "[WARNING] Numba is not installed. Terrain calculation will use the "
            "slow Python fallback. Install with: conda install -n sgmrm numba"
        )
    elif args.numba_threads > 0:
        set_num_threads(args.numba_threads)

    schedule = pd.read_csv(args.schedule_csv)
    if "no_service_flag" in schedule.columns:
        schedule = schedule[schedule["no_service_flag"] == False].copy()  # noqa: E712
    if args.regions.strip():
        allowed = {item.strip() for item in args.regions.split(",") if item.strip()}
        schedule = schedule[schedule["region"].isin(allowed)].copy()
    schedule = schedule.sort_values(["region", "frame_id"])
    if args.max_frames is not None:
        schedule = schedule.head(args.max_frames)
    if schedule.empty:
        print("No service frames selected.")
        return

    print(f"Frames to generate: {len(schedule)}")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    timescale = load.timescale()
    print(f"Loading TLE catalog: {args.tle_path}")
    tle_catalog = build_tle_catalog(args.tle_path, timescale)
    print(f"TLE satellites loaded: {len(tle_catalog)}")

    if not args.worker_mode:
        write_dataset_config(out_root, args)
    manifest: list[dict[str, Any]] = []
    state: RegionState | None = None
    current_region = ""
    total_start = time.perf_counter()

    for sequence_index, (_, row) in enumerate(schedule.iterrows(), start=1):
        region = str(row["region"])
        frame_id = int(row["frame_id"])
        time_utc = str(row["time_utc"])
        sat_id = str(int(row["service_sat_id"]))
        elapsed_s = float(row["elapsed_s"])
        frame_name = (
            f"{region}_frame{frame_id:04d}_{safe_time_tag(time_utc)}_sat{sat_id}"
        )
        frame_dir = base_root / frame_name
        if args.skip_existing and (frame_dir / "gt_pr_dbm.npy").is_file():
            print(f"[{sequence_index}/{len(schedule)}] skip {frame_name}")
            continue

        if state is None or region != current_region:
            state = prepare_region_state(root, out_root, region, args)
            current_region = region
            print(f"[region] loaded {region}, shape={state.core_dem.shape}")

        assert state is not None
        frame_start = time.perf_counter()
        target_time = parse_utc(time_utc)
        scalar_time = timescale.from_datetime(target_time)
        satellite = select_satellite(tle_catalog, sat_id, float(scalar_time.tt))

        stage_start = time.perf_counter()
        elevation, azimuth, slant_km, model_b, geometry_detail = (
            compute_geometry_and_model_b(
                state, satellite, target_time, timescale, args
            )
        )
        geometry_seconds = time.perf_counter() - stage_start

        slant_m = slant_km.astype(np.float32) * 1000.0
        y0, _y1, x0, _x1 = state.core_slice
        curvature_enabled = not args.disable_earth_curvature

        stage_start = time.perf_counter()
        terrain_lite = compute_terrain_loss_fast(
            state.expanded_dem,
            state.core_dem,
            elevation,
            azimuth,
            slant_m,
            y0,
            x0,
            args.resolution_m,
            args.freq_ghz,
            args.lite_ray_step_m,
            args.lite_ray_start_m,
            args.min_elevation_deg,
            args.terrain_clearance_m,
            args.lite_cap_db,
            args.k_factor,
            curvature_enabled,
        )
        lite_seconds = time.perf_counter() - stage_start
        coarse = model_b - terrain_lite

        stage_start = time.perf_counter()
        terrain_hf = compute_terrain_loss_fast(
            state.expanded_dem,
            state.core_dem,
            elevation,
            azimuth,
            slant_m,
            y0,
            x0,
            args.resolution_m,
            args.freq_ghz,
            args.hf_ray_step_m,
            args.hf_ray_start_m,
            args.min_elevation_deg,
            args.terrain_clearance_m,
            args.hf_cap_db,
            args.k_factor,
            curvature_enabled,
        )
        hf_seconds = time.perf_counter() - stage_start
        model_d_hf = model_b - terrain_hf

        stage_start = time.perf_counter()
        weather_loss = evaluate_weather_loss(
            state.weather_x_km,
            state.weather_y_km,
            state.rain_cells,
            state.rain_texture,
            elapsed_s,
            args.rain_max_loss_db,
        )
        noise_rng = np.random.default_rng(
            stable_seed(f"{region}_noise_{frame_id}", args.model_e_seed)
        )
        noise = noise_rng.normal(
            0.0,
            args.measurement_noise_std_db,
            size=model_d_hf.shape,
        ).astype(np.float32)
        gt = (
            model_d_hf
            - weather_loss
            - state.clutter_loss
            - args.system_bias_loss_db
            + noise
        ).astype(np.float32)
        residual_dhf_to_e = (gt - model_d_hf).astype(np.float32)
        model_e_seconds = time.perf_counter() - stage_start

        arrays = {
            "coarse_pr_dbm.npy": coarse,
            "gt_pr_dbm.npy": gt,
            "terrain_loss_lite_db.npy": terrain_lite,
            "terrain_loss_hf_db.npy": terrain_hf,
            "weather_loss_db.npy": weather_loss,
            "residual_dhf_to_e_db.npy": residual_dhf_to_e,
        }
        total_residual = gt - coarse
        metadata = to_jsonable(row.to_dict())
        metadata.update(
            {
                "frame_name": frame_name,
                "storage_profile": "minimal3_direct",
                "metadata_path": str(state.metadata_path),
                "coarse_for_main_task": "coarse_pr_dbm.npy",
                "gt_for_main_task": "gt_pr_dbm.npy",
                "map_shape_yx": list(coarse.shape),
                "dtype": "float32",
                "generation_time_seconds": {
                    "geometry_model_ab": geometry_seconds,
                    **geometry_detail,
                    "terrain_lite": lite_seconds,
                    "terrain_hf": hf_seconds,
                    "model_e": model_e_seconds,
                },
                "geometry_backend": "cpu_ecef_enu_numba",
                "total_residual_statistics_db": {
                    "min": float(np.min(total_residual)),
                    "max": float(np.max(total_residual)),
                    "mean": float(np.mean(total_residual)),
                    "std": float(np.std(total_residual)),
                },
            }
        )
        save_start = time.perf_counter()
        atomic_write_frame(frame_dir, arrays, metadata)
        save_seconds = time.perf_counter() - save_start
        manifest.append(metadata)

        frame_seconds = time.perf_counter() - frame_start
        elapsed_total = time.perf_counter() - total_start
        average_seconds = elapsed_total / sequence_index
        remaining_seconds = average_seconds * (len(schedule) - sequence_index)
        print(
            f"[{sequence_index}/{len(schedule)}] {frame_name} "
            f"total={frame_seconds:.2f}s geom+A+B={geometry_seconds:.2f}s "
            f"(SGP4={geometry_detail.get('satellite_sgp4_itrs', 0.0):.3f}s, "
            f"CPU-fused={geometry_detail.get('geometry_model_ab_fused_numba', geometry_detail.get('geometry_model_ab_numpy', 0.0)):.3f}s) "
            f"D-lite={lite_seconds:.2f}s D-HF={hf_seconds:.2f}s E={model_e_seconds:.2f}s "
            f"save={save_seconds:.2f}s "
            f"ETA={remaining_seconds / 3600.0:.2f}h"
        )

    if args.worker_mode:
        print(
            f"Worker generation finished: {out_root} "
            f"({len(manifest)} newly completed frames; shared manifest deferred)"
        )
        return

    # Rebuild the manifest from every completed frame so resume/skip runs also
    # produce a complete index rather than only indexing newly generated frames.
    complete_manifest: list[dict[str, Any]] = []
    for meta_path in sorted(base_root.glob("*/meta.json")):
        complete_manifest.append(json.loads(meta_path.read_text(encoding="utf-8")))

    manifest_json = out_root / "dataset_v0_manifest.json"
    manifest_json.write_text(
        json.dumps(to_jsonable(complete_manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(complete_manifest).to_csv(
        out_root / "dataset_v0_manifest.csv", index=False
    )
    print(
        f"Dataset generation finished: {out_root} "
        f"({len(complete_manifest)} completed frames)"
    )


if __name__ == "__main__":
    main()
