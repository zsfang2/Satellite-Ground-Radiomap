#!/usr/bin/env python3
"""Build a fixed-20-second service-satellite schedule for Dataset-v0.

For each region and each frame, the script first checks whether the current
serving satellite is still above the elevation threshold. If yes, it keeps the
same satellite. If not, it selects the currently highest-elevation satellite
whose elevation is above the threshold. This avoids frame-by-frame ping-pong
handover while still checking the requirement before every map frame.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84


def parse_utc(text: str) -> datetime:
    text = text.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def find_default_tle(root: Path, date_tag: str) -> Path:
    candidates = [
        root / "data" / "l1_space" / f"{date_tag}_leo_payload.tle",
        root / "data" / "l1_space" / f"{date_tag}_latest_per_norad.tle",
        root / "data" / "l1_space" / f"{date_tag}_unique_pairs.tle",
        root / "data" / "l1_space" / f"{date_tag}.tle",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find TLE file. Tried:\n" + "\n".join(str(p) for p in candidates) +
        "\nPlease pass --tle-path explicitly."
    )


def load_regions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Region config must be a JSON list.")
    required = {"region", "center_lon", "center_lat"}
    for item in data:
        missing = required - set(item)
        if missing:
            raise ValueError(f"Region item missing {missing}: {item}")
        item.setdefault("center_elev_m", 0.0)
    return data


def load_latest_tles(tle_path: Path, ts) -> dict[str, EarthSatellite]:
    lines = [
        x.strip()
        for x in tle_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if x.strip()
    ]
    if len(lines) < 2:
        raise ValueError(f"TLE file is empty or invalid: {tle_path}")

    latest: dict[str, EarthSatellite] = {}
    for i in range(0, len(lines) - 1, 2):
        l1, l2 = lines[i], lines[i + 1]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        norad = l1[2:7].strip()
        try:
            sat = EarthSatellite(l1, l2, norad, ts)
        except Exception:
            continue
        old = latest.get(norad)
        if old is None or sat.epoch.tt > old.epoch.tt:
            latest[norad] = sat
    if not latest:
        raise ValueError(f"No valid satellites loaded from {tle_path}")
    return latest


def elev_bin(elev: float) -> str:
    if not np.isfinite(elev):
        return "no_service"
    if elev >= 70.0:
        return "high"
    if elev >= 40.0:
        return "mid"
    if elev >= 25.0:
        return "low"
    return "below_threshold"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Dataset-v0 service schedule with a fixed 20-second "
            "frame interval and a command-line configurable time range."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--regions-json", type=Path, default=None)
    parser.add_argument("--tle-path", type=Path, default=None)
    parser.add_argument(
        "--start-time",
        default="2025-01-01T00:00:00Z",
        help="UTC start time, for example 2025-01-01T00:00:00Z.",
    )
    period = parser.add_mutually_exclusive_group()
    period.add_argument(
        "--end-time",
        default=None,
        help=(
            "Exclusive UTC end time. Example: 2025-01-02T00:00:00Z "
            "generates exactly one day from a Jan 1 start."
        ),
    )
    period.add_argument(
        "--duration-hours",
        type=float,
        default=None,
        help="Schedule duration in hours. Default: 24 hours.",
    )
    period.add_argument(
        "--duration-s",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--min-elev-deg", type=float, default=25.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    regions_json = args.regions_json or (root / "config" / "regions_dataset_v0.json")
    if not regions_json.exists():
        raise FileNotFoundError(f"Cannot find region config: {regions_json}")
    regions = load_regions(regions_json)

    start = parse_utc(args.start_time)
    if args.end_time is not None:
        end = parse_utc(args.end_time)
        duration_s = int(round((end - start).total_seconds()))
    elif args.duration_s is not None:
        duration_s = int(args.duration_s)
        end = start + timedelta(seconds=duration_s)
    else:
        duration_hours = 24.0 if args.duration_hours is None else float(args.duration_hours)
        duration_s = int(round(duration_hours * 3600.0))
        end = start + timedelta(seconds=duration_s)

    if duration_s <= 0:
        raise ValueError("The requested schedule duration must be positive.")

    step_s = 20
    date_tag = start.strftime("%Y-%m-%d")
    tle_path = args.tle_path or find_default_tle(root, date_tag)
    if not tle_path.exists():
        raise FileNotFoundError(f"Cannot find TLE file: {tle_path}")

    output = args.output or (root / "results" / "dataset_v0" / "service_schedule_20s.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_offsets = np.arange(0, duration_s, step_s, dtype=int)
    datetimes = [start + timedelta(seconds=int(s)) for s in frame_offsets]

    ts = load.timescale()
    times = ts.from_datetimes(datetimes)
    sats = load_latest_tles(tle_path, ts)
    sat_items = list(sats.items())

    print(f"Loaded {len(sat_items)} satellites from {tle_path}")
    print(
        f"Time range: {start.strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"to {end.strftime('%Y-%m-%dT%H:%M:%SZ')} (end exclusive)"
    )
    print(
        f"Frames per region: {len(datetimes)}, step={step_s}s, "
        f"duration={duration_s}s, total rows={len(datetimes) * len(regions)}"
    )

    all_rows: list[dict[str, Any]] = []

    for region in regions:
        region_name = str(region["region"])
        observer = wgs84.latlon(
            latitude_degrees=float(region["center_lat"]),
            longitude_degrees=float(region["center_lon"]),
            elevation_m=float(region.get("center_elev_m", 0.0)),
        )

        elev_by_sat: dict[str, np.ndarray] = {}
        az_by_sat: dict[str, np.ndarray] = {}
        range_by_sat: dict[str, np.ndarray] = {}

        print(f"Computing visibility for region={region_name} ...")
        for idx, (norad, sat) in enumerate(sat_items, start=1):
            try:
                topoc = (sat - observer).at(times)
                alt, az, distance = topoc.altaz()
                elev_by_sat[norad] = np.asarray(alt.degrees, dtype=np.float32)
                az_by_sat[norad] = np.asarray(az.degrees, dtype=np.float32)
                range_by_sat[norad] = np.asarray(distance.km, dtype=np.float32)
            except Exception as exc:
                if idx <= 5:
                    print(f"  skip {norad}: {exc}")
            if idx % 1000 == 0:
                print(f"  processed {idx}/{len(sat_items)} satellites")

        current_sat: str | None = None
        pass_id = -1

        for frame_id, dt in enumerate(datetimes):
            handover = False
            no_service = False

            keep_current = False
            if current_sat is not None and current_sat in elev_by_sat:
                current_elev = float(elev_by_sat[current_sat][frame_id])
                keep_current = current_elev >= args.min_elev_deg

            if not keep_current:
                best_sat = None
                best_elev = -999.0
                for norad, elev_arr in elev_by_sat.items():
                    elev = float(elev_arr[frame_id])
                    if elev >= args.min_elev_deg and elev > best_elev:
                        best_sat = norad
                        best_elev = elev

                if best_sat is None:
                    if current_sat is not None:
                        handover = True
                    current_sat = None
                    no_service = True
                else:
                    handover = best_sat != current_sat
                    current_sat = best_sat
                    if handover:
                        pass_id += 1

            if current_sat is None:
                row = {
                    "region": region_name,
                    "display_name": region.get("display_name", region_name),
                    "frame_id": frame_id,
                    "time_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "elapsed_s": int(frame_offsets[frame_id]),
                    "service_sat_id": "",
                    "elevation_center_deg": np.nan,
                    "azimuth_center_deg": np.nan,
                    "slant_range_center_km": np.nan,
                    "handover_flag": bool(handover),
                    "no_service_flag": True,
                    "pass_id": pass_id,
                    "elevation_bin": "no_service",
                    "min_elev_deg": args.min_elev_deg,
                    "center_lon": float(region["center_lon"]),
                    "center_lat": float(region["center_lat"]),
                }
            else:
                elev = float(elev_by_sat[current_sat][frame_id])
                row = {
                    "region": region_name,
                    "display_name": region.get("display_name", region_name),
                    "frame_id": frame_id,
                    "time_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "elapsed_s": int(frame_offsets[frame_id]),
                    "service_sat_id": current_sat,
                    "elevation_center_deg": elev,
                    "azimuth_center_deg": float(az_by_sat[current_sat][frame_id]),
                    "slant_range_center_km": float(range_by_sat[current_sat][frame_id]),
                    "handover_flag": bool(handover),
                    "no_service_flag": False,
                    "pass_id": pass_id,
                    "elevation_bin": elev_bin(elev),
                    "min_elev_deg": args.min_elev_deg,
                    "center_lon": float(region["center_lon"]),
                    "center_lat": float(region["center_lat"]),
                }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    summary = {
        "start_time_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time_utc_exclusive": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": duration_s,
        "duration_hours": duration_s / 3600.0,
        "step_s": step_s,
        "min_elev_deg": args.min_elev_deg,
        "tle_path": str(tle_path),
        "regions_json": str(regions_json),
        "num_regions": len(regions),
        "num_rows": int(len(df)),
        "num_no_service": int(df["no_service_flag"].sum()),
        "num_handover": int(df["handover_flag"].sum()),
        "by_region": df.groupby("region").agg(
            frames=("frame_id", "count"),
            no_service=("no_service_flag", "sum"),
            handovers=("handover_flag", "sum"),
            mean_elev=("elevation_center_deg", "mean"),
            min_elev=("elevation_center_deg", "min"),
            max_elev=("elevation_center_deg", "max"),
        ).reset_index().to_dict(orient="records"),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== service schedule done =====")
    print("Output:", output)
    print("Summary:", output.with_suffix(".summary.json"))
    print(df.groupby(["region", "elevation_bin"]).size().unstack(fill_value=0))
    print("Handovers by region:")
    print(df.groupby("region")["handover_flag"].sum())


if __name__ == "__main__":
    main()
