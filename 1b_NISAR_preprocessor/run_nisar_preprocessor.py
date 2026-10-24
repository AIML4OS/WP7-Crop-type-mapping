#!/usr/bin/env python
"""
================================================================================
AIML CropMapper Cloud - Phase 1b: Unified NISAR L-Band SAR Preprocessor
================================================================================
End-to-end operational pipeline for NASA/ISRO NISAR Level-2 Geocoded Polarimetric
Covariance (L2 GCOV) time-series extraction, cross-track antenna de-tapering,
seamless multi-track orbital cycle mosaicking, S1/S2 EPSG:3857 (10.0 m) grid
co-registration, and BigTIFF stacking across any European country (EU-27).

Key Capabilities:
  - Dual Data Source Discovery: Local `/eodata/NISAR` HDF5 repository fast-path
    and NASA ASF DAAC search (`asf_search` or built-in ASF REST API fallback).
  - Physical L-Band Polarimetry: Extracts HHHH (dB), HVHV (dB), RVI_L (4*HV/(HH+HV)),
    and HH/HV Ratio with 2D adaptive cross-track range-edge de-tapering, along-track
    slit healing, and C^1 inverse-distance-squared feathering.
  - S1 & S2 Pixel-for-Pixel Compatibility: Automatically matches the exact
    EPSG:3857 (10.0 m) pixel grid of Sentinel-1 & Sentinel-2 stacks in
    `workingDirs/<COUNTRY>/orbit_<ORBIT>/1_input_stacks/`.

Execution Examples:
  # 1. Interactive CLI setup wizard (zero arguments):
  python run_nisar_preprocessor.py

  # 2. Full pipeline for a specific orbit track (e.g. Netherlands Orbit 88):
  python run_nisar_preprocessor.py --track NL/orbit_88 -s 2025-10-15 -e 2026-09-15 --stage A

  # 3. Full pipeline for all discovered orbits of a country (e.g. Poland or Portugal):
  python run_nisar_preprocessor.py --country PL -s 2025-10-15 -e 2026-09-15 --stage A

  # 4. Run only Stage 3 (Re-stack existing cycle mosaics into S1-aligned BigTIFF):
  python run_nisar_preprocessor.py --track PT/orbit_52 --stage 3
================================================================================
"""

import argparse
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Constrain OpenMP/MKL threads before importing scientific libraries
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

script_dir = Path(__file__).resolve().parent
modules_dir = script_dir / "modules"
for p in [script_dir, modules_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nisar_asf_search import (
    AUX_DIR,
    BASE_DIR,
    FindNisarASF,
    get_s1_reference_grid,
    load_country_or_belt_geometry
)
from nisar_pipeline import CalculateTrackNISAR

try:
    import asf_search as asf
    HAS_ASF = True
except ImportError:
    asf = None
    HAS_ASF = False


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def discover_country_orbits(country_code: str) -> List[int]:
    c_dir = BASE_DIR / country_code.upper()
    if not c_dir.exists():
        return []
    found = []
    for d in sorted(c_dir.glob("orbit_*")):
        m = re.search(r"orbit_(\d+)", d.name)
        if m and d.is_dir():
            found.append(int(m.group(1)))
    return sorted(list(set(found)))


def load_nisar_config(config_path: Optional[Path] = None) -> dict:
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.extend([
        script_dir / "config_nisar.json",
        script_dir / "config_nisar.example.json",
        script_dir.parent / "docs" / "1b_NISAR_toolbox" / "config.json"
    ])
    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return {
        "product_type": "GCOV",
        "flight_direction": "DESCENDING",
        "target_epsg": 3857,
        "pixel_resolution_meters": 10.0,
        "grouping_mode": "cycle",
        "cycle_days": 12,
        "border_trim_pixels": 8,
        "blend_width_pixels": 350,
        "min_aoi_coverage_pct": 50.0
    }


class NISARPreprocessorPipeline:
    """Unified 3-stage NISAR L2 GCOV Preprocessor compatible with S1 and S2."""

    def __init__(
        self,
        country: str,
        orbit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        config_file: Optional[Path] = None,
        grouping_mode: Optional[str] = None,
        overwrite: bool = False
    ):
        self.country = country.upper()
        self.orbit = int(orbit) if orbit is not None else None
        self.track = f"{self.country}/orbit_{self.orbit}" if self.orbit else self.country
        self.track_dir = (BASE_DIR / self.country / f"orbit_{self.orbit}") if self.orbit else (BASE_DIR / self.country)
        self.config = load_nisar_config(config_file)
        if grouping_mode:
            self.config["grouping_mode"] = grouping_mode
        if overwrite:
            self.config["overwrite_existing"] = True

        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else datetime.date(2025, 2, 1)
        self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else datetime.date(2025, 10, 15)

    def _authenticate_earthdata(self):
        username = os.environ.get("EARTHDATA_USERNAME") or self.config.get("earthdata_username")
        password = os.environ.get("EARTHDATA_PASSWORD") or self.config.get("earthdata_password")
        if HAS_ASF and username and password and "YOUR_" not in username:
            try:
                logging.info(f"Authenticating with NASA Earthdata as user: {username}")
                return asf.ASFSession().auth_with_creds(username, password)
            except Exception as e:
                logging.warning(f"NASA Earthdata asf_search session auth warning: {e}")
        return None

    def _resolve_target_grid_and_cutline(self):
        s1_grid = get_s1_reference_grid(self.track_dir)
        country_info = load_country_or_belt_geometry(
            self.country,
            aux_dir=AUX_DIR,
            target_epsg=int(self.config.get("target_epsg", 3857)),
            pixel_res=float(self.config.get("pixel_resolution_meters", 10.0))
        )
        if s1_grid is not None:
            logging.info(
                f"Matched Sentinel-1 reference grid: {s1_grid['ref_raster'].name} "
                f"({s1_grid['size'][0]}x{s1_grid['size'][1]} px at {s1_grid['x_res']:.1f}m, EPSG:{s1_grid['epsg']})"
            )
            cutline_shp = country_info["cutline_shp"] if country_info else None
            return s1_grid, cutline_shp

        if country_info is not None:
            logging.info(
                f"Using Eurostat NUTS / country boundary grid for {self.country}: "
                f"({country_info['size'][0]}x{country_info['size'][1]} px at 10.0m, EPSG:{country_info['epsg']})"
            )
            return country_info, country_info["cutline_shp"]

        raise FileNotFoundError(f"Could not resolve S1 reference grid or country shapefile for {self.track}")

    def run(self, stage: str = "A"):
        logging.info(f"\n============================================================")
        logging.info(f" AIML CropMapper Cloud - NISAR L-Band SAR Preprocessor")
        logging.info(f" Target Track : {self.track}")
        logging.info(f" Time Window  : {self.start_date} -> {self.end_date}")
        logging.info(f" Stage        : {stage}")
        logging.info(f"============================================================")

        target_grid, cutline_shp = self._resolve_target_grid_and_cutline()
        stage_u = stage.strip().upper()

        if stage_u in ("A", "1", "2"):
            session = self._authenticate_earthdata()
            finder = FindNisarASF()
            local_repo = Path(self.config.get("local_nisar_repository", "/eodata/NISAR"))
            products = finder.search_products(
                geometry_wkt=target_grid["wkt_4326"],
                start_date=self.start_date,
                end_date=self.end_date,
                product_type=self.config.get("product_type", "GCOV"),
                flight_direction=self.config.get("flight_direction", "DESCENDING"),
                session=session,
                local_dir=local_repo if local_repo.exists() else None
            )
            if not products:
                logging.warning(
                    f"No NISAR L2 GCOV scenes found for {self.track} between {self.start_date} and {self.end_date}. "
                    f"Downstream classifier will automatically fall back to Sentinel-1 + Sentinel-2."
                )
                return

            processor = CalculateTrackNISAR(
                list_to_download=products,
                target_grid=target_grid,
                track_dir=self.track_dir,
                country_code=self.country,
                configuration=self.config,
                session=session,
                cutline_shp=cutline_shp
            )
            processor.stage_1_and_2_extract_and_mosaic()
            if stage_u == "A":
                processor.stage_3_stack_timeseries(self.start_date, self.end_date)
        elif stage_u == "3":
            processor = CalculateTrackNISAR(
                list_to_download=[],
                target_grid=target_grid,
                track_dir=self.track_dir,
                country_code=self.country,
                configuration=self.config,
                session=None,
                cutline_shp=cutline_shp
            )
            processor.stage_3_stack_timeseries(self.start_date, self.end_date)


def interactive_wizard():
    print("""
================================================================================
     AIML CropMapper Cloud - NISAR L-Band SAR Preprocessor Setup Wizard
================================================================================
 Welcome! This wizard configures NISAR L2 GCOV polarimetric extraction,
 seamless cycle mosaicking, and EPSG:3857 (10.0 m) S1/S2 grid stacking.
================================================================================""")
    country = input(" Enter 2-letter Country Code (e.g. NL, PT, PL, FR, DE) [PL]: ").strip().upper() or "PL"
    orbits = discover_country_orbits(country)
    orbit_val = None
    if orbits:
        print(f" Discovered existing S1 orbits for {country}: {orbits}")
        orb_in = input(f" Enter single orbit number or press Enter for ALL orbits: ").strip()
        if orb_in.isdigit():
            orbit_val = int(orb_in)
    s_date = input(" Enter Start Date (YYYY-MM-DD) [2025-10-15]: ").strip() or "2025-10-15"
    e_date = input(" Enter End Date   (YYYY-MM-DD) [2026-09-15]: ").strip() or "2026-09-15"
    stage = input(" Enter Stage ('A' for All, '1' Extract, '2' Mosaic, '3' Stack) [A]: ").strip().upper() or "A"

    if orbit_val is None and orbits:
        for o_num in orbits:
            pipe = NISARPreprocessorPipeline(country=country, orbit=o_num, start_date=s_date, end_date=e_date)
            pipe.run(stage=stage)
    else:
        pipe = NISARPreprocessorPipeline(country=country, orbit=orbit_val, start_date=s_date, end_date=e_date)
        pipe.run(stage=stage)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Unified NISAR L-Band SAR (L2 GCOV) Preprocessor for AIML CropMapper Cloud.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-t", "--track", default=None, help="Track identifier (e.g. NL/orbit_88, PT/orbit_52)")
    parser.add_argument("-c", "--country", default=None, help="2-letter EU country code (e.g. NL, PT, PL, FR, DE)")
    parser.add_argument("-o", "--orbit", type=int, default=None, help="Relative S1 reference orbit number")
    parser.add_argument("-s", "--start_date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("-e", "--end_date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--stage", default="A", choices=["A", "1", "2", "3", "a"], help="Stage to run: 'A' (all), '1' (extract), '2' (mosaic), '3' (stack)")
    parser.add_argument("--grouping_mode", default=None, choices=["cycle", "dekada", "date"], help="Temporal grouping mode (default: cycle [12-day NISAR orbital cycle])")
    parser.add_argument("-j", "--config", default=None, help="Path to config_nisar.json")
    parser.add_argument("-f", "--overwrite", action="store_true", help="Force overwrite of existing mosaics and stacks")

    args = parser.parse_args()

    if not args.track and not args.country:
        interactive_wizard()
        return

    country = args.country.upper() if args.country else None
    orbit = args.orbit
    if args.track:
        norm_t = args.track.replace("\\", "/")
        if "/" in norm_t:
            parts = norm_t.split("/")
            country = parts[0].upper()
            m = re.search(r"(\d+)", parts[1])
            if m:
                orbit = int(m.group(1))
        else:
            country = norm_t.upper()

    if not country:
        parser.error("Either --track (-t) or --country (-c) must be specified.")

    cfg_path = Path(args.config) if args.config else None

    if orbit is not None:
        pipe = NISARPreprocessorPipeline(
            country=country,
            orbit=orbit,
            start_date=args.start_date,
            end_date=args.end_date,
            config_file=cfg_path,
            grouping_mode=args.grouping_mode,
            overwrite=args.overwrite
        )
        pipe.run(stage=args.stage)
    else:
        orbits = discover_country_orbits(country)
        if orbits:
            for o_num in orbits:
                pipe = NISARPreprocessorPipeline(
                    country=country,
                    orbit=o_num,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    config_file=cfg_path,
                    grouping_mode=args.grouping_mode,
                    overwrite=args.overwrite
                )
                pipe.run(stage=args.stage)
        else:
            pipe = NISARPreprocessorPipeline(
                country=country,
                orbit=None,
                start_date=args.start_date,
                end_date=args.end_date,
                config_file=cfg_path,
                grouping_mode=args.grouping_mode,
                overwrite=args.overwrite
            )
            pipe.run(stage=args.stage)


if __name__ == "__main__":
    main()
