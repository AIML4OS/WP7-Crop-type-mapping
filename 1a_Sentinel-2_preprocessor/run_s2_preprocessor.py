#!/usr/bin/env python
"""
run_s2_preprocessor.py - Unified Sentinel-2 Multi-Temporal Preprocessing Pipeline.

Provides a standardized English CLI and interactive menu for:
  Stage 1: Ingestion & SCL Cloud Masking (CREODIAS local / CDSE API)
  Stage 2: Multi-temporal Synthetic DOY Interpolation (14 target dates, 9 spectral bands)
  Stage 3: Mosaicking, Sub-pixel S1 SAR Grid Matching & 126-band BigTIFF Stacking

Usage examples:
  # 1. Run all stages for a single orbit:
  python run_s2_preprocessor.py --track NL/orbit_88 --stage A

  # 2. Run all stages for an entire country using greedy search:
  python run_s2_preprocessor.py --country NL --stage A

  # 3. Interactive menu:
  python run_s2_preprocessor.py --track NL/orbit_88
"""

import argparse
import datetime
import importlib
import logging
import os
import pathlib
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Ensure local and modules imports work cleanly
script_dir = Path(__file__).resolve().parent
modules_dir = script_dir / "modules"
for p in [script_dir, modules_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDir"))
DEFAULT_DOYS = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]

COUNTRY_ORBITS = {
    'NL': [88, 161],
    'PL': [22, 29, 73, 95, 102, 124, 146, 168, 175],
    'IE': [30, 74, 103, 132, 147],
    'FR': [8, 30, 37, 59, 81, 88, 103, 110, 132, 139, 153, 161],
    'AT': [22, 29, 73, 95, 102, 124, 146, 168],
    'PT': [81, 153, 161, 8, 88, 110],
    'DE': [22, 29, 73, 95, 102, 124, 146, 168, 175],
    'ES': [8, 81, 88, 153, 161],
    'IT': [44, 117, 146, 168]
}


def detect_s2_source() -> str:
    """Auto-detects whether CREODIAS local mount is available or CDSE API should be used."""
    local_repo = os.environ.get("S2_REPO_PATH", r"Y:\Sentinel-2\MSI\L2A")
    eodata_path = Path("/eodata/Sentinel-2")
    if Path(local_repo).exists() or eodata_path.exists():
        return "creodias"
    return "cdse"


def discover_country_orbits(country_code: str) -> List[int]:
    """Finds existing orbit folders or falls back to standard greedy orbit list."""
    c_dir = BASE_DIR / country_code.upper()
    if c_dir.exists():
        found = []
        for d in c_dir.glob("orbit_*"):
            m = re.search(r'orbit_(\d+)', d.name)
            if m:
                found.append(int(m.group(1)))
        if found:
            return sorted(list(set(found)))
    return COUNTRY_ORBITS.get(country_code.upper(), [88, 161])


class Sentinel2Pipeline:
    def __init__(
        self,
        country: str,
        orbit: Optional[int] = None,
        start_date: str = "2024-10-15",
        end_date: str = "2025-09-15",
        source: str = "auto",
        cloud_cover: float = 80.0,
        doys: List[int] = DEFAULT_DOYS,
        threads: int = 4
    ):
        self.country = country.upper()
        self.orbit = orbit
        self.start_date_str = start_date
        self.end_date_str = end_date
        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        self.cloud_cover = cloud_cover
        self.doys = doys
        self.threads = threads
        self.source = detect_s2_source() if source == "auto" else source.lower()

        if self.orbit:
            self.track = f"{self.country}/orbit_{self.orbit}"
        else:
            self.track = f"{self.country} (All greedy orbits)"

    def stage_1_download_extract(self):
        """Stage 1: Download or Extract L2A bands with SCL cloud masking."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 1/3] Sentinel-2 L2A Ingestion & SCL Masking ({self.source.upper()})")
        logging.info(f" Track: {self.track} | Date range: {self.start_date_str} to {self.end_date_str}")
        logging.info(f"============================================================")

        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for orb in orbits_to_run:
            if self.source == "creodias":
                extract_mod = importlib.import_module("1a_extract_creodias_s2")
                extract_mod.process_orbit_creodias_s2(
                    country_code=self.country,
                    orbit_num=orb,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    max_cloud_cover=self.cloud_cover,
                    max_workers=self.threads
                )
            else:
                download_mod = importlib.import_module("1b_download_cdse_s2")
                download_mod.process_orbit_cdse_s2(
                    country_code=self.country,
                    orbit_num=orb,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    cloud_cover=self.cloud_cover,
                    max_workers=self.threads
                )

    def stage_2_time_series(self):
        """Stage 2: Synthetic DOY time-series interpolation."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 2/3] Sentinel-2 Synthetic DOY Time-Series Interpolation")
        logging.info(f" Track: {self.track} | Target DOYs: {len(self.doys)} dates")
        logging.info(f"============================================================")

        ts_mod = importlib.import_module("2_time_series_s2")
        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for orb in orbits_to_run:
            track_name = f"{self.country}/orbit_{orb}"
            ts_mod.run_time_series_for_track(
                track=track_name,
                doys=self.doys,
                max_workers=self.threads
            )

    def stage_3_mosaic_stack(self):
        """Stage 3: Mosaicking, S1 SAR grid matching and BigTIFF stacking."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 3/3] Sentinel-2 Mosaicking, S1 Grid Matching & BigTIFF Stacking")
        logging.info(f" Track: {self.track}")
        logging.info(f"============================================================")

        mosaic_mod = importlib.import_module("3_mosaic_stack_clip_s2")
        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for orb in orbits_to_run:
            track_name = f"{self.country}/orbit_{orb}"
            mosaic_mod.mosaic_stack_clip_single_track(
                track=track_name,
                country_code=self.country,
                target_epsg=3857,
                doys=self.doys,
                max_workers=self.threads,
                overwrite=False,
                build_overviews=True
            )

    def run_all(self):
        """Executes all 3 stages sequentially."""
        self.stage_1_download_extract()
        self.stage_2_time_series()
        self.stage_3_mosaic_stack()
        logging.info(f"\n[SUCCESS] Sentinel-2 preprocessing pipeline completed successfully for {self.track}!")


def interactive_menu(pipeline: Sentinel2Pipeline):
    while True:
        menu_text = f"""
============================================================
 Sentinel-2 Multi-Temporal Preprocessing Pipeline
 Track  : {pipeline.track}
 Source : [{pipeline.source.upper()}] (CREODIAS local / CDSE API)
 Range  : {pipeline.start_date_str} to {pipeline.end_date_str}
============================================================
 [1] Stage 1: Ingestion & SCL cloud masking ({pipeline.source.upper()})
 [2] Stage 2: Multi-temporal synthetic DOY interpolation (14 dates)
 [3] Stage 3: Mosaicking, S1 grid matching & 126-band BigTIFF stack
 -----------------------------------------------------------
 [S] Switch data source (Toggle: CREODIAS <-> CDSE)
 [A] Run all stages automatically (1 -> 2 -> 3)
 [Q] Quit
============================================================
 Enter choice: """
        try:
            choice = input(menu_text).strip().upper()
            if choice == '1': pipeline.stage_1_download_extract()
            elif choice == '2': pipeline.stage_2_time_series()
            elif choice == '3': pipeline.stage_3_mosaic_stack()
            elif choice == 'S':
                pipeline.source = 'cdse' if pipeline.source == 'creodias' else 'creodias'
                print(f"\n    Data source switched to: {pipeline.source.upper()}")
            elif choice == 'A': pipeline.run_all()
            elif choice == 'Q': break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting pipeline.")
            break


def main():
    parser = argparse.ArgumentParser(description="Unified Sentinel-2 Multi-Temporal Preprocessing Pipeline.")
    parser.add_argument('-t', '--track', default=None, help="Track identifier, e.g. NL/orbit_88, PL/orbit_22")
    parser.add_argument('-c', '--country', default=None, help="Country code, e.g. NL, PL, FR, PT")
    parser.add_argument('-o', '--orbit', type=int, default=None, help="Specific relative orbit number")
    parser.add_argument('--stage', default=None, choices=['A', '1', '2', '3'], help="Stage to execute: 'A' (all), '1', '2', '3'")
    parser.add_argument('--source', default='auto', choices=['auto', 'creodias', 'cdse'], help="Data source (default: auto)")
    parser.add_argument('-s', '--start_date', default='2024-10-15', help="Acquisition start date (YYYY-MM-DD)")
    parser.add_argument('-e', '--end_date', default='2025-09-15', help="Acquisition end date (YYYY-MM-DD)")
    parser.add_argument('--cloud_cover', type=float, default=80.0, help="Max scene cloud cover percentage (default: 80.0)")
    parser.add_argument('--doys', nargs='+', type=int, default=DEFAULT_DOYS, help="Target DOYs list")
    parser.add_argument('--threads', type=int, default=4, help="Worker threads (default: 4)")

    args = parser.parse_args()

    country = args.country
    orbit = args.orbit
    if args.track:
        norm_track = args.track.replace('\\', '/')
        parts = norm_track.split('/')
        country = parts[0].upper()
        if len(parts) > 1:
            m = re.search(r'orbit_(\d+)', parts[1])
            if m:
                orbit = int(m.group(1))

    if not country:
        parser.error("Either --track (-t) or --country (-c) must be specified.")

    pipeline = Sentinel2Pipeline(
        country=country,
        orbit=orbit,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
        cloud_cover=args.cloud_cover,
        doys=args.doys,
        threads=args.threads
    )

    if args.stage is None:
        interactive_menu(pipeline)
    else:
        stage_choice = args.stage.strip().upper()
        if stage_choice == 'A': pipeline.run_all()
        elif stage_choice == '1': pipeline.stage_1_download_extract()
        elif stage_choice == '2': pipeline.stage_2_time_series()
        elif stage_choice == '3': pipeline.stage_3_mosaic_stack()


if __name__ == '__main__':
    main()
