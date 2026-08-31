#!/usr/bin/env python
"""
================================================================================
AIML CropMapper Cloud - Sentinel-2 Optical Preprocessing Pipeline
================================================================================
Master orchestrator for multi-temporal Sentinel-2 L2A optical reflectance data.

Features:
  - Automated Ingestion & SCL Masking: Local CREODIAS (/eodata, Y:) or Copernicus CDSE API.
  - Multi-temporal Synthetic DOY Interpolation: Pure Python linear interpolation across 14 dates.
  - Sub-pixel S1 SAR Grid Matching & BigTIFF Stacking: 126-band mosaic aligned with SAR reference.
  - Country-wide Greedy Search: Automated discovery of all S1/S2 orbits for an entire country.

Execution Examples:
  # 1. Full automated pipeline for an entire country with explicit agricultural season dates:
  python run_s2_preprocessor.py --country PT -s 2024-10-15 -e 2025-09-15 --stage A

  # 2. Full automated pipeline for a single orbit with custom dates:
  python run_s2_preprocessor.py --track PT/orbit_81 -s 2024-10-15 -e 2025-09-15 --stage A

  # 3. Force downloading directly from Copernicus Data Space (CDSE API):
  python run_s2_preprocessor.py --country PT --source cdse -s 2024-10-15 -e 2025-09-15 --stage A

  # 4. Force local extraction on CREODIAS cloud (/eodata or Y: drive):
  python run_s2_preprocessor.py --country PT --source creodias -s 2024-10-15 -e 2025-09-15 --stage A

  # 5. Run only individual stages:
  python run_s2_preprocessor.py --track PT/orbit_81 -s 2024-10-15 -e 2025-09-15 --stage 1  # Ingestion & SCL Cloud Masking only
  python run_s2_preprocessor.py --track PT/orbit_81 -s 2024-10-15 -e 2025-09-15 --stage 2  # Synthetic DOY Interpolation only
  python run_s2_preprocessor.py --track PT/orbit_81 -s 2024-10-15 -e 2025-09-15 --stage 3  # Mosaicking & BigTIFF Stacking only

  # 6. Interactive English CLI setup wizard (prompts for dates and orbits):
  python run_s2_preprocessor.py
================================================================================
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

BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDirs"))
DEFAULT_DOYS = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]

COUNTRY_ORBITS = {
    'AL': [80, 153],  # DESCENDING (100.0%) - 2 orbits
    'AT': [22, 95, 124, 168],  # DESCENDING (100.0%) - 4 orbits
    'BE': [37, 110],  # DESCENDING (100.0%) - 2 orbits
    'BG': [7, 36, 80, 109],  # DESCENDING (100.0%) - 4 orbits
    'CH': [15, 88],  # ASCENDING (100.0%) - 2 orbits
    'CY': [94, 167],  # DESCENDING (100.0%) - 2 orbits
    'CZ': [22, 95, 124],  # DESCENDING (100.0%) - 3 orbits
    'DE': [66, 95, 139, 168],  # DESCENDING (100.0%) - 4 orbits
    'DK': [139, 168],  # DESCENDING (100.0%) - 2 orbits
    'EE': [51, 80],  # DESCENDING (100.0%) - 2 orbits
    'EL': [7, 36, 80, 109],  # DESCENDING (99.91%) - 4 orbits
    'ES': [1, 30, 59, 74, 103, 132, 147],  # ASCENDING (100.0%) - 7 orbits (Mainland + Balearics)
    'FI': [7, 95, 124, 153],  # DESCENDING (100.0%) - 4 orbits
    'FR': [30, 59, 88, 103, 132, 161],  # ASCENDING (100.0%) - 6 orbits
    'GR': [7, 36, 80, 109],  # DESCENDING (99.91%) - 4 orbits
    'HR': [22, 51, 124],  # DESCENDING (100.0%) - 3 orbits
    'HU': [51, 124, 153],  # DESCENDING (100.0%) - 3 orbits
    'IE': [1, 74],  # ASCENDING (100.0%) - 2 orbits
    'IS': [53, 82, 111],  # DESCENDING (100.0%) - 3 orbits
    'IT': [15, 44, 88, 117, 146],  # ASCENDING (100.0%) - 5 orbits
    'LI': [66],  # DESCENDING (100.0%) - 1 orbits
    'LT': [58, 131],  # ASCENDING (99.95%) - 2 orbits
    'LU': [37],  # DESCENDING (100.0%) - 1 orbits
    'LV': [7, 51, 80],  # DESCENDING (100.0%) - 3 orbits
    'ME': [153],  # DESCENDING (100.0%) - 1 orbits
    'MK': [80],  # DESCENDING (100.0%) - 1 orbits
    'MT': [124],  # DESCENDING (100.0%) - 1 orbits
    'NL': [37, 110],  # DESCENDING (100.0%) - 2 orbits
    'NO': [8, 37, 66, 124, 153, 168],  # DESCENDING (100.0%) - 6 orbits (Mainland Norway)
    'PL': [29, 73, 102, 131, 175],  # ASCENDING (100.0%) - 5 orbits
    'PT': [52, 125],  # DESCENDING (100.0%) - 2 orbits
    'RO': [29, 58, 102, 131],  # ASCENDING (100.0%) - 4 orbits
    'RS': [102, 175],  # ASCENDING (100.0%) - 2 orbits
    'SE': [22, 66, 168],  # DESCENDING (100.0%) - 3 orbits
    'SI': [22, 124],  # DESCENDING (100.0%) - 2 orbits
    'SK': [51, 124, 153],  # DESCENDING (100.0%) - 3 orbits
    'TR': [21, 36, 50, 65, 94, 123, 138, 152, 167],  # DESCENDING (99.96%) - 9 orbits
    'UK': [23, 52, 81, 154],  # DESCENDING (100.0%) - 4 orbits
}


def detect_s2_source() -> str:
    """Auto-detects whether CREODIAS local mount is available or CDSE API should be used."""
    local_repo = os.environ.get("S2_REPO_PATH", r"Y:\Sentinel-2\MSI\L2A")
    eodata_path = Path("/eodata/Sentinel-2")
    if Path(local_repo).exists() or eodata_path.exists():
        return "creodias"
    return "cdse"


def discover_country_orbits(country_code: str) -> List[int]:
    """Finds existing orbit folders or falls back to master orbit registry."""
    found = set()
    for b in [BASE_DIR, Path(r"D:/AIML_CropMapper_Cloud/workingDir")]:
        c_dir = b / country_code.upper()
        if c_dir.exists():
            for d in c_dir.glob("orbit_*"):
                m = re.search(r'orbit_(\d+)', d.name)
                if m:
                    found.add(int(m.group(1)))
    if found:
        return sorted(list(found))
    json_path = Path(__file__).resolve().parent.parent / "auxiliary_files" / "country_orbits_complete.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if country_code.upper() in data and data[country_code.upper()]:
                    return data[country_code.upper()]
        except Exception:
            pass
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
        threads: int = 8,
        overwrite: bool = False
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
        self.overwrite = overwrite
        self.source = detect_s2_source() if source == "auto" else source.lower()

        if self.orbit:
            self.track = f"{self.country}/orbit_{self.orbit}"
        else:
            self.track = f"{self.country} (All greedy orbits)"

    def stage_1_download_extract(self):
        """Stage 1: Download or Extract L2A bands with SCL cloud masking into shared country S2 pool."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 1/3] Sentinel-2 L2A Ingestion & SCL Masking ({self.source.upper()})")
        logging.info(f" Track: {self.track} | Date range: {self.start_date_str} to {self.end_date_str}")
        logging.info(f"============================================================")

        if self.orbit:
            orbits_to_run = [self.orbit]
            for orb in orbits_to_run:
                if self.source == "creodias":
                    extract_mod = importlib.import_module("s2_extract_creodias")
                    extract_mod.process_orbit_creodias_s2(
                        country_code=self.country,
                        orbit_num=orb,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        max_cloud_cover=self.cloud_cover,
                        max_workers=self.threads
                    )
                else:
                    download_mod = importlib.import_module("s2_download_cdse")
                    download_mod.process_orbit_cdse_s2(
                        country_code=self.country,
                        orbit_num=orb,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        cloud_cover=self.cloud_cover,
                        max_workers=self.threads
                    )
        else:
            if self.source == "creodias":
                extract_mod = importlib.import_module("s2_extract_creodias")
                extract_mod.process_country_creodias_s2(
                    country_code=self.country,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    max_cloud_cover=self.cloud_cover,
                    max_workers=self.threads
                )
            else:
                download_mod = importlib.import_module("s2_download_cdse")
                download_mod.process_country_cdse_s2(
                    country_code=self.country,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    cloud_cover=self.cloud_cover,
                    max_workers=self.threads
                )

    def stage_2_time_series(self):
        """Stage 2: Synthetic DOY time-series interpolation across shared country S2 pool."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 2/3] Sentinel-2 Synthetic DOY Time-Series Interpolation")
        logging.info(f" Track: {self.track} | Target DOYs: {len(self.doys)} dates")
        logging.info(f"============================================================")

        ts_mod = importlib.import_module("s2_time_series")
        if self.orbit:
            ts_mod.run_time_series_for_track(
                track=f"{self.country}/orbit_{self.orbit}",
                doys=self.doys,
                max_workers=self.threads,
                overwrite=self.overwrite
            )
        else:
            ts_mod.run_time_series(
                country=self.country,
                doys=self.doys,
                max_workers=self.threads,
                overwrite=self.overwrite
            )

    def stage_3_mosaic_stack(self):
        """Stage 3: Mosaicking from shared country S2 pool, S1 SAR grid matching and BigTIFF stacking."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 3/3] Sentinel-2 Mosaicking, S1 Grid Matching & BigTIFF Stacking")
        logging.info(f" Track: {self.track}")
        logging.info(f"============================================================")

        mosaic_mod = importlib.import_module("s2_mosaic_stack")
        pipeline_mod = importlib.import_module("s2_pipeline")
        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for idx, orb in enumerate(orbits_to_run):
            track_name = f"{self.country}/orbit_{orb}"
            # For the first orbit, respect self.overwrite. For subsequent orbits in the same run,
            # attempt instant reuse/fast-warp of the first orbit's completed stack (saving ~70h per orbit).
            if idx > 0 and pipeline_mod.try_reuse_existing_country_s2_stack(track_name, self.country, overwrite=False):
                continue

            mosaic_mod.mosaic_stack_clip_single_track(
                track=track_name,
                country_code=self.country,
                target_epsg=3857,
                doys=self.doys,
                max_workers=self.threads,
                overwrite=self.overwrite,
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
 [D] Change Date Range (Current: {pipeline.start_date_str} to {pipeline.end_date_str})
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
            elif choice == 'D':
                new_s = input(f" Enter new start date [YYYY-MM-DD] (current: {pipeline.start_date_str}): ").strip()
                new_e = input(f" Enter new end date [YYYY-MM-DD] (current: {pipeline.end_date_str}): ").strip()
                if new_s:
                    pipeline.start_date_str = new_s
                    pipeline.start_date = datetime.datetime.strptime(new_s, "%Y-%m-%d").date()
                if new_e:
                    pipeline.end_date_str = new_e
                    pipeline.end_date = datetime.datetime.strptime(new_e, "%Y-%m-%d").date()
                print(f"\n    Date range updated to: {pipeline.start_date_str} -> {pipeline.end_date_str}")
            elif choice == 'S':
                pipeline.source = 'cdse' if pipeline.source == 'creodias' else 'creodias'
                print(f"\n    Data source switched to: {pipeline.source.upper()}")
            elif choice == 'A': pipeline.run_all()
            elif choice == 'Q': break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting pipeline.")
            break


def interactive_setup_wizard():
    """Interactive CLI wizard shown when run_s2_preprocessor.py is called without arguments."""
    print("""
============================================================
 AIML CropMapper Cloud - Sentinel-2 Optical Preprocessor Wizard
============================================================""")

    # Step 1: Discover tracks
    tracks = []
    if BASE_DIR.exists():
        for c_dir in sorted(BASE_DIR.iterdir()):
            if c_dir.is_dir() and len(c_dir.name) in [2, 3]:
                for orb_dir in sorted(c_dir.glob("orbit_*")):
                    if orb_dir.is_dir():
                        tracks.append(f"{c_dir.name}/{orb_dir.name}")

    selected_track = None
    selected_country = None
    selected_orbit = None

    if tracks:
        print(" Discovered tracks in working directory:")
        for idx, t in enumerate(tracks, 1):
            print(f"  [{idx}] {t}")
        print("  [C] Enter custom track (e.g. PL/orbit_22)")
        print("  [N] Process entire country (e.g. NL, PL, FR)")
        print("  [Q] Quit")
        choice = input("\n Select track or option [1-%d/C/N/Q] (default: 1): " % len(tracks)).strip().upper()
        if choice == 'Q': return
        elif choice == 'C':
            selected_track = input(" Enter track identifier (e.g. NL/orbit_88): ").strip()
        elif choice == 'N':
            selected_country = input(" Enter country code (e.g. NL, PL, FR): ").strip().upper()
        elif choice.isdigit() and 1 <= int(choice) <= len(tracks):
            selected_track = tracks[int(choice) - 1]
        else:
            selected_track = tracks[0]
    else:
        val = input(" Enter track (e.g. NL/orbit_88) or country (e.g. NL): ").strip()
        if '/' in val:
            selected_track = val
        else:
            selected_country = val.upper()

    if selected_track:
        parts = selected_track.replace('\\', '/').split('/')
        selected_country = parts[0].upper()
        if len(parts) > 1:
            m = re.search(r'orbit_(\d+)', parts[1])
            if m:
                selected_orbit = int(m.group(1))

    # Step 2: Select Source
    detected = detect_s2_source()
    print(f"""
============================================================
 Select Data Ingestion Source:
  [1] Auto-detect (Current: {detected.upper()})
  [2] CREODIAS Local Storage (/eodata or Y: drive)
  [3] Copernicus Data Space Ecosystem API (CDSE download)
============================================================""")
    src_choice = input(" Enter choice [1-3] (default: 1): ").strip()
    sources = {'1': 'auto', '2': 'creodias', '3': 'cdse'}
    source = sources.get(src_choice, 'auto')

    # Step 3: Date Range
    start_date = input(" Enter acquisition start date [YYYY-MM-DD] (default: 2024-10-15): ").strip() or "2024-10-15"
    end_date = input(" Enter acquisition end date [YYYY-MM-DD] (default: 2025-09-15): ").strip() or "2025-09-15"

    pipeline = Sentinel2Pipeline(
        country=selected_country,
        orbit=selected_orbit,
        start_date=start_date,
        end_date=end_date,
        source=source
    )
    interactive_menu(pipeline)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Sentinel-2 Multi-Temporal Preprocessing Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive wizard (zero arguments):
  python run_s2_preprocessor.py

  # Single orbit automated run:
  python run_s2_preprocessor.py --track NL/orbit_88 --stage A

  # Entire country automated run:
  python run_s2_preprocessor.py --country NL --stage A
"""
    )
    parser.add_argument('-t', '--track', default=None, help="Track identifier (e.g. NL/orbit_88, PL/orbit_22)")
    parser.add_argument('-c', '--country', default=None, help="Country code (e.g. NL, PL, FR, PT, ES, DE)")
    parser.add_argument('-o', '--orbit', type=int, default=None, help="Specific relative orbit number")
    parser.add_argument('--stage', default=None, choices=['A', '1', '2', '3'], help="Stage to execute: 'A' (all), '1' (ingest), '2' (time series), '3' (stack)")
    parser.add_argument('--source', default='auto', choices=['auto', 'creodias', 'cdse'], help="Data source: 'auto' (detect local), 'creodias', 'cdse' (default: auto)")
    parser.add_argument('-s', '--start_date', default='2024-10-15', help="Acquisition start date (YYYY-MM-DD, default: 2024-10-15)")
    parser.add_argument('-e', '--end_date', default='2025-09-15', help="Acquisition end date (YYYY-MM-DD, default: 2025-09-15)")
    parser.add_argument('--cloud_cover', type=float, default=80.0, help="Max scene cloud cover percentage (default: 80.0)")
    parser.add_argument('--doys', nargs='+', type=int, default=DEFAULT_DOYS, help="Target DOYs list (default: 14 dates)")
    parser.add_argument('--threads', type=int, default=8, help="Worker threads for parallel processing (default: 8)")
    parser.add_argument('--overwrite', action='store_true', help="Force re-generation of existing intermediate and final stacks")

    args = parser.parse_args()

    # Zero arguments: open interactive wizard!
    if not args.track and not args.country and not args.stage:
        interactive_setup_wizard()
        return

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
        threads=args.threads,
        overwrite=args.overwrite
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
