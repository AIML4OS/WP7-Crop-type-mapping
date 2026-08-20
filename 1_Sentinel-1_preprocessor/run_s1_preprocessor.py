#!/usr/bin/env python
"""
================================================================================
AIML CropMapper Cloud - Sentinel-1 SAR Preprocessing Pipeline
================================================================================
Master orchestrator for multi-temporal Sentinel-1 GRDH SAR data (Sigma0 VH/VV).

Features:
  - Automated Ingestion & Calibration: Local CREODIAS (/eodata, Y:) or Copernicus CDSE API.
  - Multi-temporal Coregistration: ESA SNAP GPT for precise pixel alignment.
  - Time-series Stacking & Area Clipping: Sub-pixel resampling and BigTIFF generation.
  - Country-wide Greedy Search: Minimum set cover orbit detection across EU countries.

Execution Examples:
  # 1. Full automated pipeline for a single orbit (all stages):
  python run_s1_preprocessor.py --track NL/orbit_88 --stage A

  # 2. Full automated pipeline for an entire country (Greedy search):
  python run_s1_preprocessor.py --country NL --stage A

  # 3. Force downloading directly from Copernicus Data Space (CDSE API):
  python run_s1_preprocessor.py --track NL/orbit_88 --source cdse --stage A

  # 4. Force local extraction on CREODIAS cloud (/eodata or Y: drive):
  python run_s1_preprocessor.py --track NL/orbit_88 --source creodias --stage A

  # 5. Run only individual stages:
  python run_s1_preprocessor.py --track NL/orbit_88 --stage 1  # Ingestion & Calibration only
  python run_s1_preprocessor.py --track NL/orbit_88 --stage 2  # SNAP Coregistration only
  python run_s1_preprocessor.py --track NL/orbit_88 --stage 3  # Time-series Stacking & Clipping only

  # 6. Custom acquisition date range and worker threads:
  python run_s1_preprocessor.py --track NL/orbit_88 -s 2024-10-15 -e 2025-09-15 --threads 8 --stage A

  # 7. Interactive English CLI menu:
  python run_s1_preprocessor.py --track NL/orbit_88
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


def detect_s1_source() -> str:
    """Auto-detects whether CREODIAS local mount is available or CDSE API should be used."""
    local_repo = os.environ.get("S1_REPO_PATH", r"Y:\Sentinel-1\SAR\IW_GRDH_1S")
    eodata_path = Path("/eodata/Sentinel-1")
    if Path(local_repo).exists() or eodata_path.exists():
        return "creodias"
    return "cdse"


def discover_country_orbits(country_code: str) -> List[int]:
    """Finds existing orbit folders or falls back to standard greedy orbit list."""
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
    return COUNTRY_ORBITS.get(country_code.upper(), [88, 161])


class Sentinel1Pipeline:
    def __init__(
        self,
        country: str,
        orbit: Optional[int] = None,
        start_date: str = "2024-10-15",
        end_date: str = "2025-09-15",
        source: str = "auto",
        threads: int = 4
    ):
        self.country = country.upper()
        self.orbit = orbit
        self.start_date = start_date
        self.end_date = end_date
        self.threads = threads
        self.source = detect_s1_source() if source == "auto" else source.lower()

        if self.orbit:
            self.track = f"{self.country}/orbit_{self.orbit}"
        else:
            self.track = f"{self.country} (All greedy orbits)"

    def stage_1_calibration(self):
        """Stage 1: Calibration & Slicing via CREODIAS or CDSE API."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 1/3] Sentinel-1 SAR Ingestion & Calibration ({self.source.upper()})")
        logging.info(f" Track: {self.track} | Date range: {self.start_date} to {self.end_date}")
        logging.info(f"============================================================")

        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for orb in orbits_to_run:
            if self.source == "creodias":
                calib_mod = importlib.import_module("s1_calibration_creodias")
                calib_mod.process_orbit(
                    country=self.country,
                    orbit=orb,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
            else:
                calib_cdse_mod = importlib.import_module("s1_calibration_cdse")
                calib_cdse_mod.process_orbit_cdse(
                    country=self.country,
                    orbit=orb,
                    start_date=self.start_date,
                    end_date=self.end_date
                )

    def stage_2_coregistration(self):
        """Stage 2: SNAP Multi-temporal Coregistration."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 2/3] Sentinel-1 Multi-temporal Coregistration (SNAP)")
        logging.info(f" Track: {self.track}")
        logging.info(f"============================================================")

        coreg_mod = importlib.import_module("s1_coregistration")
        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for orb in orbits_to_run:
            track_name = f"{self.country}/orbit_{orb}"
            coreg_mod.process_track(track_name)

    def stage_3_stack_clip(self):
        """Stage 3: Multi-temporal Stacking & Area Clipping (BigTIFF)."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 3/3] Sentinel-1 Time-series Stacking & Clipping (BigTIFF)")
        logging.info(f" Track: {self.track}")
        logging.info(f"============================================================")

        stack_mod = importlib.import_module("s1_stack_clip")
        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for orb in orbits_to_run:
            track_name = f"{self.country}/orbit_{orb}"
            stack_mod.process_track(track_name)

    def run_all(self):
        """Executes all 3 stages sequentially."""
        self.stage_1_calibration()
        self.stage_2_coregistration()
        self.stage_3_stack_clip()
        logging.info(f"\n[SUCCESS] Sentinel-1 preprocessing pipeline completed successfully for {self.track}!")


def interactive_menu(pipeline: Sentinel1Pipeline):
    while True:
        menu_text = f"""
============================================================
 Sentinel-1 SAR Preprocessing Pipeline (Sigma0 VH/VV)
 Track  : {pipeline.track}
 Source : [{pipeline.source.upper()}] (CREODIAS local / CDSE API)
 Range  : {pipeline.start_date} to {pipeline.end_date}
============================================================
 [1] Stage 1: Ingestion & calibration ({pipeline.source.upper()})
 [2] Stage 2: SNAP coregistration & terrain correction
 [3] Stage 3: Multi-temporal stacking & AOI clipping
 -----------------------------------------------------------
 [S] Switch data source (Toggle: CREODIAS <-> CDSE)
 [A] Run all stages automatically (1 -> 2 -> 3)
 [Q] Quit
============================================================
 Enter choice: """
        try:
            choice = input(menu_text).strip().upper()
            if choice == '1': pipeline.stage_1_calibration()
            elif choice == '2': pipeline.stage_2_coregistration()
            elif choice == '3': pipeline.stage_3_stack_clip()
            elif choice == 'S':
                pipeline.source = 'cdse' if pipeline.source == 'creodias' else 'creodias'
                print(f"\n    Data source switched to: {pipeline.source.upper()}")
            elif choice == 'A': pipeline.run_all()
            elif choice == 'Q': break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting pipeline.")
            break


def interactive_setup_wizard():
    """Interactive CLI wizard shown when run_s1_preprocessor.py is called without arguments."""
    print("""
============================================================
 AIML CropMapper Cloud - Sentinel-1 SAR Preprocessor Wizard
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
    detected = detect_s1_source()
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

    pipeline = Sentinel1Pipeline(
        country=selected_country,
        orbit=selected_orbit,
        start_date=start_date,
        end_date=end_date,
        source=source
    )
    interactive_menu(pipeline)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Sentinel-1 SAR Preprocessing Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive wizard (zero arguments):
  python run_s1_preprocessor.py

  # Single orbit automated run:
  python run_s1_preprocessor.py --track NL/orbit_88 --stage A

  # Entire country greedy run:
  python run_s1_preprocessor.py --country NL --stage A
"""
    )
    parser.add_argument('-t', '--track', default=None, help="Track identifier (e.g. NL/orbit_88, PL/orbit_22)")
    parser.add_argument('-c', '--country', default=None, help="Country code (e.g. NL, PL, FR, PT, ES, DE)")
    parser.add_argument('-o', '--orbit', type=int, default=None, help="Specific relative orbit number")
    parser.add_argument('--stage', default=None, choices=['A', '1', '2', '3'], help="Stage to execute: 'A' (all), '1' (calib), '2' (snap), '3' (stack)")
    parser.add_argument('--source', default='auto', choices=['auto', 'creodias', 'cdse'], help="Data source: 'auto' (detect local), 'creodias', 'cdse' (default: auto)")
    parser.add_argument('-s', '--start_date', default='2024-10-15', help="Acquisition start date (YYYY-MM-DD, default: 2024-10-15)")
    parser.add_argument('-e', '--end_date', default='2025-09-15', help="Acquisition end date (YYYY-MM-DD, default: 2025-09-15)")
    parser.add_argument('--threads', type=int, default=4, help="Worker threads for parallel processing (default: 4)")

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

    pipeline = Sentinel1Pipeline(
        country=country,
        orbit=orbit,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
        threads=args.threads
    )

    if args.stage is None:
        interactive_menu(pipeline)
    else:
        stage_choice = args.stage.strip().upper()
        if stage_choice == 'A': pipeline.run_all()
        elif stage_choice == '1': pipeline.stage_1_calibration()
        elif stage_choice == '2': pipeline.stage_2_coregistration()
        elif stage_choice == '3': pipeline.stage_3_stack_clip()


if __name__ == '__main__':
    main()
