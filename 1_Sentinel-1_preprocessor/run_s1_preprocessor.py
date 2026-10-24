#!/usr/bin/env python
"""
================================================================================
AIML CropMapper Cloud - Sentinel-1 SAR Preprocessing Pipeline
================================================================================
Master orchestrator for multi-temporal Sentinel-1 GRDH SAR data (Sigma0 VH/VV).

Supports two data ingestion pathways and two processing engines:
  Data Ingestion Pathways (--source):
    1. Copernicus Data Space Ecosystem API (cdse): Automated OData catalog query
       and .SAFE archive download.
    2. Local CreoDIAS Repository (creodias / cog): Direct high-speed reading from
       mounted /eodata (or local drive) .SAFE and .SAFE-COG archives.

  Processing Engines (--engine):
    1. Pure Python Engine (python - default): 100% standalone Python/NumPy/SciPy/GDAL
       pipeline performing 2D Thermal Noise Removal (TNR), GRD Border Noise Removal
       (BNR), Sigma0 LUT Calibration, Range-Doppler Terrain Correction with
       Copernicus 30m Global DEM onto a locked 10.0m EPSG:3857 grid, and
       Multi-Temporal Quegan Speckle Filtering without ESA SNAP.
    2. ESA SNAP GPT Engine (snap): Graph Processing Tool pipeline using
       Copernicus 30m Global DEM with bilinear interpolation.

Execution Examples:
  # 1. Interactive CLI setup wizard (zero arguments):
  python run_s1_preprocessor.py

  # 2. Pure Python full pipeline using CreoDIAS local COG/SAFE repository:
  python run_s1_preprocessor.py --country PT --source creodias --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A --threads 4

  # 3. Pure Python full pipeline downloading directly from Copernicus CDSE API:
  python run_s1_preprocessor.py --country PT --source cdse --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A --threads 4

  # 4. Single orbit track execution in Pure Python (e.g. Portugal Orbit 52):
  python run_s1_preprocessor.py --track PT/orbit_52 --source creodias --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A

  # 5. Pure Python with 14-DOY temporal interpolation matching Sentinel-2 DOY nodes:
  python run_s1_preprocessor.py --track PT/orbit_52 --engine python --temporal_mode doy --stage 3

  # 6. Legacy ESA SNAP GPT engine execution:
  python run_s1_preprocessor.py --track PT/orbit_52 --source cog --engine snap -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A
================================================================================
"""

import argparse
import datetime
import importlib
import json
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


def _load_s1_config() -> dict:
    cfg_path = script_dir / "config_s1.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to parse {cfg_path}: {e}")
    return {}


_CFG = _load_s1_config()
_PATHS = _CFG.get("paths", {})

BASE_DIR = Path(
    os.environ.get(
        "AIML_WORKING_DIR",
        _PATHS.get("working_dir", str(script_dir.parent / "workingDirs")),
    )
)
AUX_DIR = Path(
    os.environ.get(
        "AIML_AUX_DIR",
        _PATHS.get("aux_dir", str(script_dir.parent / "auxiliary_files")),
    )
)

COUNTRY_ORBITS = {
    'AL': [80, 153],
    'AT': [22, 95, 124, 168],
    'BE': [37, 110],
    'BG': [7, 36, 80, 109],
    'CH': [15, 88],
    'CY': [94, 167],
    'CZ': [22, 95, 124],
    'DE': [66, 95, 139, 168],
    'DK': [139, 168],
    'EE': [51, 80],
    'EL': [7, 36, 80, 109],
    'ES': [1, 30, 59, 74, 103, 132, 147],
    'FI': [7, 95, 124, 153],
    'FR': [30, 59, 88, 103, 132, 161],
    'GR': [7, 36, 80, 109],
    'HR': [22, 51, 124],
    'HU': [51, 124, 153],
    'IE': [1, 74],
    'IS': [53, 82, 111],
    'IT': [15, 44, 88, 117, 146],
    'LI': [66],
    'LT': [58, 131],
    'LU': [37],
    'LV': [7, 51, 80],
    'ME': [153],
    'MK': [80],
    'MT': [124],
    'NL': [37, 110],
    'NO': [8, 37, 66, 124, 153, 168],
    'PL': [29, 73, 102, 175],
    'PT': [52, 125],
    'RO': [29, 58, 102, 131],
    'RS': [102, 175],
    'SE': [22, 66, 168],
    'SI': [22, 124],
    'SK': [51, 124, 153],
    'TR': [21, 36, 50, 65, 94, 123, 138, 152, 167],
    'UK': [23, 52, 81, 154],
}


def detect_s1_source() -> str:
    """Auto-detects whether COG, CREODIAS local mount, or CDSE API should be used."""
    cog_candidates = [
        Path(os.environ.get("S1_COG_REPO_PATH", "/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG")),
        Path(r"Y:\Sentinel-1\SAR\IW_GRDH_1S-COG"),
    ]
    for p in cog_candidates:
        if p.exists():
            return "cog"

    local_repo = Path(
        os.environ.get("S1_REPO_PATH", _PATHS.get("s1_repo_path", "/eodata/Sentinel-1/SAR/IW_GRDH_1S"))
    )
    eodata_path = Path("/eodata/Sentinel-1")
    if local_repo.exists() or eodata_path.exists():
        return "creodias"
    return "cdse"


def discover_country_orbits(country_code: str) -> List[int]:
    """Finds existing orbit folders or falls back to greedy optimal orbit registry."""
    c = country_code.upper()
    found = set()
    for b in [BASE_DIR, Path(r"D:/AIML_CropMapper_Cloud/workingDir")]:
        c_dir = b / c
        if c_dir.exists():
            for d in c_dir.glob("orbit_*"):
                m = re.search(r'orbit_(\d+)', d.name)
                if m:
                    has_slice = (d / "slice_assembly").exists() and any((d / "slice_assembly").iterdir())
                    has_wrapped = (d / "wrapped").exists() and any((d / "wrapped").iterdir())
                    has_final = (d / "S1_final_preprocessing").exists() and any((d / "S1_final_preprocessing").glob("*.dim"))
                    has_stack = (d / "1_input_stacks").exists() and any((d / "1_input_stacks").glob("*.tif"))
                    if has_slice or has_wrapped or has_final or has_stack:
                        found.add(int(m.group(1)))
    if found:
        return sorted(list(found))

    greedy_path = AUX_DIR / "country_optimal_greedy_orbits_clean.json"
    if greedy_path.exists():
        try:
            with open(greedy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if c in data and isinstance(data[c], dict) and data[c].get("orbits"):
                return [int(o) for o in data[c]["orbits"]]
        except Exception:
            pass

    if c in COUNTRY_ORBITS:
        return COUNTRY_ORBITS[c]

    json_path = AUX_DIR / "country_orbits_complete.json"
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if c in data and data[c]:
                    return [int(o) for o in data[c]]
        except Exception:
            pass
    return []


class Sentinel1Pipeline:
    def __init__(
        self,
        country: str,
        orbit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exclude_winter: bool = False,
        source: str = "auto",
        engine: str = "python",
        filter_window: int = 7,
        threads: int = 4,
        overwrite: bool = False
    ):
        self.country = country.upper()
        self.orbit = orbit
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_winter = exclude_winter
        self.threads = threads
        self.overwrite = overwrite
        self.source = detect_s1_source() if source == "auto" else source.lower()
        self.engine = engine.lower()
        self.filter_window = filter_window

        if self.orbit:
            self.track = f"{self.country}/orbit_{self.orbit}"
        else:
            self.track = f"{self.country} (All greedy orbits)"

    def prompt_dates(self):
        """Forces the user to enter valid start and end dates."""
        print("\n--- Mandatory Acquisition Dates Configuration ---")
        while True:
            s_input = input(" Enter acquisition start date [YYYY-MM-DD] (e.g. 2024-10-15): ").strip()
            try:
                datetime.datetime.strptime(s_input, "%Y-%m-%d")
                self.start_date = s_input
                break
            except ValueError:
                print(" [ERROR] Invalid date format! Please enter date as YYYY-MM-DD.")

        while True:
            e_input = input(" Enter acquisition end date [YYYY-MM-DD] (e.g. 2025-09-15): ").strip()
            try:
                datetime.datetime.strptime(e_input, "%Y-%m-%d")
                self.end_date = e_input
                break
            except ValueError:
                print(" [ERROR] Invalid date format! Please enter date as YYYY-MM-DD.")

    def stage_1_calibration(self):
        """Stage 1: Calibration & Slicing via COG, CREODIAS SAFE, or CDSE API."""
        if not self.start_date or not self.end_date:
            self.prompt_dates()

        winter_str = " [Excluding Winter 01.12-14.02]" if self.exclude_winter else ""
        logging.info(f"\n============================================================")
        logging.info(
            f" [Stage 1/3] Sentinel-1 SAR Ingestion & Calibration "
            f"(Source: {self.source.upper()} | Engine: {self.engine.upper()}){winter_str}"
        )
        logging.info(f" Track: {self.track} | Date range: {self.start_date} to {self.end_date}")
        logging.info(f"============================================================")

        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for orb in orbits_to_run:
            track_dir = BASE_DIR / self.country / f"orbit_{orb}"
            wrapped_dir = track_dir / "wrapped"
            final_proc_dir = track_dir / "S1_final_preprocessing"
            has_wrapped = wrapped_dir.exists() and (
                any(wrapped_dir.glob('*.dim')) or (wrapped_dir / "python_stack_manifest.json").exists()
            )
            has_final = final_proc_dir.exists() and any(final_proc_dir.glob('*.dim'))

            if (has_wrapped or has_final) and not self.overwrite:
                logging.info(
                    f"[{self.country}/orbit_{orb}] Coregistered stack or final processed products already exist. "
                    f"Skipping Stage 1 Calibration."
                )
                continue

            if self.source == "cog":
                calib_cog_mod = importlib.import_module("s1_calibration_cog")
                calib_cog_mod.process_orbit_cog(
                    country=self.country,
                    orbit=orb,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    exclude_winter=self.exclude_winter,
                    max_workers=self.threads,
                    engine=self.engine,
                    overwrite=self.overwrite,
                )
            elif self.source == "creodias":
                calib_mod = importlib.import_module("s1_calibration_creodias")
                calib_mod.process_orbit(
                    country=self.country,
                    orbit=orb,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    exclude_winter=self.exclude_winter,
                    max_workers=self.threads,
                    engine=self.engine,
                    overwrite=self.overwrite,
                )
            else:
                calib_cdse_mod = importlib.import_module("s1_calibration_cdse")
                calib_cdse_mod.process_orbit_cdse(
                    country=self.country,
                    orbit=orb,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    exclude_winter=self.exclude_winter,
                    max_workers=self.threads,
                    engine=self.engine,
                    overwrite=self.overwrite,
                )

    def stage_2_coregistration(self):
        """Stage 2: Multi-temporal coregistration (Pure Python grid alignment or ESA SNAP GPT)."""
        logging.info(f"\n============================================================")
        logging.info(f" [Stage 2/3] Sentinel-1 Multi-Temporal Coregistration (Engine: {self.engine.upper()})")
        logging.info(f" Track: {self.track}")
        logging.info(f"============================================================")

        coreg_mod = importlib.import_module("s1_coregistration")
        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for orb in orbits_to_run:
            coreg_mod.process_orbit_coregistration(
                country=self.country,
                orbit=orb,
                overwrite=self.overwrite,
                engine=self.engine,
            )

    def stage_3_stack_clip(self):
        """Stage 3: Multi-temporal despeckling, Web Mercator stacking, and NUTS2 clipping."""
        logging.info(f"\n============================================================")
        logging.info(
            f" [Stage 3/3] Sentinel-1 Multi-Temporal Despeckling, Stacking & Clipping "
            f"(Engine: {self.engine.upper()})"
        )
        logging.info(f" Track: {self.track}")
        logging.info(f"============================================================")

        stack_mod = importlib.import_module("s1_stack_clip")
        orbits_to_run = [self.orbit] if self.orbit else discover_country_orbits(self.country)
        for orb in orbits_to_run:
            stack_mod.process_orbit_stack_and_clip(
                country=self.country,
                orbit=orb,
                overwrite=self.overwrite,
                engine=self.engine,
                filter_window=self.filter_window,
            )

    def run_all(self):
        """Executes all 3 stages sequentially."""
        if not self.start_date or not self.end_date:
            self.prompt_dates()
        self.stage_1_calibration()
        self.stage_2_coregistration()
        self.stage_3_stack_clip()
        logging.info(f"\n[SUCCESS] Sentinel-1 preprocessing pipeline completed successfully for {self.track}!")


def interactive_menu(pipeline: Sentinel1Pipeline):
    while True:
        range_str = f"{pipeline.start_date} to {pipeline.end_date}" if (pipeline.start_date and pipeline.end_date) else "[NOT SET - REQUIRED]"
        menu_text = f"""
============================================================
 Sentinel-1 SAR Preprocessing Pipeline (Sigma0 VH/VV)
 Track  : {pipeline.track}
 Source : [{pipeline.source.upper()}] (CREODIAS local / CDSE API)
 Engine : [{pipeline.engine.upper()}] (Pure Python / ESA SNAP)
 Range  : {range_str}
============================================================
 [1] Stage 1: Ingestion & calibration ({pipeline.source.upper()} | {pipeline.engine.upper()})
 [2] Stage 2: Multi-temporal coregistration ({pipeline.engine.upper()})
 [3] Stage 3: Despeckling, stacking & NUTS2 clipping ({pipeline.engine.upper()})
 -----------------------------------------------------------
 [D] Change / Enter acquisition dates
 [S] Switch data source (Cycle: COG -> CREODIAS -> CDSE)
 [E] Switch processing engine (Toggle: PYTHON <-> SNAP)
 [A] Run all stages automatically (1 -> 2 -> 3)
 [Q] Quit
============================================================
 Enter choice: """
        try:
            choice = input(menu_text).strip().upper()
            if choice == '1':
                pipeline.stage_1_calibration()
            elif choice == '2':
                pipeline.stage_2_coregistration()
            elif choice == '3':
                pipeline.stage_3_stack_clip()
            elif choice == 'D':
                pipeline.prompt_dates()
            elif choice == 'S':
                cycle = {'cog': 'creodias', 'creodias': 'cdse', 'cdse': 'cog'}
                pipeline.source = cycle.get(pipeline.source, 'creodias')
                print(f"\n    Data source switched to: {pipeline.source.upper()}")
            elif choice == 'E':
                pipeline.engine = 'snap' if pipeline.engine == 'python' else 'python'
                print(f"\n    Processing engine switched to: {pipeline.engine.upper()}")
            elif choice == 'A':
                pipeline.run_all()
            elif choice == 'Q':
                break
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
        print("  [C] Enter custom track (e.g. PL/orbit_29)")
        print("  [N] Process entire country (e.g. NL, PL, FR, PT)")
        print("  [Q] Quit")
        choice = input("\n Select track or option [1-%d/C/N/Q] (default: 1): " % len(tracks)).strip().upper()
        if choice == 'Q':
            return
        elif choice == 'C':
            selected_track = input(" Enter track identifier (e.g. NL/orbit_37): ").strip()
        elif choice == 'N':
            selected_country = input(" Enter country code (e.g. NL, PL, FR, PT): ").strip().upper()
        elif choice.isdigit() and 1 <= int(choice) <= len(tracks):
            selected_track = tracks[int(choice) - 1]
        else:
            selected_track = tracks[0]
    else:
        val = input(" Enter track (e.g. NL/orbit_37) or country (e.g. NL): ").strip()
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
  [2] CreoDIAS Cloud-Optimized GeoTIFF (/eodata/.../IW_GRDH_1S-COG)
  [3] CreoDIAS Standard SAFE (/eodata/.../IW_GRDH_1S)
  [4] Copernicus Data Space Ecosystem API (CDSE download)
============================================================""")
    src_choice = input(" Enter choice [1-4] (default: 1): ").strip()
    sources = {'1': 'auto', '2': 'cog', '3': 'creodias', '4': 'cdse'}
    source = sources.get(src_choice, 'auto')

    # Step 3: Select Processing Engine
    print("""
============================================================
 Select SAR Processing Engine:
  [1] Pure Python Engine (No ESA SNAP required - Recommended)
  [2] ESA SNAP GPT Engine (Requires local gpt binary)
============================================================""")
    eng_choice = input(" Enter choice [1-2] (default: 1): ").strip()
    engine = 'snap' if eng_choice == '2' else 'python'

    # Step 4: Date Range & Winter Exclusion
    print("\n--- Mandatory Acquisition Dates Configuration ---")
    while True:
        start_date = input(" Enter acquisition start date [YYYY-MM-DD] (e.g. 2024-10-15): ").strip()
        try:
            datetime.datetime.strptime(start_date, "%Y-%m-%d")
            break
        except ValueError:
            print(" [ERROR] Invalid date format! You must provide a valid date formatted as YYYY-MM-DD.")

    while True:
        end_date = input(" Enter acquisition end date [YYYY-MM-DD] (e.g. 2025-09-15): ").strip()
        try:
            datetime.datetime.strptime(end_date, "%Y-%m-%d")
            break
        except ValueError:
            print(" [ERROR] Invalid date format! You must provide a valid date formatted as YYYY-MM-DD.")

    winter_ans = input(" Exclude winter dormancy period 01.12-14.02? [Y/n] (default: Y): ").strip().upper()
    exclude_winter = (winter_ans != 'N')

    pipeline = Sentinel1Pipeline(
        country=selected_country,
        orbit=selected_orbit,
        start_date=start_date,
        end_date=end_date,
        exclude_winter=exclude_winter,
        source=source,
        engine=engine,
    )
    interactive_menu(pipeline)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Sentinel-1 SAR Preprocessing Pipeline (Pure Python & ESA SNAP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive wizard (zero arguments):
  python run_s1_preprocessor.py

  # Pure Python single orbit run from CreoDIAS repository:
  python run_s1_preprocessor.py --track PT/orbit_52 --source creodias --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A

  # Pure Python country-wide run downloading from Copernicus CDSE API:
  python run_s1_preprocessor.py --country PT --source cdse --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A --threads 4
"""
    )
    parser.add_argument('-t', '--track', default=None, help="Track identifier (e.g. NL/orbit_37, PL/orbit_29, PT/orbit_52)")
    parser.add_argument('-c', '--country', default=None, help="Country code (e.g. NL, PL, FR, PT, ES, DE)")
    parser.add_argument('-o', '--orbit', type=int, default=None, help="Specific relative orbit number")
    parser.add_argument('--stage', default=None, choices=['A', '1', '2', '3'], help="Stage to execute: 'A' (all), '1' (calib), '2' (coreg), '3' (stack)")
    parser.add_argument('--source', default='auto', choices=['auto', 'cog', 'creodias', 'cdse'], help="Data source: 'auto' (detect local), 'cog', 'creodias', 'cdse' (default: auto)")
    parser.add_argument('--engine', default='python', choices=['python', 'snap', 'auto'], help="SAR processing engine: 'python' (pure Python, default), 'snap' (ESA SNAP GPT), or 'auto'")
    parser.add_argument('--filter_window', type=int, default=7, help="Multi-temporal Quegan speckle filter window size for Python engine (default: 7)")
    parser.add_argument('-s', '--start_date', default=None, help="Acquisition start date (YYYY-MM-DD, required for Stage 1/A)")
    parser.add_argument('-e', '--end_date', default=None, help="Acquisition end date (YYYY-MM-DD, required for Stage 1/A)")
    parser.add_argument('--exclude_winter', action='store_true', help="Exclude winter dormancy period (December 1 to February 14) from processing (default: False)")
    parser.add_argument('--threads', type=int, default=4, help="Worker threads for parallel processing (default: 4)")
    parser.add_argument('--overwrite', action='store_true', help="Force re-processing of intermediate/final products")

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

    # Validate mandatory dates for Stage 1 and Stage A
    if args.stage in ['A', '1'] and (not args.start_date or not args.end_date):
        parser.error("Start date (-s/--start_date) and end date (-e/--end_date) in format YYYY-MM-DD are required when running Stage 1 or All stages.")

    pipeline = Sentinel1Pipeline(
        country=country,
        orbit=orbit,
        start_date=args.start_date,
        end_date=args.end_date,
        exclude_winter=args.exclude_winter,
        source=args.source,
        engine=args.engine,
        filter_window=args.filter_window,
        threads=args.threads,
        overwrite=args.overwrite
    )

    if args.stage is None:
        interactive_menu(pipeline)
    else:
        stage_choice = args.stage.strip().upper()
        if stage_choice == 'A':
            pipeline.run_all()
        elif stage_choice == '1':
            pipeline.stage_1_calibration()
        elif stage_choice == '2':
            pipeline.stage_2_coregistration()
        elif stage_choice == '3':
            pipeline.stage_3_stack_clip()


if __name__ == '__main__':
    main()
