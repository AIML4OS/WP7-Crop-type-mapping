#!/usr/bin/env python
"""
sentinel2_preprocessor.py - Master Pipeline Orchestrator for Sentinel-2 Data
supporting automatic detection of Sentinel-1 orbits for a given country,
sequential end-to-end processing per orbit, and exact alignment with Sentinel-1 SAR stacks.

Execution examples:
  # 1. Automatic discovery and sequential end-to-end processing of all S1 orbits for country:
  python sentinel2_preprocessor.py -s 2024-10-15 -e 2025-09-15 -c NL --source creodias --mode all
  python sentinel2_preprocessor.py -s 2024-10-15 -e 2025-09-15 -c PL --source creodias --mode all

  # 2. Sequential processing using Copernicus CDSE download:
  python sentinel2_preprocessor.py -s 2024-10-15 -e 2025-09-15 -c NL --source cdse --mode all

  # 3. Single orbit override:
  python sentinel2_preprocessor.py -s 2024-10-15 -e 2025-09-15 -c NL -o 88 --source creodias --mode all

  # 4. Only process existing converted TIFs:
  python sentinel2_preprocessor.py -c NL --mode process
"""

import argparse
import datetime
import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Add directory to sys.path for local imports
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import importlib
extract_creodias = importlib.import_module("1a_extract_creodias_s2")
download_cdse = importlib.import_module("1b_download_cdse_s2")
time_series = importlib.import_module("2_time_series_s2")
mosaic_stack = importlib.import_module("3_mosaic_stack_clip_s2")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def run_s2_pipeline_for_orbit(
    country_code: str,
    orbit_num: int,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    source: str = "creodias",
    mode: str = "all",
    cloud_cover: float = 80.0,
    doys: list = time_series.DEFAULT_DOYS,
    threads: int = 4,
    all_scenes_cache: Optional[list] = None
):
    track_name = f"{country_code}/orbit_{orbit_num}"
    logging.info(f"\n################################################################################")
    logging.info(f" >>> STARTING S2 PIPELINE FOR TRACK: {track_name} (Orbit {orbit_num}) <<<")
    logging.info(f"################################################################################")

    # Step 1: Ingestion / Extraction / Download
    if mode in ['all', 'download', 'download_only']:
        if not start_date or not end_date:
            raise ValueError("Start date (-s) and end date (-e) must be specified for download mode.")

        if source.lower() == 'creodias':
            logging.info(f"\n--- [Step 1a] Ingesting & converting from CREODIAS (Y:) for {track_name} ---")
            extract_creodias.process_orbit_creodias_s2(
                country_code=country_code,
                orbit_num=orbit_num,
                start_date=start_date,
                end_date=end_date,
                all_scenes=all_scenes_cache,
                max_cloud_cover=cloud_cover,
                max_workers=threads
            )
        elif source.lower() == 'cdse':
            logging.info(f"\n--- [Step 1b] Downloading & converting from CDSE API for {track_name} ---")
            download_cdse.process_orbit_cdse_s2(
                country_code=country_code,
                orbit_num=orbit_num,
                start_date=start_date,
                end_date=end_date,
                cloud_cover=cloud_cover,
                max_workers=threads
            )

    if mode in ['download', 'download_only']:
        logging.info(f"Download & conversion complete for track {track_name}.")
        return

    # Step 2: Synthetic Time-series Interpolation
    if mode in ['all', 'process', 'process_only']:
        logging.info(f"\n--- [Step 2] Pure Python Synthetic Time-Series Generation (DOY interpolation) for {track_name} ---")
        time_series.run_time_series_for_track(
            track=track_name,
            doys=doys,
            max_workers=threads
        )

        # Step 3: Mosaic, Reproject, Clip, and Stack
        logging.info(f"\n--- [Step 3] Mosaicking, Reprojecting, NUTS Clipping, and Multi-Band Stacking for {track_name} ---")
        mosaic_stack.mosaic_stack_clip_single_track(
            track=track_name,
            country_code=country_code,
            doys=doys,
            max_workers=threads
        )

    logging.info(f" >>> FINISHED S2 PIPELINE FOR TRACK: {track_name} <<<\n")


def run_s2_master_pipeline(
    country: str,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    orbit: Optional[int] = None,
    track: Optional[str] = None,
    source: str = "creodias",
    mode: str = "all",
    cloud_cover: float = 80.0,
    doys: list = time_series.DEFAULT_DOYS,
    threads: int = 4,
    repo_path: Optional[str] = None
):
    country_code = country.upper() if country else (track.split('/')[0].upper() if track else "PL")

    if repo_path:
        extract_creodias.S2_REPO_PATH = Path(repo_path)

    # 1. Determine target orbits
    if orbit is not None:
        target_orbits = [orbit]
        logging.info(f"Processing manually specified orbit: {target_orbits}")
    elif track:
        match = re.search(r'orbit_(\d+)', track)
        target_orbits = [int(match.group(1))] if match else [88]
        logging.info(f"Processing manually specified track: {track} (Orbit {target_orbits[0]})")
    else:
        target_orbits = extract_creodias.discover_s1_orbits(country_code)

    logging.info(f"================================================================================")
    logging.info(f" Sentinel-2 Pipeline Orchestrator")
    logging.info(f" Country: {country_code} | Target Orbits: {target_orbits}")
    logging.info(f" Source: {source.upper()} | Mode: {mode.upper()}")
    if start_date and end_date:
        logging.info(f" Date range: {start_date} to {end_date}")
    logging.info(f"================================================================================")

    # Cache CREODIAS scan if using CREODIAS source
    all_scenes_cache = None
    if source.lower() == 'creodias' and mode in ['all', 'download', 'download_only'] and start_date and end_date:
        all_scenes_cache = extract_creodias.scan_creodias_for_dates(extract_creodias.S2_REPO_PATH, start_date, end_date, cloud_cover)

    # Sequential processing per orbit
    for o_num in target_orbits:
        run_s2_pipeline_for_orbit(
            country_code=country_code,
            orbit_num=o_num,
            start_date=start_date,
            end_date=end_date,
            source=source,
            mode=mode,
            cloud_cover=cloud_cover,
            doys=doys,
            threads=threads,
            all_scenes_cache=all_scenes_cache
        )

    logging.info(f"================================================================================")
    logging.info(f" ALL Sentinel-2 Pipelines for country {country_code} (Orbits: {target_orbits}) COMPLETED SUCCESSFULLY!")
    logging.info(f"================================================================================")


def main():
    parser = argparse.ArgumentParser(description="Master Sentinel-2 Pipeline Orchestrator automatically detecting S1 orbits.")
    parser.add_argument('-c', '--country', required=True, help="Country code, e.g. PL, NL, FR, PT, AT")
    parser.add_argument('-s', '--start_date', default=None, help="Start date (YYYY-MM-DD), e.g. 2024-10-15")
    parser.add_argument('-e', '--end_date', default=None, help="End date (YYYY-MM-DD), e.g. 2025-09-15")
    parser.add_argument('-o', '--orbit', type=int, default=None, help="Optional single orbit number override (e.g. 88, 161)")
    parser.add_argument('-t', '--track', default=None, help="Optional track relative path (e.g. NL/orbit_88)")
    parser.add_argument('--source', choices=['creodias', 'cdse'], default='creodias', help="Data source: 'creodias' (Y: drive) or 'cdse' (Copernicus API)")
    parser.add_argument('-m', '--mode', choices=['all', 'download', 'process', 'download_only', 'process_only'], default='all', help="Execution mode")
    parser.add_argument('--cloud_cover', type=float, default=80.0, help="Maximum cloud cover (0-100, default: 80)")
    parser.add_argument('--doys', nargs='+', type=int, default=time_series.DEFAULT_DOYS, help="List of target DOY integers")
    parser.add_argument('--threads', type=int, default=8, help="Worker threads (default: 8)")
    parser.add_argument('--repo_path', type=str, default=None, help="Override CREODIAS repository path")
    parser.add_argument('--username', default=download_cdse.CDSE_USERNAME, help="CDSE Username")
    parser.add_argument('--password', default=download_cdse.CDSE_PASSWORD, help="CDSE Password")

    args = parser.parse_args()

    start_dt = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_dt = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    if args.username:
        download_cdse.CDSE_USERNAME = args.username
    if args.password:
        download_cdse.CDSE_PASSWORD = args.password

    run_s2_master_pipeline(
        country=args.country,
        start_date=start_dt,
        end_date=end_dt,
        orbit=args.orbit,
        track=args.track,
        source=args.source,
        mode=args.mode,
        cloud_cover=args.cloud_cover,
        doys=args.doys,
        threads=args.threads,
        repo_path=args.repo_path
    )


if __name__ == '__main__':
    main()
