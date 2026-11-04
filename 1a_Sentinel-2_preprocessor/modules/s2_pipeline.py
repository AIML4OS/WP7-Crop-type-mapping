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
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Add directory to sys.path for local imports
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import importlib
extract_creodias = importlib.import_module("s2_extract_creodias")
download_cdse = importlib.import_module("s2_download_cdse")
time_series = importlib.import_module("s2_time_series")
mosaic_stack = importlib.import_module("s2_mosaic_stack")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


import shutil


def try_reuse_existing_country_s2_stack(track_name: str, country_code: str, overwrite: bool = False) -> bool:
    """
    Checks if a valid orbit-clipped multi-temporal S2 stack for this specific track already exists.
    Since each orbit has its own distinct S1 swath footprint, stacks from different orbits are never reused across orbits.
    """
    from osgeo import gdal
    norm_track = track_name.replace('\\', '/')
    track_dir = mosaic_stack.BASE_DIR / norm_track
    dest_proc_dir = track_dir / "1_input_stacks"
    if not dest_proc_dir.exists() or overwrite:
        return False

    for cand in sorted(dest_proc_dir.glob("*_S2_timeseries.tif")):
        if cand.exists() and cand.stat().st_size > 100 * 1024 * 1024 and not cand.name.endswith(".tmp.tif"):
            try:
                ds_check = gdal.Open(str(cand))
                if ds_check and ds_check.RasterCount >= 126 and ds_check.GetMetadataItem("ORBIT_CLIPPED") == "1":
                    ds_check = None
                    logging.info(f"Target orbit-clipped Sentinel-2 stack {cand.name} already exists and is complete ({cand.stat().st_size / (1024**3):.1f} GB). Skipping S2 pipeline!")
                    return True
                ds_check = None
            except Exception:
                pass
    return False


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
    all_scenes_cache: Optional[list] = None,
    overwrite: bool = False,
    enable_whittaker: bool = True,
    whittaker_lambda: float = 15.0
):
    track_name = f"{country_code}/orbit_{orbit_num}"
    logging.info(f"\n################################################################################")
    logging.info(f" >>> STARTING S2 PIPELINE FOR TRACK: {track_name} (Orbit {orbit_num}) <<<")
    logging.info(f"################################################################################")

    # Step 0: Check if a matching country-wide S2 stack already exists in another orbit / repository
    if not overwrite:
        if try_reuse_existing_country_s2_stack(track_name, country_code, overwrite=overwrite):
            logging.info(f" >>> FINISHED S2 PIPELINE FOR TRACK: {track_name} (Reused Country S2 Stack) <<<\n")
            return

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
            max_workers=threads,
            overwrite=overwrite,
            enable_whittaker=enable_whittaker,
            whittaker_lambda=whittaker_lambda
        )

        # Step 3: Mosaic, Reproject, Clip, and Stack
        logging.info(f"\n--- [Step 3] Mosaicking, Reprojecting, NUTS Clipping, and Multi-Band Stacking for {track_name} ---")
        mosaic_stack.mosaic_stack_clip_single_track(
            track=track_name,
            country_code=country_code,
            doys=doys,
            max_workers=threads,
            overwrite=overwrite
        )

    logging.info(f" >>> FINISHED S2 PIPELINE FOR TRACK: {track_name} <<<\n")


def run_s2_pipeline_for_track(
    track_name: str,
    country_code: str,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    source: str = "creodias",
    mode: str = "all",
    cloud_cover: float = 80.0,
    doys: list = time_series.DEFAULT_DOYS,
    threads: int = 4,
    all_scenes_cache: Optional[list] = None,
    overwrite: bool = False,
    enable_whittaker: bool = True,
    whittaker_lambda: float = 15.0
):
    orbit_match = re.search(r'orbit_(\d+)', track_name)
    orbit_num = int(orbit_match.group(1)) if orbit_match else 0
    return run_s2_pipeline_for_orbit(
        country_code=country_code,
        orbit_num=orbit_num,
        start_date=start_date,
        end_date=end_date,
        source=source,
        mode=mode,
        cloud_cover=cloud_cover,
        doys=doys,
        threads=threads,
        all_scenes_cache=all_scenes_cache,
        overwrite=overwrite,
        enable_whittaker=enable_whittaker,
        whittaker_lambda=whittaker_lambda
    )


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
    threads: int = 8,
    repo_path: Optional[str] = None,
    overwrite: bool = False,
    enable_whittaker: bool = True,
    whittaker_lambda: float = 15.0
):
    country_code = country.upper() if country else (track.split('/')[0].upper() if track else "PL")

    if repo_path:
        extract_creodias.S2_REPO_PATH = Path(repo_path)

    if track:
        run_s2_pipeline_for_track(
            track_name=track,
            country_code=country_code,
            start_date=start_date,
            end_date=end_date,
            source=source,
            mode=mode,
            cloud_cover=cloud_cover,
            doys=doys,
            threads=threads,
            overwrite=overwrite,
            enable_whittaker=enable_whittaker,
            whittaker_lambda=whittaker_lambda
        )
    else:
        if orbit is not None:
            track_name = f"{country_code}/orbit_{orbit}"
            run_s2_pipeline_for_track(
                track_name=track_name,
                country_code=country_code,
                start_date=start_date,
                end_date=end_date,
                source=source,
                mode=mode,
                cloud_cover=cloud_cover,
                doys=doys,
                threads=threads,
                overwrite=overwrite,
                enable_whittaker=enable_whittaker,
                whittaker_lambda=whittaker_lambda
            )
        else:
            orbits = extract_creodias.discover_s1_orbits(country_code)
            logging.info(f"\n#####################################################################")
            logging.info(f"   STARTING PER-ORBIT SENTINEL-2 PIPELINE FOR COUNTRY: {country_code} (Orbits: {orbits})")
            logging.info(f"   SOURCE: {source.upper()} | MODE: {mode.upper()} | THREADS: {threads}")
            logging.info(f"#####################################################################\n")

            if orbits:
                for orb in orbits:
                    run_s2_pipeline_for_orbit(
                        country_code=country_code,
                        orbit_num=orb,
                        start_date=start_date,
                        end_date=end_date,
                        source=source,
                        mode=mode,
                        cloud_cover=cloud_cover,
                        doys=doys,
                        threads=threads,
                        overwrite=overwrite,
                        enable_whittaker=enable_whittaker,
                        whittaker_lambda=whittaker_lambda
                    )
            else:
                # Fallback if no orbits configured
                if mode in ['all', 'download', 'download_only']:
                    if not start_date or not end_date:
                        raise ValueError("Start date (-s) and end date (-e) must be specified for download mode.")
                    if source.lower() == 'creodias':
                        extract_creodias.process_country_creodias_s2(
                            country_code=country_code,
                            start_date=start_date,
                            end_date=end_date,
                            max_cloud_cover=cloud_cover,
                            max_workers=threads
                        )
                    else:
                        download_cdse.process_country_cdse_s2(
                            country_code=country_code,
                            start_date=start_date,
                            end_date=end_date,
                            cloud_cover=cloud_cover,
                            max_workers=threads
                        )
                if mode in ['all', 'process', 'process_only']:
                    time_series.run_time_series(
                        country=country_code,
                        doys=doys,
                        max_workers=threads,
                        overwrite=overwrite,
                        enable_whittaker=enable_whittaker,
                        whittaker_lambda=whittaker_lambda
                    )
                    mosaic_stack.mosaic_stack_clip_s2(
                        country=country_code,
                        doys=doys,
                        max_workers=threads,
                        overwrite=overwrite
                    )

        logging.info(f"\n=====================================================================")
        logging.info(f" Sentinel-2 Pipeline for country {country_code} COMPLETED SUCCESSFULLY!")
        logging.info(f"=====================================================================\n")


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
    parser.add_argument('--overwrite', action='store_true', help="Force overwrite of existing mosaics/stacks")
    parser.add_argument('--enable_whittaker', dest='whittaker', action='store_true', default=True, help="Enable Whittaker smoothing across DOYs (default: True)")
    parser.add_argument('--no_whittaker', dest='whittaker', action='store_false', help="Disable Whittaker smoothing")
    parser.add_argument('--whittaker_lambda', type=float, default=15.0, help="Smoothing penalty lambda for Whittaker (default: 15.0)")
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
        repo_path=args.repo_path,
        overwrite=args.overwrite,
        enable_whittaker=args.whittaker,
        whittaker_lambda=args.whittaker_lambda
    )


if __name__ == '__main__':
    main()
