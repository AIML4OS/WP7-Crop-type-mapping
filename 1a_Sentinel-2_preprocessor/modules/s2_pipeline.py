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
    Checks if a valid multi-temporal S2 stack for this country already exists
    in another orbit or shared repository. If found, instantly links or aligns it,
    saving ~70 hours of repetitive downloading and mosaicking.
    """
    from osgeo import gdal
    norm_track = track_name.replace('\\', '/')
    sanitized_track = norm_track.replace('/', '_')
    track_dir = mosaic_stack.BASE_DIR / norm_track
    dest_proc_dir = track_dir / "1_input_stacks"
    dest_s2_stack = dest_proc_dir / f"{sanitized_track}_S2_timeseries.tif"

    if dest_s2_stack.exists() and dest_s2_stack.stat().st_size > 100 * 1024 * 1024 and not overwrite:
        try:
            ds_check = gdal.Open(str(dest_s2_stack))
            if ds_check and ds_check.RasterCount >= 126:
                logging.info(f"Target Sentinel-2 stack {dest_s2_stack.name} already exists and is complete ({dest_s2_stack.stat().st_size / (1024**3):.1f} GB). Skipping S2 pipeline!")
                return True
        except Exception:
            pass

    s1_ref = mosaic_stack.get_s1_raster_reference(track_dir)
    if not s1_ref:
        return False

    target_w = s1_ref['width']
    target_h = s1_ref['height']
    target_gt = s1_ref['gt']
    target_bounds = s1_ref['bounds']
    target_proj = s1_ref['proj']

    # Search for candidates across the country
    candidate_paths = [
        mosaic_stack.BASE_DIR / country_code.upper() / "S2" / f"{country_code.upper()}_S2_timeseries.tif",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / country_code.upper() / "S2" / f"{country_code.upper()}_S2_timeseries.tif",
    ]
    for c_dir in [mosaic_stack.BASE_DIR / country_code.upper(), Path(r"D:/AIML_CropMapper_Cloud/workingDir") / country_code.upper()]:
        if c_dir.exists():
            for o_stack in c_dir.glob("orbit_*/1_input_stacks/*S2*.tif"):
                if o_stack.resolve() != dest_s2_stack.resolve() and not o_stack.name.endswith(".tmp.tif"):
                    candidate_paths.append(o_stack)

    for cand in candidate_paths:
        if cand.exists() and cand.stat().st_size > 100 * 1024 * 1024:
            try:
                ds_cand = gdal.Open(str(cand))
                if not ds_cand or ds_cand.RasterCount < 126:
                    continue

                cand_w = ds_cand.RasterXSize
                cand_h = ds_cand.RasterYSize
                cand_gt = ds_cand.GetGeoTransform()

                # Case A: Exact or near-exact match (within 10 meters / 1 pixel shift)
                if cand_w == target_w and cand_h == target_h and abs(cand_gt[0] - target_gt[0]) < 10.0 and abs(cand_gt[3] - target_gt[3]) < 10.0:
                    dest_proc_dir.mkdir(parents=True, exist_ok=True)
                    if dest_s2_stack.exists():
                        try: dest_s2_stack.unlink()
                        except: pass
                    try:
                        os.link(str(cand), str(dest_s2_stack))
                        logging.info(f" >>> [OPTIMIZATION] Created instant hardlink from {cand.name} to {dest_s2_stack.name} (0 bytes duplicated) <<<")
                    except Exception:
                        shutil.copy2(str(cand), str(dest_s2_stack))
                        logging.info(f" >>> [OPTIMIZATION] Copied {cand.name} to {dest_s2_stack.name} <<<")

                    # Also link overviews if present
                    cand_ovr = cand.parent / f"{cand.name}.ovr"
                    if cand_ovr.exists():
                        dest_ovr = dest_proc_dir / f"{dest_s2_stack.name}.ovr"
                        try:
                            if dest_ovr.exists(): dest_ovr.unlink()
                            os.link(str(cand_ovr), str(dest_ovr))
                        except: pass

                    logging.info(f" >>> [OPTIMIZATION] Reused existing full-country S2 stack ({cand.stat().st_size / (1024**3):.1f} GB) for {track_name}! (Saved ~70h processing time) <<<\n")
                    return True

                # Case B: Slight bounding box difference (e.g. 1-2 rows diff) - Warp existing stack in ~1-2 minutes instead of 70 hours
                elif abs(cand_w - target_w) <= 10 and abs(cand_h - target_h) <= 10:
                    dest_proc_dir.mkdir(parents=True, exist_ok=True)
                    logging.info(f" >>> [OPTIMIZATION] Found country S2 stack {cand.name} with minor boundary shift ({cand_w}x{cand_h} vs target {target_w}x{target_h}). Fast-warping in ~1-2 mins instead of 70 hours...")
                    warp_opts = gdal.WarpOptions(
                        outputBounds=target_bounds,
                        width=target_w,
                        height=target_h,
                        dstSRS=target_proj,
                        resampleAlg='bilinear',
                        creationOptions=['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=6', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'],
                        callback=gdal.TermProgress_nocb
                    )
                    ds_warp = gdal.Warp(str(dest_s2_stack), str(cand), options=warp_opts)
                    if ds_warp:
                        ds_warp.FlushCache()
                        ds_warp = None
                        logging.info(f" >>> [OPTIMIZATION] Fast-warp complete for {dest_s2_stack.name}! Building overviews...")
                        ds_out = gdal.Open(str(dest_s2_stack), gdal.GA_Update)
                        if ds_out:
                            gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
                            gdal.SetConfigOption('PREDICTOR_OVERVIEW', '2')
                            gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
                            ds_out.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32, 64])
                            ds_out = None
                        logging.info(f" >>> [OPTIMIZATION] Successfully adapted existing country S2 stack for {track_name}! (Saved ~70 hours!) <<<\n")
                        return True
            except Exception as e:
                logging.warning(f"Error checking S2 candidate {cand}: {e}")
                continue

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
    overwrite: bool = False
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
            max_workers=threads
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
    overwrite: bool = False
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
        overwrite=overwrite
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
    overwrite: bool = False
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
            overwrite=overwrite
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
                overwrite=overwrite
            )
        else:
            logging.info(f"\n#####################################################################")
            logging.info(f"   STARTING UNIFIED SENTINEL-2 PIPELINE FOR COUNTRY: {country_code}")
            logging.info(f"   SOURCE: {source.upper()} | MODE: {mode.upper()} | THREADS: {threads}")
            logging.info(f"#####################################################################\n")

            # Step 1: Ingestion
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

            if mode in ['download', 'download_only']:
                logging.info(f"Country download & conversion complete for {country_code}.")
                return

            # Step 2: Synthetic Time-series Interpolation
            if mode in ['all', 'process', 'process_only']:
                time_series.run_time_series(
                    country=country_code,
                    doys=doys,
                    max_workers=threads,
                    overwrite=overwrite
                )

                # Step 3: Mosaic & Stack into Country Master BigTIFF
                mosaic_stack.mosaic_stack_clip_single_track(
                    track=country_code,
                    country_code=country_code,
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
        overwrite=args.overwrite
    )


if __name__ == '__main__':
    main()
