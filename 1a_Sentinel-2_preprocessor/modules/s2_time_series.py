#!/usr/bin/env python
"""
2_time_series_s2.py - Pure Python Sentinel-2 Synthetic Time-Series Generation.

Supports running by Country Code (-c) across all greedy orbits or by track (-t).
Interpolates for 9 spectral bands: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12'] across target DOYs.

Usage examples:
  python 2_time_series_s2.py -c PL
  python 2_time_series_s2.py -c NL -o 88
  python 2_time_series_s2.py -t NL/orbit_88
"""

import argparse
import datetime
import glob
import logging
import os
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from osgeo import gdal

# ================= CONFIGURATION =================
BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDirs"))
DEFAULT_DOYS = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]
S2_SPECTRAL_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

COUNTRY_ORBITS = {
    'NL': [15, 37, 88, 110, 139, 161],
    'PL': [22, 29, 73, 95, 102, 124, 146, 168, 175],
    'IE': [30, 74, 103, 132, 147],
    'FR': [8, 30, 37, 59, 81, 88, 103, 110, 132, 139, 153, 161],
    'AT': [22, 29, 73, 95, 102, 124, 146, 168],
    'PT': [81, 153, 161, 8, 88, 110],
    'DE': [22, 29, 73, 95, 102, 124, 146, 168, 175],
    'ES': [8, 81, 88, 153, 161],
    'IT': [44, 117, 146, 168]
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def extract_date_and_doy_from_filepath(filepath: Path) -> Tuple[datetime.date, int]:
    fname = filepath.name
    match = re.search(r'20\d{6}', fname)
    if match:
        dt = datetime.datetime.strptime(match.group(0), "%Y%m%d").date()
        return dt, dt.timetuple().tm_yday

    parent_name = filepath.parent.name
    match = re.search(r'20\d{6}', parent_name)
    if match:
        dt = datetime.datetime.strptime(match.group(0), "%Y%m%d").date()
        return dt, dt.timetuple().tm_yday

    mtime = datetime.datetime.fromtimestamp(filepath.stat().st_mtime).date()
    return mtime, mtime.timetuple().tm_yday


def generate_s2_time_series_for_tile(
    tile_name: str,
    tile_tif_dir: Path,
    result_synthetic_dir: Path,
    doys: List[int]
) -> bool:
    result_synthetic_dir.mkdir(parents=True, exist_ok=True)
    clean_tile = tile_name.upper().replace('T', '')

    b02_paths = list(tile_tif_dir.glob("**/*_B02*.tif"))
    if not b02_paths:
        return False

    acquisitions = []
    for b02_p in b02_paths:
        dt, doy = extract_date_and_doy_from_filepath(b02_p)
        acquisitions.append({
            'date': dt,
            'doy': doy,
            'b02_path': b02_p
        })

    acquisitions.sort(key=lambda x: x['doy'])
    year = acquisitions[0]['date'].year

    for target_doy in doys:
        day_folder = result_synthetic_dir / f"day{target_doy}_{year}"
        day_folder.mkdir(parents=True, exist_ok=True)

        d1_acq = None
        d2_acq = None
        for acq in acquisitions:
            if acq['doy'] <= target_doy:
                d1_acq = acq
            if acq['doy'] >= target_doy and d2_acq is None:
                d2_acq = acq

        if d1_acq is None: d1_acq = acquisitions[0]
        if d2_acq is None: d2_acq = acquisitions[-1]

        d1 = d1_acq['doy']
        d2 = d2_acq['doy']
        weight = 0.0 if d1 == d2 else float(np.clip((target_doy - d1) / float(d2 - d1), 0.0, 1.0))

        for band in S2_SPECTRAL_BANDS:
            out_band_file = day_folder / f"{band}.tif"
            if out_band_file.exists() and out_band_file.stat().st_size > 1024:
                continue

            path1_str = str(d1_acq['b02_path']).replace('B02', band)
            path2_str = str(d2_acq['b02_path']).replace('B02', band)

            p1 = Path(path1_str) if Path(path1_str).exists() else d1_acq['b02_path']
            p2 = Path(path2_str) if Path(path2_str).exists() else d2_acq['b02_path']

            ds1 = gdal.Open(str(p1))
            ds2 = gdal.Open(str(p2))
            if ds1 is None or ds2 is None:
                continue

            arr1 = ds1.GetRasterBand(1).ReadAsArray().astype(np.float32)
            arr2 = ds2.GetRasterBand(1).ReadAsArray().astype(np.float32)

            interp_arr = np.clip((1.0 - weight) * arr1 + weight * arr2, 0, 65535).astype(np.uint16)

            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                str(out_band_file), ds1.RasterXSize, ds1.RasterYSize, 1, gdal.GDT_UInt16,
                options=['COMPRESS=LZW', 'TILED=YES']
            )
            out_ds.SetGeoTransform(ds1.GetGeoTransform())
            out_ds.SetProjection(ds1.GetProjection())
            out_ds.GetRasterBand(1).WriteArray(interp_arr)
            out_ds.GetRasterBand(1).SetNoDataValue(0)
            out_ds = None
            ds1 = None
            ds2 = None

    return True


def run_time_series_for_track(track: str, doys: List[int], max_workers: int = 8):
    import threading
    s2_base = BASE_DIR / track / "S2"
    if not s2_base.exists():
        return

    tile_dirs = [d for d in s2_base.iterdir() if d.is_dir() and d.name.endswith("_tif")]
    if not tile_dirs:
        return

    total_tiles = len(tile_dirs)
    done_tiles = 0
    lock = threading.Lock()

    logging.info(f"Generating synthetic time-series for track {track} ({total_tiles} tiles, Workers: {max_workers})...")

    def _worker_tile(t_dir):
        nonlocal done_tiles
        clean_tile_name = t_dir.name.replace('_tif', '')
        res = generate_s2_time_series_for_tile(clean_tile_name, t_dir, s2_base / clean_tile_name / "_synthetic_s2", doys)
        with lock:
            done_tiles += 1
            pct = (done_tiles / total_tiles) * 100.0
            logging.info(f"  [TIME-SERIES PROGRESS] Track {track}: {done_tiles}/{total_tiles} tiles completed ({pct:.1f}%) - Last: {clean_tile_name}")
        return res

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_worker_tile, tile_dirs))


def run_time_series(
    country: Optional[str] = None,
    track: Optional[str] = None,
    orbit: Optional[int] = None,
    doys: List[int] = DEFAULT_DOYS,
    max_workers: int = 4
):
    if track:
        run_time_series_for_track(track, doys, max_workers)
    elif country:
        country_code = country.upper()
        country_dir = BASE_DIR / country_code
        if orbit is not None:
            orbits = [orbit]
        elif country_dir.exists():
            orbits = [int(re.search(r'orbit_(\d+)', d.name).group(1)) for d in country_dir.glob("orbit_*") if re.search(r'orbit_(\d+)', d.name)]
            if not orbits:
                orbits = COUNTRY_ORBITS.get(country_code, [88, 161])
        else:
            orbits = COUNTRY_ORBITS.get(country_code, [88, 161])

        logging.info(f"Generating synthetic time-series for country {country_code} across orbits {orbits}...")
        for o in orbits:
            run_time_series_for_track(f"{country_code}/orbit_{o}", doys, max_workers)


def main():
    parser = argparse.ArgumentParser(description="Generate Sentinel-2 synthetic multi-temporal time series for country/track.")
    parser.add_argument('-c', '--country', default=None, help="Country code, e.g. PL, NL, FR, PT")
    parser.add_argument('-t', '--track', default=None, help="Track/Orbit relative path, e.g. NL/orbit_88")
    parser.add_argument('-o', '--orbit', type=int, default=None, help="Specify single orbit")
    parser.add_argument('--doys', nargs='+', type=int, default=DEFAULT_DOYS, help="DOY targets")
    parser.add_argument('--threads', type=int, default=4, help="Worker threads (default: 4)")

    args = parser.parse_args()

    run_time_series(
        country=args.country,
        track=args.track,
        orbit=args.orbit,
        doys=args.doys,
        max_workers=args.threads
    )


if __name__ == '__main__':
    main()
