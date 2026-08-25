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
    doys: List[int],
    overwrite: bool = False
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

    acquisitions.sort(key=lambda x: (x['date'].year, x['doy']))
    years = [a['date'].year for a in acquisitions]
    year = max(years) if years else datetime.date.today().year

    # Open first band to retrieve raster dimensions and projection
    ds0 = gdal.Open(str(acquisitions[0]['b02_path']))
    if ds0 is None:
        return False
    raster_w = ds0.RasterXSize
    raster_h = ds0.RasterYSize
    geo_trans = ds0.GetGeoTransform()
    proj_wkt = ds0.GetProjection()
    ds0 = None

    for target_doy in doys:
        day_folder = result_synthetic_dir / f"day{target_doy}_{year}"
        day_folder.mkdir(parents=True, exist_ok=True)

        before_cands = [a for a in acquisitions if a['doy'] <= target_doy]
        before_cands.sort(key=lambda x: x['doy'], reverse=True)

        after_cands = [a for a in acquisitions if a['doy'] >= target_doy]
        after_cands.sort(key=lambda x: x['doy'])

        if not before_cands: before_cands = list(after_cands)
        if not after_cands: after_cands = list(before_cands)

        for band in S2_SPECTRAL_BANDS:
            out_band_file = day_folder / f"{band}.tif"
            if not overwrite and out_band_file.exists() and out_band_file.stat().st_size > 1024:
                continue

            # Build seamless composite before target_doy
            arr_before = np.zeros((raster_h, raster_w), dtype=np.float32)
            doy_before = np.zeros((raster_h, raster_w), dtype=np.float32)
            for acq in before_cands[:6]:
                p = Path(str(acq['b02_path']).replace('B02', band))
                if p.exists():
                    ds = gdal.Open(str(p))
                    if ds:
                        arr_i = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
                        ds = None
                        mask_fill = (arr_before == 0) & (arr_i > 0)
                        arr_before[mask_fill] = arr_i[mask_fill]
                        doy_before[mask_fill] = acq['doy']
                        if np.all(arr_before > 0):
                            break

            # Build seamless composite after target_doy
            arr_after = np.zeros((raster_h, raster_w), dtype=np.float32)
            doy_after = np.zeros((raster_h, raster_w), dtype=np.float32)
            for acq in after_cands[:6]:
                p = Path(str(acq['b02_path']).replace('B02', band))
                if p.exists():
                    ds = gdal.Open(str(p))
                    if ds:
                        arr_i = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
                        ds = None
                        mask_fill = (arr_after == 0) & (arr_i > 0)
                        arr_after[mask_fill] = arr_i[mask_fill]
                        doy_after[mask_fill] = acq['doy']
                        if np.all(arr_after > 0):
                            break

            # Seamless pixel-wise interpolation
            interp_arr = np.zeros((raster_h, raster_w), dtype=np.float32)
            both_mask = (arr_before > 0) & (arr_after > 0)
            only_b = (arr_before > 0) & (arr_after == 0)
            only_a = (arr_before == 0) & (arr_after > 0)

            denom = np.maximum(doy_after - doy_before, 1.0)
            weight = np.clip((target_doy - doy_before) / denom, 0.0, 1.0)
            interp_arr[both_mask] = (1.0 - weight[both_mask]) * arr_before[both_mask] + weight[both_mask] * arr_after[both_mask]
            interp_arr[only_b] = arr_before[only_b]
            interp_arr[only_a] = arr_after[only_a]

            # Fallback for any unpopulated pixels: check all remaining acquisitions
            zero_mask = (interp_arr == 0)
            if np.any(zero_mask):
                for acq in acquisitions:
                    p = Path(str(acq['b02_path']).replace('B02', band))
                    if p.exists():
                        ds = gdal.Open(str(p))
                        if ds:
                            arr_i = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
                            ds = None
                            fill_m = zero_mask & (arr_i > 0)
                            interp_arr[fill_m] = arr_i[fill_m]
                            zero_mask = (interp_arr == 0)
                            if not np.any(zero_mask):
                                break

            final_arr = np.clip(interp_arr, 0, 65535).astype(np.uint16)

            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(
                str(out_band_file), raster_w, raster_h, 1, gdal.GDT_UInt16,
                options=['COMPRESS=LZW', 'TILED=YES']
            )
            out_ds.SetGeoTransform(geo_trans)
            out_ds.SetProjection(proj_wkt)
            out_ds.GetRasterBand(1).WriteArray(final_arr)
            out_ds.GetRasterBand(1).SetNoDataValue(0)
            out_ds = None

    return True


def run_time_series_for_track(track: str, doys: List[int], max_workers: int = 8, overwrite: bool = False):
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

    logging.info(f"Generating seamless synthetic time-series for track {track} ({total_tiles} tiles, Workers: {max_workers})...")

    def _worker_tile(t_dir):
        nonlocal done_tiles
        clean_tile_name = t_dir.name.replace('_tif', '')
        res = generate_s2_time_series_for_tile(clean_tile_name, t_dir, s2_base / clean_tile_name / "_synthetic_s2", doys, overwrite=overwrite)
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
