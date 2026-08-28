#!/usr/bin/env python
"""
3_mosaic_stack_clip_s2.py - Mosaic, Reproject, Clip and Stack Sentinel-2 Synthetic Time Series
using Country Code (-c) and Greedy Orbits matching Sentinel-1.

Usage examples:
  python 3_mosaic_stack_clip_s2.py -c PL
  python 3_mosaic_stack_clip_s2.py -c NL
  python 3_mosaic_stack_clip_s2.py -t NL/orbit_88
"""

import argparse
import datetime
import glob
import logging
import os
import pathlib
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from osgeo import gdal, ogr, osr

# ================= CONFIGURATION =================
BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDirs"))
AUX_DIR = Path(os.environ.get("AIML_AUX_DIR", r"D:/AIML_CropMapper_Cloud/auxiliary_files"))
SHAPEFILES_DIR = AUX_DIR / "shapefiles_nuts"
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


def get_country_shapefile(country_code: str) -> Optional[Path]:
    country_dir = SHAPEFILES_DIR / country_code.upper()
    if country_dir.exists():
        shps = list(country_dir.glob("*.shp"))
        if shps:
            return shps[0]
    shps = list(SHAPEFILES_DIR.glob(f"**/*{country_code.upper()}*.shp"))
    return shps[0] if shps else None


def get_s1_raster_reference(track_dir: Path) -> Optional[Dict]:
    candidate_dirs = [
        track_dir / "1_input_stacks",
        track_dir,
        track_dir / "processed_raster",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / track_dir.relative_to(BASE_DIR) / "1_input_stacks" if BASE_DIR in track_dir.parents else None,
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / track_dir.relative_to(BASE_DIR) / "processed_raster" if BASE_DIR in track_dir.parents else None,
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / track_dir.relative_to(BASE_DIR) if BASE_DIR in track_dir.parents else None
    ]
    for proc_dir in candidate_dirs:
        if proc_dir and proc_dir.exists():
            s1_tifs = list(proc_dir.glob("*_VH_VV*.tif")) + list(proc_dir.glob("*Sigma0*.tif")) + list(proc_dir.glob("S1_*_stack*.tif"))
            if s1_tifs:
                ds = gdal.Open(str(s1_tifs[0]))
                if ds:
                    proj = ds.GetProjection()
                    gt = ds.GetGeoTransform()
                    w = ds.RasterXSize
                    h = ds.RasterYSize
                    res_x = abs(gt[1])
                    res_y = abs(gt[5])
                    min_x = gt[0]
                    max_y = gt[3]
                    max_x = min_x + w * gt[1]
                    min_y = max_y + h * gt[5]
                    ds = None
                    return {
                        'proj': proj,
                        'gt': gt,
                        'res_x': res_x,
                        'res_y': res_y,
                        'bounds': (min_x, min_y, max_x, max_y),
                        'width': w,
                        'height': h,
                        'ref_file': s1_tifs[0]
                    }
    return None


def mosaic_single_band_doy(
    band_input_files: List[Path],
    output_tif: Path,
    shp_cutline: Optional[Path] = None,
    target_epsg: int = 3857,
    ref_proj: Optional[str] = None,
    res_x: float = 10.0,
    res_y: float = 10.0,
    output_bounds: Optional[List[float]] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    overwrite: bool = False
) -> bool:
    if output_tif.exists() and output_tif.stat().st_size > 1024:
        if not overwrite:
            return True
        # Smart check: if overwrite is requested, but file already has exact matching target geometry, skip it!
        if target_width and target_height:
            try:
                ds_check = gdal.Open(str(output_tif))
                if ds_check:
                    if ds_check.RasterXSize == target_width and ds_check.RasterYSize == target_height:
                        ds_check = None
                        return True
                    ds_check = None
            except:
                pass

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    existing_files = [str(f) for f in band_input_files if f.exists()]
    if not existing_files:
        return False

    warp_options_kwargs = {
        'format': 'GTiff',
        'srcNodata': 0,
        'dstNodata': 0,
        'multithread': True,
        'warpOptions': ["NUM_THREADS=ALL_CPUS"],
        'resampleAlg': gdal.GRA_Bilinear,
        'creationOptions': ["COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=6", "TILED=YES", "BIGTIFF=YES"],
        'xRes': res_x,
        'yRes': res_y
    }

    if ref_proj:
        warp_options_kwargs['dstSRS'] = ref_proj
    else:
        warp_options_kwargs['dstSRS'] = f"EPSG:{target_epsg}"

    if output_bounds:
        warp_options_kwargs['outputBounds'] = output_bounds

    if shp_cutline and shp_cutline.exists():
        warp_options_kwargs['cutlineDSName'] = str(shp_cutline)
        warp_options_kwargs['cropToCutline'] = (output_bounds is None)

    try:
        warp_opts = gdal.WarpOptions(**warp_options_kwargs)
        gdal.Warp(str(output_tif), existing_files, options=warp_opts)
        return True
    except Exception as e:
        logging.error(f"Error mosaicking {output_tif.name}: {e}")
        return False


def mosaic_stack_clip_single_track(
    track: str,
    country_code: str,
    target_epsg: int = 3857,
    doys: List[int] = DEFAULT_DOYS,
    max_workers: int = 8,
    overwrite: bool = False,
    build_overviews: bool = True
):
    norm_track = track.replace('\\', '/')
    sanitized_track = norm_track.replace('/', '_')
    track_dir = BASE_DIR / track

    candidate_s2_bases = [
        BASE_DIR / country_code.upper() / "S2",
        track_dir / "S2",
        track_dir / "_temp_processing" / "s2_optical",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / country_code.upper() / "S2",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / track / "S2"
    ]
    s2_base = candidate_s2_bases[0]
    for c in candidate_s2_bases:
        if c.exists() and (list(c.glob("**/_synthetic_s2")) or list(c.glob("day*_*"))):
            s2_base = c
            break

    if not s2_base.exists():
        logging.warning(f"Could not locate S2 synthetic repository for {track} (checked {candidate_s2_bases})")
        return

    out_final_dir = track_dir / "s2_doy_mosaics"
    out_proc_dir = track_dir / "1_input_stacks"

    # Backward compatibility: migrate legacy deep directory if present
    old_doy_dir = track_dir / "_temp_processing" / "s2_optical" / "3_doy_mosaics" / "mosaic"
    if old_doy_dir.exists() and not out_final_dir.exists():
        logging.info(f"Moving existing DOY mosaics from {old_doy_dir} to clean path {out_final_dir}...")
        try:
            shutil.move(str(old_doy_dir), str(out_final_dir))
            shutil.rmtree(str(track_dir / "_temp_processing"), ignore_errors=True)
        except Exception as e:
            logging.warning(f"Could not migrate legacy directory: {e}")

    out_final_dir.mkdir(parents=True, exist_ok=True)
    out_proc_dir.mkdir(parents=True, exist_ok=True)

    shp_cutline = get_country_shapefile(country_code)

    ref_proj, res_x, res_y = (None, 10.0, 10.0)
    output_bounds = None
    target_width = None
    target_height = None
    s1_ref = get_s1_raster_reference(track_dir)
    if s1_ref:
        ref_proj = s1_ref['proj']
        res_x = s1_ref['res_x']
        res_y = s1_ref['res_y']
        output_bounds = s1_ref['bounds']
        target_width = s1_ref['width']
        target_height = s1_ref['height']
        logging.info(f"Matching S1 SAR reference geometry for {track}: {target_width}x{target_height} ({res_x}m x {res_y}m), Bounds: {output_bounds}")

    synthetic_dirs = list(s2_base.glob("**/_synthetic_s2"))
    if not synthetic_dirs:
        return

    logging.info(f"Track {track}: mosaicking from {len(synthetic_dirs)} tile sources...")

    year = 2024
    for syn_d in synthetic_dirs:
        day_folders = list(syn_d.glob("day*_*"))
        if day_folders:
            match = re.search(r'day\d+_(\d{4})', day_folders[0].name)
            if match:
                year = int(match.group(1))
                break

    mosaic_tasks = []
    mosaic_bands_list = []

    for doy in doys:
        day_str = f"day{doy}_{year}"
        out_doy_dir = out_final_dir / day_str
        out_doy_dir.mkdir(parents=True, exist_ok=True)

        for band in S2_SPECTRAL_BANDS:
            band_filename = f"{band}.tif"
            out_band_tif = out_doy_dir / band_filename
            input_band_files = [syn_d / day_str / band_filename for syn_d in synthetic_dirs]
            mosaic_tasks.append((input_band_files, out_band_tif))
            mosaic_bands_list.append((out_band_tif, band, doy, year))

    import threading
    total_bands = len(mosaic_tasks)
    done_bands = 0
    lock = threading.Lock()

    logging.info(f"Track {track}: warping {total_bands} single-band DOY mosaics (Workers: {max_workers})...")

    def _worker_mosaic(task):
        nonlocal done_bands
        res = mosaic_single_band_doy(
            task[0], task[1], shp_cutline, target_epsg, ref_proj, res_x, res_y,
            output_bounds, target_width, target_height, overwrite
        )
        with lock:
            done_bands += 1
            if done_bands % 5 == 0 or done_bands == total_bands:
                pct = (done_bands / total_bands) * 100.0
                logging.info(f"  [MOSAIC PROGRESS] Track {track}: {done_bands}/{total_bands} bands completed ({pct:.1f}%)")
        return res

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_worker_mosaic, mosaic_tasks))

    valid_layers = []
    band_descriptions = []
    for out_band_tif, band, doy, year in mosaic_bands_list:
        if out_band_tif.exists() and out_band_tif.stat().st_size > 1024:
            valid_layers.append(str(out_band_tif))
            cal_date = datetime.date(year, 1, 1) + datetime.timedelta(days=int(doy) - 1)
            date_str = cal_date.strftime("%d%b%Y")
            desc = f"S2_{band}_{date_str}_doy{doy}"
            band_descriptions.append(desc)

    if not valid_layers:
        logging.warning(f"No valid mosaic layers generated for track {track}.")
        return

    out_vrt = out_final_dir / f"{sanitized_track}_S2_timeseries_temp.vrt"
    out_final_tif = out_proc_dir / f"{sanitized_track}_S2_timeseries.tif"
    out_final_tmp = out_proc_dir / f"{sanitized_track}_S2_timeseries.tmp.tif"

    logging.info(f"Assembling VRT and translating final {len(band_descriptions)}-band multi-temporal GeoTIFF stack...")
    vrt_opts = gdal.BuildVRTOptions(separate=True)
    gdal.BuildVRT(str(out_vrt), valid_layers, options=vrt_opts)

    trans_opts = gdal.TranslateOptions(
        creationOptions=['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=6', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'],
        callback=gdal.TermProgress_nocb
    )
    if out_final_tmp.exists():
        try: out_final_tmp.unlink()
        except: pass

    ds_final = gdal.Translate(str(out_final_tmp), str(out_vrt), options=trans_opts)

    if ds_final:
        for b_idx, desc in enumerate(band_descriptions, start=1):
            band_obj = ds_final.GetRasterBand(b_idx)
            band_obj.SetDescription(desc)
            band_obj.SetNoDataValue(0)

        if build_overviews:
            logging.info(f"Building compressed pyramid overviews (2, 4, 8, 16, 32, 64) for {out_final_tif.name}...")
            gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
            gdal.SetConfigOption('PREDICTOR_OVERVIEW', '2')
            gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
            ds_final.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32, 64], callback=gdal.TermProgress_nocb)

        ds_final.FlushCache()
        ds_final = None

        if out_final_tif.exists():
            try: out_final_tif.unlink()
            except Exception as e:
                logging.warning(f"Could not remove old file {out_final_tif.name}: {e}")

        if out_final_tmp.exists():
            out_final_tmp.rename(out_final_tif)

    if out_vrt.exists():
        try: out_vrt.unlink()
        except: pass

    # Always keep final DOY mosaics intact for inspection and instant re-stacking.
    # Auto-cleanup raw MGRS tile folders, *_tif, and _synthetic_s2 to free ~600 GB of disk space.
    if out_final_tif.exists() and out_final_tif.stat().st_size > 100 * 1024 * 1024:
        s2_root = track_dir / 'S2'
        if s2_root.exists():
            logging.info(f"Auto-cleanup: removing raw S2 granules and tile tifs for {track} to free disk space (~600 GB)...")
            shutil.rmtree(str(s2_root), ignore_errors=True)

    logging.info(f"SUCCESS: Sentinel-2 Multi-Temporal Stack saved to {out_final_tif} ({len(band_descriptions)} bands)!")


def discover_s1_orbits(country_code: str) -> List[int]:
    country_dir = BASE_DIR / country_code
    if country_dir.exists():
        found = [int(re.search(r'orbit_(\d+)', d.name).group(1)) for d in country_dir.glob("orbit_*") if re.search(r'orbit_(\d+)', d.name)]
        if found: return sorted(list(set(found)))
    return COUNTRY_ORBITS.get(country_code.upper(), [88, 161])


def mosaic_stack_clip_s2(
    country: Optional[str] = None,
    track: Optional[str] = None,
    orbit: Optional[int] = None,
    target_epsg: int = 3857,
    doys: List[int] = DEFAULT_DOYS,
    max_workers: int = 8,
    overwrite: bool = False,
    build_overviews: bool = True
):
    if track:
        norm_track = track.replace('\\', '/')
        if '/' in norm_track:
            c_code = norm_track.split('/')[0]
        else:
            c_code = norm_track.split('_')[0]
        mosaic_stack_clip_single_track(norm_track, c_code, target_epsg, doys, max_workers, overwrite, build_overviews)
    elif country:
        country_code = country.upper()
        if orbit is not None:
            orbits = [orbit]
        else:
            orbits = discover_s1_orbits(country_code)

        for o_num in orbits:
            track_name = f"{country_code}/orbit_{o_num}"
            try:
                mosaic_stack_clip_single_track(track_name, country_code, target_epsg, doys, max_workers, overwrite, build_overviews)
            except Exception as e:
                logging.error(f"Error processing track {track_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Mosaic, Reproject, Clip, and Stack Sentinel-2 Synthetic Time Series.")
    parser.add_argument('-c', '--country', default=None, help="Country code, e.g. PL, NL, FR, PT, AT")
    parser.add_argument('-t', '--track', default=None, help="Track path, e.g. NL/orbit_88")
    parser.add_argument('-o', '--orbit', type=int, default=None, help="Optional single orbit number")
    parser.add_argument('--epsg', type=int, default=3857, help="Target EPSG code (default: 3857)")
    parser.add_argument('--doys', nargs='+', type=int, default=DEFAULT_DOYS, help="List of target DOYs")
    parser.add_argument('--threads', type=int, default=8, help="Worker threads (default: 8)")
    parser.add_argument('--overwrite', action='store_true', help="Force re-mosaicking and overwrite existing rasters")
    parser.add_argument('--no_overviews', action='store_true', help="Skip building pyramid overviews")

    args = parser.parse_args()

    if not args.country and not args.track:
        parser.error("Either --country (-c) or --track (-t) must be specified.")

    mosaic_stack_clip_s2(
        country=args.country,
        track=args.track,
        orbit=args.orbit,
        target_epsg=args.epsg,
        doys=args.doys,
        max_workers=args.threads,
        overwrite=args.overwrite,
        build_overviews=not args.no_overviews
    )


if __name__ == '__main__':
    main()
