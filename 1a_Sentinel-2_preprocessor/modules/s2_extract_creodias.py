#!/usr/bin/env python
"""
1a_extract_creodias_s2.py - Direct Sentinel-2 L2A Extraction & Conversion from CREODIAS mounted drive (Y:)
with automatic detection of Sentinel-1 orbits for a given country.

Usage examples:
  # Automatically detect all S1 orbits in workingDir/<country> (or greedy orbits) and process sequentially:
  python 1a_extract_creodias_s2.py -s 2024-10-15 -e 2025-09-15 -c NL
  python 1a_extract_creodias_s2.py -s 2024-10-15 -e 2025-09-15 -c PL

  # Process a specific single orbit:
  python 1a_extract_creodias_s2.py -s 2024-10-15 -e 2025-09-15 -c NL -o 88
"""

import argparse
import datetime
import glob
import json
import logging
import os
import pathlib
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from osgeo import gdal, ogr, osr

BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDirs"))
AUX_DIR = Path(os.environ.get("AIML_AUX_DIR", r"D:/AIML_CropMapper_Cloud/auxiliary_files"))
SHAPEFILES_DIR = AUX_DIR / "shapefiles_nuts"
S2_REPO_PATH = Path(os.environ.get("S2_REPO_PATH", r"Y:\Sentinel-2\MSI\L2A"))

# Auto-load from JSON configuration files
try:
    _root_dir = Path(__file__).resolve().parent.parent
    _cfg_paths = [
        Path(__file__).resolve().parent / "config_s2.json",
        _root_dir / "config.json"
    ]
    for _cp in _cfg_paths:
        if _cp.exists():
            with open(_cp, 'r', encoding='utf-8') as _f:
                _data = json.load(_f)
                if "paths" in _data:
                    if "working_dir" in _data["paths"]:
                        BASE_DIR = Path(_data["paths"]["working_dir"])
                    if "aux_dir" in _data["paths"]:
                        AUX_DIR = Path(_data["paths"]["aux_dir"])
                        SHAPEFILES_DIR = AUX_DIR / "shapefiles_nuts"
                    if "s2_repo_path" in _data["paths"]:
                        S2_REPO_PATH = Path(_data["paths"]["s2_repo_path"])
except Exception:
    pass

S2_BANDS_20M = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']

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

COUNTRY_MGRS_TILES = {
    'NL': ['31UDR', '31UER', '31UES', '31UET', '31UFR', '31UFS', '31UFT', '31UGR', '31UGS', '31UGT'],
    'PT': [
        '29SMC', '29SMD', '29SNC', '29SND', '29SPC', '29SPD', '29SQC', '29SQD',
        '29SMB', '29SNB', '29SPB', '29SQB', '29SMA', '29SNA', '29SPA', '29SQA',
        '29TNE', '29TNF', '29TNG', '29TPE', '29TPF', '29TPG', '29TQE', '29TQF', '29TQG'
    ],
    'PL': [
        '33UUT', '33UUS', '33UVR', '33UVS', '33UVT', '33UWU', '33UWV', '33UWR', '33UWS', '33UWT',
        '34UCU', '34UCV', '34UCA', '34UCB', '34UCC', '34UCD', '34UCE', '34UCF',
        '34UDU', '34UDV', '34UDA', '34UDB', '34UDC', '34UDD', '34UDE', '34UDF',
        '34UEU', '34UEV', '34UEA', '34UEB', '34UEC', '34UED', '34UEE', '34UEF',
        '34UFU', '34UFV', '34UFA', '34UFB', '34UFC', '34UFD', '34UFE', '34UFF',
        '34UGU', '34UGV', '34UGA', '34UGB', '34UGC', '34UGD', '34UGE', '34UGF',
        '35ULA', '35ULB', '35ULC', '35ULD', '35ULE', '35ULF',
        '33UXT', '33UXU', '33UXV', '34VDC', '34VDD', '34VDE', '34VDF'
    ],
    'IE': ['29UNU', '29UNV', '29UPU', '29UPV', '29UQU', '29UQV', '29UNT', '29UPT', '29UQT', '29UNS', '29UPS', '29UQS'],
    'AT': ['32UPU', '32UPV', '32UQU', '32UQV', '33UUP', '33UUQ', '33UVP', '33UVQ', '33UWP', '33UWQ', '33UXP', '33UXQ'],
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def load_shapefile_geometry_ogr(shp_path: Path, target_epsg=4326) -> Optional[ogr.Geometry]:
    """Loads all geometries from a shapefile, transforms to target EPSG, and returns union."""
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(str(shp_path))
    if not ds:
        return None
    layer = ds.GetLayer()
    src_srs = layer.GetSpatialRef()
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(target_epsg)

    coord_trans = None
    if src_srs and not src_srs.IsSame(dst_srs):
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        coord_trans = osr.CoordinateTransformation(src_srs, dst_srs)

    union_geom = None
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom:
            cloned = geom.Clone()
            if coord_trans:
                cloned.Transform(coord_trans)
            if union_geom is None:
                union_geom = cloned
            else:
                union_geom = union_geom.Union(cloned)
    ds = None
    return union_geom


def get_country_geometry(country_code: str) -> Optional[ogr.Geometry]:
    """Dynamically locates and loads the NUTS shapefile for the given country code."""
    country_dir = SHAPEFILES_DIR / country_code.upper()
    shp_path = country_dir / f"NUTS2_{country_code.upper()}.shp"
    if not shp_path.exists() and country_dir.exists():
        shp_files = list(country_dir.glob("*.shp"))
        if shp_files:
            shp_path = shp_files[0]
    if not shp_path.exists():
        logging.error(f"Shapefile for country {country_code} not found in {SHAPEFILES_DIR}")
        return None

    return load_shapefile_geometry_ogr(shp_path)


def discover_s1_orbits(country_code: str) -> List[int]:
    """
    Automatically detects existing S1 orbits in workingDir/<country>/orbit_* or falls back to greedy candidate list.
    """
    country_code = country_code.upper()
    country_dir = BASE_DIR / country_code
    orbits = set()

    if country_dir.exists():
        for o_dir in country_dir.glob("orbit_*"):
            match = re.search(r'orbit_(\d+)', o_dir.name)
            if match:
                orbits.add(int(match.group(1)))

    if orbits:
        logging.info(f"Automatically detected {len(orbits)} Sentinel-1 orbit(s) in workingDir for {country_code}: {sorted(orbits)}")
        return sorted(orbits)

    cand = COUNTRY_ORBITS.get(country_code, [])
    logging.info(f"No existing orbit folders in workingDir/{country_code}. Using candidate greedy orbits: {cand}")
    return cand


def get_s1_orbit_extent_geometry(country_code: str, orbit_num: int) -> Optional[ogr.Geometry]:
    """Retrieves bounding polygon for S1 orbit raster if present, else fallback to country geometry."""
    track_dir = BASE_DIR / country_code.upper() / f"orbit_{orbit_num}"
    proc_dir = track_dir / "processed_raster"

    if proc_dir.exists():
        s1_tifs = list(proc_dir.glob("*_VH_VV*.tif"))
        if s1_tifs:
            ds = gdal.Open(str(s1_tifs[0]))
            if ds:
                gt = ds.GetGeoTransform()
                w = ds.RasterXSize
                h = ds.RasterYSize
                proj_wkt = ds.GetProjection()

                min_x = gt[0]
                max_x = gt[0] + w * gt[1]
                max_y = gt[3]
                min_y = gt[3] + h * gt[5]

                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(min_x, min_y)
                ring.AddPoint(max_x, min_y)
                ring.AddPoint(max_x, max_y)
                ring.AddPoint(min_x, max_y)
                ring.AddPoint(min_x, min_y)

                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                src_srs = osr.SpatialReference()
                src_srs.ImportFromWkt(proj_wkt)
                dst_srs = osr.SpatialReference()
                dst_srs.ImportFromEPSG(4326)

                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

                transform = osr.CoordinateTransformation(src_srs, dst_srs)
                poly.Transform(transform)
                ds = None
                return poly

    return get_country_geometry(country_code)


def parse_metadata_cloud_cover(safe_dir: Path) -> float:
    """Extracts cloudy pixel percentage from MTD_MSIL2A.xml in SAFE directory."""
    try:
        xml_path = safe_dir / "MTD_MSIL2A.xml"
        if not xml_path.exists():
            candidates = list(safe_dir.glob("*.xml"))
            if candidates:
                xml_path = candidates[0]
            else:
                return 0.0
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag.endswith("Cloud_Coverage_Assessment") or elem.tag.endswith("CLOUDY_PIXEL_PERCENTAGE"):
                if elem.text:
                    return float(elem.text)
    except Exception:
        pass
    return 0.0


# Optimize GDAL for network SMB reading and suppress heavy directory crawling
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.jp2,.tif,.xml')
gdal.SetConfigOption('GDAL_CACHEMAX', '2048')

_GLOBAL_SCAN_CACHE: Dict[Tuple, List[Dict]] = {}


def extract_tile_and_date_from_safe_name(safe_name: str) -> tuple:
    tile_match = re.search(r'_T([0-9]{2}[A-Z]{3})_', safe_name)
    tile = tile_match.group(1) if tile_match else "UNKNOWN"
    date_match = re.search(r'_(20\d{6})T', safe_name)
    dt = datetime.datetime.strptime(date_match.group(1), "%Y%m%d").date() if date_match else datetime.date(2000, 1, 1)
    return tile, dt


def find_b02_path_in_safe(safe_dir: Path) -> Optional[Path]:
    try:
        g_dir = safe_dir / 'GRANULE'
        if not g_dir.exists():
            return None
        for granule in g_dir.iterdir():
            if granule.is_dir():
                r20m_dir = granule / 'IMG_DATA' / 'R20m'
                if r20m_dir.exists():
                    b02_files = list(r20m_dir.glob("*_B02_20m.jp2"))
                    if b02_files:
                        return b02_files[0]
                    b02_tif = list(r20m_dir.glob("*_B02_20m.tif"))
                    if b02_tif:
                        return b02_tif[0]
                r10m_dir = granule / 'IMG_DATA' / 'R10m'
                if r10m_dir.exists():
                    b02_files = list(r10m_dir.glob("*_B02_10m.jp2")) + list(r10m_dir.glob("*_B02*.jp2"))
                    if b02_files:
                        return b02_files[0]
    except Exception as e:
        logging.debug(f"Could not inspect granules in {safe_dir.name}: {e}")
        return None
    return None


def convert_safe_to_geotiff(safe_dir: Path, output_dest_dir: Path) -> bool:
    try:
        output_dest_dir.mkdir(parents=True, exist_ok=True)

        # Fast local check: if all 10 bands already exist locally, skip touching the network drive completely
        all_bands_exist = True
        for band in S2_BANDS_20M:
            dst_tif = output_dest_dir / f"{safe_dir.stem}_{band}_20m.tif"
            if not (dst_tif.exists() and dst_tif.stat().st_size > 1024):
                all_bands_exist = False
                break
        if all_bands_exist:
            return True

        b02_file = find_b02_path_in_safe(safe_dir)
        if not b02_file or not b02_file.exists():
            return False

        success = True
        b02_str = str(b02_file)
        for band in S2_BANDS_20M:
            band_dst_tif = output_dest_dir / f"{safe_dir.stem}_{band}_20m.tif"
            if band_dst_tif.exists() and band_dst_tif.stat().st_size > 1024:
                continue

            band_src = Path(b02_str.replace('_B02_20m', f'_{band}_20m'))
            if not band_src.exists():
                band_src = Path(b02_str.replace('B02', band))
            if not band_src.exists():
                try:
                    candidates = list(b02_file.parent.glob(f"*_{band}_20m.jp2"))
                    if not candidates:
                        candidates = list(b02_file.parent.glob(f"*{band}*.jp2"))
                    if candidates:
                        band_src = candidates[0]
                    else:
                        continue
                except Exception:
                    continue

            try:
                ds = gdal.Open(str(band_src))
                if ds is None:
                    continue
                options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER', 'NUM_THREADS=ALL_CPUS'])
                out_ds = gdal.Translate(str(band_dst_tif), ds, options=options)
                out_ds = None  # Explicitly close and flush to disk to avoid Windows handle locks
                ds = None
                if not (band_dst_tif.exists() and band_dst_tif.stat().st_size > 1024):
                    success = False
            except Exception as e:
                logging.debug(f"Failed converting band {band_src.name}: {e}")
                if band_dst_tif.exists():
                    try: band_dst_tif.unlink()
                    except: pass
                success = False

        if not any(output_dest_dir.glob("*.tif")):
            try:
                output_dest_dir.rmdir()
            except Exception:
                pass
        return success
    except Exception as e:
        logging.warning(f"Error converting scene {safe_dir.name}: {e}")
        return False


def scan_creodias_for_dates(
    repo_path: Path,
    start_date: datetime.date,
    end_date: datetime.date,
    target_tiles: Optional[List[str]] = None,
    max_cloud_cover: float = 80.0
) -> List[Dict]:
    tiles_key = tuple(sorted(set(t.upper().replace('T', '') for t in target_tiles))) if target_tiles else None
    cache_key = (str(repo_path), str(start_date), str(end_date), tiles_key, max_cloud_cover)
    if cache_key in _GLOBAL_SCAN_CACHE:
        cached_all = _GLOBAL_SCAN_CACHE[cache_key]
        logging.info(f"Loaded {len(cached_all)} matching S2 SAFE scenes from memory cache.")
        return list(cached_all)

    logging.info(f"Scanning CREODIAS repository ({repo_path}) for dates {start_date} to {end_date} (Target MGRS tiles: {len(target_tiles) if target_tiles else 'ALL'})...")
    matched = []
    if not repo_path.exists():
        logging.error(f"CREODIAS S2 path not found: {repo_path}")
        return matched

    target_tile_tags = [f"_T{t.upper().replace('T', '')}_" for t in target_tiles] if target_tiles else None

    curr_date = start_date
    checked = set()
    total_days = (end_date - start_date).days + 1
    day_idx = 0

    while curr_date <= end_date:
        day_idx += 1
        day_dir = repo_path / str(curr_date.year) / f"{curr_date.month:02d}" / f"{curr_date.day:02d}"
        
        # Robust network retry against Windows SMB I/O timeouts (WinError 1117)
        day_exists = False
        for attempt in range(3):
            try:
                if os.path.exists(str(day_dir)):
                    day_exists = True
                    break
            except OSError:
                time.sleep(0.3)

        if day_exists and day_dir not in checked:
            checked.add(day_dir)
            for attempt in range(3):
                try:
                    with os.scandir(str(day_dir)) as it:
                        for entry in it:
                            name = entry.name
                            if name.startswith("S2") and name.endswith(".SAFE"):
                                # Fast tile name filter before opening XML over the network
                                if target_tile_tags is None or any(tag in name for tag in target_tile_tags):
                                    tile_name, acq_date = extract_tile_and_date_from_safe_name(name)
                                    if start_date <= acq_date <= end_date:
                                        safe_path = Path(entry.path)
                                        cloud_pct = parse_metadata_cloud_cover(safe_path)
                                        if cloud_pct <= max_cloud_cover:
                                            matched.append({
                                                'title': name,
                                                'safe_path': safe_path,
                                                'tile': tile_name,
                                                'date': acq_date,
                                                'cloud_cover': cloud_pct
                                            })
                    break
                except OSError as e:
                    if attempt == 2:
                        logging.warning(f"Network glitch scanning day directory {day_dir}: {e}")
                    time.sleep(0.5)
        curr_date += datetime.timedelta(days=1)

    _GLOBAL_SCAN_CACHE[cache_key] = list(matched)
    logging.info(f"Found {len(matched)} matching S2 SAFE scenes in CREODIAS repo.")
    return matched


def get_s1_orbit_extent_geometry(country_code: str, orbit_num: int) -> Optional[ogr.Geometry]:
    track_dir = BASE_DIR / country_code.upper() / f"orbit_{orbit_num}"
    candidate_dirs = [
        track_dir / "1_input_stacks",
        track_dir,
        track_dir / "processed_raster",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / country_code.upper() / f"orbit_{orbit_num}" / "1_input_stacks",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / country_code.upper() / f"orbit_{orbit_num}" / "processed_raster",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / country_code.upper() / f"orbit_{orbit_num}"
    ]

    for proc_dir in candidate_dirs:
        if proc_dir.exists():
            s1_tifs = list(proc_dir.glob("*_VH_VV*.tif")) + list(proc_dir.glob("*Sigma0*.tif")) + list(proc_dir.glob("S1_*_stack*.tif"))
            if s1_tifs:
                ds = gdal.Open(str(s1_tifs[0]))
                if ds:
                    gt = ds.GetGeoTransform()
                    w = ds.RasterXSize
                    h = ds.RasterYSize
                    proj_wkt = ds.GetProjection()

                    min_x = gt[0]
                    max_x = gt[0] + w * gt[1]
                    max_y = gt[3]
                    min_y = gt[3] + h * gt[5]

                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    ring.AddPoint(min_x, min_y)
                    ring.AddPoint(max_x, min_y)
                    ring.AddPoint(max_x, max_y)
                    ring.AddPoint(min_x, max_y)
                    ring.AddPoint(min_x, min_y)

                    poly = ogr.Geometry(ogr.wkbPolygon)
                    poly.AddGeometry(ring)

                    src_srs = osr.SpatialReference()
                    src_srs.ImportFromWkt(proj_wkt)
                    dst_srs = osr.SpatialReference()
                    dst_srs.ImportFromEPSG(4326)

                    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

                    transform = osr.CoordinateTransformation(src_srs, dst_srs)
                    poly.Transform(transform)
                    ds = None
                    return poly

    return get_country_geometry(country_code)


def process_orbit_creodias_s2(
    country_code: str,
    orbit_num: int,
    start_date: datetime.date,
    end_date: datetime.date,
    all_scenes: Optional[List[Dict]] = None,
    max_cloud_cover: float = 80.0,
    max_workers: int = 8,
    target_s2_dir: Optional[Path] = None
):
    country_code = country_code.upper()
    track_name = f"{country_code}/orbit_{orbit_num}"
    logging.info(f"\n========================================================")
    logging.info(f" INGESTING & CONVERTING S2 FOR TRACK: {track_name} (Workers: {max_workers})")
    logging.info(f"========================================================")

    if target_s2_dir is not None:
        dest_track_s2 = target_s2_dir
    else:
        shared_s2 = BASE_DIR / country_code / "S2"
        orbit_s2 = BASE_DIR / track_name / "S2"
        dest_track_s2 = shared_s2 if shared_s2.exists() else orbit_s2
    dest_track_s2.mkdir(parents=True, exist_ok=True)

    if all_scenes is None:
        target_tiles = COUNTRY_MGRS_TILES.get(country_code, None)
        all_scenes = scan_creodias_for_dates(S2_REPO_PATH, start_date, end_date, target_tiles=target_tiles, max_cloud_cover=max_cloud_cover)

    if not all_scenes:
        logging.warning("No S2 products available to convert.")
        return

    # Filter scenes already converted
    scenes_to_process = []
    for sc in all_scenes:
        tile_upper = sc['tile'].upper()
        dest_prod_tif_dir = dest_track_s2 / f"{tile_upper}_tif" / sc['title']
        check_b02 = dest_prod_tif_dir / f"{sc['title']}_B02_20m.tif"
        if not (check_b02.exists() and check_b02.stat().st_size > 1024):
            # Also check fallback shared directory
            shared_check = BASE_DIR / country_code / "S2" / f"{tile_upper}_tif" / sc['title'] / f"{sc['title']}_B02_20m.tif"
            if not (shared_check.exists() and shared_check.stat().st_size > 1024):
                scenes_to_process.append(sc)

    total_scenes = len(scenes_to_process)
    logging.info(f"Remaining S2 products to convert for {track_name}: {total_scenes} (out of {len(all_scenes)} total)")

    if total_scenes == 0:
        logging.info(f"All Sentinel-2 products for {track_name} are already converted to GeoTIFF!")
        return

    converted_count = 0
    lock = threading.Lock()

    def _worker_convert_creodias(sc: dict):
        nonlocal converted_count
        try:
            tile_upper = sc['tile'].upper()
            out_tile_dir = dest_track_s2 / f"{tile_upper}_tif" / sc['title']
            convert_safe_to_geotiff(sc['safe_path'], out_tile_dir)
        except Exception as e:
            logging.debug(f"Skipping damaged scene {sc.get('title')}: {e}")
        finally:
            with lock:
                converted_count += 1
                if converted_count % 10 == 0 or converted_count == total_scenes:
                    pct = (converted_count / total_scenes) * 100.0
                    logging.info(f"  [CREODIAS CONVERSION] {converted_count}/{total_scenes} products completed ({pct:.1f}%)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_worker_convert_creodias, scenes_to_process))

    logging.info(f"SUCCESS: Sentinel-2 extraction & conversion completed for track {track_name}!")


def process_country_creodias_s2(
    country_code: str,
    start_date: datetime.date,
    end_date: datetime.date,
    max_cloud_cover: float = 80.0,
    max_workers: int = 8
):
    country_code = country_code.upper()
    dest_country_s2 = BASE_DIR / country_code / "S2"
    dest_country_s2.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n========================================================")
    logging.info(f" CREODIAS SENTINEL-2 EXTRACTION FOR COUNTRY: {country_code} (Target: {dest_country_s2})")
    logging.info(f" Date range: {start_date} to {end_date} | Cloud cover <= {max_cloud_cover}% | Workers: {max_workers}")
    logging.info(f"========================================================")

    target_tiles = COUNTRY_MGRS_TILES.get(country_code, None)
    all_scenes = scan_creodias_for_dates(S2_REPO_PATH, start_date, end_date, target_tiles=target_tiles, max_cloud_cover=max_cloud_cover)

    if not all_scenes:
        logging.warning(f"No Sentinel-2 products found on CREODIAS for country {country_code}.")
        return

    # Filter scenes already converted in shared country pool
    scenes_to_process = []
    for sc in all_scenes:
        tile_upper = sc['tile'].upper()
        dest_prod_tif_dir = dest_country_s2 / f"{tile_upper}_tif" / sc['title']
        check_b02 = dest_prod_tif_dir / f"{sc['title']}_B02_20m.tif"
        if not (check_b02.exists() and check_b02.stat().st_size > 1024):
            scenes_to_process.append(sc)

    total_scenes = len(scenes_to_process)
    logging.info(f"Remaining S2 products to convert for country {country_code}: {total_scenes} (Already processed: {len(all_scenes) - total_scenes})")

    if total_scenes == 0:
        logging.info(f"All Sentinel-2 products for country {country_code} are already converted to GeoTIFF!")
        return

    converted_count = 0
    lock = threading.Lock()

    def _worker_convert_creodias(sc: dict):
        nonlocal converted_count
        try:
            tile_upper = sc['tile'].upper()
            out_tile_dir = dest_country_s2 / f"{tile_upper}_tif" / sc['title']
            convert_safe_to_geotiff(sc['safe_path'], out_tile_dir)
        except Exception as e:
            logging.debug(f"Skipping damaged scene {sc.get('title')}: {e}")
        finally:
            with lock:
                converted_count += 1
                if converted_count % 10 == 0 or converted_count == total_scenes:
                    pct = (converted_count / total_scenes) * 100.0
                    logging.info(f"  [CREODIAS CONVERSION] {converted_count}/{total_scenes} products completed ({pct:.1f}%)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_worker_convert_creodias, scenes_to_process))

    logging.info(f"SUCCESS: Sentinel-2 extraction & conversion completed for country {country_code}!\n")


def main():
    parser = argparse.ArgumentParser(description="Extract & Convert Sentinel-2 L2A from CREODIAS Y: drive.")
    parser.add_argument('-s', '--start_date', required=True, help="Start date (YYYY-MM-DD), e.g. 2024-10-15")
    parser.add_argument('-e', '--end_date', required=True, help="End date (YYYY-MM-DD), e.g. 2025-09-15")
    parser.add_argument('-c', '--country', required=True, help="Country code, e.g. PL, NL, FR, PT, AT")
    parser.add_argument('-o', '--orbit', type=int, default=None, help="Optional single orbit override")
    parser.add_argument('--cloud_cover', type=float, default=80.0, help="Max cloud cover (default: 80)")
    parser.add_argument('--threads', type=int, default=8, help="Worker threads (default: 8)")
    parser.add_argument('--repo_path', type=str, default=None, help="Override CREODIAS repo path")

    args = parser.parse_args()

    start_dt = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()

    global S2_REPO_PATH
    if args.repo_path:
        S2_REPO_PATH = Path(args.repo_path)

    if args.orbit:
        process_orbit_creodias_s2(
            country_code=args.country,
            orbit_num=args.orbit,
            start_date=start_dt,
            end_date=end_dt,
            max_cloud_cover=args.cloud_cover,
            max_workers=args.threads
        )
    else:
        process_country_creodias_s2(
            country_code=args.country,
            start_date=start_dt,
            end_date=end_dt,
            max_cloud_cover=args.cloud_cover,
            max_workers=args.threads
        )


if __name__ == '__main__':
    main()
