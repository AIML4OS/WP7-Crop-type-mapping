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
import logging
import os
import pathlib
import re
import sys
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
    'NL': [88, 161, 15, 37, 110, 139],
    'PL': [22, 73, 124, 175, 29, 95, 102, 146, 168],
    'IE': [30, 74, 103, 132, 147],
    'FR': [8, 30, 37, 59, 81, 88, 103, 110, 132, 139, 153, 161],
    'AT': [22, 29, 73, 95, 102, 124, 146, 168],
    'PT': [161, 81, 153, 8, 88, 110],
    'DE': [22, 29, 73, 95, 102, 124, 146, 168, 175],
    'ES': [8, 81, 88, 153, 161],
    'IT': [44, 117, 146, 168]
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

    cand = COUNTRY_ORBITS.get(country_code, [88, 161])
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
        xml_candidates = list(safe_dir.glob("*MTD_MSIL2A*.xml")) + list(safe_dir.glob("*.xml"))
        for xml_path in xml_candidates:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            for elem in root.iter():
                if elem.tag.endswith("Cloud_Coverage_Assessment") or elem.tag.endswith("CLOUDY_PIXEL_PERCENTAGE"):
                    if elem.text:
                        return float(elem.text)
    except Exception:
        pass
    return 0.0


def extract_tile_and_date_from_safe_name(safe_name: str) -> tuple:
    tile_match = re.search(r'_T([0-9]{2}[A-Z]{3})_', safe_name)
    tile = tile_match.group(1) if tile_match else "UNKNOWN"
    date_match = re.search(r'_(20\d{6})T', safe_name)
    dt = datetime.datetime.strptime(date_match.group(1), "%Y%m%d").date() if date_match else datetime.date(2000, 1, 1)
    return tile, dt


def find_b02_path_in_safe(safe_dir: Path) -> Optional[Path]:
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
    return None


def convert_safe_to_geotiff(safe_dir: Path, output_dest_dir: Path) -> bool:
    output_dest_dir.mkdir(parents=True, exist_ok=True)
    b02_file = find_b02_path_in_safe(safe_dir)
    if not b02_file or not b02_file.exists():
        return False

    b02_dest_tif = output_dest_dir / f"{safe_dir.stem}_B02_20m.tif"
    if b02_dest_tif.exists() and b02_dest_tif.stat().st_size > 1024:
        return True

    success = True
    b02_str = str(b02_file)
    for band in S2_BANDS_20M:
        band_src = Path(b02_str.replace('_B02_20m', f'_{band}_20m'))
        if not band_src.exists():
            band_src = Path(b02_str.replace('B02', band))
        if not band_src.exists():
            candidates = list(b02_file.parent.glob(f"*_{band}_20m.jp2"))
            if not candidates:
                candidates = list(b02_file.parent.glob(f"*{band}*.jp2"))
            if candidates:
                band_src = candidates[0]
            else:
                continue

        band_dst_tif = output_dest_dir / f"{safe_dir.stem}_{band}_20m.tif"
        if band_dst_tif.exists() and band_dst_tif.stat().st_size > 1024:
            continue

        band_dst_tmp = output_dest_dir / f"{safe_dir.stem}_{band}_20m.tmp.tif"
        try:
            ds = gdal.Open(str(band_src))
            if ds is None:
                continue
            options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER', 'NUM_THREADS=ALL_CPUS'])
            gdal.Translate(str(band_dst_tmp), ds, options=options)
            ds = None
            if band_dst_tmp.exists() and band_dst_tmp.stat().st_size > 1024:
                if band_dst_tif.exists():
                    band_dst_tif.unlink()
                band_dst_tmp.rename(band_dst_tif)
            else:
                success = False
        except Exception as e:
            logging.error(f"Error converting {band_src.name}: {e}")
            if band_dst_tmp.exists():
                try: band_dst_tmp.unlink()
                except: pass
            success = False
    return success


def scan_creodias_for_dates(
    repo_path: Path,
    start_date: datetime.date,
    end_date: datetime.date,
    max_cloud_cover: float = 80.0
) -> List[Dict]:
    logging.info(f"Scanning CREODIAS repository ({repo_path}) for dates {start_date} to {end_date}...")
    matched = []
    if not repo_path.exists():
        logging.error(f"CREODIAS S2 path not found: {repo_path}")
        return matched

    curr_date = start_date
    checked = set()
    while curr_date <= end_date:
        day_dir = repo_path / str(curr_date.year) / f"{curr_date.month:02d}" / f"{curr_date.day:02d}"
        if day_dir.exists() and day_dir not in checked:
            checked.add(day_dir)
            for safe_entry in day_dir.glob("S2*_MSIL2A_*.SAFE"):
                tile_name, acq_date = extract_tile_and_date_from_safe_name(safe_entry.name)
                if start_date <= acq_date <= end_date:
                    cloud_pct = parse_metadata_cloud_cover(safe_entry)
                    if cloud_pct <= max_cloud_cover:
                        matched.append({
                            'title': safe_entry.name,
                            'safe_path': safe_entry,
                            'tile': tile_name,
                            'date': acq_date,
                            'cloud_cover': cloud_pct
                        })
        curr_date += datetime.timedelta(days=1)

    logging.info(f"Found {len(matched)} matching S2 SAFE scenes in CREODIAS repo.")
    return matched


def get_s1_orbit_extent_geometry(country_code: str, orbit_num: int) -> Optional[ogr.Geometry]:
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


def process_orbit_creodias_s2(
    country_code: str,
    orbit_num: int,
    start_date: datetime.date,
    end_date: datetime.date,
    all_scenes: Optional[List[Dict]] = None,
    max_cloud_cover: float = 80.0,
    max_workers: int = 8
):
    country_code = country_code.upper()
    track_name = f"{country_code}/orbit_{orbit_num}"
    logging.info(f"\n========================================================")
    logging.info(f" INGESTING & CONVERTING S2 FOR TRACK: {track_name} (Workers: {max_workers})")
    logging.info(f"========================================================")

    dest_track_s2 = BASE_DIR / track_name / "S2"
    dest_track_s2.mkdir(parents=True, exist_ok=True)

    if all_scenes is None:
        all_scenes = scan_creodias_for_dates(S2_REPO_PATH, start_date, end_date, max_cloud_cover)

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
        tile_upper = sc['tile'].upper()
        out_tile_dir = dest_track_s2 / f"{tile_upper}_tif" / sc['title']
        convert_safe_to_geotiff(sc['safe_path'], out_tile_dir)
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
    orbit: Optional[int] = None,
    max_cloud_cover: float = 80.0,
    max_workers: int = 8
):
    country_code = country_code.upper()
    if orbit is not None:
        selected_orbits = [orbit]
    else:
        selected_orbits = discover_s1_orbits(country_code)

    logging.info(f"=== PROCESSING SENTINEL-2 FOR COUNTRY: {country_code} | TARGET ORBITS: {selected_orbits} ===")
    all_scenes = scan_creodias_for_dates(S2_REPO_PATH, start_date, end_date, max_cloud_cover)

    for o_num in selected_orbits:
        process_orbit_creodias_s2(country_code, o_num, start_date, end_date, all_scenes, max_cloud_cover, max_workers)


def main():
    parser = argparse.ArgumentParser(description="Extract & Convert Sentinel-2 L2A from CREODIAS Y: automatically detecting S1 orbits.")
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

    process_country_creodias_s2(
        country_code=args.country,
        start_date=start_dt,
        end_date=end_dt,
        orbit=args.orbit,
        max_cloud_cover=args.cloud_cover,
        max_workers=args.threads
    )


if __name__ == '__main__':
    main()
