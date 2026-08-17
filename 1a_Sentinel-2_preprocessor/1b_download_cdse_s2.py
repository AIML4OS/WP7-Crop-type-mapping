#!/usr/bin/env python
"""
1b_download_cdse_s2.py - Download and Convert Sentinel-2 L2A via CDSE OData API
with automatic detection of Sentinel-1 orbits for a given country.

Usage examples:
  # Automatically detect all S1 orbits in workingDir/<country> (or greedy orbits) and process sequentially:
  python 1b_download_cdse_s2.py -s 2024-10-15 -e 2025-09-15 -c NL
  python 1b_download_cdse_s2.py -s 2024-10-15 -e 2025-09-15 -c PL

  # Process single orbit override:
  python 1b_download_cdse_s2.py -s 2024-10-15 -e 2025-09-15 -c NL -o 88
"""

import argparse
import datetime
import glob
import json
import logging
import os
import pathlib
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from zipfile import ZipFile, BadZipfile

import requests
from osgeo import gdal, ogr, osr

# ================= CONFIGURATION =================
BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDir"))
AUX_DIR = Path(os.environ.get("AIML_AUX_DIR", r"D:/AIML_CropMapper_Cloud/auxiliary_files"))
SHAPEFILES_DIR = AUX_DIR / "shapefiles_nuts"

CDSE_USERNAME = os.environ.get("CDSE_USERNAME", "")
CDSE_PASSWORD = os.environ.get("CDSE_PASSWORD", "")

# Auto-load from JSON configuration files
try:
    _root_dir = Path(__file__).resolve().parent.parent
    _cfg_paths = [
        Path(__file__).resolve().parent / "config_s2.json",
        _root_dir / "config_cdse.json",
        _root_dir / "config.json"
    ]
    for _cp in _cfg_paths:
        if _cp.exists():
            with open(_cp, 'r', encoding='utf-8') as _f:
                _data = json.load(_f)
                if not CDSE_USERNAME:
                    CDSE_USERNAME = _data.get("cdse", {}).get("username") or _data.get("username", "")
                if not CDSE_PASSWORD:
                    CDSE_PASSWORD = _data.get("cdse", {}).get("password") or _data.get("password", "")
                if "paths" in _data:
                    if "working_dir" in _data["paths"]:
                        BASE_DIR = Path(_data["paths"]["working_dir"])
                    if "aux_dir" in _data["paths"]:
                        AUX_DIR = Path(_data["paths"]["aux_dir"])
                        SHAPEFILES_DIR = AUX_DIR / "shapefiles_nuts"
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

_cached_token = None
_token_fetch_time = 0
_token_lock = threading.Lock()


class CDSETokenManager:
    @staticmethod
    def get_token(username: str, password: str) -> str:
        global _cached_token, _token_fetch_time
        with _token_lock:
            now = time.time()
            if _cached_token and (now - _token_fetch_time) < 480:
                return _cached_token

            token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
            data = {
                'client_id': 'cdse-public',
                'grant_type': 'password',
                'username': username,
                'password': password
            }
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            resp = requests.post(token_url, data=data, headers=headers, timeout=30)
            resp.raise_for_status()
            token = resp.json()['access_token']
            _cached_token = token
            _token_fetch_time = now
            return _cached_token


def load_shapefile_geometry_ogr(shp_path: Path, target_epsg=4326) -> Optional[ogr.Geometry]:
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
    country_dir = SHAPEFILES_DIR / country_code.upper()
    shp_path = country_dir / f"NUTS2_{country_code.upper()}.shp"
    if not shp_path.exists() and country_dir.exists():
        shp_files = list(country_dir.glob("*.shp"))
        if shp_files:
            shp_path = shp_files[0]
    if not shp_path.exists():
        return None
    return load_shapefile_geometry_ogr(shp_path)


def discover_s1_orbits(country_code: str) -> List[int]:
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
    logging.info(f"Using candidate greedy orbits for {country_code}: {cand}")
    return cand


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


class Sentinel2FinderCDSE:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def search_by_geometry(
        self,
        geom: ogr.Geometry,
        start_date: datetime.date,
        end_date: datetime.date,
        cloud_cover: float = 80.0
    ) -> List[Dict]:
        start_str = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
        end_str = end_date.strftime("%Y-%m-%dT23:59:59.999Z")
        env = geom.GetEnvelope() # minX, maxX, minY, maxY
        wkt_poly = f"POLYGON(({env[0]} {env[2]}, {env[1]} {env[2]}, {env[1]} {env[3]}, {env[0]} {env[3]}, {env[0]} {env[2]}))"

        next_url = (
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
            f"$filter=Collection/Name eq 'SENTINEL-2' and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
            f"ContentDate/Start ge {start_str} and ContentDate/Start le {end_str} and "
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover}) and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_poly}')&$top=1000&$orderby=ContentDate/Start asc"
        )

        products = []
        try:
            while next_url:
                resp = requests.get(next_url, timeout=45)
                resp.raise_for_status()
                data = resp.json()
                items = data.get('value', [])
                for item in items:
                    name = item['Name']
                    tile_match = re.search(r'_T([0-9]{2}[A-Z]{3})_', name)
                    tile_name = tile_match.group(1) if tile_match else "UNKNOWN"

                    products.append({
                        'id': item['Id'],
                        'title': name,
                        'size': item.get('ContentLength', 0),
                        'tile': tile_name,
                        'start_date': item.get('ContentDate', {}).get('Start', ''),
                        'download_url': f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({item['Id']})/$value"
                    })
                next_url = data.get('@odata.nextLink')
            return products
        except Exception as e:
            logging.error(f"Error querying CDSE: {e}")
            return products


def download_single_product(product: Dict, output_dir: Path, username: str, password: str) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{product['title']}.zip"

    if zip_path.exists() and zip_path.stat().st_size >= product['size'] * 0.95 and product['size'] > 0:
        return zip_path

    token = CDSETokenManager.get_token(username, password)
    headers = {'Authorization': f'Bearer {token}'}

    download_url = product['download_url']
    try:
        # Step 1: Request with allow_redirects=False to catch the redirect and preserve Authorization header
        resp = requests.get(download_url, headers=headers, allow_redirects=False, timeout=30)
        
        if resp.status_code == 401:
            global _cached_token, _token_fetch_time
            _cached_token = None
            _token_fetch_time = 0
            token = CDSETokenManager.get_token(username, password)
            headers = {'Authorization': f'Bearer {token}'}
            resp = requests.get(download_url, headers=headers, allow_redirects=False, timeout=30)

        if resp.status_code in [301, 302, 303, 307, 308]:
            redirect_url = resp.headers.get("Location")
            if redirect_url:
                resp = requests.get(redirect_url, headers=headers, stream=True, timeout=60)

        resp.raise_for_status()

        total_size = int(resp.headers.get('content-length', product.get('size', 0)))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunk

        with open(str(zip_path), 'wb') as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_tot = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r    Downloading: {mb_done:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)    ")
                        sys.stdout.flush()
        print("")
        return zip_path
    except Exception as e:
        print("")
        logging.error(f"Download failed for {product['title']}: {e}")
        if zip_path.exists():
            try: zip_path.unlink()
            except: pass
        return None


def convert_safe_to_geotiff(safe_dir: Path, output_dest_dir: Path) -> bool:
    output_dest_dir.mkdir(parents=True, exist_ok=True)
    g_dir = safe_dir / 'GRANULE'
    if not g_dir.exists():
        return False

    b02_file = None
    for granule in g_dir.iterdir():
        if granule.is_dir():
            r20m_dir = granule / 'IMG_DATA' / 'R20m'
            if r20m_dir.exists():
                b02_candidates = list(r20m_dir.glob("*_B02_20m.jp2"))
                if b02_candidates:
                    b02_file = b02_candidates[0]
                    break

    if not b02_file or not b02_file.exists():
        return False

    b02_str = str(b02_file)
    success = True
    for band in S2_BANDS_20M:
        band_src = Path(b02_str.replace('_B02_20m', f'_{band}_20m'))
        if not band_src.exists():
            band_src = Path(b02_str.replace('B02', band))
        if not band_src.exists():
            continue

        band_dst_tif = output_dest_dir / f"{safe_dir.stem}_{band}_20m.tif"
        if band_dst_tif.exists() and band_dst_tif.stat().st_size > 1024:
            continue

        try:
            ds = gdal.Open(str(band_src))
            if ds is None:
                continue
            options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'])
            gdal.Translate(str(band_dst_tif), ds, options=options)
            ds = None
        except Exception as e:
            logging.error(f"Failed converting {band_src.name}: {e}")
            success = False
    return success


def process_orbit_cdse_s2(
    country_code: str,
    orbit_num: int,
    start_date: datetime.date,
    end_date: datetime.date,
    username: str = CDSE_USERNAME,
    password: str = CDSE_PASSWORD,
    cloud_cover: float = 80.0,
    max_workers: int = 4
):
    country_code = country_code.upper()
    track_name = f"{country_code}/orbit_{orbit_num}"
    logging.info(f"\n========================================================")
    logging.info(f" CDSE S2 DOWNLOAD & CONVERSION FOR TRACK: {track_name}")
    logging.info(f"========================================================")

    dest_track_s2 = BASE_DIR / track_name / "S2"
    dest_track_s2.mkdir(parents=True, exist_ok=True)

    orbit_geom = get_s1_orbit_extent_geometry(country_code, orbit_num)
    if not orbit_geom:
        logging.error(f"Cannot resolve spatial extent for {track_name}")
        return

    finder = Sentinel2FinderCDSE(username, password)
    products = finder.search_by_geometry(orbit_geom, start_date, end_date, cloud_cover)
    logging.info(f"Discovered {len(products)} Sentinel-2 products on CDSE intersecting {track_name}.")

    total_prods = len(products)
    for idx, prod in enumerate(products, start=1):
        tile_upper = prod['tile'].upper()
        dest_prod_tif_dir = dest_track_s2 / f"{tile_upper}_tif" / prod['title']

        check_b02 = dest_prod_tif_dir / f"{prod['title']}_B02_20m.tif"
        if check_b02.exists() and check_b02.stat().st_size > 1024:
            continue

        logging.info(f"[{idx}/{total_prods}] Downloading & Converting: {prod['title']} (Tile: {tile_upper}, Size: {prod['size'] / (1024*1024):.1f} MB)")

        tile_raw_dir = dest_track_s2 / tile_upper
        tile_raw_dir.mkdir(parents=True, exist_ok=True)
        unzipped_safe = tile_raw_dir / f"{prod['title']}.SAFE"

        if not unzipped_safe.exists():
            zip_file = download_single_product(prod, tile_raw_dir, username, password)
            if not zip_file or not zip_file.exists():
                continue
            try:
                logging.info(f"    Extracting ZIP archive for {prod['title']}...")
                with ZipFile(str(zip_file), 'r') as z:
                    z.extractall(str(tile_raw_dir))
                try: zip_file.unlink()
                except: pass
            except BadZipfile:
                try: zip_file.unlink()
                except: pass
                continue

        if unzipped_safe.exists():
            logging.info(f"    Converting 20m bands to GeoTIFF...")
            convert_safe_to_geotiff(unzipped_safe, dest_prod_tif_dir)
            try: shutil.rmtree(str(unzipped_safe))
            except: pass

    logging.info(f"SUCCESS: CDSE S2 download & conversion completed for track {track_name}!")


def process_country_cdse_s2(
    country_code: str,
    start_date: datetime.date,
    end_date: datetime.date,
    orbit: Optional[int] = None,
    username: str = CDSE_USERNAME,
    password: str = CDSE_PASSWORD,
    cloud_cover: float = 80.0,
    max_workers: int = 4
):
    country_code = country_code.upper()
    if orbit is not None:
        selected_orbits = [orbit]
    else:
        selected_orbits = discover_s1_orbits(country_code)

    logging.info(f"=== CDSE SENTINEL-2 DOWNLOAD FOR COUNTRY: {country_code} | TARGET ORBITS: {selected_orbits} ===")

    for o_num in selected_orbits:
        process_orbit_cdse_s2(country_code, o_num, start_date, end_date, username, password, cloud_cover, max_workers)


def main():
    parser = argparse.ArgumentParser(description="Download and convert Sentinel-2 L2A from CDSE automatically detecting S1 orbits.")
    parser.add_argument('-s', '--start_date', required=True, help="Start date (YYYY-MM-DD), e.g. 2024-10-15")
    parser.add_argument('-e', '--end_date', required=True, help="End date (YYYY-MM-DD), e.g. 2025-09-15")
    parser.add_argument('-c', '--country', required=True, help="Country code, e.g. PL, NL, FR, PT, AT")
    parser.add_argument('-o', '--orbit', type=int, default=None, help="Optional single orbit override")
    parser.add_argument('--cloud_cover', type=float, default=80.0, help="Maximum cloud cover (default: 80)")
    parser.add_argument('--threads', type=int, default=4, help="Worker threads (default: 4)")
    parser.add_argument('--username', default=CDSE_USERNAME, help="CDSE Username")
    parser.add_argument('--password', default=CDSE_PASSWORD, help="CDSE Password")

    args = parser.parse_args()

    start_dt = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()

    process_country_cdse_s2(
        country_code=args.country,
        start_date=start_dt,
        end_date=end_dt,
        orbit=args.orbit,
        username=args.username,
        password=args.password,
        cloud_cover=args.cloud_cover,
        max_workers=args.threads
    )


if __name__ == '__main__':
    main()
