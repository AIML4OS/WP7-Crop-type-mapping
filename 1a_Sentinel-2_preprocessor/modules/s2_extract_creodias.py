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
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from osgeo import gdal, ogr, osr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", str(_PROJECT_ROOT / "workingDirs")))
AUX_DIR = Path(os.environ.get("AIML_AUX_DIR", str(_PROJECT_ROOT / "auxiliary_files")))
SHAPEFILES_DIR = AUX_DIR / "shapefiles_nuts"
def safe_exists(p) -> bool:
    """Safely check if path exists, catching OSError (including Errno 5 / EIO on FUSE/s3fs mounts)."""
    if p is None:
        return False
    try:
        return Path(p).exists()
    except OSError:
        return False
    except Exception:
        return False


_DEFAULT_S2_REPO = "/eodata/Sentinel-2/MSI/L2A" if (os.name != "nt" or safe_exists("/eodata")) else r"Y:\Sentinel-2\MSI\L2A"
S2_REPO_PATH = Path(os.environ.get("S2_REPO_PATH", _DEFAULT_S2_REPO))

# Auto-load from JSON configuration files
try:
    _root_dir = Path(__file__).resolve().parent.parent
    _cfg_paths = [
        _root_dir / "config_s2.json",
        _root_dir / "config.json",
        Path(__file__).resolve().parent / "config_s2.json"
    ]
    for _cp in _cfg_paths:
        if _cp.exists():
            with open(_cp, 'r', encoding='utf-8') as _f:
                _data = json.load(_f)
                if "paths" in _data:
                    if "AIML_WORKING_DIR" not in os.environ and "working_dir" in _data["paths"]:
                        BASE_DIR = Path(_data["paths"]["working_dir"])
                    if "AIML_AUX_DIR" not in os.environ and "aux_dir" in _data["paths"]:
                        AUX_DIR = Path(_data["paths"]["aux_dir"])
                        SHAPEFILES_DIR = AUX_DIR / "shapefiles_nuts"
                    if "S2_REPO_PATH" not in os.environ and "s2_repo_path" in _data["paths"]:
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
    'IS': [9, 16, 111, 118, 155],  # DESCENDING+ASCENDING (100.0%) - 5 orbits
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
    'NL': [
        '31UES', '31UET', '31UEU',
        '31UFS', '31UFT', '31UFU', '31UFV',
        '31UGS', '31UGT', '31UGU', '31UGV'
    ],
    'PT': [
        '29SMA', '29SMB', '29SMC', '29SMD',
        '29SNA', '29SNB', '29SNC', '29SND',
        '29SPA', '29SPB', '29SPC', '29SPD',
        '29TME', '29TNE', '29TNF', '29TNG',
        '29TPE', '29TPF', '29TPG', '29TQF', '29TQG'
    ],
    'PL': [
        '33UVA', '33UVS', '33UVT', '33UVU', '33UVV',
        '33UWA', '33UWR', '33UWS', '33UWT', '33UWU', '33UWV',
        '33UXA', '33UXR', '33UXS', '33UXT', '33UXU', '33UXV',
        '33UYQ', '33UYR', '33UYS', '33UYT', '33UYU', '33UYV',
        '34UCA', '34UCB', '34UCC', '34UCD', '34UCE', '34UCF', '34UCV',
        '34UDA', '34UDB', '34UDC', '34UDD', '34UDE', '34UDF', '34UDV',
        '34UEA', '34UEB', '34UEC', '34UED', '34UEE', '34UEF', '34UEV',
        '34UFA', '34UFB', '34UFC', '34UFD', '34UFE', '34UFF', '34UFV',
        '34UGA', '34UGB'
    ],
    'IE': [
        '29ULT', '29UMA', '29UMS', '29UMT', '29UMU', '29UMV',
        '29UNA', '29UNB', '29UNT', '29UNU', '29UNV',
        '29UPA', '29UPB', '29UPT', '29UPU', '29UPV', '29UQU'
    ],
    'AT': [
        '32TNS', '32TNT', '32TPS', '32TPT', '32TQS', '32TQT', '32UQU',
        '33TUM', '33TUN', '33TVM', '33TVN', '33TWM', '33TWN', '33TXN',
        '33UUP', '33UVP', '33UVQ', '33UWP', '33UWQ', '33UXP', '33UXQ'
    ],
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def load_shapefile_geometry_ogr(shp_path: Path, target_epsg: int = 4326, country_code: Optional[str] = None) -> Optional[ogr.Geometry]:
    """Loads all continental geometries from a NUTS shapefile, transforms to target EPSG, and returns a validated union."""
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(str(shp_path))
    if not ds:
        return None
    layer = ds.GetLayer()
    src_srs = layer.GetSpatialRef()

    srs_4326 = osr.SpatialReference()
    srs_4326.ImportFromEPSG(4326)
    srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    to_4326 = None
    if src_srs and not src_srs.IsSame(srs_4326):
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        to_4326 = osr.CoordinateTransformation(src_srs, srs_4326)

    c_up = (country_code or "").upper()
    union_geom = ogr.Geometry(ogr.wkbMultiPolygon)
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if not geom:
            continue
        cloned = geom.Clone()
        if to_4326:
            cloned.Transform(to_4326)
        env = cloned.GetEnvelope()  # minX, maxX, minY, maxY in EPSG:4326
        # Exclude non-continental overseas territories (French Guiana/DOM-TOM, Canary Islands, Svalbard/Jan Mayen)
        if c_up == 'FR' and (env[0] < -10.0 or env[2] < 38.0):
            continue
        if c_up == 'ES' and (env[0] < -11.0 or env[2] < 34.0):
            continue
        if c_up == 'NO' and env[3] > 72.0:
            continue

        cloned_valid = cloned.MakeValid()
        if cloned_valid is None or cloned_valid.IsEmpty():
            continue
        g_type = cloned_valid.GetGeometryType() & ~ogr.wkb25Bit
        if g_type == ogr.wkbPolygon:
            union_geom.AddGeometry(cloned_valid)
        elif g_type in (ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
            for i in range(cloned_valid.GetGeometryCount()):
                sub_g = cloned_valid.GetGeometryRef(i)
                if sub_g and (sub_g.GetGeometryType() & ~ogr.wkb25Bit) == ogr.wkbPolygon:
                    union_geom.AddGeometry(sub_g.Clone())
    ds = None
    if union_geom.IsEmpty():
        return None

    res_geom = union_geom.MakeValid()
    try:
        buf = res_geom.Buffer(0.0)
        if buf is not None and not buf.IsEmpty():
            res_geom = buf
    except Exception:
        pass

    if target_epsg != 4326:
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(target_epsg)
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct_out = osr.CoordinateTransformation(srs_4326, dst_srs)
        res_geom.Transform(ct_out)
        res_geom = res_geom.MakeValid()
    return res_geom


def get_country_geometry(country_code: str, target_epsg: int = 4326) -> Optional[ogr.Geometry]:
    """Dynamically locates and loads the NUTS shapefile for the given country code."""
    c_up = country_code.upper()
    candidates = [
        SHAPEFILES_DIR / c_up / f"NUTS2_{c_up}.shp",
        AUX_DIR / "NUTS2" / f"NUTS2_{c_up}.shp",
    ]
    country_dir = SHAPEFILES_DIR / c_up
    if country_dir.exists():
        candidates.extend(sorted(country_dir.glob("*.shp")))
    shp_path = next((p for p in candidates if p.exists()), None)
    if shp_path is None:
        logging.error(f"Shapefile for country {c_up} not found in {SHAPEFILES_DIR}")
        return None

    return load_shapefile_geometry_ogr(shp_path, target_epsg=target_epsg, country_code=c_up)


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


def _extract_active_s1_raster_footprint(s1_tif_path: Path, target_epsg: int = 4326) -> Optional[ogr.Geometry]:
    """Extracts the active (non-nodata) footprint polygon from a Sentinel-1 stack using a high-resolution overview."""
    try:
        ds = gdal.Open(str(s1_tif_path))
        if not ds:
            return None
        w = ds.RasterXSize
        h = ds.RasterYSize
        gt = ds.GetGeoTransform()
        proj_wkt = ds.GetProjection()
        nb = ds.RasterCount

        ov_w = max(512, min(1024, w // 20))
        ov_h = max(512, min(1024, h // 20))
        mask = np.zeros((ov_h, ov_w), dtype=np.uint8)
        for b_idx in sorted(set([1, max(1, nb // 2), nb])):
            arr = ds.GetRasterBand(b_idx).ReadAsArray(buf_xsize=ov_w, buf_ysize=ov_h)
            if arr is not None:
                mask |= ((arr != 0) & (~np.isnan(arr))).astype(np.uint8)
        ds = None

        if not np.any(mask):
            return None

        padded = np.pad(mask, 1, mode='edge')
        dilated = (
            padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1] |
            padded[1:-1, :-2] | padded[1:-1, 2:] | padded[:-2, :-2] |
            padded[:-2, 2:] | padded[2:, :-2] | padded[2:, 2:]
        )

        mem_drv = gdal.GetDriverByName('MEM')
        mem_ds = mem_drv.Create('', ov_w, ov_h, 1, gdal.GDT_Byte)
        ov_gt = (gt[0], gt[1] * w / ov_w, 0.0, gt[3], 0.0, gt[5] * h / ov_h)
        mem_ds.SetGeoTransform(ov_gt)
        mem_ds.SetProjection(proj_wkt)
        mem_ds.GetRasterBand(1).WriteArray(dilated)

        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(proj_wkt)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        ogr_drv = ogr.GetDriverByName('MEM') or ogr.GetDriverByName('Memory')
        ogr_ds = ogr_drv.CreateDataSource('mem_s1_fp')
        ogr_lyr = ogr_ds.CreateLayer('fp', src_srs, ogr.wkbPolygon)
        ogr_lyr.CreateField(ogr.FieldDefn('val', ogr.OFTInteger))
        gdal.Polygonize(mem_ds.GetRasterBand(1), mem_ds.GetRasterBand(1), ogr_lyr, 0, [])

        s1_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for feat in ogr_lyr:
            if feat.GetField('val') == 1:
                g = feat.GetGeometryRef()
                if g:
                    s1_geom.AddGeometry(g.Clone())
        ogr_ds = None
        mem_ds = None

        if s1_geom.IsEmpty():
            return None

        s1_geom = s1_geom.MakeValid().Buffer(200.0).SimplifyPreserveTopology(150.0).MakeValid().Buffer(0.0)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(target_epsg)
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not src_srs.IsSame(dst_srs):
            ct = osr.CoordinateTransformation(src_srs, dst_srs)
            s1_geom.Transform(ct)
            s1_geom = s1_geom.MakeValid()
        return s1_geom
    except Exception:
        return None


def _query_s1_orbit_swath_from_cdse(country_geom_4326: ogr.Geometry, orbit_num: int) -> Optional[ogr.Geometry]:
    """Queries CDSE OData for Sentinel-1 IW_GRDH_1S footprints of relativeOrbitNumber over the country."""
    try:
        env = country_geom_4326.GetEnvelope()
        wkt = (
            f"POLYGON(({env[0]:.4f} {env[2]:.4f}, {env[1]:.4f} {env[2]:.4f}, "
            f"{env[1]:.4f} {env[3]:.4f}, {env[0]:.4f} {env[3]:.4f}, {env[0]:.4f} {env[2]:.4f}))"
        )
        flt = (
            "Collection/Name eq 'SENTINEL-1' and "
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S') and "
            f"Attributes/OData.CSC.IntegerAttribute/any(att:att/Name eq 'relativeOrbitNumber' and att/OData.CSC.IntegerAttribute/Value eq {int(orbit_num)}) and "
            "ContentDate/Start gt 2024-06-01T00:00:00.000Z and ContentDate/Start lt 2024-06-25T00:00:00.000Z and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
        )
        url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter={urllib.parse.quote(flt)}&$top=200"
        req = urllib.request.Request(url, headers={"User-Agent": "AIML-CropMapper/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        swath_union = ogr.Geometry(ogr.wkbMultiPolygon)
        for item in data.get("value", []):
            fp = item.get("GeoFootprint")
            if fp:
                g = ogr.CreateGeometryFromJson(json.dumps(fp))
                if g:
                    g = g.MakeValid()
                    gt = g.GetGeometryType() & ~ogr.wkb25Bit
                    if gt == ogr.wkbPolygon:
                        swath_union.AddGeometry(g)
                    elif gt == ogr.wkbMultiPolygon:
                        for i in range(g.GetGeometryCount()):
                            swath_union.AddGeometry(g.GetGeometryRef(i).Clone())
        if not swath_union.IsEmpty():
            return swath_union.MakeValid().Buffer(0.0)
    except Exception:
        pass
    return None


def get_s1_orbit_extent_geometry(country_code: str, orbit_num: int, target_epsg: int = 4326) -> Optional[ogr.Geometry]:
    """
    Retrieves the active spatial footprint polygon for a specific S1 orbit clipped to the country's NUTS2 boundary.
    Uses the S1 stack raster's active pixels when available, or queries the S1 relative orbit swath from CDSE.
    """
    c_up = country_code.upper()
    country_geom = get_country_geometry(c_up, target_epsg=target_epsg)
    track_dir = BASE_DIR / c_up / f"orbit_{orbit_num}"
    candidate_dirs = [
        track_dir / "1_input_stacks",
        track_dir,
        track_dir / "processed_raster",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / c_up / f"orbit_{orbit_num}" / "1_input_stacks",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / c_up / f"orbit_{orbit_num}" / "processed_raster",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / c_up / f"orbit_{orbit_num}"
    ]

    for proc_dir in candidate_dirs:
        if proc_dir.exists():
            s1_tifs = (
                list(proc_dir.glob("*_VH_VV*.tif")) +
                list(proc_dir.glob("*Sigma0*.tif")) +
                list(proc_dir.glob("S1_*_stack*.tif"))
            )
            if s1_tifs:
                fp_geom = _extract_active_s1_raster_footprint(s1_tifs[0], target_epsg=target_epsg)
                if fp_geom is not None and not fp_geom.IsEmpty():
                    if country_geom is not None and not country_geom.IsEmpty():
                        inter = fp_geom.Intersection(country_geom)
                        if inter is not None and not inter.IsEmpty():
                            return inter.MakeValid()
                    return fp_geom

    # Fallback when S1 raster does not exist on disk yet: intersect S1 relative orbit swath with country geometry
    if country_geom is not None and not country_geom.IsEmpty():
        country_4326 = country_geom if target_epsg == 4326 else get_country_geometry(c_up, target_epsg=4326)
        if country_4326 is not None:
            swath_4326 = _query_s1_orbit_swath_from_cdse(country_4326, orbit_num)
            if swath_4326 is not None and not swath_4326.IsEmpty():
                inter_4326 = swath_4326.Intersection(country_4326)
                if inter_4326 is not None and not inter_4326.IsEmpty():
                    inter_valid = inter_4326.MakeValid()
                    if target_epsg != 4326:
                        srs_4326 = osr.SpatialReference()
                        srs_4326.ImportFromEPSG(4326)
                        srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                        dst_srs = osr.SpatialReference()
                        dst_srs.ImportFromEPSG(target_epsg)
                        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                        inter_valid.Transform(osr.CoordinateTransformation(srs_4326, dst_srs))
                    return inter_valid.MakeValid()
    return country_geom


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


def _safe_listdir(dir_path: Path, retries: int = 3) -> List[str]:
    """Safely list directory entries on FUSE/s3fs mounts with retries, avoiding per-entry stat calls."""
    for attempt in range(retries):
        try:
            return os.listdir(str(dir_path))
        except FileNotFoundError:
            return []
        except OSError:
            if attempt < retries - 1:
                time.sleep(0.4 * (attempt + 1))
    return []


def _resolve_safe_band_paths(safe_dir: Path) -> Dict[str, Path]:
    """
    Resolves source JP2/TIF paths for all 10 target S2 bands using at most 3 os.listdir calls
    (GRANULE, R10m, R20m) to prevent FUSE/s3fs metadata throttling.
    """
    band_map: Dict[str, Path] = {}
    g_dir = safe_dir / 'GRANULE'
    granules = [g for g in _safe_listdir(g_dir) if not g.startswith('.')]
    if not granules:
        return band_map

    for g_name in granules:
        img_data_dir = g_dir / g_name / 'IMG_DATA'
        r10m_dir = img_data_dir / 'R10m'
        r20m_dir = img_data_dir / 'R20m'

        r20m_entries = _safe_listdir(r20m_dir)
        r10m_entries = _safe_listdir(r10m_dir)
        if not r20m_entries and not r10m_entries:
            # Flat IMG_DATA fallback
            flat_entries = _safe_listdir(img_data_dir)
            for band in S2_BANDS_20M:
                matches = [f for f in flat_entries if f"_{band}" in f and (f.endswith('.jp2') or f.endswith('.tif'))]
                if matches:
                    band_map[band] = img_data_dir / matches[0]
            if band_map:
                return band_map
            continue

        for band in S2_BANDS_20M:
            # Prefer native 10m resolution from R10m for B02, B03, B04 when available
            if band in ('B02', 'B03', 'B04') and r10m_entries:
                m10 = [f for f in r10m_entries if f"_{band}_10m." in f and (f.endswith('.jp2') or f.endswith('.tif'))]
                if not m10:
                    m10 = [f for f in r10m_entries if f"_{band}" in f and (f.endswith('.jp2') or f.endswith('.tif'))]
                if m10:
                    band_map[band] = r10m_dir / m10[0]
                    continue

            if r20m_entries:
                m20 = [f for f in r20m_entries if f"_{band}_20m." in f and (f.endswith('.jp2') or f.endswith('.tif'))]
                if not m20:
                    m20 = [f for f in r20m_entries if f"_{band}" in f and (f.endswith('.jp2') or f.endswith('.tif'))]
                if m20:
                    band_map[band] = r20m_dir / m20[0]

        if 'B02' in band_map:
            return band_map

    return band_map


def find_b02_path_in_safe(safe_dir: Path) -> Optional[Path]:
    band_map = _resolve_safe_band_paths(safe_dir)
    return band_map.get('B02')


def detect_boa_add_offset(safe_dir: Path) -> int:
    """
    Detects ESA Processing Baseline 04.00+ / 05.xx BOA_ADD_OFFSET (typically -1000 DN).
    Since Jan 25, 2022 (N0400+), ESA adds +1000 DN to all L2A spectral reflectance bands.
    """
    try:
        m = re.search(r'_N(\d{4})_', safe_dir.name)
        if m:
            baseline_num = int(m.group(1))
            if baseline_num >= 400:
                return -1000
            else:
                return 0
        xml_path = safe_dir / "MTD_MSIL2A.xml"
        if xml_path.exists():
            tree = ET.parse(str(xml_path))
            for elem in tree.getroot().iter():
                if elem.tag.endswith("BOA_ADD_OFFSET") and elem.text:
                    return int(float(elem.text))
    except Exception:
        pass
    return 0


def convert_safe_to_geotiff(safe_dir: Path, output_dest_dir: Path) -> bool:
    try:
        # Fast local check: if all 10 bands already exist locally, skip touching the network drive completely
        if output_dest_dir.exists():
            all_bands_exist = True
            for band in S2_BANDS_20M:
                dst_tif = output_dest_dir / f"{safe_dir.stem}_{band}_20m.tif"
                if not (dst_tif.exists() and dst_tif.stat().st_size > 1024):
                    all_bands_exist = False
                    break
            if all_bands_exist:
                return True

        band_map = _resolve_safe_band_paths(safe_dir)
        if 'B02' not in band_map:
            return False

        output_dest_dir.mkdir(parents=True, exist_ok=True)
        boa_offset = detect_boa_add_offset(safe_dir)
        success = True

        for band in S2_BANDS_20M:
            band_dst_tif = output_dest_dir / f"{safe_dir.stem}_{band}_20m.tif"
            if band_dst_tif.exists() and band_dst_tif.stat().st_size > 1024:
                continue

            band_src = band_map.get(band)
            if band_src is None:
                success = False
                continue

            try:
                ds = gdal.Open(str(band_src))
                if ds is None:
                    success = False
                    continue
                if band != 'SCL' and boa_offset != 0:
                    # Harmonize ESA Processing Baseline 04.00+/05.xx BOA_ADD_OFFSET (-1000 DN)
                    arr = ds.GetRasterBand(1).ReadAsArray()
                    if arr is not None:
                        corr = np.where(arr > 0, np.clip(arr.astype(np.int32) + boa_offset, 1, 65535), 0).astype(np.uint16)
                        drv = gdal.GetDriverByName('GTiff')
                        out_ds = drv.Create(
                            str(band_dst_tif), ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_UInt16,
                            options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER', 'NUM_THREADS=2']
                        )
                        out_ds.SetGeoTransform(ds.GetGeoTransform())
                        out_ds.SetProjection(ds.GetProjection())
                        out_ds.SetMetadataItem("BOA_OFFSET_CORRECTED", "1")
                        out_b = out_ds.GetRasterBand(1)
                        out_b.SetNoDataValue(0)
                        out_b.WriteArray(corr)
                        out_ds = None
                    ds = None
                else:
                    options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER', 'NUM_THREADS=2'])
                    out_ds = gdal.Translate(str(band_dst_tif), ds, options=options)
                    if out_ds is not None and band != 'SCL':
                        out_ds.SetMetadataItem("BOA_OFFSET_CORRECTED", "1")
                    out_ds = None  # Explicitly close and flush to disk to avoid Windows handle locks
                    ds = None
                if not (band_dst_tif.exists() and band_dst_tif.stat().st_size > 1024):
                    success = False
            except Exception as e:
                logging.debug(f"Failed converting band {band_src.name}: {e}")
                if band_dst_tif.exists():
                    try:
                        band_dst_tif.unlink()
                    except Exception:
                        pass
                success = False

        if output_dest_dir.exists() and not any(output_dest_dir.glob("*.tif")):
            try:
                output_dest_dir.rmdir()
                if output_dest_dir.parent.exists() and not any(output_dest_dir.parent.iterdir()):
                    output_dest_dir.parent.rmdir()
            except Exception:
                pass
        return success
    except Exception as e:
        logging.warning(f"Error converting scene {safe_dir.name}: {e}")
        return False


def _query_cdse_odata_s2_scenes(
    repo_path: Path,
    start_date: datetime.date,
    end_date: datetime.date,
    target_tiles: Optional[List[str]] = None,
    max_cloud_cover: float = 80.0,
    target_geom: Optional[ogr.Geometry] = None
) -> Optional[List[Dict]]:
    """
    Fast pre-lookup of Sentinel-2 L2A scenes via the Copernicus Data Space (CDSE) OData Catalog API.
    Bypasses slow FUSE/s3fs directory scanning across 25,000+ global daily scenes on /eodata.
    Filters scenes by spatial intersection with target_geom (orbit or country footprint) when provided.
    """
    try:
        start_str = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
        end_str = end_date.strftime("%Y-%m-%dT23:59:59.999Z")

        filter_parts = [
            "Collection/Name eq 'SENTINEL-2'",
            "contains(Name,'MSIL2A')",
            f"ContentDate/Start ge {start_str}",
            f"ContentDate/Start le {end_str}",
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {float(max_cloud_cover):.2f})"
        ]

        if target_geom is not None and not target_geom.IsEmpty():
            env = target_geom.GetEnvelope()  # minX, maxX, minY, maxY in EPSG:4326
            wkt_poly = (
                f"POLYGON(({env[0]:.4f} {env[2]:.4f}, {env[1]:.4f} {env[2]:.4f}, "
                f"{env[1]:.4f} {env[3]:.4f}, {env[0]:.4f} {env[3]:.4f}, {env[0]:.4f} {env[2]:.4f}))"
            )
            filter_parts.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_poly}')")
        elif target_tiles:
            clean_tiles = sorted(set(t.upper().lstrip('T') for t in target_tiles))
            tile_clauses = [f"contains(Name,'_T{t}_')" for t in clean_tiles]
            filter_parts.append("(" + " or ".join(tile_clauses) + ")")

        allowed_tile_set = set(t.upper().lstrip('T') for t in target_tiles) if target_tiles else None

        flt = " and ".join(filter_parts)
        next_url = (
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
            f"$filter={urllib.parse.quote(flt)}&$expand=Attributes&$top=1000&$orderby=ContentDate/Start%20asc"
        )

        raw_items = []
        while next_url:
            req = urllib.request.Request(next_url, headers={"User-Agent": "AIML-CropMapper/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw_items.extend(data.get("value", []))
            next_url = data.get("@odata.nextLink")

        # Deduplicate reprocessed versions of the exact same (tile, sensing_timestamp, orbit), keeping latest baseline/processing date
        dedup_map: Dict[Tuple[str, str, str], Dict] = {}
        repo_is_eodata = str(repo_path).startswith("/eodata")

        for item in raw_items:
            name = item.get("Name", "")
            if not name:
                continue
            if not name.endswith(".SAFE"):
                name = f"{name}.SAFE"

            tile_name, acq_date = extract_tile_and_date_from_safe_name(name)
            if not (start_date <= acq_date <= end_date):
                continue

            tile_upper = tile_name.upper().lstrip('T')
            if allowed_tile_set is not None and tile_upper not in allowed_tile_set:
                continue

            if target_geom is not None and not target_geom.IsEmpty():
                fp = item.get("GeoFootprint")
                if fp:
                    try:
                        fp_geom = ogr.CreateGeometryFromJson(json.dumps(fp) if isinstance(fp, dict) else str(fp))
                        if fp_geom is not None and not fp_geom.Intersects(target_geom):
                            continue
                    except Exception:
                        pass

            cloud_pct = 0.0
            for att in item.get("Attributes", []):
                if att.get("Name") == "cloudCover" and att.get("Value") is not None:
                    try:
                        cloud_pct = float(att["Value"])
                    except (ValueError, TypeError):
                        pass
                    break

            if cloud_pct > max_cloud_cover:
                continue

            s3_path = item.get("S3Path", "")
            if repo_is_eodata and s3_path and s3_path.startswith("/eodata/"):
                safe_path = Path(s3_path)
            else:
                safe_path = repo_path / f"{acq_date.year}" / f"{acq_date.month:02d}" / f"{acq_date.day:02d}" / name

            m_sense = re.search(r'_(20\d{6}T\d{6})_', name)
            m_orb = re.search(r'_(R\d{3})_', name)
            sense_key = m_sense.group(1) if m_sense else str(acq_date)
            orb_key = m_orb.group(1) if m_orb else "R000"
            dedup_key = (tile_name, sense_key, orb_key)

            if dedup_key not in dedup_map or name > dedup_map[dedup_key]['title']:
                dedup_map[dedup_key] = {
                    'title': name,
                    'safe_path': safe_path,
                    'tile': tile_name,
                    'date': acq_date,
                    'cloud_cover': cloud_pct
                }

        matched = list(dedup_map.values())
        matched.sort(key=lambda x: (x['date'], x['tile'], x['title']))
        if matched:
            found_tiles = sorted(set(m['tile'] for m in matched))
            logging.info(f"OData spatial filter matched {len(found_tiles)} MGRS tiles: {found_tiles}")
        return matched
    except Exception as e:
        logging.warning(f"CDSE OData catalog pre-lookup failed ({e}); falling back to directory scan.")
        return None


def scan_creodias_for_dates(
    repo_path: Path,
    start_date: datetime.date,
    end_date: datetime.date,
    target_tiles: Optional[List[str]] = None,
    max_cloud_cover: float = 80.0,
    target_geom: Optional[ogr.Geometry] = None
) -> List[Dict]:
    tiles_key = tuple(sorted(set(t.upper().lstrip('T') for t in target_tiles))) if target_tiles else None
    geom_key = target_geom.ExportToWkt()[:120] if (target_geom is not None and not target_geom.IsEmpty()) else None
    cache_key = (str(repo_path), str(start_date), str(end_date), tiles_key, max_cloud_cover, geom_key)
    if cache_key in _GLOBAL_SCAN_CACHE:
        cached_all = _GLOBAL_SCAN_CACHE[cache_key]
        logging.info(f"Loaded {len(cached_all)} matching S2 SAFE scenes from memory cache.")
        return list(cached_all)

    logging.info(f"Scanning CREODIAS repository ({repo_path}) for dates {start_date} to {end_date} (Target MGRS tiles: {len(target_tiles) if target_tiles else 'AUTO-GEOM'})...")
    matched = []
    if not safe_exists(repo_path):
        logging.error(f"CREODIAS S2 path not found: {repo_path}")
        return matched

    # Fast OData catalog lookup (avoids scanning 25,000+ global daily folders over s3fs/FUSE)
    odata_matched = _query_cdse_odata_s2_scenes(
        repo_path=repo_path,
        start_date=start_date,
        end_date=end_date,
        target_tiles=target_tiles,
        max_cloud_cover=max_cloud_cover,
        target_geom=target_geom
    )
    if odata_matched is not None and len(odata_matched) > 0:
        _GLOBAL_SCAN_CACHE[cache_key] = list(odata_matched)
        logging.info(f"Found {len(odata_matched)} matching S2 SAFE scenes in CREODIAS repo via OData catalog.")
        return odata_matched

    target_tile_tags = [f"_T{t.upper().lstrip('T')}_" for t in target_tiles] if target_tiles else None

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

    orbit_geom = get_s1_orbit_extent_geometry(country_code, orbit_num, target_epsg=4326)
    if all_scenes is None:
        target_tiles = COUNTRY_MGRS_TILES.get(country_code, None)
        all_scenes = scan_creodias_for_dates(
            S2_REPO_PATH, start_date, end_date,
            target_tiles=target_tiles,
            max_cloud_cover=max_cloud_cover,
            target_geom=orbit_geom
        )

    if not all_scenes:
        logging.warning("No S2 products available to convert.")
        return

    # Filter scenes already converted
    scenes_to_process = []
    for sc in all_scenes:
        tile_upper = sc['tile'].upper()
        stem_name = sc['title'][:-5] if sc['title'].endswith('.SAFE') else sc['title']
        dest_prod_tif_dir = dest_track_s2 / f"{tile_upper}_tif" / sc['title']
        check_b02_stem = dest_prod_tif_dir / f"{stem_name}_B02_20m.tif"
        check_b02_full = dest_prod_tif_dir / f"{sc['title']}_B02_20m.tif"
        if not ((check_b02_stem.exists() and check_b02_stem.stat().st_size > 1024) or (check_b02_full.exists() and check_b02_full.stat().st_size > 1024)):
            # Also check fallback shared directory
            shared_dir = BASE_DIR / country_code / "S2" / f"{tile_upper}_tif" / sc['title']
            shared_check_stem = shared_dir / f"{stem_name}_B02_20m.tif"
            shared_check_full = shared_dir / f"{sc['title']}_B02_20m.tif"
            if not ((shared_check_stem.exists() and shared_check_stem.stat().st_size > 1024) or (shared_check_full.exists() and shared_check_full.stat().st_size > 1024)):
                scenes_to_process.append(sc)

    total_scenes = len(scenes_to_process)
    logging.info(f"Remaining S2 products to convert for {track_name}: {total_scenes} (out of {len(all_scenes)} total)")

    if total_scenes == 0:
        logging.info(f"All Sentinel-2 products for {track_name} are already converted to GeoTIFF!")
        return

    converted_count = 0
    success_count = 0
    lock = threading.Lock()

    def _worker_convert_creodias(sc: dict):
        nonlocal converted_count, success_count
        ok = False
        try:
            tile_upper = sc['tile'].upper()
            out_tile_dir = dest_track_s2 / f"{tile_upper}_tif" / sc['title']
            ok = convert_safe_to_geotiff(sc['safe_path'], out_tile_dir)
        except Exception as e:
            logging.debug(f"Skipping damaged scene {sc.get('title')}: {e}")
        finally:
            with lock:
                converted_count += 1
                if ok:
                    success_count += 1
                if converted_count % 10 == 0 or converted_count == total_scenes:
                    pct = (converted_count / total_scenes) * 100.0
                    logging.info(f"  [CREODIAS CONVERSION] {converted_count}/{total_scenes} products processed ({pct:.1f}%) | Succeeded: {success_count}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_worker_convert_creodias, scenes_to_process))

    if total_scenes > 0 and success_count == 0:
        err_msg = (
            f"Sentinel-2 extraction failed for track {track_name}: 0 out of {total_scenes} products succeeded. "
            f"The source repository at {S2_REPO_PATH} may be corrupt, inaccessible, or lacking readable product files."
        )
        logging.error(err_msg)
        raise RuntimeError(err_msg)

    logging.info(f"SUCCESS: Sentinel-2 extraction & conversion completed for track {track_name}! ({success_count}/{total_scenes} succeeded)")


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

    country_geom = get_country_geometry(country_code, target_epsg=4326)
    target_tiles = COUNTRY_MGRS_TILES.get(country_code, None)
    all_scenes = scan_creodias_for_dates(
        S2_REPO_PATH, start_date, end_date,
        target_tiles=target_tiles,
        max_cloud_cover=max_cloud_cover,
        target_geom=country_geom
    )

    if not all_scenes:
        logging.warning(f"No Sentinel-2 products found on CREODIAS for country {country_code}.")
        return

    # Filter scenes already converted in shared country pool
    scenes_to_process = []
    for sc in all_scenes:
        tile_upper = sc['tile'].upper()
        stem_name = sc['title'][:-5] if sc['title'].endswith('.SAFE') else sc['title']
        dest_prod_tif_dir = dest_country_s2 / f"{tile_upper}_tif" / sc['title']
        check_b02_stem = dest_prod_tif_dir / f"{stem_name}_B02_20m.tif"
        check_b02_full = dest_prod_tif_dir / f"{sc['title']}_B02_20m.tif"
        if not ((check_b02_stem.exists() and check_b02_stem.stat().st_size > 1024) or (check_b02_full.exists() and check_b02_full.stat().st_size > 1024)):
            scenes_to_process.append(sc)

    total_scenes = len(scenes_to_process)
    logging.info(f"Remaining S2 products to convert for country {country_code}: {total_scenes} (Already processed: {len(all_scenes) - total_scenes})")

    if total_scenes == 0:
        logging.info(f"All Sentinel-2 products for country {country_code} are already converted to GeoTIFF!")
        return

    converted_count = 0
    success_count = 0
    lock = threading.Lock()

    def _worker_convert_creodias(sc: dict):
        nonlocal converted_count, success_count
        ok = False
        try:
            tile_upper = sc['tile'].upper()
            out_tile_dir = dest_country_s2 / f"{tile_upper}_tif" / sc['title']
            ok = convert_safe_to_geotiff(sc['safe_path'], out_tile_dir)
        except Exception as e:
            logging.debug(f"Skipping damaged scene {sc.get('title')}: {e}")
        finally:
            with lock:
                converted_count += 1
                if ok:
                    success_count += 1
                if converted_count % 10 == 0 or converted_count == total_scenes:
                    pct = (converted_count / total_scenes) * 100.0
                    logging.info(f"  [CREODIAS CONVERSION] {converted_count}/{total_scenes} products processed ({pct:.1f}%) | Succeeded: {success_count}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_worker_convert_creodias, scenes_to_process))

    if total_scenes > 0 and success_count == 0:
        err_msg = (
            f"Sentinel-2 extraction failed for country {country_code}: 0 out of {total_scenes} products succeeded. "
            f"The source repository at {S2_REPO_PATH} may be corrupt, inaccessible, or lacking readable product files."
        )
        logging.error(err_msg)
        raise RuntimeError(err_msg)

    logging.info(f"SUCCESS: Sentinel-2 extraction & conversion completed for country {country_code}! ({success_count}/{total_scenes} succeeded)\n")


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
