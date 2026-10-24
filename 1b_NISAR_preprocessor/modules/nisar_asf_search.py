"""
================================================================================
AIML CropMapper Cloud - NISAR L-Band SAR Discovery & ASF DAAC Search Module
================================================================================
Handles geometry resolution for any European country (Eurostat GISCO NUTS) or
Sentinel-1 reference orbit track, and queries NASA ASF DAAC (via `asf_search`
or direct NASA ASF REST API fallback) and local `/eodata/NISAR` repositories
for NISAR Level-2 Geocoded Polarimetric Covariance (L2 GCOV) products.
================================================================================
"""

import datetime
import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from osgeo import gdal, ogr, osr

try:
    import asf_search as asf
    HAS_ASF = True
except ImportError:
    asf = None
    HAS_ASF = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = _PROJECT_ROOT / "workingDirs"
AUX_DIR = _PROJECT_ROOT / "auxiliary_files"


def get_s1_reference_grid(track_dir: Path) -> Optional[Dict]:
    """
    Inspects the Sentinel-1 reference stack in `<track_dir>/1_input_stacks` (or country orbit folders)
    to extract exact target projection, pixel resolution, dimensions, and WKT footprint in EPSG:4326.
    """
    candidates = []
    input_stacks = track_dir / "1_input_stacks"
    if input_stacks.exists():
        candidates.extend(sorted(input_stacks.glob("*_VH_VV*.tif")))
        candidates.extend(sorted(input_stacks.glob("*Sigma0*.tif")))

    if not candidates and track_dir.exists():
        for orb_dir in sorted(track_dir.glob("orbit_*")):
            st_dir = orb_dir / "1_input_stacks"
            if st_dir.exists():
                candidates.extend(sorted(st_dir.glob("*_VH_VV*.tif")))

    for tif_path in candidates:
        ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
        if ds is None:
            continue
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        gt = ds.GetGeoTransform()
        proj_wkt = ds.GetProjection()
        ds = None

        min_x = gt[0]
        max_y = gt[3]
        max_x = min_x + cols * gt[1]
        min_y = max_y + rows * gt[5]

        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(proj_wkt)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        epsg_code = int(src_srs.GetAttrValue("AUTHORITY", 1) or 3857)

        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(src_srs, wgs84_srs)

        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]:
            lon, lat, _ = transform.TransformPoint(x, y)
            ring.AddPoint_2D(lon, lat)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        return {
            "ref_raster": tif_path,
            "epsg": epsg_code,
            "extent": [min_x, min_y, max_x, max_y],
            "size": (cols, rows),
            "x_res": abs(gt[1]),
            "y_res": abs(gt[5]),
            "wkt_4326": poly.ExportToWkt(),
            "geom_4326": poly
        }
    return None


def load_country_or_belt_geometry(
    country_code: str,
    aux_dir: Path = AUX_DIR,
    custom_shp: Optional[Path] = None,
    target_epsg: int = 3857,
    pixel_res: float = 10.0
) -> Optional[Dict]:
    """
    Loads the AOI polygon for any European country (from Eurostat GISCO NUTS or country shapefile)
    or a custom belt shapefile, returning both the EPSG:4326 WKT for ASF search and the snapped
    target grid in `target_epsg` (default EPSG:3857 at 10.0 m).
    """
    c_upper = country_code.upper()
    shp_candidates = []
    if custom_shp and Path(custom_shp).exists():
        shp_candidates.append(Path(custom_shp))

    shp_candidates.extend([
        aux_dir / "shapefiles" / f"{c_upper}_borders_3857.shp",
        aux_dir / "shapefiles" / f"{c_upper}_3857.shp",
        aux_dir / "shapefiles" / f"{c_upper}.shp",
        aux_dir / "nuts" / "NUTS_RG_01M_2021_3857_LEVL_2.shp",
        aux_dir / "nuts" / "NUTS_RG_01M_2021_3857.shp"
    ])

    chosen_shp = next((p for p in shp_candidates if p.exists()), None)
    if chosen_shp is None:
        logging.error(f"Could not find boundary shapefile for {c_upper} in {aux_dir}")
        return None

    ds = ogr.Open(str(chosen_shp), 0)
    if ds is None:
        return None
    layer = ds.GetLayer()
    src_srs = layer.GetSpatialRef()
    if src_srs:
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    if "NUTS_RG" in chosen_shp.name:
        layer.SetAttributeFilter(f"CNTR_CODE = '{c_upper}'")

    union_geom = None
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is not None:
            g_clone = geom.Clone()
            union_geom = g_clone if union_geom is None else union_geom.Union(g_clone)
    ds = None

    if union_geom is None or union_geom.IsEmpty():
        logging.error(f"Empty geometry for {c_upper} in {chosen_shp.name}")
        return None

    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(target_epsg)
    tgt_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    geom_tgt = union_geom.Clone()
    if src_srs and not src_srs.IsSame(tgt_srs):
        tx_to_tgt = osr.CoordinateTransformation(src_srs, tgt_srs)
        geom_tgt.Transform(tx_to_tgt)

    env = geom_tgt.GetEnvelope()  # minX, maxX, minY, maxY
    min_x = float(int(env[0] // pixel_res) * pixel_res)
    max_y = float(int((env[3] + pixel_res - 1e-6) // pixel_res) * pixel_res)
    max_x = float(int((env[1] + pixel_res - 1e-6) // pixel_res) * pixel_res)
    min_y = float(int(env[2] // pixel_res) * pixel_res)
    cols = max(1, int(round((max_x - min_x) / pixel_res)))
    rows = max(1, int(round((max_y - min_y) / pixel_res)))

    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geom_4326 = union_geom.Clone()
    if src_srs and not src_srs.IsSame(wgs84_srs):
        tx_to_4326 = osr.CoordinateTransformation(src_srs, wgs84_srs)
        geom_4326.Transform(tx_to_4326)

    # Simplify complex country polygons for ASF HTTP query URL length limits while keeping convex/buffer safety
    search_geom = geom_4326.SimplifyPreserveTopology(0.02)
    if search_geom is None or search_geom.IsEmpty():
        search_geom = geom_4326.ConvexHull()
    search_geom.FlattenTo2D()

    return {
        "cutline_shp": chosen_shp,
        "country_code": c_upper,
        "epsg": target_epsg,
        "extent": [min_x, min_y, max_x, max_y],
        "size": (cols, rows),
        "x_res": pixel_res,
        "y_res": pixel_res,
        "wkt_4326": search_geom.ExportToWkt(),
        "geom_4326": geom_4326,
        "geom_target": geom_tgt
    }


class FindNisarASF:
    """
    Handles discovery and search of NISAR L2 GCOV products via:
      1. Local `/eodata/NISAR` or cache directory `.h5` archives (instant zero-download path).
      2. NASA ASF DAAC (`asf_search` library if installed, or direct ASF Search REST API fallback).
    """

    def __init__(self, geometry_directory: Optional[Path] = None):
        self.geometry_directory = Path(geometry_directory) if geometry_directory else None
        self.polygons = self._load_belt_geometries() if (self.geometry_directory and self.geometry_directory.exists()) else []

    def _load_belt_geometries(self) -> List[Dict]:
        polygons = []
        if not self.geometry_directory or not self.geometry_directory.exists():
            return polygons

        shp_files = sorted(list(self.geometry_directory.glob("*.shp")) + list(self.geometry_directory.glob("*.SHP")))
        seen = set()
        for shp_file in shp_files:
            if shp_file.name.lower() in seen:
                continue
            seen.add(shp_file.name.lower())
            ds = ogr.Open(str(shp_file), 0)
            if ds is None:
                continue
            layer = ds.GetLayer()
            src_srs = layer.GetSpatialRef()
            wgs84 = osr.SpatialReference()
            wgs84.ImportFromEPSG(4326)
            wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            tx = osr.CoordinateTransformation(src_srs, wgs84) if (src_srs and not src_srs.IsSame(wgs84)) else None

            union_geom = None
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom is not None:
                    g = geom.Clone()
                    if tx:
                        g.Transform(tx)
                    union_geom = g if union_geom is None else union_geom.Union(g)
            ds = None
            if union_geom is not None:
                union_geom.FlattenTo2D()
                polygons.append({
                    "file": shp_file,
                    "belt": shp_file.stem.upper(),
                    "wkt": union_geom.ExportToWkt()
                })
        return polygons

    @staticmethod
    def search_local_h5(
        local_dir: Path,
        start_date: datetime.date,
        end_date: datetime.date,
        product_type: str = "GCOV"
    ) -> List[Dict]:
        """Scans local repository (e.g., `/eodata/NISAR`) for pre-staged NISAR `.h5` granules."""
        if not local_dir or not local_dir.exists():
            return []
        found = []
        for h5_file in sorted(local_dir.rglob(f"*{product_type}*.h5")):
            title = h5_file.stem
            if "_SHSH_" in title or "_SVSV_" in title:
                continue
            parts = title.split("_")
            start_str = None
            path_num = None
            if len(parts) >= 12 and parts[5].isdigit():
                path_num = int(parts[5])
                for p in parts:
                    if len(p) == 15 and "T" in p and p[:8].isdigit():
                        start_str = f"{p[:4]}-{p[4:6]}-{p[6:8]}T{p[9:11]}:{p[11:13]}:{p[13:15]}Z"
                        break
            if start_str:
                d_val = datetime.datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                if not (start_date <= d_val <= end_date):
                    continue
            else:
                start_str = datetime.datetime.fromtimestamp(h5_file.stat().st_mtime).strftime("%Y-%m-%dT00:00:00Z")

            found.append({
                "title": title,
                "start_time": start_str,
                "stop_time": start_str,
                "url": str(h5_file),
                "local_h5": h5_file,
                "bytes": h5_file.stat().st_size,
                "flight_direction": "DESCENDING",
                "path_number": path_num,
                "asf_product": None
            })
        if found:
            logging.info(f"Discovered {len(found)} local NISAR {product_type} HDF5 granules in {local_dir}")
        return found

    def search_products(
        self,
        geometry_wkt: str,
        start_date: datetime.date,
        end_date: datetime.date,
        product_type: str = "GCOV",
        flight_direction: Optional[str] = "DESCENDING",
        session=None,
        local_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Queries local `/eodata/NISAR` repository first, then queries NASA ASF DAAC
        (via `asf_search` or direct ASF REST GeoJSON endpoint fallback).
        """
        if local_dir and Path(local_dir).exists():
            local_hits = self.search_local_h5(Path(local_dir), start_date, end_date, product_type)
            if local_hits:
                return local_hits

        start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_str = end_date.strftime("%Y-%m-%dT23:59:59Z")
        logging.info(f"Searching NASA ASF DAAC for NISAR {product_type}: {start_date} to {end_date} ({flight_direction})")

        if HAS_ASF:
            try:
                search_opts = {
                    "platform": getattr(asf.PLATFORM, "NISAR", "NISAR"),
                    "intersectsWith": geometry_wkt,
                    "start": start_str,
                    "end": end_str
                }
                if product_type:
                    search_opts["processingLevel"] = product_type
                if flight_direction and flight_direction.upper() != "ALL":
                    search_opts["flightDirection"] = flight_direction.upper()

                if session:
                    opts = asf.ASFSearchOptions(session=session)
                    results = asf.geo_search(**search_opts, opts=opts)
                else:
                    results = asf.geo_search(**search_opts)

                products = []
                for res in results:
                    props = res.properties
                    title = props.get("sceneName") or props.get("fileID") or props.get("fileName") or ""
                    if "_SHSH_" in title or "_SVSV_" in title:
                        continue
                    products.append({
                        "title": title,
                        "start_time": props.get("startTime"),
                        "stop_time": props.get("stopTime"),
                        "url": props.get("url"),
                        "bytes": props.get("bytes", 0),
                        "flight_direction": props.get("flightDirection"),
                        "path_number": props.get("pathNumber"),
                        "geometry": getattr(res, "geometry", None),
                        "asf_product": res
                    })
                logging.info(f"Found {len(products)} NISAR granules via asf_search.")
                return products
            except Exception as e:
                logging.warning(f"asf_search query failed ({e}); falling back to direct ASF Search REST API...")

        # Direct NASA ASF Search REST API fallback (zero external package dependencies)
        try:
            params = {
                "platform": "NISAR",
                "processingLevel": product_type,
                "intersectsWith": geometry_wkt,
                "start": start_str,
                "end": end_str,
                "output": "geojson"
            }
            if flight_direction and flight_direction.upper() != "ALL":
                params["flightDirection"] = flight_direction.upper()

            url = "https://api.daac.asf.alaska.edu/services/search/param?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": "AIML-CropMapper-Cloud/2.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            features = data.get("features", [])
            products = []
            for feat in features:
                props = feat.get("properties", {})
                title = props.get("sceneName") or props.get("fileID") or props.get("fileName") or ""
                if "_SHSH_" in title or "_SVSV_" in title:
                    continue
                products.append({
                    "title": title,
                    "start_time": props.get("startTime"),
                    "stop_time": props.get("stopTime"),
                    "url": props.get("url"),
                    "bytes": props.get("bytes", 0),
                    "flight_direction": props.get("flightDirection"),
                    "path_number": props.get("pathNumber"),
                    "geometry": feat.get("geometry"),
                    "asf_product": None
                })
            logging.info(f"Found {len(products)} NISAR granules via NASA ASF REST API.")
            return products
        except Exception as e:
            logging.error(f"Error querying NASA ASF REST API for NISAR: {e}")
            return []
