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
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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


def get_country_shapefile(country_code: str, aux_dir: Path = AUX_DIR) -> Optional[Path]:
    """
    Locates the official national administrative shapefile (NUTS2) for any European country,
    searching national NUTS directories and automatically falling back to Eurostat GISCO
    NUTS_RG_01M_2021_3857.geojson if an individual shapefile does not exist.
    """
    c_up = country_code.upper()
    candidates = [
        aux_dir / "shapefiles_national_NUTS2" / c_up / f"NUTS2_{c_up}.shp",
        aux_dir / "shapefiles_national_NUTS2" / f"NUTS2_{c_up}.shp",
        aux_dir / "shapefiles_national_NUTS2" / f"{c_up}.shp",
        aux_dir / "shapefiles_nuts" / c_up / f"NUTS2_{c_up}.shp",
        aux_dir / "shapefiles_nuts" / f"NUTS2_{c_up}.shp",
        aux_dir / "shapefiles_nuts" / c_up / f"{c_up}.shp",
        aux_dir / "shapefiles_nuts" / f"{c_up}.shp",
        aux_dir / "shapefiles" / c_up / f"NUTS2_{c_up}.shp",
        aux_dir / "shapefiles" / f"NUTS2_{c_up}.shp",
        aux_dir / "shapefiles" / f"{c_up}_borders_3857.shp",
        aux_dir / "shapefiles" / f"{c_up}_3857.shp",
        aux_dir / "shapefiles" / f"{c_up}.shp",
        aux_dir / "nuts" / f"NUTS2_{c_up}.shp",
        aux_dir / "nuts" / "NUTS_RG_01M_2021_3857_LEVL_2.shp",
        aux_dir / "nuts" / "NUTS_RG_01M_2021_3857.shp"
    ]

    for p in candidates:
        if p.exists():
            return p

    # Fallback: extract national boundary from Eurostat GISCO GeoJSON
    for gisco_candidate in [
        aux_dir / "shapefiles_nuts" / "NUTS_RG_01M_2021_3857.geojson",
        aux_dir / "nuts" / "NUTS_RG_01M_2021_3857.geojson",
        aux_dir / "shapefiles" / "NUTS_RG_01M_2021_3857.geojson"
    ]:
        if gisco_candidate.exists():
            try:
                target_dir = aux_dir / "shapefiles_nuts" / c_up
                target_dir.mkdir(parents=True, exist_ok=True)
                out_shp = target_dir / f"NUTS2_{c_up}.shp"
                src_ds = ogr.Open(str(gisco_candidate), 0)
                if src_ds is not None:
                    src_lyr = src_ds.GetLayer()
                    src_lyr.SetAttributeFilter(f"CNTR_CODE = '{c_up}'")
                    drv = ogr.GetDriverByName("ESRI Shapefile")
                    if out_shp.exists():
                        drv.DeleteDataSource(str(out_shp))
                    dst_ds = drv.CreateDataSource(str(out_shp))
                    dst_lyr = dst_ds.CreateLayer(f"NUTS2_{c_up}", src_lyr.GetSpatialRef(), ogr.wkbMultiPolygon)
                    cnt = 0
                    for feat in src_lyr:
                        geom = feat.GetGeometryRef()
                        if geom is not None:
                            out_feat = ogr.Feature(dst_lyr.GetLayerDefn())
                            out_feat.SetGeometry(geom.Clone())
                            dst_lyr.CreateFeature(out_feat)
                            out_feat = None
                            cnt += 1
                    dst_ds = None
                    src_ds = None
                    if cnt > 0 and out_shp.exists():
                        logging.info(f"Extracted country shapefile for {c_up} from GISCO GeoJSON: {out_shp}")
                        return out_shp
            except Exception as e:
                logging.warning(f"Could not extract country boundary from GISCO GeoJSON for {c_up}: {e}")

    return None


def get_country_geometry(country_code: str, target_epsg: int = 4326, aux_dir: Path = AUX_DIR) -> Optional[ogr.Geometry]:
    """Loads and unifies the country geometry from its administrative NUTS2 boundary shapefile."""
    shp_path = get_country_shapefile(country_code, aux_dir=aux_dir)
    if not shp_path:
        return None

    ds = ogr.Open(str(shp_path), 0)
    if ds is None:
        return None
    layer = ds.GetLayer()
    src_srs = layer.GetSpatialRef()
    if src_srs:
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    srs_4326 = osr.SpatialReference()
    srs_4326.ImportFromEPSG(4326)
    srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_4326 = osr.CoordinateTransformation(src_srs, srs_4326) if (src_srs and not src_srs.IsSame(srs_4326)) else None

    c_up = country_code.upper()
    if "NUTS_RG" in shp_path.name:
        layer.SetAttributeFilter(f"CNTR_CODE = '{c_up}'")

    union_geom = None
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is not None:
            cloned = geom.Clone()
            if to_4326:
                cloned.Transform(to_4326)
            env = cloned.GetEnvelope()  # minX, maxX, minY, maxY in EPSG:4326
            # Exclude non-continental overseas territories (French Guiana/DOM-TOM, Canary Islands, Svalbard/Jan Mayen)
            if c_up == "FR" and (env[0] < -10.0 or env[2] < 38.0):
                continue
            if c_up == "ES" and (env[0] < -11.0 or env[2] < 34.0):
                continue
            if c_up == "NO" and env[3] > 72.0:
                continue

            g_clone = geom.Clone()
            union_geom = g_clone if union_geom is None else union_geom.Union(g_clone)
    ds = None

    if union_geom is None or union_geom.IsEmpty():
        return None

    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(target_epsg)
    tgt_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    if src_srs and not src_srs.IsSame(tgt_srs):
        tx = osr.CoordinateTransformation(src_srs, tgt_srs)
        union_geom.Transform(tx)

    union_geom = union_geom.MakeValid()
    try:
        buf = union_geom.Buffer(0.0)
        if buf is not None and not buf.IsEmpty():
            union_geom = buf
    except Exception:
        pass
    union_geom.FlattenTo2D()
    return union_geom


def get_s1_reference_grid(track_dir: Path) -> Optional[Dict]:
    """
    Inspects the Sentinel-1 reference stack in `<track_dir>/1_input_stacks` (or associated directories)
    to extract exact target projection, pixel resolution, dimensions, GeoTransform, and WKT footprint.
    """
    t_resolved = Path(track_dir).resolve()
    base_resolved = BASE_DIR.resolve()
    is_under_base = base_resolved in t_resolved.parents

    candidate_dirs = [
        t_resolved / "1_input_stacks",
        t_resolved / "wrapped",
        t_resolved,
        t_resolved / "processed_raster",
        t_resolved / "S1_final_preprocessing",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / t_resolved.relative_to(base_resolved) / "1_input_stacks" if is_under_base else None,
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / t_resolved.relative_to(base_resolved) / "processed_raster" if is_under_base else None,
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / t_resolved.relative_to(base_resolved) if is_under_base else None
    ]
    if t_resolved.exists():
        candidate_dirs.extend(sorted(list(t_resolved.glob("orbit_*/1_input_stacks"))))
        candidate_dirs.extend(sorted(list(t_resolved.glob("orbit_*/wrapped"))))
        candidate_dirs.extend(sorted(list(t_resolved.glob("orbit_*"))))

    candidates = []
    for proc_dir in candidate_dirs:
        if proc_dir and proc_dir.exists():
            candidates.extend(sorted(proc_dir.glob("*_VH_VV*.tif")))
            candidates.extend(sorted(proc_dir.glob("*Sigma0*.tif")))
            candidates.extend(sorted(proc_dir.glob("S1_*_stack*.tif")))

    # Exclude temporary, overview, or mask files
    candidates = [
        f for f in candidates
        if f.is_file() and not f.name.endswith(".tmp") and not f.name.endswith(".tmp.tif")
        and not f.name.endswith(".ovr") and not f.name.endswith(".xml") and not f.name.startswith("_")
    ]

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
            "gt": gt,
            "proj": proj_wkt,
            "wkt_4326": poly.ExportToWkt(),
            "geom_4326": poly
        }
    return None


def _extract_active_s1_raster_footprint(s1_tif_path: Path, target_epsg: int = 4326) -> Optional[ogr.Geometry]:
    """
    Extracts the active (non-nodata) footprint polygon from an operational Sentinel-1 stack
    using a fast 256x256 subsampled overview.
    """
    try:
        ds = gdal.Open(str(s1_tif_path), gdal.GA_ReadOnly)
        if not ds:
            return None
        w = ds.RasterXSize
        h = ds.RasterYSize
        gt = ds.GetGeoTransform()
        proj_wkt = ds.GetProjection()
        nb = ds.RasterCount

        ov_w = max(128, min(256, w // 20))
        ov_h = max(128, min(256, h // 20))
        mask = np.zeros((ov_h, ov_w), dtype=np.uint8)
        for b_idx in sorted(set([1, max(1, nb // 2), nb])):
            arr = ds.GetRasterBand(b_idx).ReadAsArray(buf_xsize=ov_w, buf_ysize=ov_h)
            if arr is not None:
                mask |= ((arr != 0) & (~np.isnan(arr))).astype(np.uint8)
        ds = None

        if not np.any(mask):
            return None

        padded = np.pad(mask, 1, mode="edge")
        dilated = (
            padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1] |
            padded[1:-1, :-2] | padded[1:-1, 2:] | padded[:-2, :-2] |
            padded[:-2, 2:] | padded[2:, :-2] | padded[2:, 2:]
        )

        mem_drv = gdal.GetDriverByName("MEM")
        mem_ds = mem_drv.Create("", ov_w, ov_h, 1, gdal.GDT_Byte)
        ov_gt = (gt[0], gt[1] * w / ov_w, 0.0, gt[3], 0.0, gt[5] * h / ov_h)
        mem_ds.SetGeoTransform(ov_gt)
        mem_ds.SetProjection(proj_wkt)
        mem_ds.GetRasterBand(1).WriteArray(dilated)

        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(proj_wkt)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        ogr_drv = ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")
        ogr_ds = ogr_drv.CreateDataSource("mem_s1_fp")
        ogr_lyr = ogr_ds.CreateLayer("fp", src_srs, ogr.wkbPolygon)
        ogr_lyr.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))
        gdal.Polygonize(mem_ds.GetRasterBand(1), mem_ds.GetRasterBand(1), ogr_lyr, 0, [])

        s1_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for feat in ogr_lyr:
            if feat.GetField("val") == 1:
                g = feat.GetGeometryRef()
                if g:
                    g_type = g.GetGeometryType() & ~ogr.wkb25Bit
                    if g_type == ogr.wkbPolygon:
                        s1_geom.AddGeometry(g.Clone())
                    elif g_type in (ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
                        for i in range(g.GetGeometryCount()):
                            sub_g = g.GetGeometryRef(i)
                            if sub_g and (sub_g.GetGeometryType() & ~ogr.wkb25Bit) == ogr.wkbPolygon:
                                s1_geom.AddGeometry(sub_g.Clone())
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
    except Exception as e:
        logging.warning(f"Could not extract active S1 raster footprint from {s1_tif_path.name}: {e}")
        return None


def _query_s1_orbit_swath_from_cdse(
    country_geom_4326: ogr.Geometry,
    orbit_num: int,
    country_code: Optional[str] = None
) -> Optional[ogr.Geometry]:
    """
    Queries CDSE OData for Sentinel-1 IW_GRDH_1S footprints of relativeOrbitNumber over the country envelope.
    Caches the swath footprint locally partitioned by country code to guarantee fast and offline execution
    without multi-country swath collisions.
    """
    cache_dir = AUX_DIR / "orbit_swaths"
    c_suffix = f"_{country_code.upper()}" if country_code else ""
    cache_file = cache_dir / f"s1_orbit_{orbit_num}{c_suffix}.geojson"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                geo_data = json.load(f)
            cached_geom = ogr.CreateGeometryFromJson(json.dumps(geo_data))
            if cached_geom and not cached_geom.IsEmpty():
                if country_geom_4326 is not None:
                    inter_chk = cached_geom.Intersection(country_geom_4326)
                    if inter_chk is not None and not inter_chk.IsEmpty() and inter_chk.GetArea() > 0.005:
                        return cached_geom.MakeValid()
                else:
                    return cached_geom.MakeValid()
        except Exception:
            pass

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
        req = urllib.request.Request(url, headers={"User-Agent": "AIML-CropMapper-NISAR/2.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        swath_union = ogr.Geometry(ogr.wkbMultiPolygon)
        for item in data.get("value", []):
            fp = item.get("GeoFootprint")
            if fp:
                g = ogr.CreateGeometryFromJson(json.dumps(fp))
                if g and not g.IsEmpty():
                    g_valid = g.MakeValid()
                    swath_union = swath_union.Union(g_valid)
        if not swath_union.IsEmpty():
            valid_swath = swath_union.MakeValid().Buffer(0.0)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(valid_swath.ExportToJson())
            except Exception:
                pass
            return valid_swath
    except Exception as e:
        logging.warning(f"CDSE Sentinel-1 orbit {orbit_num} swath query warning: {e}")
    return None


def get_s1_orbit_extent_geometry(
    country_code: str,
    orbit_num: int,
    target_epsg: int = 4326,
    aux_dir: Path = AUX_DIR
) -> Optional[ogr.Geometry]:
    """
    Retrieves the active spatial footprint polygon for a specific Sentinel-1 relative orbit
    clipped to the country's official administrative boundary (NUTS2).
    Uses the active pixels of an on-disk S1 stack when available, or intersects the CDSE
    relative orbit swath with the country boundary.
    """
    c_up = country_code.upper()
    country_geom = get_country_geometry(c_up, target_epsg=target_epsg, aux_dir=aux_dir)
    track_dir = BASE_DIR / c_up / f"orbit_{orbit_num}"
    t_resolved = track_dir.resolve()
    base_resolved = BASE_DIR.resolve()
    is_under_base = base_resolved in t_resolved.parents
    candidate_dirs = [
        t_resolved / "1_input_stacks",
        t_resolved,
        t_resolved / "processed_raster",
        t_resolved / "wrapped",
        t_resolved / "S1_final_preprocessing",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / t_resolved.relative_to(base_resolved) / "1_input_stacks" if is_under_base else None,
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / t_resolved.relative_to(base_resolved) / "processed_raster" if is_under_base else None,
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / t_resolved.relative_to(base_resolved) if is_under_base else None
    ]

    for proc_dir in candidate_dirs:
        if proc_dir and proc_dir.exists():
            s1_tifs = [
                f for f in (
                    sorted(proc_dir.glob("*_VH_VV*.tif")) +
                    sorted(proc_dir.glob("*Sigma0*.tif")) +
                    sorted(proc_dir.glob("S1_*_stack*.tif"))
                )
                if f.is_file() and not f.name.endswith(".tmp") and not f.name.endswith(".tmp.tif")
                and not f.name.endswith(".ovr") and not f.name.endswith(".xml") and not f.name.startswith("_")
            ]
            if s1_tifs:
                fp_geom = _extract_active_s1_raster_footprint(s1_tifs[0], target_epsg=target_epsg)
                if fp_geom is not None and not fp_geom.IsEmpty():
                    if country_geom is not None and not country_geom.IsEmpty():
                        inter = fp_geom.Intersection(country_geom)
                        if inter is not None and not inter.IsEmpty():
                            return inter.MakeValid().Buffer(0.0)
                    return fp_geom

    # Fallback: intersect S1 relative orbit swath with country boundary
    if country_geom is not None and not country_geom.IsEmpty():
        country_4326 = country_geom if target_epsg == 4326 else get_country_geometry(c_up, target_epsg=4326, aux_dir=aux_dir)
        if country_4326 is not None:
            swath_4326 = _query_s1_orbit_swath_from_cdse(country_4326, orbit_num, country_code=c_up)
            if swath_4326 is not None and not swath_4326.IsEmpty():
                inter_4326 = swath_4326.Intersection(country_4326)
                if inter_4326 is not None and not inter_4326.IsEmpty():
                    inter_valid = inter_4326.MakeValid().Buffer(0.0)
                    if target_epsg != 4326:
                        srs_4326 = osr.SpatialReference()
                        srs_4326.ImportFromEPSG(4326)
                        srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                        dst_srs = osr.SpatialReference()
                        dst_srs.ImportFromEPSG(target_epsg)
                        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                        inter_valid.Transform(osr.CoordinateTransformation(srs_4326, dst_srs))
                    return inter_valid.MakeValid().Buffer(0.0)

    return country_geom


def prepare_orbit_cutline_shapefile(
    country_code: str,
    norm_track: str,
    out_dir: Path,
    target_epsg: int = 3857,
    aux_dir: Path = AUX_DIR
) -> Tuple[Optional[Path], Optional[ogr.Geometry]]:
    """
    Creates a dissolved, topologically valid single-feature cutline shapefile in target_epsg
    representing the intersection of the active Sentinel-1 orbit footprint and the country's NUTS2 boundary.
    Prevents GEOS TopologyException during gdal.Warp and clips each orbit's NISAR mosaic to its exact S1 extent.
    """
    try:
        orbit_match = re.search(r"orbit_(\d+)", norm_track)
        if orbit_match:
            orbit_num = int(orbit_match.group(1))
            clip_geom = get_s1_orbit_extent_geometry(country_code, orbit_num, target_epsg=target_epsg, aux_dir=aux_dir)
        else:
            clip_geom = get_country_geometry(country_code, target_epsg=target_epsg, aux_dir=aux_dir)

        if clip_geom is None or clip_geom.IsEmpty():
            fallback_shp = get_country_shapefile(country_code, aux_dir=aux_dir)
            return fallback_shp, None

        clip_geom = clip_geom.MakeValid()
        try:
            buf = clip_geom.Buffer(0.0)
            if buf is not None and not buf.IsEmpty():
                clip_geom = buf
        except Exception:
            pass
        clip_geom = ogr.ForceToMultiPolygon(clip_geom)
        clip_geom.FlattenTo2D()

        out_dir.mkdir(parents=True, exist_ok=True)
        cut_shp = out_dir / "_clean_orbit_cutline.shp"
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if cut_shp.exists():
            try:
                drv.DeleteDataSource(str(cut_shp))
            except Exception:
                pass

        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(target_epsg)
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        ds_cut = drv.CreateDataSource(str(cut_shp))
        if ds_cut is None:
            return get_country_shapefile(country_code, aux_dir=aux_dir), clip_geom

        lyr_cut = ds_cut.CreateLayer("cutline", dst_srs, ogr.wkbMultiPolygon)
        feat = ogr.Feature(lyr_cut.GetLayerDefn())
        feat.SetGeometry(clip_geom)
        lyr_cut.CreateFeature(feat)
        feat = None
        ds_cut = None

        return cut_shp, clip_geom
    except Exception as e:
        logging.warning(f"Could not build dissolved orbit cutline for {norm_track} ({e}); falling back to country shapefile.")
        return get_country_shapefile(country_code, aux_dir=aux_dir), None


def discover_country_orbits(
    country_code: str,
    base_dir: Path = BASE_DIR,
    aux_dir: Path = AUX_DIR
) -> List[int]:
    """
    Finds existing on-disk orbit folders or looks up the country's Sentinel-1 orbit registry
    across `country_optimal_greedy_orbits_clean.json` and `country_orbits_complete.json`
    for all 38 European countries in the catalog.
    """
    c_up = country_code.upper()
    found = set()

    # 1. On-disk working directories
    for b in [base_dir, Path(r"D:/AIML_CropMapper_Cloud/workingDir")]:
        c_dir = b / c_up
        if c_dir.exists():
            for d in c_dir.glob("orbit_*"):
                m = re.search(r"orbit_(\d+)", d.name)
                if m and d.is_dir():
                    found.add(int(m.group(1)))
    if found:
        return sorted(list(found))

    # 2. Optimal greedy orbits registry (clean)
    for greedy_name in ["country_optimal_greedy_orbits_clean.json", "country_optimal_greedy_orbits.json"]:
        greedy_path = aux_dir / greedy_name
        if greedy_path.exists():
            try:
                with open(greedy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if c_up in data:
                    val = data[c_up]
                    if isinstance(val, dict) and val.get("orbits"):
                        return [int(o) for o in val["orbits"]]
                    elif isinstance(val, list):
                        return [int(o) for o in val]
            except Exception:
                pass

    # 3. Complete country orbits registry
    complete_path = aux_dir / "country_orbits_complete.json"
    if complete_path.exists():
        try:
            with open(complete_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if c_up in data and data[c_up]:
                return [int(o) for o in data[c_up]]
        except Exception:
            pass

    return []


def resolve_target_grid_and_cutline(
    country_code: str,
    orbit_num: Optional[int],
    track_dir: Path,
    target_epsg: int = 3857,
    pixel_res: float = 10.0,
    aux_dir: Path = AUX_DIR
) -> Tuple[Dict, Optional[Path]]:
    """
    Resolves the target pixel grid (bounding box, width, height, resolution, and GeoTransform)
    and the cutline shapefile representing the Sentinel-1 relative orbit footprint intersected
    with the official NUTS2 country boundary.
    """
    c_up = country_code.upper()
    norm_track = f"{c_up}/orbit_{orbit_num}" if orbit_num is not None else c_up

    cutline_shp, clip_geom = prepare_orbit_cutline_shapefile(
        country_code=c_up,
        norm_track=norm_track,
        out_dir=track_dir,
        target_epsg=target_epsg,
        aux_dir=aux_dir
    )

    s1_grid = get_s1_reference_grid(track_dir)
    if s1_grid is not None:
        logging.info(
            f"Matched Sentinel-1 reference grid: {s1_grid['ref_raster'].name} "
            f"({s1_grid['size'][0]}x{s1_grid['size'][1]} px at {s1_grid['x_res']:.1f}m, EPSG:{s1_grid['epsg']})"
        )
        if clip_geom is not None and not clip_geom.IsEmpty():
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            src_srs = osr.SpatialReference()
            src_srs.ImportFromEPSG(target_epsg)
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            geom_4326 = clip_geom.Clone()
            geom_4326.Transform(osr.CoordinateTransformation(src_srs, wgs84_srs))
            search_geom = geom_4326.SimplifyPreserveTopology(0.02)
            if search_geom is None or search_geom.IsEmpty():
                search_geom = geom_4326.ConvexHull()
            search_geom.FlattenTo2D()
            s1_grid["wkt_4326"] = search_geom.ExportToWkt()
            s1_grid["geom_4326"] = geom_4326
            s1_grid["geom_target"] = clip_geom
        else:
            s1_grid["geom_target"] = None
        return s1_grid, cutline_shp

    if clip_geom is not None and not clip_geom.IsEmpty():
        env = clip_geom.GetEnvelope()  # minX, maxX, minY, maxY in target_epsg
        min_x = float(int(env[0] // pixel_res) * pixel_res)
        min_y = float(int(env[2] // pixel_res) * pixel_res)
        max_x = float(int((env[1] + pixel_res - 1e-6) // pixel_res) * pixel_res)
        max_y = float(int((env[3] + pixel_res - 1e-6) // pixel_res) * pixel_res)
        cols = max(1, int(round((max_x - min_x) / pixel_res)))
        rows = max(1, int(round((max_y - min_y) / pixel_res)))

        gt = (min_x, pixel_res, 0.0, max_y, 0.0, -pixel_res)
        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromEPSG(target_epsg)
        tgt_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        proj_wkt = tgt_srs.ExportToWkt()

        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geom_4326 = clip_geom.Clone()
        geom_4326.Transform(osr.CoordinateTransformation(tgt_srs, wgs84_srs))
        search_geom = geom_4326.SimplifyPreserveTopology(0.02)
        if search_geom is None or search_geom.IsEmpty():
            search_geom = geom_4326.ConvexHull()
        search_geom.FlattenTo2D()

        grid_info = {
            "ref_raster": None,
            "epsg": target_epsg,
            "extent": [min_x, min_y, max_x, max_y],
            "size": (cols, rows),
            "x_res": pixel_res,
            "y_res": pixel_res,
            "gt": gt,
            "proj": proj_wkt,
            "wkt_4326": search_geom.ExportToWkt(),
            "geom_4326": geom_4326,
            "geom_target": clip_geom
        }
        logging.info(f"Generated snapped 10m orbit geometry for {norm_track}: {cols}x{rows} px, Bounds: {grid_info['extent']}")
        return grid_info, cutline_shp

    country_info = load_country_or_belt_geometry(c_up, aux_dir=aux_dir, target_epsg=target_epsg, pixel_res=pixel_res)
    if country_info is not None:
        return country_info, country_info.get("cutline_shp")

    raise FileNotFoundError(f"Could not resolve S1 reference grid or country shapefile for {norm_track}")


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
    chosen_shp = Path(custom_shp) if (custom_shp and Path(custom_shp).exists()) else get_country_shapefile(c_upper, aux_dir=aux_dir)

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

    srs_4326 = osr.SpatialReference()
    srs_4326.ImportFromEPSG(4326)
    srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_4326 = osr.CoordinateTransformation(src_srs, srs_4326) if (src_srs and not src_srs.IsSame(srs_4326)) else None

    if "NUTS_RG" in chosen_shp.name:
        layer.SetAttributeFilter(f"CNTR_CODE = '{c_upper}'")

    union_geom = None
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is not None:
            cloned = geom.Clone()
            if to_4326:
                cloned.Transform(to_4326)
            env_4326 = cloned.GetEnvelope()  # minX, maxX, minY, maxY in EPSG:4326
            # Exclude non-continental overseas territories (French Guiana/DOM-TOM, Canary Islands, Svalbard/Jan Mayen)
            if c_upper == "FR" and (env_4326[0] < -10.0 or env_4326[2] < 38.0):
                continue
            if c_upper == "ES" and (env_4326[0] < -11.0 or env_4326[2] < 34.0):
                continue
            if c_upper == "NO" and env_4326[3] > 72.0:
                continue

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

    gt = (min_x, pixel_res, 0.0, max_y, 0.0, -pixel_res)
    proj_wkt = tgt_srs.ExportToWkt()

    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    wgs84_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    geom_4326 = union_geom.Clone()
    if src_srs and not src_srs.IsSame(wgs84_srs):
        tx_to_4326 = osr.CoordinateTransformation(src_srs, wgs84_srs)
        geom_4326.Transform(tx_to_4326)

    # Simplify complex country polygons for ASF HTTP query URL length limits
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
        "gt": gt,
        "proj": proj_wkt,
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
            for p in parts:
                if len(p) == 15 and "T" in p and p[:8].isdigit():
                    start_str = f"{p[:4]}-{p[4:6]}-{p[6:8]}T{p[9:11]}:{p[11:13]}:{p[13:15]}Z"
                    break
                elif len(p) == 8 and p.isdigit() and 2020 <= int(p[:4]) <= 2035:
                    start_str = f"{p[:4]}-{p[4:6]}-{p[6:8]}T00:00:00Z"
                    break

            if len(parts) >= 6 and parts[5].isdigit():
                path_num = int(parts[5])
            else:
                m_path = re.search(r"(?:track|path|orbit)_?(\d+)", title, re.IGNORECASE)
                if m_path:
                    path_num = int(m_path.group(1))

            if start_str:
                try:
                    d_val = datetime.datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                    if not (start_date <= d_val <= end_date):
                        continue
                except Exception:
                    pass
            else:
                start_str = datetime.datetime.fromtimestamp(h5_file.stat().st_mtime).strftime("%Y-%m-%dT00:00:00Z")

            geom_dict = None
            if HAS_H5PY:
                try:
                    with h5py.File(h5_file, "r") as h5_f:
                        g_cand = None
                        for cp in [
                            "science/LSAR/GCOV/grids/frequencyA",
                            "science/LSAR/GCOV/grids/frequencyB",
                            "GCOV/grids/frequencyA",
                            "grids/frequencyA"
                        ]:
                            if cp in h5_f:
                                g_cand = h5_f[cp]
                                break
                        if g_cand is not None:
                            xs = g_cand.get("xCoordinates") or g_cand.get("xCoordinate") or g_cand.get("x")
                            ys = g_cand.get("yCoordinates") or g_cand.get("yCoordinate") or g_cand.get("y")
                            if xs is not None and ys is not None:
                                min_x, max_x = float(min(xs[0], xs[-1])), float(max(xs[0], xs[-1]))
                                min_y, max_y = float(min(ys[0], ys[-1])), float(max(ys[0], ys[-1]))
                                epsg = 4326
                                proj_ds = g_cand.get("projection")
                                if proj_ds is not None and hasattr(proj_ds, "attrs") and "epsg_code" in proj_ds.attrs:
                                    try:
                                        epsg = int(proj_ds.attrs["epsg_code"])
                                    except Exception:
                                        pass
                                srs_s = osr.SpatialReference()
                                srs_s.ImportFromEPSG(epsg)
                                srs_w = osr.SpatialReference()
                                srs_w.ImportFromEPSG(4326)
                                srs_w.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                                tx = osr.CoordinateTransformation(srs_s, srs_w)
                                ring = ogr.Geometry(ogr.wkbLinearRing)
                                for x, y in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]:
                                    lon, lat, _ = tx.TransformPoint(x, y)
                                    ring.AddPoint_2D(lon, lat)
                                poly = ogr.Geometry(ogr.wkbPolygon)
                                poly.AddGeometry(ring)
                                geom_dict = json.loads(poly.ExportToJson())
                except Exception:
                    pass

            found.append({
                "title": title,
                "start_time": start_str,
                "stop_time": start_str,
                "url": str(h5_file),
                "local_h5": h5_file,
                "bytes": h5_file.stat().st_size,
                "flight_direction": "DESCENDING",
                "path_number": path_num,
                "geometry": geom_dict,
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
        local_dir: Optional[Path] = None,
        source: str = "auto"
    ) -> List[Dict]:
        """
        Queries local `/eodata/NISAR` repository first (when `source` is `local` or `auto`),
        then queries NASA ASF DAAC (`source` in `asf` or `auto`) via `asf_search` or direct ASF REST GeoJSON endpoint fallback.
        """
        src_mode = (source or "auto").lower()
        if src_mode in ("local", "auto") and local_dir and Path(local_dir).exists():
            local_hits = self.search_local_h5(Path(local_dir), start_date, end_date, product_type)
            if local_hits or src_mode == "local":
                return local_hits
        elif src_mode == "local":
            logging.warning(f"Local NISAR repository not found at {local_dir}")
            return []

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

            url = "https://api.daac.asf.alaska.edu/services/search/param"
            post_data = urllib.parse.urlencode(params).encode("utf-8")
            req = urllib.request.Request(url, data=post_data, headers={"User-Agent": "AIML-CropMapper-Cloud/2.0"})
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
