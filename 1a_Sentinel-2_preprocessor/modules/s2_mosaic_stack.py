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
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", str(_PROJECT_ROOT / "workingDirs")))
AUX_DIR = Path(os.environ.get("AIML_AUX_DIR", str(_PROJECT_ROOT / "auxiliary_files")))
SHAPEFILES_DIR = AUX_DIR / "shapefiles_nuts"
DEFAULT_DOYS = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]
S2_SPECTRAL_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

REGIONAL_DOYS = {
    'CONTINENTAL_CENTRAL': [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287],
    'MEDITERRANEAN': [45, 65, 85, 105, 120, 135, 150, 165, 180, 200, 220, 240, 260, 280],
    'NORTHERN': [105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300]
}


def get_regional_doys(country_code: str) -> List[int]:
    """Returns the optimal 14-date phenological DOY sequence for the specified country."""
    c = country_code.upper()
    if c in {'PT', 'ES', 'IT', 'EL', 'GR', 'CY', 'MT', 'HR', 'AL', 'TR', 'ME', 'MK'}:
        return REGIONAL_DOYS['MEDITERRANEAN']
    elif c in {'SE', 'FI', 'EE', 'LT', 'LV', 'NO', 'IS'}:
        return REGIONAL_DOYS['NORTHERN']
    return REGIONAL_DOYS['CONTINENTAL_CENTRAL']


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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def get_country_shapefile(country_code: str) -> Optional[Path]:
    c_up = country_code.upper()
    country_dir = SHAPEFILES_DIR / c_up
    if country_dir.exists():
        shps = list(country_dir.glob("*.shp"))
        if shps:
            return shps[0]
    shps = list(SHAPEFILES_DIR.glob(f"**/*{c_up}*.shp"))
    if shps:
        return shps[0]

    # Automatic fallback: extract country boundary from Eurostat GISCO NUTS_RG_01M_2021_3857.geojson
    geojson_path = SHAPEFILES_DIR / "NUTS_RG_01M_2021_3857.geojson"
    if geojson_path.exists():
        try:
            country_dir.mkdir(parents=True, exist_ok=True)
            out_shp = country_dir / f"NUTS2_{c_up}.shp"
            src_ds = ogr.Open(str(geojson_path), 0)
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
                    return out_shp
        except Exception as e:
            logging.warning(f"Could not extract country shapefile from GISCO GeoJSON for {c_up}: {e}")
    return None


def get_s1_raster_reference(track_dir: Path) -> Optional[Dict]:
    candidate_dirs = [
        track_dir / "1_input_stacks",
        track_dir / "wrapped",
        track_dir,
        track_dir / "processed_raster",
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / track_dir.relative_to(BASE_DIR) / "1_input_stacks" if BASE_DIR in track_dir.parents else None,
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / track_dir.relative_to(BASE_DIR) / "processed_raster" if BASE_DIR in track_dir.parents else None,
        Path(r"D:/AIML_CropMapper_Cloud/workingDir") / track_dir.relative_to(BASE_DIR) if BASE_DIR in track_dir.parents else None
    ]
    if track_dir.exists():
        candidate_dirs.extend(list(track_dir.glob("orbit_*/1_input_stacks")))
        candidate_dirs.extend(list(track_dir.glob("orbit_*/wrapped")))
        candidate_dirs.extend(list(track_dir.glob("orbit_*")))
    for proc_dir in candidate_dirs:
        if proc_dir and proc_dir.exists():
            s1_tifs = (
                list(proc_dir.glob("*_VH_VV*.tif")) +
                list(proc_dir.glob("*_VH_VV*.tif.tmp")) +
                list(proc_dir.glob("*Sigma0*.tif")) +
                list(proc_dir.glob("S1_*_stack*.tif")) +
                list(proc_dir.glob("_nuts_mask.tif"))
            )
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


def prepare_orbit_cutline_shapefile(
    country_code: str,
    norm_track: str,
    out_dir: Path,
    target_epsg: int = 3857
) -> Tuple[Optional[Path], Optional[ogr.Geometry]]:
    """
    Creates a dissolved, topologically valid single-feature cutline shapefile in target_epsg
    representing the intersection of the active S1 orbit footprint and the country's NUTS2 boundary.
    Prevents GEOS TopologyException during gdal.Warp and clips each orbit's S2 mosaic to its exact S1 extent.
    """
    try:
        import s2_extract_creodias as _ext
        country_geom = _ext.get_country_geometry(country_code, target_epsg=target_epsg)
        orbit_match = re.search(r'orbit_(\d+)', norm_track)
        if orbit_match:
            orbit_num = int(orbit_match.group(1))
            clip_geom = _ext.get_s1_orbit_extent_geometry(country_code, orbit_num, target_epsg=target_epsg)
            if clip_geom is not None and not clip_geom.IsEmpty() and country_geom is not None and not country_geom.IsEmpty():
                try:
                    # Buffer the active SAR raster footprint slightly before intersecting with country_geom,
                    # preventing coastal serrations or discretization truncation
                    buf_dist = (1000.0 / 111320.0) if target_epsg == 4326 else 1000.0
                    buf = clip_geom.Buffer(buf_dist)
                    if buf is not None and not buf.IsEmpty():
                        inter = buf.Intersection(country_geom)
                        if inter is not None and not inter.IsEmpty():
                            clip_geom = inter.MakeValid()
                except Exception:
                    pass
        else:
            clip_geom = country_geom

        if clip_geom is None or clip_geom.IsEmpty():
            return get_country_shapefile(country_code), None

        clip_geom = clip_geom.MakeValid()
        try:
            buf = clip_geom.Buffer(0.0)
            if buf is not None and not buf.IsEmpty():
                clip_geom = buf
        except Exception:
            pass

        # Ensure clip_geom is strictly a Polygon or MultiPolygon without non-polygon slivers for shapefile layer
        g_type = clip_geom.GetGeometryType() & ~ogr.wkb25Bit
        if g_type not in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
            clean_mp = ogr.Geometry(ogr.wkbMultiPolygon)
            if g_type == ogr.wkbGeometryCollection:
                for i in range(clip_geom.GetGeometryCount()):
                    sub_g = clip_geom.GetGeometryRef(i)
                    if sub_g:
                        sub_t = sub_g.GetGeometryType() & ~ogr.wkb25Bit
                        if sub_t == ogr.wkbPolygon:
                            clean_mp.AddGeometry(sub_g.Clone())
                        elif sub_t == ogr.wkbMultiPolygon:
                            for j in range(sub_g.GetGeometryCount()):
                                clean_mp.AddGeometry(sub_g.GetGeometryRef(j).Clone())
            if not clean_mp.IsEmpty():
                clip_geom = clean_mp

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
            return get_country_shapefile(country_code), clip_geom
        lyr_cut = ds_cut.CreateLayer("cutline", dst_srs, ogr.wkbMultiPolygon)
        feat = ogr.Feature(lyr_cut.GetLayerDefn())
        feat.SetGeometry(clip_geom)
        lyr_cut.CreateFeature(feat)
        feat = None
        ds_cut = None
        return cut_shp, clip_geom
    except Exception as e:
        logging.warning(f"Could not build dissolved orbit cutline for {norm_track} ({e}); falling back to raw country shapefile.")
        return get_country_shapefile(country_code), None


def _synthetic_dir_intersects_geom(syn_dir: Path, clip_geom_target_srs: Optional[ogr.Geometry], target_epsg: int = 3857) -> bool:
    """Checks whether a tile's _synthetic_s2 raster bounding box intersects clip_geom_target_srs."""
    if clip_geom_target_srs is None or clip_geom_target_srs.IsEmpty():
        return True
    try:
        sample_tifs = list(syn_dir.glob("day*/*.tif"))
        if not sample_tifs:
            return False
        ds = gdal.Open(str(sample_tifs[0]))
        if not ds:
            return True
        gt = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        proj_wkt = ds.GetProjection()
        ds = None

        min_x, max_x = gt[0], gt[0] + w * gt[1]
        max_y, min_y = gt[3], gt[3] + h * gt[5]
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
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(target_epsg)
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not src_srs.IsSame(dst_srs):
            poly.Transform(osr.CoordinateTransformation(src_srs, dst_srs))
        return bool(poly.Intersects(clip_geom_target_srs))
    except Exception:
        return True


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
    if output_tif.exists() and output_tif.stat().st_size > 1024 and not overwrite:
        return True

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
        'creationOptions': ["COMPRESS=ZSTD", "PREDICTOR=2", "ZSTD_LEVEL=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]
    }

    if output_bounds and target_width and target_height:
        warp_options_kwargs['width'] = int(target_width)
        warp_options_kwargs['height'] = int(target_height)
    else:
        warp_options_kwargs['xRes'] = res_x
        warp_options_kwargs['yRes'] = res_y

    if ref_proj:
        warp_options_kwargs['dstSRS'] = ref_proj
    else:
        warp_options_kwargs['dstSRS'] = f"EPSG:{target_epsg}"

    if output_bounds:
        warp_options_kwargs['outputBounds'] = output_bounds

    if shp_cutline and shp_cutline.exists():
        warp_options_kwargs['cutlineDSName'] = str(shp_cutline)
        warp_options_kwargs['cropToCutline'] = (output_bounds is None)

    gdal.SetConfigOption('GDAL_WARP_IGNORE_BAD_CUTLINE_GEOMETRY', 'YES')
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    try:
        warp_opts = gdal.WarpOptions(**warp_options_kwargs)
        gdal.Warp(str(output_tif), existing_files, options=warp_opts)
        return True
    except Exception as e:
        logging.error(f"Error mosaicking {output_tif.name}: {e}")
        return False
    finally:
        gdal.PopErrorHandler()


def mosaic_stack_clip_single_track(
    track: str,
    country_code: str,
    target_epsg: int = 3857,
    doys: Optional[List[int]] = None,
    max_workers: int = 8,
    overwrite: bool = False,
    build_overviews: bool = True
):
    import json as _json
    if doys is None or doys == DEFAULT_DOYS:
        doys = get_regional_doys(country_code)
        logging.info(f"Auto-selected regional DOY phenological profile for {country_code}: {doys}")

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
    if "/" not in norm_track and "\\" not in norm_track:
        out_proc_dir = BASE_DIR / country_code.upper() / "S2"
    else:
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

    if not overwrite:
        existing_stacks = list(out_proc_dir.glob(f"{sanitized_track}*_S2_timeseries.tif"))
        if existing_stacks and any(s.stat().st_size > 1024 * 1024 * 1024 for s in existing_stacks):
            complete_stack = next(s for s in existing_stacks if s.stat().st_size > 1024 * 1024 * 1024)
            logging.info(f"Sentinel-2 multi-temporal stack already exists for {track}: {complete_stack.name} ({complete_stack.stat().st_size / (1024**3):.1f} GB). Skipping Stage 3 (use --overwrite to force rebuild).")
            return

    shp_cutline, clip_geom = prepare_orbit_cutline_shapefile(country_code, norm_track, out_final_dir, target_epsg=target_epsg)

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
    elif clip_geom is not None and not clip_geom.IsEmpty():
        env = clip_geom.GetEnvelope()  # minX, maxX, minY, maxY in target_epsg
        min_x = float(int(env[0] // res_x) * res_x)
        min_y = float(int(env[2] // res_y) * res_y)
        max_x = float(int((env[1] + res_x - 1e-6) // res_x) * res_x)
        max_y = float(int((env[3] + res_y - 1e-6) // res_y) * res_y)
        output_bounds = (min_x, min_y, max_x, max_y)
        target_width = int(round((max_x - min_x) / res_x))
        target_height = int(round((max_y - min_y) / res_y))
        logging.info(f"Computed snapped 10m orbit geometry for {track}: {target_width}x{target_height}, Bounds: {output_bounds}")

    all_synthetic_dirs = sorted(s2_base.glob("**/_synthetic_s2"))
    if not all_synthetic_dirs:
        logging.warning(f"No _synthetic_s2 directories found in {s2_base}. Please run Stage 2 first.")
        return

    allowed_tiles = None
    try:
        import s2_extract_creodias as _ext
        if country_code.upper() in _ext.COUNTRY_MGRS_TILES:
            allowed_tiles = set(t.upper().lstrip('T') for t in _ext.COUNTRY_MGRS_TILES[country_code.upper()])
    except Exception:
        pass

    synthetic_dirs = []
    for syn_d in all_synthetic_dirs:
        tile_code = syn_d.parent.name.upper().lstrip('T')
        if allowed_tiles is not None and tile_code not in allowed_tiles:
            continue
        if clip_geom is not None and not _synthetic_dir_intersects_geom(syn_d, clip_geom, target_epsg=target_epsg):
            continue
        synthetic_dirs.append(syn_d)

    if not synthetic_dirs:
        synthetic_dirs = all_synthetic_dirs

    selected_tiles = sorted(syn_d.parent.name.upper() for syn_d in synthetic_dirs)

    # Invalidate stale intermediate DOY mosaics if tile set or orbit clipping changed
    sig_file = out_final_dir / "_mosaic_signature.json"
    current_sig = {
        "track": norm_track,
        "tiles": selected_tiles,
        "width": target_width,
        "height": target_height,
        "orbit_clipped": True
    }
    force_rewarp = overwrite
    if not force_rewarp:
        if sig_file.exists():
            try:
                old_sig = _json.loads(sig_file.read_text(encoding="utf-8"))
                if old_sig != current_sig:
                    logging.info(f"Track {track}: tile set or clipping geometry changed ({old_sig.get('tiles')} -> {selected_tiles}); rebuilding DOY mosaics.")
                    force_rewarp = True
            except Exception:
                pass
        try:
            sig_file.write_text(_json.dumps(current_sig, indent=2), encoding="utf-8")
        except Exception:
            pass

    year = None
    for syn_d in synthetic_dirs:
        day_folders = list(syn_d.glob("day*_*"))
        if day_folders:
            match = re.search(r'day\d+_(\d{4})', day_folders[0].name)
            if match:
                year = int(match.group(1))
                break

    if year is None:
        logging.warning(f"No generated synthetic DOY folders (day*_<year>) found in {s2_base}. Please run Stage 2 (time-series interpolation) first.")
        return

    logging.info(f"Track {track}: mosaicking & clipping from {len(synthetic_dirs)} intersecting MGRS tiles {selected_tiles} (Acquisition Year: {year})...")

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

    warp_workers = 1 if (shp_cutline and shp_cutline.exists()) else min(max_workers, 2)
    logging.info(f"Track {track}: warping {total_bands} single-band DOY mosaics (Warp Workers: {warp_workers}, Internal Threads: ALL_CPUS)...")

    def _worker_mosaic(task):
        nonlocal done_bands
        res = mosaic_single_band_doy(
            task[0], task[1], shp_cutline, target_epsg, ref_proj, res_x, res_y,
            output_bounds, target_width, target_height, force_rewarp
        )
        with lock:
            done_bands += 1
            pct = (done_bands / total_bands) * 100.0
            doy_name = task[1].parent.name
            band_name = task[1].name
            logging.info(f"  [MOSAIC PROGRESS] Track {track}: {done_bands}/{total_bands} bands completed ({pct:.1f}%) - Finished: {doy_name}/{band_name}")
        return res

    with ThreadPoolExecutor(max_workers=warp_workers) as executor:
        list(executor.map(_worker_mosaic, mosaic_tasks))

    try:
        sig_file.write_text(_json.dumps(current_sig, indent=2), encoding="utf-8")
    except Exception:
        pass

    valid_layers = []
    band_descriptions = []
    min_cal_date = None
    max_cal_date = None
    for out_band_tif, band, doy, year in mosaic_bands_list:
        if out_band_tif.exists() and out_band_tif.stat().st_size > 1024:
            valid_layers.append(str(out_band_tif))
            cal_date = datetime.date(year, 1, 1) + datetime.timedelta(days=int(doy) - 1)
            if min_cal_date is None or cal_date < min_cal_date:
                min_cal_date = cal_date
            if max_cal_date is None or cal_date > max_cal_date:
                max_cal_date = cal_date
            date_str = cal_date.strftime("%d%b%Y")
            desc = f"S2_{band}_{date_str}_doy{doy}"
            band_descriptions.append(desc)

    if not valid_layers:
        logging.warning(f"No valid mosaic layers generated for track {track}.")
        return

    date_range_str = f"_{max_cal_date.strftime('%Y%m%d')}_{min_cal_date.strftime('%Y%m%d')}" if (min_cal_date and max_cal_date) else ""
    out_vrt = out_final_dir / f"{sanitized_track}{date_range_str}_S2_timeseries_temp.vrt"
    if "/" not in norm_track and "\\" not in norm_track:
        out_proc_dir = BASE_DIR / country_code.upper() / "S2"
        out_proc_dir.mkdir(parents=True, exist_ok=True)
    out_final_tif = out_proc_dir / f"{sanitized_track}{date_range_str}_S2_timeseries.tif"
    out_final_tmp = out_proc_dir / f"{sanitized_track}{date_range_str}_S2_timeseries.tmp.tif"

    logging.info(f"Assembling VRT and translating final {len(band_descriptions)}-band multi-temporal GeoTIFF stack ({out_final_tif.name})...")
    vrt_opts = gdal.BuildVRTOptions(separate=True)
    gdal.BuildVRT(str(out_vrt), valid_layers, options=vrt_opts)

    trans_opts = gdal.TranslateOptions(
        creationOptions=['COMPRESS=ZSTD', 'PREDICTOR=2', 'ZSTD_LEVEL=3', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'],
        callback=gdal.TermProgress_nocb
    )
    if out_final_tmp.exists():
        try: out_final_tmp.unlink()
        except: pass

    ds_final = gdal.Translate(str(out_final_tmp), str(out_vrt), options=trans_opts)

    if ds_final:
        ds_final.SetMetadataItem("BOA_OFFSET_CORRECTED", "1")
        ds_final.SetMetadataItem("ORBIT_CLIPPED", "1")
        ds_final.SetMetadataItem("MGRS_TILES", ",".join(selected_tiles))
        for b_idx, desc in enumerate(band_descriptions, start=1):
            band_obj = ds_final.GetRasterBand(b_idx)
            band_obj.SetDescription(desc)
            band_obj.SetNoDataValue(0)

        if build_overviews:
            logging.info(f"Building compressed pyramid overviews (2, 4, 8, 16, 32, 64) for {out_final_tif.name}...")
            gdal.SetConfigOption('COMPRESS_OVERVIEW', 'ZSTD')
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

        # Only link to shared country master S2 stack when running country-wide (not per-orbit clipped)
        if "/" not in norm_track and "\\" not in norm_track:
            shared_s2_dir = BASE_DIR / country_code.upper() / "S2"
            shared_s2_dir.mkdir(parents=True, exist_ok=True)
            canonical_s2 = shared_s2_dir / f"{country_code.upper()}_S2_timeseries.tif"
            if out_final_tif.exists() and canonical_s2.resolve() != out_final_tif.resolve():
                try:
                    if canonical_s2.exists():
                        canonical_s2.unlink()
                    os.link(str(out_final_tif), str(canonical_s2))
                except Exception:
                    pass

    if out_vrt.exists():
        try: out_vrt.unlink()
        except: pass

    if out_final_tif.exists() and out_final_tif.stat().st_size > 100 * 1024 * 1024 and out_final_dir.exists():
        try:
            shutil.rmtree(out_final_dir, ignore_errors=True)
            logging.info(f"Cleaned up temporary intermediate DOY mosaics directory: {out_final_dir.name}")
        except Exception:
            pass

    logging.info(f"SUCCESS: Sentinel-2 Multi-Temporal Stack saved to {out_final_tif} ({len(band_descriptions)} bands)!")


def discover_s1_orbits(country_code: str) -> List[int]:
    c_up = country_code.upper()
    country_dir = BASE_DIR / c_up
    if country_dir.exists():
        found = [
            int(re.search(r'orbit_(\d+)', d.name).group(1))
            for d in country_dir.glob("orbit_*")
            if d.is_dir() and re.search(r'orbit_(\d+)', d.name) and any((d / "1_input_stacks").glob("*.tif"))
        ]
        if found:
            return sorted(list(set(found)))
        found_all = [int(re.search(r'orbit_(\d+)', d.name).group(1)) for d in country_dir.glob("orbit_*") if re.search(r'orbit_(\d+)', d.name)]
        if found_all:
            return sorted(list(set(found_all)))
    return COUNTRY_ORBITS.get(c_up, [])


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

        if orbits:
            for o_num in orbits:
                track_name = f"{country_code}/orbit_{o_num}"
                try:
                    mosaic_stack_clip_single_track(track_name, country_code, target_epsg, doys, max_workers, overwrite, build_overviews)
                except Exception as e:
                    logging.error(f"Error processing track {track_name}: {e}")
        else:
            mosaic_stack_clip_single_track(country_code, country_code, target_epsg, doys, max_workers, overwrite, build_overviews)


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
