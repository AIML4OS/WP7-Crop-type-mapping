#!/usr/bin/env python
"""
2_build_agricultural_mask.py
============================
Universal Agricultural Mask Generator for AIML CropMapper.

Supports 2 modes of building binary cropland masks (1 = agricultural, 0 = non-agricultural):
  Mode A (LPIS Vectors): Rasterizes official cadastral parcel boundaries (.shp, .gpkg, .geojson)
  Mode B (Copernicus HRL): Unzips and mosaics Copernicus CLMS High Resolution Layer crop tiles

Usage examples:
  # Mode A (From LPIS Vector Data - Recommended for highest accuracy):
  python tools/2_build_agricultural_mask.py -c NL --lpis path/to/brp.gpkg
  python tools/2_build_agricultural_mask.py -c PL --lpis path/to/arimr.shp
  python tools/2_build_agricultural_mask.py -c PT --lpis path/to/isip.shp --ref_raster workingDir/PT/orbit_161/processed_raster/PT_orbit_161_VH_VV.tif

  # Mode B (From Copernicus HRL / CLMS Raster Tiles):
  python tools/2_build_agricultural_mask.py -c NL
  python tools/2_build_agricultural_mask.py -c PL --target_crs EPSG:3857
"""

import os
import sys
import zipfile
import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from osgeo import gdal, ogr, osr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUX_DIR = PROJECT_ROOT / 'auxiliary_files'
AGRIMASKS_DIR = AUX_DIR / 'raster_files' / 'AgriMasks'
NUTS_DIR = AUX_DIR / 'shapefiles_nuts'
WORKING_DIR = PROJECT_ROOT / 'workingDir'

CLASS_3_INCLUDE = {
    1110, 1120, 1150, 1430, 1130, 1210, 1220, 1310, 1320, 1410, 1420, 1440, 3100
}

CLASS_ALL_EXCLUDE = {
    0, 2000, 3000, 5000, 65535
}


def find_nuts_boundary(country: str) -> Optional[Path]:
    country_nuts = NUTS_DIR / country.upper()
    if country_nuts.exists():
        for pat in [f'NUTS0_{country.upper()}.shp', f'NUTS2_{country.upper()}.shp', '*.shp']:
            matches = list(country_nuts.glob(pat))
            if matches:
                return matches[0]
    return None


def find_reference_raster(country: str) -> Optional[Path]:
    """Finds any processed S1 or S2 raster in workingDir to inherit exact pixel grid & CRS."""
    country_working = WORKING_DIR / country.upper()
    if country_working.exists():
        rasters = list(country_working.glob("**/processed_raster/*.tif"))
        if rasters:
            return rasters[0]
    return None


def build_mask_from_lpis(
    country: str,
    lpis_path: str,
    ref_raster: Optional[str] = None,
    target_crs: str = "EPSG:3857",
    resolution: float = 10.0,
    clip_shp: Optional[str] = None
):
    country = country.upper()
    lpis_file = Path(lpis_path)
    if not lpis_file.exists():
        print(f"  [ERROR] LPIS vector file not found: {lpis_file}")
        sys.exit(1)

    country_mask_dir = AGRIMASKS_DIR / country
    country_mask_dir.mkdir(parents=True, exist_ok=True)
    epsg_clean = target_crs.replace(':', '').lower()
    out_mask_path = country_mask_dir / f"{country}_agri_mask_lpis_{epsg_clean}.tif"

    print(f"\n=======================================================")
    print(f" Rasterizing LPIS Parcels to Binary Mask for: {country}")
    print(f" Source LPIS: {lpis_file}")
    print(f"=======================================================")

    # Determine reference geometry (CRS, Extent, Resolution)
    ref_proj = None
    ref_bounds = None
    res_x, res_y = resolution, resolution

    resolved_ref = Path(ref_raster) if ref_raster else find_reference_raster(country)
    if resolved_ref and resolved_ref.exists():
        print(f"  Inheriting exact spatial grid from reference raster: {resolved_ref.name}")
        ds_ref = gdal.Open(str(resolved_ref))
        if ds_ref:
            ref_proj = ds_ref.GetProjection()
            gt = ds_ref.GetGeoTransform()
            res_x, res_y = abs(gt[1]), abs(gt[5])
            min_x = gt[0]
            max_y = gt[3]
            max_x = min_x + gt[1] * ds_ref.RasterXSize
            min_y = max_y + gt[5] * ds_ref.RasterYSize
            ref_bounds = [min_x, min_y, max_x, max_y]
            ds_ref = None

    if not ref_bounds:
        cutline = Path(clip_shp) if clip_shp else find_nuts_boundary(country)
        if cutline and cutline.exists():
            print(f"  Calculating bounds from NUTS boundary: {cutline.name}")
            ds_shp = ogr.Open(str(cutline))
            layer = ds_shp.GetLayer()
            ext = layer.GetExtent()  # minX, maxX, minY, maxY
            ref_bounds = [ext[0], ext[2], ext[1], ext[3]]
            ds_shp = None

    rasterize_kwargs = {
        'format': 'GTiff',
        'burnValues': [1],
        'initValues': [0],
        'outputType': gdal.GDT_Byte,
        'creationOptions': ['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=IF_SAFER', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512'],
        'xRes': res_x,
        'yRes': res_y
    }

    if ref_proj:
        rasterize_kwargs['outputSRS'] = ref_proj
    else:
        rasterize_kwargs['outputSRS'] = target_crs

    if ref_bounds:
        rasterize_kwargs['outputBounds'] = ref_bounds

    print("  Executing high-performance GDAL Rasterize...")
    opts = gdal.RasterizeOptions(**rasterize_kwargs)
    res_ds = gdal.Rasterize(str(out_mask_path), str(lpis_file), options=opts)
    if res_ds is not None:
        res_ds.FlushCache()
        res_ds = None
        print(f"  [OK] Successfully created LPIS agricultural mask: {out_mask_path}")

        # Copy to default auto-discovery locations
        auto_mask = AUX_DIR / "raster_files" / f"{country}_agri_mask.tif"
        auto_mask.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(out_mask_path), str(auto_mask))
            print(f"  [OK] Copied to standard pipeline location: {auto_mask}")
        except Exception:
            pass


def find_or_extract_tifs(country_dir: Path, temp_dir: Path) -> list:
    results_dir = country_dir / 'Results'
    if not results_dir.exists():
        country_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        return []

    zip_files = list(results_dir.glob('*.zip')) + list(results_dir.glob('*.ZIP'))
    tif_files = []

    for zf in zip_files:
        print(f"  Unpacking archive: {zf.name}...")
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(temp_dir)
            tifs_in_zip = [f for f in temp_dir.rglob('*.tif') if not f.name.startswith('.')]
            tif_files.extend(tifs_in_zip)
        except Exception as e:
            print(f"  ERROR extracting {zf.name}: {e}")

    if not tif_files:
        already_tifs = [
            f for f in results_dir.rglob('*.tif')
            if 'agri_mask' not in f.name and not f.name.startswith('.')
        ]
        if already_tifs:
            return already_tifs

    return tif_files


def reclassify_to_binary(src_path: str, dst_path: str, include_set: set):
    import numpy as np
    ds = gdal.Open(src_path)
    if ds is None:
        return

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.int32)

    out = np.zeros(data.shape, dtype=np.uint8)
    for val in include_set:
        out[data == val] = 1

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        dst_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    )
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(out)
    out_ds.FlushCache()
    out_ds = None
    ds = None


def reclassify_allcrops_to_binary(src_path: str, dst_path: str, exclude_set: set):
    import numpy as np
    ds = gdal.Open(src_path)
    if ds is None:
        return

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.int32)

    out = np.ones(data.shape, dtype=np.uint8)
    for val in exclude_set:
        out[data == val] = 0

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        dst_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    )
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(out)
    out_ds.FlushCache()
    out_ds = None
    ds = None


def mosaic_and_reproject(tif_files: list, output_path: str, target_crs: str, clip_shp: str = None):
    print(f"  Mosaicking + reprojecting {len(tif_files)} tiles to {target_crs}...")
    warp_kwargs = {
        'format': 'GTiff',
        'dstSRS': target_crs,
        'resampleAlg': gdal.GRA_NearestNeighbour,
        'creationOptions': ['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=IF_SAFER', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    }

    if clip_shp and os.path.exists(clip_shp):
        warp_kwargs['cutlineDSName'] = clip_shp
        warp_kwargs['cropToCutline'] = True

    warp_opts = gdal.WarpOptions(**warp_kwargs)
    res = gdal.Warp(output_path, [str(f) for f in tif_files], options=warp_opts)
    if res is not None:
        res.FlushCache()
        res = None
        print(f"  [OK] Saved agricultural mask: {output_path}")


def process_hrl_mask(country: str, target_crs: str = 'EPSG:3857', clip_shp: str = None, no_clip: bool = False):
    country = country.upper()
    country_dir = AGRIMASKS_DIR / country
    print(f"\n=======================================================")
    print(f" Building Agricultural Mask from HRL CLMS for: {country}")
    print(f"=======================================================")

    temp_dir = Path(tempfile.mkdtemp(prefix=f"agrimask_{country}_"))

    try:
        tif_files = find_or_extract_tifs(country_dir, temp_dir)
        if not tif_files:
            print(f"  No raw HRL/CLMS data found in {country_dir / 'Results'}.")
            print(f"  Place downloaded HRL ZIP or TIF files in {country_dir / 'Results'} and re-run, or use --lpis <path>.")
            return

        print(f"  Found {len(tif_files)} source raster tiles.")

        # Reclassify Variant A (3class arable)
        reclass_a_dir = temp_dir / 'reclass_a'
        reclass_a_dir.mkdir(exist_ok=True)
        reclass_a_files = []
        for idx, tf in enumerate(tif_files, 1):
            dst = reclass_a_dir / f"reclass_a_{idx}.tif"
            reclassify_to_binary(str(tf), str(dst), CLASS_3_INCLUDE)
            reclass_a_files.append(dst)

        # Reclassify Variant B (all crops)
        reclass_b_dir = temp_dir / 'reclass_b'
        reclass_b_dir.mkdir(exist_ok=True)
        reclass_b_files = []
        for idx, tf in enumerate(tif_files, 1):
            dst = reclass_b_dir / f"reclass_b_{idx}.tif"
            reclassify_allcrops_to_binary(str(tf), str(dst), CLASS_ALL_EXCLUDE)
            reclass_b_files.append(dst)

        # Resolve clip shapefile
        cutline = None
        if not no_clip:
            if clip_shp and os.path.exists(clip_shp):
                cutline = clip_shp
            else:
                cutline_p = find_nuts_boundary(country)
                cutline = str(cutline_p) if cutline_p else None

        epsg_clean = target_crs.replace(':', '').lower()
        out_a = country_dir / f"{country}_agri_mask_3class_{epsg_clean}.tif"
        out_b = country_dir / f"{country}_agri_mask_allcrops_{epsg_clean}.tif"

        mosaic_and_reproject(reclass_a_files, str(out_a), target_crs, cutline)
        mosaic_and_reproject(reclass_b_files, str(out_b), target_crs, cutline)

        auto_mask = AUX_DIR / "raster_files" / f"{country}_agri_mask.tif"
        auto_mask.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(out_b), str(auto_mask))
            print(f"  [OK] Copied default mask -> {auto_mask}")
        except Exception:
            pass

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"SUCCESS: Agricultural mask processing for {country} complete.")


def main():
    parser = argparse.ArgumentParser(description="Build binary agricultural mask from LPIS Vector Data or Copernicus HRL CLMS.")
    parser.add_argument('-c', '--country', required=True, help="Country code, e.g. NL, PL, PT, FR, IE")
    parser.add_argument('--lpis', default=None, help="Path to LPIS cadastral parcel vector file (.shp, .gpkg, .geojson) for direct vector mask generation")
    parser.add_argument('--ref_raster', default=None, help="Reference GeoTIFF raster to inherit exact grid, resolution, and CRS")
    parser.add_argument('--target_crs', default='EPSG:3857', help="Target CRS (default: EPSG:3857)")
    parser.add_argument('--res', type=float, default=10.0, help="Pixel resolution in meters (default: 10.0)")
    parser.add_argument('--clip_shp', default=None, help="Custom shapefile path for boundary clipping")
    parser.add_argument('--no_clip', action='store_true', help="Do not clip to country boundaries")

    args = parser.parse_args()

    if args.lpis:
        build_mask_from_lpis(
            country=args.country,
            lpis_path=args.lpis,
            ref_raster=args.ref_raster,
            target_crs=args.target_crs,
            resolution=args.res,
            clip_shp=args.clip_shp
        )
    else:
        process_hrl_mask(args.country, args.target_crs, args.clip_shp, args.no_clip)


if __name__ == '__main__':
    main()
