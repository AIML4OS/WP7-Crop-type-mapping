#!/usr/bin/env python
"""
2_build_agricultural_mask.py
============================
Unzips, mosaics, clips, and creates binary agricultural cropland masks
from Copernicus HRL Crop Type / CLMS data.

Usage examples:
  python tools/2_build_agricultural_mask.py -c NL
  python tools/2_build_agricultural_mask.py -c PL
  python tools/2_build_agricultural_mask.py -c PT --target_crs EPSG:3857
"""

import os
import sys
import zipfile
import argparse
import shutil
import tempfile
from pathlib import Path
from osgeo import gdal, ogr

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
AUX_DIR = PROJECT_ROOT / 'auxiliary_files'
AGRIMASKS_DIR = AUX_DIR / 'raster_files' / 'AgriMasks'
NUTS_DIR = AUX_DIR / 'shapefiles_nuts'

CLASS_3_INCLUDE = {
    1110, 1120, 1150, 1430, 1130, 1210, 1220, 1310, 1320, 1410, 1420, 1440, 3100
}

CLASS_ALL_EXCLUDE = {
    0, 2000, 3000, 5000, 65535
}


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


def find_nuts_boundary(country: str) -> Path | None:
    country_nuts = NUTS_DIR / country.upper()
    if country_nuts.exists():
        for pat in [f'NUTS0_{country.upper()}.shp', f'NUTS2_{country.upper()}.shp', '*.shp']:
            matches = list(country_nuts.glob(pat))
            if matches:
                return matches[0]
    return None


def process_country_mask(country: str, target_crs: str = 'EPSG:3857', clip_shp: str = None, no_clip: bool = False):
    country = country.upper()
    country_dir = AGRIMASKS_DIR / country
    print(f"\n=======================================================")
    print(f" Building Agricultural Mask for country: {country}")
    print(f"=======================================================")

    temp_dir = Path(tempfile.mkdtemp(prefix=f"agrimask_{country}_"))

    try:
        tif_files = find_or_extract_tifs(country_dir, temp_dir)
        if not tif_files:
            print(f"  No raw HRL/CLMS data found in {country_dir / 'Results'}.")
            print(f"  Place downloaded HRL ZIP or TIF files in {country_dir / 'Results'} and re-run.")
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

        # Also place in auxiliary_files/raster_files/{country}_agri_mask.tif for auto-discovery
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
    parser = argparse.ArgumentParser(description="Build binary agricultural mask from Copernicus HRL / CLMS.")
    parser.add_argument('-c', '--country', required=True, help="Country code, e.g. NL, PL, PT, FR, IE")
    parser.add_argument('--target_crs', default='EPSG:3857', help="Target CRS (default: EPSG:3857)")
    parser.add_argument('--clip_shp', default=None, help="Custom shapefile path for boundary clipping")
    parser.add_argument('--no_clip', action='store_true', help="Do not clip to country boundaries")

    args = parser.parse_args()
    process_country_mask(args.country, args.target_crs, args.clip_shp, args.no_clip)


if __name__ == '__main__':
    main()
