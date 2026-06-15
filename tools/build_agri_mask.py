"""
build_agri_mask.py
==================
Unzips, mosaics, clips, and creates binary agricultural masks
from Copernicus HRL Crop Type 2023 data.

NOTE: Input data must be downloaded MANUALLY. The script does NOT download it automatically,
since the Copernicus CLMS portal requires user authentication.

DOWNLOAD AND DATA PREPARATION INSTRUCTIONS:
  1. Log in to the Copernicus Land Monitoring Service (CLMS) portal.
  2. Search for and download the "High Resolution Layer: Crop Type 2023" (HRL CTY 2023) layer
     for the area of interest (e.g., as ZIP tiles).
  3. Create the directory (if it doesn't exist):
     auxiliary_files/raster_files/AgriMasks/<COUNTRY>/Results/
     (where <COUNTRY> is the country code, e.g., PL, FR, IE).
  4. Place the downloaded ZIP files directly inside the "Results/" folder.
  5. Run the script (example below).

Generates 2 mask variants (both BINARY: 0=no crops, 1=crops, 255=NoData):

  Variant A: 3-class arable crops mask (spring / winter / winter rapeseed)
             -> 1 where any of these 3 classes exist
  Variant B: all crops mask (including permanent crops)
             -> 1 where any crop exists

NOTE: Both masks are BINARY (0/1). The value 0 is NOT tagged as NoData.
      NoData = 255 (area outside the range of source data).

Usage:
  python tools/build_agri_mask.py --country IE
  python tools/build_agri_mask.py --country IE --target_crs EPSG:3857
  python tools/build_agri_mask.py --country IE --clip_shp path/to/boundary.shp
  python tools/build_agri_mask.py --country IE --no_clip

Directory structure (input data):
  auxiliary_files/
    raster_files/
      AgriMasks/
        <COUNTRY>/
          Results/    <- Copernicus CLMS zip files (DOWNLOAD MANUALLY!)

    shapefiles_nuts/
      <COUNTRY>/
        NUTS2_<COUNTRY>.shp  <- country boundary for clipping (auto-detected)

Results are saved to:
  auxiliary_files/
    raster_files/
      AgriMasks/
        <COUNTRY>/
          <COUNTRY>_agri_mask_3class_<CRS>.tif   <- Variant A (binary)
          <COUNTRY>_agri_mask_allcrops_<CRS>.tif <- Variant B (binary)
"""

import os
import sys
import zipfile
import argparse
import shutil
import tempfile
from pathlib import Path


# -----------------------------------------------------------------------
# BASE PATH CONFIGURATION
# Script location: D:/AIML_CropMapper_Cloud/tools/
# -----------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent                          # tools/
PROJECT_ROOT = SCRIPT_DIR.parent                              # D:/AIML_CropMapper_Cloud/
AUX_DIR      = PROJECT_ROOT / 'auxiliary_files'
AGRIMASKS_DIR = AUX_DIR / 'raster_files' / 'AgriMasks'
NUTS_DIR      = AUX_DIR / 'shapefiles_nuts'


# -----------------------------------------------------------------------
# HRL CTY 2023 CLASS DEFINITIONS
# Source: CLMS_HRLVLCC_CTY_R10.qml
# -----------------------------------------------------------------------

# Variant A: Arable crops to include (1=spring crops, 2=winter crops, 3=winter rapeseed)
# -> binary mask: these pixels will yield a value of 1 in the output
CLASS_3_INCLUDE = {
    # Winter crops
    1110,   # Wheat (mainly winter wheat)
    1120,   # Barley (partially winter barley)
    1150,   # Other Cereals (other winter cereals)
    # Winter rapeseed
    1430,   # Rapeseed
    # Spring crops and others
    1130,   # Maize
    1210,   # Fresh Vegetables
    1220,   # Dry Pulses
    1310,   # Potatoes
    1320,   # Sugar Beet
    1410,   # Sunflower
    1420,   # Soybeans
    1440,   # Flax, cotton and hemp
    3100,   # Unclassified arable crop
}

# Variant B: All crops (including permanent crops)
ALL_CROPS_INCLUDE = {
    1110, 1120, 1130, 1140, 1150,   # Cereals
    1210, 1220,                     # Vegetables and pulses
    1310, 1320,                     # Root crops
    1410, 1420, 1430, 1440,         # Oilseeds and industrial crops
    2100, 2200, 2310, 2320,         # Permanent crops (vineyards, olives, orchards, nuts)
    3100, 3200,                     # Unclassified crops
}

NODATA_VAL = None   # No NoData value - identical to EU_arable_areas_mask_3857.tif
                    # Value 0 = no crops / out of bounds, 1 = crops


# -----------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------

def unzip_tiles(results_dir: Path, temp_dir: Path) -> list:
    """Unzips all ZIP files from the Results directory into temp_dir.
    Returns a list of paths to the extracted TIF files."""
    zip_files = sorted(results_dir.glob('*.zip'))
    if not zip_files:
        print(f"  [WARNING] No ZIP files found in: {results_dir}")
        return []

    tif_files = []
    print(f"  Unzipping {len(zip_files)} ZIP files...")
    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path, 'r') as zf:
            tif_names = [
                n for n in zf.namelist()
                if n.lower().endswith('.tif') and not n.endswith('.aux.xml')
            ]
            for tif_name in tif_names:
                out_path = temp_dir / Path(tif_name).name
                if not out_path.exists():
                    zf.extract(tif_name, temp_dir)
                    extracted = temp_dir / tif_name
                    if extracted != out_path and extracted.exists():
                        shutil.move(str(extracted), str(out_path))
                tif_files.append(str(out_path))
        print(f"    OK: {zf_path.name}")

    # Check for already extracted TIFs in Results (e.g., if one was already unzipped manually)
    already_tifs = [
        str(p) for p in results_dir.glob('**/*.tif')
        if not p.name.endswith('.aux.xml')
    ]
    if already_tifs and not zip_files:
        print(f"  Found {len(already_tifs)} TIF files directly in the Results directory.")
        return already_tifs

    return tif_files


def reclassify_to_binary(src_path: str, dst_path: str, include_set: set):
    """
    Reclassifies the raster into a binary mask:
      1 -> pixel belongs to a class in include_set (crops)
      0 -> no crops OR out of data bounds (65535 in HRL CTY -> 0)

    NO NoData value - identical to EU_arable_areas_mask_3857.tif.
    The value 0 is a valid value (no crops), not NoData.
    """
    from osgeo import gdal
    import numpy as np

    ds = gdal.Open(src_path)
    if ds is None:
        print(f"  ERROR: Cannot open: {src_path}")
        return

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.int32)

    # Binary mask: 1 = crop, 0 = everything else (including 65535 = outside tile)
    out = np.zeros(data.shape, dtype=np.uint8)
    for val in include_set:
        out[data == val] = 1
    # 65535 (out of HRL CTY bounds) -> remains 0 (already set by zeros)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        dst_path,
        ds.RasterXSize, ds.RasterYSize, 1,
        gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    )
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(out)
    # We do NOT set NoData - identical to EU_arable_areas_mask_3857.tif
    out_ds.FlushCache()
    out_ds = None
    ds = None
    print(f"    Reclassification: {Path(src_path).name} -> {Path(dst_path).name}")


def _detect_shp_crs(shp_path: str) -> str | None:
    """
    Detects the true CRS of the shapefile based on the coordinate range.
    Used as a workaround when the .prj file declares an incorrect CRS.
    """
    from osgeo import ogr
    ds = ogr.Open(shp_path)
    if not ds:
        return None
    layer = ds.GetLayer()
    srs = layer.GetSpatialRef()
    ext = layer.GetExtent()  # (minX, maxX, minY, maxY)
    declared = srs.GetAuthorityCode(None) if srs else None

    # Check range - EPSG:3857 has coordinates in millions (approx. +-20M)
    # EPSG:3035 has coordinates in hundreds of thousands / a few millions
    # EPSG:4326 has coordinates in range +-180 / +-90
    max_coord = max(abs(ext[0]), abs(ext[1]), abs(ext[2]), abs(ext[3]))
    if max_coord < 180:
        real_crs = 'EPSG:4326'
    elif max_coord < 10_000_000 and abs(ext[2]) > 1_000_000:
        real_crs = 'EPSG:3035'
    else:
        real_crs = 'EPSG:3857'

    if declared != real_crs.split(':')[1]:
        print(f"  [WARNING] CRS in .prj: EPSG:{declared}, detected from extent: {real_crs}")
        print(f"            Extent: {ext}, will use {real_crs} as cutlineSRS")
    return real_crs


def mosaic_and_reproject(tif_files: list, output_path: str, target_crs: str,
                         clip_shp: str = None):
    """
    Mosaics the tiles and reprojects them to target_crs.
    Result: a clean binary raster (0/1) WITHOUT any NoData value.
      0 = no crops, 1 = crops
    Uses gdal.Warp directly with the file list (no VRT) to avoid
    issues with zero values during BuildVRT on Windows.
    """
    from osgeo import gdal

    print(f"  Mosaicking + reprojecting {len(tif_files)} tiles to {target_crs}...")

    warp_kwargs = {
        'format': 'GTiff',
        'dstSRS': target_crs,
        'resampleAlg': gdal.GRA_NearestNeighbour,
        'creationOptions': ['COMPRESS=DEFLATE', 'TILED=YES',
                            'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'BIGTIFF=YES'],
        'multithread': True,
        'warpOptions': ['NUM_THREADS=ALL_CPUS'],
    }

    if clip_shp:
        print(f"  Clipping to boundary using: {clip_shp}")
        warp_kwargs['cutlineDSName'] = clip_shp
        warp_kwargs['cropToCutline'] = True
        
        # Detect and set the correct CRS for the cutline (useful when coordinate systems differ)
        real_shp_crs = _detect_shp_crs(clip_shp)
        if real_shp_crs:
            warp_kwargs['cutlineSRS'] = real_shp_crs

    # gdal.Warp accepts the list of files directly - no VRT needed
    ds = gdal.Warp(
        output_path,
        tif_files,          # list of reclassified tiles in EPSG:3035
        **warp_kwargs
    )
    if ds is None:
        print(f"  ERROR: gdal.Warp failed!")
        return
    ds.FlushCache()
    ds = None

    print(f"  Saved: {output_path}")


def resolve_clip_shp(country_code: str) -> str | None:
    """Searches for a country boundary shapefile in the standard NUTS directory."""
    nuts_country_dir = NUTS_DIR / country_code
    if not nuts_country_dir.exists():
        return None
    # Search for: NUTS2_<COUNTRY>.shp or any other .shp file
    candidates = [
        nuts_country_dir / f"NUTS2_{country_code}.shp",
        nuts_country_dir / f"NUTS1_{country_code}.shp",
        *list(nuts_country_dir.glob("*.shp")),
    ]
    for p in candidates:
        if p.exists():
            print(f"  Country boundary: {p}")
            return str(p)
    return None


def build_mask_for_country(country_code: str, results_dir: Path, output_dir: Path,
                           target_crs: str, clip_shp: str = None, force: bool = False):
    """Main function: builds binary agricultural masks for a given country."""
    print(f"\n{'='*60}")
    print(f" Processing: {country_code}  ({target_crs})")
    print(f"{'='*60}")

    crs_tag = target_crs.replace(':', '').lower().replace('epsg', 'epsg')
    out_3class   = output_dir / f"{country_code}_agri_mask_3class_{crs_tag}.tif"
    out_allcrops = output_dir / f"{country_code}_agri_mask_allcrops_{crs_tag}.tif"

    if out_3class.exists() and out_allcrops.exists() and not force:
        print(f"  Masks already exist. Use --force to regenerate.")
        print(f"  {out_3class}")
        print(f"  {out_allcrops}")
        return

    # We use a persistent intermediate directory instead of tempfile.TemporaryDirectory.
    # On Windows, GDAL keeps file handles open, preventing TemporaryDirectory from deleting them,
    # which can cause the output files to be empty/zero-sized.
    temp_path = output_dir / "intermediate"
    temp_path.mkdir(parents=True, exist_ok=True)

    try:
        # [1] Unzip
        print("\n[1/4] Unzipping ZIP files...")
        raw_tifs = unzip_tiles(results_dir, temp_path)
        if not raw_tifs:
            print("  ERROR: No TIF files found after extraction!")
            return

        # [2] Reclassify each tile -> binary 0/1
        print(f"\n[2/4] Reclassifying {len(raw_tifs)} tiles (binary 0/1)...")
        tifs_3class   = []
        tifs_allcrops = []

        for src_tif in raw_tifs:
            stem = Path(src_tif).stem
            dst_3class   = str(temp_path / f"{stem}_3class.tif")
            dst_allcrops = str(temp_path / f"{stem}_allcrops.tif")
            reclassify_to_binary(src_tif, dst_3class,   include_set=CLASS_3_INCLUDE)
            reclassify_to_binary(src_tif, dst_allcrops, include_set=ALL_CROPS_INCLUDE)
            tifs_3class.append(dst_3class)
            tifs_allcrops.append(dst_allcrops)

        # [3] Mosaic + reproject + clip
        print("\n[3/4] Mosaicking, reprojecting, clipping - Variant A (3-class -> binary)...")
        mosaic_and_reproject(tifs_3class,   str(out_3class),   target_crs, clip_shp)

        print("\n[3/4] Mosaicking, reprojecting, clipping - Variant B (all crops -> binary)...")
        mosaic_and_reproject(tifs_allcrops, str(out_allcrops), target_crs, clip_shp)

    finally:
        # Delete intermediate directory after all GDAL operations finish
        import gc
        gc.collect()   # Force release of GDAL file handles
        try:
            shutil.rmtree(str(temp_path))
            print(f"\n  Deleted intermediate directory: {temp_path}")
        except Exception as e:
            print(f"\n  [INFO] Cannot delete intermediate directory (can be deleted manually): {e}")


    # [4] Summary
    print("\n[4/4] Done!")
    for f, label in [(out_3class, "Variant A (3-class binary)"),
                     (out_allcrops, "Variant B (all crops binary)")]:
        size_mb = f.stat().st_size / 1024**2 if f.exists() else 0
        print(f"  {label}: {f}  [{size_mb:.1f} MB]")

    print()
    print("  Legend (both masks binary, no NoData value):")
    print("    0 = no crops / area outside HRL CTY data")
    print("    1 = crops")
    print("    (identical structure as EU_arable_areas_mask_3857.tif)")
    print()
    print("  Classes included in Variant A (3-class):")
    print("    spring:  maize, vegetables, pulses, roots, oilseeds, unclassified")
    print("    winter:  wheat, barley, other winter cereals")
    print("    rape:    rapeseed (1430)")


# -----------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Builds binary agricultural masks from Copernicus HRL CTY 2023',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/build_agri_mask.py --country IE
  python tools/build_agri_mask.py --country IE --target_crs EPSG:3857
  python tools/build_agri_mask.py --country IE --no_clip
  python tools/build_agri_mask.py --country IE --clip_shp D:/my/boundary.shp
  python tools/build_agri_mask.py --country PL --force
        """
    )
    parser.add_argument('--country', '-c', required=True,
                        help='Country code (IE, AT, PL, DE, ...)')
    parser.add_argument('--target_crs', default='EPSG:3857',
                        help='Target coordinate reference system (default EPSG:3857)')
    parser.add_argument('--clip_shp', default=None,
                        help='Path to shapefile for clipping (overrides auto-detection)')
    parser.add_argument('--no_clip', action='store_true',
                        help='Do not clip to country boundaries')
    parser.add_argument('--results_dir', default=None,
                        help='Directory with ZIP files (defaults to AgriMasks/<COUNTRY>/Results/)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (defaults to AgriMasks/<COUNTRY>/)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files')
    args = parser.parse_args()

    country = args.country.upper()

    # ZIP directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = AGRIMASKS_DIR / country / 'Results'

    # Output directory -> AgriMasks/<COUNTRY>/
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = AGRIMASKS_DIR / country

    if not results_dir.exists():
        print(f"ERROR: Data directory does not exist: {results_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Boundary for clipping
    clip_shp = None
    if args.no_clip:
        print("  Clipping disabled (--no_clip)")
    elif args.clip_shp:
        clip_shp = args.clip_shp
    else:
        clip_shp = resolve_clip_shp(country)
        if clip_shp is None:
            print(f"  [INFO] Country boundary shapefile not found for '{country}' in {NUTS_DIR / country}")
            print(f"  The mask will NOT be clipped to the country boundaries.")
            print(f"  Use --clip_shp or add the file to: {NUTS_DIR / country / f'NUTS2_{country}.shp'}")

    build_mask_for_country(
        country_code=country,
        results_dir=results_dir,
        output_dir=output_dir,
        target_crs=args.target_crs,
        clip_shp=clip_shp,
        force=args.force,
    )


if __name__ == '__main__':
    main()
