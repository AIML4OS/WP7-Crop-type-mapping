#!/usr/bin/env python
"""
5_build_raster_overviews.py - Universal High-Performance GDAL Pyramid/Overviews Generator.

Generates compressed (LZW) multiscale pyramid overviews for massive GeoTIFF stacks,
Sentinel-1/2 rasters, and classification maps so they render instantaneously in QGIS / ArcGIS.

Usage examples:
  python tools/5_build_raster_overviews.py -i workingDir/NL/orbit_88/processed_raster/NL_orbit_88_S2_timeseries.tif
  python tools/5_build_raster_overviews.py -d workingDir/NL/orbit_88/processed_raster/
  python tools/5_build_raster_overviews.py -c NL
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
from osgeo import gdal

BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDir"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def build_overviews_for_file(
    tif_path: Path,
    levels: Optional[List[int]] = None,
    resampling: str = "AVERAGE",
    compress: str = "LZW"
) -> bool:
    if not tif_path.exists():
        logging.error(f"File not found: {tif_path}")
        return False

    if levels is None:
        levels = [2, 4, 8, 16, 32, 64]

    logging.info(f"Building overviews {levels} ({resampling}, {compress}) for: {tif_path.name}...")
    gdal.SetConfigOption('COMPRESS_OVERVIEW', compress)
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.SetConfigOption('BIGTIFF_OVERVIEW', 'IF_NEEDED')

    try:
        ds = gdal.Open(str(tif_path), gdal.GA_Update)
        if ds is None:
            logging.error(f"Could not open file in update mode: {tif_path}")
            return False

        ds.BuildOverviews(resampling, levels, callback=gdal.TermProgress_nocb)
        ds.FlushCache()
        ds = None
        logging.info(f"SUCCESS: Overviews built for {tif_path.name}!")
        return True
    except Exception as e:
        logging.error(f"Error building overviews for {tif_path.name}: {e}")
        return False


def build_overviews_for_directory(
    directory: Path,
    levels: Optional[List[int]] = None,
    resampling: str = "AVERAGE",
    pattern: str = "*.tif"
):
    tifs = list(directory.glob(pattern))
    logging.info(f"Found {len(tifs)} GeoTIFF files matching '{pattern}' in {directory}...")
    for f in tifs:
        build_overviews_for_file(f, levels, resampling)


def main():
    parser = argparse.ArgumentParser(description="Universal GDAL Pyramid Overviews Generator.")
    parser.add_argument('-i', '--input', type=str, default=None, help="Path to single GeoTIFF file")
    parser.add_argument('-d', '--directory', type=str, default=None, help="Directory containing GeoTIFF files")
    parser.add_argument('-c', '--country', type=str, default=None, help="Country code in workingDir (e.g. NL, PL)")
    parser.add_argument('--levels', nargs='+', type=int, default=[2, 4, 8, 16, 32, 64], help="Overview levels (default: 2 4 8 16 32 64)")
    parser.add_argument('--resampling', default="AVERAGE", choices=["NEAREST", "AVERAGE", "GAUSS", "CUBIC", "MODE"], help="Resampling method (default: AVERAGE, use NEAREST/MODE for classification maps)")
    parser.add_argument('--compress', default="LZW", choices=["LZW", "DEFLATE", "JPEG", "NONE"], help="Compression for overviews (default: LZW)")

    args = parser.parse_args()

    if args.input:
        build_overviews_for_file(Path(args.input), args.levels, args.resampling, args.compress)
    elif args.directory:
        build_overviews_for_directory(Path(args.directory), args.levels, args.resampling)
    elif args.country:
        country_dir = BASE_DIR / args.country.upper()
        if country_dir.exists():
            for proc_dir in country_dir.glob("orbit_*/processed_raster"):
                build_overviews_for_directory(proc_dir, args.levels, args.resampling)
        else:
            logging.error(f"Country directory not found: {country_dir}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
