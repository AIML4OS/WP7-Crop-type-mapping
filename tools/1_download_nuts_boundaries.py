#!/usr/bin/env python
"""
1_download_nuts_boundaries.py
=============================
Downloads and prepares official GISCO NUTS administrative boundaries (NUTS2 and NUTS0)
for European countries.

Usage examples:
  python tools/1_download_nuts_boundaries.py -c NL
  python tools/1_download_nuts_boundaries.py -c PL
  python tools/1_download_nuts_boundaries.py -c PT
  python tools/1_download_nuts_boundaries.py --all
"""

import argparse
import logging
import os
import pathlib
import shutil
import sys
import urllib.request
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

GISCO_NUTS_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/ref-nuts-2021-01m.shp.zip"
WORKSPACE_DIR = pathlib.Path(__file__).resolve().parent.parent
TEMP_DIR = WORKSPACE_DIR / "workingDir" / "temp_nuts"
OUTPUT_BASE_DIR = WORKSPACE_DIR / "auxiliary_files" / "shapefiles_nuts"


def download_and_extract_nuts():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TEMP_DIR / "nuts_2021.zip"

    if not zip_path.exists() or zip_path.stat().st_size < 1000000:
        logging.info(f"Downloading official NUTS shapefiles from GISCO ({GISCO_NUTS_URL})...")
        urllib.request.urlretrieve(GISCO_NUTS_URL, zip_path)
        logging.info("Download completed.")

    extract_dir = TEMP_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    nested_zips = list(extract_dir.glob("**/*RG*4326*.zip"))
    if not nested_zips:
        nested_zips = list(extract_dir.glob("**/*.zip"))

    nested_extract_dir = extract_dir / "nested"
    nested_extract_dir.mkdir(parents=True, exist_ok=True)
    for nz in nested_zips:
        try:
            with zipfile.ZipFile(nz, 'r') as z:
                z.extractall(nested_extract_dir)
        except Exception:
            pass

    shp_files = list(nested_extract_dir.glob("**/*RG*4326*.shp"))
    if not shp_files:
        shp_files = list(nested_extract_dir.glob("**/*.shp"))
    if not shp_files:
        shp_files = list(extract_dir.glob("**/*.shp"))

    if not shp_files:
        raise FileNotFoundError("No shapefiles found in extracted GISCO archive.")

    return shp_files[0]


def process_nuts(country_code: str = None, all_countries: bool = False):
    try:
        import geopandas as gpd
    except ImportError:
        logging.error("geopandas is required. Please activate the aiml_env conda environment.")
        sys.exit(1)

    shp_path = download_and_extract_nuts()
    logging.info(f"Loading GISCO shapefile: {shp_path.name}...")
    gdf = gpd.read_file(shp_path)

    cntr_col = 'CNTR_CODE' if 'CNTR_CODE' in gdf.columns else ('country_co' if 'country_co' in gdf.columns else None)
    levl_col = 'LEVL_CODE' if 'LEVL_CODE' in gdf.columns else ('levl_code' if 'levl_code' in gdf.columns else None)

    if not cntr_col or not levl_col:
        logging.error(f"Cannot find CNTR_CODE/LEVL_CODE columns in {list(gdf.columns)}")
        sys.exit(1)

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    if country_code:
        target_countries = [country_code.upper()]
    elif all_countries:
        target_countries = sorted(gdf[cntr_col].unique().tolist())
    else:
        target_countries = ['NL', 'PL', 'PT', 'FR', 'IE', 'AT', 'DE', 'ES', 'IT']

    logging.info(f"Processing NUTS boundaries for countries: {target_countries}")

    for c in target_countries:
        c_dir = OUTPUT_BASE_DIR / c
        c_dir.mkdir(parents=True, exist_ok=True)

        # NUTS 2 (Regions for clipping)
        gdf_nuts2 = gdf[(gdf[cntr_col] == c) & (gdf[levl_col] == 2)]
        if not gdf_nuts2.empty:
            out_nuts2 = c_dir / f"NUTS2_{c}.shp"
            gdf_nuts2.to_file(out_nuts2)
            logging.info(f"  [OK] Saved NUTS2_{c}.shp ({len(gdf_nuts2)} regions) -> {out_nuts2}")

        # NUTS 0 (Country boundary)
        gdf_nuts0 = gdf[(gdf[cntr_col] == c) & (gdf[levl_col] == 0)]
        if not gdf_nuts0.empty:
            out_nuts0 = c_dir / f"NUTS0_{c}.shp"
            gdf_nuts0.to_file(out_nuts0)
            logging.info(f"  [OK] Saved NUTS0_{c}.shp (National Boundary) -> {out_nuts0}")

    # Clean temporary files
    try:
        shutil.rmtree(str(TEMP_DIR), ignore_errors=True)
    except Exception:
        pass

    logging.info("SUCCESS: NUTS boundary database preparation completed.")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare GISCO NUTS boundaries.")
    parser.add_argument('-c', '--country', help="Country code, e.g. NL, PL, PT, FR, AT")
    parser.add_argument('--all', action='store_true', help="Process all available European countries")

    args = parser.parse_args()
    process_nuts(country_code=args.country, all_countries=args.all)


if __name__ == '__main__':
    main()
