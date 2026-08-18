#!/usr/bin/env python
"""
4_generate_crop_priors.py
=========================
Calculates statistical real-world crop acreages (Bayesian Prior Probabilities)
from country-wide LPIS vector datasets and saves priors.json.

Usage examples:
  python tools/4_generate_crop_priors.py -c NL --input path/to/brp.gpkg --crop_col GEWAS
  python tools/4_generate_crop_priors.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME
  python tools/4_generate_crop_priors.py -c PT --input path/to/isip.shp --crop_col CATEGORIA
"""

import argparse
import json
import logging
import os
import pathlib
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLES_BASE_DIR = PROJECT_ROOT / "auxiliary_files" / "shapefiles_samples"


def generate_priors(
    country: str,
    input_vector_path: str,
    crop_col: str,
    output_path: str = None,
    area_col: str = None
):
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        logging.error("geopandas is required. Run in aiml_env conda environment.")
        sys.exit(1)

    country = country.upper()
    in_path = pathlib.Path(input_vector_path)
    if not in_path.exists():
        logging.error(f"Input file not found: {in_path}")
        sys.exit(1)

    if output_path:
        out_fp = pathlib.Path(output_path)
    else:
        out_fp = SAMPLES_BASE_DIR / country / "priors.json"

    out_fp.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading vector dataset: {in_path} ...")
    gdf = gpd.read_file(str(in_path))

    if crop_col not in gdf.columns:
        logging.error(f"Crop column '{crop_col}' not found. Available columns: {list(gdf.columns)}")
        sys.exit(1)

    logging.info("Calculating crop polygon areas...")
    if area_col and area_col in gdf.columns:
        gdf['calc_area'] = gdf[area_col].astype(float)
    else:
        if gdf.crs and gdf.crs.is_geographic:
            logging.info("Projecting to EPSG:3857 to calculate square meters...")
            gdf['calc_area'] = gdf.to_crs(epsg=3857).geometry.area
        else:
            gdf['calc_area'] = gdf.geometry.area

    # Group by crop name and sum area
    grouped = gdf.groupby(crop_col)['calc_area'].sum().reset_index()
    grouped = grouped[grouped[crop_col].astype(str).str.strip() != ""]

    total_area = grouped['calc_area'].sum()
    grouped['proportion'] = grouped['calc_area'] / total_area

    priors_dict = {}
    for _, row in grouped.iterrows():
        key = str(row[crop_col]).lower().strip()
        priors_dict[key] = round(float(row['proportion']), 6)

    sorted_priors = dict(sorted(priors_dict.items(), key=lambda x: x[1], reverse=True))

    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump(sorted_priors, f, indent=2, ensure_ascii=False)

    logging.info(f"\n=======================================================")
    logging.info(f" SUCCESS: Generated priors.json with {len(sorted_priors)} classes.")
    logging.info(f" Saved to: {out_fp}")
    logging.info(f"=======================================================")
    logging.info("\nTop 7 Crop Types by Real Acreage:")
    for crop, prop in list(sorted_priors.items())[:7]:
        logging.info(f"  - {crop.title()}: {prop * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate name-based priors.json for Bayesian classification.")
    parser.add_argument('-c', '--country', required=True, help="Country code, e.g. NL, PL, PT, FR, AT")
    parser.add_argument('-i', '--input', required=True, help="Path to raw unbalanced shapefile/GeoPackage")
    parser.add_argument('--crop_col', required=True, help="Column name containing crop names")
    parser.add_argument('-o', '--output', default=None, help="Custom output priors.json path")
    parser.add_argument('--area_col', default=None, help="Optional precalculated area column name")

    args = parser.parse_args()
    generate_priors(
        country=args.country,
        input_vector_path=args.input,
        crop_col=args.crop_col,
        output_path=args.output,
        area_col=args.area_col
    )


if __name__ == '__main__':
    main()
