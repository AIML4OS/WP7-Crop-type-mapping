#!/usr/bin/env python
"""
3_prepare_classification_samples.py
===================================
Universal Agricultural Crop Sample Point Generator from LPIS/Cadastral Vector Databases.

Reads parcel polygons (Shapefile, GeoPackage, GeoJSON, SQLite), filters micro-parcels,
standardizes crop labels, generates interior sampling points (representative points / centroids),
balances class counts, and outputs ready-to-train samples.shp for the classifier.

Usage examples:
  # Netherlands (NL BRP dataset):
  python tools/3_prepare_classification_samples.py -c NL --input path/to/brp.gpkg --crop_col GEWAS --min_area_ha 0.2

  # Poland (PL ARiMR dataset):
  python tools/3_prepare_classification_samples.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME --max_samples_per_class 3000

  # Portugal (PT LPIS dataset):
  python tools/3_prepare_classification_samples.py -c PT --input path/to/isip.shp --crop_col CATEGORIA --min_area_ha 0.3
"""

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLES_BASE_DIR = PROJECT_ROOT / "auxiliary_files" / "shapefiles_samples"


def load_mapping_json(mapping_path: Optional[str]) -> Dict[str, str]:
    if mapping_path and os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load mapping JSON {mapping_path}: {e}")
    return {}


def prepare_samples(
    country: str,
    input_vector_path: str,
    crop_col: str,
    output_dir: Optional[str] = None,
    min_area_ha: float = 0.1,
    max_samples_per_class: int = 2500,
    min_samples_per_class: int = 20,
    sampling_method: str = "representative",
    mapping_json_path: Optional[str] = None,
    target_crs: str = "EPSG:4326"
):
    try:
        import geopandas as gpd
        import pandas as pd
        import numpy as np
    except ImportError:
        logging.error("geopandas, pandas, and numpy are required. Please run inside aiml_env environment.")
        sys.exit(1)

    country = country.upper()
    in_path = pathlib.Path(input_vector_path)
    if not in_path.exists():
        logging.error(f"Input vector file not found: {in_path}")
        sys.exit(1)

    out_folder = pathlib.Path(output_dir) if output_dir else (SAMPLES_BASE_DIR / country)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_shp_path = out_folder / "samples.shp"

    logging.info(f"Loading vector layer: {in_path} ...")
    gdf = gpd.read_file(str(in_path))
    logging.info(f"Loaded {len(gdf)} parcels. Original CRS: {gdf.crs}")

    if crop_col not in gdf.columns:
        logging.error(f"Crop column '{crop_col}' not found in attributes! Available columns: {list(gdf.columns)}")
        sys.exit(1)

    # 1. Clean invalid geometries and empty crop names
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf[crop_col].notna() & (gdf[crop_col].astype(str).str.strip() != "")].copy()

    # 2. Filter by minimum parcel area
    if min_area_ha > 0:
        if gdf.crs and gdf.crs.is_geographic:
            # Temporarily calculate area in projected CRS EPSG:3857
            areas_m2 = gdf.to_crs(epsg=3857).geometry.area
        else:
            areas_m2 = gdf.geometry.area
        areas_ha = areas_m2 / 10000.0
        gdf['parcel_area_ha'] = areas_ha
        initial_count = len(gdf)
        gdf = gdf[gdf['parcel_area_ha'] >= min_area_ha].copy()
        logging.info(f"Filtered parcels >= {min_area_ha} ha: {initial_count} -> {len(gdf)} parcels.")

    # 3. Apply optional mapping dictionary
    mapping_dict = load_mapping_json(mapping_json_path)
    if mapping_dict:
        logging.info(f"Applying custom crop category mapping dictionary ({len(mapping_dict)} mappings)...")
        gdf['standard_crop'] = gdf[crop_col].astype(str).map(lambda x: mapping_dict.get(x, x))
    else:
        gdf['standard_crop'] = gdf[crop_col].astype(str).str.strip()

    # 4. Filter rare classes below threshold
    class_counts = gdf['standard_crop'].value_counts()
    valid_crops = class_counts[class_counts >= min_samples_per_class].index.tolist()
    gdf = gdf[gdf['standard_crop'].isin(valid_crops)].copy()

    logging.info(f"Identified {len(valid_crops)} valid crop classes with >= {min_samples_per_class} parcels.")

    # 5. Stratified sampling per class (balance dataset)
    sampled_dfs = []
    for c_name, group in gdf.groupby('standard_crop'):
        if len(group) > max_samples_per_class:
            sampled_group = group.sample(n=max_samples_per_class, random_state=42)
        else:
            sampled_group = group
        sampled_dfs.append(sampled_group)

    balanced_gdf = pd.concat(sampled_dfs, ignore_index=True)
    balanced_gdf = gpd.GeoDataFrame(balanced_gdf, geometry='geometry', crs=gdf.crs)

    # 6. Generate interior point geometry (guaranteed inside polygon)
    logging.info(f"Generating interior sampling points ({sampling_method})...")
    if sampling_method == "centroid":
        point_geoms = balanced_gdf.geometry.centroid
    else:
        # representative_point is guaranteed to be strictly inside the polygon
        point_geoms = balanced_gdf.geometry.representative_point()

    # 7. Build integer crop_id mapping
    sorted_unique_crops = sorted(balanced_gdf['standard_crop'].unique().tolist())
    crop_to_id = {name: idx + 1 for idx, name in enumerate(sorted_unique_crops)}

    pts_gdf = gpd.GeoDataFrame({
        'crop_id': balanced_gdf['standard_crop'].map(crop_to_id).astype(int),
        'crop_name': balanced_gdf['standard_crop'].astype(str),
        'geometry': point_geoms
    }, crs=gdf.crs)

    # 8. Reproject to target CRS
    if target_crs and pts_gdf.crs != target_crs:
        pts_gdf = pts_gdf.to_crs(target_crs)

    # 9. Save shapefile and summary table
    pts_gdf.to_file(str(out_shp_path))
    csv_summary = out_folder / "samples_summary.csv"
    summary_df = pts_gdf['crop_name'].value_counts().reset_index()
    summary_df.columns = ['crop_name', 'sample_count']
    summary_df['crop_id'] = summary_df['crop_name'].map(crop_to_id)
    summary_df = summary_df[['crop_id', 'crop_name', 'sample_count']].sort_values('crop_id')
    summary_df.to_csv(str(csv_summary), index=False)

    logging.info(f"\n=======================================================")
    logging.info(f" SUCCESS: Generated {len(pts_gdf)} sample points for {country}!")
    logging.info(f" Shapefile saved to: {out_shp_path}")
    logging.info(f" Summary saved to:   {csv_summary}")
    logging.info(f"=======================================================")
    logging.info("\nCrop Class Distribution:")
    for _, row in summary_df.iterrows():
        logging.info(f"  [ID {int(row['crop_id']):02d}] {row['crop_name']}: {row['sample_count']} samples")


def main():
    parser = argparse.ArgumentParser(description="Universal Agricultural Crop Sample Point Generator from LPIS/Cadastral vectors.")
    parser.add_argument('-c', '--country', required=True, help="Country code, e.g. NL, PL, PT, FR, AT")
    parser.add_argument('-i', '--input', required=True, help="Path to raw LPIS or reference polygon vector layer (.shp, .gpkg, .geojson)")
    parser.add_argument('--crop_col', required=True, help="Column name containing crop name or crop code")
    parser.add_argument('-o', '--output_dir', default=None, help="Custom output directory (default: auxiliary_files/shapefiles_samples/<COUNTRY>)")
    parser.add_argument('--min_area_ha', type=float, default=0.1, help="Minimum parcel area in hectares (default: 0.1 ha)")
    parser.add_argument('--max_samples_per_class', type=int, default=2500, help="Maximum samples per crop class (default: 2500)")
    parser.add_argument('--min_samples_per_class', type=int, default=20, help="Minimum samples threshold per class (default: 20)")
    parser.add_argument('--sampling_method', choices=['representative', 'centroid'], default='representative', help="Point placement method")
    parser.add_argument('--mapping_json', default=None, help="Optional JSON dictionary file for merging subcategories")
    parser.add_argument('--target_crs', default='EPSG:4326', help="Target CRS (default: EPSG:4326)")

    args = parser.parse_args()

    prepare_samples(
        country=args.country,
        input_vector_path=args.input,
        crop_col=args.crop_col,
        output_dir=args.output_dir,
        min_area_ha=args.min_area_ha,
        max_samples_per_class=args.max_samples_per_class,
        min_samples_per_class=args.min_samples_per_class,
        sampling_method=args.sampling_method,
        mapping_json_path=args.mapping_json,
        target_crs=args.target_crs
    )


if __name__ == '__main__':
    main()
