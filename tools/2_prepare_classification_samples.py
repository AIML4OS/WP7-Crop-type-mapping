#!/usr/bin/env python
"""
2_prepare_classification_samples.py
===================================
Universal Agricultural Crop Sample Point Generator from LPIS/Cadastral Vector Databases.

Reads parcel polygons (Shapefile, GeoPackage, GeoJSON, SQLite), filters micro-parcels,
standardizes crop labels, generates interior sampling points (representative points / centroids),
balances class counts, and outputs ready-to-train samples.shp and priors.json for the classifier.

Features:
  - Fast-path SQL-based streaming extraction for massive national GeoPackages (e.g. Dutch BRP 2.3M parcels).
  - Built-in 18-class agricultural mapping for the Netherlands (S1 + S2 + NISAR multi-sensor optimized).
  - Adaptive negative buffering (10m -> 5m -> interior point) to eliminate boundary mixed pixels.
  - Generates synchronized samples.shp, samples_summary.csv, and priors.json.

Usage examples:
  # Netherlands (NL BRP 2026 dataset - 18 classes):
  python tools/2_prepare_classification_samples.py -c NL --input auxiliary_files/shapefiles_samples/NL/gewaspercelen_concept_2026.gpkg --crop_col gewas --max_samples_per_class 2000

  # Poland (PL ARiMR dataset):
  python tools/2_prepare_classification_samples.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME --max_samples_per_class 3000

  # Portugal (PT LPIS dataset):
  python tools/2_prepare_classification_samples.py -c PT --input path/to/isip.shp --crop_col CATEGORIA --min_area_ha 0.3
"""

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLES_BASE_DIR = PROJECT_ROOT / "auxiliary_files" / "shapefiles_samples"

# Standard 18-class taxonomy for the Netherlands (S1 + S2 + NISAR multi-modal optimized)
NL_BRP_MAPPING = {
    "Asparagus": "gewas LIKE '%Asperge%'",
    "Beans": "gewas LIKE '%Boon%' OR gewas LIKE '%bonen%' OR gewas LIKE '%Veldboon%' OR gewas LIKE '%Tuinboon%'",
    "Sugar Beets": "gewas LIKE '%Bieten, suiker%'",
    "Carrots": "(gewas LIKE '%peen%' OR gewas LIKE '%Wortel%') AND gewas NOT LIKE '%Witlof%'",
    "Chicory": "gewas LIKE '%Cichorei%' OR gewas LIKE '%Witlof%'",
    "Flower Bulbs": "gewas LIKE '%Tulp%' OR gewas LIKE '%Lelie%' OR gewas LIKE '%bloembol%' OR gewas LIKE '%Hyacint%' OR gewas LIKE '%Narcis%' OR gewas LIKE '%Krokus%' OR gewas LIKE '%Gladiool%'",
    "Grassland": "gewas LIKE '%Grasland, blijvend%' OR gewas LIKE '%Grasland, tijdelijk%'",
    "Maize": "gewas LIKE '%Mais, snij%' OR gewas LIKE '%Mais, korrel%' OR gewas LIKE '%Maiskolvensilage%' OR gewas LIKE '%Mais, corncob mix%'",
    "Oats": "gewas LIKE '%Haver%'",
    "Onions": "gewas LIKE '%Uien%'",
    "Orchards": "(gewas LIKE '%Peren%' OR (gewas LIKE '%Appel%' AND gewas NOT LIKE '%Aardappel%') OR gewas LIKE '%Kersen%' OR gewas LIKE '%Pruimen%' OR gewas LIKE '%boomgaard%')",
    "Peas": "gewas LIKE '%Erwt%'",
    "Potatoes": "gewas LIKE '%Aardappelen, consumptie%' OR gewas LIKE '%Aardappelen, poot%' OR gewas LIKE '%Aardappelen, zetmeel%'",
    "Rye": "gewas LIKE '%Rogge%'",
    "Spring Barley": "gewas LIKE '%Gerst, zomer%'",
    "Spring Wheat": "gewas LIKE '%Tarwe, zomer%'",
    "Winter Barley": "gewas LIKE '%Gerst, winter%'",
    "Winter Wheat": "gewas LIKE '%Tarwe, winter%'"
}


def load_mapping_json(mapping_path: Optional[str]) -> Dict[str, str]:
    if mapping_path and os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load mapping JSON {mapping_path}: {e}")
    return {}


def _extract_nl_brp_fast_path(
    in_path: pathlib.Path,
    max_samples_per_class: int,
    min_area_ha: float
) -> Tuple[object, Dict[str, float]]:
    """
    High-performance SQL-based stratified parcel sampler for Dutch BRP GeoPackage.
    Avoids loading 2.3M full polygons into memory.
    """
    import pyogrio
    import pandas as pd
    import geopandas as gpd

    logging.info(f"Using high-performance SQL extraction on Dutch BRP GeoPackage: {in_path.name}")
    min_area_m2 = min_area_ha * 10000.0

    collected_gdfs = []
    crop_counts = {}

    for crop_name, sql_cond in NL_BRP_MAPPING.items():
        query = (
            f"SELECT id, gewas, geom FROM brp_gewas "
            f"WHERE ({sql_cond}) "
            f"ORDER BY RANDOM() LIMIT {max_samples_per_class * 2}"
        )
        try:
            gdf_crop = pyogrio.read_dataframe(str(in_path), sql=query)
            if gdf_crop is None or len(gdf_crop) == 0:
                logging.warning(f"  [{crop_name}]: 0 matching parcels found.")
                continue

            # Calculate area and filter micro-parcels
            if gdf_crop.crs and gdf_crop.crs.is_geographic:
                areas_m2 = gdf_crop.to_crs(epsg=3857).geometry.area
            else:
                areas_m2 = gdf_crop.geometry.area

            gdf_crop = gdf_crop[areas_m2 >= min_area_m2].copy()

            if len(gdf_crop) > max_samples_per_class:
                gdf_crop = gdf_crop.sample(n=max_samples_per_class, random_state=42)

            gdf_crop['standard_crop'] = crop_name
            collected_gdfs.append(gdf_crop)
            crop_counts[crop_name] = len(gdf_crop)
            logging.info(f"  [{crop_name}]: Extracted {len(gdf_crop)} valid parcels.")
        except Exception as e:
            logging.error(f"Error querying {crop_name}: {e}")

    if not collected_gdfs:
        raise RuntimeError("No parcels were extracted from Dutch BRP!")

    combined_gdf = pd.concat(collected_gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs=collected_gdfs[0].crs)

    # Approximate national crop acreage priors from 2026 total counts
    total_samples = sum(crop_counts.values())
    priors = {k.lower(): round(v / total_samples, 6) for k, v in crop_counts.items()}

    return combined_gdf, priors


def prepare_samples(
    country: str,
    input_vector_path: str,
    crop_col: str,
    output_dir: Optional[str] = None,
    min_area_ha: float = 0.1,
    max_samples_per_class: int = 2000,
    min_samples_per_class: int = 20,
    sampling_method: str = "representative",
    negative_buffer_m: float = 10.0,
    mapping_json_path: Optional[str] = None,
    target_crs: str = "EPSG:3857"
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
    out_priors_path = out_folder / "priors.json"

    # Fast-path for Netherlands BRP GeoPackage
    if country == "NL" and in_path.suffix.lower() == ".gpkg" and not mapping_json_path:
        balanced_gdf, auto_priors = _extract_nl_brp_fast_path(
            in_path=in_path,
            max_samples_per_class=max_samples_per_class,
            min_area_ha=min_area_ha
        )
    else:
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
        auto_priors = None

    # 6. Generate interior point geometry with negative buffer to avoid boundary contamination
    logging.info(f"Generating interior sampling points ({sampling_method}, negative buffer = {negative_buffer_m}m)...")
    metric_gdf = balanced_gdf.to_crs(epsg=3857)
    if negative_buffer_m > 0:
        buffered_geoms = metric_gdf.geometry.buffer(-negative_buffer_m)
        collapsed = buffered_geoms.is_empty
        if collapsed.any():
            logging.info(f"  Adaptive buffer: {collapsed.sum()}/{len(metric_gdf)} parcels collapsed at -{negative_buffer_m}m, applying -5m / raw fallback.")
            half_buf = metric_gdf.loc[collapsed, 'geometry'].buffer(-negative_buffer_m * 0.5)
            buffered_geoms[collapsed] = half_buf
            still_collapsed = buffered_geoms.is_empty
            if still_collapsed.any():
                buffered_geoms[still_collapsed] = metric_gdf.loc[still_collapsed, 'geometry']
    else:
        buffered_geoms = metric_gdf.geometry

    if sampling_method == "centroid":
        point_geoms_metric = buffered_geoms.centroid
    else:
        point_geoms_metric = buffered_geoms.representative_point()

    point_geoms = gpd.GeoSeries(point_geoms_metric, crs="EPSG:3857").to_crs(balanced_gdf.crs)

    # 7. Build integer crop_id mapping (ordered alphabetically)
    sorted_unique_crops = sorted(balanced_gdf['standard_crop'].unique().tolist())
    crop_to_id = {name: idx + 1 for idx, name in enumerate(sorted_unique_crops)}

    pts_gdf = gpd.GeoDataFrame({
        'crop_id': balanced_gdf['standard_crop'].map(crop_to_id).astype(int),
        'crop_name': balanced_gdf['standard_crop'].astype(str),
        'geometry': point_geoms
    }, crs=balanced_gdf.crs)

    # 8. Reproject to target CRS
    if target_crs and pts_gdf.crs != target_crs:
        pts_gdf = pts_gdf.to_crs(target_crs)

    # 9. Save shapefile, summary table, and priors.json
    pts_gdf.to_file(str(out_shp_path))
    csv_summary = out_folder / "samples_summary.csv"
    summary_df = pts_gdf['crop_name'].value_counts().reset_index()
    summary_df.columns = ['crop_name', 'sample_count']
    summary_df['crop_id'] = summary_df['crop_name'].map(crop_to_id)
    summary_df = summary_df[['crop_id', 'crop_name', 'sample_count']].sort_values('crop_id')
    summary_df.to_csv(str(csv_summary), index=False)

    # Export synchronized priors.json
    priors_dict = {}
    total_pts = len(pts_gdf)
    for _, row in summary_df.iterrows():
        priors_dict[row['crop_name'].lower()] = round(row['sample_count'] / total_pts, 4)

    with open(out_priors_path, 'w', encoding='utf-8') as pf:
        json.dump(priors_dict, pf, indent=2)

    logging.info(f"\n=======================================================")
    logging.info(f" SUCCESS: Generated {len(pts_gdf)} sample points across {len(sorted_unique_crops)} classes for {country}!")
    logging.info(f" Shapefile saved to: {out_shp_path}")
    logging.info(f" Summary saved to:   {csv_summary}")
    logging.info(f" Priors saved to:    {out_priors_path}")
    logging.info(f"=======================================================")
    logging.info("\nCrop Class Distribution (Variant A - 18 classes):")
    for _, row in summary_df.iterrows():
        logging.info(f"  [ID {int(row['crop_id']):02d}] {row['crop_name']}: {row['sample_count']} samples ({priors_dict[row['crop_name'].lower()]*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Universal Agricultural Crop Sample Point Generator from LPIS/Cadastral vectors.")
    parser.add_argument('-c', '--country', required=True, help="Country code, e.g. NL, PL, PT, FR, AT")
    parser.add_argument('-i', '--input', required=True, help="Path to raw LPIS or reference polygon vector layer (.shp, .gpkg, .geojson)")
    parser.add_argument('--crop_col', required=True, help="Column name containing crop name or crop code")
    parser.add_argument('-o', '--output_dir', default=None, help="Custom output directory (default: auxiliary_files/shapefiles_samples/<COUNTRY>)")
    parser.add_argument('--min_area_ha', type=float, default=0.1, help="Minimum parcel area in hectares (default: 0.1 ha)")
    parser.add_argument('--max_samples_per_class', type=int, default=2000, help="Maximum samples per crop class (default: 2000)")
    parser.add_argument('--min_samples_per_class', type=int, default=20, help="Minimum samples threshold per class (default: 20)")
    parser.add_argument('--sampling_method', choices=['representative', 'centroid'], default='representative', help="Point placement method")
    parser.add_argument('--negative_buffer_m', type=float, default=10.0, help="Negative buffer in meters to avoid boundary/mixed pixels (default: 10.0m)")
    parser.add_argument('--mapping_json', default=None, help="Optional JSON dictionary file for merging subcategories")
    parser.add_argument('--target_crs', default='EPSG:3857', help="Target CRS (default: EPSG:3857)")

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
        negative_buffer_m=args.negative_buffer_m,
        mapping_json_path=args.mapping_json,
        target_crs=args.target_crs
    )


if __name__ == '__main__':
    main()
