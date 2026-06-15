import argparse
import os
import json
from pathlib import Path
import geopandas as gpd
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Generate name-based priors.json from raw unbalanced LPIS vector dataset.")
    parser.add_argument("--input", required=True, help="Path to raw unbalanced shapefile, geopackage, or DBF table.")
    parser.add_argument("--output", required=True, help="Path where to save the generated priors.json.")
    parser.add_argument("--crop_col", default="crop_name", help="Column name containing crop names (e.g. crop_name or crop_id).")
    parser.add_argument("--area_col", help="Optional column containing polygon area. If not provided, it will be calculated from geometry.")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    print(f"Loading dataset: {input_path} ...")
    
    # If it is a DBF, we can load it using geopandas (via pandas/dbfread or by reading it as shp)
    # Geopandas read_file supports .dbf, .shp, .gpkg, .geojson, etc.
    gdf = gpd.read_file(str(input_path))
    
    if args.crop_col not in gdf.columns:
        raise ValueError(f"Crop column '{args.crop_col}' not found in dataset. Available columns: {gdf.columns.tolist()}")
        
    print("Calculating crop areas...")
    
    # Calculate area if not provided in a column
    if args.area_col:
        if args.area_col not in gdf.columns:
            raise ValueError(f"Area column '{args.area_col}' not found in dataset.")
        gdf['calc_area'] = gdf[args.area_col].astype(float)
    else:
        if gdf.geometry is not None:
            print("Calculating area from geometry (make sure dataset is in a projected coordinate system, e.g. EPSG:3857)...")
            gdf['calc_area'] = gdf.geometry.area
        else:
            raise ValueError("No geometry found and no area_col provided. Cannot calculate areas.")
            
    # Group by crop name/ID and sum area
    grouped = gdf.groupby(args.crop_col)['calc_area'].sum().reset_index()
    
    # Filter out empty or nodata names
    grouped = grouped[grouped[args.crop_col].astype(str).str.strip() != ""]
    
    # Normalize to get proportions
    total_area = grouped['calc_area'].sum()
    grouped['proportion'] = grouped['calc_area'] / total_area
    
    # Convert to dictionary (making keys lowercase if they are strings)
    priors_dict = {}
    for _, row in grouped.iterrows():
        key = row[args.crop_col]
        if isinstance(key, str):
            key = key.lower().strip()
        priors_dict[key] = round(float(row['proportion']), 6)
        
    # Sort by proportion descending
    sorted_priors = dict(sorted(priors_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Ensure directory exists and write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_priors, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully generated priors.json with {len(sorted_priors)} classes.")
    print(f"Saved to: {output_path}")
    print("\nTop 5 crops by area:")
    for crop, prop in list(sorted_priors.items())[:5]:
        print(f"  - {crop}: {prop * 100:.2f}%")

if __name__ == "__main__":
    main()
