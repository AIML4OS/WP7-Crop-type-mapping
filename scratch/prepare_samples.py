import os
import json
import urllib.request
import urllib.parse
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Output directory
OUTPUT_DIR = r"d:\AIML_CropMapper_Cloud\auxiliary_files\shapefiles_samples\NL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bounding box for Flevoland (NL23) in EPSG:4326
# min-x, min-y, max-x, max-y
BBOX = "5.0604,52.2496,5.9684,52.8065"

# Target crop translation and mapping
CROP_MAPPING = {
    # Grassland
    "Grasland, blijvend": "Grassland",
    "Grasland, tijdelijk": "Grassland",
    "Grasland, natuurlijk. Met landbouwactiviteiten.": "Grassland",
    # Maize
    "Mais, snij-": "Maize",
    "Mais, korrel-": "Maize",
    # Onions
    "Uien, gele zaai-": "Onions",
    "Uien, rode zaai-": "Onions",
    "Uien, plant-": "Onions",
    "Uien, zilver-": "Onions",
    # Potatoes
    "Aardappelen, consumptie": "Potatoes",
    "Aardappelen, poot-": "Potatoes",
    "Aardappelen, zetmeel-": "Potatoes",
    # Sugar Beet
    "Bieten, suiker-": "Sugar Beet",
    # Winter Wheat
    "Tarwe, winter-": "Winter Wheat"
}

# Crop IDs sorted alphabetically by English name
# Grassland: 1
# Maize: 2
# Onions: 3
# Potatoes: 4
# Sugar Beet: 5
# Winter Wheat: 6
CROP_IDS = {
    "Grassland": 1,
    "Maize": 2,
    "Onions": 3,
    "Potatoes": 4,
    "Sugar Beet": 5,
    "Winter Wheat": 6
}

def fetch_features():
    features_list = []
    limit = 1000
    base_url = "https://api.pdok.nl/rvo/gewaspercelen/ogc/v1/collections/brpgewas/items"
    
    # We construct the URL with query parameters
    params = {
        "bbox": BBOX,
        "limit": limit,
        "f": "json"
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    page = 1
    max_features = 20000  # Cap the total features to download to prevent infinite loops
    
    logging.info("Starting BRP 2025 download for Flevoland...")
    
    while url and len(features_list) < max_features:
        logging.info(f"Downloading page {page} (Features collected: {len(features_list)})...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            curr_features = data.get("features", [])
            if not curr_features:
                logging.info("No more features found.")
                break
                
            features_list.extend(curr_features)
            
            # Find next link
            links = data.get("links", [])
            next_url = None
            for link in links:
                if link.get("rel") == "next":
                    next_url = link.get("href")
                    break
            
            url = next_url
            page += 1
            
        except Exception as e:
            logging.error(f"Error downloading page {page}: {e}")
            break
            
    logging.info(f"Download complete. Total raw features downloaded: {len(features_list)}")
    return features_list

def process_features(features):
    logging.info("Processing features and filtering crops...")
    
    records = []
    for feat in features:
        props = feat.get("properties", {})
        geom_dict = feat.get("geometry")
        if not geom_dict or not props:
            continue
            
        dutch_crop = props.get("gewas")
        if dutch_crop not in CROP_MAPPING:
            continue
            
        eng_crop = CROP_MAPPING[dutch_crop]
        crop_id = CROP_IDS[eng_crop]
        
        try:
            geom = shape(geom_dict)
            records.append({
                "crop_name": eng_crop,
                "crop_id": crop_id,
                "dutch_name": dutch_crop,
                "geometry": geom
            })
        except Exception as e:
            continue
            
    # Create GeoDataFrame in WGS84
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    logging.info(f"Filtered to target classes. Total fields: {len(gdf)}")
    return gdf

def select_and_extract_centroids(gdf):
    logging.info("Reprojecting to RD New (EPSG:28992) for metric calculations...")
    gdf_metric = gdf.to_crs("EPSG:28992")
    
    # Calculate area in hectares
    gdf_metric["area_ha"] = gdf_metric.geometry.area / 10000.0
    logging.info(f"Average parcel size per crop type (ha):")
    stats = gdf_metric.groupby("crop_name")["area_ha"].agg(["count", "mean", "min", "max"])
    print(stats)
    
    # Filter out fields smaller than 0.5 hectares (5000 sqm) to avoid boundary/mixed pixel issues
    min_area_ha = 0.5
    gdf_filtered = gdf_metric[gdf_metric["area_ha"] >= min_area_ha].copy()
    logging.info(f"Filtered out fields < {min_area_ha} ha. Remaining fields: {len(gdf_filtered)}")
    
    # Extract centroids
    logging.info("Calculating centroids...")
    centroids = []
    
    for idx, row in gdf_filtered.iterrows():
        poly = row.geometry
        cnt = poly.centroid
        
        # Verify if centroid is strictly inside the polygon
        if poly.contains(cnt):
            # Create a copy of the row, replace geometry with point centroid
            row_cnt = row.copy()
            row_cnt.geometry = cnt
            centroids.append(row_cnt)
            
    gdf_cnt = gpd.GeoDataFrame(centroids, crs="EPSG:28992")
    logging.info(f"Centroids calculated. Centroids inside polygons: {len(gdf_cnt)}")
    
    # Stratified sampling: 200 samples per crop class
    # (150 for training, 50 for validation - split done by Stage 2)
    samples_per_class = 200
    sampled_dfs = []
    
    for crop, group in gdf_cnt.groupby("crop_name"):
        if len(group) < samples_per_class:
            logging.warning(f"Class '{crop}' has only {len(group)} fields. Taking all.")
            sampled_dfs.append(group)
        else:
            logging.info(f"Sampling {samples_per_class} fields for '{crop}' (out of {len(group)})")
            sampled_dfs.append(group.sample(n=samples_per_class, random_state=42))
            
    final_gdf = pd.concat(sampled_dfs)
    
    # Reproject back to WGS 84 (EPSG:4326) for standard distribution
    final_gdf = final_gdf.to_crs("EPSG:4326")
    
    # Keep only target columns
    final_gdf = final_gdf[["crop_id", "crop_name", "area_ha", "geometry"]]
    
    return final_gdf

def main():
    # 1. Fetch
    raw_features = fetch_features()
    if not raw_features:
        logging.error("Failed to fetch features.")
        return
        
    # 2. Process
    gdf_polygons = process_features(raw_features)
    if gdf_polygons.empty:
        logging.error("No target features found.")
        return
        
    # 3. Analyze, filter, and extract centroids
    gdf_samples = select_and_extract_centroids(gdf_polygons)
    
    # 4. Save to Shapefile
    out_path = os.path.join(OUTPUT_DIR, "samples.shp")
    logging.info(f"Saving training samples to {out_path}...")
    gdf_samples.to_file(out_path, engine="pyogrio")
    logging.info("Shapefile successfully created and saved!")
    
    # Print final counts
    print("\nFinal sample counts per class:")
    print(gdf_samples["crop_name"].value_counts())

if __name__ == "__main__":
    main()
