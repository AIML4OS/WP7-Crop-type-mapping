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
BBOX = "5.0604,52.2496,5.9684,52.8065"

# Target crop translation and mapping (20 classes)
CROP_MAPPING = {
    # 1. Brassicas & Leafy Vegetables (B)
    "Broccoli, productie": "Brassicas & Leafy Vegetables",
    "Bloemkool, zomer, productie": "Brassicas & Leafy Vegetables",
    "Bloemkool, winter, productie": "Brassicas & Leafy Vegetables",
    "Spruitkool/spruitjes, productie": "Brassicas & Leafy Vegetables",
    "Rodekool, productie": "Brassicas & Leafy Vegetables",
    "Spitskool, productie": "Brassicas & Leafy Vegetables",
    "Witte kool, productie": "Brassicas & Leafy Vegetables",
    "Sla, overig, productie": "Brassicas & Leafy Vegetables",
    "Sla, radicchio rosso, productie": "Brassicas & Leafy Vegetables",
    "Prei, winter, productie": "Brassicas & Leafy Vegetables",
    "Spruitkool/spruitjes, zaden en opkweekmateriaal": "Brassicas & Leafy Vegetables",
    "Sla, groentezaden open grond": "Brassicas & Leafy Vegetables",
    "Koolgewassen, overige": "Brassicas & Leafy Vegetables",

    # 2. Clover (C)
    "Klaver, rode, groenbemesting, vanggewas": "Clover",
    "Klaver, witte, groenbemesting, vanggewas": "Clover",
    "Klaver": "Clover",

    # 3. Fallow & Cover Crops (Fa)
    "Groene braak, spontane opkomst": "Fallow & Cover Crops",
    "Groene braak, ingezaaid": "Fallow & Cover Crops",
    "Engels raaigras, groenbemesting, vanggewas": "Fallow & Cover Crops",
    "Overige gras, groenbemesting, vanggewas": "Fallow & Cover Crops",
    "Overige groenbemesters, vlinderbloemige-": "Fallow & Cover Crops",
    "Bladrammenas": "Fallow & Cover Crops",
    "Engels raaigras, graszaad": "Fallow & Cover Crops",
    "Tagetes patula (Afrikaantje)": "Fallow & Cover Crops",
    "Overige groenbemesters, niet-vlinderbloemige-, niet zijnde gras": "Fallow & Cover Crops",
    "Bladrammenas, groenbemesting, vanggewas": "Fallow & Cover Crops",
    "Gele mosterd, groenbemesting, vanggewas": "Fallow & Cover Crops",

    # 4. Flower Bulbs (Fl)
    "Tulp, bloembollen en -knollen": "Flower Bulbs",
    "Lelie, bloembollen en -knollen": "Flower Bulbs",
    "Gladiool, bloembollen en - knollen": "Flower Bulbs",
    "Pioenroos, overige bloemkwekerijgewassen": "Flower Bulbs",
    "Iris, Bolvormend": "Flower Bulbs",
    "Overige bloemen, overige bloemkwekerijgewassen": "Flower Bulbs",
    "Bloemzaden open grond": "Flower Bulbs",

    # 5. Grassland (G)
    "Grasland, blijvend": "Grassland",
    "Grasland, tijdelijk": "Grassland",
    "Grasland, natuurlijk. Met landbouwactiviteiten.": "Grassland",
    "Stroken wild gras": "Grassland",

    # 6. Legume Vegetables (Le)
    "Erwten, groene/gele (groen te oogsten)": "Legume Vegetables",
    "Bonen, veld- (onder andere duiven-, paarden-, wierbonen)": "Legume Vegetables",
    "Stamsperziebonen (=stamslabonen), productie": "Legume Vegetables",
    "Peulen, productie": "Legume Vegetables",
    "Stoksnijbonen en stokslabonen, productie": "Legume Vegetables",
    "Sojabonen": "Legume Vegetables",
    "Peulvruchten, overige": "Legume Vegetables",

    # 7. Lucerne (Lu)
    "Luzerne": "Lucerne",

    # 8. Maize (M)
    "Mais, snij-": "Maize",
    "Maiskolvensilage": "Maize",
    "Mais, suiker-": "Maize",
    "Mais, korrel-": "Maize",

    # 9. Nurseries & Ornamental Shrubs (N)
    "Sierheesters en klimplanten, pot- en containerveld": "Nurseries & Ornamental Shrubs",
    "Boomkwekerijgewassen": "Nurseries & Ornamental Shrubs",
    "Bos- en haagplanten, open grond": "Nurseries & Ornamental Shrubs",
    "Laanbomen/parkbomen, spillen, open grond": "Nurseries & Ornamental Shrubs",
    "Laanbomen/parkbomen, opzetters, open grond": "Nurseries & Ornamental Shrubs",
    "Vruchtbomen, onderstammen, open grond": "Nurseries & Ornamental Shrubs",
    "Laanbomen/parkbomen, opzetters, pot- en containerveld": "Nurseries & Ornamental Shrubs",

    # 10. Oats (Oa)
    "Haver": "Oats",

    # 11. Onions (On)
    "Uien, gele zaai-": "Onions",
    "Uien, rode zaai-": "Onions",
    "Uien, poot en plant, 1e jaars": "Onions",
    "Uien, poot en plant, 2e jaars": "Onions",
    "Uien, zilver-": "Onions",
    "Sjalotten": "Onions",

    # 12. Orchards & Fruits (Or)
    "Peren. Aangeplant voorafgaande aan lopende seizoen.": "Orchards & Fruits",
    "Appels. Aangeplant voorafgaande aan lopende seizoen.": "Orchards & Fruits",
    "Bessen, rode": "Orchards & Fruits",
    "Aardbeien op stellingen, productie": "Orchards & Fruits",
    "Vruchtbomen, overig, open grond": "Orchards & Fruits",
    "Vruchtbomen, moerbomen, open grond": "Orchards & Fruits",
    "Vruchtbomen, overig, pot- en containerveld": "Orchards & Fruits",
    "Appels. Dit seizoen aangeplant.": "Orchards & Fruits",
    "Peren. Dit seizoen aangeplant.": "Orchards & Fruits",

    # 13. Potatoes (P)
    "Aardappelen, consumptie": "Potatoes",
    "Aardappelen, poot NAK": "Potatoes",
    "Aardappelen, bestrijdingsmaatregel AM": "Potatoes",
    "Aardappelen, poot TBM": "Potatoes",
    "Aardappelen, zetmeel-": "Potatoes",

    # 14. Root & Tuber Vegetables (R)
    "Kroten/rode beets, productie": "Root & Tuber Vegetables",
    "Kroten/rode bieten, productie": "Root & Tuber Vegetables",
    "Winterpeen, productie": "Root & Tuber Vegetables",
    "Witlofwortel, productie": "Root & Tuber Vegetables",
    "Pastinaak, productie": "Root & Tuber Vegetables",
    "Knolselderij, productie": "Root & Tuber Vegetables",
    "Bospeen, productie": "Root & Tuber Vegetables",
    "Wortelen, overige": "Root & Tuber Vegetables",

    # 15. Sugar Beet (Su)
    "Bieten, suiker-": "Sugar Beet",
    "Bieten, voeder-": "Sugar Beet",

    # 16. Summer Barley (Sum Ba)
    "Gerst, zomer-": "Summer Barley",

    # 17. Summer Wheat (Sum Wh)
    "Tarwe, zomer-": "Summer Wheat",

    # 18. Triticale (T)
    "Triticale": "Triticale",

    # 19. Winter Barley (Wi Ba)
    "Gerst, winter-": "Winter Barley",

    # 20. Winter Wheat (Wi Wh)
    "Tarwe, winter-": "Winter Wheat"
}

# Crop IDs sorted alphabetically by English name
CROP_IDS = {
    "Brassicas & Leafy Vegetables": 1,
    "Clover": 2,
    "Fallow & Cover Crops": 3,
    "Flower Bulbs": 4,
    "Grassland": 5,
    "Legume Vegetables": 6,
    "Lucerne": 7,
    "Maize": 8,
    "Nurseries & Ornamental Shrubs": 9,
    "Oats": 10,
    "Onions": 11,
    "Orchards & Fruits": 12,
    "Potatoes": 13,
    "Root & Tuber Vegetables": 14,
    "Sugar Beet": 15,
    "Summer Barley": 16,
    "Summer Wheat": 17,
    "Triticale": 18,
    "Winter Barley": 19,
    "Winter Wheat": 20
}

def fetch_features():
    features_list = []
    limit = 1000
    base_url = "https://api.pdok.nl/rvo/gewaspercelen/ogc/v1/collections/brpgewas/items"
    
    params = {
        "bbox": BBOX,
        "limit": limit,
        "f": "json"
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    page = 1
    max_features = 40000  # We increased the limit to 40,000 to cover more fields for rare crops
    
    logging.info("Starting BRP 2025 download for Flevoland (large catalog)...")
    
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
            
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    logging.info(f"Filtered to target classes. Total fields: {len(gdf)}")
    return gdf

def select_and_extract_centroids(gdf):
    logging.info("Reprojecting to RD New (EPSG:28992) for metric calculations...")
    gdf_metric = gdf.to_crs("EPSG:28992")
    
    gdf_metric["area_ha"] = gdf_metric.geometry.area / 10000.0
    
    # Filter out fields smaller than 0.5 hectares (5000 sqm)
    min_area_ha = 0.5
    gdf_filtered = gdf_metric[gdf_metric["area_ha"] >= min_area_ha].copy()
    logging.info(f"Filtered out fields < {min_area_ha} ha. Remaining fields: {len(gdf_filtered)}")
    
    # Extract centroids strictly inside polygons
    logging.info("Calculating centroids...")
    centroids = []
    
    for idx, row in gdf_filtered.iterrows():
        poly = row.geometry
        cnt = poly.centroid
        
        if poly.contains(cnt):
            row_cnt = row.copy()
            row_cnt.geometry = cnt
            centroids.append(row_cnt)
            
    gdf_cnt = gpd.GeoDataFrame(centroids, crs="EPSG:28992")
    logging.info(f"Centroids inside polygons: {len(gdf_cnt)}")
    
    # Stratified sampling: 200 samples per crop class
    # (or take all if class has fewer than 200 fields)
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
    final_gdf = final_gdf.to_crs("EPSG:4326")
    final_gdf = final_gdf[["crop_id", "crop_name", "area_ha", "geometry"]]
    
    return final_gdf

def main():
    raw_features = fetch_features()
    if not raw_features:
        logging.error("Failed to fetch features.")
        return
        
    gdf_polygons = process_features(raw_features)
    if gdf_polygons.empty:
        logging.error("No target features found.")
        return
        
    gdf_samples = select_and_extract_centroids(gdf_polygons)
    
    out_path = os.path.join(OUTPUT_DIR, "samples.shp")
    logging.info(f"Saving training samples to {out_path}...")
    gdf_samples.to_file(out_path, engine="pyogrio")
    logging.info("Shapefile successfully created and saved!")
    
    print("\nFinal sample counts per class:")
    counts = gdf_samples["crop_name"].value_counts()
    for name, count in counts.items():
        print(f"  {name}: {count}")

if __name__ == "__main__":
    main()
