import os
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

GPKG_PATH = r"d:\AIML_CropMapper_Cloud\auxiliary_files\raster_files\AgriMasks\NL\brpgewaspercelen_definitief_2025.gpkg"
OUTPUT_DIR = r"d:\AIML_CropMapper_Cloud\auxiliary_files\shapefiles_samples\NL"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_SHP_PATH = os.path.join(OUTPUT_DIR, "samples.shp")

# Target crop translation and mapping (20 classes)
CROP_MAPPING = {
    # 1. Brassicas & Leafy Vegetables
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

    # 2. Clover
    "Klaver, rode, groenbemesting, vanggewas": "Clover",
    "Klaver, witte, groenbemesting, vanggewas": "Clover",
    "Klaver": "Clover",

    # 3. Fallow & Cover Crops
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

    # 4. Flower Bulbs
    "Tulp, bloembollen en -knollen": "Flower Bulbs",
    "Lelie, bloembollen en -knollen": "Flower Bulbs",
    "Gladiool, bloembollen en - knollen": "Flower Bulbs",
    "Pioenroos, overige bloemkwekerijgewassen": "Flower Bulbs",
    "Iris, Bolvormend": "Flower Bulbs",
    "Overige bloemen, overige bloemkwekerijgewassen": "Flower Bulbs",
    "Bloemzaden open grond": "Flower Bulbs",

    # 5. Grassland
    "Grasland, blijvend": "Grassland",
    "Grasland, tijdelijk": "Grassland",
    "Grasland, natuurlijk. Met landbouwactiviteiten.": "Grassland",
    "Stroken wild gras": "Grassland",

    # 6. Legume Vegetables
    "Erwten, groene/gele (groen te oogsten)": "Legume Vegetables",
    "Bonen, veld- (onder andere duiven-, paarden-, wierbonen)": "Legume Vegetables",
    "Stamsperziebonen (=stamslabonen), productie": "Legume Vegetables",
    "Peulen, productie": "Legume Vegetables",
    "Stoksnijbonen en stokslabonen, productie": "Legume Vegetables",
    "Sojabonen": "Legume Vegetables",
    "Peulvruchten, overige": "Legume Vegetables",

    # 7. Lucerne
    "Luzerne": "Lucerne",

    # 8. Maize
    "Mais, snij-": "Maize",
    "Maiskolvensilage": "Maize",
    "Mais, suiker-": "Maize",
    "Mais, korrel-": "Maize",

    # 9. Nurseries & Ornamental Shrubs
    "Sierheesters en klimplanten, pot- en containerveld": "Nurseries & Ornamental Shrubs",
    "Boomkwekerijgewassen": "Nurseries & Ornamental Shrubs",
    "Bos- en haagplanten, open grond": "Nurseries & Ornamental Shrubs",
    "Laanbomen/parkbomen, spillen, open grond": "Nurseries & Ornamental Shrubs",
    "Laanbomen/parkbomen, opzetters, open grond": "Nurseries & Ornamental Shrubs",
    "Vruchtbomen, onderstammen, open grond": "Nurseries & Ornamental Shrubs",
    "Laanbomen/parkbomen, opzetters, pot- en containerveld": "Nurseries & Ornamental Shrubs",

    # 10. Oats
    "Haver": "Oats",

    # 11. Onions
    "Uien, gele zaai-": "Onions",
    "Uien, rode zaai-": "Onions",
    "Uien, poot en plant, 1e jaars": "Onions",
    "Uien, poot en plant, 2e jaars": "Onions",
    "Uien, zilver-": "Onions",
    "Sjalotten": "Onions",

    # 12. Orchards & Fruits
    "Peren. Aangeplant voorafgaande aan lopende seizoen.": "Orchards & Fruits",
    "Appels. Aangeplant voorafgaande aan lopende seizoen.": "Orchards & Fruits",
    "Bessen, rode": "Orchards & Fruits",
    "Aardbeien op stellingen, productie": "Orchards & Fruits",
    "Vruchtbomen, overig, open grond": "Orchards & Fruits",
    "Vruchtbomen, moerbomen, open grond": "Orchards & Fruits",
    "Vruchtbomen, overig, pot- en containerveld": "Orchards & Fruits",
    "Appels. Dit seizoen aangeplant.": "Orchards & Fruits",
    "Peren. Dit seizoen aangeplant.": "Orchards & Fruits",

    # 13. Potatoes
    "Aardappelen, consumptie": "Potatoes",
    "Aardappelen, poot NAK": "Potatoes",
    "Aardappelen, bestrijdingsmaatregel AM": "Potatoes",
    "Aardappelen, poot TBM": "Potatoes",
    "Aardappelen, zetmeel-": "Potatoes",

    # 14. Root & Tuber Vegetables
    "Kroten/rode beets, productie": "Root & Tuber Vegetables",
    "Kroten/rode bieten, productie": "Root & Tuber Vegetables",
    "Winterpeen, productie": "Root & Tuber Vegetables",
    "Witlofwortel, productie": "Root & Tuber Vegetables",
    "Pastinaak, productie": "Root & Tuber Vegetables",
    "Knolselderij, productie": "Root & Tuber Vegetables",
    "Bospeen, productie": "Root & Tuber Vegetables",
    "Wortelen, overige": "Root & Tuber Vegetables",

    # 15. Sugar Beet
    "Bieten, suiker-": "Sugar Beet",
    "Bieten, voeder-": "Sugar Beet",

    # 16. Summer Barley
    "Gerst, zomer-": "Summer Barley",

    # 17. Summer Wheat
    "Tarwe, zomer-": "Summer Wheat",

    # 18. Triticale
    "Triticale": "Triticale",

    # 19. Winter Barley
    "Gerst, winter-": "Winter Barley",

    # 20. Winter Wheat
    "Tarwe, winter-": "Winter Wheat"
}

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

def main():
    if not os.path.exists(GPKG_PATH):
        logging.error(f"GeoPackage not found: {GPKG_PATH}")
        logging.error("Please download the 2.93 GB GPKG from RVO/PDOK first.")
        sys.exit(1)
        
    start_time = time.time()
    logging.info("Reading whole Netherlands BRP 2025 GeoPackage for training samples...")
    
    # Read only geometry and 'gewas' column to save memory
    try:
        gdf = gpd.read_file(GPKG_PATH, engine="pyogrio", columns=["gewas"])
    except Exception as e:
        logging.error(f"Failed to read GPKG: {e}")
        sys.exit(1)
        
    logging.info(f"Loaded {len(gdf)} fields. Filtering target crop types...")
    
    # Filter crop types
    gdf_filtered = gdf[gdf["gewas"].isin(CROP_MAPPING.keys())].copy()
    gdf_filtered["crop_name"] = gdf_filtered["gewas"].map(CROP_MAPPING)
    gdf_filtered["crop_id"] = gdf_filtered["crop_name"].map(CROP_IDS)
    logging.info(f"Filtered to {len(gdf_filtered)} target crop fields.")
    
    # Reproject to metric Amersfoort / RD New (EPSG:28992) for size check and centroid calculation
    logging.info("Reprojecting to RD New (EPSG:28992)...")
    gdf_metric = gdf_filtered.to_crs("EPSG:28992")
    
    # Area filter: fields >= 0.5 ha
    gdf_metric["area_ha"] = gdf_metric.geometry.area / 10000.0
    min_area_ha = 0.5
    gdf_size_ok = gdf_metric[gdf_metric["area_ha"] >= min_area_ha].copy()
    logging.info(f"Filtered out fields < {min_area_ha} ha. Remaining: {len(gdf_size_ok)}")
    
    # Centroid check (must be inside polygon) - Vectorized for speed
    logging.info("Calculating centroids inside polygons (vectorized)...")
    centroids = gdf_size_ok.geometry.centroid
    is_inside = gdf_size_ok.geometry.contains(centroids)
    gdf_cnt = gdf_size_ok[is_inside].copy()
    gdf_cnt.geometry = centroids[is_inside]
    logging.info(f"Valid interior centroids: {len(gdf_cnt)}")
    
    # Stratified Sampling: 1000 per class across the entire country
    samples_per_class = 1000
    sampled_groups = []
    
    for crop, group in gdf_cnt.groupby("crop_name"):
        if len(group) < samples_per_class:
            logging.warning(f"Class '{crop}' has only {len(group)} fields in the whole country. Taking all.")
            sampled_groups.append(group)
        else:
            logging.info(f"Sampling {samples_per_class} fields for '{crop}' (out of {len(group)} country-wide)")
            sampled_groups.append(group.sample(n=samples_per_class, random_state=42))
            
    final_gdf = pd.concat(sampled_groups)
    
    # Reproject back to WGS 84 (EPSG:4326)
    final_gdf = final_gdf.to_crs("EPSG:4326")
    final_gdf = final_gdf[["crop_id", "crop_name", "area_ha", "geometry"]]
    
    # Save to Shapefile
    logging.info(f"Saving country-wide samples to {OUTPUT_SHP_PATH}...")
    final_gdf.to_file(OUTPUT_SHP_PATH, engine="pyogrio")
    
    logging.info(f"Success! Created shapefile with {len(final_gdf)} points.")
    print("\nFinal country-wide sample counts per class:")
    counts = final_gdf["crop_name"].value_counts()
    for name, count in counts.items():
        print(f"  {name}: {count}")
    logging.info(f"Process completed in {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()
