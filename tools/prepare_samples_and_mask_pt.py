import os
import sys
import time
import logging
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from osgeo import gdal, osr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Input and Output paths
SHP_INPUT_PATH = r"D:\AIML_CropMapper_Cloud\auxiliary_files\shapefiles_samples\PT\samples_all.shp"

SAMPLES_OUT_DIR = r"D:\AIML_CropMapper_Cloud\auxiliary_files\shapefiles_samples\PT"
os.makedirs(SAMPLES_OUT_DIR, exist_ok=True)
SAMPLES_OUT_PATH = os.path.join(SAMPLES_OUT_DIR, "samples.shp")

MASKS_OUT_DIR = r"D:\AIML_CropMapper_Cloud\auxiliary_files\raster_files\AgriMasks\PT"
os.makedirs(MASKS_OUT_DIR, exist_ok=True)
MASK_ALLCROPS_PATH = os.path.join(MASKS_OUT_DIR, "PT_agri_mask_allcrops_epsg3857.tif")

# 1. Define crop category mappings (grouped for robust SAR backscatter classification)
CROP_MAPPING = {
    # 1. Almond
    "Amendoa": "Almond",
    
    # 2. Annual Forage Mix
    "Consociações Anuais E Outras Cult. Forrag. Anuais": "Annual Forage Mix",
    
    # 3. Barley (winter cereal)
    "Cevada": "Barley",
    
    # 4. Chestnut
    "Castanha": "Chestnut",
    
    # 5. Cork Oak (merged forest/woodland declarations)
    "Sobreiro Para Produção De Cortiça": "Cork Oak",
    "Povoamento De Sobreiros": "Cork Oak",
    
    # 6. Fallow
    "Pousio": "Fallow",
    
    # 7. Grassland & Pastures (merged pasture/meadow/ryegrass subclasses)
    "Pastagens Permanentes": "Grassland & Pastures",
    "Pastagens Arbustivas": "Grassland & Pastures",
    "Prados Temporários": "Grassland & Pastures",
    "Azevem": "Grassland & Pastures",
    
    # 8. Maize
    "Milho": "Maize",
    
    # 9. Mixed Permanent Crops
    "Misto Culturas Permanentes": "Mixed Permanent Crops",
    
    # 10. Non-grazing Shrubland
    "Superfície Arbustiva Não Pastoreável": "Non-grazing Shrubland",
    
    # 11. Oats (winter cereal)
    "Aveia": "Oats",
    
    # 12. Olive Grove
    "Olival": "Olive Grove",
    
    # 13. Peas
    "Ervilha": "Peas",
    
    # 14. Potato
    "Batata": "Potato",
    
    # 15. Rice
    "Arroz": "Rice",
    
    # 16. Rye (winter cereal)
    "Centeio": "Rye",
    
    # 17. Stone Pine (agro-forestry wood class)
    "Pinhão": "Stone Pine",
    
    # 18. Triticale (winter cereal)
    "Triticale": "Triticale",
    
    # 19. Vineyard
    "Vinha": "Vineyard",
    
    # 20. Wheat (winter cereal)
    "Trigo": "Wheat"
}

# 20 standard grouped crop classes sorted alphabetically
CROP_IDS = {
    "Almond": 1,
    "Annual Forage Mix": 2,
    "Barley": 3,
    "Chestnut": 4,
    "Cork Oak": 5,
    "Fallow": 6,
    "Grassland & Pastures": 7,
    "Maize": 8,
    "Mixed Permanent Crops": 9,
    "Non-grazing Shrubland": 10,
    "Oats": 11,
    "Olive Grove": 12,
    "Peas": 13,
    "Potato": 14,
    "Rice": 15,
    "Rye": 16,
    "Stone Pine": 17,
    "Triticale": 18,
    "Vineyard": 19,
    "Wheat": 20
}


def prepare_samples():
    logging.info("--- STEP 1: Creating Stratified Point Samples ---")
    start = time.time()
    
    logging.info(f"Reading input shapefile: {SHP_INPUT_PATH}")
    gdf = gpd.read_file(SHP_INPUT_PATH, engine="pyogrio")
    logging.info(f"Loaded {len(gdf)} parcels.")
    
    # Apply class mapping
    logging.info("Mapping raw crop classes to grouped categories...")
    gdf["crop_name_mapped"] = gdf["crop_name"].map(CROP_MAPPING)
    
    # Filter only target crop classes
    gdf_filtered = gdf[gdf["crop_name_mapped"].notna()].copy()
    gdf_filtered["crop_name"] = gdf_filtered["crop_name_mapped"]
    gdf_filtered["crop_id"] = gdf_filtered["crop_name"].map(CROP_IDS)
    logging.info(f"Filtered to target classes. Remaining: {len(gdf_filtered)} parcels.")
    
    # Reproject to metric system (EPSG:3763) for accurate area & inside centroid calculations
    logging.info("Reprojecting to Portuguese National Coordinate System (EPSG:3763)...")
    gdf_metric = gdf_filtered.to_crs("EPSG:3763")
    
    # Calculate area in ha
    gdf_metric["area_ha"] = gdf_metric.geometry.area / 10000.0
    
    # Area filter: fields >= 0.5 ha (standard field size threshold)
    min_area_ha = 0.5
    gdf_size_ok = gdf_metric[gdf_metric["area_ha"] >= min_area_ha].copy()
    logging.info(f"Filtered out fields < {min_area_ha} ha. Remaining: {len(gdf_size_ok)} parcels.")
    
    # Extract centroids strictly inside geometries
    logging.info("Calculating centroids inside field geometries...")
    centroids = gdf_size_ok.geometry.centroid
    is_inside = gdf_size_ok.geometry.contains(centroids)
    
    gdf_cnt = gdf_size_ok[is_inside].copy()
    gdf_cnt.geometry = centroids[is_inside]
    
    # Fallback to representative point (which is guaranteed to lie inside) for concave geometries
    gdf_outside = gdf_size_ok[~is_inside].copy()
    if len(gdf_outside) > 0:
        logging.info(f"Handling {len(gdf_outside)} concave fields using representative points...")
        gdf_outside.geometry = gdf_outside.geometry.representative_point()
        gdf_cnt = pd.concat([gdf_cnt, gdf_outside], ignore_index=True)
        
    logging.info(f"Total valid interior reference points: {len(gdf_cnt)}")
    
    # Stratified Sampling: 1500 points per crop class
    samples_per_class = 1500
    sampled_groups = []
    
    for crop, group in gdf_cnt.groupby("crop_name"):
        if len(group) < samples_per_class:
            logging.warning(f"Class '{crop}' has only {len(group)} fields. Taking all.")
            sampled_groups.append(group)
        else:
            logging.info(f"Sampling {samples_per_class} fields for '{crop}' (out of {len(group)})")
            sampled_groups.append(group.sample(n=samples_per_class, random_state=42))
            
    final_gdf = pd.concat(sampled_groups)
    
    # Reproject back to WGS 84 (EPSG:4326)
    final_gdf = final_gdf.to_crs("EPSG:4326")
    final_gdf = final_gdf[["crop_id", "crop_name", "area_ha", "geometry"]]
    
    # Save shapefile
    logging.info(f"Saving stratified samples to: {SAMPLES_OUT_PATH}")
    final_gdf.to_file(SAMPLES_OUT_PATH, engine="pyogrio")
    
    logging.info(f"Samples preparation finished in {time.time() - start:.1f} seconds. Created {len(final_gdf)} points.")
    
    print("\nSample counts per class:")
    counts = final_gdf["crop_name"].value_counts()
    for name, count in counts.items():
        print(f"  {name}: {count}")

def generate_agricultural_mask():
    logging.info("\n--- STEP 2: Generating Country-Wide Binary Agricultural Mask ---")
    start = time.time()
    
    # Read columns and geometry
    logging.info("Reading input shapefile...")
    gdf = gpd.read_file(SHP_INPUT_PATH, engine="pyogrio")
    
    # Blacklist keywords for forestry/shrubland/non-agricultural classes
    BLACKLIST_KEYWORDS = [
        "sobreiro", "sobreiros", "pinhão", "pinheiro", "eucalipto", "azinheiras", "azinheira",
        "folhosas", "resinosas", "carvalho", "quercus", "florestal", "florestais",
        "arbustiva não pastoreável", "não agricola", "água", "charca", "lagoa", "vala", 
        "muro", "património", "galeria", "ripícola", "bosquete", "reliquiais", "talude", 
        "sebe", "corta-vento", "povoamento"
    ]

    def is_agricultural(crop_name):
        if not crop_name:
            return False
        c_lower = crop_name.lower().strip()
        for kw in BLACKLIST_KEYWORDS:
            if kw in c_lower:
                return False
        if "elemento linear" in c_lower:
            return False
        return True

    # Filter to active agricultural classes (exclude forestry/shrubland/non-agri)
    gdf["is_agri"] = gdf["crop_name"].apply(is_agricultural)
    gdf_agri = gdf[gdf["is_agri"]].copy()
    logging.info(f"Filtered to agricultural classes. Remaining: {len(gdf_agri)} parcels.")
    
    # Project to Web Mercator (EPSG:3857) for masks
    logging.info("Reprojecting to Web Mercator (EPSG:3857)...")
    gdf_3857 = gdf_agri.to_crs("EPSG:3857")
    
    # Get spatial bounds
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    logging.info(f"Bounds in EPSG:3857: minx={minx:.1f}, miny={miny:.1f}, maxx={maxx:.1f}, maxy={maxy:.1f}")
    
    # Align to 10m resolution grid
    pixel_size = 10.0
    minx = np.floor(minx / pixel_size) * pixel_size
    miny = np.floor(miny / pixel_size) * pixel_size
    maxx = np.ceil(maxx / pixel_size) * pixel_size
    maxy = np.ceil(maxy / pixel_size) * pixel_size
    
    cols = int((maxx - minx) / pixel_size)
    rows = int((maxy - miny) / pixel_size)
    logging.info(f"Target raster size: {cols} cols x {rows} rows ({cols*rows/1e6:.1f} Mpx)")
    
    # Save temporary shapefile for rasterization to avoid OOM
    temp_shp = os.path.join(MASKS_OUT_DIR, "temp_agri_parcels.shp")
    logging.info(f"Saving temporary shapefile to {temp_shp}...")
    gdf_3857.to_file(temp_shp, engine="pyogrio")
    
    # Rasterize using GDAL
    logging.info(f"Rasterizing active croplands to: {MASK_ALLCROPS_PATH}")
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        MASK_ALLCROPS_PATH,
        cols, rows, 1,
        gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    )
    out_ds.SetGeoTransform([minx, pixel_size, 0, maxy, 0, -pixel_size])
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    out_ds.SetProjection(srs.ExportToWkt())
    
    # Fill with 0 (background)
    band = out_ds.GetRasterBand(1)
    band.Fill(0)
    
    # Rasterize: Burn 1
    gdal.Rasterize(out_ds, temp_shp, burnValues=[1], allTouched=False)
    out_ds.FlushCache()
    
    # Build pyramids for QGIS
    logging.info("Building pyramids (overviews)...")
    out_ds.BuildOverviews(resampling="NEAREST", overviewlist=[2, 4, 8, 16, 32])
    out_ds = None
    
    # Clean up temporary shapefile
    logging.info("Cleaning up temporary shapefiles...")
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        p = temp_shp.replace('.shp', ext)
        if os.path.exists(p):
            os.remove(p)
            
    logging.info(f"Mask generation completed in {time.time() - start:.1f} seconds.")

def main():
    prepare_samples()
    generate_agricultural_mask()
    logging.info("PT Crop Data Preparation Complete!")

if __name__ == "__main__":
    main()
