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
SHP_INPUT_PATH = r"d:\AIML_CropMapper_Cloud\auxiliary_files\raster_files\AgriMasks\PT\PT_2021_EC21.shp"

SAMPLES_OUT_DIR = r"d:\AIML_CropMapper_Cloud\auxiliary_files\shapefiles_samples\PT"
os.makedirs(SAMPLES_OUT_DIR, exist_ok=True)
SAMPLES_OUT_PATH = os.path.join(SAMPLES_OUT_DIR, "samples.shp")

MASKS_OUT_DIR = r"d:\AIML_CropMapper_Cloud\auxiliary_files\raster_files\AgriMasks\PT"
os.makedirs(MASKS_OUT_DIR, exist_ok=True)
MASK_3CLASS_PATH = os.path.join(MASKS_OUT_DIR, "PT_agri_mask_3class_epsg3857.tif")
MASK_ALLCROPS_PATH = os.path.join(MASKS_OUT_DIR, "PT_agri_mask_allcrops_epsg3857.tif")

# 1. Define crop category mappings
GRASSLAND_CROPS = {
    'PERMANENT PASTURES', 'PASTURE WITH BUSHES', 'TEMPORARY MEADOWS', 'ryegrass',
    'lolium_ryegrass', 'temporary_grass', 'pasture_meadow_grassland_grass', 'clover',
    'lucerne', 'alfalfa_lucerne', 'COMMON BIRDSFOOT', 'legumes_harvested_green',
    'COMMON LAND PASTURE'
}
OLIVES = {'OLIVE VALLEY'}
VINEYARDS = {'VINEYARD'}
MAIZE = {
    'CORN', 'MAIZE; POTATOES', 'MAIZE; OTHER VEGETABLES', 'CORN; OTHER VEGETABLES; POTATO',
    'MAIZE; BEANS', 'CORN; OATS', 'CORN; ANNUAL AND OTHER CULT. FORAGE ANNUALS',
    'PUMPKINS AND COURGETTES; CORN', 'MAIZE; POTATO; ONION', 'CORN; TEMPORARY MEADOWS',
    'MAIZE; POTATOES; OTHER VEGETABLES; OATS', 'MAIZE; BEANS; POTATOES',
    'MAIZE; BEANS; POTATOES; ANNUAL INTERCROPS AND OTHER CROPS. FORRAG. ANNUALS',
    'MAIZE; VEGETABLES', 'MAIZE; BEANS; OTHER VEGETABLES; POTATOES; INTERCROPPING AND OTHER ANNUAL CROPS. FORRAG. ANNUALS'
}
RICE = {'RICE', 'ELEGIBLE LANDSCAPE FEATRURES - RICE', 'NON ELEGIBLE LANDSCAPE FEATURES - RICE (COMP. MAA)'}
WHEAT = {
    'WHEAT', 'WHEAT; CORN', 'WHEAT; OUTRAS HORTÍCOLAS', 'WHEAT; LUPIN', 'WHEAT; MELON',
    'WHEAT; VEGETABLES', 'WHEAT; ANNUAL INTERCROPS AND OTHER FODDER CROPS FORRAG. ANNUALS',
    'WHEAT; CORN; OTHER VEGETABLES'
}
BARLEY = {'BARLEY', 'Barley; lolium'}
OATS = {
    'OAT', 'OATS; BEANS', 'OATS; ANNUAL INTERCROPS AND OTHER FODDER CROPS FORRAG. ANNUALS',
    'POTATO; OATS', 'POTATOES; OATS', 'BATATA;AVEIA'
}
RYE = {
    'RYE', 'RYE; CORN', 'RYE; MAIZE; OTHER VEGETABLES; POTATOES', 'RYE; MAIZE; POTATO',
    'RYE; MAIZE; BEANS; POTATOES; OTHER VEGETABLES', 'CEREALS; SOY', 'RYE; OTHER VEGETABLES; POTATOES',
    'RYE; POTATO', 'RYE; ANNUAL INTERCROPS AND OTHER FODDER CROPS FORRAG. ANNUALS'
}
TRITICALE = {'TRITICALE'}
OTHER_CEREALS = {
    'SORGHUM', 'OTHER CEREALS', 'OTHER CEREALS; CORN', 'OTHER CEREALS; OTHER VEGETABLES; POTATOES',
    'OTHER CEREALS; BEANS', 'OTHER CEREALS; CABBAGE; OTHER VEGETABLES; ONION'
}
POTATOES = {
    'POTATO', 'SWEET POTATO', 'BATATA DOCE', 'BATATA', 'BATATA DOCE;MILHO',
    'SWEET POTATO; CORN', 'SWEET POTATO; CORN; COURGETTE', 'SWEET POTATO; POTATO',
    'SWEET POTATO; BEANS', 'BATATA;CEBOLA', 'BATATA;CONSOCIAÇÕES ANUAIS E OUTRAS CULT. FORRAG. ANUAIS',
    'POTATO; ONION', 'POTATO; ANNUAL AND OTHER CULT. FORAGE ANNUALS', 'POTATOES; OTHER VEGETABLES'
}
BEETS = {'BEETROOT', 'BEETROOT_BEETS'}
LEGUMES = {
    'BEAN', 'LUPINE', 'TREMOCILHA', 'Lupinus luteus', 'FAVA', 'ERVILHA', 'GRÃO DE BICO', 'PEA', 'CHICKPEA',
    'BEANS; BROAD BEAN', 'BEAN; PEA; OTHER VEGETABLES; BROAD BEAN', 'sweet_lupins', 'beans', 'peas', 'chickpeas',
    'LUPINE; BROAD BEAN', 'FEIJÃO;BATATA', 'BEANS; POTATOES', 'FEIJÃO;OUTRAS HORTÍCOLAS', 'BEANS; OTHER VEGETABLES',
    'FEIJÃO;BATATA;OUTRAS HORTÍCOLAS', 'BEAN; POTATO; OTHER VEGETABLES', 'FEIJÃO;NABO', 'BEANS; TURNIP',
    'FEIJÃO;CEBOLA', 'BEANS; ONION', 'COUVE;TREMOÇO;FAVA', 'CABBAGE; LUPIN; BEAN BEAN'
}
VEGETABLES = {
    'OTHER VEGETABLES', 'ABÓBORAS E ABOBORINHAS', 'PUMPKINS AND COURGETTES', 'COURGETTE', 'MELÃO', 'MELANCIA',
    'ALHO FRANCÊS', 'ALHO', 'ALFACE', 'NABO', 'NABIÇA', 'CENOURA', 'TOMATE', 'COUVE', 'GREEN CABBAGE',
    'PIMENTO', 'ESPINAFRE', 'pumpkin_squash_gourd', 'zucchini_courgette', 'fresh_vegetables', 'melon',
    'watermelon', 'garlic', 'onions', 'turnips', 'carrots_daucus', 'tomato', 'spinach', 'cress',
    'salads_lettuce_leaf_vegetables', 'brassica_oleracea_cabbage', 'piper_pepper',
    'PUMPKINS AND COURGETTES; POTATOES; OTHER VEGETABLES', 'PUMPKINS AND COURGETTES; POTATOES',
    'PUMPKINS AND COURGETTES; OTHER VEGETABLES', 'PUMPKINS AND COURGETTES; OATS', 'PUMPKINS AND PUMPKINS; CORN; OTHER VEGETABLES',
    'PUMPKINS AND COURGETTES; CORN; TURNIP; POTATO', 'PUMPKINS AND COURGETTES; CHICKPEA; CORN; BEAN; POTATO; OTHER VEGETABLES; ANNUAL AND OTHER CULT. FORAGE ANNUALS',
    'PUMPKINS AND COURGETTES; LUPIN', 'PUMPKINS AND COURGETTES; TEMPORARY MEADOWS', 'PUMPKINS AND COURGETTES; ONION',
    'CABBAGE; OATS', 'CABBAGE;POTATO;ONION', 'COUVE;BATATA;CEBOLA', 'TURNIP; TURNIP', 'NABIÇA;NABO',
    'OTHER VEGETABLES; POTATOES', 'OTHER VEGETABLES; BEAN', 'OTHER VEGETABLES; LUPINS',
    'OTHER VEGETABLES; POTATOES; ANNUAL CONSOCIATIONS AND OTHER CULT. FORAGE ANNUALS', 'OTHER VEGETABLES; ONION',
    'lolium; OTHER VEGETABLES', 'TEMPORARY MEADOWS; OTHER VEGETABLES', 'FALLOW; OTHER VEGETABLES',
    'MELON; OTHER VEGETABLES; ANNUAL AND OTHER CULT. FORAGE ANNUALS'
}
FRUITS = {
    'PERA', 'MAÇÃ', 'PÊSSEGO', 'AMEIXA', 'MARMELO', 'CEREJA', 'GINJA', 'DAMASCO', 'KIWI', 'FIGO', 'FIGO DA INDIA',
    'MORANGO', 'FRAMBOESA', 'AMORA', 'MIRTILO', 'OUTRAS FRUTOS FRESCOS', 'OUTROS PEQUENOS FRUTOS', 'OUTROS FRUTOS SUB-TROPICAIS',
    'orchards_fruits', 'apples', 'pears', 'quinces', 'plums', 'apricots', 'cherry_cherries', 'blueberry',
    'blackberry', 'raspberry_raspberries', 'strawberries', 'fig', 'persimmon', 'kiwi', 'avocado',
    'sour cherry', 'plum', 'peach', 'cherry', 'apricot', 'fig_da_india', 'strawberry', 'raspberry',
    'blackberry', 'blueberry'
}
CITRUS = {'LARANJA', 'LIMÃO', 'OUTROS CITRINOS', 'citrus_plantations'}
NUTS = {
    'CASTANHA', 'AMENDOA', 'NOZ', 'PINHÃO', 'AMENDOIM', 'PISTACIOS', 'sweet_chestnuts', 'almond', 'nuts', 'pistachio',
    'CHESTNUT PLANTATION; OTHER HARDWOOD PLANTATION', 'POVOAMENTO CASTANHEIRO;POVOAMENTO OUTRAS FOLHOSAS',
    'POVOAMENTO CASTANHEIRO', 'CHESTNUT PLANTATIONS'
}
FALLOW = {'POUSIO', 'fallow_land_not_crop', 'FALLOWING/ INTERCROPPING (INTERRUPTED CULTIVATEN TO MAKE SOIL MORE FERTILE)'}
FORESTS = {
    'SOBREIRO', 'POVOAMENTO DE SOBREIROS', 'POVOAMENTO AZINHEIRAS', 'POVOAMENTO DE PINHEIRO MANSO',
    'POVOAMENTO DE EUCALIPTO', 'POVOAMENTO OUTRAS FOLHOSAS', 'POVOAMENTO OUTRAS RESINOSAS', 'POVOAMENTO F MISTO',
    'BOSQUETES', 'ACEIRO FLORESTAL', 'GALERIA RIPÍCOLA', 'OUTRAS SUPERFÍCIES FLORESTAIS', 'tree_wood_forest',
    'oak', 'eucalyptus', 'CORK OAK FOR CORK PRODUCTION', 'CORK OAK PLANTATION', 'PLANTATION OF OTHER HARDWOODS',
    'EVERGREEN OAK PLANTATION', 'PINE TREES PLANTATION', 'OTHER CONIFEROUS FORESTS', 'MIXED Forest',
    'EUCALYPTUS PLANTATION', 'BLACK OAK NEGRAL PLANTATION', 'FOREST FIREBREAKS', 'RIPARIAN GALLERY',
    'OTHER FOREST SURFACES', 'ELEGIBLE LANDSCAPE FEATURES - HEDGES AND WINDBREAKS', 'RIPARIAN GALLERY',
    'SETTLEMENT OF PINE TREES; OTHER FOREST SURFACES', 'OTHER FOREST SURFACES; CORK GROVE',
    'PLANTATION OF OTHER HARDWOODS; PLANTATION OF CORK OAK', 'SETTLEMENT OF OTHER HARDWOODS; SETTLEMENT OF OTHER RESIN',
    'PLANTATION OF OTHER HARDWOODS; PLANTATION OF OTHER RESIN; OTHER F MIXED PLANTATION', 'SETTLEMENT OTHER HARDWOOD; F MIXED SETTLEMENT',
    'POVOAMENTO OUTRAS FOLHOSAS;POVOAMENTO DE SOBREIROS', 'POVOAMENTO OUTRAS FOLHOSAS;POVOAMENTO OUTRAS RESINOSAS',
    'POVOAMENTO OUTRAS FOLHOSAS;POVOAMENTO OUTRAS RESINOSAS;POVOAMENTO F MISTO', 'POVOAMENTO OUTRAS FOLHOSAS;POVOAMENTO F MISTO',
    'POVOAMENTO DE PINHEIRO MANSO;OUTRAS SUPERFÍCIES FLORESTAIS', 'OUTRAS SUPERFÍCIES FLORESTAIS;POVOAMENTO DE SOBREIROS'
}

CROP_MAPPING_IDS = {
    "Grassland & Pastures": 1,
    "Olive Groves": 2,
    "Vineyards": 3,
    "Maize": 4,
    "Rice": 5,
    "Wheat": 6,
    "Barley": 7,
    "Oats": 8,
    "Rye": 9,
    "Triticale": 10,
    "Other Cereals": 11,
    "Potatoes": 12,
    "Beets": 13,
    "Legumes & Pulses": 14,
    "Vegetables": 15,
    "Orchards & Fruits": 16,
    "Citrus": 17,
    "Nuts": 18,
    "Fallow Land": 19,
    "Forests & Woodlands": 20
}

# Reverse mapping to look up class by crop name
def get_class_name(crop_name):
    if not crop_name or pd.isna(crop_name) or crop_name.strip() == '' or crop_name == 'NOT KNOWN':
        return None
    if crop_name in GRASSLAND_CROPS: return "Grassland & Pastures"
    if crop_name in OLIVES: return "Olive Groves"
    if crop_name in VINEYARDS: return "Vineyards"
    if crop_name in MAIZE: return "Maize"
    if crop_name in RICE: return "Rice"
    if crop_name in WHEAT: return "Wheat"
    if crop_name in BARLEY: return "Barley"
    if crop_name in OATS: return "Oats"
    if crop_name in RYE: return "Rye"
    if crop_name in TRITICALE: return "Triticale"
    if crop_name in OTHER_CEREALS: return "Other Cereals"
    if crop_name in POTATOES: return "Potatoes"
    if crop_name in BEETS: return "Beets"
    if crop_name in LEGUMES: return "Legumes & Pulses"
    if crop_name in VEGETABLES: return "Vegetables"
    if crop_name in FRUITS: return "Orchards & Fruits"
    if crop_name in CITRUS: return "Citrus"
    if crop_name in NUTS: return "Nuts"
    if crop_name in FALLOW: return "Fallow Land"
    if crop_name in FORESTS: return "Forests & Woodlands"
    return None

def classify_crop(name):
    if not name or pd.isna(name) or name.strip() == '' or name == 'NOT KNOWN':
        return 'NOT KNOWN'
    cname = get_class_name(name)
    if cname: return cname
    return 'OTHER UNCLASSIFIED'

def prepare_samples():
    logging.info("--- STEP 1: Creating Stratified Samples Shapefile ---")
    start = time.time()
    
    # Read columns and geometry
    logging.info("Reading PT Shapefile...")
    gdf = gpd.read_file(SHP_INPUT_PATH, engine="pyogrio")
    logging.info(f"Loaded {len(gdf)} parcels.")
    
    # Project to metric CRS (EPSG:3763) for area and interior centroids
    logging.info("Reprojecting to Portuguese National System (EPSG:3763)...")
    gdf_metric = gdf.to_crs("EPSG:3763")
    
    # Apply class mapping
    logging.info("Mapping crop classes...")
    gdf_metric["crop_name"] = gdf_metric["EC_trans_n"].apply(get_class_name)
    gdf_metric = gdf_metric[gdf_metric["crop_name"].notna()].copy()
    gdf_metric["crop_id"] = gdf_metric["crop_name"].map(CROP_MAPPING_IDS)
    
    # Calculate area in ha
    gdf_metric["area_ha"] = gdf_metric.geometry.area / 10000.0
    
    # Area filter: fields >= 0.1 ha (optimized for small fields in Portugal)
    min_area_ha = 0.1
    gdf_filtered = gdf_metric[gdf_metric["area_ha"] >= min_area_ha].copy()
    logging.info(f"Filtered to fields >= {min_area_ha} ha. Remaining: {len(gdf_filtered)}")
    
    # Calculate interior centroids (to guarantee point falls within polygon boundary)
    logging.info("Calculating interior centroids...")
    centroids = gdf_filtered.geometry.centroid
    is_inside = gdf_filtered.geometry.contains(centroids)
    gdf_cnt = gdf_filtered[is_inside].copy()
    gdf_cnt.geometry = centroids[is_inside]
    logging.info(f"Valid interior centroids found: {len(gdf_cnt)}")
    
    # Stratified Sampling: 1000 per class
    samples_per_class = 1000
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

def generate_agricultural_masks():
    logging.info("--- STEP 2: Generating Country-Wide Agricultural Masks ---")
    start = time.time()
    
    # We will read only crop_name and geometry
    logging.info("Loading parcel geometries...")
    gdf = gpd.read_file(SHP_INPUT_PATH, engine="pyogrio", columns=["EC_trans_n"])
    gdf["Class"] = gdf["EC_trans_n"].apply(classify_crop)
    
    # Project to Web Mercator (EPSG:3857) for masks
    logging.info("Reprojecting parcels to EPSG:3857...")
    gdf_3857 = gdf.to_crs("EPSG:3857")
    
    # Use exact bounds of the crop geometries (mainland Portugal Continental only, excluding Azores/Madeira)
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    logging.info(f"Mainland crop bounds in EPSG:3857: minx={minx:.1f}, miny={miny:.1f}, maxx={maxx:.1f}, maxy={maxy:.1f}")
    
    # Align to 10m grid
    pixel_size = 10.0
    minx = np.floor(minx / pixel_size) * pixel_size
    miny = np.floor(miny / pixel_size) * pixel_size
    maxx = np.ceil(maxx / pixel_size) * pixel_size
    maxy = np.ceil(maxy / pixel_size) * pixel_size
    
    cols = int((maxx - minx) / pixel_size)
    rows = int((maxy - miny) / pixel_size)
    logging.info(f"Target raster size: {cols} cols x {rows} rows ({cols*rows/1e6:.1f} Mpx)")
    
    # Define filters:
    # Wariant B (allcrops): Include everything EXCEPT forests/woodlands
    gdf_allcrops = gdf_3857[gdf_3857["Class"] != "Forests & Woodlands"].copy()
    
    # Wariant A (3class): Exclude permanent crops (Grassland, Olives, Vineyards, Orchards, Citrus, Nuts, Forests)
    exclude_arable = {"Grassland & Pastures", "Olive Groves", "Vineyards", "Orchards & Fruits", "Citrus", "Nuts", "Forests & Woodlands"}
    gdf_arable = gdf_3857[~gdf_3857["Class"].isin(exclude_arable)].copy()
    
    temp_arable_shp = os.path.join(MASKS_OUT_DIR, "temp_arable.shp")
    temp_allcrops_shp = os.path.join(MASKS_OUT_DIR, "temp_allcrops.shp")
    
    logging.info("Saving temporary shapefiles for rasterization...")
    gdf_arable.to_file(temp_arable_shp, engine="pyogrio")
    gdf_allcrops.to_file(temp_allcrops_shp, engine="pyogrio")
    
    def rasterize_to_file(shp_path, out_tif_path, description):
        logging.info(f"Rasterizing {description}...")
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            out_tif_path,
            cols, rows, 1,
            gdal.GDT_Byte,
            options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
        )
        out_ds.SetGeoTransform([minx, pixel_size, 0, maxy, 0, -pixel_size])
        
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        out_ds.SetProjection(srs.ExportToWkt())
        
        # Fill with 0
        band = out_ds.GetRasterBand(1)
        band.Fill(0)
        
        # Rasterize: Burn 1
        gdal.Rasterize(out_ds, shp_path, burnValues=[1], allTouched=False)
        out_ds.FlushCache()
        out_ds = None
        logging.info(f"Saved mask to {out_tif_path}")

    # Generate both masks
    rasterize_to_file(temp_arable_shp, MASK_3CLASS_PATH, "Arable Crops & Unknown (3-Class Variant)")
    rasterize_to_file(temp_allcrops_shp, MASK_ALLCROPS_PATH, "All Agricultural Crops & Unknown Variant")
    
    # Clean up temp shapefiles
    logging.info("Cleaning up temporary shapefiles...")
    for temp_shp in [temp_arable_shp, temp_allcrops_shp]:
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            p = temp_shp.replace('.shp', ext)
            if os.path.exists(p):
                os.remove(p)
                
    logging.info(f"Masks generation finished in {time.time() - start:.1f} seconds.")

def main():
    prepare_samples()
    generate_agricultural_masks()
    logging.info("PT Crop Data Preparation Complete!")

if __name__ == "__main__":
    main()
