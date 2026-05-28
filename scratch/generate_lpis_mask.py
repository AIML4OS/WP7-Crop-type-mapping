import os
import json
import urllib.request
import urllib.parse
import geopandas as gpd
import numpy as np
from shapely.geometry import shape
from osgeo import gdal, osr
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Output paths
OUTPUT_DIR = r"d:\AIML_CropMapper_Cloud\auxiliary_files\raster_files\AgriMasks\NL"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_MASK_PATH = os.path.join(OUTPUT_DIR, "NL_agri_mask_allcrops_epsg3857.tif")

# Bounding box for Flevoland (NL23) in EPSG:4326
BBOX = "5.0604,52.2496,5.9684,52.8065"

def fetch_all_parcels():
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
    max_features = 40000  # Fetch up to 40,000 features to get a complete representation of Flevoland fields
    
    logging.info("Downloading all agricultural parcels from PDOK (Flevoland BBOX)...")
    
    while url and len(features_list) < max_features:
        logging.info(f"Downloading page {page} (Parcels collected: {len(features_list)})...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            curr_features = data.get("features", [])
            if not curr_features:
                logging.info("No more features found.")
                break
                
            features_list.extend(curr_features)
            
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
            
    logging.info(f"Download complete. Total parcels fetched: {len(features_list)}")
    return features_list

def rasterize_parcels(features):
    logging.info("Parsing parcel geometries...")
    geoms = []
    for feat in features:
        geom_dict = feat.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
            geoms.append(geom)
        except:
            continue
            
    # Create GeoDataFrame in WGS84
    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
    logging.info(f"Reprojecting to Web Mercator (EPSG:3857)...")
    gdf_3857 = gdf.to_crs("EPSG:3857")
    
    # Dissolve to speed up rasterization and create a clean boundary mask
    logging.info("Dissolving geometries...")
    dissolved = gdf_3857.unary_union
    
    # Calculate bounding box bounds in EPSG:3857
    minx, miny, maxx, maxy = dissolved.bounds
    logging.info(f"Bounds in EPSG:3857: minx={minx:.1f}, miny={miny:.1f}, maxx={maxx:.1f}, maxy={maxy:.1f}")
    
    # Align to 10m grid
    pixel_size = 10.0
    minx = np.floor(minx / pixel_size) * pixel_size
    miny = np.floor(miny / pixel_size) * pixel_size
    maxx = np.ceil(maxx / pixel_size) * pixel_size
    maxy = np.ceil(maxy / pixel_size) * pixel_size
    
    cols = int((maxx - minx) / pixel_size)
    rows = int((maxy - miny) / pixel_size)
    
    logging.info(f"Raster dimensions: cols={cols}, rows={rows}")
    
    # Create temporary Shapefile to use with gdal.Rasterize
    temp_shp = os.path.join(OUTPUT_DIR, "temp_parcels.shp")
    gpd.GeoDataFrame(geometry=[dissolved], crs="EPSG:3857").to_file(temp_shp, engine="pyogrio")
    
    # Setup output GeoTIFF using GDAL
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        OUTPUT_MASK_PATH,
        cols, rows, 1,
        gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    )
    
    # Set geotransform (top-left X, X resolution, 0, top-left Y, 0, -Y resolution)
    out_ds.SetGeoTransform([minx, pixel_size, 0, maxy, 0, -pixel_size])
    
    # Set projection EPSG:3857
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    out_ds.SetProjection(srs.ExportToWkt())
    
    # Initialize with 0
    band = out_ds.GetRasterBand(1)
    band.Fill(0)
    
    # Rasterize: burn value 1 into the raster
    logging.info("Rasterizing LPIS parcels...")
    gdal.Rasterize(out_ds, temp_shp, burnValues=[1])
    
    out_ds.FlushCache()
    out_ds = None
    
    # Clean up temporary shapefile
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        p = temp_shp.replace('.shp', ext)
        if os.path.exists(p):
            os.remove(p)
            
    logging.info(f"Agricultural mask successfully saved to: {OUTPUT_MASK_PATH}")

def main():
    parcels = fetch_all_parcels()
    if not parcels:
        logging.error("No parcels fetched.")
        return
    rasterize_parcels(parcels)

if __name__ == "__main__":
    main()
