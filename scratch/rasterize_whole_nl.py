import os
import geopandas as gpd
from osgeo import gdal, osr
import numpy as np
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

GPKG_PATH = r"d:\AIML_CropMapper_Cloud\auxiliary_files\raster_files\AgriMasks\NL\brpgewaspercelen_definitief_2025.gpkg"
OUTPUT_MASK_PATH = r"d:\AIML_CropMapper_Cloud\auxiliary_files\raster_files\AgriMasks\NL\NL_agri_mask_allcrops_epsg3857.tif"

def main():
    if not os.path.exists(GPKG_PATH):
        logging.error(f"File not found: {GPKG_PATH}")
        logging.error("Please download the 2.93 GB GPKG from:")
        logging.error("https://service.pdok.nl/rvo/gewaspercelen/atom/downloads/brpgewaspercelen_definitief_2025.gpkg")
        sys.exit(1)
        
    start_time = time.time()
    logging.info("Loading whole Netherlands BRP 2025 GeoPackage (only geometry)...")
    
    # We read using pyogrio for fast multi-threaded reading
    try:
        gdf = gpd.read_file(GPKG_PATH, engine="pyogrio", columns=[])
    except Exception as e:
        logging.error(f"Failed to read GPKG: {e}")
        sys.exit(1)
        
    logging.info(f"Loaded {len(gdf)} parcels. Reprojecting to EPSG:3857...")
    gdf_3857 = gdf.to_crs("EPSG:3857")
    
    # Calculate bounding box bounds
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    logging.info(f"Netherlands Bounds (EPSG:3857): minx={minx:.1f}, miny={miny:.1f}, maxx={maxx:.1f}, maxy={maxy:.1f}")
    
    # Align to 10m grid
    pixel_size = 10.0
    minx = np.floor(minx / pixel_size) * pixel_size
    miny = np.floor(miny / pixel_size) * pixel_size
    maxx = np.ceil(maxx / pixel_size) * pixel_size
    maxy = np.ceil(maxy / pixel_size) * pixel_size
    
    cols = int((maxx - minx) / pixel_size)
    rows = int((maxy - miny) / pixel_size)
    
    logging.info(f"Creating output raster: {cols} x {rows} pixels...")
    
    # Save a temporary shapefile to feed into GDAL rasterize
    temp_shp = os.path.join(os.path.dirname(OUTPUT_MASK_PATH), "temp_nl_bounds.shp")
    gdf_3857.to_file(temp_shp, engine="pyogrio")
    
    # Create GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        OUTPUT_MASK_PATH,
        cols, rows, 1,
        gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'BIGTIFF=YES']
    )
    
    out_ds.SetGeoTransform([minx, pixel_size, 0, maxy, 0, -pixel_size])
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    out_ds.SetProjection(srs.ExportToWkt())
    
    band = out_ds.GetRasterBand(1)
    band.Fill(0)
    
    logging.info("Rasterizing all parcels (burning value 1)...")
    gdal.Rasterize(out_ds, temp_shp, burnValues=[1])
    
    out_ds.FlushCache()
    out_ds = None
    
    # Cleanup temp shapefile
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        p = temp_shp.replace('.shp', ext)
        if os.path.exists(p):
            os.remove(p)
            
    logging.info(f"Success! Whole country agricultural mask saved to: {OUTPUT_MASK_PATH}")
    logging.info(f"Process took {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()
