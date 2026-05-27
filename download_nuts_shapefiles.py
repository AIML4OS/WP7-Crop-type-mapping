import os
import pathlib
import urllib.request
import zipfile
import shutil
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/ref-nuts-2021-20m.shp.zip"
WORKSPACE_DIR = pathlib.Path("d:/AIML_CropMapper_Cloud")
TEMP_DIR = WORKSPACE_DIR / "workingDir" / "temp_nuts"
OUTPUT_BASE_DIR = WORKSPACE_DIR / "auxiliary_files" / "shapefiles_nuts"

def main():
    try:
        import geopandas as gpd
    except ImportError:
        logging.error("geopandas is not installed in this environment! Please run under the conda environment with geopandas.")
        sys.exit(1)

    # 1. Prepare directories
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TEMP_DIR / "nuts_2021.zip"

    # 2. Download ZIP
    logging.info(f"Downloading NUTS shapefiles from {URL}...")
    try:
        urllib.request.urlretrieve(URL, zip_path)
        logging.info("Download completed successfully.")
    except Exception as e:
        logging.error(f"Failed to download NUTS shapefiles: {e}")
        sys.exit(1)

    # 3. Extract ZIP
    extract_dir = TEMP_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Extracting ZIP to {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logging.info("Extraction completed.")
    except Exception as e:
        logging.error(f"Failed to extract ZIP: {e}")
        sys.exit(1)

    # 4. Extract nested zip files
    nested_zips = list(extract_dir.glob("**/*RG*4326*.zip"))
    if not nested_zips:
        nested_zips = list(extract_dir.glob("**/*.zip"))
    
    nested_extract_dir = extract_dir / "nested_extracted"
    nested_extract_dir.mkdir(parents=True, exist_ok=True)
    
    for nz in nested_zips:
        logging.info(f"Extracting nested ZIP: {nz.name} to {nested_extract_dir}...")
        try:
            with zipfile.ZipFile(nz, 'r') as zip_ref:
                zip_ref.extractall(nested_extract_dir)
        except Exception as e:
            logging.error(f"Failed to extract nested ZIP {nz}: {e}")

    # Now find the shapefile in the nested extraction
    shp_files = list(nested_extract_dir.glob("**/*.shp"))
    if not shp_files:
        shp_files = list(extract_dir.glob("**/*.shp"))
    
    if not shp_files:
        logging.error("No .shp files found in the extracted archives!")
        sys.exit(1)
        
    # Prefer files with 'RG' and '4326' in the name
    rg_files = [f for f in shp_files if "RG" in f.name and "4326" in f.name]
    if not rg_files:
        rg_files = [f for f in shp_files if "RG" in f.name]
    if rg_files:
        shp_path = rg_files[0]
    else:
        shp_path = shp_files[0]
        
    logging.info(f"Reading shapefile: {shp_path}")
    
    # 5. Read shapefile and process
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        logging.error(f"Failed to read shapefile with GeoPandas: {e}")
        sys.exit(1)

    logging.info(f"Shapefile loaded. Total features: {len(gdf)}")
    logging.info(f"Columns: {list(gdf.columns)}")

    # Standard columns in GISCO NUTS shapefiles:
    # 'LEVL_CODE': NUTS Level (0, 1, 2, 3)
    # 'CNTR_CODE': Country Code (AT, PL, DE, etc.)
    # 'NUTS_ID': Region ID (AT11, PL21, etc.)
    
    # Check if necessary columns exist
    if 'LEVL_CODE' not in gdf.columns or 'CNTR_CODE' not in gdf.columns:
        # Let's inspect columns to see if names match (sometimes they are lowercase or slightly different)
        col_map = {col.upper(): col for col in gdf.columns}
        levl_col = col_map.get('LEVL_CODE')
        cntr_col = col_map.get('CNTR_CODE')
        if not levl_col or not cntr_col:
            logging.error(f"Could not find LEVL_CODE or CNTR_CODE columns in shapefile. Columns: {gdf.columns}")
            sys.exit(1)
    else:
        levl_col = 'LEVL_CODE'
        cntr_col = 'CNTR_CODE'

    # Filter NUTS Level 2 (LEVL_CODE == 2)
    # Note: LEVL_CODE might be string or integer depending on how shapefile is loaded.
    gdf_level2 = gdf[gdf[levl_col].astype(str) == '2']
    logging.info(f"Filtered to NUTS Level 2. Total NUTS2 features: {len(gdf_level2)}")

    if len(gdf_level2) == 0:
        logging.error("No NUTS Level 2 features found! Please check columns or dataset.")
        sys.exit(1)

    # 6. Group and save for each country
    countries = gdf_level2[cntr_col].unique()
    logging.info(f"Found {len(countries)} country codes: {sorted(countries)}")

    for country in countries:
        country_code = str(country).upper()
        country_gdf = gdf_level2[gdf_level2[cntr_col] == country]
        
        # Save output path
        country_out_dir = OUTPUT_BASE_DIR / country_code
        country_out_dir.mkdir(parents=True, exist_ok=True)
        out_shp_path = country_out_dir / f"NUTS2_{country_code}.shp"
        
        logging.info(f"Saving {len(country_gdf)} NUTS2 regions for {country_code} to {out_shp_path}...")
        try:
            country_gdf.to_file(out_shp_path)
        except Exception as e:
            logging.error(f"Failed to save shapefile for {country_code}: {e}")

        # If it's Greece ('EL'), also write to 'GR'
        if country_code == "EL":
            gr_out_dir = OUTPUT_BASE_DIR / "GR"
            gr_out_dir.mkdir(parents=True, exist_ok=True)
            gr_shp_path = gr_out_dir / "NUTS2_GR.shp"
            logging.info(f"Greece detected. Also saving to GR: {gr_shp_path}")
            try:
                # We need to update CNTR_CODE to 'GR' in the copy to keep it consistent
                gr_gdf = country_gdf.copy()
                gr_gdf[cntr_col] = "GR"
                gr_gdf.to_file(gr_shp_path)
            except Exception as e:
                logging.error(f"Failed to save shapefile for GR: {e}")

    # 7. Cleanup
    logging.info("Cleaning up temporary download files...")
    try:
        shutil.rmtree(TEMP_DIR)
        logging.info("Cleanup completed successfully.")
    except Exception as e:
        logging.warning(f"Failed to clean up temporary folder {TEMP_DIR}: {e}")

    logging.info("Successfully finished building EU NUTS2 Shapefile Database!")

if __name__ == "__main__":
    main()
