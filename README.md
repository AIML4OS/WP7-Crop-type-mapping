# Sentinel-1 OBIA Crop Type Mapping Pipeline (v2.0)

A modular, automated, and optimized Earth Observation pipeline for Sentinel-1 GRD SAR data preprocessing, object-based image analysis (OBIA), and crop classification using Artificial Neural Networks (ANN) and Segment Anything (SAM). 

This version (v2.0) introduces dynamic country-wide processing, automatic relative orbit optimization with Set Cover, separation of Ascending/Descending tracks, and seamless multi-orbit merging for any European country.

---

## Directory Structure
- `1_Sentinel-1_preprocessor/`: Calibration, slice assembly, coregistration, and temporal stacking/clipping.
- `2_OBIA_classifier/`: Segmentation (SAM/Felzenszwalb), feature extraction, training, classification, and country-level merging.
- `download_nuts_shapefiles.py`: Downloader and builder for European NUTS2 administrative boundaries.
- `auxiliary_files/`: Excluded from git. Contains auxiliary shapefiles, raster masks, and models.
- `workingDir/`: Excluded from git. Output workspace for intermediate and final rasters.

---

## Setup & Prerequisites
1. **Python Environment**: Use the `satmirol_env` Conda environment.
2. **SNAP**: ESA SNAP (specifically `gpt.exe`) must be installed and configured. Ensure the environment variable `SNAP_GPT_EXE` points to the correct executable path.
3. **GDAL**: GDAL bindings for Python are required (bundled inside `satmirol_env`).

---

## Step-by-Step Execution Guide

### Step 1: Build the NUTS2 Administrative Boundaries Database
Before running preprocessing, download the official Eurostat NUTS 2021 shapefiles. This script downloads the zipped boundaries, filters NUTS2 level regions, and extracts them into structured folders per country.

**Execution Command:**
```bash
python download_nuts_shapefiles.py
```
- **Outputs**: Individual country boundaries are stored in `auxiliary_files/shapefiles_nuts/{COUNTRY}/NUTS2_{COUNTRY}.shp`.

---

### Step 2: Prepare the Agricultural Mask
To restrict the classification to agricultural areas, the pipeline uses the Copernicus High Resolution Layer (HRL) Crop Type 2023 product.

#### A. Manual Data Download
1. Log in to the [Copernicus Land Monitoring Service (CLMS) Portal](https://land.copernicus.eu/).
2. Search for the **High Resolution Layer: Crop Type 2023 (HRL CTY 2023)**.
3. Download the zipped product tiles (`.zip`) covering your target country.
4. Place the downloaded ZIP files in the following directory:
   `auxiliary_files/raster_files/AgriMasks/<COUNTRY>/Results/`
   *(e.g., `auxiliary_files/raster_files/AgriMasks/PL/Results/` for Poland).*

#### B. Generate the Binary Mask
Once the ZIP files are placed, run `build_agri_mask.py` to extract, reclassify (into a binary 0/1 mask), mosaic, reproject to `EPSG:3857`, and clip the mask to the country boundary.

**Execution Command:**
```bash
python 2_OBIA_classifier/build_agri_mask.py --country <COUNTRY>
```
- **Outputs**: 
  - `auxiliary_files/raster_files/AgriMasks/<COUNTRY>/<COUNTRY>_agri_mask_3class_epsg3857.tif` (arable crops only)
  - `auxiliary_files/raster_files/AgriMasks/<COUNTRY>/<COUNTRY>_agri_mask_allcrops_epsg3857.tif` (all crops, including permanent)

---

### Step 3: Sentinel-1 Slice Calibration & Assembly
This script scans the raw Sentinel-1 GRD SAFE repository, determines the best pass direction (Ascending or Descending) for the country, runs the Greedy Set Cover solver to select the minimum number of relative orbits, and processes each orbit sequentially.

**Execution Command:**
```bash
python 1_Sentinel-1_preprocessor/1_AIML_S1_slice_calibration.py -s <START_DATE> -e <END_DATE> -c <COUNTRY>
```
*Example (Poland, 2024-10-15 to 2024-11-30):*
```bash
python 1_Sentinel-1_preprocessor/1_AIML_S1_slice_calibration.py -s 2024-10-15 -e 2024-11-30 -c PL
```
- **Note**: For Cloud-Optimized GeoTIFFs (COG), run the alternative `1_AIML_S1_slice_calibration_COG.py` script.
- **Outputs**: Sliced and calibrated `.dim` files stored sequentially in `workingDir/<COUNTRY>/orbit_<ORBIT_NUMBER>/slice_assembly/`.

---

### Step 4: Temporal Stack Coregistration
Run coregistration to align all time-series slices for each orbit. This runs a cross-correlation and warp inside SNAP sequentially.

**Execution Command:**
```bash
python 1_Sentinel-1_preprocessor/2_AIML_S1_coregistration.py -t <COUNTRY>/orbit_<ORBIT>
```
*Example (multiple orbits):*
```bash
python 1_Sentinel-1_preprocessor/2_AIML_S1_coregistration.py -t PL/orbit_12 PL/orbit_88
```
- **Outputs**: Calibrated, coregistered VH/VV polarizations under `workingDir/<COUNTRY>/orbit_<ORBIT>/S1_final_preprocessing/`.

---

### Step 5: Stack Clipping
Create a virtual raster stack (VRT) for all dates and clip the stack using GDAL to the country's NUTS2 boundaries.

**Execution Command:**
```bash
python 1_Sentinel-1_preprocessor/3_AIML_S1_stack_clip.py -t <COUNTRY>/orbit_<ORBIT>
```
*Example:*
```bash
python 1_Sentinel-1_preprocessor/3_AIML_S1_stack_clip.py -t PL/orbit_12 PL/orbit_88
```
- **Outputs**: Clipped Multi-temporal GeoTIFF stacks saved to `workingDir/<COUNTRY>/orbit_<ORBIT>/processed_raster/`.

---

### Step 6: Object-Based Image Analysis (OBIA) Classifier
Generate segmentation (using SAM or Felzenszwalb), extract object features (mean, std deviation per segment across time), train an Artificial Neural Network (MLP), and classify the segments.

**Execution Command (with SAM):**
```bash
python 2_OBIA_classifier/1_OBIA_vector_classifier_modular_ANN_SAM.py --track <COUNTRY>/orbit_<ORBIT>
```
**Execution Command (Standard ANN - faster):**
```bash
python 2_OBIA_classifier/1_OBIA_vector_classifier_modular_ANN.py --track <COUNTRY>/orbit_<ORBIT>
```
*Example:*
```bash
python 2_OBIA_classifier/1_OBIA_vector_classifier_modular_ANN.py --track PL/orbit_12
```
- **Outputs**: Classification maps, confidence maps, and validation spreadsheets under `workingDir/<COUNTRY>/orbit_<ORBIT>/classification_results/`.

---

### Step 7: Country-Wide Classification Merging
After classifying all orbits for a country, warp and merge them into a single, seamless country-level classification map.

**Execution Command:**
```bash
python 2_OBIA_classifier/2_OBIA_merge_classifications.py --track <COUNTRY>
```
*Example:*
```bash
python 2_OBIA_classifier/2_OBIA_merge_classifications.py --track PL
```
- **Outputs**: Final merged GeoTIFF classification map at `workingDir/<COUNTRY>/classification_results/<COUNTRY>_final_classification.tif` and unified verification sheets.

---

## Computational Efficiency & Resource Management
- **Sequential Processing**: To prevent Out Of Memory (OOM) errors and CPU over-allocation by SNAP and GDAL, all scripts process orbits and time series sequentially.
- **Cache Optimization**: SNAP GPT instances run with configured limits (`-q 4`) to bound process memory consumption.
- **Disk Cleanups**: Output `.dim` stacks and temporary files are retained for verification but can be cleaned up manually once final classification merging is complete.
