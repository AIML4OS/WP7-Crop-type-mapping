# Sentinel-1 OBIA Crop Type Mapping Pipeline (v2.0)

An automated, object-based image analysis (OBIA) pipeline designed to process Sentinel-1 SAR time series and perform crop type classification using Artificial Neural Networks (ANN) and Segment Anything (SAM).

This version (v2.0) introduces dynamic country-level orbit optimization (using the Set Cover algorithm), separate processing of Ascending/Descending tracks, and automatic country-wide classification merging.

---

## Table of Contents
1. [Overview for Non-Experts](#overview-for-non-experts)
2. [Prerequisites & System Installation](#prerequisites--system-installation)
   - [Windows Setup](#windows-setup)
   - [Linux Setup](#linux-setup)
3. [Environment Configuration](#environment-configuration)
4. [Training Samples Specification (samples.shp)](#training-samples-specification-samplesshp)
5. [Interactive Menu & Stages Selector (ANN / SAM)](#interactive-menu--stages-selector-ann--sam)
6. [Segment Anything (SAM) Model Setup & Parameters](#segment-anything-sam-model-setup--parameters)
7. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Step 1: Download NUTS2 Boundaries](#step-1-download-nuts2-boundaries)
   - [Step 2: Prepare Copernicus HRL Crop Mask](#step-2-prepare-copernicus-hrl-crop-mask)
   - [Step 3: SAR Slice Calibration & Assembly](#step-3-sar-slice-calibration--assembly)
   - [Step 4: Stack Coregistration](#step-4-stack-coregistration)
   - [Step 5: Stack Clipping](#step-5-stack-clipping)
   - [Step 6: Object-Based Classification](#step-6-object-based-classification)
   - [Step 7: Merge Country Classification](#step-7-merge-country-classification)
8. [Troubleshooting & Performance Tuning](#troubleshooting--performance-tuning)

---

## Overview for Non-Experts
Processing radar satellite data (Sentinel-1) usually involves many complicated manual steps. This toolbox automates the entire process:
1. **Calibration & Slicing**: Converts raw radar backscatter signals (recorded as `.SAFE` directories) into physically meaningful values, merges slices of the same day, and clips them to your country's bounding box.
2. **Coregistration**: Aligns a time-series stack of images taken over several weeks/months so that pixels from different dates match perfectly.
3. **Segmentation & Classification**: Groups similar pixels into "fields" (objects) using image segmentation, calculates stats (mean backscatter values over time), trains a machine learning model, and classifies what crop is growing in each field.

---

## Prerequisites & System Installation

### Windows Setup

1. **Install Python via Miniforge**:
   - Download and install [Miniforge3](https://github.com/conda-forge/miniforge) for Windows.
   - Open **Miniforge Prompt** and create your conda environment:
     ```bash
     conda create -n satmirol_env python=3.10 gdal geopandas scikit-learn pandas openpyxl joblib -y
     conda activate satmirol_env
     ```
   - *Optional (for SAM GPU acceleration)*: Install PyTorch with CUDA:
     ```bash
     conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
     pip install segment-anything
     ```

2. **Install ESA SNAP**:
   - Download the SNAP installer from [ESA SNAP Download](https://step.esa.int/main/download/snap-download/).
   - Run the installer. You can install to the default path on drive `C:\` or choose another drive (e.g., `D:\Program Files\esa-snap`).
   - Locate the path to the executable `gpt.exe` (e.g. `D:\Program Files\esa-snap\bin\gpt.exe`). You will need to supply this exact path in the configuration.

3. **Install Orfeo ToolBox (OTB) (Optional)**:
   - Download OTB 6.2.0 Win64 and extract it to a local folder (e.g., `D:\AIML_CropMapper_Cloud\2_OBIA_classifier\OTB-6.2.0-Win64`).

---

### Linux Setup

1. **Install System Dependencies (Ubuntu/Debian)**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y gdal-bin libgdal-dev build-essential unzip
   ```

2. **Install Python Environment**:
   - Install Miniconda or Miniforge:
     ```bash
     wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
     bash Miniforge3-Linux-x86_64.sh -b
     source ~/miniforge3/bin/activate
     ```
   - Create the Conda environment:
     ```bash
     conda create -n satmirol_env python=3.10 gdal geopandas scikit-learn pandas openpyxl joblib -y
     conda activate satmirol_env
     ```
   - *Optional (for SAM)*:
     ```bash
     conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
     pip install segment-anything
     ```

3. **Install ESA SNAP**:
   - Download the SNAP Linux installer `.sh` script from ESA's website.
   - Run the installer:
     ```bash
     chmod +x esa-snap_sentinel_unix_*.sh
     ./esa-snap_sentinel_unix_*.sh -q
     ```
   - Note the installation path (typically `/usr/local/esa-snap` or `$HOME/esa-snap`). Locate the `gpt` tool in the `bin/` directory.

---

## Environment Configuration

Before running any script, you must configure the following environment variables in your terminal shell. Make sure all paths align with your actual system installation.

### On Windows (PowerShell):
```powershell
# Set the exact path to SNAP gpt.exe (change D: to C: if installed on drive C)
$env:SNAP_GPT_EXE="D:/Program Files/esa-snap/bin/gpt.exe"

# Path to SNAP auxiliary files (where orbit files are cached)
$env:SNAP_AUXDATA_PATH="C:/Users/Administrator/.snap/auxdata"

# Path to the raw Sentinel-1 GRD SAFE repository directory
$env:S1_REPO_PATH="Y:/Sentinel-1/SAR/IW_GRDH_1S"

# Output workspace directory for intermediate and final rasters
$env:AIML_WORKING_DIR="D:/AIML_CropMapper_Cloud/workingDir"

# Path to project's auxiliary files directory
$env:AIML_AUX_DIR="D:/AIML_CropMapper_Cloud/auxiliary_files"
```

### On Linux (Bash):
```bash
# Set the exact path to SNAP gpt
export SNAP_GPT_EXE="/usr/local/esa-snap/bin/gpt"

# Path to SNAP auxiliary files (where orbit files are cached)
export SNAP_AUXDATA_PATH="$HOME/.snap/auxdata"

# Path to the raw Sentinel-1 GRD SAFE repository directory
export S1_REPO_PATH="/mnt/sentinel1/SAR/IW_GRDH_1S"

# Output workspace directory for intermediate and final rasters
export AIML_WORKING_DIR="/home/user/AIML_CropMapper_Cloud/workingDir"

# Path to project's auxiliary files directory
export AIML_AUX_DIR="/home/user/AIML_CropMapper_Cloud/auxiliary_files"
```

---

## Training Samples Specification (samples.shp)

To train the machine learning classifier, you must supply a point vector dataset containing crop reference points.

### 1. File Format & Geometry
- **Format**: ESRI Shapefile or SQLite database.
- **Geometry Type**: **Point** (`OGRPoint`). Polygons are not supported; reference points should lie inside the fields.
- **Coordinate System (CRS)**: Any standard coordinate reference system (e.g. `EPSG:4326` or `EPSG:3857`). The pipeline automatically reprojects the points to match the raster coordinate system.

### 2. Required Attribute Fields
The shapefile attributes table **must** contain an integer column representing the crop classes:
- **`crop_id`** (Integer): Numeric code corresponding to the crop type or land-cover class.
  - *Note*: Positive values are parsed as training samples (e.g., `11` = Winter Wheat, `12` = Maize, `1430` = Rapeseed).
  - *Note*: Value `0` is reserved for background/ignored areas.

### 3. File Directory Hierarchy (Input Paths)
The classifier searches for your reference shapefile in the `shapefiles_samples` directory using the following hierarchy:
1. `auxiliary_files/shapefiles_samples/{COUNTRY}_{SAN_TRACK}/samples.shp` (e.g. `PL_PL_orbit_12/samples.shp`)
2. `auxiliary_files/shapefiles_samples/{COUNTRY}_{TRACK}/samples.shp` (e.g. `PL_PL/orbit_12/samples.shp`)
3. `auxiliary_files/shapefiles_samples/{SAN_TRACK}/samples.shp`
4. `auxiliary_files/shapefiles_samples/{TRACK}/samples.shp`
5. `auxiliary_files/shapefiles_samples/{COUNTRY}/samples.shp` (e.g. `PL/samples.shp` - country-wide fallback)
6. `auxiliary_files/shapefiles_samples/samples.shp` (global fallback)

### 4. Automatic Sample Splitting
In Stage 2, the pipeline automatically splits your `samples.shp` dataset:
- **70%** of the points are randomly selected and saved as `learn.shp` (for model training).
- **30%** of the points are saved as `control.shp` (for independent model validation and confusion matrix generation).

---

## Interactive Menu & Stages Selector (ANN / SAM)

When you execute the classifier script (`1_OBIA_vector_classifier_modular_ANN.py` or the SAM version), the program starts an interactive text menu in your terminal. This gives you absolute control over execution, allowing you to run everything in one go or step-by-step while adjusting hyperparameters.

### The Pipeline Menu Layout
Upon launching the script, the following menu is displayed:
```text
    --- Raster-Based OBIA Pipeline (ANN) ---
    Track: PL/orbit_12 (PL)

    [1] Stage 1: SAR Summed Segmentation (eCognition-style MRS)
    [2] Stage 2: Split Samples
    [3] Stage 3: Extract Features (Object-based Training)
    [4] Stage 4: Train ANN Classifier
    [5] Stage 5: Tiled Object-Based Inference
    [6] Stage 6: Mask Classification
    [7] Stage 7: Mask Confidence
    [8] Stage 8: Calculate Metrics

    [A] Run All Stages (Forces overwrite of Stages 5-8 to clear old bugs)
    [Q] Quit

    Enter your choice:
```

### Detailed Execution Options

#### Option `A`: Run All Stages (All-in-One Execution)
- Recommended for standard runs. 
- Automatically executes all 8 stages sequentially.
- Forces overwrite of the inference outputs (`Stage 5` to `8`) to ensure no corrupted files or caching issues are present.

#### Single-Stage Execution & Parameter Tuning
You can select individual numbers to execute specific parts of the pipeline and dynamically adjust their parameters:

- **Choice `1` (Stage 1: Segmentation)**:
  - Splitting the raster into homogenous segments.
  - You can change hyperparameters interactively when prompted (e.g. `scale`, `sigma`, `min_size` for Felzenszwalb, or Multi-Resolution Segmentation parameters).
  - Prompts:
    ```text
    Change parameters? (y/n) [n]: y
    Enter new value for 'scale' [50.0]: 40.0
    Enter new value for 'min_size' [15]: 20
    ```

- **Choice `2` (Stage 2: Split Samples)**:
  - Randomly partitions `samples.shp` into training and validation sets.
  - Prompts:
    ```text
    Change parameters? (y/n) [n]: y
    Enter new value for 'learn_frac' [0.7]: 0.8
    ```

- **Choice `3` (Stage 3: Extract Features)**:
  - Performs zonal statistics on the segments corresponding to training locations, storing outputs in a CSV file. No hyperparameters needed.

- **Choice `4` (Stage 4: Train ANN)**:
  - Trains the neural network on the extracted features.
  - Allows you to change classifier settings (e.g. MLP hidden layers architecture, max training iterations, or class balancing threshold).
  - Prompts:
    ```text
    Change parameters? (y/n) [n]: y
    Enter classifier (ann_sklearn) [ann_sklearn]: ann_sklearn
    Enter new value for 'sk_hidden_sizes' [100,50]: 120,60,30
    Enter new value for 'sk_max_iter' [500]: 800
    Enter new value for 'balance_threshold' [1000]: 1500
    ```

- **Choice `5` (Stage 5: Object-Based Inference)**:
  - Runs classification on every segment across the full raster tiles. This is performed block-by-block to prevent memory exhaustion (OOM).

- **Choice `6` & `7` (Stage 6 & 7: Cropland Masking)**:
  - Applies the binary agricultural mask (generated in Step 2) and data footprint to the final classification and confidence GeoTIFFs.

- **Choice `8` (Stage 8: Calculate Metrics)**:
  - Computes global Overall Accuracy, Kappa coefficient, per-class recall, precision, F1-score, and crop areas (in hectares). Generates the final Excel report.

---

## Segment Anything (SAM) Model Setup & Parameters

If you choose to run `1_OBIA_vector_classifier_modular_ANN_SAM.py`, the segmentation stage uses Meta AI's Segment Anything Model (SAM) instead of traditional algorithms. This requires downloading a model checkpoint.

### 1. Download SAM Checkpoint File
1. Download the high-quality **ViT-H SAM model checkpoint** (`sam_vit_h_4b8939.pth`) from the official Facebook Research repository:
   [sam_vit_h_4b8939.pth (Download Link)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
2. Create the directory `auxiliary_files/SAM_models/` (if it does not exist).
3. Save the downloaded `.pth` file directly as:
   `auxiliary_files/SAM_models/sam_vit_h_4b8939.pth`

### 2. Tuning SAM Segmentation Parameters
When running the SAM-based classifier script, select option `1` (Stage 1) to configure the SAM parameters:

- **`sam_checkpoint`** (String): Path to your downloaded `.pth` checkpoint. Defaults to `auxiliary_files/SAM_models/sam_vit_h_4b8939.pth`.
- **`sam_model_type`** (String): Type of model corresponding to the checkpoint (e.g. `vit_h` for ViT-H, `vit_l` for ViT-L, `vit_b` for ViT-B).
- **`sam_device`** (String): Hardware device to run the deep learning calculations. 
  - Set to `cuda` if you have an NVIDIA GPU with PyTorch CUDA installed (strongly recommended for speed).
  - Set to `cpu` to run on the processor (slower, but acts as a reliable fallback).
- **`tile_size`** (Integer): The size of the tile grid in pixels (default `2048`) parsed to SAM at once. If your GPU runs out of VRAM (CUDA Out Of Memory errors), decrease this to `1024` or `512`.
- **`buffer`** (Integer): Overlapping pixel boundary (default `128`) to prevent edge artifacts between tiles.

---

## Step-by-Step Execution Guide

### Step 1: Download NUTS2 Boundaries
Automated script to build country shapefiles.
- **What it does**: Connects to Eurostat, downloads European NUTS administrative boundary shapefiles, filters them to NUTS2 level, and saves them per country.
- **Command**:
  ```bash
  python download_nuts_shapefiles.py
  ```
- **Where files go**: `auxiliary_files/shapefiles_nuts/{COUNTRY}/NUTS2_{COUNTRY}.shp`

---

### Step 2: Prepare Copernicus HRL Crop Mask
Masks out non-agricultural areas (forests, cities, water) to focus classification only on cropland.

1. **Manual Download**:
   - Go to [Copernicus CLMS Portal](https://land.copernicus.eu/).
   - Download the **High Resolution Layer: Crop Type 2023** zip files covering your target country.
   - Save the ZIP files into: `auxiliary_files/raster_files/AgriMasks/<COUNTRY>/Results/`
     *(e.g., `auxiliary_files/raster_files/AgriMasks/PL/Results/` for Poland)*.

2. **Generate Binary Mask**:
   Run the processing script to mosaic, reproject, and clip the mask to the country:
   ```bash
   python 2_OBIA_classifier/build_agri_mask.py --country PL
   ```

---

### Step 3: SAR Slice Calibration & Assembly
Scans your local database of raw Sentinel-1 SAFE files, resolves which orbits and flight direction cover the country, and starts SNAP calibration.
- **Command**:
  ```bash
  python 1_Sentinel-1_preprocessor/1_AIML_S1_slice_calibration.py -s 2024-10-15 -e 2024-11-30 -c PL
  ```
  *(Change `PL` to your country code and set your target date range).*
- **COG Alternative**: For Cloud-Optimized GeoTIFF outputs, run `1_AIML_S1_slice_calibration_COG.py` instead.
- **Where files go**: Sliced BEAM-DIMAP outputs go to `workingDir/<COUNTRY>/orbit_<ORBIT>/slice_assembly/`.

---

### Step 4: Stack Coregistration
Aligns the time-series stack of images for each orbit so they match pixel-for-pixel.
- **Command**:
  ```bash
  python 1_Sentinel-1_preprocessor/2_AIML_S1_coregistration.py -t PL/orbit_12 PL/orbit_88
  ```
  *(Pass all orbit directories created in Step 3).*

---

### Step 5: Stack Clipping
Converts the aligned SNAP Dimap stacks into standard GeoTIFF format and crops them to the exact NUTS2 boundary of the country.
- **Command**:
  ```bash
  python 1_Sentinel-1_preprocessor/3_AIML_S1_stack_clip.py -t PL/orbit_12 PL/orbit_88
  ```

---

### Step 6: Object-Based Classification
Splits the image stack into objects/fields, extracts radar metrics (mean, std dev, ratios) for each object over the time-series dates, and performs classification.
- **Option A (ANN with Felzenszwalb segmentation - Faster/Recommended)**:
  ```bash
  python 2_OBIA_classifier/1_OBIA_vector_classifier_modular_ANN.py --track PL/orbit_12
  ```
- **Option B (ANN with SAM deep-learning segmentation - Resource Intensive)**:
  ```bash
  python 2_OBIA_classifier/1_OBIA_vector_classifier_modular_ANN_SAM.py --track PL/orbit_12
  ```
  *(Execute this sequentially for each orbit directory).*

---

### Step 7: Merge Country Classification
Combines the classification results from all individual orbits into a single country-wide map.
- **Command**:
  ```bash
  python 2_OBIA_classifier/2_OBIA_merge_classifications.py --track PL
  ```
- **Output**: The finalized classification raster is saved as:
  `workingDir/PL/classification_results/PL_final_classification.tif`

---

## Troubleshooting & Performance Tuning

### SNAP Memory and Cache Issues (OOM)
SNAP can consume huge amounts of RAM. If you face Java Heap Space/OOM crashes:
1. Locate SNAP's config: `gpt.vmoptions` (located in the SNAP `bin/` directory).
2. Edit the VM options (e.g. increase `-Xmx` memory threshold):
   ```text
   -Xmx16G
   -Dsnap.userdir=...
   ```
3. The scripts pass `-q 4` to GPT to limit it to 4 parallel threads. You can decrease this to `-q 2` in the scripts (`run_calibration_stage`, etc.) if memory usage is still too high.

### Orbit Files Download Failures
During step 3/4, SNAP will attempt to download Precise Orbit files. If this fails due to ESA server downtimes:
- Ensure you have an internet connection.
- In SNAP's `Apply-Orbit-File` XML node, the code sets `continueOnFail` to `true`, which allows processing to continue with lower precision orbit files if the precise files are unavailable.
