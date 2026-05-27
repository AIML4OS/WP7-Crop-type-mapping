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
4. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Step 1: Download NUTS2 Boundaries](#step-1-download-nuts2-boundaries)
   - [Step 2: Prepare Copernicus HRL Crop Mask](#step-2-prepare-copernicus-hrl-crop-mask)
   - [Step 3: SAR Slice Calibration & Assembly](#step-3-sar-slice-calibration--assembly)
   - [Step 4: Stack Coregistration](#step-4-stack-coregistration)
   - [Step 5: Stack Clipping](#step-5-stack-clipping)
   - [Step 6: Object-Based Classification](#step-6-object-based-classification)
   - [Step 7: Merge Country Classification](#step-7-merge-country-classification)
5. [Troubleshooting & Performance Tuning](#troubleshooting--performance-tuning)

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
   - Install to the default path (e.g. `C:\Program Files\esa-snap`).
   - Find `gpt.exe` inside `C:\Program Files\esa-snap\bin\gpt.exe`.

3. **Install Orfeo ToolBox (OTB) (Optional)**:
   - Download OTB 6.2.0 Win64 and extract it to `D:\AIML_CropMapper_Cloud\2_OBIA_classifier\OTB-6.2.0-Win64`.

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
   - The default installation path is usually `/usr/local/esa-snap` or `$HOME/esa-snap`. Locate the `gpt` tool in the `bin/` directory.

---

## Environment Configuration

Before running any script, you must tell the tools where SNAP and your project directories are located. 

### On Windows (PowerShell):
Run these commands in your console before executing the scripts:
```powershell
$env:SNAP_GPT_EXE="D:/Program Files/esa-snap/bin/gpt.exe"
$env:SNAP_AUXDATA_PATH="C:/Users/Administrator/.snap/auxdata"
$env:S1_REPO_PATH="Y:/Sentinel-1/SAR/IW_GRDH_1S"
$env:AIML_WORKING_DIR="D:/AIML_CropMapper_Cloud/workingDir"
$env:AIML_AUX_DIR="D:/AIML_CropMapper_Cloud/auxiliary_files"
```

### On Linux (Bash):
Run these commands in your terminal:
```bash
export SNAP_GPT_EXE="/usr/local/esa-snap/bin/gpt"
export SNAP_AUXDATA_PATH="$HOME/.snap/auxdata"
export S1_REPO_PATH="/mnt/sentinel1/SAR/IW_GRDH_1S"
export AIML_WORKING_DIR="/home/user/AIML_CropMapper_Cloud/workingDir"
export AIML_AUX_DIR="/home/user/AIML_CropMapper_Cloud/auxiliary_files"
```

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
1. Locate SNAP's config: `C:\Program Files\esa-snap\bin\gpt.vmoptions` (Windows) or `/usr/local/esa-snap/bin/gpt.vmoptions` (Linux).
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
