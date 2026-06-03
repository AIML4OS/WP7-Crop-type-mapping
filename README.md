# Sentinel-1 OBIA Crop Type Mapping Pipeline (v2.0)

An automated, object-based image analysis (OBIA) pipeline designed to process Sentinel-1 SAR time series and perform crop type classification. It integrates state-of-the-art **Geospatial Foundation Models** and classical **machine learning classifiers** to handle large-scale country-wide cropland mapping.

### Key Technologies & Models:
1. **IBM-NASA Prithvi-EO-1.0-100M Foundation Model**: Used to extract deep, robust temporal-spectral embeddings from Sentinel-1 SAR stacks to represent agricultural fields.
2. **Meta AI Segment Anything Model (SAM)**: Used for precise deep-learning-based field boundary delineation (segmentation) under challenging backscatter conditions.
3. **Orfeo ToolBox (OTB) Classifier**: Integrates OTB's high-performance machine learning suite (Random Forest, Support Vector Machines - SVM) for lightning-fast training and pixel/object classification on large spatial grids.
4. **Multi-Layer Perceptron (MLP/ANN)**: A classical artificial neural network (scikit-learn and custom architectures) optimized for object-based class prediction.

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
7. [NASA-IBM Prithvi-SAR Model Setup & Parameters](#nasa-ibm-prithvi-sar-model-setup--parameters)
8. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
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
   - Open **Miniforge Prompt** and create your conda environment. 
     > [!IMPORTANT]
     > The pipeline requires standard Python libraries for processing geospatial data, machine learning, and report generation:
     > - **Geospatial Processing**: `gdal`, `geopandas`, `rasterio`, `numpy` (for grid manipulation), and `pyogrio` (for accelerated vector database reading).
     > - **Machine Learning**: `scikit-learn` (for MLP ANN classifier).
     > - **Report & Data Handling**: `pandas`, `openpyxl` (for writing Excel metric reports), and `joblib` (for model saving/loading).
     > - **Image Processing**: `scikit-image` (required for Felzenszwalb, SLIC, and Multi-Resolution segmentation).
     ```bash
     conda create -n your_env python=3.10 gdal geopandas rasterio numpy pandas scikit-learn scikit-image openpyxl joblib pyogrio -y
     conda activate your_env
     ```
   - *Optional (for SAM GPU acceleration & Prithvi-SAR foundation model)*: Install PyTorch with CUDA, Segment Anything (SAM), and HuggingFace dependencies:
     ```bash
     conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
     pip install segment-anything segment-geospatial huggingface_hub transformers timm einops
     ```

2. **Install ESA SNAP**:
   - Download the SNAP installer from [ESA SNAP Download](https://step.esa.int/main/download/snap-download/).
   - Run the installer. You can install to the default path on drive `C:\` or choose another drive (e.g., `D:\Program Files\esa-snap`).
   - Locate the path to the executable `gpt.exe` (e.g. `D:\Program Files\esa-snap\bin\gpt.exe`). You will need to supply this exact path in the configuration.

3. **Install Orfeo ToolBox (OTB) (Optional)**:
   - Download OTB 6.2.0 Win64 and extract it to a local folder (e.g., `D:\AIML_CropMapper_Cloud\bin\OTB-6.2.0-Win64`).

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
   - Create the Conda environment with standard geospatial and data processing dependencies (`gdal`, `geopandas`, `rasterio`, `numpy`, `pandas`, `scikit-learn`, `scikit-image`, `openpyxl`, `joblib`, `pyogrio`):
     ```bash
     conda create -n your_env python=3.10 gdal geopandas rasterio numpy pandas scikit-learn scikit-image openpyxl joblib pyogrio -y
     conda activate your_env
     ```
   - *Optional (for SAM & Prithvi-SAR)*: Install deep learning frameworks and foundation model libraries:
     ```bash
     conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
     pip install segment-anything segment-geospatial huggingface_hub transformers timm einops
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
The classifier searches for your reference shapefile in the `shapefiles_samples` directory using the following hierarchy (matching the Python code search paths):
1. `auxiliary_files/shapefiles_samples/{FILE_PREFIX}/samples.shp` (e.g. `PL_orbit_12/samples.shp` or `AT_P1a/samples.shp`)
2. `auxiliary_files/shapefiles_samples/{SAN_TRACK}/samples.shp` (e.g. `PL_orbit_12/samples.shp` or `P1a/samples.shp`)
3. `auxiliary_files/shapefiles_samples/{TRACK}/samples.shp` (e.g. `PL/orbit_12/samples.shp` or `P1a/samples.shp`)
4. `auxiliary_files/shapefiles_samples/{COUNTRY}/samples.shp` (e.g. `PL/samples.shp` or `AT/samples.shp` - country fallback)
5. `auxiliary_files/shapefiles_samples/samples.shp` (global fallback)

### 4. Automatic Sample Splitting
In Stage 2, the pipeline automatically splits your `samples.shp` dataset:
- **70%** of the points are randomly selected and saved as `learn.shp` (for model training).
- **30%** of the points are saved as `control.shp` (for independent model validation and confusion matrix generation).

---

## Interactive Menu & Stages Selector (ANN / SAM)

When you execute the classifier script (`1_classify_ann.py` or the SAM version), the program starts an interactive text menu in your terminal. This gives you absolute control over execution, allowing you to run everything in one go or step-by-step while adjusting hyperparameters.

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

If you choose to run `1_classify_ann_sam.py`, the segmentation stage uses Meta AI's Segment Anything Model (SAM) instead of traditional algorithms. This requires downloading a model checkpoint.

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

## NASA-IBM Prithvi-SAR Model Setup & Parameters

The Prithvi-SAR crop classifier (`1_classify_prithvi_sar.py`) utilizes the pre-trained NASA-IBM Prithvi-EO-1.0-100M geospatial foundation model for extracting deep temporal-spectral representations from multi-date Sentinel-1 SAR stacks.

### 1. Auto-Download Architecture
The script is designed to **automatically download** all required model components from the official HuggingFace repository (`ibm-nasa-geospatial/Prithvi-EO-1.0-100M`) on the first run:
- **Architecture definition**: `prithvi_mae.py` (downloaded and saved to `auxiliary_files/Prithvi_models/`)
- **Model weights**: `Prithvi_100M.pt` (downloaded and saved to `auxiliary_files/Prithvi_models/`)

If your processing machine lacks internet access, you can manually download these two files and place them inside the `auxiliary_files/Prithvi_models/` directory prior to running the script.

### 2. Temporal Stacking Logic
Prithvi expects 6 spectral bands across 3 temporal frames (total size `[6, 3, 224, 224]`). The script automatically:
1. Groups Sentinel-1 VV/VH bands into three seasonal frames (early, mid, and late season).
2. Duplicates the 2 polarization bands to construct the 6-band input expected by the model.
3. Resizes segment crops bilinearly to `224x224` pixels.
4. Feeds them to the Prithvi ViT encoder to extract a `768`-dimensional token embedding vector per segment.

---

### Step-by-Step Execution Guide

This guide details each script in the pipeline, explaining its functionality, requirements, execution commands, and output products.

---

### Step 1: NUTS2 Boundary Database Builder (`download_nuts_shapefiles.py`)
* **Description & Logic**: Automatically downloads the official GISCO Eurostat administrative boundary dataset (1:1 Million high-resolution shapefiles) and extracts boundary polygons for 37 European countries. It builds a local GIS database of national boundaries at the NUTS2 level, which is used for spatial subsetting of the satellite data. It also duplicates boundaries for Greece under both `EL` and `GR` codes to handle standardized data querying.
* **Prerequisites & Config**: Requires `geopandas` and `pyogrio` python packages. No environment variables are needed.
* **Launch Command**:
  ```bash
  python download_nuts_shapefiles.py
  ```
* **Produced Outputs**:
  - National boundary shapefiles saved to: `auxiliary_files/shapefiles_nuts/{COUNTRY}/NUTS2_{COUNTRY}.shp`

---

### Step 2: Crop Mask Preparation (`tools/build_agri_mask.py`)
* **Description & Logic**: Mosaics and clips the Copernicus High Resolution Layer (HRL) Crop Type raster files to the exact boundary of the target country. It aligns the final raster to the Web Mercator projection (`EPSG:3857`) at a 10-meter resolution grid. Non-agricultural pixels are masked out to focus object segmentation and neural network predictions strictly on active croplands.
* **Prerequisites & Config**: The HRL ZIP files must be manually downloaded from the Copernicus CLMS portal and placed in `auxiliary_files/raster_files/AgriMasks/{COUNTRY}/Results/` (e.g. `PL/Results/`).
* **Launch Command**:
  ```bash
  python tools/build_agri_mask.py --country PL
  ```
* **Produced Outputs**:
  - Arable Crops Mask (Cereals, Rape, Maize, Vegetables): `auxiliary_files/raster_files/AgriMasks/{COUNTRY}/{COUNTRY}_agri_mask_3class_epsg3857.tif`
  - All Crops Mask (Arable + Permanent Crops and Grasslands): `auxiliary_files/raster_files/AgriMasks/{COUNTRY}/{COUNTRY}_agri_mask_allcrops_epsg3857.tif`

---

### Step 3: Sentinel-1 Slice Calibration & Assembly (`1_Sentinel-1_preprocessor/1a_slice_calibration.py` / `1b_slice_calibration_cog.py` [PREFERRED])
* **Description & Logic**: Scans the local Sentinel-1 IW GRD repository and reads spatial geometries of available `.SAFE` directories. It solves a Set Cover mathematical optimization problem (`CountryOrbitOptimizer`) to find the minimal set of relative orbits required to fully cover the country's geometry. For each orbit, it runs SNAP's Graph Processing Tool (`gpt.exe`) to perform calibration.
* **Calibration Alternatives**:
  - **Standard Calibration** (`1a_slice_calibration.py`): Performs radiometric calibration, orbit application, thermal noise removal, terrain correction, and slice assembly, saving intermediate and final outputs as raw SNAP `.dim` / `.data` BEAM-DIMAP pairs.
  - **Cloud-Optimized GeoTIFF (COG) Calibration** (`1b_slice_calibration_cog.py` [**RECOMMENDED/PREFERRED**]): Executes the same calibration steps but writes outputs directly as Cloud-Optimized GeoTIFFs (`.tif`) utilizing DEFLATE compression. 
    > [!TIP]
    > **Why the COG version is preferred:**
    > 1. **Massive Disk Space Savings**: Uses up to 5x less disk space compared to uncompressed raw BEAM-DIMAP files.
    > 2. **Cloud/Network Optimization**: Supports HTTP range requests, making it highly optimized for remote storage, virtual file systems, and cloud deployments.
    > 3. **Reduced I/O Bottlenecks**: Faster reading/writing during downstream coregistration and clipping steps.
* **Prerequisites & Config**: Requires SNAP GPT executable path and environment variables (`SNAP_GPT_EXE`, `S1_REPO_PATH`, `AIML_WORKING_DIR`, `AIML_AUX_DIR`) set in the active terminal.
* **Launch Command**:
  ```bash
  # Standard Calibration (BEAM-DIMAP outputs):
  python 1_Sentinel-1_preprocessor/1a_slice_calibration.py -s 2024-10-15 -e 2024-11-30 -c PL

  # Cloud-Optimized GeoTIFF (COG) Calibration (Preferred):
  python 1_Sentinel-1_preprocessor/1b_slice_calibration_cog.py -s 2024-10-15 -e 2024-11-30 -c PL
  ```
* **Produced Outputs**:
  - Calibrated daily SAR scenes saved in: `workingDir/{COUNTRY}/orbit_{ORBIT}/slice_assembly/` as `.dim` / `.data` pairs (or `.tif` for COG).

---

### Step 4: Multi-Temporal Stack Coregistration (`1_Sentinel-1_preprocessor/2_coregistration.py`)
* **Description & Logic**: Aligns the multi-temporal time-series of assembled Sentinel-1 scenes for each orbit. It dynamically parses the band ordering (VH/VV) from the `.dim` XML files, sorts the dates chronologically, and registers all dates to a common master scene. It then applies a multi-temporal Lee Sigma speckle filter to suppress radar noise while preserving field boundaries.
* **Prerequisites & Config**: Requires calibrated outputs from Step 3.
* **Launch Command**:
  ```bash
  python 1_Sentinel-1_preprocessor/2_coregistration.py -t PL/orbit_12 PL/orbit_88
  ```
* **Produced Outputs**:
  - Aligned time-series stack saved to: `workingDir/{COUNTRY}/orbit_{ORBIT}/coregistration/` (as `.dim` and `.data` folders).

---

### Step 5: Stack Spatial Clipping (`1_Sentinel-1_preprocessor/3_stack_clip.py`)
* **Description & Logic**: Converts the coregistered time-series SNAP stacks into standard multiband GeoTIFF format and clips them to the exact NUTS2 country boundary shapefile. It executes warping and DEFLATE compression in parallel across all CPU cores (`NUM_THREADS=ALL_CPUS`) and automatically builds overview pyramids (`BuildOverviews`) for instant visual rendering in QGIS.
* **Prerequisites & Config**: Requires NUTS2 shapefiles (from Step 1) and coregistered stacks (from Step 4).
* **Launch Command**:
  ```bash
  python 1_Sentinel-1_preprocessor/3_stack_clip.py -t PL/orbit_12 PL/orbit_88
  ```
* **Produced Outputs**:
  - Clipped Multiband GeoTIFF: `workingDir/{COUNTRY}/orbit_{ORBIT}/processed_raster/{COUNTRY}_orbit_{ORBIT}_VH_VV.tif` (along with a `.vrt` header file).

---

### Step 6: Object-Based Classification (`2_classifier/` scripts)
* **Description & Logic**: Splits the clipped image stack into homogeneous agricultural parcel objects, extracts statistical or deep learning features for each object over the Sentinel-1 timeline, trains a machine learning or deep learning classifier, and performs tiled prediction across the entire track.
* **Algorithm Options**:
  - **Option A (Felzenszwalb ANN)** - `1_classify_ann.py`: Performs Felzenszwalb segmentation on CPU. Extracts zonal statistics (mean backscatter, standard deviation, and temporal ratios) per parcel object to train a scikit-learn MLP Classifier.
  - **Option B (SAM ANN)** - `1_classify_ann_sam.py`: Employs Meta AI's Segment Anything Model (SAM) for deep learning-based boundary delineation (requires GPU / PyTorch).
  - **Option C (Prithvi-SAR)** - `1_classify_prithvi_sar.py`: Leverages the NASA-IBM geospatial foundation model to extract `768`-dimensional temporal-spectral token embeddings from segmented image patches. It includes its own built-in Segment Anything Model (SAM) segmentation stage, making the entire pipeline completely self-contained.
  - **Option D (OTB RF/SVM)** - `1_classify_otb.py`: Integrates Orfeo ToolBox (OTB) CLI tools. Performs OTB Mean-Shift segmentation on the time-series stack, extracts statistical object features, trains an OTB Random Forest or SVM classifier, and outputs classified shapefiles and rasters.
* **Prerequisites & Config**: Requires the clipped raster (from Step 5), training sample points at `auxiliary_files/shapefiles_samples/{COUNTRY}/samples.shp`, model checkpoints if running SAM/Prithvi, and OTB binary installation in `bin/OTB-6.2.0-Win64/` if running OTB classification.
* **Launch Command**:
  ```bash
  # Run Felzenszwalb ANN Classifier:
  python 2_classifier/1_classify_ann.py --track PL/orbit_12

  # Run SAM Deep-Learning Classifier:
  python 2_classifier/1_classify_ann_sam.py --track PL/orbit_12

  # Run NASA-IBM Prithvi-SAR Classifier:
  python 2_classifier/1_classify_prithvi_sar.py --track PL/orbit_12

  # Run OTB Random Forest/SVM Classifier:
  python 2_classifier/1_classify_otb.py --track PL/orbit_12
  ```
* **Produced Outputs**:
  - Segmentation Map: `workingDir/{track}/classification_results/segmentation/{file_prefix}_segmentation.tif`
  - Training/Validation Split Points: `.../samples/learn.shp` & `.../samples/control.shp`
  - Extracted Features: `.../samples/{file_prefix}_[prithvi_]learn_features.csv`
  - Trained Classifier: `.../train_model/{file_prefix}_[prithvi_]model.pkl`
  - Raw Outputs: `.../classification/{file_prefix}_[prithvi_]classified.tif` & `..._confidence_map.tif`
  - Masked Outputs: `.../classification/{file_prefix}_[prithvi_]classified_masked.tif` & `..._confidence_masked.tif`
  - Classification Accuracy Report: `.../classification/{file_prefix}_[prithvi_]metrics.xlsx`

---

### Step 7: Classification Merge (`2_classifier/2_merge_classifications.py`)
* **Description & Logic**: Mosaics and merges the classification results from all individual orbits into a single country-wide map. For overlapping zones between different orbits, the script compares confidence scores at the pixel level and selects the prediction with the **highest confidence score**. Finally, it applies a morphological **sieve filter** to dissolve small isolated pixels (slivers) and validates the merged dataset against validation points (`control.shp`).
* **Prerequisites & Config**: Requires masked rasters and `control.shp` from Step 6.
* **Launch Command**:
  ```bash
  # Merge standard ANN classifications (Felzenszwalb / SAM):
  python 2_classifier/2_merge_classifications.py --track PL

  # Merge Prithvi-SAR classifications:
  python 2_classifier/2_merge_classifications.py --track PL --suffix _prithvi
  ```
* **Produced Outputs**:
  - Merged Classification Raster: `workingDir/{COUNTRY}/classification_results/{COUNTRY}_final_classification[_prithvi].tif`
  - Country Accuracy Report: `workingDir/{COUNTRY}/classification_results/{COUNTRY}_final_metrics[_prithvi].xlsx`

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
