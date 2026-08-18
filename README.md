# AIML CropMapper Cloud: Sentinel-1 & Sentinel-2 OBIA Crop Type Mapping Pipeline (v2.5)

An automated, object-based image analysis (OBIA) pipeline designed to process **Sentinel-1 SAR** and **Sentinel-2 Multispectral Optical** time series for large-scale, country-wide crop type classification. It integrates state-of-the-art **Geospatial Foundation Models (NASA Harvest Presto)**, **Deep PyTorch Neural Networks (MLP)**, and **Extreme Gradient Boosting (XGBoost)** to achieve high-accuracy agricultural mapping across Europe.

---

### Key Technologies and Models:

1. **Multimodal S1 + S2 Hybrid Classifier (`1_classify_MLPXGB_presto_hybrid_S1S2.py`)**:
   - Fuses **Sentinel-1 SAR** backscatter (polarizations VH, VV, and ratio VH/VV) with **Sentinel-2 Optical** surface reflectances (9 spectral bands: `B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B8A`, `B11`, `B12`, plus NDVI).
   - Generates **128-dimensional temporal token embeddings** using the **NASA Harvest Presto Foundation Model** for both SAR and optical modalities.
   - Implements a unified **PyTorch Deep MLP + XGBoost GBDT Soft-Voting Ensemble** (`EnsembleClassifier`) with class-frequency weighting, dropout, batch normalization, and cosine annealing learning rate schedules.

2. **Presto-SAR Hybrid Classifier (`1_classify_ann_presto_hybrid.py`)**:
   - For all-weather radar-only mapping when Sentinel-2 optical data is obscured by persistent cloud cover. Combines physical SAR statistical features with deep Presto embeddings.

3. **Orfeo ToolBox (OTB) Classifier (`1_classify_otb.py`)**:
   - High-performance machine learning suite (Random Forest, Support Vector Machines - SVM) for lightning-fast baseline benchmarking.

4. **Multi-Orbit Sieve & Bayesian Merge (`2_merge_classifications.py`)**:
   - Country-wide multi-orbit mosaicking using pixel-level confidence comparison, Bayesian prior probability calibration, and morphological sieve filtering to dissolve isolated sliver pixels.

5. **Sentinel-2 Preprocessor Suite (`1a_Sentinel-2_preprocessor/`)**:
   - Dual-source ingestion: Local CREODIAS mounted repository (`Y:\Sentinel-2\MSI\L2A`) or remote Copernicus Data Space Ecosystem (CDSE) API with automated OData pagination.
   - Pure Python multi-temporal DOY linear interpolation across 14 agricultural phenology dates.
   - Seamless S1 spatial grid matching (matching exact CRS, bounding box, and pixel resolution) and NUTS2 administrative boundary clipping.

6. **Archived Experimental Scripts (`2_classifier/Archive_scripts/`)**:
   - Earlier single-modality experiments (`1_classify_prithvi_sar.py`, `1_classify_ann.py`, `1_classify_presto_sar.py`) have been moved to `2_classifier/Archive_scripts/` to preserve historical research reproducibility while keeping the main workflow streamlined and focused on top-performing architectures.

---

## Table of Contents
1. [Prerequisites and System Installation](#prerequisites-and-system-installation)
   - [Windows Setup](#windows-setup)
   - [Linux Setup](#linux-setup)
2. [JSON Configuration System (Safe Secret Management)](#json-configuration-system-safe-secret-management)
3. [Training Samples Specification (`samples.shp`)](#training-samples-specification-samplesshp)
4. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
   - [Step 1: Download NUTS2 Boundaries](#step-1-download-nuts2-boundaries)
   - [Step 2: Build Agricultural Cropland Mask](#step-2-build-agricultural-cropland-mask)
   - [Step 3: Sentinel-1 SAR Preprocessing (Calibration, Coregistration, Clip)](#step-3-sentinel-1-sar-preprocessing)
   - [Step 4: Sentinel-2 Optical Preprocessing (Download, DOY Interpolation, Mosaic & Clip)](#step-4-sentinel-2-optical-preprocessing)
   - [Step 5: Multimodal S1 + S2 Object-Based Classification](#step-5-multimodal-s1--s2-object-based-classification)
   - [Step 6: Country-Wide Classification Merge](#step-6-country-wide-classification-merge)
5. [Model Architecture & Segmentation Options](#model-architecture--segmentation-options)
6. [Troubleshooting & Performance Tuning](#troubleshooting--performance-tuning)
7. [How to Cite](#how-to-cite)

---

## Prerequisites and System Installation

### Windows Setup

1. **Install Python via Miniforge**:
   - Download and install [Miniforge3](https://github.com/conda-forge/miniforge) for Windows.
   - Open **Miniforge Prompt** and create your environment:
     ```bash
     mamba env create -f environment.yml
     mamba activate aiml_env
     ```
   - Make sure GDAL's native OpenJPEG plugin is installed:
     ```bash
     conda install -y -c conda-forge libgdal-jp2openjpeg
     ```

2. **Install ESA SNAP**:
   - Download SNAP from [ESA SNAP Download](https://step.esa.int/main/download/snap-download/).
   - Install to e.g. `D:\Program Files\esa-snap`.

3. **Install Orfeo ToolBox (OTB) (Optional)**:
   - Download OTB 6.2.0 Win64 and extract to `D:\AIML_CropMapper_Cloud\bin\OTB-6.2.0-Win64`.

---

### Linux Setup

1. **Install System Dependencies (Ubuntu/Debian)**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y gdal-bin libgdal-dev build-essential unzip
   ```

2. **Install Conda Environment**:
   ```bash
   mamba env create -f environment.yml
   mamba activate aiml_env
   ```

3. **Install ESA SNAP**:
   - Download SNAP Linux installer `.sh` and run:
     ```bash
     chmod +x esa-snap_sentinel_unix_*.sh
     ./esa-snap_sentinel_unix_*.sh -q
     ```

---

## JSON Configuration System (Safe Secret Management)

The toolbox features a centralized, modular configuration system using **JSON** files. You do not need to pass passwords or paths on the command line.

### 1. Configuration Files

| Config File | Purpose | Location |
| :--- | :--- | :--- |
| **`config_cdse.json`** | Copernicus CDSE credentials (username & password) | Root directory (`d:/AIML_CropMapper_Cloud/config_cdse.json`) |
| **`config.json`** | Global project paths, threads, bands, and classifier hyperparameters | Root directory (`d:/AIML_CropMapper_Cloud/config.json`) |
| **`config_s1.json`** | Sentinel-1 pipeline parameters and SNAP paths | `1_Sentinel-1_preprocessor/config_s1.json` |
| **`config_s2.json`** | Sentinel-2 pipeline parameters and DOY vectors | `1a_Sentinel-2_preprocessor/config_s2.json` |

### 2. Quick CDSE Setup
Open [`config_cdse.json`](file:///d:/AIML_CropMapper_Cloud/config_cdse.json) and enter your credentials:
```json
{
  "username": "your_email@domain.com",
  "password": "your_cdse_password"
}
```

### 3. GitHub Security (.gitignore)
All `*.json` files containing actual passwords or local paths are **strictly ignored by Git** via `.gitignore`. Only safe templates (`*.example.json`) with placeholder values are committed to GitHub:
```bash
# Push changes safely to GitHub (passwords are automatically excluded):
git add .
git commit -m "Update CropMapper toolbox"
git push origin AIML_CropMapper
```

---

## Training Samples Specification (`samples.shp`)

To train the machine learning classifier, place your point vector dataset at:
`auxiliary_files/shapefiles_samples/{COUNTRY}/samples.shp` (e.g. `PL/samples.shp` or `NL/samples.shp`).

- **Geometry Type**: Point (`OGRPoint`). Points should lie inside field boundaries.
- **Required Attribute**: **`crop_id`** (Integer): Numeric code for the crop class (e.g., `11` = Winter Wheat, `12` = Maize, `1430` = Rapeseed). Value `0` is reserved for background.
- **Sample Splitting**: The pipeline automatically partitions points into **70% training (`learn.shp`)** and **30% validation (`control.shp`)**.
- **Bayesian Prior Probabilities (`priors.json`)**: To apply prior acreage correction, place `priors.json` in `auxiliary_files/shapefiles_samples/{COUNTRY}/priors.json` with crop class proportions.

---

## Data Preparation Utilities (`tools/`)

The `tools/` directory provides unified, standardized CLI utilities for data preparation:

1. **Step 1: Download NUTS Boundaries (`tools/1_download_nuts_boundaries.py`)**:
   Downloads official GISCO NUTS2 / NUTS0 boundaries for raster spatial clipping:
   ```powershell
   python tools/1_download_nuts_boundaries.py -c NL
   python tools/1_download_nuts_boundaries.py -c PL
   python tools/1_download_nuts_boundaries.py --all
   ```

2. **Step 2: Build Agricultural Cropland Mask (`tools/2_build_agricultural_mask.py`)**:
   Creates a binary cropland mask from either **official LPIS cadastral vectors** (highest precision) or **Copernicus HRL/CLMS raster tiles**:
   ```powershell
   # Option A: From LPIS Cadastral Parcel Vectors (Recommended):
   python tools/2_build_agricultural_mask.py -c NL --lpis path/to/brp.gpkg
   python tools/2_build_agricultural_mask.py -c PL --lpis path/to/arimr.shp
   python tools/2_build_agricultural_mask.py -c PT --lpis path/to/isip.shp

   # Option B: From Copernicus HRL / CLMS Raster Tiles:
   python tools/2_build_agricultural_mask.py -c NL
   python tools/2_build_agricultural_mask.py -c PL
   ```

3. **Step 3: Prepare Classification Training Samples (`tools/3_prepare_classification_samples.py`)**:
   Universal sample generator from raw LPIS / cadastral parcel vectors (`.shp`, `.gpkg`, `.geojson`):
   ```powershell
   # Netherlands (NL BRP dataset):
   python tools/3_prepare_classification_samples.py -c NL --input path/to/brp.gpkg --crop_col GEWAS --min_area_ha 0.2

   # Poland (PL ARiMR dataset):
   python tools/3_prepare_classification_samples.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME --max_samples_per_class 3000
   ```

4. **Step 4: Calculate Bayesian Acreage Priors (`tools/4_generate_crop_priors.py`)**:
   Calculates statistical real-world crop acreages from LPIS vector datasets for prior probability calibration:
   ```powershell
   python tools/4_generate_crop_priors.py -c NL --input path/to/brp.gpkg --crop_col GEWAS
   python tools/4_generate_crop_priors.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME
   ```

---

## Step-by-Step Execution Guide

### Phase 1: Sentinel-1 SAR Preprocessing

Runs slice calibration, multi-temporal coregistration, and NUTS2 spatial clipping across greedy orbits:

```powershell
# Option A: Ingest directly from CREODIAS (Y: drive)
python 1_Sentinel-1_preprocessor/1a_slice_calibration.py -s 2024-10-15 -e 2025-09-15 -c NL

# Option B: Download directly from Copernicus CDSE API
python 1_Sentinel-1_preprocessor/1c_slice_calibration_cdse.py -s 2024-10-15 -e 2025-09-15 -c NL

# Multi-temporal stack coregistration:
python 1_Sentinel-1_preprocessor/2_coregistration.py -c NL

# Spatial clipping to NUTS2:
python 1_Sentinel-1_preprocessor/3_stack_clip.py -c NL
```
*Outputs*: `workingDir/{COUNTRY}/orbit_{ORBIT}/processed_raster/{COUNTRY}_orbit_{ORBIT}_VH_VV.tif`

---

### Step 4: Sentinel-2 Optical Preprocessing

The Sentinel-2 preprocessor automatically discovers which Sentinel-1 orbits cover the target country, downloads/extracts the optical granules, calculates synthetic time series across 14 agricultural DOYs, and mosaics the raster matched to the Sentinel-1 grid.

```powershell
# Master pipeline for all detected orbits (e.g. orbits 88 and 161 for NL):
python 1a_Sentinel-2_preprocessor/sentinel2_preprocessor.py -s 2025-03-01 -e 2025-09-15 -c NL --source cdse --mode all

# Single orbit override:
python 1a_Sentinel-2_preprocessor/sentinel2_preprocessor.py -s 2025-03-01 -e 2025-09-15 -c NL -o 88 --source cdse --mode all
```

*Outputs*:
- Synthetic DOY Folders: `workingDir/{COUNTRY}/orbit_{ORBIT}/S2_final_preprocessing/mosaic/day{DOY}_{YEAR}/`
- Final Multi-band Timeseries Raster: `workingDir/{COUNTRY}/orbit_{ORBIT}/processed_raster/{COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif`

---

### Step 5: Multimodal S1 + S2 Object-Based Classification

Executes the state-of-the-art **Presto Multimodal + PyTorch MLP + XGBoost Ensemble** classifier:

```powershell
# Run multimodal classification using existing SLIC segmentation:
python 2_classifier/1_classify_MLPXGB_presto_hybrid_S1S2.py --track NL/orbit_88 --seg_mode slic --stage A

# Run multimodal classification using LPIS cadastral parcel vectors:
python 2_classifier/1_classify_MLPXGB_presto_hybrid_S1S2.py --track NL/orbit_88 --seg_mode lpis --stage A

# Interactive menu mode (select specific stages 0-8):
python 2_classifier/1_classify_MLPXGB_presto_hybrid_S1S2.py --track NL/orbit_88 --seg_mode slic
```

*Classification Pipeline Stages*:
- `Stage 0`: Generate Valid Data Footprint Mask
- `Stage 1`: Segmentation (SLIC, SAM, LPIS, Felzenszwalb, OTB Mean-Shift)
- `Stage 2`: Partition `samples.shp` into `learn.shp` (70%) and `control.shp` (30%)
- `Stage 3`: Extract Multimodal Features (S1 SAR stats + S2 Reflectances + 128-d Presto Embeddings)
- `Stage 4`: Train PyTorch Deep MLP + XGBoost GBDT Ensemble
- `Stage 5`: Tiled Object-Based Prediction with Bayesian Prior Probability Adjustment
- `Stage 6`: Cropland Masking (`_classified_masked_*.tif`)
- `Stage 7`: Confidence Masking (`_confidence_masked_*.tif`)
- `Stage 8`: Independent Confusion Matrix & Metric Excel Report (`_metrics_*.xlsx`)

---

### Step 6: Country-Wide Classification Merge

Mosaics all orbit classifications into a single country-wide map with confidence comparison and sieve filtering:

```powershell
# Merge SLIC multimodal classifications for NL:
python 2_classifier/2_merge_classifications.py --track NL --suffix _mlpxgb_presto_slic

# Merge LPIS multimodal classifications for NL:
python 2_classifier/2_merge_classifications.py --track NL --suffix _mlpxgb_presto_lpis
```

*Outputs*:
- Country Classification Raster: `workingDir/{COUNTRY}/classification_results/{COUNTRY}_final_classification_{SUFFIX}.tif`
- Country Validation Report: `workingDir/{COUNTRY}/classification_results/{COUNTRY}_final_metrics_{SUFFIX}.xlsx`

---

## Model Architecture & Segmentation Options

### 1. Unified PyTorch MLP + XGBoost Ensemble
```
                   +-----------------------------------------------+
                   |     Multimodal Object Features (S1 + S2)      |
                   +-----------------------+-----------------------+
                                           |
                   +-----------------------+-----------------------+
                   |                                               |
                   v                                               v
     +---------------------------+                   +---------------------------+
     |     PyTorch Deep MLP      |                   |       XGBoost GBDT        |
     | (BatchNorm1d + Dropout +  |                   | (n_estimators=250,        |
     |   CosineAnnealingLR)      |                   |  max_depth=6, colsample)  |
     +-------------+-------------+                   +-------------+-------------+
                   |                                               |
                   | P(MLP)                                        | P(XGB)
                   +-----------------------+-----------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |    Soft-Voting Probability Blend      |
                       |  P = 0.65 * P(MLP) + 0.35 * P(XGB)    |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |       Bayesian Prior Calibration      |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |       Final Crop Classification       |
                       +---------------------------------------+
```

### 2. Supported Segmentation Modes (`--seg_mode`)
- **`slic`** (Recommended): Fast Simple Linear Iterative Clustering generating superpixel field segments.
- **`lpis`**: Cadastral parcel vector database rasterization (e.g. BRP / ARiMR).
- **`sam`**: Meta AI Segment Anything Model for deep boundary delineation.
- **`felzenszwalb`**: Graph-based multi-resolution segmentation.
- **`otb_meanshift`**: Orfeo ToolBox Mean-Shift spatial clustering.

---

## Troubleshooting & Performance Tuning

### 1. OpenMP / MKL Concurrency Conflicts
If you encounter `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`, set:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="4"
```

### 2. GDAL OpenJPEG JP2 Plugin
If GDAL fails to open Sentinel-2 `.jp2` files with `ERROR 4: plugin gdal_JP2OpenJPEG.dll is not available`, install the plugin:
```powershell
conda install -y -c conda-forge libgdal-jp2openjpeg=3.10.3
```

### 3. SNAP Out of Memory (OOM)
Edit `gpt.vmoptions` located in your SNAP `bin/` directory:
```text
-Xmx16G
```

---

## How to Cite

If you use this software in your research or publications, please cite:

**APA Format:**
> Slesinski, P., Kotulak, N., Roos, M., Mróz, M., Mleczko, M., Gabriel, C., Hofer, N., Belton, S., Logakrishnan, M., Kästenbauer, M., Martins, C., Pallister, I. L. M., Gonçalves, I. (2025). Sentinel-1 & Sentinel-2 OBIA Crop Type Mapping Pipeline (v2.5). [AIML4OS – One Stop Shop for Artificial Intelligence in Official Statistics](https://cros.ec.europa.eu/dashboard/aiml4os). European Commission / Eurostat. Available at: https://github.com/AIML4OS/WP7-Crop-type-mapping

**BibTeX:**
```bibtex
@software{slesinski2025cropmapper,
  author       = {Slesinski, Przemyslaw and Kotulak, Natalia and Roos, Marko and Mróz, Marek and Mleczko, Magdalena and Gabriel, Cristina and Hofer, Nina and Belton, Sam and Logakrishnan, Mohana and Kästenbauer, Mathias and Martins, Carla and Pallister, Ivana I. L. M. and Gonçalves, Isabel},
  title        = {Sentinel-1 & Sentinel-2 OBIA Crop Type Mapping Pipeline},
  version      = {2.5.0},
  year         = {2025},
  url          = {https://github.com/AIML4OS/WP7-Crop-type-mapping},
  organization = {AIML4OS – One Stop Shop for Artificial Intelligence in Official Statistics, Eurostat, European Commission}
}
```
