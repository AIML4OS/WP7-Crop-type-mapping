# AIML CropMapper Cloud: Sentinel-1 & Sentinel-2 OBIA crop type mapping pipeline (v2.5)

An automated, cloud-optimized object-based image analysis (OBIA) pipeline designed to process **Sentinel-1 SAR** and **Sentinel-2 multispectral optical** time series for large-scale, national and regional crop type classification. Developed under the European Statistical System (ESS) **AIML4OS (One Stop Shop for Artificial Intelligence in Official Statistics - Work Package 7)** project funded by Eurostat and the European Commission, this toolbox enables National Statistical Institutes (NSIs), agricultural paying agencies, and IT practitioners to generate standardized, high-accuracy crop type statistics across Europe.

---

## Table of contents
1. [Quick start guide for IT administrators](#quick-start-guide-for-it-administrators)
2. [Data inputs: what you provide vs what is automated](#data-inputs-what-you-provide-vs-what-is-automated)
3. [Plain-language glossary of terms](#plain-language-glossary-of-terms)
4. [Hardware, storage, and system sizing](#hardware-storage-and-system-sizing)
5. [Scientific methodology & foundation model architecture](#scientific-methodology--foundation-model-architecture)
   - [Sentinel-1 SAR radar remote sensing physics](#sentinel-1-sar-radar-remote-sensing-physics)
   - [Sentinel-2 multispectral optical remote sensing](#sentinel-2-multispectral-optical-remote-sensing)
   - [NASA Harvest Presto geospatial foundation model](#nasa-harvest-presto-geospatial-foundation-model)
   - [Meta AI SAM vision foundation model for parcel delineation](#meta-ai-sam-vision-foundation-model-for-parcel-delineation)
   - [Unified PyTorch Deep MLP + XGBoost fusion ensemble](#unified-pytorch-deep-mlp--xgboost-fusion-ensemble)
   - [Bayesian prior probability calibration](#bayesian-prior-probability-calibration)
6. [Complete toolbox guides and architecture](#complete-toolbox-guides-and-architecture)
   - [Toolbox 1: Sentinel-1 SAR preprocessor](#toolbox-1-sentinel-1-sar-preprocessor)
   - [Toolbox 2: Sentinel-2 optical preprocessor](#toolbox-2-sentinel-2-optical-preprocessor)
   - [Toolbox 3: Multimodal machine learning classifier](#toolbox-3-multimodal-machine-learning-classifier)
   - [Toolbox 4: Nationwide multi-orbit merger](#toolbox-4-nationwide-multi-orbit-merger)
   - [Toolbox 5: Data preparation utilities](#toolbox-5-data-preparation-utilities)
7. [Prerequisites and environment setup](#prerequisites-and-environment-setup)
   - [Windows installation](#windows-installation)
   - [Linux installation](#linux-installation)
8. [Modular JSON configuration system](#modular-json-configuration-system)
9. [Ground truth sample specifications (`samples.shp`)](#ground-truth-sample-specifications-samplesshp)
10. [Detailed step-by-step execution guide](#detailed-step-by-step-execution-guide)
    - [Phase 1: Sentinel-1 SAR preprocessing](#phase-1-sentinel-1-sar-preprocessing)
    - [Phase 2: Sentinel-2 optical preprocessing](#phase-2-sentinel-2-optical-preprocessing)
    - [Phase 3: Multimodal crop classification (stages 0 to 7)](#phase-3-multimodal-crop-classification-stages-0-to-7)
    - [Phase 4: Multi-orbit nationwide merge](#phase-4-multi-orbit-nationwide-merge)
11. [High-performance vectorized inference architecture](#high-performance-vectorized-inference-architecture)
12. [National orbit coverage and geographic territory definitions](#national-orbit-coverage-and-geographic-territory-definitions)
13. [How to inspect and use output products](#how-to-inspect-and-use-output-products)
14. [Directory and file lineage structure](#directory-and-file-lineage-structure)
15. [Troubleshooting and FAQ](#troubleshooting-and-faq)
16. [Authors and citation](#authors-and-citation)
17. [License](#license)

---

## Quick start guide for IT administrators

If you are an IT administrator or data engineer running this pipeline for the first time, follow this 5-step checklist to produce a complete crop map:

```
[Step 1: Free CDSE account] ---> [Step 2: JSON configs] ---> [Step 3: Ground truth samples]
                                                                        |
                                                                        v
[Step 5: Run classifier] <--- [Step 4: Run S1 & S2 preprocessors] <-----+
```

1. **Create a free account on the Copernicus Data Space Ecosystem (CDSE)**:
   - Register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).
2. **Set up local configuration files**:
   - Copy `1_Sentinel-1_preprocessor/config_s1.example.json` to `config_s1.json`.
   - Copy `1a_Sentinel-2_preprocessor/config_s2.example.json` to `config_s2.json`.
   - Paste your CDSE username (email) and password into both files.
3. **Place your ground truth samples**:
   - Put your training points in `auxiliary_files/shapefiles_samples/{COUNTRY}/samples.shp` (with attribute `crop_id`).
4. **Run the preprocessors**:
   - Preprocess Sentinel-1 SAR: `python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track NL/orbit_88 --source cdse --stage A`
   - Preprocess Sentinel-2 optical: `python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --track NL/orbit_88 --source cdse --stage A`
5. **Run the multimodal classifier**:
   - `python 2_classifier/run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode slic --stage A`
   *(Tip: Running `python run_classifier.py`, `python run_s1_preprocessor.py`, or `python run_s2_preprocessor.py` without arguments launches the interactive setup wizard!)*

---

## Data inputs: what you provide vs what is automated

| Data type | Source / responsibility | Description |
| :--- | :--- | :--- |
| **Sentinel-1 SAR imagery** | **Automated** (CDSE API / CreoDIAS) | Calibrated $\sigma^0$ radar backscatter time series (VH, VV). |
| **Sentinel-2 optical imagery** | **Automated** (CDSE API / CreoDIAS) | Multispectral surface reflectances (`B02`–`B12`) and NDVI. |
| **Digital Elevation Model (DEM)** | **Automated** (Copernicus 30 m) | Automatically fetched by ESA SNAP for terrain correction. |
| **NUTS administrative boundaries**| **Automated** (`tools/1_download_nuts_boundaries.py`) | Downloaded directly from Eurostat GISCO web service. |
| **Presto foundation model weights**| **Automated** (GitHub / Hugging Face) | Downloaded automatically during first execution. |
| **CDSE credentials** | **User-provided** (Required) | Free email and password entered in `config_s1.json` / `config_s2.json`. |
| **Training point samples** | **User-provided** (Required) | Vector shapefile (`samples.shp`) with field crop labels. |
| **LPIS cadastral parcel vector** | **User-provided** (Optional) | National agricultural parcel database (`.gpkg` / `.shp`). |
| **Crop acreage priors (`priors.json`)**| **User-provided** (Optional) | Statistical crop area distribution for Bayesian calibration. |

---

## Plain-language glossary of terms

* **SAR (Synthetic Aperture Radar - Sentinel-1)**: Active radar satellite sensor that penetrates clouds, rain, and darkness, measuring surface roughness, structure, and moisture.
* **Optical multispectral (Sentinel-2)**: Passive satellite sensor that captures solar reflectance across visible, near-infrared, and shortwave infrared wavelengths. Sensitive to crop greenness, chlorophyll content, and canopy water.
* **DOY (Day of Year)**: Number representing a specific calendar day (e.g., DOY 80 = March 21, DOY 203 = July 22). Used to align satellite observations across different years and orbits into standardized 10-day time steps.
* **SCL (Scene Classification Layer)**: Quality band produced by ESA Sentinel-2 processing that identifies clouds, cloud shadows, snow, and clear land.
* **OBIA (Object-Based Image Analysis)**: Methodology that classifies groups of homogeneous pixels (agricultural fields or superpixels) rather than individual isolated pixels, eliminating "salt-and-pepper" visual noise.
* **LPIS (Land Parcel Identification System)**: Official cadastral geographic database of agricultural parcels maintained by EU Member States for Common Agricultural Policy (CAP) subsidies (e.g., BRP in the Netherlands, ARiMR in Poland, ISIP in Portugal).
* **NUTS (Nomenclature of Territorial Units for Statistics)**: Standard administrative division system of the European Union (e.g., NUTS0 = country, NUTS2 = province/region).
* **BigTIFF & pyramid overviews**: GeoTIFF format extension allowing file sizes $> 4\text{ GB}$. Overviews are pre-calculated reduced-resolution preview layers that allow instant zooming and panning in GIS software without loading the entire 100+ GB file into RAM.

---

## Hardware, storage, and system sizing

| Component | Minimum requirement | Recommended production setup |
| :--- | :--- | :--- |
| **Processor (CPU)** | 8 physical cores (e.g. Intel i7 / AMD Ryzen 7) | 16 to 32 cores (e.g. AMD Ryzen 9 / Threadripper / EPYC, Intel Xeon) |
| **System memory (RAM)** | 32 GB RAM | 64 GB to 128 GB RAM (especially for multi-orbit national merges) |
| **Graphics card (GPU)** | Not strictly required (runs on CPU) | NVIDIA GPU with 8 GB+ VRAM (accelerates PyTorch MLP, Presto, and SAM) |
| **Disk storage** | 500 GB free space | 1 TB to 2 TB NVMe SSD (fast I/O is critical for multi-band BigTIFF processing) |
| **Operating System** | Windows 10/11 (64-bit) or Linux (Ubuntu 22.04+) | Windows 11 Pro 64-bit or Ubuntu Linux 22.04 / 24.04 LTS |

---

## Scientific methodology & foundation model architecture

### Sentinel-1 SAR radar remote sensing physics
Sentinel-1 operates at C-band microwave frequency ($5.405\text{ GHz}$, wavelength $\lambda \approx 5.55\text{ cm}$). Microwaves penetrate cloud cover, interacting with canopy structure, leaf density, and soil moisture:
* **$VV$ polarization**: Dominantly sensitive to surface roughness, soil moisture, and vertical cereal stems (wheat, barley).
* **$VH$ polarization**: Cross-polarization sensitive to volume scattering within the crop canopy (biomass accumulation, canopy closure).
* **$VH/VV$ ratio**: Normalizes soil moisture variations, highlighting crop phenological transitions and heading phases.

### Sentinel-2 multispectral optical remote sensing
Utilizes 9 spectral bands at 10 m and 20 m spatial resolutions (resampled to 10 m):
* **Visible bands (`B02` Blue, `B03` Green, `B04` Red)**: Sensitive to photosynthetic pigment absorption (chlorophyll $a$ and $b$).
* **RedEdge bands (`B05` 705 nm, `B06` 740 nm, `B07` 783 nm)**: Sharp reflectance transition region; highly sensitive to canopy nitrogen, leaf area index (LAI), and early senescence.
* **Narrow NIR (`B8A` 865 nm)**: Measures internal leaf cellular structure scattering, avoiding atmospheric water vapor contamination.
* **Shortwave Infrared (`B11` 1610 nm, `B12` 2190 nm)**: Strongly absorbed by liquid water; detects crop water stress and canopy dry matter.
* **14 standardized DOYs**: Interpolates cloud-free observations into regular 10-day agricultural reference dates:
  $$\text{DOYs} = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]$$

### NASA Harvest Presto geospatial foundation model
**Presto** is a geospatial transformer foundation model pretrained on global multi-sensor Earth observation time series:
* **Transformer encoder**: Self-attention mechanism across irregular multi-temporal sequences capturing crop phenological dynamics.
* **Multi-sensor tokenization**: Jointly encodes Sentinel-1 SAR ($\sigma^0_{VV}, \sigma^0_{VH}$) and Sentinel-2 optical reflectances ($B02$–$B12$).
* **Spatial & temporal position encodings**: Sinusoidal positional embeddings of geographic coordinates $(\text{latitude}, \text{longitude})$ and acquisition month tokens.
* **Latent token extraction**: Produces **128-dimensional multi-temporal token embeddings** from S1 SAR sequences and **128-dimensional embeddings** from S2 optical sequences.

### Meta AI SAM vision foundation model for parcel delineation
* Adapts Meta AI Segment Anything Model (SAM) Vision Transformers (`vit_h`, `vit_l`, `vit_b`) for remote sensing imagery.
* Generates field parcel boundaries from multi-temporal composite rasters with bilateral edge preservation and distance transform hole-filling.

### Unified PyTorch Deep MLP + XGBoost fusion ensemble
* **PyTorch Deep MLP**: 3-layer neural network with Batch Normalization (`BatchNorm1d`), Dropout ($p=0.3$), Class-weighted Cross-Entropy loss, and Cosine Annealing learning rate schedule.
* **XGBoost GBDT**: Ensemble of 250 gradient boosted decision trees (`max_depth=6`, `subsample=0.8`, `colsample_bytree=0.25`) with histogram-based splitting.
* **Soft-voting probability blend**: $\hat{P}(C_i | X) = 0.65 \cdot P_{\text{MLP}}(C_i | X) + 0.35 \cdot P_{\text{XGB}}(C_i | X)$.

### Bayesian prior probability calibration
Aligns raw machine learning predictions with real-world statistical crop acreage proportions from national agricultural registries (`priors.json`):

$$P_{\text{calibrated}}(C_i | X) = \frac{P_{\text{model}}(C_i | X) \cdot \left(\frac{P_{\text{true}}(C_i)}{P_{\text{train}}(C_i)}\right)^\gamma}{\sum_{j=1}^K P_{\text{model}}(C_j | X) \cdot \left(\frac{P_{\text{true}}(C_j)}{P_{\text{train}}(C_j)}\right)^\gamma}$$

Where $\gamma = 0.7$ is the damping exponent preventing extreme boundary distortions.

---

## Complete toolbox guides and architecture

### Toolbox 1: Sentinel-1 SAR preprocessor

Located in `1_Sentinel-1_preprocessor/`, this toolbox transforms raw Copernicus Sentinel-1 Level-1 GRD products into multi-temporal, orthorectified, calibrated backscatter stacks.

```
[Copernicus CDSE API / CreoDIAS COG]
                 |
                 v
    +---------------------------+
    |  Stage 1: Calibration     |  --> Precise orbit (POEORB), TNR, BNR, Sigma0 calibration,
    |  & slice assembly         |      and daily slice stitching into continuous orbit strips
    +-------------+-------------+
                  |
                  v
    +---------------------------+
    |  Stage 2: Coregistration  |  --> ESA SNAP multi-temporal cross-correlation coregistration
    |  (wrapped stack)          |      aligning all dates across the agricultural season
    +-------------+-------------+
                  |
                  v
    +---------------------------+
    |  Stage 3: Terrain corr.,  |  --> Range Doppler terrain correction (Copernicus 30 m DEM),
    |  BigTIFF stack & clipping |      NUTS2 clipping (EPSG:3857, 10 m), and 6 pyramid overviews
    +---------------------------+
```

* **Master runner script**: `run_s1_preprocessor.py`
* **Internal modules (`modules/`)**:
  * `s1_calibration_creodias.py`: Fast calibration from local CreoDIAS COG repositories.
  * `s1_calibration_cdse.py`: Automated retrieval and calibration from CDSE API.
  * `s1_coregistration.py`: Multi-temporal coregistration using ESA SNAP GPT (`CreateStack`).
  * `s1_stack_clip.py`: Range Doppler terrain correction using Copernicus 30 m DEM, GDAL BigTIFF stacking, and GISCO NUTS2 regional boundary clipping in `EPSG:3857`.

---

### Toolbox 2: Sentinel-2 optical preprocessor

Located in `1a_Sentinel-2_preprocessor/`, this toolbox creates cloud-free, regular 10-day synthetic optical time-series composites matching 1:1 with the Sentinel-1 SAR pixel grid.

```
[Copernicus CDSE API / CreoDIAS L2A Archive]
                     |
                     v
    +----------------------------------+
    |  Stage 1: Retrieval & Masking    |  --> S2 L2A tile download/extraction, Scene Classification
    |  (SCL Cloud & Shadow Filtering)  |      Layer (SCL) filtering for clouds, shadows, and snow
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  Stage 2: Synthetic DOY          |  --> Multi-temporal interpolation across 14 standardized DOYs:
    |  Time-Series Interpolation       |      [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  Stage 3: SAR Grid Alignment     |  --> Sub-pixel resampling to Sentinel-1 raster bounding box,
    |  & 126-Band BigTIFF Stacking     |      126-band BigTIFF creation, and 6 pyramid overviews
    +----------------------------------+
```

* **Master runner script**: `run_s2_preprocessor.py`
* **Internal modules (`modules/`)**:
  * `s2_download_cdse.py`: Automated search, download, and SCL cloud masking from CDSE API.
  * `s2_extract_creodias.py`: Direct extraction from local CreoDIAS archives (`Y:/Sentinel-2/MSI/L2A`).
  * `s2_time_series.py`: Pure Python multi-temporal interpolation across 14 standardized agricultural DOYs.
  * `s2_mosaic_stack.py`: Mosaicking, sub-pixel grid alignment to Sentinel-1 SAR raster, 126-band BigTIFF creation, and pyramid overview generation.
  * `s2_pipeline.py`: Object-oriented pipeline orchestrator.

---

### Toolbox 3: Multimodal machine learning classifier

Located in `2_classifier/`, this toolbox implements object-based image analysis (OBIA) classification using state-of-the-art multimodal fusion, deep learning foundation models, and Bayesian statistical calibration.

```
+----------------------------------------------------------------------------------------------------+
|                                      MULTIMODAL CLASSIFICATION SUITE                               |
+----------------------------------------------------------------------------------------------------+
|  [0] Footprint Gen.     --> Intersection of valid S1 SAR and S2 Optical coverage                   |
|  [1] Segmentation       --> Object delineation: SLIC superpixels / Meta AI SAM / LPIS parcels      |
|  [2] Sample Partition   --> Stratified split: 70% Learn (train) / 30% Control (validation)         |
|  [3] Feature Extraction --> S1 Stats + S2 Reflectances + Presto 128d S1/S2 Token Embeddings        |
|  [4] Model Training     --> Class-weighted PyTorch Deep MLP + XGBoost GBDT Fusion Ensemble         |
|  [5] Vectorized Infer.  --> High-performance tile inference with Bayesian Prior Calibration        |
|  [6] Cropland Masking   --> Agricultural mask application & non-cropland suppression               |
|  [7] Accuracy Reporting --> Out-of-bag validation metrics, F1-scores, and styled Excel report      |
+----------------------------------------------------------------------------------------------------+
```

* **Master runner script**: `run_classifier.py`
* **Internal engines (`modules/`)**:
  * `classifier_mlpxgb_presto.py`: Multimodal fusion ensemble combining NASA Harvest Presto transformer embeddings, PyTorch Deep MLP, and XGBoost GBDT (`[S1 + S2] [SOTA]`).
  * `classifier_presto_s1.py`: Single-radar Presto artificial neural network for SAR-only classification (`[S1 only]`).
  * `classifier_otb.py`: Orfeo ToolBox machine learning models (Random Forest / Support Vector Machines) (`[S1 + S2]`).
  * `presto_model.py`: Embedded NASA Harvest Presto transformer foundation architecture.

---

### Toolbox 4: Nationwide multi-orbit merger

Located in `2_classifier/`, this tool combines multiple single-orbit classification maps into a single, seamless nationwide raster.

```
+----------------------------------------------------------------------------------------------------+
|                         PHASE 4: NATIONWIDE MERGE PIPELINE FLOWCHART                               |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Orbit 1 Track Map]           [Orbit 2 Track Map]        |
                     | - Classified GeoTIFF          - Classified GeoTIFF       |
                     | - Confidence GeoTIFF          - Confidence GeoTIFF       |
                     +---------------------+--------------------+---------------+
                                           |                    |
                                           +---------+----------+
                                                     |
                                                     v
                     +----------------------------------------------------------+
                     | [Multi-Orbit Confidence-Weighted Blending]               |
                     | In overlapping swath zones:                              |
                     | Pixel = Orbit_1 if Conf_1 > Conf_2 else Orbit_2          |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Post-Processing & Masking]                              |
                     | 1. Morphological sieve filtering (clump size < 10 px)    |
                     | 2. National agricultural cropland mask application       |
                     | 3. Aggregated national confusion matrix computation      |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [National Products: national_products/]                  |
                     | 1. {COUNTRY}_national_crop_map_{MODE}.tif                |
                     | 2. {COUNTRY}_national_confidence_{MODE}.tif              |
                     | 3. {COUNTRY}_national_accuracy_report_{MODE}.xlsx        |
                     +----------------------------------------------------------+
```

* **Master runner script**: `run_merge.py`
* **Internal engine (`modules/`)**:
  * `multi_orbit_merger.py`: Confidence-weighted mosaic blending, morphological sieve post-processing, and national statistical aggregation.

---

### Toolbox 5: Data preparation utilities

Located in `tools/`, these standalone command-line scripts assist in preparing auxiliary datasets:

* `1_download_nuts_boundaries.py`: Downloads GISCO NUTS administrative boundaries from Eurostat in GeoJSON and Shapefile formats.
* `2_build_agricultural_mask.py`: Generates standardized high-resolution cropland masks (`EPSG:3857`, 10 m) from national LPIS parcel vectors or Copernicus CLMS/HRL raster layers.
* `3_prepare_classification_samples.py`: Extracts point ground truth samples (`samples.shp`) from national agricultural parcel databases (e.g., BRP in the Netherlands, ARiMR in Poland, ISIP in Portugal).
* `4_generate_crop_priors.py`: Calculates statistical real-world crop acreage proportions for Bayesian prior calibration (`priors.json`).
* `5_build_raster_overviews.py`: Universal multi-scale pyramid overviews builder (`[2, 4, 8, 16, 32, 64]`) with LZW/DEFLATE compression for instant QGIS/ArcGIS rendering.

---

## Prerequisites and environment setup

### Windows installation

1. **Install Python via Miniforge**:
   - Download and install [Miniforge3](https://github.com/conda-forge/miniforge) for Windows (64-bit).
   - Open **Miniforge Prompt** and run:
     ```bash
     mamba env create -f environment.yml
     mamba activate aiml_env
     conda install -y -c conda-forge libgdal-jp2openjpeg
     ```
2. **Install ESA SNAP**:
   - Download and install SNAP from [ESA SNAP Download](https://step.esa.int/main/download/snap-download/).
   - Default recommended path: `D:\Program Files\esa-snap`.

---

### Linux installation

```bash
# System packages (Ubuntu / Debian)
sudo apt-get update && sudo apt-get install -y gdal-bin libgdal-dev build-essential unzip

# Conda environment
mamba env create -f environment.yml
mamba activate aiml_env

# ESA SNAP for Linux
chmod +x esa-snap_sentinel_unix_*.sh
./esa-snap_sentinel_unix_*.sh -q
```

---

## Modular JSON configuration system

Configuration is stored in **modular JSON files** located directly in their respective component directories:

| Config file | Purpose | Location |
| :--- | :--- | :--- |
| **`config_s1.json`** | Sentinel-1 pipeline parameters, SNAP paths, and CDSE credentials | `1_Sentinel-1_preprocessor/config_s1.json` |
| **`config_s2.json`** | Sentinel-2 pipeline parameters, spectral bands, DOYs, and CDSE credentials | `1a_Sentinel-2_preprocessor/config_s2.json` |

### Initializing configuration from templates

```powershell
# Sentinel-1 configuration setup
cp 1_Sentinel-1_preprocessor/config_s1.example.json 1_Sentinel-1_preprocessor/config_s1.json

# Sentinel-2 configuration setup
cp 1a_Sentinel-2_preprocessor/config_s2.example.json 1a_Sentinel-2_preprocessor/config_s2.json
```

Edit `config_s1.json` and `config_s2.json` with your Copernicus CDSE credentials:
```json
{
  "cdse": {
    "username": "your_email@domain.com",
    "password": "your_cdse_password"
  },
  "paths": {
    "working_dir": "D:/AIML_CropMapper_Cloud/workingDirs",
    "aux_dir": "D:/AIML_CropMapper_Cloud/auxiliary_files"
  }
}
```

---

## Ground truth sample specifications (`samples.shp`)

Place your point ground truth dataset at:
`auxiliary_files/shapefiles_samples/{COUNTRY}/samples.shp` (e.g. `NL/samples.shp`, `PL/samples.shp`, `PT/samples.shp`).

* **Geometry type**: Point (`OGRPoint`). Points should be located inside agricultural fields.
* **Required attribute**: **`crop_id`** (Integer): Numerical crop class identifier (e.g., `11` = Winter Wheat, `12` = Maize, `1430` = Rapeseed).
* **Sample partitioning**: The pipeline automatically partitions points into **70% training (`learn.shp`)** and **30% validation (`control.shp`)**.

---

## Detailed step-by-step execution guide

### Phase 1: Sentinel-1 SAR preprocessing

```powershell
# Run full automated pipeline for an entire country across all orbits:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PT -s 2024-10-15 -e 2025-09-15 --stage A

# Run full automated pipeline for a single orbit track:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 -s 2024-10-15 -e 2025-09-15 --stage A

# Force downloading directly from Copernicus Data Space (CDSE API):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PT --source cdse -s 2024-10-15 -e 2025-09-15 --stage A

# Exclude winter freeze period (December 1 to February 14):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PL --exclude_winter --stage A

# Interactive menu mode (prompts for country, orbits, dates, and stages):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py
```

---

### Phase 2: Sentinel-2 optical preprocessing

```powershell
# Run full automated pipeline for an entire country across all orbits:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Run full automated pipeline for a single orbit track:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --track PT/orbit_52 -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Force downloading directly from Copernicus Data Space (CDSE API):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT --source cdse -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Interactive menu mode (prompts for country, orbits, dates, and stages):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py
```

---

### Phase 3: Multimodal crop classification (stages 0 to 7)

```powershell
# Mode 1: Multimodal Deep MLP + XGBoost + Presto with SLIC Superpixels (Recommended SOTA):
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode slic --stage A

# Run classification for an entire country across all orbits:
python 2_classifier/run_classifier.py --country PT --classifier mlpxgb_presto --seg_mode slic --stage A

# Mode 2: Official Cadastral LPIS Parcel Segmentation:
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode lpis --lpis_vector path/to/parcels.gpkg --stage A

# Mode 3: Vision Foundation Model Segmentation (Meta AI SAM):
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode sam --stage A

# Mode 4: Single-radar S1-only Presto ANN Classifier:
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier presto_s1 --seg_mode slic --stage A

# Mode 5: Orfeo ToolBox Machine Learning (Random Forest / SVM):
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier otb --seg_mode slic --stage A

# Interactive setup wizard (prompts for track, classifier model, segmentation, and stage):
python 2_classifier/run_classifier.py
```

---

### Phase 4: Multi-orbit nationwide merge

```powershell
# Merge SLIC classifications into a seamless national map:
python 2_classifier/run_merge.py --country PT --seg_mode slic

# Merge LPIS classifications into a seamless national map:
python 2_classifier/run_merge.py --country PT --seg_mode lpis
```

---

## High-performance vectorized inference architecture

In version 2.5, Stage 5 (`stage_5_classify_vector`) has been re-architected with **vectorized block-level I/O and O(1) lookup table (LUT) raster reconstruction**:

| Metric / feature | Legacy approach | Vectorized v2.5 architecture | Improvement |
| :--- | :---: | :---: | :---: |
| **Disk I/O requests per orbit** | $> 50,000,000$ random reads | **$\sim 85,000$ sequential block reads** | 🟢 **99.83% I/O reduction** |
| **Zonal statistics aggregation** | Python per-object looping | Vectorized `np.bincount` in NumPy/C | 🟢 **$> 1000\times$ faster** |
| **Tile raster reconstruction** | Iterative dictionary mask assignment | Direct array indexing `lut_pred[sub_seg]` | 🟢 **$2\text{ ms}$ per tile** |
| **Average time per tile ($2048 \times 2048$ px)** | $2.5\text{ to }4.0\text{ minutes}$ | **$4\text{ to }7\text{ seconds}$** | 🚀 **$\sim 35\times$ speedup** |
| **Total Stage 5 runtime (full country orbit)** | **$24\text{ to }48\text{ hours}$** | **$30\text{ to }50\text{ minutes}$** | ⚡ **$\sim 40\times\text{ to }60\times\text{ faster}$** |

---

## National orbit coverage and geographic territory definitions

To optimize processing time and eliminate redundant data downloads, optimal descending and ascending orbit track combinations are configured for each European territory, with polar/Arctic and remote oceanic islands excluded from agricultural classification:

| Country | Code | Processing orbits | Geographic coverage & exclusions |
| :--- | :---: | :--- | :--- |
| **Netherlands** | `NL` | `[88, 161]` (2 descending) | 100% mainland Netherlands & Wadden Islands |
| **Portugal** | `PT` | `[52, 125]` (2 descending) | 100% continental Portugal (Lisbon to Bragança) |
| **Poland** | `PL` | `[22, 51, 95, 124, 168]` (5 descending) | 100% Polish national territory |
| **Spain** | `ES` | `[1, 30, 59, 74, 103, 132, 147]` (7 ascending) | 100% mainland Spain + Balearic Islands (Mallorca, Menorca, Ibiza) |
| **France** | `FR` | `[8, 37, 81, 110, 154]` (5 descending) | 100% metropolitan France + Corsica |
| **Italy** | `IT` | `[22, 44, 95, 117, 168]` (5 descending) | 100% mainland Italy + Sicily + Sardinia |
| **Germany** | `DE` | `[37, 66, 110, 139, 168]` (5 descending) | 100% German federal territory |
| **Norway** | `NO` | `[8, 37, 66, 124, 153, 168]` (6 descending) | 100% mainland Norway (Kristiansand to Nordkapp; Svalbard excluded) |
| **Denmark** | `DK` | `[8, 37, 110, 139]` (4 descending) | 100% mainland Denmark, Zealand, Funen (Faroe Islands excluded) |
| **United Kingdom** | `UK` | `[81, 154, 103, 132]` (4 ascending/descending) | 100% Great Britain & Northern Ireland (Shetland excluded) |

---

## How to inspect and use output products

All outputs are saved in the sequential directory structure under `workingDirs/`:

1. **Per-orbit classification map (`workingDirs/{COUNTRY}/orbit_{ORBIT}/2_classification/3_maps/`)**:
   - `*_classified_masked_*.tif`: Single-band raster where each pixel value corresponds to a numeric `crop_id`.
   - `*_confidence_masked_*.tif`: Single-band floating-point raster ($0.0$ to $1.0$) indicating classification confidence.
   - Open in **QGIS** or **ArcGIS Pro**:
     - Right-click layer $\rightarrow$ **Properties** $\rightarrow$ **Symbology** $\rightarrow$ select **Paletted / Unique values**.
     - Click **Classify** to display distinct colors for each crop class.
2. **Per-orbit accuracy report (`workingDirs/{COUNTRY}/orbit_{ORBIT}/2_classification/4_reports/`)**:
   - `report_*_metrics_*.xlsx`: Excel spreadsheet containing:
     - **Overall Accuracy (OA)**: Percentage of correctly classified validation samples.
     - **Cohen's Kappa ($\kappa$)**: Statistical metric accounting for chance agreement.
     - **Full confusion matrix**: Rows represent ground truth; columns represent model predictions.
     - **Per-crop metrics**: Precision (User's Accuracy), Recall (Producer's Accuracy), and F1-score for each crop class.
3. **Nationwide merged products (`workingDirs/{COUNTRY}/national_products/`)**:
   - `{COUNTRY}_national_crop_map_{SEG_MODE}.tif`: Seamless national crop type GeoTIFF.
   - `{COUNTRY}_national_confidence_{SEG_MODE}.tif`: Seamless national confidence GeoTIFF.
   - `{COUNTRY}_national_accuracy_report_{SEG_MODE}.xlsx`: Comprehensive national statistical report.

---

## Directory and file lineage structure

```text
AIML_CropMapper_Cloud/
├── 1_Sentinel-1_preprocessor/           # Sentinel-1 SAR pipeline
│   ├── run_s1_preprocessor.py           # Unified S1 CLI & interactive wizard
│   ├── config_s1.json                   # Active S1 config (CDSE credentials, SNAP paths)
│   ├── config_s1.example.json           # Template S1 config
│   ├── modules/                         # Core S1 processing modules
│   │   ├── s1_calibration_creodias.py   # CreoDIAS local GRDH calibration
│   │   ├── s1_calibration_cdse.py       # CDSE API GRDH calibration
│   │   ├── s1_coregistration.py         # SNAP multi-temporal coregistration
│   │   └── s1_stack_clip.py             # Time-series stacking & NUTS2 clipping
│   └── Archive_scripts/                 # Frozen historical archive (read-only)
├── 1a_Sentinel-2_preprocessor/          # Sentinel-2 optical pipeline
│   ├── run_s2_preprocessor.py           # Unified S2 CLI & interactive wizard
│   ├── config_s2.json                   # Active S2 config (CDSE credentials, DOYs, bands)
│   ├── config_s2.example.json           # Template S2 config
│   ├── modules/                         # Core S2 processing modules
│   │   ├── s2_extract_creodias.py       # CreoDIAS local tile extraction & SCL masking
│   │   ├── s2_download_cdse.py          # CDSE API granule download & SCL masking
│   │   ├── s2_time_series.py            # 14-DOY synthetic time-series interpolation
│   │   ├── s2_mosaic_stack.py           # S1 grid matching & 126-band BigTIFF stacking
│   │   └── s2_pipeline.py               # Pipeline orchestrator
│   └── Archive_scripts/                 # Frozen historical archive (read-only)
├── 2_classifier/                        # Multimodal machine learning suite
│   ├── run_classifier.py                # Unified classifier CLI & interactive wizard
│   ├── run_merge.py                     # Multi-orbit national mosaic & merger
│   ├── modules/                         # Core classification engines
│   │   ├── classifier_mlpxgb_presto.py  # S1+S2 Presto + PyTorch MLP + XGBoost SOTA
│   │   ├── classifier_presto_s1.py      # Single-radar S1-only Presto ANN model
│   │   ├── classifier_otb.py            # Orfeo ToolBox machine learning models
│   │   ├── multi_orbit_merger.py        # Nationwide multi-orbit confidence merger
│   │   └── presto_model.py              # NASA Harvest Presto foundation architecture
│   └── Archive_scripts/                 # Frozen historical archive (read-only)
├── tools/                               # Standalone preparation utilities
│   ├── 1_download_nuts_boundaries.py    # GISCO NUTS boundaries downloader
│   ├── 2_build_agricultural_mask.py     # Cropland mask builder
│   ├── 3_prepare_classification_samples.py # Ground truth sample extractor
│   ├── 4_generate_crop_priors.py        # Bayesian crop acreage priors calculator
│   └── 5_build_raster_overviews.py      # Universal GDAL pyramid overviews generator
├── auxiliary_files/                     # Auxiliary data
│   ├── raster_files/AgriMasks/{COUNTRY}/# High-resolution agricultural masks & LPIS
│   ├── shapefiles_nuts/                 # Downloaded GISCO NUTS boundaries
│   ├── shapefiles_samples/{COUNTRY}/    # Ground truth samples (samples.shp) & priors.json
│   ├── Presto_models/                   # Pre-trained Presto foundation model weights
│   └── SAM_models/                      # Meta AI Segment Anything Model weights
├── workingDirs/                         # Unified sequential output working directory
│   └── {COUNTRY}/
│       ├── national_products/           # [Phase 4] Seamless national crop map & metrics (.xlsx)
│       └── orbit_{ORBIT}/
│           ├── 1_input_stacks/          # Multimodal S1 (Sigma0) & S2 (126-band) BigTIFF stacks
│           ├── 2_classification/        # Full lineage classification products
│           │   ├── 0_segmentation/      # Data footprint & OBIA segments (SLIC, SAM, LPIS)
│           │   ├── 1_samples_and_features/ # Train/test shapefiles & extracted feature vectors (CSV)
│           │   ├── 2_models/            # Serialized trained model weights (.pkl)
│           │   ├── 3_maps/              # Classified GeoTIFFs & confidence maps
│           │   └── 4_reports/           # Validation accuracy reports & confusion matrices (.xlsx)
│           └── _temp_processing/        # Isolated transient buffer (slices, tiles, DOY mosaics)
├── environment.yml                      # Conda / Mamba environment definition
└── README.md                            # Complete technical documentation
```

---

## Troubleshooting and FAQ

### 1. OpenMP / MKL concurrency conflict
If you encounter `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`, set:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="4"
```

### 2. GDAL OpenJPEG plugin missing
If GDAL fails to open Sentinel-2 `.jp2` files with `ERROR 4: plugin gdal_JP2OpenJPEG.dll is not available`:
```powershell
conda install -y -c conda-forge libgdal-jp2openjpeg=3.10.3
```

### 3. SNAP out of memory (OOM)
Edit `gpt.vmoptions` in your SNAP installation `bin/` directory:
```text
-Xmx16G
```

### 4. High disk I/O latency when opening large BigTIFF rasters
Run the pyramid overviews utility on your output rasters to enable smooth zooming:
```powershell
python tools/5_build_raster_overviews.py -i workingDirs/NL/orbit_88/1_input_stacks/NL_orbit_88_S2_timeseries.tif
```

---

## Authors and citation

If you use this software in your research or statistical production pipelines, please cite:

**APA format:**
> Slesinski, P., Kotulak, N., Roos, M., Mróz, M., Mleczko, M., Gabriel, C., Hofer, N., Belton, S., Logakrishnan, M., Kästenbauer, M., Martins, C., Pallister, I. L. M., Gonçalves, I. (2025). *Sentinel-1 & Sentinel-2 OBIA crop type mapping pipeline (v2.5)*. [AIML4OS – One Stop Shop for Artificial Intelligence in Official Statistics](https://cros.ec.europa.eu/dashboard/aiml4os), Work Package 7, European Commission / Eurostat. Available at: https://github.com/AIML4OS/WP7-Crop-type-mapping

**BibTeX:**
```bibtex
@software{slesinski2025cropmapper,
  author       = {Slesinski, Przemyslaw and Kotulak, Natalia and Roos, Marko and Mróz, Marek and Mleczko, Magdalena and Gabriel, Cristina and Hofer, Nina and Belton, Sam and Logakrishnan, Mohana and Kästenbauer, Mathias and Martins, Carla and Pallister, Ivana I. L. M. and Gonçalves, Isabel},
  title        = {Sentinel-1 & Sentinel-2 OBIA crop type mapping pipeline},
  version      = {2.5.0},
  year         = {2025},
  url          = {https://github.com/AIML4OS/WP7-Crop-type-mapping},
  organization = {AIML4OS – One Stop Shop for Artificial Intelligence in Official Statistics, Eurostat, European Commission}
}
```

---

## License

This project is licensed under the Apache License 2.0. Developed under the European Statistical System (ESS) AIML4OS initiative funded by Eurostat and the European Commission.
