# AIML CropMapper Cloud: Sentinel-1 & Sentinel-2 OBIA crop type mapping pipeline (v2.5)

An automated, cloud-optimized object-based image analysis (OBIA) pipeline designed to process **Sentinel-1 SAR** and **Sentinel-2 multispectral optical** time series for large-scale, national and regional crop type classification. Developed under the European Statistical System (ESS) **AIML4OS (One Stop Shop for Artificial Intelligence in Official Statistics - Work Package 7)** project funded by Eurostat and the European Commission, this toolbox enables National Statistical Institutes (NSIs), agricultural paying agencies, and IT practitioners to generate standardized, high-accuracy crop type statistics across Europe.

---

## Table of contents
1. [Quick start guide for IT administrators](#quick-start-guide-for-it-administrators)
2. [Data inputs: what you provide vs what is automated](#data-inputs-what-you-provide-vs-what-is-automated)
3. [Plain-language glossary of terms](#plain-language-glossary-of-terms)
4. [Hardware, storage, and system sizing](#hardware-storage-and-system-sizing)
5. [Pipeline architecture and scientific methodology](#pipeline-architecture-and-scientific-methodology)
   - [Sentinel-1 SAR radar preprocessing](#sentinel-1-sar-radar-preprocessing)
   - [Sentinel-2 multispectral optical preprocessing](#sentinel-2-multispectral-optical-preprocessing)
   - [Object-based segmentation paradigms](#object-based-segmentation-paradigms)
   - [NASA Harvest Presto foundation model embeddings](#nasa-harvest-presto-foundation-model-embeddings)
   - [Unified PyTorch MLP + XGBoost ensemble classifier](#unified-pytorch-mlp--xgboost-ensemble-classifier)
   - [Bayesian prior probability calibration](#bayesian-prior-probability-calibration)
   - [Multi-orbit confidence merge and sieve post-processing](#multi-orbit-confidence-merge-and-sieve-post-processing)
6. [Prerequisites and environment setup](#prerequisites-and-environment-setup)
   - [Windows installation](#windows-installation)
   - [Linux installation](#linux-installation)
7. [Modular JSON configuration system](#modular-json-configuration-system)
8. [Ground truth sample specifications (`samples.shp`)](#ground-truth-sample-specifications-samplesshp)
9. [Data preparation utilities (`tools/`)](#data-preparation-utilities-tools)
10. [Step-by-step execution guide](#step-by-step-execution-guide)
    - [Phase 1: Sentinel-1 SAR preprocessing](#phase-1-sentinel-1-sar-preprocessing)
    - [Phase 2: Sentinel-2 optical preprocessing](#phase-2-sentinel-2-optical-preprocessing)
    - [Phase 3: Multimodal crop classification](#phase-3-multimodal-crop-classification)
    - [Phase 4: Multi-orbit nationwide merge](#phase-4-multi-orbit-nationwide-merge)
11. [How to inspect and use output products](#how-to-inspect-and-use-output-products)
12. [Directory and file structure](#directory-and-file-structure)
13. [Troubleshooting and FAQ](#troubleshooting-and-faq)
14. [Authors and citation](#authors-and-citation)
15. [License](#license)

---

## Quick start guide for IT administrators

If you are an IT administrator or data engineer running this pipeline for the first time, follow this 5-step checklist to produce a complete crop map:

```
[Step 1: Free CDSE Account] ---> [Step 2: JSON Configs] ---> [Step 3: Ground Truth Samples]
                                                                        |
                                                                        v
[Step 5: Run Classifier] <--- [Step 4: Run S1 & S2 Preprocessors] <-----+
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
   - Preprocess Sentinel-2 Optical: `python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --track NL/orbit_88 --source cdse --stage A`
5. **Run the multimodal classifier**:
   - `python 2_classifier/run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode slic --stage A`
   *(Tip: You can also simply run `python 2_classifier/run_classifier.py` without arguments to launch the interactive setup wizard!)*

---

## Data inputs: what you provide vs what is automated

| Data type | Source / Responsibility | Description |
| :--- | :--- | :--- |
| **Sentinel-1 SAR imagery** | **Automated** (CDSE API / CREODIAS) | Calibrated $\sigma^0$ radar backscatter time series (VH, VV). |
| **Sentinel-2 optical imagery** | **Automated** (CDSE API / CREODIAS) | Multispectral surface reflectances (`B02`–`B12`) and NDVI. |
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
* **Optical Multispectral (Sentinel-2)**: Passive satellite sensor that captures solar reflectance across visible, near-infrared, and shortwave infrared wavelengths. Sensitive to crop greenness, chlorophyll content, and canopy water.
* **DOY (Day of Year)**: Number representing a specific calendar day (e.g., DOY 80 = March 21, DOY 203 = July 22). Used to align satellite observations across different years and orbits into standardized 10-day time steps.
* **SCL (Scene Classification Layer)**: Quality band produced by ESA Sentinel-2 processing that identifies clouds, cloud shadows, snow, and clear land.
* **OBIA (Object-Based Image Analysis)**: Methodology that classifies groups of homogeneous pixels (agricultural fields or superpixels) rather than individual isolated pixels, eliminating "salt-and-pepper" visual noise.
* **LPIS (Land Parcel Identification System)**: Official cadastral geographic database of agricultural parcels maintained by EU Member States for Common Agricultural Policy (CAP) subsidies (e.g., BRP in the Netherlands, ARiMR in Poland, ISIP in Portugal).
* **NUTS (Nomenclature of Territorial Units for Statistics)**: Standard administrative division system of the European Union (e.g., NUTS0 = country, NUTS2 = province/region).
* **BigTIFF & Pyramid Overviews**: GeoTIFF format extension allowing file sizes $> 4\text{ GB}$. Overviews are pre-calculated reduced-resolution preview layers that allow instant zooming and panning in GIS software without loading the entire 100+ GB file into RAM.

---

## Hardware, storage, and system sizing

| Component | Minimum requirement | Recommended production setup |
| :--- | :--- | :--- |
| **Processor (CPU)** | 8 physical cores (e.g. Intel i7 / AMD Ryzen 7) | 16 to 32 cores (e.g. AMD Ryzen 9 / Threadripper / EPYC, Intel Xeon) |
| **System Memory (RAM)** | 32 GB RAM | 64 GB to 128 GB RAM (especially for multi-orbit national merges) |
| **Graphics Card (GPU)** | Not strictly required (runs on CPU) | NVIDIA GPU with 8 GB+ VRAM (accelerates PyTorch MLP, Presto, and SAM) |
| **Disk Storage** | 500 GB free space | 1 TB to 2 TB NVMe SSD (fast I/O is critical for multi-band BigTIFF processing) |
| **Operating System** | Windows 10/11 (64-bit) or Linux (Ubuntu 22.04+) | Windows 11 Pro 64-bit or Ubuntu Linux 22.04 / 24.04 LTS |

> [!NOTE]
> Processing one standard satellite orbit pass across an entire country requires approximately 150 GB to 250 GB of disk space (including intermediate calibrated slices, synthetic DOY mosaics, and the final 126-band BigTIFF stack).

---

## Pipeline architecture and scientific methodology

### Sentinel-1 SAR radar preprocessing

The Sentinel-1 pipeline processes Level-1 Ground Range Detected (GRD) Interferometric Wide (IW) swath products:

1. **Radiometric calibration**: Automated ESA SNAP GPT graph applying precise orbit state vectors (POEORB), thermal noise removal (TNR), border noise removal (BNR), calibration to backscatter coefficient $\sigma^0$ (in dB), and Range Doppler terrain correction using Copernicus 30 m DEM.
2. **Deburst and slice assembly**: Merges consecutive slices along each orbit track into a single continuous strip.
3. **Multi-temporal coregistration**: Performs sub-pixel cross-correlation coregistration across the entire agricultural season (autumn to autumn).
4. **Administrative clipping**: Clips rasters to GISCO NUTS2 administrative regions in `EPSG:3857` at an exact 10.0 m pixel resolution.

---

### Sentinel-2 multispectral optical preprocessing

Produces cloud-free, regular 10-day synthetic optical time-series composites matching 1:1 with the Sentinel-1 SAR grid:

1. **Automated granule retrieval**: Downloads Sentinel-2 L2A tiles from the CDSE API or local CREODIAS paths (`Y:/Sentinel-2/MSI/L2A`).
2. **Cloud/shadow filtering**: Masks invalid observations using the Scene Classification Layer (SCL).
3. **Synthetic DOY interpolation**: Pure Python multi-temporal interpolation across 14 standardized agricultural DOYs:
   $$\text{DOYs} = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]$$
   Interpolation covers 9 spectral bands (`B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B8A`, `B11`, `B12`) plus dynamic NDVI, generating 126 spectral layers per orbit.
4. **Sub-pixel geometry matching**: Resamples optical mosaics to the exact bounding box and resolution of the Sentinel-1 SAR raster ($\Delta X = 0.000\text{ m}, \Delta Y = 0.000\text{ m}$).
5. **BigTIFF and pyramid creation**: Automatically builds multi-scale pyramid overviews (`[2, 4, 8, 16, 32, 64]`) with LZW compression.

---

### Object-based segmentation paradigms

Supports five segmentation options for agricultural field delineation:

```
                               +--------------------------------------------+
                               |     High-SNR Multi-Temporal Composite      |
                               +---------------------+----------------------+
                                                     |
            +-----------------------+----------------+----------------+-----------------------+
            |                       |                                 |                       |
            v                       v                                 v                       v
+-----------------------+ +--------------------+            +--------------------+ +--------------------+
|  SLIC Superpixels     | |  Meta AI SAM-Geo   |            |  LPIS Cadastral    | |  OTB / Graph-Based |
|  - 64px Tile Buffers  | |  - ViT-H/L/B Deep  |            |  - Vector Parcels  | |  - Mean-Shift      |
|  - Dynamic Density    | |  - 8-Process Pool  |            |  - BBox Spatial Q. | |  - Felzenszwalb    |
|  - Fast & Scalable    | |  - Bilateral+CLAHE |            |  - GDAL Rasterize  | |  - Benchmarking    |
+-----------+-----------+ +---------+----------+            +---------+----------+ +---------+----------+
            |                       |                                 |                       |
            +-----------------------+----------------+----------------+-----------------------+
                                                     |
                                                     v
                               +--------------------------------------------+
                               |    Homogeneous Agricultural Segments       |
                               +--------------------------------------------+
```

1. **`slic` (Simple Linear Iterative Clustering - Recommended for general operational use)**:
   - High-speed superpixel generation on multi-temporal composite rasters.
   - Buffered tiling (`buffer=64px`) prevents edge artifacts across block boundaries.
2. **`sam` (Meta AI Segment Anything Model - Deep foundation segmentation)**:
   - Vision Transformer architectures (`vit_h`, `vit_l`, `vit_b`) adapted for remote sensing.
   - Multi-process parallel inference (8 worker pool) with bilateral filtering and distance transform hole-filling.
3. **`lpis` (Official Cadastral Parcel Vectors - Cadastral ground truth)**:
   - Bounding-box spatial query (`pyogrio` bbox query) loads only relevant intersecting parcels from multi-gigabyte vector files.
   - Converts 3D/Z geometries to 2D polygons, reprojects to `EPSG:3857`, and applies GDAL rasterization.

---

### NASA Harvest Presto foundation model embeddings

**Presto** is a geospatial transformer foundation model pretrained on global multi-sensor satellite imagery:
* **Feature extraction**: Generates **128-dimensional multi-temporal token embeddings** from S1 SAR sequences and **128-dimensional embeddings** from S2 optical reflectance arrays.
* **Token fusion**: Combines multi-temporal spectral dynamics with geographic coordinates, dynamic land cover tokens, and acquisition month indices.

---

### Unified PyTorch MLP + XGBoost ensemble classifier

```
                   +-----------------------------------------------+
                   |     Multimodal Object Features (S1 + S2)      |
                   |   (SAR Stats + Optical Reflectances + Presto) |
                   +-----------------------+-----------------------+
                                           |
                   +-----------------------+-----------------------+
                   |                                               |
                   v                                               v
     +---------------------------+                   +---------------------------+
     |     PyTorch Deep MLP      |                   |       XGBoost GBDT        |
     | - BatchNorm1d + Dropout   |                   | - 250 Estimators          |
     | - ReLU Activations        |                   | - Max Depth: 6            |
     | - Dynamic Class Weights   |                   | - Subsample / Colsample   |
     | - Cosine Annealing LR     |                   | - Fast Multicore Trees    |
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
                       |  P(C|X) ~ P(X|C) * P_true(C)/P_smp(C) |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |       Final Crop Classification       |
                       |   (Raster Map + Confidence + Stats)   |
                       +---------------------------------------+
```

1. **PyTorch Deep MLP**: 3-layer architecture with Batch Normalization, Dropout, Class-weighted Cross-Entropy loss, and Cosine Annealing learning rate schedule.
2. **XGBoost GBDT**: Gradient boosted decision trees with `n_estimators=250`, `max_depth=6`, and stratified sample balancing.
3. **Soft-voting blend**: Combines predicted class probability vectors: $\hat{P} = 0.65 \cdot P_{\text{MLP}} + 0.35 \cdot P_{\text{XGB}}$.

---

### Bayesian prior probability calibration

Aligns machine learning predictions with statistical real-world crop acreage proportions:

$$P_{\text{calibrated}}(C_i | X) = \frac{P_{\text{model}}(C_i | X) \cdot \frac{P_{\text{true}}(C_i)}{P_{\text{sample}}(C_i)}}{\sum_{j=1}^{K} P_{\text{model}}(C_j | X) \cdot \frac{P_{\text{true}}(C_j)}{P_{\text{sample}}(C_j)}}$$

---

### Multi-orbit confidence merge and sieve post-processing

* **Confidence comparison**: In overlapping orbit zones, assigns pixels to the class with higher prediction confidence.
* **Morphological sieve filtering**: Removes isolated single-pixel noise (clump size $< 10$ pixels).
* **Agricultural masking**: Confines final classifications to active agricultural cropland boundaries.

---

## Prerequisites and environment setup

### Windows installation

1. **Install Python via Miniforge**:
   - Download [Miniforge3](https://github.com/conda-forge/miniforge) for Windows.
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
    "working_dir": "D:/AIML_CropMapper_Cloud/workingDir",
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

## Data preparation utilities (`tools/`)

### 1. Download GISCO NUTS boundaries (`tools/1_download_nuts_boundaries.py`)
```powershell
python tools/1_download_nuts_boundaries.py -c NL
python tools/1_download_nuts_boundaries.py -c PL
python tools/1_download_nuts_boundaries.py --all
```

### 2. Build agricultural cropland mask (`tools/2_build_agricultural_mask.py`)
```powershell
# From LPIS cadastral parcel vectors:
python tools/2_build_agricultural_mask.py -c NL --lpis path/to/brp.gpkg
python tools/2_build_agricultural_mask.py -c PL --lpis path/to/arimr.shp

# From Copernicus HRL / CLMS raster tiles:
python tools/2_build_agricultural_mask.py -c NL
```

### 3. Generate classification training samples (`tools/3_prepare_classification_samples.py`)
```powershell
# Extract points from Dutch BRP parcel database:
python tools/3_prepare_classification_samples.py -c NL --input path/to/brp.gpkg --crop_col GEWAS --min_area_ha 0.2

# Extract points from Polish ARiMR parcel database:
python tools/3_prepare_classification_samples.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME --max_samples_per_class 3000
```

### 4. Compute Bayesian crop acreage priors (`tools/4_generate_crop_priors.py`)
```powershell
python tools/4_generate_crop_priors.py -c NL --input path/to/brp.gpkg --crop_col GEWAS
python tools/4_generate_crop_priors.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME
```

### 5. Multi-scale pyramid overviews builder (`tools/5_build_raster_overviews.py`)
```powershell
# Single raster file:
python tools/5_build_raster_overviews.py -i workingDirs/NL/orbit_88/1_input_stacks/NL_orbit_88_S2_timeseries.tif

# Entire orbit input stacks directory:
python tools/5_build_raster_overviews.py -d workingDirs/NL/orbit_88/1_input_stacks/

# All country orbits and national products:
python tools/5_build_raster_overviews.py -c NL
```

---

## Step-by-step execution guide

### Phase 1: Sentinel-1 SAR preprocessing

```powershell
# Run full automated pipeline for an entire country with explicit agricultural season dates:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PT -s 2024-10-15 -e 2025-09-15 --stage A

# Run full automated pipeline for a single orbit:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_81 -s 2024-10-15 -e 2025-09-15 --stage A

# Force downloading directly from Copernicus Data Space (CDSE API):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PT --source cdse -s 2024-10-15 -e 2025-09-15 --stage A

# Interactive menu mode (prompts for dates, orbits, and stages):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py
```

---

### Phase 2: Sentinel-2 optical preprocessing

```powershell
# Run full automated pipeline for an entire country with explicit agricultural season dates:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT -s 2024-10-15 -e 2025-09-15 --stage A

# Run full automated pipeline for a single orbit:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --track PT/orbit_81 -s 2024-10-15 -e 2025-09-15 --stage A

# Force downloading directly from Copernicus Data Space (CDSE API):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT --source cdse -s 2024-10-15 -e 2025-09-15 --stage A

# Interactive menu mode (prompts for dates, orbits, and stages):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py
```

---

### Phase 3: Multimodal crop classification

```powershell
# Run full automated pipeline with SLIC superpixels (Recommended SOTA):
python 2_classifier/run_classifier.py --track PT/orbit_81 --classifier mlpxgb_presto --seg_mode slic --stage A

# Run full automated pipeline for an entire country across all orbits:
python 2_classifier/run_classifier.py --country PT --classifier mlpxgb_presto --seg_mode slic --stage A

# Run full automated pipeline with official LPIS cadastral parcel vectors:
python 2_classifier/run_classifier.py --track PT/orbit_81 --classifier mlpxgb_presto --seg_mode lpis --lpis_vector path/to/parcels.gpkg --stage A

# Run full automated pipeline with Meta AI SAM deep segmentation:
python 2_classifier/run_classifier.py --track PT/orbit_81 --classifier mlpxgb_presto --seg_mode sam --stage A

# Single-radar S1-only Presto ANN classifier:
python 2_classifier/run_classifier.py --track PT/orbit_81 --classifier presto_s1 --seg_mode slic --stage A

# Orfeo ToolBox Machine Learning classifier:
python 2_classifier/run_classifier.py --track PT/orbit_81 --classifier otb --seg_mode slic --stage A

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

## How to inspect and use output products

All outputs are saved in the sequential directory structure in `workingDirs/`:

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

## Directory and file structure

```text
AIML_CropMapper_Cloud/
├── 1_Sentinel-1_preprocessor/           # Sentinel-1 SAR pipeline
│   ├── run_s1_preprocessor.py           # Unified S1 CLI & interactive menu
│   ├── config_s1.json                   # Active S1 config (CDSE credentials, SNAP paths)
│   ├── config_s1.example.json           # Template S1 config
│   ├── modules/                         # Core S1 processing modules
│   │   ├── s1_calibration_creodias.py   # CREODIAS local GRDH calibration
│   │   ├── s1_calibration_cdse.py       # CDSE API GRDH calibration
│   │   ├── s1_coregistration.py         # SNAP multi-temporal coregistration
│   │   └── s1_stack_clip.py             # Time-series stacking & NUTS2 clipping
│   └── Archive_scripts/                 # Frozen historical archive (read-only)
├── 1a_Sentinel-2_preprocessor/          # Sentinel-2 optical pipeline
│   ├── run_s2_preprocessor.py           # Unified S2 CLI & interactive menu
│   ├── config_s2.json                   # Active S2 config (CDSE credentials, DOYs, bands)
│   ├── config_s2.example.json           # Template S2 config
│   ├── modules/                         # Core S2 processing modules
│   │   ├── s2_extract_creodias.py       # CREODIAS local tile extraction & SCL masking
│   │   ├── s2_download_cdse.py          # CDSE API granule download & SCL masking
│   │   ├── s2_time_series.py            # 14-DOY synthetic time-series interpolation
│   │   ├── s2_mosaic_stack.py           # S1 grid matching & 126-band BigTIFF stacking
│   │   └── s2_pipeline.py               # Pipeline orchestrator
│   └── Archive_scripts/                 # Frozen historical archive (read-only)
├── 2_classifier/                        # Multimodal machine learning suite
│   ├── run_classifier.py                # Unified classifier CLI & interactive menu
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

### 3. SNAP Out of Memory (OOM)
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
