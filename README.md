# AIML CropMapper Cloud: Sentinel-1 & Sentinel-2 OBIA crop type mapping pipeline (v2.5)

An automated, cloud-optimized object-based image analysis (OBIA) pipeline designed to process **Sentinel-1 SAR** and **Sentinel-2 multispectral optical** time series for large-scale, national and regional crop type classification. Developed under the European Statistical System (ESS) **AIML4OS (One Stop Shop for Artificial Intelligence in Official Statistics - Work Package 7)** project funded by Eurostat and the European Commission, this toolbox enables National Statistical Institutes (NSIs), agricultural paying agencies, and IT practitioners to generate standardized, high-accuracy crop type statistics across Europe.

---

## Table of contents
1. [Quick start guide for IT administrators](#quick-start-guide-for-it-administrators)
2. [Data inputs: what you provide vs what is automated](#data-inputs-what-you-provide-vs-what-is-automated)
3. [Plain-language glossary of terms](#plain-language-glossary-of-terms)
4. [Hardware, storage, and system sizing](#hardware-storage-and-system-sizing)
5. [Complete toolbox guides and architecture](#complete-toolbox-guides-and-architecture)
   - [Toolbox 1: Sentinel-1 SAR preprocessor](#toolbox-1-sentinel-1-sar-preprocessor)
   - [Toolbox 2: Sentinel-2 optical preprocessor](#toolbox-2-sentinel-2-optical-preprocessor)
   - [Toolbox 3: Multimodal machine learning classifier](#toolbox-3-multimodal-machine-learning-classifier)
   - [Toolbox 4: Nationwide multi-orbit merger](#toolbox-4-nationwide-multi-orbit-merger)
   - [Toolbox 5: Data preparation utilities](#toolbox-5-data-preparation-utilities)
6. [Prerequisites and environment setup](#prerequisites-and-environment-setup)
   - [Windows installation](#windows-installation)
   - [Linux installation](#linux-installation)
7. [Modular JSON configuration system](#modular-json-configuration-system)
8. [Ground truth sample specifications (`samples.shp`)](#ground-truth-sample-specifications-samplesshp)
9. [Detailed step-by-step execution guide](#detailed-step-by-step-execution-guide)
   - [Phase 1: Sentinel-1 SAR preprocessing](#phase-1-sentinel-1-sar-preprocessing)
   - [Phase 2: Sentinel-2 optical preprocessing](#phase-2-sentinel-2-optical-preprocessing)
   - [Phase 3: Multimodal crop classification (stages 0 to 7)](#phase-3-multimodal-crop-classification-stages-0-to-7)
   - [Phase 4: Multi-orbit nationwide merge](#phase-4-multi-orbit-nationwide-merge)
10. [High-performance vectorized inference architecture](#high-performance-vectorized-inference-architecture)
11. [National orbit coverage and geographic territory definitions](#national-orbit-coverage-and-geographic-territory-definitions)
12. [How to inspect and use output products](#how-to-inspect-and-use-output-products)
13. [Directory and file lineage structure](#directory-and-file-lineage-structure)
14. [Troubleshooting and FAQ](#troubleshooting-and-faq)
15. [Authors and citation](#authors-and-citation)
16. [License](#license)

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

> [!NOTE]
> Processing one standard satellite orbit pass across an entire country requires approximately 150 GB to 250 GB of transient disk space. Once processing completes, transient buffers (`calibrated/`, `slice_assembly/`, `wrapped/`, `_temp_processing/`) can be deleted, leaving only the final compact BigTIFF stacks in `1_input_stacks/`.

---

## Complete toolbox guides and architecture

### Toolbox 1: Sentinel-1 SAR preprocessor

Located in `1_Sentinel-1_preprocessor/`, this toolbox transforms raw Copernicus Sentinel-1 Level-1 GRD products into multi-temporal, orthorectified, calibrated backscatter stacks.

* **Master runner script**: `run_s1_preprocessor.py`
* **Internal processing modules (`modules/`)**:
  * `s1_calibration_creodias.py`: Fast calibration from local CreoDIAS Cloud-Optimized GeoTIFF (COG) repositories.
  * `s1_calibration_cdse.py`: Automated retrieval and calibration from the Copernicus Data Space Ecosystem (CDSE API).
  * `s1_coregistration.py`: Multi-temporal coregistration using ESA SNAP GPT (`CreateStack`).
  * `s1_stack_clip.py`: Range Doppler terrain correction using Copernicus 30 m DEM, GDAL BigTIFF stacking, and GISCO NUTS2 regional boundary clipping in `EPSG:3857`.

#### Processing stages:
* **Stage 1 (Calibration & slice assembly)**: Downloads or reads raw SAR slices, applies precise orbit files (POEORB), thermal noise removal (TNR), border noise removal (BNR), calibrates to $\sigma^0$ (in dB), and merges daily slices into continuous orbit tracks.
* **Stage 2 (Multi-temporal coregistration)**: Aligns all acquisition dates across the agricultural season to a common master grid using cross-correlation.
* **Stage 3 (Terrain correction & BigTIFF stacking)**: Orthorectifies radar backscatter using Copernicus 30 m DEM, creates the final dual-polarization (`VH`, `VV`) multi-temporal BigTIFF stack, clips it to the country footprint, and builds 6 pyramid overview levels.

---

### Toolbox 2: Sentinel-2 optical preprocessor

Located in `1a_Sentinel-2_preprocessor/`, this toolbox creates cloud-free, regular 10-day synthetic optical time-series composites matching 1:1 with the Sentinel-1 SAR pixel grid.

* **Master runner script**: `run_s2_preprocessor.py`
* **Internal processing modules (`modules/`)**:
  * `s2_download_cdse.py`: Automated search, download, and SCL cloud masking from CDSE API.
  * `s2_extract_creodias.py`: Direct extraction from local CreoDIAS archives (`Y:/Sentinel-2/MSI/L2A`).
  * `s2_time_series.py`: Pure Python multi-temporal interpolation across 14 standardized agricultural DOYs.
  * `s2_mosaic_stack.py`: Mosaicking, sub-pixel grid alignment to Sentinel-1 SAR raster, 126-band BigTIFF creation, and pyramid overview generation.
  * `s2_pipeline.py`: Object-oriented pipeline orchestrator.

#### Processing stages:
* **Stage 1 (Granule download / extraction & cloud masking)**: Gathers Sentinel-2 L2A tiles covering the country orbit, applying Scene Classification Layer (SCL) masks to remove clouds, cirrus, cloud shadows, and snow.
* **Stage 2 (Synthetic DOY time-series interpolation)**: Generates regular 10-day observations across 14 agricultural reference dates:
  $$\text{DOYs} = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]$$
  Covers 9 spectral bands (`B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B8A`, `B11`, `B12`) plus dynamic NDVI, producing 126 spectral layers per orbit.
* **Stage 3 (Mosaicking, SAR grid matching & BigTIFF stacking)**: Warps and resamples the optical mosaic to match the exact bounding box and resolution of the Sentinel-1 SAR stack ($\Delta X = 0.000\text{ m}, \Delta Y = 0.000\text{ m}$ at 10.0 m resolution), saving the product as a BigTIFF with DEFLATE compression and 6 pyramid overviews.

---

### Toolbox 3: Multimodal machine learning classifier

Located in `2_classifier/`, this toolbox implements object-based image analysis (OBIA) classification using state-of-the-art multimodal fusion, deep learning foundation models, and Bayesian statistical calibration.

* **Master runner script**: `run_classifier.py`
* **Internal classification engines (`modules/`)**:
  * `classifier_mlpxgb_presto.py`: Multimodal fusion ensemble combining NASA Harvest Presto transformer embeddings, PyTorch Deep MLP, and XGBoost GBDT (`[S1 + S2] [SOTA]`).
  * `classifier_presto_s1.py`: Single-radar Presto artificial neural network for SAR-only classification (`[S1 only]`).
  * `classifier_otb.py`: Orfeo ToolBox machine learning models (Random Forest / Support Vector Machines) (`[S1 + S2]`).
  * `presto_model.py`: Embedded NASA Harvest Presto transformer foundation architecture.

#### Complete breakdown of classification stages (0 to 7):

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

* **Stage 0 (Footprint generation)**: Computes the spatial intersection of valid Sentinel-1 radar and Sentinel-2 optical data, creating a binary mask (`*_data_footprint.tif`).
* **Stage 1 (Object-based segmentation)**: Delineates agricultural parcel boundaries using one of three supported methods:
  * `slic`: Fast Simple Linear Iterative Clustering superpixels with 64 px tile buffering.
  * `sam`: Deep Vision Transformer segmentation (Meta AI Segment Anything Model).
  * `lpis`: Official European cadastral parcel vector database rasterization.
* **Stage 2 (Stratified sample split)**: Partitions ground truth points (`samples.shp`) into independent training (`learn_*.shp`, 70%) and validation (`control_*.shp`, 30%) sets with per-class stratification.
* **Stage 3 (Multimodal feature extraction)**: Extracts temporal backscatter profiles, optical reflectances, and queries the NASA Harvest Presto foundation model to obtain 128-dimensional SAR and 128-dimensional optical embeddings.
* **Stage 4 (Train fusion ensemble)**: Fits a class-balanced PyTorch Deep MLP neural network (BatchNorm, Dropout, Cosine Annealing LR) and an XGBoost GBDT model on the extracted multimodal features.
* **Stage 5 (Object-based inference with Bayesian priors)**: Executes tile-based prediction across the entire raster, weighting raw model probabilities by statistical crop acreage distributions:
  $$P_{\text{calibrated}}(C_i | X) = \frac{P_{\text{model}}(C_i | X) \cdot \frac{P_{\text{true}}(C_i)}{P_{\text{sample}}(C_i)}}{\sum_{j=1}^{K} P_{\text{model}}(C_j | X) \cdot \frac{P_{\text{true}}(C_j)}{P_{\text{sample}}(C_j)}}$$
* **Stage 6 (Apply agricultural masks)**: Restricts predictions to active agricultural areas using the country agricultural mask (`*_agri_mask_*.tif`).
* **Stage 7 (Calculate accuracy metrics & export Excel report)**: Evaluates the classification against the independent 30% control dataset, computing Overall Accuracy (OA), Cohen's Kappa ($\kappa$), per-crop Precision, Recall, and F1-scores, and exports an Excel workbook (`report_*.xlsx`).

---

### Toolbox 4: Nationwide multi-orbit merger

Located in `2_classifier/`, this tool combines multiple single-orbit classification maps into a single, seamless nationwide raster.

* **Master runner script**: `run_merge.py`
* **Internal engine (`modules/`)**:
  * `multi_orbit_merger.py`: Confidence-weighted mosaic blending, morphological sieve post-processing, and national statistical aggregation.

#### Key capabilities:
* **Confidence-weighted blending**: In overlapping swath zones between adjacent satellite tracks, assigns each pixel to the orbit classification that exhibited higher confidence ($P_{\text{conf}}$).
* **Morphological sieve filtering**: Removes isolated single-pixel noise (clump size $< 10$ pixels).
* **National reporting**: Aggregates validation metrics across all national tracks into a unified national Excel report.

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
