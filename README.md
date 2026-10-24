# AIML CropMapper Cloud: Sentinel-1 & Sentinel-2 OBIA crop type mapping pipeline (v3.0)

An automated, cloud-optimized object-based image analysis (OBIA) pipeline designed to process **Sentinel-1 SAR** and **Sentinel-2 multispectral optical** time series for large-scale, national and regional crop type classification. Developed under the European Statistical System (ESS) **AIML4OS (One Stop Shop for Artificial Intelligence in Official Statistics - Work Package 7)** project funded by Eurostat and the European Commission, this toolbox enables National Statistical Institutes (NSIs), agricultural paying agencies, and IT practitioners to generate standardized, high-accuracy crop type statistics across Europe.

---

## Table of contents
1. [Quick start guide for IT administrators](#quick-start-guide-for-it-administrators)
2. [Data inputs: what you provide vs what is automated](#data-inputs-what-you-provide-vs-what-is-automated)
3. [Plain-language glossary of terms](#plain-language-glossary-of-terms)
4. [Hardware, storage, and system sizing](#hardware-storage-and-system-sizing)
5. [Scientific methodology & foundation model architecture](#scientific-methodology--foundation-model-architecture)
   - [Sentinel-1 SAR radar remote sensing physics & polarimetry](#sentinel-1-sar-radar-remote-sensing-physics--polarimetry)
   - [Sentinel-2 multispectral optical remote sensing & vegetation indices](#sentinel-2-multispectral-optical-remote-sensing--vegetation-indices)
   - [NASA Harvest Presto geospatial foundation model](#nasa-harvest-presto-geospatial-foundation-model)
   - [Object-based image analysis (OBIA) & segmentation deep dive](#object-based-image-analysis-obia--segmentation-deep-dive)
   - [Multimodal feature fusion architecture](#multimodal-feature-fusion-architecture)
   - [Hybrid multi-paradigm machine learning ensemble](#hybrid-multi-paradigm-machine-learning-ensemble)
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
9. [Ground truth sample specifications (samples.shp)](#ground-truth-sample-specifications-samplesshp)
10. [Detailed step-by-step execution guide](#detailed-step-by-step-execution-guide-a-to-z-manual)
    - [Phase 1: Sentinel-1 SAR preprocessing](#phase-1-sentinel-1-sar-preprocessing-run_s1_preprocessorpy)
    - [Phase 2: Sentinel-2 optical preprocessing](#phase-2-sentinel-2-optical-preprocessing-run_s2_preprocessorpy)
    - [Phase 3: Multimodal crop classification (stages 1 to 8)](#phase-3-multimodal-machine-learning-classifier-run_classifierpy)
    - [Phase 4: Multi-orbit nationwide merge](#phase-4-nationwide-multi-orbit-merge-run_mergepy)
    - [Phase 5: How to visualize and validate maps in QGIS & ArcGIS Pro](#phase-5-how-to-visualize-and-validate-maps-in-qgis--arcgis-pro)
11. [Cloud-native geospatial engineering & memory architecture](#cloud-native-geospatial-engineering--memory-architecture)
12. [High-performance vectorized inference architecture](#high-performance-vectorized-inference-architecture)
13. [Directory and file lineage structure](#directory-and-file-lineage-structure)
14. [Troubleshooting and FAQ](#troubleshooting-and-faq)
15. [Third-party software, foundation models, and data licenses](#third-party-software-foundation-models-and-data-licenses)
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
   - Enter your CDSE username (email) and password into both files.
3. **Place your ground truth samples**:
   - Put your training points in `auxiliary_files/shapefiles_samples/{COUNTRY}/samples.shp` (with attribute `crop_id`).
4. **Run the preprocessors**:
   - Preprocess Sentinel-1 SAR: `python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track NL/orbit_88 --source cdse -s 2024-10-15 -e 2025-09-15 --stage A`
   - Preprocess Sentinel-2 optical: `python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --track NL/orbit_88 --source cdse -s 2024-10-15 -e 2025-09-15 --stage A`
5. **Run the multimodal classifier**:
   - `python 2_classifier/run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode slic --stage A`
   *(Tip: Running `python run_classifier.py`, `python run_s1_preprocessor.py`, or `python run_s2_preprocessor.py` without arguments launches the interactive setup wizard!)*

---

## Data inputs: what you provide vs what is automated

| Data type | Source / responsibility | Description |
| :--- | :--- | :--- |
| **Sentinel-1 SAR imagery** | **Automated** (CDSE API / CreoDIAS) | Calibrated Sigma0 radar backscatter time series (VH, VV). |
| **Sentinel-2 optical imagery** | **Automated** (CDSE API / CreoDIAS) | Multispectral surface reflectances (`B02`–`B12`) and NDVI. |
| **Digital Elevation Model (DEM)** | **Automated** (Copernicus 30 m) | Automatically fetched by ESA SNAP for terrain correction. |
| **NUTS administrative boundaries**| **Automated** (`tools/1_download_nuts_boundaries.py`) | Downloaded directly from Eurostat GISCO web service. |
| **Presto foundation model weights**| **Automated** (GitHub / Hugging Face) | Downloaded automatically during first execution. |
| **CDSE credentials** | **User-provided** (Required) | Free email and password entered in `config_s1.json` / `config_s2.json`. |
| **Training point samples** | **User-provided** (Required) | Vector shapefile (`samples.shp`) with field crop labels. |
| **LPIS cadastral parcel vector** | **User-provided** (Optional) | National agricultural parcel database (`.gpkg` / `.shp`). |
| **Dynamic Bayesian acreage calibration** | **Automated** (Intelligent dynamic Bayesian) | 100% automated calibration combining physical field area multi-lingual taxonomy and training loss bias inversion (no external JSON required). |

---

## Plain-language glossary of terms

* **SAR (Synthetic Aperture Radar - Sentinel-1)**: Active radar satellite sensor that penetrates clouds, rain, and darkness, measuring surface roughness, structure, and moisture.
* **Optical multispectral (Sentinel-2)**: Passive satellite sensor that captures solar reflectance across visible, near-infrared, and shortwave infrared wavelengths. Sensitive to crop greenness, chlorophyll content, and canopy water.
* **DOY (Day of Year)**: Number representing a specific calendar day (e.g., DOY 80 = March 21, DOY 203 = July 22). Used to align satellite observations across different years and orbits into standardized 10-day time steps.
* **SCL (Scene Classification Layer)**: Quality band produced by ESA Sentinel-2 processing that identifies clouds, cloud shadows, snow, and clear land.
* **OBIA (Object-Based Image Analysis)**: Methodology that classifies groups of homogeneous pixels (agricultural fields or superpixels) rather than individual isolated pixels, eliminating "salt-and-pepper" visual noise.
* **LPIS (Land Parcel Identification System)**: Official cadastral geographic database of agricultural parcels maintained by EU Member States for Common Agricultural Policy (CAP) subsidies (e.g., BRP in the Netherlands, ARiMR in Poland, ISIP in Portugal).
* **NUTS (Nomenclature of Territorial Units for Statistics)**: Standard administrative division system of the European Union (e.g., NUTS0 = country, NUTS2 = province/region).
* **BigTIFF & pyramid overviews**: GeoTIFF format extension allowing file sizes greater than 4 GB. Overviews are pre-calculated reduced-resolution preview layers that allow instant zooming and panning in GIS software without loading the entire 100+ GB file into RAM.

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

### Sentinel-1 SAR radar remote sensing physics & polarimetry

Sentinel-1 operates at C-band Synthetic Aperture Radar frequency (5.405 GHz, wavelength ~5.55 cm). Microwaves penetrate cloud cover, precipitation, and solar illumination variations, directly measuring canopy volumetric structure, dielectric constant, and soil moisture.

```
+----------------------------------------------------------------------------------------------------+
|                             SENTINEL-1 SAR POLARIMETRIC SCATTERING REGIMES                         |
+----------------------------------------------------------------------------------------------------+
|  [VV Polarization]  --> Co-polarized wave interacting with vertical cereal stems & soil roughness  |
|                         (Direct surface backscatter & dielectric soil moisture sensitivity)        |
|  [VH Polarization]  --> Cross-polarized wave generated by multiple volume scattering in canopy     |
|                         (Direct proxy for green biomass accumulation, Leaf Area Index & closure)   |
|  [VH/VV Ratio]      --> Normalized polarimetric ratio canceling soil roughness & moisture effects  |
|                         (Tracks crop phenological transitions, stem elongation, heading & ripening)|
+----------------------------------------------------------------------------------------------------+
```

#### 1. Radar backscatter calibration
Raw Sentinel-1 images contain digital number (DN) values. The pipeline calibrates them to normalized radar cross section (Sigma0) using look-up tables (LUT), removing antenna gain and range spreading loss.

#### 2. Radiometric terrain correction (RTC) & decibel transformation
To eliminate topographic illumination distortions on sloped terrain, backscatter is orthorectified using Copernicus 30 m DEM and converted to decibels:
`dB = 10 * log10(Sigma0)`
This standardizes the radar signal dynamic range between -28 dB and 0 dB.

#### 3. Agricultural polarimetric indices
* **Radar Vegetation Index (RVI)**: Measures canopy volume scattering relative to total backscatter:
  `RVI = 4 * VH / (VV + VH)`
* **Polarimetric cross-ratio (CR)**:
  `CR [dB] = VH [dB] - VV [dB]`

---

### Sentinel-2 multispectral optical remote sensing & vegetation indices

Captures Bottom-Of-Atmosphere (BOA) surface reflectances across 9 spectral bands at 10 m and 20 m spatial resolutions (resampled to 10.0 m in `EPSG:3857`). Atmospheric correction removes Rayleigh scattering, aerosol optical depth, and ozone/water absorption.

#### 1. Spectral band sensitivities
* **Visible bands (`B02` Blue 490 nm, `B03` Green 560 nm, `B04` Red 665 nm)**: Sensitive to photosynthetic chlorophyll absorption.
* **RedEdge bands (`B05` 705 nm, `B06` 740 nm, `B07` 783 nm)**: Capture the steep reflectance transition edge; highly sensitive to canopy nitrogen, leaf chlorophyll concentration, and early senescence.
* **Narrow NIR (`B8A` 865 nm)**: Measures internal leaf mesophyll cellular scattering, avoiding atmospheric water vapor absorption in broad `B08`.
* **Shortwave Infrared (`B11` 1610 nm, `B12` 2190 nm)**: Sensitive to foliar water content and dry cellulose matter.

#### 2. Narrow-band agricultural indices
* **Normalized Difference Vegetation Index (NDVI)**:
  `NDVI = (B8A - B04) / (B8A + B04)`
* **Red-Edge Chlorophyll Index (NDRE1 & NDRE2)**:
  `NDRE1 = (B06 - B05) / (B06 + B05)`
  `NDRE2 = (B07 - B05) / (B07 + B05)`
* **Normalized Difference Water Index (NDWI / NDII)**:
  `NDWI = (B8A - B11) / (B8A + B11)`

#### 3. Standardized 14-DOY agricultural nodes & regional phenological profiles
Temporal observations are interpolated into 14 standardized agricultural reference dates (Day of Year) tailored to specific European bioclimatic zones:
* **Mediterranean profile (`PT`, `ES`, `IT`, `EL`, `CY`, etc.)**:
  DOYs: `[45, 65, 85, 105, 120, 135, 150, 165, 180, 200, 220, 240, 260, 280]`
  Captures early February green-up (DOY 45 ~ Feb 14), early cereal heading, summer drought senescence, and autumn emergence.
* **Continental & Central European profile (`PL`, `NL`, `DE`, `FR`, etc.)**:
  DOYs: `[80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]`
  Optimized for spring green-up (DOY 80 ~ March 21), peak summer vegetative growth, and autumn harvest.
* **Northern European profile (`SE`, `FI`, `EE`, `LT`, `LV`, `NO`)**:
  DOYs: `[105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300]`
  Calibrated for delayed boreal snowmelt and late autumn harvest windows.

#### 4. Whittaker penalized least-squares smoother (`--enable_whittaker`)
To suppress residual cloud artifacts, shadow dips, and atmospheric fluctuations across the reconstructed sequence, the pipeline incorporates an optional, vectorized Whittaker smoother (`lambda = 15.0`). It balances fidelity to observations with a curvature smoothness penalty, yielding continuous, physically realistic phenological trajectories.

#### 5. High-throughput ZSTD compression & temporal stack naming
* **Zstandard (ZSTD level 3, predictor 2)**: Optical BigTIFF cubes utilize high-speed ZSTD compression, yielding 35% to 45% reduction in storage footprint with significantly faster multi-threaded decompression rates than legacy DEFLATE.
* **Temporal acquisition signature**: Mosaicked BigTIFF stacks feature explicit temporal range indicators matching the Sentinel-1 SAR convention:
  `{COUNTRY}_{END_DATE}_{START_DATE}_S2_timeseries.tif` (e.g. `PT_20251007_20241015_S2_timeseries.tif`) with an instant hardlink alias to `{COUNTRY}_S2_timeseries.tif`.

---

### NASA Harvest Presto geospatial foundation model

**Presto** is a global geospatial transformer foundation model pre-trained on multi-sensor Earth observation time series:
* **Multi-sensor joint tokenization**: Simultaneously projects multi-temporal Sentinel-1 SAR (VV, VH) and Sentinel-2 optical reflectances (`B02`–`B12`) into a unified high-dimensional latent space.
* **Sinusoidal spatio-temporal embeddings**: Encodes continuous geographic coordinates (latitude, longitude) and seasonal acquisition months, conditioning model attention on geographic and bioclimatic context.
* **Latent representation extraction**: Outputs **128-dimensional multi-temporal token embeddings** from S1 SAR sequences and **128-dimensional embeddings** from S2 optical sequences, capturing complex non-linear phenological dynamics superior to standard handcrafted summary metrics.

---

### Object-based image analysis (OBIA) & segmentation deep dive

#### 1. Multimodal data footprint generation (`*_data_footprint.tif`)
Satellite tracks are captured along inclined orbits (descending ~ -12 deg, ascending ~ +12 deg), creating slanted swath edges and variable spatial bounds between radar (S1) and optical (S2) sensors.

In Stage 1, the pipeline computes a **binary multimodal data footprint raster**:
* **Physical sensor intersection**: Calculates the exact pixel-wise binary intersection of valid Sentinel-1 radar data, valid Sentinel-2 optical data, and regional NUTS boundaries (1 = valid data, 0 = NoData/outside swath).
* **Eliminates edge distortion**: Strictly bounds downstream sample generation, feature extraction, and tile inference to pixels where all radar and optical bands contain valid physical measurements.
* **Prevents no-data classification**: Ensures that partial-coverage edges outside the satellite swath are masked out, preventing false classifications along boundary margins.

#### 2. High-SNR SAR composite generation for OBIA segmentation
Single-date Synthetic Aperture Radar (SAR) imagery inherently suffers from multiplicative speckle noise caused by random interference of coherent waves. 

Before performing segmentation, Stage 2 generates a dedicated 1-channel high-SNR SAR raster to establish clean, stationary parcel boundaries:
* **Primary engine: Seasonal SAR composite (`_create_seasonal_sar_composite`)**:
  - Output: `{COUNTRY}_{ORBIT}_sar_seasonal_composite.tif`
  - Specifically isolates Sentinel-1 acquisitions during peak vegetative growth months (**April to August**: `apr`, `may`, `jun`, `jul`, `aug`).
  - During April–August, agricultural crop canopies reach maximum biomass and structural differentiation against bare soil, drainage ditches, and farm tracks. Field margins are geometrically sharpest during this period.
  - Multi-temporal averaging across these seasonal dates reduces radar speckle variance by `1/N`, providing an optimal input raster for edge-sensitive clustering.
* **Fallback / alternative engine: High-SNR summed composite (`_create_summed_composite`)**:
  - Output: `{COUNTRY}_{ORBIT}_summed_composite.tif`
  - Computes a multi-temporal summation across all valid radar bands (both VV and VH polarizations) spanning the full multi-month agricultural timeline.
  - Summing backscatter values in linear/log domain accumulates coherent surface backscatter while averaging out zero-mean incoherent phase noise, dramatically raising the Signal-to-Noise Ratio (SNR).
  - Automatically deployed whenever fewer than 4 valid acquisitions are available in the seasonal window, or in SAR-only baseline architectures (`1_classify_ann.py`, `1_classify_presto_sar.py`, `1_classify_ann_presto_hybrid.py`).
* **Optical fallback**: If no SAR raster is present, the pipeline falls back directly to multi-spectral optical composites.

#### 3. Segmentation modes in detail

```
+----------------------------------------------------------------------------------------------------+
|                                    OBIA SEGMENTATION METHODOLOGIES                                 |
+----------------------------------------------------------------------------------------------------+
|  [Mode 1: SLIC Superpixels]  --> Simple Linear Iterative Clustering on buffered 2048x2048 tiles     |
|                                  (64px halo + Union-Find tile seam stitching; UInt32 segment raster)|
|  [Mode 2: Meta AI SAM]       --> Vision Transformer (ViT) foundation model parcel delineation      |
|                                  (Bilateral edge preservation & distance transform hole-filling)    |
|  [Mode 3: LPIS Cadastre]     --> Ingestion of official European farmer declaration vectors (GPKG)  |
|                                  (High-speed R-Tree spatial indexing & 10m rasterization)          |
+----------------------------------------------------------------------------------------------------+
```

* **Mode 1: SLIC (Simple Linear Iterative Clustering - Recommended for general operational use)**:
  - Adapts k-means clustering in a 5-dimensional feature-spatial space combining the high-SNR radiometric composite (seasonal or summed composite) and spatial coordinates.
  - **Adaptive target parcel size**: Dynamically sets target parcel size based on country agronomic profile via `--slic_segment_ha` (default: 1.8 ha for Southern/Eastern Europe `PT`/`ES`/`IT`/`GR`/`PL`; 3.0 ha for Western Europe `NL`/`FR`/`DE`).
  - **High-elasticity boundary tracking**: Compactness parameter (`m = 0.05`) balances boundary adherence with regular parcel shape.
  - **Seamless buffered tiling**: Processes rasters in 2048 x 2048 pixel tiles with a **64-pixel overlapping halo buffer**.
  - **Disjoint Set Union (Union-Find) tile seam stitching (`_stitch_tile_seams`)**: Directly solves rectilinear grid edge artifacts by running an automated Union-Find boundary stitching pass across horizontal and vertical block seams (`x = 2048, 4096...` and `y = 2048, 4096...`). Adjacent segment labels on tile boundaries are seamlessly merged if they belong to the same continuous field parcel with consistent spectral properties.
  - **Region Adjacency Graph (RAG) spectral fusion (`--enable_slic_rag`)**: Optional experimental post-segmentation fusion pass merging adjacent superpixels whose spectral distance falls below `--slic_rag_thresh` (default: 0.02).
* **Mode 2: Meta AI SAM (Segment Anything Model - Deep foundation segmentation)**:
  - Utilizes Meta AI Vision Transformer backbones (`vit_h`, `vit_l`, `vit_b`) pre-trained on over 1 billion masks.
  - Automatically distributes a regular prompt grid across composite rasters.
  - Applies bilateral edge filtering and morphological distance transform hole-filling to close gaps along field boundaries.
* **Mode 3: LPIS (Land Parcel Identification System - Official cadastral parcels)**:
  - Ingests official national cadastral parcel boundary datasets (`.gpkg` or `.shp`) provided by Member State paying agencies (e.g. BRP in the Netherlands, ISIP in Portugal, ARiMR in Poland).
  - Employs **R-Tree spatial indexing** to assign every registered agricultural polygon a unique 32-bit `segment_id` rasterized at 10.0 m in `EPSG:3857` and strictly masks with the multimodal data footprint.

---


### Multimodal feature fusion architecture

```
+----------------------------------------------------------------------------------------------------+
|                             MULTIMODAL DUAL-TIER FEATURE FUSION PIPELINE                           |
+----------------------------------------------------------------------------------------------------+
|  [Tier 1: Physical Handcrafted Remote Sensing Features]                                             |
|  - S1 Temporal Statistics (VH, VV, VH/VV): Mean, Std, Min, Max, Dynamic Range (Max-Min), Slope     |
|  - S2 Multi-Spectral Reflectances: 14 DOYs * 9 Bands (B02-B12) = 126 spectral temporal features     |
|  - S2 Dynamic Indices: Multi-temporal NDVI, NDRE1, NDRE2, and NDWI canopy water trajectories       |
|                                                  |                                                 |
|  [Tier 2: NASA Harvest Presto Geospatial Foundation Model Latent Embeddings]                       |
|  - S1 SAR Transformer Latent Representation: 128-dimensional multi-temporal token embedding        |
|  - S2 Optical Transformer Latent Representation: 128-dimensional multi-temporal token embedding    |
|                                                  |                                                 |
|  [Concatenated High-Dimensional Feature Vector X]                                                  |
|  X = [ F_S1_stats || F_S2_spectral || E_Presto_S1 (128d) || E_Presto_S2 (128d) ]                  |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                      HYBRID MULTI-PARADIGM ENSEMBLE CLASSIFICATION SUITE                           |
+----------------------------------------------------------------------------------------------------+
|  [Branch A: PyTorch Deep MLP]                 |  [Branch B: XGBoost GBDT Engine]                   |
|  - 3 Dense Layers (512 -> 256 -> 128)         |  - 250 Gradient Boosted Decision Trees             |
|  - BatchNorm1d & Dropout (p=0.3)              |  - Histogram-based splitting (tree_method='hist')  |
|  - Class-Weighted Cross-Entropy Loss          |  - Subsample=0.8, Colsample_bytree=0.25            |
|  - Cosine Annealing Learning Rate Schedule    |  - Max depth = 6, learning rate = 0.08             |
|  -> Probability Vector P_MLP                  |  -> Probability Vector P_XGB                       |
+-----------------------------------------------+----------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|  [Soft-Voting Probability Blend]: P_final = 0.65 * P_MLP + 0.35 * P_XGB                            |
|  [Bayesian Prior Calibration]: P_calibrated adjusted by physical field size and training bias     |
+----------------------------------------------------------------------------------------------------+
```

#### 1. Tier 1: Handcrafted temporal, spectral & object texture features (100D feature suite)
* **Sentinel-1 SAR temporal moments**: For each polarization (VV, VH) and cross-ratio (VH/VV), the pipeline extracts statistical moments across the agricultural calendar:
  Mean, Standard Deviation (Std), Minimum (Min), Maximum (Max), and Dynamic Range (Max - Min).
  These metrics capture surface roughness, canopy closure speed, and abrupt drops in backscatter caused by harvesting.
* **Specialized SAR phenological indices (RVI & Dual-Pol vegetation metrics)**:
  `RVI = 4 * VH / (VV + VH)`
  Tracks canopy volume fraction and vegetative biomass accumulation independent of soil surface dielectric variations.
* **Seasonal SAR backscatter dynamics (Delta_VH)**:
  `Delta_VH = VH_summer - VH_spring`
  Provides discriminative power to differentiate winter cereals (established autumn canopy) from spring-sown crops (bare soil in early spring transitioning to lush vegetation by mid-summer).
* **Intra-object texture variance & dispersion (field-level std & IQR)**:
  Captures internal parcel spatial heterogeneity: `std` and `IQR` (interquartile range) for VH backscatter and NDVI reflectances. Distinguishes dense uniform field canopies (wheat, barley) from wide-row crops (maize, sugar beets) and structured woody vegetation (vineyards, orchards, olive groves).
* **Sentinel-2 multi-spectral trajectories**: Standardized 14-DOY reflectances across 9 spectral bands (14 x 9 = 126 features) plus time-series vegetation indices (NDVI, NDRE1, NDRE2, NDWI) tracking chlorophyll absorption, red-edge shift, and canopy moisture.

#### 2. Tier 2: NASA Harvest Presto 256-dimensional foundation embeddings
NASA Harvest Presto encodes raw multi-temporal sequences using self-attention transformer blocks, mapping seasonal dynamics into two 128-dimensional latent vectors:
* `E_S1` (128 dimensions): SAR dynamics.
* `E_S2` (128 dimensions): Optical dynamics.

#### 3. Unified concatenated feature vector
The complete feature vector combines domain-specific physical interpretability with deep self-supervised representation learning:
`X_fused = [ F_S1_stats_texture || F_S2_spectral_texture || E_Presto_S1 || E_Presto_S2 ]`

---

### Hybrid multi-paradigm machine learning ensemble

Combining Deep Neural Networks (PyTorch MLP) with Gradient Boosted Decision Trees (XGBoost) provides complementary strengths:
* **Deep Neural Networks (MLP)** excel at modeling smooth, continuous high-dimensional manifolds and projecting dense transformer embeddings.
* **Gradient Boosted Decision Trees (XGBoost)** excel at modeling discrete tabular decision thresholds, sharp spectral cutoffs, and step-function phenological transitions.
* Combining both architectures yields significantly lower generalization error and lower prediction variance than either model operating alone.

#### 1. PyTorch Deep MLP architecture & regularization
* **Network topology**: Input Layer (D dims) -> `Dense(512)` -> `BatchNorm1d` -> `ReLU` -> `Dropout(p=0.3)` -> `Dense(256)` -> `BatchNorm1d` -> `ReLU` -> `Dropout(p=0.3)` -> `Dense(128)` -> Output (K classes).
* **Square-root balanced class loss weights**: Corrects for class imbalance between dominant broadacre crops and minor specialty crops without gradient explosion:
  `w_c = sqrt(N / (K * N_c))`
  Smooths extreme sample disparities, ensuring rare specialty crops (tomatoes, orchards, pulses) generate sufficient gradient updates without destabilizing backpropagation on dominant classes.
* **Cosine Annealing learning rate schedule**: Smoothly decays learning rate across training epochs to avoid local minima.

#### 2. XGBoost GBDT engine
* Trained on 250 trees with histogram-based split binning (`tree_method='hist'`), maximum depth `max_depth=6`, learning rate 0.08, sample subsampling `subsample=0.8`, and column feature subsampling `colsample_bytree=0.25`.

#### 3. Soft-voting probability blend
Combines posterior probability distributions from both models:
`P_final = 0.65 * P_MLP + 0.35 * P_XGB`
Where 65% weight is assigned to the deep MLP and 35% weight to XGBoost.

#### 4. Dynamic geographic Bayesian prior calibration
Supervised models trained with weighted loss functions can introduce bias toward rare classes during spatial inference, while models trained on raw unweighted samples can overpredict dominant classes. The classifier implements a fully automated, dynamic two-stage Bayesian calibration:
1. **Physical geographic parcel area distribution**:
   Assigns expected physical crop parcel sizes from an agronomic multi-lingual taxonomy (supporting Polish, English, Portuguese, Dutch, French, German, Spanish, and Italian) across 5 agronomic scale tiers:
   * **Tier 1 (70 000 m² / 7.0 ha)**: Broadacre cereals & grasslands (maize, wheat, rapeseed, permanent pastures, meadows, fallow).
   * **Tier 2 (30 000 - 40 000 m² / 3-4 ha)**: Standard field & industrial crops (barley, rye, oats, triticale, sorghum, sugar beets, peas, lupins, olive groves, cotton).
   * **Tier 3 (20 000 - 25 000 m² / 2-2.5 ha)**: Orchards, vineyards & tubers (vineyards, fruit/nut orchards, citrus, potatoes).
   * **Tier 4 (10 000 - 15 000 m² / 1-1.5 ha)**: Medium & niche field crops (soybeans, buckwheat, field beans, vegetables & legumes, nurseries, flax, hemp).
   * **Tier 5 (2 000 - 5 000 m² / 0.2-0.5 ha)**: Intensive horticulture & specialty vegetables (tomatoes, onions, carrots, cucumbers, cabbage, strawberries, tobacco, asparagus, flower bulbs).
2. **Inversion of training loss bias**: Counteracts the artificial loss weighting applied during neural network training.
3. **Power-law exponential damping & clipping**: Damps adjustments by exponent gamma = 0.7 and bounds weights within `[0.01, 10.0]`.
4. **Strict zeroing of non-existent classes**: If a crop class has 0 samples in the orbit footprint, its weight is strictly set to 0.0, preventing ghost predictions for crops absent at that latitude.

---

## Complete toolbox guides and architecture

### Toolbox 1: Sentinel-1 SAR preprocessor

Located in `1_Sentinel-1_preprocessor/`, this toolbox transforms raw Copernicus Sentinel-1 Level-1 GRD products into multi-temporal, orthorectified, calibrated backscatter stacks. It supports **two data ingestion pathways** (`--source cdse` for Copernicus Data Space API downloads and `--source creodias` / `cog` for local `/eodata` archives) and **two SAR processing engines** (`--engine python` for 100% Pure Python processing without ESA SNAP, and `--engine snap` for legacy ESA SNAP GPT execution).

```
[Copernicus CDSE API (--source cdse) / CreoDIAS Local Archive (--source creodias | cog)]
                                          |
                                          v
    +--------------------------------------------------------------------------+
    |  Stage 1: Radiometric calibration, orthorectification & slice assembly   |
    |  - Pure Python (--engine python): 2D TNR, BNR, Sigma0 LUT spline,        |
    |    Range-Doppler Terrain Correction (Copernicus 30 m DEM) & stitching    |
    |  - ESA SNAP (--engine snap): POEORB, TNR, BNR, Sigma0 & SliceAssembly    |
    +-------------------------------------+------------------------------------+
                                          |
                                          v
    +--------------------------------------------------------------------------+
    |  Stage 2: Multi-temporal coregistration                                  |
    |  - Pure Python: Master 10.0m EPSG:3857 grid alignment manifest           |
    |  - ESA SNAP: CreateStack cross-correlation & Copernicus 30 m DEM TC      |
    +-------------------------------------+------------------------------------+
                                          |
                                          v
    +--------------------------------------------------------------------------+
    |  Stage 3: Multi-temporal despeckling, NUTS2 clipping & BigTIFF stacking  |
    |  - Vectorized Multi-Temporal Quegan Speckle Filter in linear power domain|
    |  - Optional 14-DOY temporal interpolation (--temporal_mode doy)          |
    |  - Linear-to-dB, NUTS2 clipping (10.0m EPSG:3857) & DEFLATE BigTIFF      |
    +--------------------------------------------------------------------------+
```

* **Master runner script**: `run_s1_preprocessor.py`
* **Internal modules (`modules/`)**:
  * `s1_python_processor.py`: Standalone Pure Python Sentinel-1 SAR calibration, 2D thermal & border noise removal, Range-Doppler terrain correction (Copernicus 30 m Global DEM), multi-temporal Quegan speckle filter, and BigTIFF stacker.
  * `s1_calibration_cog.py`: High-speed ingestion and calibration from local CreoDIAS Cloud-Optimized GeoTIFF (`.SAFE-COG`) repository.
  * `s1_calibration_creodias.py`: Fast ingestion and calibration from local CreoDIAS `.SAFE` archives (`/eodata/Sentinel-1/SAR/IW_GRDH_1S`).
  * `s1_calibration_cdse.py`: Automated scene discovery, download, and calibration from the Copernicus CDSE OData API.
  * `s1_coregistration.py`: Multi-temporal coregistration (Pure Python grid alignment & ESA SNAP `CreateStack`).
  * `s1_stack_clip.py`: Multi-temporal Quegan despeckling, Range-Doppler terrain correction using Copernicus 30 m DEM, GDAL BigTIFF stacking, and GISCO NUTS2 regional boundary clipping in `EPSG:3857`.

---

### Toolbox 2: Sentinel-2 optical preprocessor

Located in `1a_Sentinel-2_preprocessor/`, this toolbox creates cloud-free, regular 10-day synthetic optical time-series composites matching 1:1 with the Sentinel-1 SAR pixel grid using an optimized **country-level shared repository architecture** (`workingDirs/{COUNTRY}/S2/`).

```
[Copernicus CDSE API / CreoDIAS L2A Archive]
                     |
                     v
    +----------------------------------+
    |  Stage 1: Retrieval & masking    |  --> S2 L2A tile download/extraction to shared country pool,
    |  (SCL cloud & shadow filtering)  |      SCL filtering for clouds, shadows, snow, and invalid pixels
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  Stage 2: Synthetic DOY          |  --> Multi-temporal spline interpolation across 14 standardized DOYs
    |  time-series interpolation       |      (computed once per country tile, eliminating duplicate processing)
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  Stage 3: Per-orbit stacking     |  --> Sub-pixel warping to Sentinel-1 raster bounding box,
    |  & 126-band BigTIFF generation   |      126-band BigTIFF creation, and 6 pyramid overviews per orbit
    +----------------------------------+
```

* **Master runner script**: `run_s2_preprocessor.py`
* **Internal modules (`modules/`)**:
  * `s2_download_cdse.py`: Automated search, download, and SCL cloud masking from CDSE API directly to `workingDirs/{COUNTRY}/S2/`.
  * `s2_extract_creodias.py`: Direct extraction from local CreoDIAS archives (`Y:/Sentinel-2/MSI/L2A`).
  * `s2_time_series.py`: Pure Python multi-temporal interpolation across 14 standardized agricultural DOYs per country tile.
  * `s2_mosaic_stack.py`: Dynamic discovery of shared country tiles, sub-pixel grid alignment to Sentinel-1 SAR raster, 126-band BigTIFF creation, and pyramid overview generation.
  * `s2_pipeline.py`: Object-oriented pipeline orchestrator.

---

### Toolbox 3: Multimodal machine learning classifier

Located in `2_classifier/`, this toolbox implements object-based image analysis (OBIA) classification using multimodal fusion, deep learning foundation models, and Bayesian statistical calibration.

```
+----------------------------------------------------------------------------------------------------+
|                                      MULTIMODAL CLASSIFICATION SUITE                               |
+----------------------------------------------------------------------------------------------------+
|  [1] Footprint Gen.     --> Intersection of valid S1 SAR and S2 Optical coverage                   |
|  [2] Segmentation       --> Object delineation: SLIC superpixels / Meta AI SAM / LPIS parcels      |
|  [3] Sample Partition   --> Stratified split: 70% Learn (train) / 30% Control (validation)         |
|  [4] Feature Extraction --> S1 Stats + S2 Reflectances + Presto 128d S1/S2 Token Embeddings        |
|  [5] Model Training     --> Class-weighted PyTorch Deep MLP + XGBoost GBDT Fusion Ensemble         |
|  [6] Vectorized Infer.  --> High-performance tile inference with dynamic Bayesian calibration       |
|  [7] Cropland Masking   --> Agricultural mask application & non-cropland suppression               |
|  [8] Accuracy Reporting --> Out-of-bag validation metrics, F1-scores, and styled Excel report      |
+----------------------------------------------------------------------------------------------------+
```

* **Master runner script**: `run_classifier.py`
* **Internal engines (`modules/`)**:
  * `classifier_mlpxgb_presto.py`: Enhanced multimodal fusion ensemble combining NASA Harvest Presto transformer embeddings, PyTorch Deep MLP, XGBoost GBDT, vectorized chunk I/O, and spatial uncertainty estimation (`[S1 + S2] [SOTA]`). Aliases: `mlpxgb_presto_s1s2`, `s1s2`.
  * `multi_orbit_merger.py`: Nationwide multi-orbit confidence merger and seamless blender.
  * `presto_model.py`: Embedded NASA Harvest Presto transformer foundation architecture.
  *(Note: Legacy engines `classifier_otb.py` and `classifier_presto_s1.py` have been archived to `Archive_scripts/`)*

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
                     | Pixel = Orbit_1 if Conf_1 >= Conf_2 else Orbit_2         |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Post-Processing & Masking]                              |
                     | 1. Morphological sieve filtering (clump size < 50 px)    |
                     | 2. National agricultural cropland mask application       |
                     | 3. Aggregated national confusion matrix computation      |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [National Products: national_products/]                  |
                     | 1. {COUNTRY}_national_crop_map{suffix}.tif               |
                     | 2. {COUNTRY}_final_metrics{suffix}.xlsx                  |
                     +----------------------------+-----------------------------+
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

## Detailed step-by-step execution guide (A-to-Z manual)

This section provides an exhaustive, beginner-friendly, end-to-end operational guide for running each phase of the pipeline.

```
========================================================================================================
                                      END-TO-END PIPELINE ARCHITECTURE
========================================================================================================
 [PHASE 1: Sentinel-1 SAR]       [PHASE 2: Sentinel-2 Optical]       [INPUTS: samples.shp + NUTS2]
           |                                  |                                       |
           v                                  v                                       v
  S1 Dual-Pol Stack (VH/VV)        S2 126-Band Timeseries            Ground Truth & Administrative Mask
  (Float32, 10m, EPSG:3857)        (Int16, 14 DOYs, 9 bands)         (EPSG:3857, crop_id attributes)
           \                                  /                                       /
            \                                /                                       /
             +------------------------------+---------------------------------------+
                                            |
                                            v
                      [PHASE 3: Multimodal Machine Learning Classifier]
                      - Stage 1: Multimodal Data Footprint Generation
                      - Stage 2: High-SNR SAR Composites (Seasonal & Summed) & OBIA Segmentation
                      - Stage 3: Stratified Sample Partitioning (70% Train / 30% Test)
                      - Stage 4: Feature Extraction (S1 Stats + S2 + Presto 128d Embeddings)
                      - Stage 5: Model Training (PyTorch Deep MLP + XGBoost Ensemble)
                      - Stage 6: High-Speed Vectorized Tile Inference (Bayesian Priors)
                      - Stage 7: Cropland & Data Footprint Masking
                      - Stage 8: Validation Accuracy Assessment & Excel Report (.xlsx)
                                            |
                                            v
                         [PHASE 4: Nationwide Multi-Orbit Merger]
                         - Multi-orbit confidence-weighted blending in overlaps
                         - Morphological sieve post-processing
                         - Seamless national crop map & statistical aggregation
========================================================================================================
```

---

### Phase 1: Sentinel-1 SAR preprocessing (`run_s1_preprocessor.py`)

#### What happens behind the scenes:
1. **Stage 1 (Radiometric Calibration, Orthorectification & Slice Assembly)**:
   - Supports **two data ingestion pathways**: direct reading from local CreoDIAS storage (`--source creodias` / `--source cog`) or automated downloading from the Copernicus Data Space Ecosystem OData API (`--source cdse`).
   - Performs 2D thermal noise removal (TNR), GRD border noise removal (BNR), and radiometric calibration from raw digital numbers ($\text{DN}$) to physical linear radar cross-section ($\sigma^0$).
   - In **Pure Python mode (`--engine python`, default)**, orthorectifies each slice using exact Range-Doppler foreshortening equations with the **Copernicus 30 m Global DEM** directly onto a locked 10.0 m `EPSG:3857` grid and stitches daily slices into `slice_assembly/{YYYYMMDD}.tif` without requiring ESA SNAP.
2. **Stage 2 (Multi-Temporal Coregistration)**:
   - In **Pure Python mode (`--engine python`)**, aligns all daily orthorectified slices onto the master 10.0 m `EPSG:3857` grid and writes `wrapped/python_stack_manifest.json`.
   - In **ESA SNAP mode (`--engine snap`)**, runs SNAP `CreateStack` sub-pixel cross-correlation and Range-Doppler terrain correction with the Copernicus 30 m Global DEM (`BILINEAR_INTERPOLATION`).
3. **Stage 3 (Multi-Temporal Despeckling, Stacking & NUTS2 Clipping)**:
   - Applies the vectorized **Multi-Temporal Quegan Speckle Filter** in linear power domain (`--filter_window 7`), suppressing granular radar speckle while preserving 10 m field edges.
   - Optionally interpolates the SAR time series onto 14 standardized regional agricultural DOYs (`--temporal_mode doy`) matching Sentinel-2.
   - Converts backscatter to decibels (`dB = 10 * log10(Sigma0)`), clips to the country NUTS2 administrative boundary in `EPSG:3857` at 10.0 m resolution, and writes the final multi-temporal dual-polarization BigTIFF (`1_input_stacks/`) with 6 pyramid overview layers (`.ovr`).

#### How to execute:
```bash
# 1. Pure Python full automated run from local CreoDIAS archive (no ESA SNAP required):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PT --source creodias --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A --threads 4

# 2. Pure Python full automated run downloading from Copernicus CDSE API:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PT --source cdse --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A --threads 4

# 3. Process a single orbit track in Pure Python (e.g. Portugal orbit 52):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 --source creodias --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A

# 4. Exclude winter freeze period (December 1 to February 14) for Poland:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PL --source creodias --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A

# 5. Legacy ESA SNAP GPT engine execution:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 --source cog --engine snap -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A

# 6. Interactive wizard mode (guides you step-by-step):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py
```

#### Expected output artifact:
* `workingDirs/{COUNTRY}/orbit_{ORBIT}/1_input_stacks/{COUNTRY}_orbit_{ORBIT}_{DATE_RANGE}_VH_VV.tif`  
  *(Multi-temporal dual-pol Float32 BigTIFF, 10 m, EPSG:3857, with embedded `.ovr` pyramids).*

---

### Phase 2: Sentinel-2 optical preprocessing (`run_s2_preprocessor.py`)

#### What happens behind the scenes:
1. **Stage 1 (Granule Retrieval & SCL Cloud Masking)**:
   - Queries Copernicus CDSE API (or CreoDIAS local archive) for Sentinel-2 L2A BOA surface reflectance tiles intersecting the country territory.
   - Extracts 9 spectral bands (`B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B8A`, `B11`, `B12`) and the Scene Classification Layer (`SCL`).
   - Strictly masks out clouds, thin cirrus, cloud shadows, snow, and defective pixels, saving clean GeoTIFFs to `_temp_processing/{TILE}_tif/`.
2. **Stage 2 (14-DOY Synthetic Time-Series Interpolation & Whittaker Smoothing)**:
   - Evaluates the temporal distribution of cloud-free observations across 14 standardized agricultural reference dates tailored to regional European profiles (Mediterranean DOY 45–280, Continental DOY 80–287, Northern DOY 105–300).
   - Computes multi-core linear interpolation between nearest forward and backward clear observations for all 9 bands.
   - Applies an optional vectorized **Whittaker penalized least-squares smoother** (`--enable_whittaker`, `lambda = 15.0`) across the temporal DOY profile, suppressing cloud spikes and shadow artifacts to ensure continuous phenological trajectories.
   - Saves synthetic composite tiles to `workingDirs/{COUNTRY}/S2/{TILE}/_synthetic_s2/day{DOY}_{YEAR}/`.
3. **Stage 3 (SAR Grid Matching, ZSTD Stacking & Timestamped Naming)**:
   - Reads the Sentinel-1 SAR stack bounding box and spatial resolution as the master geometric reference.
   - Performs sub-pixel resampling (`dX = 0.000 m, dY = 0.000 m`) ensuring 100% pixel-for-pixel alignment between optical reflectances and radar backscatter.
   - Compiles all 14 DOYs x 9 spectral bands into a unified 126-band BigTIFF using **high-speed ZSTD level 3 compression (`PREDICTOR=2`)** with 6 pyramid overview layers.
   - Generates standardized temporal filenames matching the SAR standard: `{COUNTRY}_{END_DATE}_{START_DATE}_S2_timeseries.tif` (with instant backward-compatible alias).

#### How to execute:
```powershell
# Recommended: Full automated run for Portugal using official CDSE API and Whittaker smoother:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT --source cdse -s 2024-10-15 -e 2025-09-15 --enable_whittaker --stage A --threads 8

# Process a single orbit track:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --track PT/orbit_52 --source cdse -s 2024-10-15 -e 2025-09-15 --enable_whittaker --stage A --threads 8

# Interactive wizard mode (guides you step-by-step):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py
```

#### Expected output artifact:
* `workingDirs/{COUNTRY}/S2/{COUNTRY}_{END_DATE}_{START_DATE}_S2_timeseries.tif`  
  *(126-band ZSTD-compressed optical BigTIFF, 10 m, EPSG:3857, matched 1:1 with SAR grid).*

---

### Phase 3: Multimodal machine learning classifier (`run_classifier.py`)

#### What happens in each classification stage (stages 1 to 8):

| Stage | Name | Action & scientific description | Generated output file |
| :--- | :--- | :--- | :--- |
| **1** | **Multimodal Data Footprint** | Computes binary spatial intersection of valid S1, S2, and NUTS2 regional boundaries (1 = valid, 0 = NoData). | `0_segmentation/*_data_footprint.tif` |
| **2** | **High-SNR SAR Composites & OBIA** | Generates high-SNR SAR composite (**seasonal April–August composite** `*_sar_seasonal_composite.tif` or **full-season summed composite** `*_summed_composite.tif`) to suppress speckle. Delineates agricultural parcels using SLIC superpixels (with 64px halo buffer & Union-Find seam stitching), Meta AI SAM, or official LPIS cadastre. | `0_segmentation/*_sar_seasonal_composite.tif`<br>`0_segmentation/*_summed_composite.tif`<br>`0_segmentation/*_segmentation_{MODE}.tif` |
| **3** | **Stratified Sample Split** | Spatially intersects `samples.shp` with segment parcels. Partitions points into **70% training (`*_learn_{MODE}.shp`)** and **30% independent validation (`*_control_{MODE}.shp`)** with balanced class stratification. | `1_samples_and_features/*_learn_{MODE}.shp`<br>`1_samples_and_features/*_control_{MODE}.shp` |
| **4** | **Multimodal Feature Extraction** | Extracts linear-power SAR temporal backscatter statistics, 14-DOY optical reflectances (`B02`–`B12`), physical Red-Edge and SAR indices (**RVI, CR, Delta_VH, NDVI, NDRE, NDWI, EVI, texture variance**), and **128d Joint, S1, and S2 Presto token embeddings** (~450+ features). | `1_samples_and_features/*_mlpxgb_presto_learn_features_{MODE}.csv` |
| **5** | **Model Training** | Fits class-weighted PyTorch Deep MLP (`BatchNorm1d`, `Dropout`, Cosine Annealing) and XGBoost GBDT (250 trees, histogram split). Combines models via soft-voting ensemble blend (65% MLP + 35% XGBoost). | `2_models/*_mlpxgb_presto_model_{MODE}.pkl` |
| **6** | **Asynchronous Vectorized Tile Inference** | Runs an **asynchronous CPU producer thread** prefetching 2048 x 2048 blocks and computing vectorized `np.bincount` zonal aggregation, while the **GPU consumer** runs batched Presto embeddings and MLP forward passes in parallel. Applies dynamic Bayesian prior calibration and outputs classification, confidence, Shannon entropy uncertainty, and prediction margin maps. | `3_maps/*_classified_mlpxgb_presto_{MODE}.tif`<br>`3_maps/*_confidence_mlpxgb_presto_{MODE}.tif`<br>`3_maps/*_entropy_mlpxgb_presto_{MODE}.tif`<br>`3_maps/*_margin_mlpxgb_presto_{MODE}.tif` |
| **7** | **Cropland Masking & Sieve Filter** | Applies high-resolution agricultural mask (`AgriMasks/{COUNTRY}/`) and morphological sieve filtering (<10 px), suppressing non-agricultural surfaces (forests, urban, water bodies). | `3_maps/*_classified_masked_mlpxgb_presto_{MODE}.tif`<br>`3_maps/*_confidence_masked_mlpxgb_presto_{MODE}.tif`<br>`3_maps/*_entropy_masked_mlpxgb_presto_{MODE}.tif`<br>`3_maps/*_margin_masked_mlpxgb_presto_{MODE}.tif` |
| **8** | **Accuracy Assessment** | Evaluates predictions against the independent 30% control dataset. Computes full confusion matrix, Overall Accuracy (OA), Cohen's Kappa, User's Accuracy (Precision), Producer's Accuracy (Recall), and F1-scores. | `4_reports/report_*_mlpxgb_presto_{MODE}.xlsx` |

#### How to execute:
```powershell
# Recommended SOTA: Multimodal Deep MLP + XGBoost + Presto with SLIC Superpixels (forcing fresh retraining with --overwrite):
python 2_classifier/run_classifier.py --country PT --classifier mlpxgb_presto --seg_mode slic --overwrite --stage A

# Run single orbit track:
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode slic --overwrite --stage A

# Run specific single stages (e.g. Stage 4 Feature Extraction or Stage 6 Inference):
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 4
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 6

# Run with official cadastral LPIS parcel vectors:
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode lpis --lpis_vector auxiliary_files/raster_files/AgriMasks/PT/parcels.gpkg --stage A

# Run with Meta AI SAM vision foundation model segmentation:
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode sam --stage A

# Run single-radar S1-only Presto ANN classifier:
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier presto_s1 --seg_mode slic --stage A

# Interactive setup wizard (guides you step-by-step):
python 2_classifier/run_classifier.py
```

---

### Phase 4: Nationwide multi-orbit merge (`run_merge.py`)

#### What happens behind the scenes:
1. Discovers all processed single-orbit classification maps and confidence rasters for the target country.
2. Identifies overlapping swath zones between adjacent satellite orbits.
3. In overlapping pixels, applies **confidence-weighted blending**:
   For each overlapping pixel, the classification from the orbit with higher prediction confidence is selected.
4. Applies morphological sieve filtering (removing isolated pixel noise clumps < 50 pixels) and enforces the nationwide agricultural mask.
5. Aggregates confusion matrices from all orbits into a consolidated national statistical performance report.

#### How to execute:
```powershell
# Merge SLIC superpixel classification maps across all orbits for Portugal (with SOTA ensemble):
python 2_classifier/run_merge.py --country PT --classifier mlpxgb_presto --seg_mode slic

# Merge LPIS cadastral classification maps across all orbits for the Netherlands:
python 2_classifier/run_merge.py --country NL --classifier mlpxgb_presto --seg_mode lpis

# Merge Meta AI SAM classifications using confidence blending:
python 2_classifier/run_merge.py --country PT --classifier mlpxgb_presto --seg_mode sam --method confidence
```

#### Expected output artifacts:
* `workingDirs/{COUNTRY}/national_products/{COUNTRY}_national_crop_map{suffix}.tif`
* `workingDirs/{COUNTRY}/national_products/{COUNTRY}_final_metrics{suffix}.xlsx`

---

### Phase 5: How to visualize and validate maps in QGIS & ArcGIS Pro

#### 1. Loading and styling the Crop Map in QGIS:
1. Open **QGIS** and click **Layer -> Add Layer -> Add Raster Layer...**
2. Select `workingDirs/{COUNTRY}/national_products/{COUNTRY}_national_crop_map_mlpxgb_presto_slic.tif` (or per-orbit `*_classified_masked_slic.tif`).
3. Right-click the loaded layer in the Layers panel -> **Properties** -> select **Symbology** tab.
4. Set **Render type** to **Paletted / Unique values**.
5. Click **Classify** at the bottom:
   - Each numerical `crop_id` (e.g. `11` = Winter Wheat, `12` = Maize, `1430` = Rapeseed) will receive a unique color.
6. Click **OK** to render the interactive crop classification map.

#### 2. Inspecting Classification Confidence:
1. Add `*_confidence_masked_slic.tif` as a raster layer in QGIS.
2. In **Properties -> Symbology**, set **Render type** to **Singleband pseudocolor**.
3. Choose a color ramp (e.g. `Viridis` or `RdYlGn`), with minimum value `0.0` (red / low confidence) to maximum value `1.0` (green / high confidence).
4. Overlay with transparency over the classified map to inspect model certainty across field interiors and boundaries.

#### 3. Analyzing the Excel Accuracy Report:
1. Open `report_{COUNTRY}_{ORBIT}_metrics_slic.xlsx` or national `*_final_metrics_*.xlsx` in Microsoft Excel or LibreOffice Calc.
2. Inspect the **Summary Sheet**:
   - **Overall Accuracy (OA)**: Percentage of correct crop identifications (target > 85% to 90%).
   - **Cohen's Kappa**: Chance-corrected statistical agreement metric.
3. Inspect the **Confusion Matrix Sheet**:
   - Rows represent ground truth; columns represent model predictions.
   - Off-diagonal values reveal specific confusion pairs (e.g., Winter Wheat vs Winter Barley).
4. Inspect **Per-Class Metrics**:
   - **User's Accuracy (Precision)**: Reliability of predicted crop pixels.
   - **Producer's Accuracy (Recall)**: Completeness of ground truth detection.
   - **F1-Score**: Harmonic mean of Precision and Recall.

---

## Cloud-native geospatial engineering & memory architecture

To process nationwide multi-temporal satellite rasters exceeding 100 GB per orbit on standard institutional workstations and cloud nodes, the pipeline incorporates advanced high-performance computing (HPC) patterns:

```
+----------------------------------------------------------------------------------------------------+
|                               CLOUD-NATIVE HIGH-PERFORMANCE COMPUTING STACK                        |
+----------------------------------------------------------------------------------------------------+
|  [Layer 1: Block-Level I/O]    --> 2048x2048 contiguous C-order block reading (99.83% syscall cut)|
|  [Layer 2: Fast Reduction]     --> Vectorized np.bincount zonal aggregation in compiled C/NumPy    |
|  [Layer 3: Batched Inference]  --> Presto Transformer + PyTorch MLP GPU/CPU vector evaluation      |
|  [Layer 4: O(1) LUT Indexing]  --> Direct array index raster reconstruction (2 ms per tile)        |
|  [Layer 5: Thread Isolation]   --> OMP/MKL thread pinning preventing kernel context thrashing      |
+----------------------------------------------------------------------------------------------------+
```

### 1. Vectorized block I/O vs random access
* **Legacy GIS approaches**: Iterate through polygon geometries or sample points one-by-one, issuing millions of micro-seek system calls (`read()` / `fseek()`) to the filesystem, causing severe disk thrashing on network/HDD storage.
* **v3.0 Architecture**: Divides the raster into regular 2048 x 2048 processing blocks. All 170+ radar and optical bands are read sequentially in a single contiguous memory operation, eliminating 99.83% of filesystem I/O system calls.

### 2. O(1) array indexing lookup table (LUT) reconstruction
* Rather than performing spatial raster masking for every individual polygon, predictions and confidence values for all segment objects in a tile are stored in a 1D lookup array:
  `Raster_classified = LUT_pred[Segment_tile]`
* **Performance**: Reconstructs a 2048 x 2048 classified tile in **~2 ms**, compared to 2.5 to 4 minutes in legacy dictionary iteration.

### 3. OpenMP & BLAS multi-threading isolation
When running Python `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`, internal scientific libraries (OpenMP, Intel MKL, OpenBLAS) default to spawning multiple threads per process. On a 16-core CPU running 8 worker processes, this spawns 8 x 16 = 128 threads, causing severe CPU cache thrashing and kernel lock contention.

The pipeline strictly isolates worker environments:
```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```
Ensuring 100% linear multi-core scaling without thread over-subscription.

---

## High-performance vectorized inference architecture

In version 3.0, Stage 6 (`stage_6_classify_vector`) has been re-architected with **vectorized block-level I/O and O(1) lookup table (LUT) raster reconstruction**:

| Metric / feature | Legacy approach | Vectorized v3.0 architecture | Improvement |
| :--- | :---: | :---: | :---: |
| **Disk I/O requests per orbit** | > 50 000 000 random reads | **~85 000 sequential block reads** | 🟢 **99.83% I/O reduction** |
| **Zonal statistics aggregation** | Python per-object looping | Vectorized `np.bincount` in NumPy/C | 🟢 **> 1000x faster** |
| **Tile raster reconstruction** | Iterative dictionary mask assignment | Direct array indexing `lut_pred[sub_seg]` | 🟢 **2 ms per tile** |
| **Average time per tile (2048 x 2048 px)** | 2.5 to 4.0 minutes | **4 to 7 seconds** | 🚀 **~35x speedup** |
| **Total Stage 6 runtime (full country orbit)** | **24 to 48 hours** | **30 to 50 minutes** | ⚡ **~40x to 60x faster** |

### Asynchronous GPU-CPU producer-consumer pipeline
To maximize GPU tensor core utilization across massive multimodal satellite BigTIFFs, spatial tile inference employs a background prefetching producer-consumer threading model:
1. **Background CPU producer**: Runs in a dedicated background worker (`threading.Thread`), sequentially prefetching the next 2048 x 2048 tile block from disk, computing vectorized `np.bincount` zonal aggregations, and packing candidate tensors.
2. **Foreground GPU consumer**: Concurrently runs batched Presto embedding extraction and PyTorch Deep MLP forward passes on the current GPU tensor buffer (`cuda`).
3. **Latency masking**: Disk read and feature assembly latency is completely hidden behind GPU tensor execution, cutting tile inference bottlenecks and achieving near 100% hardware saturation.

---

## Directory and file lineage structure

```text
AIML_CropMapper_Cloud/
├── 1_Sentinel-1_preprocessor/           # Sentinel-1 SAR pipeline
│   ├── run_s1_preprocessor.py           # Unified S1 CLI & interactive wizard
│   ├── config_s1.json                   # Active S1 config (CDSE credentials, SNAP paths)
│   ├── config_s1.example.json           # Template S1 config
│   ├── modules/                         # Core S1 processing modules
│   │   ├── s1_python_processor.py       # Pure Python SAR engine (no ESA SNAP required)
│   │   ├── s1_calibration_cog.py        # CreoDIAS local COG calibration
│   │   ├── s1_calibration_creodias.py   # CreoDIAS local SAFE calibration
│   │   ├── s1_calibration_cdse.py       # CDSE API GRDH calibration
│   │   ├── s1_coregistration.py         # Multi-temporal coregistration (Python & SNAP)
│   │   └── s1_stack_clip.py             # Quegan despeckling, stacking & NUTS2 clipping
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

### Complete artifacts & output file lineage catalog

The table below catalogs every intermediate and final output artifact generated across all pipeline stages:

| Stage / phase | Directory path | File name pattern | Data type / bands | Purpose & scientific description |
| :--- | :--- | :--- | :---: | :--- |
| **S1 Stage 1** | `_temp_processing/1_calibrated/` | `S1*_Sigma0_{POL}.tif` | `Float32`, 1 band | Single-slice radiometric Sigma0 radar backscatter. |
| **S1 Stage 2** | `_temp_processing/2_wrapped_stack/`| `wrapped_stack.dim / .data` | ENVI/BEAM, 2N b. | Multi-temporal coregistered SAR stack across all DOYs. |
| **S1 Stage 3** | `1_input_stacks/` | `{COUNTRY}_orbit_{ORBIT}_*_VH_VV.tif` | `Float32`, 2N b. | **Final SAR BigTIFF**: Terrain corrected, 10m, EPSG:3857 with `.ovr`. |
| **S2 Stage 1** | `_temp_processing/{TILE}_tif/` | `*_{BAND}_20m.tif` | `UInt16`, 1 band | Masked surface reflectance per scene (`B02`–`B12`, `SCL`). |
| **S2 Stage 2** | `_temp_processing/{TILE}/_synthetic_s2/` | `day{DOY}_{YEAR}/{BAND}.tif` | `UInt16`, 1 band | 14-DOY synthetic cloud-free interpolated spectral bands. |
| **S2 Stage 3** | `1_input_stacks/` | `{COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif` | `Int16`, 126 bands | **Final Optical BigTIFF**: 14 dates x 9 bands, 10m grid matched to S1. |
| **Classif. 1** | `2_classification/0_segmentation/`| `{COUNTRY}_{ORBIT}_data_footprint.tif` | `Byte`, 1 band | **Binary Valid Footprint**: Spatial intersection of valid S1, S2, and NUTS2. |
| **Classif. 2** | `2_classification/0_segmentation/`| `{COUNTRY}_{ORBIT}_sar_seasonal_composite.tif` | `Float32`, 1 band | **Primary Seasonal SAR Composite**: April–August multi-temporal speckle-reduced backscatter. |
| **Classif. 2** | `2_classification/0_segmentation/`| `{COUNTRY}_{ORBIT}_summed_composite.tif` | `Float32`, 1 band | **Summed Composite**: Multi-temporal sum across all SAR bands (high-SNR fallback). |
| **Classif. 2** | `2_classification/0_segmentation/`| `{COUNTRY}_{ORBIT}_segmentation_{MODE}.tif` | `UInt32`, 1 band | **OBIA Segment Objects**: Integer raster with unique parcel identifiers. |
| **Classif. 3** | `2_classification/1_samples_and_features/` | `{COUNTRY}_{ORBIT}_learn_{MODE}.shp` | Vector Points | Stratified 70% training points with spatial attributes. |
| **Classif. 3** | `2_classification/1_samples_and_features/` | `{COUNTRY}_{ORBIT}_control_{MODE}.shp` | Vector Points | Stratified 30% independent validation points. |
| **Classif. 4** | `2_classification/1_samples_and_features/` | `{COUNTRY}_{ORBIT}_mlpxgb_presto_learn_features_{MODE}.csv` | Tabular CSV | Extracted S1 statistics, S2 reflectances, physical indices, and 128d Presto tokens (~450+ feats). |
| **Classif. 5** | `2_classification/2_models/` | `{COUNTRY}_{ORBIT}_mlpxgb_presto_model_{MODE}.pkl` | Pickle Checkpoint | Serialized PyTorch Deep MLP weights + XGBoost GBDT trees + label encoder + scaler. |
| **Classif. 6** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_classified_mlpxgb_presto_{MODE}.tif` | `UInt16`, 1 band | Raw pixel crop classification raster (pixel value = numeric `crop_id`). |
| **Classif. 6** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_confidence_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 band | Softmax ensemble probability confidence map (`0.0` to `1.0`). |
| **Classif. 6** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_entropy_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 band | Normalized Shannon entropy spatial uncertainty map (`0.0` = certain, `1.0` = high confusion). |
| **Classif. 6** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_margin_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 band | Prediction margin map (`P_first - P_second`, margin between top-2 crop candidates). |
| **Classif. 7** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_classified_masked_mlpxgb_presto_{MODE}.tif` | `UInt16`, 1 band | **Final Track Crop Map**: Cropland-masked & footprint-clipped GeoTIFF with sieve filter. |
| **Classif. 7** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_confidence_masked_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 band | **Final Track Confidence Map**: Cropland-masked confidence GeoTIFF. |
| **Classif. 7** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_entropy_masked_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 band | **Final Track Entropy Map**: Cropland-masked spatial entropy uncertainty GeoTIFF. |
| **Classif. 7** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_margin_masked_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 band | **Final Track Margin Map**: Cropland-masked prediction margin GeoTIFF. |
| **Classif. 8** | `2_classification/4_reports/` | `report_{COUNTRY}_{ORBIT}_mlpxgb_presto_{MODE}.xlsx` | Styled Excel | Detailed validation report: Overall Accuracy, Cohen's Kappa, Confusion Matrix, F1-scores. |
| **Merge 4** | `national_products/` | `{COUNTRY}_national_crop_map{suffix}.tif` | `Int32`, 1 band | **Seamless National Crop Map**: Confidence-blended multi-orbit BigTIFF. |
| **Merge 4** | `national_products/` | `{COUNTRY}_final_metrics{suffix}.xlsx` | Styled Excel | Aggregated nationwide statistical validation report. |

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

## Third-party software, foundation models, and data licenses

This project builds upon, integrates, and acknowledges key open-source libraries, pretrained foundation models, scientific processing engines, and public Earth observation datasets. All components are used in full compliance with their respective open licenses:

### AI foundation models and deep learning
* **NASA Harvest Presto Geospatial Foundation Model** ([MIT License](https://github.com/nasaharvest/presto/blob/main/LICENSE)): Multi-temporal transformer embeddings across Sentinel-1 SAR and Sentinel-2 optical time series ([Tseng et al., 2023](https://doi.org/10.48550/arXiv.2304.14065)).
* **Meta AI Segment Anything Model (SAM) & GeoSAM** ([Apache 2.0 License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)): Zero-shot deep vision segmentation foundation model for agricultural field and parcel boundary delineation ([Kirillov et al., 2023](https://doi.org/10.48550/arXiv.2304.02643)).
* **PyTorch & Torchvision** ([BSD 3-Clause License](https://github.com/pytorch/pytorch/blob/main/LICENSE)): Deep learning tensor computation framework powering the Deep MLP classifier and transformer model inference ([Paszke et al., 2019](https://pytorch.org)).
* **XGBoost (Extreme Gradient Boosting)** ([Apache 2.0 License](https://github.com/dmlc/xgboost/blob/master/LICENSE)): High-performance gradient boosted decision trees for tabular multimodal feature classification ([Chen & Guestrin, 2016](https://doi.org/10.1145/2939672.2939785)).

### Geospatial, scientific, and image processing libraries
* **GDAL / OGR / OSR** ([MIT / X11-style License](https://gdal.org/license.html)): High-throughput geospatial raster/vector I/O, BigTIFF handling, warping, and pyramid overviews.
* **scikit-learn** ([BSD 3-Clause License](https://scikit-learn.org/stable/about.html#citing-scikit-learn)): Robust feature scaling, stratified sampling, and classification evaluation metrics ([Pedregosa et al., 2011](https://scikit-learn.org)).
* **scikit-image** ([BSD 3-Clause License](https://scikit-image.org/docs/stable/license.html)): SLIC edge-constrained superpixel segmentation and image morphology ([van der Walt et al., 2014](https://doi.org/10.7717/peerj.453)).
* **SciPy** ([BSD 3-Clause License](https://scipy.org/license/)): Distance transforms, morphological filters, and spatial connected component analysis ([Virtanen et al., 2020](https://doi.org/10.1038/s41592-019-0686-2)).
* **NumPy** ([BSD 3-Clause License](https://numpy.org/license/)): Vectorized multi-dimensional array operations and time-series reductions ([Harris et al., 2020](https://doi.org/10.1038/s41586-020-2649-2)).
* **pandas** ([BSD 3-Clause License](https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html#license)): Tabular data structures and sample attribute manipulation.
* **GeoPandas & Shapely** ([BSD 3-Clause License](https://geopandas.org/en/stable/about.html)): Spatial vector analysis, NUTS administrative subsetting, and polygon geometry operations.
* **Rasterio & RioXarray** ([BSD 3-Clause / Apache 2.0](https://rasterio.readthedocs.io)): Windowed raster streaming and coordinate reference system transformations.
* **openpyxl** ([MIT License](https://openpyxl.readthedocs.io)): Generation of multi-tab Microsoft Excel (`.xlsx`) validation reports.
* **joblib** ([BSD 3-Clause License](https://joblib.readthedocs.io)): High-efficiency machine learning model serialization.

### External software and processing engines
* **ESA SNAP (Sentinel Application Platform) & Graph Processing Tool (GPT)** ([GPLv3 / ESA SNAP License](https://step.esa.int/main/toolboxes/snap/)): Sentinel-1 SAR radiometric calibration, orbit application, thermal noise removal, and Range-Doppler terrain correction. *(Invoked as an external command-line tool; not bundled with this repository)*.

### Satellite data, boundaries, and cadastral reference layers
* **Copernicus Sentinel-1 SAR & Sentinel-2 Optical Data** ([Copernicus Open Access Data Policy](https://dataspace.copernicus.eu/terms-and-conditions)): Free, full, and open Earth observation imagery provided by the European Union and the European Space Agency (ESA).
* **Eurostat GISCO Administrative Boundaries (NUTS 2024)** ([Eurostat Copyright Notice](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units)): European administrative boundary shapefiles provided under open data terms with attribution.
* **Copernicus Land Monitoring Service (CLMS)** ([Copernicus Land Monitoring Policy](https://land.copernicus.eu)): High Resolution Layer crop type and agricultural mask datasets provided by the European Environment Agency (EEA) and the European Union.
* **Land Parcel Identification System (LPIS) / GSAA Cadastral Data**: National agricultural parcel datasets provided by EU Member State Paying Agencies and Ministries of Agriculture under open government data licenses compliant with the EU Open Data Directive (Directive (EU) 2019/1024) and INSPIRE Directive (2007/2/EC).

---

## Authors and citation

If you use this software in your research or statistical production pipelines, please cite:

**APA format:**
> Slesinski, P., Kotulak, N., Roos, M., Mróz, M., Mleczko, M., Gabriel, C., Hofer, N., Belton, S., Logakrishnan, M., Kästenbauer, M., Martins, C., Pallister, I. L. M., Gonçalves, I. (2025). *Sentinel-1 & Sentinel-2 OBIA crop type mapping pipeline (v3.0)*. [AIML4OS – One Stop Shop for Artificial Intelligence in Official Statistics](https://cros.ec.europa.eu/dashboard/aiml4os), Work Package 7, European Commission / Eurostat. Available at: https://github.com/AIML4OS/WP7-Crop-type-mapping

**BibTeX:**
```bibtex
@software{slesinski2025cropmapper,
  author       = {Slesinski, Przemyslaw and Kotulak, Natalia and Roos, Marko and Mróz, Marek and Mleczko, Magdalena and Gabriel, Cristina and Hofer, Nina and Belton, Sam and Logakrishnan, Mohana and Kästenbauer, Mathias and Martins, Carla and Pallister, Ivana I. L. M. and Gonçalves, Isabel},
  title        = {Sentinel-1 & Sentinel-2 OBIA crop type mapping pipeline},
  version      = {3.0.0},
  year         = {2025},
  url          = {https://github.com/AIML4OS/WP7-Crop-type-mapping},
  organization = {AIML4OS – One Stop Shop for Artificial Intelligence in Official Statistics, Eurostat, European Commission}
}
```

---

## License

This project is licensed under the Apache License 2.0. Developed under the European Statistical System (ESS) AIML4OS initiative funded by Eurostat and the European Commission.
