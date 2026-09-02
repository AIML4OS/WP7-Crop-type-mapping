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

### Sentinel-1 SAR radar remote sensing physics & polarimetry

Sentinel-1 operates at C-band Synthetic Aperture Radar frequency ($f = 5.405\text{ GHz}$, wavelength $\lambda \approx 5.55\text{ cm}$). Microwaves penetrate cloud cover, precipitation, and solar illumination variations, directly measuring canopy volumetric structure, dielectric constant ($\varepsilon$), and soil moisture.

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

#### 1. Fundamental radar equation for distributed agricultural targets
The received radar backscatter power $P_r$ from an agricultural field of ground resolution area $A_{\text{ground}}$ is governed by:

$$P_r = \frac{P_t G^2 \lambda^2 \sigma^0 A_{\text{ground}}}{(4\pi)^3 R^4}$$

Where:
* $P_t$ is transmitter peak power, $G$ is antenna gain, and $R$ is slant range distance.
* $\sigma^0$ is the normalized radar cross section (radar backscatter coefficient):
  $$\sigma^0_i = \frac{|DN_i|^2}{A_i^2}$$
  Where $DN_i$ is digital number amplitude and $A_i$ is the calibration look-up table (LUT) gain.

#### 2. Radiometric terrain correction (RTC) & decibel transformation
To eliminate topographic illumination distortions on sloped terrain, backscatter is orthorectified using Copernicus 30 m DEM and converted to decibels:

$$\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\sigma^0)$$

#### 3. Agricultural polarimetric indices
* **Radar Vegetation Index (RVI)**: Measures canopy volume scattering relative to total backscatter:
  $$\text{RVI} = \frac{4 \cdot \sigma^0_{VH}}{\sigma^0_{VV} + \sigma^0_{VH}}$$
* **Polarimetric Cross-Ratio (CR)**:
  $$\text{CR} = \frac{\sigma^0_{VH}}{\sigma^0_{VV}} \implies \text{CR}_{\text{dB}} = \sigma^0_{VH,\text{dB}} - \sigma^0_{VV,\text{dB}}$$

---

### Sentinel-2 multispectral optical remote sensing & vegetation indices

Captures Bottom-Of-Atmosphere (BOA) surface reflectances across 9 spectral bands at 10 m and 20 m spatial resolutions (resampled to 10.0 m in `EPSG:3857`):

$$\rho_{\text{BOA}}(\lambda) = \frac{\pi \cdot (L_{\text{TOA}}(\lambda) - L_{\text{path}}(\lambda))}{\tau_v(\lambda) \cdot [E_0(\lambda) \cdot \cos \theta_s \cdot \tau_s(\lambda) + E_{\text{down}}(\lambda)]}$$

#### 1. Spectral band sensitivities
* **Visible bands (`B02` Blue 490 nm, `B03` Green 560 nm, `B04` Red 665 nm)**: Sensitive to photosynthetic chlorophyll $a$ and $b$ absorption.
* **RedEdge bands (`B05` 705 nm, `B06` 740 nm, `B07` 783 nm)**: Capture the steep reflectance transition edge; highly sensitive to canopy nitrogen, leaf chlorophyll concentration, and early senescence.
* **Narrow NIR (`B8A` 865 nm)**: Measures internal leaf mesophyll cellular scattering, avoiding atmospheric water vapor absorption in broad `B08`.
* **Shortwave Infrared (`B11` 1610 nm, `B12` 2190 nm)**: Sensitive to foliar water content and dry cellulose matter.

#### 2. Narrow-band agricultural indices
* **Normalized Difference Vegetation Index (NDVI)**:
  $$\text{NDVI} = \frac{\text{B8A} - \text{B04}}{\text{B8A} + \text{B04}}$$
* **Red-Edge Chlorophyll Index (NDRE1 & NDRE2)**:
  $$\text{NDRE1} = \frac{\text{B06} - \text{B05}}{\text{B06} + \text{B05}}, \quad \text{NDRE2} = \frac{\text{B07} - \text{B05}}{\text{B07} + \text{B05}}$$
* **Normalized Difference Water Index (NDWI / NDII)**:
  $$\text{NDWI} = \frac{\text{B8A} - \text{B11}}{\text{B8A} + \text{B11}}$$

#### 3. Standardized 14-DOY agricultural nodes
Temporal observations are interpolated into 14 standardized agricultural reference dates (Day of Year):
$$\text{DOYs} = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]$$
Constructing a unified $14\text{ DOYs} \times 9\text{ bands} = 126\text{-band}$ multi-temporal spectral cube.

---

### NASA Harvest Presto geospatial foundation model

**Presto** is a global geospatial transformer foundation model pre-trained on multi-sensor Earth observation time series:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

* **Multi-sensor joint tokenization**: Simultaneously projects multi-temporal Sentinel-1 SAR ($\sigma^0_{VV}, \sigma^0_{VH}$) and Sentinel-2 optical reflectances ($B02$–$B12$) into a unified high-dimensional latent space.
* **Sinusoidal spatio-temporal embeddings**: Encodes continuous geographic coordinates $(\text{latitude}, \text{longitude})$ and seasonal acquisition months, conditioning model attention on geographic and bioclimatic context.
* **Latent representation extraction**: Outputs **128-dimensional multi-temporal token embeddings** from S1 SAR sequences and **128-dimensional embeddings** from S2 optical sequences, capturing complex non-linear phenological dynamics superior to standard handcrafted summary metrics.

### Object-based image analysis (OBIA) & segmentation deep dive

#### 1. Multi-temporal S1 SAR summed / mean composite generation
Single-date Synthetic Aperture Radar (SAR) imagery inherently suffers from **multiplicative speckle noise** caused by random constructive and destructive interference of coherent backscatter waves from distributed scatterers within a resolution cell. 

In our pipeline, before performing segmentation, we compute a **multi-temporal mean amplitude composite** across all valid radar acquisitions spanning the entire agricultural season ($N \approx 20\text{--}40$ dates):

$$\bar{\sigma}^0_{\text{temporal}} = \frac{1}{N} \sum_{k=1}^N \sigma^0_k(x, y)$$

* **Speckle variance reduction**: According to radar statistical theory, temporal multi-look averaging reduces speckle variance by a factor of $1/N$, effectively achieving an Equivalent Number of Looks ($\text{ENL}$) equal to $N$:
  $$\text{Var}(\bar{I}) = \frac{\sigma^2}{N}$$
* **Stationary boundary enhancement**: While crops undergo seasonal phenological cycles, permanent agricultural landscape elements (field boundaries, farm tracks, drainage ditches, hedgerows, and parcel borders) maintain stable geometric and dielectric boundaries. Multi-temporal averaging sharpens these boundaries while smoothing out transient field-interior speckle, producing the ideal input for high-precision segmentation.

#### 2. Segmentation modes in detail

```
+----------------------------------------------------------------------------------------------------+
|                                    OBIA SEGMENTATION METHODOLOGIES                                 |
+----------------------------------------------------------------------------------------------------+
|  [Mode 1: SLIC Superpixels]  --> Simple Linear Iterative Clustering on buffered 2048x2048 tiles     |
|                                  (64px halo prevents tile seam artifacts; UInt32 segment raster)   |
|  [Mode 2: Meta AI SAM]       --> Vision Transformer (ViT) foundation model parcel delineation      |
|                                  (Bilateral edge preservation & distance transform hole-filling)    |
|  [Mode 3: LPIS Cadastre]     --> Ingestion of official European farmer declaration vectors (GPKG)  |
|                                  (High-speed R-Tree spatial indexing & 10m rasterization)          |
+----------------------------------------------------------------------------------------------------+
```

* **Mode 1: SLIC (Simple Linear Iterative Clustering - Recommended for general operational use)**:
  - Adapts k-means clustering in a 5-dimensional feature-spatial space $(I_1, I_2, \dots, I_k, x, y)$.
  - **Distance metric**:
    $$D_s = \sqrt{d_{\text{color}}^2 + \left(\frac{d_{xy}}{S}\right)^2 \cdot m^2}$$
    Where $S = \sqrt{N_{\text{pixels}} / K}$ is the superpixel grid step, $m$ is the compactness parameter balancing boundary adherence vs regular parcel shape ($m = 10.0$).
  - **Seamless buffered tiling**: Processes rasters in $2048 \times 2048$ pixel tiles with a **64-pixel overlapping halo buffer**. Segment boundaries are harmonized across tile edges to prevent edge cutoff artifacts.
* **Mode 2: Meta AI SAM (Segment Anything Model - Deep foundation segmentation)**:
  - Utilizes Meta AI Vision Transformer backbones (`vit_h`, `vit_l`, `vit_b`) pre-trained on $>1$ billion masks.
  - Automatically distributes a regular prompt grid across multi-spectral/SAR composites.
  - Applies **bilateral edge filtering** and **morphological distance transform hole-filling** to close gaps along field boundaries.
* **Mode 3: LPIS (Land Parcel Identification System - Official cadastral parcels)**:
  - Ingests official national cadastral parcel boundary datasets (`.gpkg` or `.shp`) provided by Member State paying agencies (e.g. BRP in the Netherlands, ISIP in Portugal, ARiMR in Poland).
  - Employs **R-Tree spatial indexing** to assign every registered agricultural polygon a unique 32-bit `segment_id` rasterized at $10.0\text{ m}$ in `EPSG:3857`.

#### 3. Multimodal data footprint generation (`*_footprint.tif`)
Satellite tracks are captured along inclined orbits (descending $\approx -12^\circ$, ascending $\approx +12^\circ$), creating slanted swath edges and variable spatial bounds between radar (S1) and optical (S2) sensors.

In Stage 0, the pipeline computes a **binary multimodal data footprint raster**:

$$\text{Footprint}(x, y) = \begin{cases} 1 & \text{if } \text{Valid}(\text{S1}_{\text{SAR}}(x, y)) \land \text{Valid}(\text{S2}_{\text{optical}}(x, y)) \land \text{Valid}(\text{NUTS2}(x, y)) \\ 0 & \text{otherwise} \end{cases}$$

* **Eliminates edge distortion**: Strictly bounds feature extraction and model inference to pixels where all $170+$ radar and optical bands have valid, uncorrupted physical measurements.
* **Prevents no-data classification**: Ensures that partial-coverage edges outside the satellite swath are masked out, preventing false classifications along boundary margins.

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
|  [Concatenated High-Dimensional Feature Vector X in R^D]                                           |
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
|  -> Probability Vector P_MLP in [0, 1]^K      |  -> Probability Vector P_XGB in [0, 1]^K           |
+-----------------------------------------------+----------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|  [Soft-Voting Probability Blend]: P_hat(C_k | X) = 0.65 * P_MLP(C_k | X) + 0.35 * P_XGB(C_k | X)  |
|  [Bayesian Prior Calibration]: P_calibrated(C_k | X) adjusted by real-world crop acreage priors    |
+----------------------------------------------------------------------------------------------------+
```

#### 1. Tier 1: Handcrafted temporal & spectral features
* **Sentinel-1 SAR temporal moments**: For each polarization ($VV$, $VH$) and cross-ratio ($VH/VV$), the pipeline extracts statistical moments across the agricultural calendar:
  $$\mu = \frac{1}{N}\sum_{t=1}^N \sigma^0_t, \quad \sigma = \sqrt{\frac{1}{N}\sum_{t=1}^N (\sigma^0_t - \mu)^2}, \quad \text{Min}, \quad \text{Max}, \quad \Delta \text{dB} = \text{Max} - \text{Min}$$
  These metrics capture surface roughness, canopy closure speed, and abrupt drops in backscatter caused by harvesting.
* **Sentinel-2 multi-spectral trajectories**: Standardized 14-DOY reflectances across 9 spectral bands ($14 \times 9 = 126$ features) plus time-series vegetation indices ($\text{NDVI}(t)$, $\text{NDRE1}(t)$, $\text{NDRE2}(t)$, $\text{NDWI}(t)$) tracking chlorophyll absorption, red-edge shift, and canopy moisture.

#### 2. Tier 2: NASA Harvest Presto 256-dimensional foundation embeddings
NASA Harvest Presto encodes raw multi-temporal sequences using self-attention transformer blocks, mapping seasonal dynamics into two 128-dimensional latent vectors:
$$E_{\text{S1}} \in \mathbb{R}^{128} \quad (\text{SAR dynamics}), \qquad E_{\text{S2}} \in \mathbb{R}^{128} \quad (\text{Optical dynamics})$$

#### 3. Unified concatenated feature vector
The complete feature vector combines domain-specific physical interpretability with deep self-supervised representation learning:
$$X_{\text{fused}} = \left[ F_{\text{S1\_stats}} \,\|\, F_{\text{S2\_spectral}} \,\|\, E_{\text{Presto\_S1}} \,\|\, E_{\text{Presto\_S2}} \right] \in \mathbb{R}^{D}$$

---

### Hybrid multi-paradigm machine learning ensemble

Why combine Deep Neural Networks (PyTorch MLP) with Gradient Boosted Decision Trees (XGBoost)?
* **Deep Neural Networks (MLP)** excel at modeling smooth, continuous high-dimensional manifolds and projecting dense transformer embeddings.
* **Gradient Boosted Decision Trees (XGBoost)** excel at modeling discrete tabular decision thresholds, sharp spectral cutoffs, and step-function phenological transitions.
* **Complementary inductive biases**: Combining both architectures yields significantly lower generalization error and lower prediction variance than either model operating alone.

#### 1. PyTorch Deep MLP architecture & regularization
* **Network Topology**: Input Layer ($D$ dims) $\rightarrow$ `Dense(512)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.3)` $\rightarrow$ `Dense(256)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.3)` $\rightarrow$ `Dense(128)` $\rightarrow$ Output ($K$ classes).
* **Class-Weighted Cross-Entropy Loss**: Corrects for class imbalance between dominant cereals and minor specialty crops:
  $$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^K w_c \cdot y_{i,c} \cdot \log\left(\frac{\exp(z_{i,c})}{\sum_{j=1}^K \exp(z_{i,j})}\right), \quad \text{where } w_c = \frac{N}{K \cdot N_c}$$
* **Cosine Annealing Learning Rate Schedule**: Smoothly decays learning rate to escape local minima:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$

#### 2. XGBoost GBDT engine
* Trained on 250 trees with histogram-based split binning (`tree_method='hist'`), maximum depth `max_depth=6`, learning rate $\eta = 0.08$, sample subsampling `subsample=0.8`, and column feature subsampling `colsample_bytree=0.25`.

#### 3. Soft-voting probability blend
Combines posterior probability distributions from both models:
$$\hat{P}(C_k | X) = \alpha \cdot P_{\text{MLP}}(C_k | X) + (1 - \alpha) \cdot P_{\text{XGB}}(C_k | X)$$
Where $\alpha = 0.65$ (MLP ensemble weight) and $1 - \alpha = 0.35$ (XGBoost ensemble weight).

#### 4. Bayesian prior probability calibration
Machine learning models trained on balanced samples overestimate rare crops and underestimate dominant crops. The pipeline applies Bayesian calibration to align raw model probabilities with official agricultural registry crop acreages (`priors.json`):

$$P_{\text{calibrated}}(C_k | X) = \frac{\hat{P}(C_k | X) \cdot \left(\frac{P_{\text{true}}(C_k)}{P_{\text{train}}(C_k)}\right)^\gamma}{\sum_{j=1}^K \hat{P}(C_j | X) \cdot \left(\frac{P_{\text{true}}(C_j)}{P_{\text{train}}(C_j)}\right)^\gamma}$$

Where:
* $\hat{P}(C_k | X)$ is the soft-voting ensemble probability.
* $P_{\text{true}}(C_k)$ is the true statistical crop area proportion obtained from paying agency declarations.
* $P_{\text{train}}(C_k)$ is the training sample proportion.
* $\gamma = 0.7$ is the calibration damping exponent preventing extreme boundary distortion.

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

Located in `1a_Sentinel-2_preprocessor/`, this toolbox creates cloud-free, regular 10-day synthetic optical time-series composites matching 1:1 with the Sentinel-1 SAR pixel grid using an optimized **country-level shared repository architecture** (`workingDirs/{COUNTRY}/S2/`).

```
[Copernicus CDSE API / CreoDIAS L2A Archive]
                     |
                     v
    +----------------------------------+
    |  Stage 1: Retrieval & Masking    |  --> S2 L2A tile download/extraction to shared country pool,
    |  (SCL Cloud & Shadow Filtering)  |      SCL filtering for clouds, shadows, snow, and invalid pixels
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  Stage 2: Synthetic DOY          |  --> Multi-temporal spline interpolation across 14 standardized DOYs
    |  Time-Series Interpolation       |      (computed ONCE per country tile, eliminating duplicate processing)
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  Stage 3: Per-Orbit Stacking     |  --> Sub-pixel warping to Sentinel-1 raster bounding box,
    |  & 126-Band BigTIFF Generation   |      126-band BigTIFF creation, and 6 pyramid overviews per orbit
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
  * `multi_orbit_merger.py`: Nationwide multi-orbit confidence merger and seamless blender.
  * `presto_model.py`: Embedded NASA Harvest Presto transformer foundation architecture.
  *(Note: Legacy engines `classifier_otb.py` and `classifier_presto_s1.py` have been moved to `Archive_scripts/`)*

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
                      - Stage 0: Multimodal Data Footprint & S1 Mean Composite
                      - Stage 1: OBIA Segmentation (SLIC / SAM / LPIS)
                      - Stage 2: Stratified Sample Partitioning (70% Train / 30% Test)
                      - Stage 3: Feature Extraction (S1 Stats + S2 + Presto 128d Embeddings)
                      - Stage 4: Model Training (PyTorch Deep MLP + XGBoost Ensemble)
                      - Stage 5: High-Speed Vectorized Tile Inference (Bayesian Priors)
                      - Stage 6: Cropland & Data Footprint Masking
                      - Stage 7: Validation Accuracy Assessment & Excel Report (.xlsx)
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
1. **Stage 1 (Radiometric Calibration & Slice Assembly)**:
   - Queries Copernicus CDSE API (or local CreoDIAS COG storage) for all Sentinel-1 GRDH acquisitions covering the target track and dates.
   - Automatically downloads precise orbit files (POEORB) for 5 cm satellite positioning accuracy.
   - Performs thermal noise removal (TNR) and border noise removal (BNR).
   - Calibrates raw digital numbers to physical radar backscatter ($\sigma^0$).
   - Stitches along-track daily slices into continuous orbit swaths (`_temp_processing/1_calibrated/`).
2. **Stage 2 (Multi-Temporal Coregistration)**:
   - Launches ESA SNAP Graph Processing Tool (GPT).
   - Selects an optimal mid-season master scene and aligns all multi-temporal slave dates with sub-pixel cross-correlation, assembling a coregistered multi-temporal stack (`_temp_processing/2_wrapped_stack/`).
3. **Stage 3 (Terrain Correction, Stacking & NUTS2 Clipping)**:
   - Orthorectifies radar geometry using the Copernicus 30 m DEM (Range Doppler terrain correction).
   - Converts backscatter to decibels ($\text{dB} = 10 \log_{10}(\sigma^0)$).
   - Resamples and clips to the country NUTS2 administrative boundary in `EPSG:3857` at an exact $10.0\text{ m}$ pixel resolution.
   - Writes the final multi-temporal dual-polarization BigTIFF (`1_input_stacks/`) and generates 6 multi-scale pyramid overview layers (`.ovr`).

#### How to execute:
```powershell
# Recommended: Full automated run for Portugal (processing all national orbits sequentially):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PT -s 2024-10-15 -e 2025-09-15 --stage A

# Process a single orbit track:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 -s 2024-10-15 -e 2025-09-15 --stage A

# Force downloading directly from Copernicus Data Space (CDSE API):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PT --source cdse -s 2024-10-15 -e 2025-09-15 --stage A

# Exclude winter freeze period (December 1 to February 14) - recommended for Central/Northern Europe:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PL --exclude_winter --stage A

# Interactive wizard mode (guides you step-by-step):
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
   - Strictly masks out clouds (high/medium prob), thin cirrus, cloud shadows, snow, and defective pixels, saving clean GeoTIFFs to `_temp_processing/{TILE}_tif/`.
2. **Stage 2 (14-DOY Synthetic Time-Series Interpolation)**:
   - Evaluates the temporal distribution of cloud-free observations across the 14 standardized agricultural reference dates:
     $$\text{DOYs} = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]$$
   - Computes pure Python/NumPy multi-core linear interpolation between nearest forward and backward clear observations for all 9 bands, saving synthetic composite tiles to `_temp_processing/{TILE}/_synthetic_s2/day{DOY}_{YEAR}/`.
3. **Stage 3 (SAR Grid Matching & 126-Band BigTIFF Stacking)**:
   - Reads the Sentinel-1 SAR stack bounding box and spatial resolution as the master geometric reference.
   - Performs sub-pixel resampling ($\Delta X = 0.000\text{ m}, \Delta Y = 0.000\text{ m}$) ensuring 100% pixel-for-pixel alignment between optical reflectances and radar backscatter.
   - Compiles all 14 DOYs $\times$ 9 spectral bands into a unified 126-band BigTIFF (`Int16`, DEFLATE compressed) and builds 6 pyramid overview layers.

#### How to execute:
```powershell
# Recommended: Full automated run for Portugal using official CDSE API (4 parallel workers):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT --source cdse -s 2025-02-15 -e 2025-09-15 --stage A --threads 4

# Process a single orbit track:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --track PT/orbit_52 --source cdse -s 2025-02-15 -e 2025-09-15 --stage A --threads 4

# Interactive wizard mode (guides you step-by-step):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py
```

#### Expected output artifact:
* `workingDirs/{COUNTRY}/orbit_{ORBIT}/1_input_stacks/{COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif`  
  *(126-band Int16 optical BigTIFF, 10 m, EPSG:3857, matched 1:1 with SAR grid).*

---

### Phase 3: Multimodal machine learning classifier (`run_classifier.py`)

#### What happens in each classification stage (Stages 0 to 7):

| Stage | Name | Action & scientific description | Generated output file |
| :---: | :--- | :--- | :--- |
| **0** | **Data Footprint & SAR Composite** | Generates multi-temporal mean amplitude SAR composite ($\bar{\sigma}^0$) to suppress speckle. Computes binary spatial intersection of valid S1, S2, and NUTS2 boundaries. | `0_segmentation/*_s1_composite.tif`<br>`0_segmentation/*_data_footprint.tif` |
| **1** | **OBIA Segmentation** | Delineates homogeneous agricultural parcels using SLIC superpixels (with 64px halo buffer), Meta AI SAM foundation vision transformers, or official LPIS vector cadastre. | `0_segmentation/*_segmentation_{MODE}.tif` |
| **2** | **Stratified Sample Split** | Spatially intersects `samples.shp` with segment parcels. Partitions points into **70% training (`learn.shp`)** and **30% independent validation (`control.shp`)** with balanced class stratification. | `1_samples_and_features/learn_{MODE}.shp`<br>`1_samples_and_features/control_{MODE}.shp` |
| **3** | **Feature Extraction** | Extracts temporal backscatter statistics (mean, min, max, std of VH, VV, VH/VV), 14-DOY optical reflectances (`B02`–`B12`), dynamic NDVI, and **128d S1 + 128d S2 NASA Harvest Presto token embeddings**. | `1_samples_and_features/*_features_{MODE}.csv`<br>`1_samples_and_features/features_scaler.pkl` |
| **4** | **Model Training** | Fits class-weighted PyTorch Deep MLP (`BatchNorm1d`, `Dropout`, Cosine Annealing) and XGBoost GBDT (250 trees, histogram split). Combines models via soft-voting ensemble blend ($0.65\text{ MLP} + 0.35\text{ XGB}$). | `2_models/*_model_{MODE}.pkl`<br>`2_models/presto_encoder.pt` |
| **5** | **Vectorized Tile Inference** | Reads input rasters in $2048 \times 2048$ blocks. Uses fast `np.bincount` zonal aggregation, batched Presto embedding forward passes, Bayesian prior calibration, and $O(1)$ LUT raster reconstruction. | `3_maps/*_classified_{MODE}.tif`<br>`3_maps/*_confidence_{MODE}.tif` |
| **6** | **Cropland Masking** | Applies high-resolution agricultural mask (`AgriMasks/{COUNTRY}/`) and data footprint, suppressing non-agricultural surfaces (forests, urban, water bodies). | `3_maps/*_classified_masked_{MODE}.tif`<br>`3_maps/*_confidence_masked_{MODE}.tif` |
| **7** | **Accuracy Assessment** | Evaluates predictions against the independent 30% control dataset. Computes full confusion matrix, Overall Accuracy (OA), Cohen's Kappa ($\kappa$), User's Accuracy (Precision), Producer's Accuracy (Recall), and F1-scores. | `4_reports/report_*_metrics_{MODE}.xlsx` |

#### How to execute:
```powershell
# Recommended SOTA: Multimodal Deep MLP + XGBoost + Presto with SLIC Superpixels:
python 2_classifier/run_classifier.py --country PT --classifier mlpxgb_presto --seg_mode slic --stage A

# Run single orbit track:
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode slic --stage A

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
   $$\text{Pixel}_{\text{final}} = \begin{cases} \text{Crop}_{\text{Orbit 1}} & \text{if } \text{Confidence}_{\text{Orbit 1}} \ge \text{Confidence}_{\text{Orbit 2}} \\ \text{Crop}_{\text{Orbit 2}} & \text{otherwise} \end{cases}$$
4. Applies morphological sieve filtering (removing isolated pixel noise $< 10\text{ pixels}$) and enforces the nationwide agricultural mask.
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
* `workingDirs/{COUNTRY}/national_products/{COUNTRY}_national_crop_map_{SEG_MODE}.tif`
* `workingDirs/{COUNTRY}/national_products/{COUNTRY}_national_confidence_{SEG_MODE}.tif`
* `workingDirs/{COUNTRY}/national_products/{COUNTRY}_national_accuracy_report_{SEG_MODE}.xlsx`

---

### Phase 5: How to visualize and validate maps in QGIS & ArcGIS Pro

#### 1. Loading and styling the Crop Map in QGIS:
1. Open **QGIS** and click **Layer $\rightarrow$ Add Layer $\rightarrow$ Add Raster Layer...**
2. Select `workingDirs/{COUNTRY}/national_products/{COUNTRY}_national_crop_map_slic.tif` (or per-orbit `*_classified_masked_slic.tif`).
3. Right-click the loaded layer in the Layers panel $\rightarrow$ **Properties** $\rightarrow$ select **Symbology** tab.
4. Set **Render type** to **Paletted / Unique values**.
5. Click **Classify** at the bottom:
   - Each numerical `crop_id` (e.g. `11` = Winter Wheat, `12` = Maize, `1430` = Rapeseed) will receive a unique vibrant color.
6. Click **OK** to render the interactive crop classification map.

#### 2. Inspecting Classification Confidence:
1. Add `*_confidence_masked_slic.tif` as a raster layer in QGIS.
2. In **Properties $\rightarrow$ Symbology**, set **Render type** to **Singleband pseudocolor**.
3. Choose a color ramp (e.g. `Viridis` or `RdYlGn`), with minimum value `0.0` (red / low confidence) to maximum value `1.0` (green / high confidence).
4. Overlay with transparency over the classified map to inspect model certainty across field interiors and boundaries.

#### 3. Analyzing the Excel Accuracy Report:
1. Open `report_{COUNTRY}_{ORBIT}_metrics_slic.xlsx` in Microsoft Excel or LibreOffice Calc.
2. Inspect the **Summary Sheet**:
   - **Overall Accuracy (OA)**: Percentage of correct crop identifications (target $> 85\text{--}90\%$).
   - **Cohen's Kappa ($\kappa$)**: Chance-corrected statistical agreement metric.
3. Inspect the **Confusion Matrix Sheet**:
   - Rows represent ground truth; columns represent model predictions.
   - Off-diagonal values reveal specific confusion pairs (e.g., Winter Wheat vs Winter Barley).
4. Inspect **Per-Class Metrics**:
   - **User's Accuracy (Precision)**: Reliability of predicted crop pixels.
   - **Producer's Accuracy (Recall)**: Completeness of ground truth detection.
   - **F1-Score**: Harmonic mean of Precision and Recall.

---

## Cloud-native geospatial engineering & memory architecture

To process nationwide multi-temporal satellite rasters exceeding $100\text{ GB}$ per orbit on standard institutional workstations and cloud nodes, the pipeline incorporates advanced high-performance computing (HPC) patterns:

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
* **v2.5 Architecture**: Divides the raster into regular $2048 \times 2048$ processing blocks. All $170+$ radar and optical bands are read sequentially in a single contiguous memory operation, eliminating $99.83\%$ of filesystem I/O system calls.

### 2. $O(1)$ array indexing lookup table (LUT) reconstruction
* Rather than performing spatial raster masking for every individual polygon, predictions and confidence values for all segment objects in a tile are stored in a 1D lookup array:
  $$\text{Raster}_{\text{classified}} = \text{LUT}_{\text{pred}}[\text{Segment}_{\text{tile}}]$$
* **Performance**: Reconstructs a $2048 \times 2048$ classified tile in **$\approx 2\text{ ms}$**, compared to $2.5\text{ to }4\text{ minutes}$ in legacy dictionary iteration.

### 3. OpenMP & BLAS multi-threading isolation
When running Python `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`, internal scientific libraries (OpenMP, Intel MKL, OpenBLAS) default to spawning $T$ threads per process. On a 16-core CPU running 8 worker processes, this spawns $8 \times 16 = 128$ threads, causing severe CPU cache thrashing and kernel lock contention.

The pipeline strictly isolates worker environments:
```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```
Ensuring $100\%$ linear multi-core scaling without thread over-subscription.

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
| **S1 Stage 1** | `_temp_processing/1_calibrated/` | `S1*_Sigma0_{POL}.tif` | `Float32`, 1 band | Single-slice radiometric $\sigma^0$ radar backscatter. |
| **S1 Stage 2** | `_temp_processing/2_wrapped_stack/`| `wrapped_stack.dim / .data` | ENVI/BEAM, $2N$ b. | Multi-temporal coregistered SAR stack across all DOYs. |
| **S1 Stage 3** | `1_input_stacks/` | `{COUNTRY}_orbit_{ORBIT}_*_VH_VV.tif` | `Float32`, $2N$ b. | **Final SAR BigTIFF**: Terrain corrected, 10m, EPSG:3857 with `.ovr`. |
| **S2 Stage 1** | `_temp_processing/{TILE}_tif/` | `*_{BAND}_20m.tif` | `UInt16`, 1 band | Masked surface reflectance per scene (`B02`–`B12`, `SCL`). |
| **S2 Stage 2** | `_temp_processing/{TILE}/_synthetic_s2/` | `day{DOY}_{YEAR}/{BAND}.tif` | `UInt16`, 1 band | 14-DOY synthetic cloud-free interpolated spectral bands. |
| **S2 Stage 3** | `1_input_stacks/` | `{COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif` | `Int16`, 126 bands | **Final Optical BigTIFF**: 14 dates $\times$ 9 bands, 10m grid matched to S1. |
| **Classif. 0** | `2_classification/0_segmentation/`| `{COUNTRY}_{ORBIT}_s1_composite.tif` | `Float32`, 1 band | **SAR Multi-temporal Mean Composite**: Speckle-reduced amplitude for OBIA. |
| **Classif. 0** | `2_classification/0_segmentation/`| `{COUNTRY}_{ORBIT}_data_footprint.tif` | `Byte`, 1 band | **Binary Valid Footprint**: Spatial intersection of valid S1, S2, and NUTS2. |
| **Classif. 1** | `2_classification/0_segmentation/`| `{COUNTRY}_{ORBIT}_segmentation_{MODE}.tif` | `UInt32`, 1 band | **OBIA Segment Objects**: Integer raster with unique parcel identifiers. |
| **Classif. 2** | `2_classification/1_samples_and_features/` | `learn_{MODE}.shp / control_{MODE}.shp` | Vector Points | Stratified 70% training and 30% independent validation samples. |
| **Classif. 3** | `2_classification/1_samples_and_features/` | `{COUNTRY}_{ORBIT}_features_{MODE}.csv` | Tabular CSV | Extracted S1 temporal stats, S2 reflectances, and 128d Presto tokens. |
| **Classif. 3** | `2_classification/1_samples_and_features/` | `features_scaler.pkl` | Pickle Object | Saved feature standardization scaler for inference. |
| **Classif. 4** | `2_classification/2_models/` | `{COUNTRY}_{ORBIT}_model_{MODE}.pkl` | Pickle Object | Serialized PyTorch Deep MLP + XGBoost ensemble checkpoint. |
| **Classif. 5** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_classified_{MODE}.tif` | `UInt16`, 1 band | Raw predicted crop classification raster (`crop_id` values). |
| **Classif. 5** | `2_classification/3_maps/` | `{COUNTRY}_{ORBIT}_confidence_{MODE}.tif` | `Float32`, 1 band | Softmax ensemble probability confidence map ($0.0$ to $1.0$). |
| **Classif. 6** | `2_classification/3_maps/` | `*_classified_masked_{MODE}.tif` | `UInt16`, 1 band | **Final Track Crop Map**: Cropland-masked & footprint-bounded raster. |
| **Classif. 6** | `2_classification/3_maps/` | `*_confidence_masked_{MODE}.tif` | `Float32`, 1 band | **Final Track Confidence Map**: Cropland-masked confidence raster. |
| **Classif. 7** | `2_classification/4_reports/` | `report_{COUNTRY}_{ORBIT}_metrics_{MODE}.xlsx` | Styled Excel | Detailed validation report: OA, $\kappa$, Confusion Matrix, F1-scores. |
| **Merge 4** | `national_products/` | `{COUNTRY}_national_crop_map_{MODE}.tif` | `UInt16`, 1 band | **Seamless National Crop Map**: Confidence-blended multi-orbit BigTIFF. |
| **Merge 4** | `national_products/` | `{COUNTRY}_national_confidence_{MODE}.tif` | `Float32`, 1 band | **Seamless National Confidence Map**: Multi-orbit confidence raster. |
| **Merge 4** | `national_products/` | `{COUNTRY}_national_accuracy_report_{MODE}.xlsx` | Styled Excel | Aggregated nationwide statistical validation report. |
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
