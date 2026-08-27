# Multimodal machine learning classifier & national merger toolbox

This toolbox provides an enterprise-grade, object-based image analysis (OBIA) crop type classification suite powered by multimodal fusion of **Sentinel-1 SAR** and **Sentinel-2 optical** time series, **NASA Harvest Presto geospatial foundation model embeddings**, deep neural networks, gradient boosted decision trees, and Bayesian statistical prior calibration.

---

## Scientific methodology & artificial intelligence architecture

### 1. Multimodal Dual-Tier Feature Fusion Architecture

The classification suite combines handcrafted physical remote sensing features with self-supervised geospatial foundation model latent representations:

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

#### Tier 1: Handcrafted Temporal & Spectral Features
* **Sentinel-1 SAR temporal moments**: For each polarization ($VV$, $VH$) and cross-ratio ($VH/VV$), the pipeline extracts statistical moments across the agricultural calendar:
  $$\mu = \frac{1}{N}\sum_{t=1}^N \sigma^0_t, \quad \sigma = \sqrt{\frac{1}{N}\sum_{t=1}^N (\sigma^0_t - \mu)^2}, \quad \text{Min}, \quad \text{Max}, \quad \Delta \text{dB} = \text{Max} - \text{Min}$$
  These metrics capture surface roughness, canopy closure speed, and abrupt drops in backscatter caused by harvesting.
* **Sentinel-2 multi-spectral trajectories**: Standardized 14-DOY reflectances across 9 spectral bands ($14 \times 9 = 126$ features) plus time-series vegetation indices ($\text{NDVI}(t)$, $\text{NDRE1}(t)$, $\text{NDRE2}(t)$, $\text{NDWI}(t)$) tracking chlorophyll absorption, red-edge shift, and canopy moisture.

#### Tier 2: NASA Harvest Presto 256-Dimensional Foundation Embeddings
NASA Harvest Presto encodes raw multi-temporal sequences using self-attention transformer blocks, mapping seasonal dynamics into two 128-dimensional latent vectors:
$$E_{\text{S1}} \in \mathbb{R}^{128} \quad (\text{SAR dynamics}), \qquad E_{\text{S2}} \in \mathbb{R}^{128} \quad (\text{Optical dynamics})$$

#### Unified Concatenated Feature Vector
The complete feature vector combines domain-specific physical interpretability with deep self-supervised representation learning:
$$X_{\text{fused}} = \left[ F_{\text{S1\_stats}} \,\|\, F_{\text{S2\_spectral}} \,\|\, E_{\text{Presto\_S1}} \,\|\, E_{\text{Presto\_S2}} \right] \in \mathbb{R}^{D}$$

---

### 2. Hybrid Multi-Paradigm Machine Learning Ensemble

Why combine Deep Neural Networks (PyTorch MLP) with Gradient Boosted Decision Trees (XGBoost)?
* **Deep Neural Networks (MLP)** excel at modeling smooth, continuous high-dimensional manifolds and projecting dense transformer embeddings.
* **Gradient Boosted Decision Trees (XGBoost)** excel at modeling discrete tabular decision thresholds, sharp spectral cutoffs, and step-function phenological transitions.
* **Complementary inductive biases**: Combining both architectures yields significantly lower generalization error and lower prediction variance than either model operating alone.

#### PyTorch Deep MLP Architecture & Regularization
* **Network Topology**: Input Layer ($D$ dims) $\rightarrow$ `Dense(512)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.3)` $\rightarrow$ `Dense(256)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.3)` $\rightarrow$ `Dense(128)` $\rightarrow$ Output ($K$ classes).
* **Class-Weighted Cross-Entropy Loss**: Corrects for class imbalance between dominant cereals and minor specialty crops:
  $$\mathcal{L}_{\text{MLP}} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^K w_c \cdot y_{i,c} \cdot \log\left(\frac{\exp(z_{i,c})}{\sum_{j=1}^K \exp(z_{i,j})}\right), \quad \text{where } w_c = \frac{N}{K \cdot N_c}$$
* **Cosine Annealing Learning Rate Schedule**: Smoothly decays learning rate to escape local minima:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$

#### XGBoost GBDT Engine
* Trained on 250 trees with histogram-based split binning (`tree_method='hist'`), maximum depth `max_depth=6`, learning rate $\eta = 0.08$, sample subsampling `subsample=0.8`, and column feature subsampling `colsample_bytree=0.25`.

#### Soft-Voting Probability Blend
Combines posterior probability distributions from both models:
$$\hat{P}(C_k | X) = \alpha \cdot P_{\text{MLP}}(C_k | X) + (1 - \alpha) \cdot P_{\text{XGB}}(C_k | X)$$
Where $\alpha = 0.65$ (MLP ensemble weight) and $1 - \alpha = 0.35$ (XGBoost ensemble weight).

#### Bayesian Prior Probability Calibration
Machine learning models trained on balanced samples overestimate rare crops and underestimate dominant crops. The pipeline applies Bayesian calibration to align raw model probabilities with official agricultural registry crop acreages (`priors.json`):

$$P_{\text{calibrated}}(C_k | X) = \frac{\hat{P}(C_k | X) \cdot \left(\frac{P_{\text{true}}(C_k)}{P_{\text{train}}(C_k)}\right)^\gamma}{\sum_{j=1}^K \hat{P}(C_j | X) \cdot \left(\frac{P_{\text{true}}(C_j)}{P_{\text{train}}(C_j)}\right)^\gamma}$$

Where:
* $\hat{P}(C_k | X)$ is the soft-voting ensemble probability.
* $P_{\text{true}}(C_k)$ is the true statistical crop area proportion obtained from paying agency declarations.
* $P_{\text{train}}(C_k)$ is the training sample proportion.
* $\gamma = 0.7$ is the calibration damping exponent preventing extreme boundary distortion.

---

## Processing flowchart & classification pipeline architecture

```
+----------------------------------------------------------------------------------------------------+
|                         MULTIMODAL MACHINE LEARNING PIPELINE FLOWCHART                             |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 0: Multimodal Data Footprint Generation]          |
                     | Intersection of valid S1 SAR and S2 Optical coverage     |
                     | -> Output: {COUNTRY}_{ORBIT}_data_footprint.tif          |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 1: Object-Based Image Segmentation]               |
                     | Delineation of agricultural field objects:               |
                     | - Mode 1: SLIC Superpixels (Fast, 64px buffered tiles)   |
                     | - Mode 2: Meta AI SAM (Vision Transformer Foundation)    |
                     | - Mode 3: LPIS Cadastre (Official European Parcel GPKG)  |
                     | -> Output: {COUNTRY}_{ORBIT}_segmentation_{MODE}.tif     |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 2: Stratified Sample Point Partitioning]          |
                     | Splits samples.shp into:                                 |
                     | - 70% Training / Learn dataset (learn_{MODE}.shp)        |
                     | - 30% Independent Validation (control_{MODE}.shp)        |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 3: Multimodal Feature Extraction]                 |
                     | 1. S1 temporal backscatter stats (VH, VV, VH/VV)         |
                     | 2. S2 surface reflectances (B02-B12) & dynamic NDVI      |
                     | 3. NASA Harvest Presto 128d S1 + 128d S2 Token Embeddings|
                     | -> Output: {COUNTRY}_{ORBIT}_features_{MODE}.csv         |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 4: Fusion Ensemble Training]                      |
                     | 1. Class-weighted PyTorch Deep MLP (BatchNorm, Dropout)  |
                     | 2. XGBoost GBDT (250 trees, max_depth=6, hist-split)     |
                     | 3. Fit soft-voting ensemble & standard scaler            |
                     | -> Output: {COUNTRY}_{ORBIT}_model_{MODE}.pkl            |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 5: High-Performance Vectorized Inference]         |
                     | 1. Tile-level block I/O (Read full 2048x2048 tile once)  |
                     | 2. Vectorized np.bincount zonal stats across 170 bands   |
                     | 3. Presto batched embedding forward pass & MLP+XGB infer |
                     | 4. Bayesian prior calibration adjustment                 |
                     | 5. O(1) LUT raster reconstruction                        |
                     | -> Output: classified_{MODE}.tif & confidence_{MODE}.tif |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 6: Cropland & Data Footprint Masking]             |
                     | Suppresses non-agricultural areas (forests, water, urban)|
                     | -> Output: classified_masked.tif & confidence_masked.tif |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 7: Out-of-Bag Validation & Excel Reporting]       |
                     | Evaluates against 30% control dataset:                   |
                     | - Overall Accuracy (OA) and Cohen's Kappa (kappa)        |
                     | - Full confusion matrix (ground truth vs predictions)    |
                     | - Per-crop Precision, Recall, and F1-scores              |
                     | -> Output: report_{COUNTRY}_{ORBIT}_metrics.xlsx         |
                     +----------------------------------------------------------+
```

---

## Phase 4: Nationwide multi-orbit merging flowchart (`run_merge.py`)

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

---

## Classifier models & modality options

| Model identifier | Modality | Engine | Description |
| :--- | :---: | :---: | :--- |
| **`mlpxgb_presto`** | `[S1 + S2]` | Presto + MLP + XGBoost | **Recommended SOTA**: 128d Presto tokens, 3-layer deep MLP, and 250 XGBoost trees with soft-voting blend ($0.65\text{ MLP} + 0.35\text{ XGB}$). |
| **`presto_s1`** | `[S1 only]` | Presto ANN | Single-radar SAR classification using temporal $\sigma^0$ sequences and 128d Presto embeddings. |
| **`otb`** | `[S1 + S2]` | Orfeo ToolBox | Standard machine learning baseline (Random Forest / SVM). |
| **`mlp`** | `[S1 + S2]` | PyTorch Deep MLP | Pure deep neural network with BatchNorm, Dropout, and Cosine Annealing learning rate. |
| **`xgb`** | `[S1 + S2]` | XGBoost GBDT | Pure gradient boosted decision tree classifier with histogram-based splitting. |

---

## Segmentation & Data Footprinting Deep Dive

### 1. Multi-Temporal S1 SAR Mean Amplitude Composite for Segmentation
Directly segmenting single-date SAR imagery is problematic due to **multiplicative speckle noise** ($\text{Rayleigh}$ or $\text{Gamma}$ distributed intensity fluctuations). 

To overcome this, before running image segmentation, Stage 1 generates a **multi-temporal mean amplitude composite** across all valid radar dates in the agricultural calendar ($N \approx 20\text{--}40$ acquisitions):

$$\bar{\sigma}^0_{\text{temporal}}(x, y) = \frac{1}{N} \sum_{k=1}^N \sigma^0_k(x, y)$$

* **Speckle variance suppression**: The temporal multi-look averaging reduces speckle noise variance by a factor of $1/N$, yielding an effective number of looks $\text{ENL} \approx N$.
* **Stationary parcel edge enhancement**: While vegetative backscatter changes over the season, physical landscape boundaries (field margins, ditches, roads, hedgerows, fences) remain geometrically stationary throughout the year. Temporal averaging highlights these stationary boundaries while smoothing interior field variance, providing an optimal input raster for boundary delineation.

### 2. OBIA Segmentation Algorithms

#### Mode 1: SLIC (Simple Linear Iterative Clustering - Recommended)
* Operates in a 5D space combining radiometric intensity and 2D geographic coordinates:
  $$D = \sqrt{d_{\text{color}}^2 + \left(\frac{d_{xy}}{S}\right)^2 \cdot m^2}$$
  Where $S = \sqrt{N / K}$ is the superpixel grid interval, and $m = 10.0$ is the compactness parameter enforcing regular polygon geometry.
* **Buffered tile processing**: Rasters are segmented in $2048 \times 2048$ blocks with a **64-pixel halo buffer**. Edge segments are matched and merged across boundaries to completely prevent tile seam artifacts.
* **32-bit Integer Raster**: Segments are outputted as a `UInt32` GeoTIFF (`*_segmentation_slic.tif`) where every discrete parcel receives a globally unique non-zero integer identifier (`segment_id`).

#### Mode 2: Meta AI SAM (Segment Anything Model)
* Adapts Meta AI Vision Transformers (`vit_h`, `vit_l`, `vit_b`) pre-trained on over 1 billion segmentation masks.
* Automatically distributes a structured prompt grid across composite rasters.
* Implements **bilateral edge filtering** and **distance transform hole-filling** to close internal parcel gaps and produce smooth field geometries.

#### Mode 3: LPIS Cadastre (Official European Agricultural Parcels)
* Ingests official vector databases (`.gpkg` or `.shp`) provided by national paying agencies (e.g., BRP in the Netherlands, ISIP in Portugal, ARiMR in Poland).
* Uses high-speed spatial bounding-box filtering (`pyogrio` with R-Tree spatial indexing) to extract intersecting parcels.
* Reprojects and rasterizes vectors into a $10.0\text{ m}$ `UInt32` grid in `EPSG:3857`.

### 3. Multimodal Data Footprint Generation (`*_data_footprint.tif`)
Because Sentinel-1 SAR orbits are tilted (inclined polar orbits) while Sentinel-2 optical granules follow UTM MGRS tiles, their valid observation swaths do not perfectly overlap. 

In Stage 0, the pipeline generates a **multimodal data footprint mask**:

$$\text{Footprint}(x, y) = \begin{cases} 1 & \text{if } \text{Valid}(\text{S1}_{\text{SAR}}(x, y)) \land \text{Valid}(\text{S2}_{\text{optical}}(x, y)) \land \text{Valid}(\text{NUTS2}(x, y)) \\ 0 & \text{otherwise} \end{cases}$$

* **Edge artifact elimination**: Strictly restricts downstream training, feature extraction, and tile inference to regions with valid, uncorrupted data across all sensor channels.
* **Prevents border classification anomalies**: Ensures that pixels with partial satellite coverage outside the orbit swath are properly treated as `NoData` ($0$).

---

## Execution commands

### 1. Interactive setup wizard (recommended)
```powershell
python run_classifier.py
```

### 2. SOTA multimodal classification (`[S1 + S2]`)
```powershell
# SLIC superpixels (fast, fully automated):
python run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode slic --stage A

# Run classification for an entire country across all orbits:
python run_classifier.py --country PT --classifier mlpxgb_presto --seg_mode slic --stage A

# Official cadastral parcel vectors (LPIS):
python run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode lpis --lpis_vector path/to/parcels.gpkg --stage A

# Meta AI SAM deep learning segmentation:
python run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode sam --stage A
```

### 3. Single-radar classification (`[S1 only]`)
```powershell
python run_classifier.py --track PT/orbit_52 --classifier presto_s1 --seg_mode slic --stage A
```

### 4. Running specific single stages (0 to 7)
```powershell
# Run only feature extraction (Stage 3):
python run_classifier.py --track PT/orbit_52 --stage 3

# Run only model training (Stage 4):
python run_classifier.py --track PT/orbit_52 --stage 4

# Run only vectorized inference with Bayesian priors (Stage 5):
python run_classifier.py --track PT/orbit_52 --stage 5

# Run only accuracy metrics and export Excel report (Stage 7):
python run_classifier.py --track PT/orbit_52 --stage 7
```

### 5. Phase 4: Nationwide multi-orbit merge
```powershell
# Merge SLIC classifications into a seamless national map:
python run_merge.py --country PT --seg_mode slic

# Merge LPIS classifications into a seamless national map:
python run_merge.py --country PT --seg_mode lpis
```

---

## Command-line arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-t, --track` | string | `None` | Satellite track identifier (e.g. `PT/orbit_52`, `NL/orbit_88`). |
| `-c, --country` | string | `None` | Country code (e.g. `PT`, `NL`, `PL`, `ES`, `FR`, `DE`). |
| `--classifier` | choice | `mlpxgb_presto` | Classifier model: `mlpxgb_presto` [S1+S2 SOTA], `presto_s1` [S1 only], `otb` [S1+S2], `mlp` [S1+S2], `xgb` [S1+S2]. |
| `--seg_mode` | choice | `slic` | Segmentation mode: `slic` (superpixels), `sam` (Meta AI), `lpis` (cadastre). |
| `--stage` | string | `None` | Stage to execute: `A` (all stages 0-7), or single stage `0`..`7`. |
| `--mlp_weight` | float | `0.65` | Weight of MLP in fusion ensemble (0.0 to 1.0; remaining weight assigned to XGBoost). |
| `--s1_raster` | string | `None` | Optional explicit path override to Sentinel-1 BigTIFF raster. |
| `--s2_raster` | string | `None` | Optional explicit path override to Sentinel-2 BigTIFF raster. |
| `--lpis_vector` | string | `None` | Optional explicit path to official LPIS parcel vector file (`.shp`, `.gpkg`). |

---

## Complete Output Products & Intermediate Artifacts

All classification outputs are stored sequentially by stage in `workingDirs/{COUNTRY}/orbit_{ORBIT}/2_classification/`:

| Stage | Directory | Output file name pattern | Format / Type | Purpose & description |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_s1_composite.tif` | `Float32`, 1 b. | Multi-temporal mean SAR amplitude composite for segmentation. |
| **0** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_data_footprint.tif` | `Byte`, 1 b. | Binary mask ($1=\text{valid}, 0=\text{nodata}$) of S1, S2, and NUTS intersection. |
| **1** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_segmentation_{MODE}.tif` | `UInt32`, 1 b. | Object segmentation raster with unique integer `segment_id` per parcel. |
| **2** | `1_samples_and_features/` | `learn_{MODE}.shp` | Vector Points | Stratified 70% training points with spatial attributes. |
| **2** | `1_samples_and_features/` | `control_{MODE}.shp` | Vector Points | Stratified 30% independent validation points. |
| **3** | `1_samples_and_features/` | `{COUNTRY}_{ORBIT}_features_{MODE}.csv` | Tabular CSV | Extracted S1 temporal statistics, S2 reflectances, and 128d Presto tokens. |
| **3** | `1_samples_and_features/` | `features_scaler.pkl` | Pickle Checkpoint | Serialized feature standardization scaler (`StandardScaler`). |
| **4** | `2_models/` | `{COUNTRY}_{ORBIT}_model_{MODE}.pkl` | Pickle Checkpoint | Serialized PyTorch Deep MLP weights + XGBoost GBDT trees + label encoder. |
| **4** | `2_models/` | `presto_encoder.pt` | PyTorch Tensor | Pre-trained Presto transformer feature extractor weights. |
| **5** | `3_maps/` | `{COUNTRY}_{ORBIT}_classified_{MODE}.tif` | `UInt16`, 1 b. | Raw pixel crop classification raster (pixel value = numeric `crop_id`). |
| **5** | `3_maps/` | `{COUNTRY}_{ORBIT}_confidence_{MODE}.tif` | `Float32`, 1 b. | Softmax ensemble probability confidence map ($0.0\text{ to }1.0$). |
| **6** | `3_maps/` | `*_classified_masked_{MODE}.tif` | `UInt16`, 1 b. | **Final Single-Orbit Crop Map**: Cropland-masked & footprint-clipped GeoTIFF. |
| **6** | `3_maps/` | `*_confidence_masked_{MODE}.tif` | `Float32`, 1 b. | **Final Single-Orbit Confidence Map**: Cropland-masked confidence GeoTIFF. |
| **7** | `4_reports/` | `report_{COUNTRY}_{ORBIT}_metrics_{MODE}.xlsx` | Styled Excel | Accuracy workbook: Overall Accuracy, $\kappa$, Confusion Matrix, F1-scores. |
| **Merge** | `national_products/` | `{COUNTRY}_national_crop_map_{MODE}.tif` | `UInt16`, 1 b. | **Seamless National Crop Map**: Confidence-blended multi-orbit BigTIFF. |
| **Merge** | `national_products/` | `{COUNTRY}_national_confidence_{MODE}.tif` | `Float32`, 1 b. | **Seamless National Confidence Map**: Multi-orbit confidence raster. |
| **Merge** | `national_products/` | `{COUNTRY}_national_accuracy_report_{MODE}.xlsx` | Styled Excel | Aggregated nationwide statistical validation report. |
