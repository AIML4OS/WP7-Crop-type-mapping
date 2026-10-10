# Multimodal machine learning classifier & national merger toolbox

This toolbox provides an enterprise-grade, object-based image analysis (OBIA) crop type classification suite powered by multimodal fusion of **Sentinel-1 SAR** and **Sentinel-2 optical** time series, **NASA Harvest Presto geospatial foundation model embeddings**, deep neural networks, gradient boosted decision trees, and Bayesian statistical prior calibration.

---

## Scientific methodology & artificial intelligence architecture

### 1. Multimodal dual-tier feature fusion architecture

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

#### Tier 1: Handcrafted temporal, spectral & object texture features (100D feature suite)
* **Sentinel-1 SAR temporal moments**: For each polarization (VV, VH) and cross-ratio (VH/VV), the pipeline extracts statistical moments across the agricultural calendar:
  Mean, Standard Deviation (Std), Minimum (Min), Maximum (Max), and Dynamic Range (Max - Min).
  These metrics capture surface roughness, canopy closure speed, and abrupt drops in backscatter caused by harvesting.
* **Specialized SAR phenological indices (RVI & Dual-Pol vegetation metrics)**:
  `RVI = 4 * VH / (VV + VH)`
  Tracks canopy volume fraction and vegetative biomass accumulation independent of soil surface dielectric variations.
* **Seasonal SAR backscatter dynamics (Delta_VH)**:
  `Delta_VH = VH_summer - VH_spring`
  Differentiates winter cereals (established autumn canopy) from spring-sown crops (bare soil in early spring transitioning to lush vegetation by mid-summer).
* **Intra-object texture variance & dispersion (field-level std & IQR)**:
  Captures internal parcel spatial heterogeneity: `std` and `IQR` (interquartile range) for VH backscatter and NDVI reflectances. Distinguishes dense uniform field canopies (wheat, barley) from wide-row crops (maize, sugar beets) and structured woody vegetation (vineyards, orchards, olive groves).
* **Sentinel-2 multi-spectral trajectories**: Standardized 14-DOY reflectances across 9 spectral bands (14 x 9 = 126 features) plus time-series vegetation indices (NDVI, NDRE1, NDRE2, NDWI) tracking chlorophyll absorption, red-edge shift, and canopy moisture.

#### Tier 2: NASA Harvest Presto 256-dimensional foundation embeddings
NASA Harvest Presto encodes raw multi-temporal sequences using self-attention transformer blocks, mapping seasonal dynamics into two 128-dimensional latent vectors:
* `E_S1` (128 dimensions): SAR temporal dynamics.
* `E_S2` (128 dimensions): Optical multispectral phenology.

#### Unified concatenated feature vector
The complete feature vector combines domain-specific physical interpretability with deep self-supervised representation learning into a unified vector:
`X_fused = [ F_S1_stats_texture || F_S2_spectral_texture || E_Presto_S1 || E_Presto_S2 ]`

---

### 2. Hybrid multi-paradigm machine learning ensemble

Combining Deep Neural Networks (PyTorch MLP) with Gradient Boosted Decision Trees (XGBoost) provides complementary strengths:
* **Deep Neural Networks (MLP)** excel at modeling smooth, continuous high-dimensional manifolds and projecting dense transformer embeddings.
* **Gradient Boosted Decision Trees (XGBoost)** excel at modeling discrete tabular decision thresholds, sharp spectral cutoffs, and step-function phenological transitions.
* Combining both architectures yields significantly lower generalization error and prediction variance than either model operating alone.

#### PyTorch Deep MLP architecture & regularization
* **Network topology**: Input Layer (D dims) -> `Dense(512)` -> `BatchNorm1d` -> `ReLU` -> `Dropout(p=0.3)` -> `Dense(256)` -> `BatchNorm1d` -> `ReLU` -> `Dropout(p=0.3)` -> `Dense(128)` -> Output (K classes).
* **Square-root balanced class loss weights**: Corrects for class imbalance between dominant broadacre crops and minor specialty crops without gradient explosion:
  `w_c = sqrt(N / (K * N_c))`
  Smooths extreme sample disparities, ensuring rare specialty crops (tomatoes, orchards, pulses) generate sufficient gradient updates without destabilizing backpropagation on dominant classes.
* **Cosine Annealing learning rate schedule**: Smoothly decays learning rate across training epochs to avoid local minima.

#### XGBoost GBDT engine
* Trained on 250 trees with histogram-based split binning (`tree_method='hist'`), maximum depth `max_depth=6`, learning rate 0.08, sample subsampling `subsample=0.8`, and column feature subsampling `colsample_bytree=0.25`.

#### Soft-voting probability blend
Combines posterior probability distributions from both models:
`P_final = 0.65 * P_MLP + 0.35 * P_XGB`
Where 65% weight is assigned to the deep MLP and 35% weight to XGBoost.

#### Dynamic geographic Bayesian prior calibration
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

### 3. Asynchronous GPU-CPU producer-consumer inference pipeline
Large-scale spatial inference across nationwide multi-temporal satellite stacks (>150 GB) is traditionally constrained by sequential disk read latencies. The classification engine implements a multi-threaded asynchronous producer-consumer pipeline:
* **Background CPU producer thread (`threading.Thread`)**: Prefetches the next candidate 2048 x 2048 tile block from NVMe storage, computes vectorized `np.bincount` zonal pixel-to-object statistics across all 170+ SAR and optical bands, and constructs high-dimensional tensor batches in system RAM.
* **Concurrent GPU consumer**: Simultaneously executes forward inference passes through NASA Harvest Presto transformer embeddings and the PyTorch Deep MLP on the active GPU tensor buffer (`cuda`).
* **Zero GPU idle time**: Disk I/O latency is completely masked behind GPU matrix multiplication, maximizing hardware saturation and accelerating national map generation by ~2.5x.

---

## Processing flowchart & classification pipeline architecture

```
+----------------------------------------------------------------------------------------------------+
|                         MULTIMODAL MACHINE LEARNING PIPELINE FLOWCHART                             |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 1: Multimodal Data Footprint Generation]          |
                     | Intersection of valid S1 SAR and S2 Optical coverage     |
                     | -> Output: {COUNTRY}_{ORBIT}_data_footprint.tif          |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 2: Object-Based Image Segmentation]               |
                     | Delineation of agricultural field objects:               |
                     | - Mode 1: SLIC Superpixels (Fast, 64px buffered tiles)   |
                     | - Mode 2: Meta AI SAM (Vision Transformer Foundation)    |
                     | - Mode 3: LPIS Cadastre (Official European Parcel GPKG)  |
                     | -> Output: {COUNTRY}_{ORBIT}_segmentation_{MODE}.tif     |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 3: Stratified Sample Point Partitioning]          |
                     | Splits samples.shp into:                                 |
                     | - 70% Training / Learn dataset (learn_{MODE}.shp)        |
                     | - 30% Independent Validation (control_{MODE}.shp)        |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 4: Multimodal Feature Extraction]                 |
                     | 1. S1 temporal backscatter stats (VH, VV, VH/VV)         |
                     | 2. S2 surface reflectances (B02-B12) & dynamic NDVI      |
                     | 3. NASA Harvest Presto 128d S1 + 128d S2 Token Embeddings|
                     | -> Output: {COUNTRY}_{ORBIT}_features_{MODE}.csv         |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 5: Fusion Ensemble Training]                      |
                     | 1. Class-weighted PyTorch Deep MLP (BatchNorm, Dropout)  |
                     | 2. XGBoost GBDT (250 trees, max_depth=6, hist-split)     |
                     | 3. Fit soft-voting ensemble & standard scaler            |
                     | -> Output: {COUNTRY}_{ORBIT}_model_{MODE}.pkl            |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 6: High-Performance Vectorized Inference]         |
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
                     | [Stage 7: Cropland & Data Footprint Masking]             |
                     | Suppresses non-agricultural areas (forests, water, urban)|
                     | -> Output: classified_masked.tif & confidence_masked.tif |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 8: Out-of-Bag Validation & Excel Reporting]       |
                     | Evaluates against 30% control dataset:                   |
                     | - Overall Accuracy (OA) and Cohen's Kappa (kappa)        |
                     | - Full confusion matrix (ground truth vs predictions)    |
                     | - Per-crop Precision, Recall, and F1-scores              |
                     | -> Output: report_{COUNTRY}_{ORBIT}_metrics.xlsx         |
                     +----------------------------+-----------------------------+
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

---

## Classifier models & modality options

| Model identifier | Modality | Engine | Status / Description |
| :--- | :---: | :---: | :--- |
| **`mlpxgb_presto`** | `[S1 + S2]` | Presto + MLP + XGBoost | **Recommended SOTA**: 128d Presto tokens, 3-layer deep MLP, and 250 XGBoost trees with soft-voting blend (65% MLP + 35% XGBoost). Aliases: `mlpxgb_presto_s1s2`, `s1s2`. |
| **`mlp`** | `[S1 + S2]` | PyTorch Deep MLP | Pure deep neural network with BatchNorm, Dropout, and Cosine Annealing learning rate. |
| **`xgb`** | `[S1 + S2]` | XGBoost GBDT | Pure gradient boosted decision tree classifier with histogram-based splitting. |
| **`presto_s1`** | `[S1 only]` | Presto ANN | *Archived*: Single-radar SAR classification using temporal Sigma0 sequences and 128d Presto embeddings. |
| **`otb`** | `[S1 + S2]` | Orfeo ToolBox | *Archived*: Classical machine learning baseline (Random Forest / SVM). |

---

## Segmentation & data footprinting deep dive

### 1. Multi-temporal S1 SAR mean amplitude composite for segmentation
Directly segmenting single-date SAR imagery is problematic due to multiplicative speckle noise. 

To overcome this, before running image segmentation, Stage 1 generates a **multi-temporal mean amplitude composite** across all valid radar dates in the agricultural calendar (N ~ 20 to 40 acquisitions):
* **Speckle variance suppression**: Temporal multi-look averaging reduces speckle noise variance by a factor of 1/N.
* **Stationary parcel edge enhancement**: While vegetative backscatter changes over the season, physical landscape boundaries (field margins, ditches, roads, hedgerows, fences) remain geometrically stationary throughout the year. Temporal averaging highlights these stationary boundaries while smoothing interior field variance, providing an optimal input raster for boundary delineation.

### 2. OBIA segmentation algorithms

#### Mode 1: SLIC (Simple Linear Iterative Clustering - recommended)
* Operates in a 5D feature space combining multi-temporal radiometric composite (S1 SAR temporal backscatter moments + S2 multi-spectral composite) and 2D spatial coordinates.
* **Adaptive target parcel size**: Dynamically sets target parcel size based on country agronomic profile (1.8 ha for Southern/Eastern Europe PT/ES/PL, 3.0 ha for Western Europe NL/FR/DE).
* **High-elasticity boundary tracking**: Compactness parameter (`m = 0.05`) allows superpixel boundaries to conform precisely to physical parcel boundaries, hedgerows, and drainage canals.
* **Buffered tile processing**: Rasters are segmented in 2048 x 2048 blocks with a **64-pixel halo buffer**. Edge segments are matched and merged across boundaries to completely prevent tile seam artifacts.
* **32-bit integer raster**: Segments are output as a `UInt32` GeoTIFF (`*_segmentation_slic.tif`) where every discrete parcel receives a globally unique non-zero integer identifier (`segment_id`).

#### Mode 2: Meta AI SAM (Segment Anything Model)
* Adapts Meta AI Vision Transformers (`vit_h`, `vit_l`, `vit_b`) pre-trained on over 1 billion segmentation masks.
* Automatically distributes a structured prompt grid across composite rasters.
* Implements bilateral edge filtering and distance transform hole-filling to close internal parcel gaps and produce smooth field geometries.

#### Mode 3: LPIS cadastre (official European agricultural parcels)
* Ingests official vector databases (`.gpkg` or `.shp`) provided by national paying agencies (e.g., BRP in the Netherlands, ISIP in Portugal, ARiMR in Poland).
* Uses high-speed spatial bounding-box filtering (`pyogrio` with R-Tree spatial indexing) to extract intersecting parcels.
* Reprojects and rasterizes vectors into a 10.0 m `UInt32` grid in `EPSG:3857`.

### 3. Multimodal data footprint generation (`*_data_footprint.tif`)
Because Sentinel-1 SAR orbits are tilted (inclined polar orbits) while Sentinel-2 optical granules follow UTM MGRS tiles, their valid observation swaths do not perfectly overlap. 

In Stage 1, the pipeline generates a **multimodal data footprint mask**:
* Computes the exact binary intersection of valid Sentinel-1, valid Sentinel-2, and NUTS2 regional boundaries (1 = valid, 0 = NoData).
* **Edge artifact elimination**: Strictly restricts downstream training, feature extraction, and tile inference to regions with valid, uncorrupted data across all sensor channels.
* **Prevents border classification anomalies**: Ensures that pixels with partial satellite coverage outside the orbit swath are properly treated as `NoData` (0).

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

### 4. Running specific single stages (1 to 8)
```powershell
# Run only feature extraction (Stage 4):
python run_classifier.py --track PT/orbit_52 --stage 4

# Run only model training (Stage 5):
python run_classifier.py --track PT/orbit_52 --stage 5

# Run only vectorized inference with Bayesian priors (Stage 6):
python run_classifier.py --track PT/orbit_52 --stage 6

# Run only accuracy metrics and export Excel report (Stage 8):
python run_classifier.py --track PT/orbit_52 --stage 8
```

### 5. Phase 4: Nationwide multi-orbit merge
```powershell
# Merge SLIC classifications into a seamless national map with SOTA ensemble:
python run_merge.py --country PT --classifier mlpxgb_presto --seg_mode slic

# Merge LPIS cadastral classifications for the Netherlands:
python run_merge.py --country NL --classifier mlpxgb_presto --seg_mode lpis

# Merge Meta AI SAM classifications using confidence blending:
python run_merge.py --country PT --classifier mlpxgb_presto --seg_mode sam --method confidence
```

---

## Command-line arguments

### Classifier runner (`run_classifier.py`)

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-t, --track` | string | `None` | Satellite track identifier (e.g. `PT/orbit_52`, `NL/orbit_88`). |
| `-c, --country` | string | `None` | Country code (e.g. `PT`, `NL`, `PL`, `ES`, `FR`, `DE`). |
| `--classifier` | choice | `mlpxgb_presto` | Classifier model: `mlpxgb_presto` [S1+S2 SOTA], `mlp` [S1+S2], `xgb` [S1+S2], `presto_s1` [S1 only], `otb` [S1+S2]. |
| `--seg_mode` | choice | `slic` | Segmentation mode: `slic` (superpixels), `sam` (Meta AI), `lpis` (cadastre). |
| `--stage` | string | `None` | Stage to execute: `A` (all stages 1-8), or single stage `1`..`8` (legacy `0`..`7` supported). |
| `--mlp_weight` | float | `0.65` | Weight of MLP in fusion ensemble (0.0 to 1.0; remaining weight assigned to XGBoost). |
| `--s1_raster` | string | `None` | Optional explicit path override to Sentinel-1 BigTIFF raster. |
| `--s2_raster` | string | `None` | Optional explicit path override to Sentinel-2 BigTIFF raster. |
| `--lpis_vector` | string | `None` | Optional explicit path to official LPIS parcel vector file (`.shp`, `.gpkg`). |
| `--slic_segment_ha` | float | `None` | Target superpixel parcel area in hectares for SLIC (adaptive: 1.8 ha for PT/ES/PL, 3.0 ha for NL/FR/DE). |
| `--slic_compactness` | float | `0.05` | SLIC boundary compactness (default: 0.05, elastic adherence to landscape boundaries). |
| `--slic_rag_thresh` | float | `0.02` | Region Adjacency Graph (RAG) spectral fusion distance threshold for SLIC. |
| `--enable_slic_rag` | flag | `False` | Enable experimental Region Adjacency Graph (RAG) spectral fusion pass for SLIC. |
| `--overwrite` | flag | `False` | Force re-generation of feature vectors, model training, and classification maps. |

### Merger runner (`run_merge.py`)

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-c, --country` | string | **Required** | Country code (e.g. `NL`, `PL`, `PT`, `FR`, `ES`, `DE`). |
| `--classifier` | choice | `auto` | Model suffix to merge: `auto`, `mlpxgb_presto`, `mlpxgb_presto_s1s2`, `s1s2`, `presto_s1`, `otb`. |
| `--seg_mode` | choice | `slic` | Segmentation mode: `slic` (superpixels), `sam` (Meta AI), `lpis` (cadastre). |
| `--method` | choice | `confidence` | Blending method across overlapping tracks: `confidence`, `priority`, `majority`. |

---

## Complete output products & intermediate artifacts

All classification outputs are stored sequentially by stage in `workingDirs/{COUNTRY}/orbit_{ORBIT}/2_classification/`:

| Stage | Directory | Output file name pattern | Format / Type | Purpose & description |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_s1_composite.tif` | `Float32`, 1 b. | Multi-temporal mean SAR amplitude composite for segmentation. |
| **1** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_data_footprint.tif` | `Byte`, 1 b. | Binary mask (1=valid, 0=NoData) of S1, S2, and NUTS intersection. |
| **2** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_segmentation_{MODE}.tif` | `UInt32`, 1 b. | Object segmentation raster with unique integer `segment_id` per parcel. |
| **3** | `1_samples_and_features/` | `learn_{MODE}.shp` | Vector Points | Stratified 70% training points with spatial attributes. |
| **3** | `1_samples_and_features/` | `control_{MODE}.shp` | Vector Points | Stratified 30% independent validation points. |
| **4** | `1_samples_and_features/` | `{COUNTRY}_{ORBIT}_features_{MODE}.csv` | Tabular CSV | Extracted S1 temporal statistics, S2 reflectances, and 128d Presto tokens. |
| **4** | `1_samples_and_features/` | `features_scaler.pkl` | Pickle Checkpoint | Serialized feature standardization scaler (`StandardScaler`). |
| **5** | `2_models/` | `{COUNTRY}_{ORBIT}_model_{MODE}.pkl` | Pickle Checkpoint | Serialized PyTorch Deep MLP weights + XGBoost GBDT trees + label encoder. |
| **5** | `2_models/` | `presto_encoder.pt` | PyTorch Tensor | Pre-trained Presto transformer feature extractor weights. |
| **6** | `3_maps/` | `{COUNTRY}_{ORBIT}_classified_{MODE}.tif` | `UInt16`, 1 b. | Raw pixel crop classification raster (pixel value = numeric `crop_id`). |
| **6** | `3_maps/` | `{COUNTRY}_{ORBIT}_confidence_{MODE}.tif` | `Float32`, 1 b. | Softmax ensemble probability confidence map (0.0 to 1.0). |
| **7** | `3_maps/` | `*_classified_masked_{MODE}.tif` | `UInt16`, 1 b. | **Final Single-Orbit Crop Map**: Cropland-masked & footprint-clipped GeoTIFF. |
| **7** | `3_maps/` | `*_confidence_masked_{MODE}.tif` | `Float32`, 1 b. | **Final Single-Orbit Confidence Map**: Cropland-masked confidence GeoTIFF. |
| **8** | `4_reports/` | `report_{COUNTRY}_{ORBIT}_metrics_{MODE}.xlsx` | Styled Excel | Accuracy workbook: Overall Accuracy, Cohen's Kappa, Confusion Matrix, F1-scores. |
| **Merge** | `national_products/` | `{COUNTRY}_national_crop_map{suffix}.tif` | `Int32`, 1 b. | **Seamless National Crop Map**: Confidence-blended multi-orbit BigTIFF. |
| **Merge** | `national_products/` | `{COUNTRY}_final_metrics{suffix}.xlsx` | Styled Excel | Aggregated nationwide statistical validation report. |
