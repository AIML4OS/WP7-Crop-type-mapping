# Multimodal machine learning classifier & national merger toolbox

This toolbox provides an enterprise-grade, object-based image analysis (OBIA) crop type classification suite powered by multimodal fusion of **Sentinel-1 SAR** and **Sentinel-2 optical** time series, **NASA Harvest Presto geospatial foundation model embeddings**, deep neural networks, gradient boosted decision trees, and Bayesian statistical prior calibration.

---

## Scientific methodology & artificial intelligence architecture

### 1. NASA Harvest Presto geospatial foundation model
**Presto** is a state-of-the-art transformer foundation model pretrained on global, multi-sensor Earth observation time series:
* **Transformer encoder architecture**: Operates across irregular multi-temporal sequences using self-attention mechanisms to capture complex crop phenological dynamics and vegetative life cycles.
* **Multi-sensor tokenization**: Encodes raw Sentinel-1 radar backscatter ($\sigma^0_{VV}, \sigma^0_{VH}$) and Sentinel-2 optical reflectances ($B02$–$B12$) into unified high-dimensional latent space.
* **Spatial & temporal position encodings**: Integrates continuous geographic coordinates $(\text{latitude}, \text{longitude})$ through sinusoidal positional embeddings and seasonal acquisition month tokens, providing geographic and climatological context.
* **Latent token extraction**: Generates **128-dimensional multi-temporal token embeddings** from S1 SAR sequences and **128-dimensional embeddings** from S2 optical sequences, providing representations that surpass standard handcrafted index statistics.

### 2. Vision foundation model: Meta AI SAM (Segment Anything)
* Utilizes deep Vision Transformer (ViT) backbones (`vit_h`, `vit_l`, `vit_b`) adapted for satellite remote sensing.
* Generates field parcel boundaries from multi-temporal composite rasters with sub-pixel boundary precision, bilateral edge preservation, and distance transform hole-filling.

### 3. Unified PyTorch Deep MLP + XGBoost fusion ensemble
* **PyTorch Deep MLP**: 3-layer neural network with Batch Normalization (`BatchNorm1d`), Dropout ($p=0.3$), Class-weighted Cross-Entropy loss, and Cosine Annealing learning rate schedule:
  $$\mathcal{L} = -\sum_{c=1}^C w_c \cdot y_c \cdot \log(p_c)$$
* **XGBoost GBDT**: Ensemble of 250 gradient boosted decision trees (`max_depth=6`, `subsample=0.8`, `colsample_bytree=0.25`) with histogram-based splitting (`tree_method='hist'`).
* **Soft-voting probability blend**: Combines predicted class probability distributions from both models:
  $$\hat{P}(C_i | X) = 0.65 \cdot P_{\text{MLP}}(C_i | X) + 0.35 \cdot P_{\text{XGB}}(C_i | X)$$

### 4. Bayesian prior probability calibration
Standard machine learning models trained on balanced samples overestimate rare crops and underestimate dominant crops in real landscapes. The pipeline resolves this by applying Bayesian prior probability calibration:

$$P_{\text{calibrated}}(C_i | X) = \frac{P_{\text{model}}(C_i | X) \cdot \left(\frac{P_{\text{true}}(C_i)}{P_{\text{train}}(C_i)}\right)^\gamma}{\sum_{j=1}^K P_{\text{model}}(C_j | X) \cdot \left(\frac{P_{\text{true}}(C_j)}{P_{\text{train}}(C_j)}\right)^\gamma}$$

Where:
* $P_{\text{model}}(C_i | X)$ is the soft-voting ensemble prediction probability.
* $P_{\text{true}}(C_i)$ is the true statistical crop area proportion obtained from national agricultural registries (`priors.json`).
* $P_{\text{train}}(C_i)$ is the training sample proportion.
* $\gamma = 0.7$ is the calibration damping exponent preventing extreme boundary distortions.

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

## Segmentation modes

1. **`slic` (Simple Linear Iterative Clustering - Recommended for general operational use)**:
   - High-speed superpixel generation on multi-temporal composite rasters.
   - Buffered tiling (`buffer=64px`) prevents edge artifacts across block boundaries.
2. **`sam` (Meta AI Segment Anything Model - Deep foundation segmentation)**:
   - Vision Transformer architectures (`vit_h`, `vit_l`, `vit_b`) adapted for remote sensing.
   - Multi-process parallel inference (8 worker pool) with bilateral filtering and distance transform hole-filling.
3. **`lpis` (Official Cadastral Parcel Vectors - Cadastral ground truth)**:
   - Bounding-box spatial query (`pyogrio` bbox query) loads only relevant intersecting parcels from multi-gigabyte vector files.
   - Converts geometries to 2D polygons, reprojects to `EPSG:3857`, and applies GDAL rasterization.

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

## Output products

All intermediate and final classification outputs are organized by lineage in `workingDirs/{COUNTRY}/orbit_{ORBIT}/2_classification/`:

* **`0_segmentation/`**:
  * `{COUNTRY}_{ORBIT}_data_footprint.tif`: Binary valid observation mask.
  * `{COUNTRY}_{ORBIT}_segmentation_{SEG_MODE}.tif`: Labeled polygon segment ID raster.
* **`1_samples_and_features/`**:
  * `learn_{SEG_MODE}.shp`: 70% stratified training sample points.
  * `control_{SEG_MODE}.shp`: 30% independent validation sample points.
  * `*_features_{SEG_MODE}.csv`: Extracted feature table (S1 stats + S2 reflectances + Presto 128d embeddings).
* **`2_models/`**:
  * `*_model_{SEG_MODE}.pkl`: Serialized trained model checkpoint (scaler, MLP weights, XGBoost trees, class mappings).
* **`3_maps/`**:
  * `*_classified_{SEG_MODE}.tif`: Raw object classification raster.
  * `*_confidence_{SEG_MODE}.tif`: Classification probability confidence map ($0.0$ to $1.0$).
  * `*_classified_masked_{SEG_MODE}.tif`: Final cropland-masked crop classification raster.
  * `*_confidence_masked_{SEG_MODE}.tif`: Final cropland-masked confidence map.
* **`4_reports/`**:
  * `report_*.xlsx`: Styled Excel validation report containing Overall Accuracy (OA), Cohen's Kappa ($\kappa$), full confusion matrix, and per-crop Precision, Recall, and F1-scores.
* **`national_products/` (after `run_merge.py`)**:
  * `{COUNTRY}_national_crop_map_{SEG_MODE}.tif`: Seamless national crop type GeoTIFF.
  * `{COUNTRY}_national_confidence_{SEG_MODE}.tif`: Seamless national confidence GeoTIFF.
  * `{COUNTRY}_national_accuracy_report_{SEG_MODE}.xlsx`: Comprehensive national statistical report.
