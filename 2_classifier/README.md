# Multimodal machine learning classifier & national merger toolbox

This toolbox provides object-based image analysis (OBIA) crop type classification using multimodal fusion of **Sentinel-1 SAR** and **Sentinel-2 optical** time series, **NASA Harvest Presto foundation model embeddings**, deep neural networks, gradient boosted decision trees, and Bayesian statistical prior calibration.

---

## Architecture overview

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

---

## File structure

* **`run_classifier.py`**: Unified master CLI runner and interactive terminal wizard for Stages 0 through 7.
* **`run_merge.py`**: Multi-orbit nationwide confidence mosaic merger (Phase 4).
* **`modules/`**: Core classification engines:
  * `classifier_mlpxgb_presto.py`: Multimodal fusion ensemble combining NASA Harvest Presto transformer embeddings, PyTorch Deep MLP, and XGBoost GBDT (`[S1 + S2] [SOTA]`).
  * `classifier_presto_s1.py`: Single-radar Presto artificial neural network for SAR-only classification (`[S1 only]`).
  * `classifier_otb.py`: Orfeo ToolBox machine learning models (Random Forest / Support Vector Machines) (`[S1 + S2]`).
  * `multi_orbit_merger.py`: Nationwide multi-orbit confidence merger and morphological sieve post-processing.
  * `presto_model.py`: Embedded NASA Harvest Presto transformer foundation architecture.

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
