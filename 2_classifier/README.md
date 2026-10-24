# Multimodal machine learning classifier & national merger toolbox

This toolbox provides an enterprise-grade, object-based image analysis (OBIA) crop type classification suite powered by multimodal fusion of **Sentinel-1 C-band SAR**, **Sentinel-2 multispectral optical**, and optional **NISAR L-band SAR (`L2 GCOV`)** time series (with **automatic seamless fallback to `Sentinel-1 + Sentinel-2`** whenever NISAR data is not present), **NASA Harvest Presto geospatial foundation model embeddings**, deep neural networks, gradient boosted decision trees, and Bayesian statistical prior calibration.

---

## Scientific methodology & artificial intelligence architecture

### 1. Multimodal dual-tier feature fusion architecture

The classification suite combines handcrafted physical remote sensing features with self-supervised geospatial foundation model latent representations across two or three sensors (`Sentinel-1 + Sentinel-2`, automatically upgrading to `Sentinel-1 + Sentinel-2 + NISAR` when a NISAR L-band stack is present in `1_input_stacks/`):

```
+----------------------------------------------------------------------------------------------------+
|                             MULTIMODAL DUAL-TIER FEATURE FUSION PIPELINE                           |
+----------------------------------------------------------------------------------------------------+
|  [Tier 1: Physical Handcrafted Remote Sensing Features]                                             |
|  - S1 C-Band Temporal Stats (VH, VV, VH/VV): Linear-power Mean (dB), CV Texture, RVI, CR, Delta_VH |
|  - S2 Multi-Spectral Reflectances: 14 DOYs * 9 Bands (B02-B12) = 126 spectral temporal features     |
|  - S2 Biophysical Indices: Multi-temporal NDVI, NDRE1 (B06/B05), NDRE2 (B07/B05), NDWI & texture    |
|  - [Optional] NISAR L-Band Stats (HH, HV, HH/HV): RVI_L, CR_L, Delta_HV_L & Dual-Freq L-vs-C Ratios|
|                                                  |                                                 |
|  [Tier 2: NASA Harvest Presto Geospatial Foundation Model Latent Embeddings (384D / 512D)]          |
|  - Joint S1+S2 Cross-Attention Representation: 128d embedding (full S1 timeline + aligned S2 DOYs)  |
|  - S1 C-Band SAR Transformer Representation:   128d full-time-series token embedding                |
|  - S2 Optical Transformer Representation:      128d multi-temporal token embedding                  |
|  - [Optional] NISAR L-Band SAR Representation: 128d multi-temporal L-band token embedding           |
|                                                  |                                                 |
|  [Concatenated High-Dimensional Feature Vector X (~550+ features, or ~700+ with NISAR)]             |
|  X = [ F_S1_stats || F_S2_spectral || F_phys_indices (|| F_NISAR_L) || E_Presto (384d / 512d) ]    |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                      HYBRID MULTI-PARADIGM ENSEMBLE CLASSIFICATION SUITE                           |
+----------------------------------------------------------------------------------------------------+
|  [Branch A: PyTorch Deep MLP]                 |  [Branch B: XGBoost GBDT Engine]                   |
|  - 3 Dense Layers (512 -> 256 -> 128)         |  - 350 Gradient Boosted Decision Trees             |
|  - BatchNorm1d & Dropout (p=0.3)              |  - Histogram-based splitting (tree_method='hist')  |
|  - Class-Weighted Cross-Entropy Loss          |  - Subsample=0.85, Colsample_bytree=0.85           |
|  - Cosine Annealing Learning Rate Schedule    |  - Max depth = 7, learning rate = 0.08             |
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
* **Sentinel-1 SAR temporal moments**: For each polarization (VV, VH) and cross-ratio (VH/VV), the pipeline extracts statistical moments across the agricultural calendar (aggregated in the physical linear power domain before converting back to dB):
  Mean, Coefficient of Variation (CV = Std / Mean), Minimum (Min), Maximum (Max), and Dynamic Range (Max - Min).
  These metrics capture surface roughness, canopy closure speed, and abrupt drops in backscatter caused by harvesting.
* **Specialized SAR phenological indices (RVI & Dual-Pol vegetation metrics)**:
  `RVI = 4 * VH / (VV + VH)` and `CR = VH_dB - VV_dB`
  Tracks canopy volume fraction and vegetative biomass accumulation independent of soil surface dielectric variations.
* **Seasonal SAR backscatter dynamics (Delta_VH & Delta_VV)**:
  `Delta_VH = VH_summer - VH_spring`
  Differentiates winter cereals (established autumn canopy) from spring-sown crops (bare soil in early spring transitioning to lush vegetation by mid-summer).
* **Intra-object texture variance & dispersion (field-level std & CV)**:
  Captures internal parcel spatial heterogeneity: scale-invariant SAR Coefficient of Variation (`CV`) and optical intra-parcel standard deviation (`s2_intra_std`, `ndvi_intra_std`). Distinguishes dense uniform field canopies (wheat, barley) from wide-row crops (maize, sugar beets) and structured woody vegetation (vineyards, orchards, olive groves).
* **Sentinel-2 multi-spectral trajectories & Red-Edge indices**: Standardized 14-DOY reflectances across 9 spectral bands (14 x 9 = 126 features) plus biophysical time-series vegetation indices (`NDVI = (B8A - B04)/(B8A + B04)`, `NDRE1 = (B06 - B05)/(B06 + B05)`, `NDRE2 = (B07 - B05)/(B07 + B05)`, `NDWI = (B8A - B11)/(B8A + B11)`) tracking chlorophyll absorption, red-edge inflection shift across `B05`/`B06`/`B07`, and canopy moisture.
* **Optional NISAR L-band SAR & dual-frequency L/C-band canopy penetration features (with automatic `S1 + S2` fallback)**:
  When a co-registered NISAR L-band stack (`*_NISAR_GCOV_L_timeseries.tif`) is detected in `1_input_stacks/` (or supplied via `--nisar_raster`), the pipeline extracts:
  - L-band polarimetric moments (`nisar_hh_mean_dB`, `nisar_hv_mean_dB`, `nisar_hv_std`, `nisar_hh_hv_ratio_dB`),
  - L-band Radar Vegetation Index (`RVI_L = 4 * HV / (HH + HV)`), L-band cross-ratio (`CR_L = HV_dB - HH_dB`), and spring-to-summer biomass accumulation (`nisar_delta_hv_spring_summer`),
  - **Dual-frequency L-vs-C-band ratios (`nisar_s1_crosspol_diff_dB`, `nisar_s1_copol_diff_dB`, `nisar_s1_rvi_diff`)** comparing 24 cm L-band penetration with 5.6 cm C-band canopy scattering to separate woody/high-biomass crops (maize, rapeseed, orchards, vineyards, potatoes) from fine-stemmed cereals.
  - **Seamless fallback**: If no NISAR stack is present (or if a trained `.pkl` checkpoint was trained on `S1 + S2` only), the classifier automatically runs in standard `Sentinel-1 + Sentinel-2` mode with zero configuration changes.

#### Tier 2: NASA Harvest Presto 384-dimensional / 512-dimensional foundation embeddings
NASA Harvest Presto encodes multi-temporal sequences using self-attention transformer blocks, mapping seasonal dynamics into 128-dimensional latent vectors:
* `E_Joint` (128 dimensions): True multimodal cross-attention preserving the **full native Sentinel-1 radar time series (`T_s1` dates)** as the primary timeline and temporally interpolating the 14 cloud-free Sentinel-2 DOY composites onto the full S1 sequence.
* `E_S1` (128 dimensions): Full-resolution C-band SAR temporal dynamics.
* `E_S2` (128 dimensions): Optical multispectral phenology.
* `E_NISAR` (128 dimensions, optional): Full-series L-band SAR transformer embedding (`presto_nisar_0..127`) extracted when NISAR data is present.

#### Unified concatenated feature vector
The complete feature vector combines domain-specific physical interpretability with deep self-supervised representation learning into a unified vector:
`X_fused = [ F_S1_stats_texture || F_S2_spectral_texture || F_phys_indices (|| F_NISAR_L) || E_Presto_Joint || E_Presto_S1 || E_Presto_S2 (|| E_Presto_NISAR) ]`

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
   Assigns expected physical crop parcel sizes from an agronomic multi-lingual EU-27 & EuroCrops (`HCAT2`) taxonomy across 5 agronomic scale tiers (with country-specific fallback isolation and Web Mercator $\cos^2(\varphi)$ latitude area correction for SLIC superpixels and national hectarage reporting):
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
                     | [Stage 2: High-SNR Composites & Object Segmentation]     |
                     | 1. High-SNR SAR composite creation:                      |
                     |    - Primary: Seasonal SAR composite (April-August avg)  |
                     |      -> {COUNTRY}_{ORBIT}_sar_seasonal_composite.tif     |
                     |    - Fallback: Multi-temporal summed composite (all DOYs)|
                     |      -> {COUNTRY}_{ORBIT}_summed_composite.tif           |
                     | 2. Object boundary delineation & seam stitching:         |
                     |    - Mode 1: SLIC Superpixels (buffered tiles + DSU seam |
                     |      stitching + optional RAG spectral fusion)           |
                     |    - Mode 2: Meta AI SAM (Vision Transformer Foundation) |
                     |    - Mode 3: LPIS Cadastre (Official European GPKG/SHP)  |
                     | -> Output: {COUNTRY}_{ORBIT}_segmentation_{MODE}.tif     |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 3: Parcel-Grouped Stratified Sample Partitioning] |
                     | Groups sample points by parcel/segment ID (seg_id) to    |
                     | prevent spatial data leakage across train/control sets:  |
                     | - 70% Training / Learn ({COUNTRY}_{ORBIT}_learn_{MODE}.shp)
                     | - 30% Independent Control ({COUNTRY}_{ORBIT}_control_{MODE}.shp)
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 4: Multimodal Feature Extraction]                 |
                     | 1. S1 temporal backscatter stats in linear power domain  |
                     | 2. S2 surface reflectances (B02-B12) & dynamic NDVI      |
                     | 3. Physical Red-Edge (NDRE1/2) & SAR (RVI, CR, Delta_VH) |
                     | 4. Presto 384d (Joint S1->S2 aligned + S1 + S2) tokens   |
                     | -> Output: {COUNTRY}_{ORBIT}_mlpxgb_presto_learn_features_{MODE}.csv
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 5: Fusion Ensemble Training & Outlier Cleaning]   |
                     | 1. Robust IQR feature normalization & 2% outlier pruning |
                     | 2. Class-weighted PyTorch Deep MLP + XGBoost GBDT (350t) |
                     | 3. Soft-voting ensemble fit (65% MLP + 35% XGBoost)      |
                     | -> Output: {COUNTRY}_{ORBIT}_mlpxgb_presto_model_{MODE}.pkl
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 6: High-Performance Vectorized Inference]         |
                     | 1. Tile-level block I/O (Read full 2048x2048 tile once)  |
                     | 2. Vectorized np.bincount zonal stats across 170+ bands  |
                     | 3. Presto batched embedding forward pass & MLP+XGB infer |
                     | 4. Dynamic Bayesian prior calibration & O(1) LUT map     |
                     | -> Output: classified, confidence, entropy & margin maps |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 7: Cropland & Data Footprint Masking]             |
                     | Suppresses non-agricultural surfaces via national mask   |
                     | and applies morphological sieve filtering (<10 px)       |
                     | -> Output: classified_masked, confidence_masked,         |
                     |            entropy_masked & margin_masked GeoTIFFs       |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 8: Out-of-Bag Validation & Excel Reporting]       |
                     | Evaluates against independent 30% control dataset:       |
                     | - Overall Accuracy (OA) and Cohen's Kappa                |
                     | - Full confusion matrix (ground truth vs predictions)    |
                     | - Per-crop Precision, Recall, and F1-scores              |
                     | -> Output: report_{COUNTRY}_{ORBIT}_mlpxgb_presto_{MODE}.xlsx
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
                     | [Multi-Orbit Swath Overlap Blending (--method)]          |
                     | - confidence: Select pixel with max posterior confidence |
                     | - majority:   Majority vote across orbits + conf tiebreak|
                     | - priority:   First valid non-zero orbit in track order  |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Post-Processing, Pyramids & National Validation]        |
                     | 1. Cropland-masked seam sieve filter (slivers < 4 px)    |
                     | 2. Multi-scale pyramid overviews [2, 4, 8, 16, 32, 64]   |
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

### 1. Multimodal data footprint generation (`*_data_footprint.tif`)

Because Sentinel-1 SAR swaths follow tilted polar orbital geometries while Sentinel-2 optical granules follow UTM MGRS tiles, their operational swaths never completely align along peripheral boundaries. 

In Stage 1, the pipeline computes a **binary multimodal data footprint mask**:
* **Physical sensor intersection**: Calculates the exact pixel-wise binary intersection of valid Sentinel-1 radar data, valid Sentinel-2 optical data, and regional NUTS boundaries (1 = valid data, 0 = NoData).
* **Edge artifact elimination**: Strictly restricts downstream sample generation, feature extraction, and tile inference to spatial bounds where all sensor bands contain uncorrupted observations.
* **Prevents border classification anomalies**: Ensures that pixels with partial satellite coverage outside the orbit swath are properly masked as `NoData` (0).

### 2. High-SNR SAR composites for OBIA segmentation

Directly segmenting single-date SAR imagery is problematic due to multiplicative speckle noise inherent to coherent microwave imaging. To establish clean, stationary parcel boundaries, Stage 2 generates a dedicated 1-channel high-SNR SAR raster before executing superpixel (SLIC) or deep foundation (SAM) segmentation:

#### Engine 1: Seasonal SAR composite (`_create_seasonal_sar_composite`)
* **Output file**: `{COUNTRY}_{ORBIT}_sar_seasonal_composite.tif`
* **Seasonal windowing**: Specifically isolates Sentinel-1 radar acquisitions during peak vegetative growth months (**April to August**: `apr`, `may`, `jun`, `jul`, `aug`).
* **Canopy structural contrast**: During April–August, agricultural crop canopies reach maximum biomass and structural differentiation against bare soil, drainage ditches, and farm tracks. Field margins are geometrically sharpest during this period.
* **Speckle suppression**: Multi-temporal temporal averaging across the selected seasonal dates reduces radar speckle variance by a factor of `1/N`, providing an optimal input raster for edge-sensitive clustering.

#### Engine 2: High-SNR summed composite (`_create_summed_composite`)
* **Output file**: `{COUNTRY}_{ORBIT}_summed_composite.tif`
* **Full-calendar temporal sum**: Computes a multi-temporal summation across all valid radar bands (both VV and VH polarizations) spanning the complete multi-month agricultural timeline.
* **Coherent signal accumulation**: Summing backscatter values in linear/log domain accumulates coherent surface backscatter while averaging out zero-mean incoherent phase noise, dramatically raising the Signal-to-Noise Ratio (SNR).
* **Robust fallback**: Automatically deployed whenever fewer than 4 valid acquisitions are available in the seasonal window, or in historical and SAR-only baseline architectures (`1_classify_ann.py`, `1_classify_presto_sar.py`, `1_classify_ann_presto_hybrid.py`).
* **Optical fallback**: If no SAR raster is present, the pipeline falls back directly to multi-spectral optical composites.

### 3. OBIA segmentation algorithms

#### Mode 1: SLIC (Simple Linear Iterative Clustering - recommended)
* Operates in a 5D feature space combining the high-SNR radiometric composite (seasonal or summed SAR composite) and 2D spatial coordinates.
* **Adaptive target parcel size**: Dynamically sets target parcel size based on country agronomic profile via `--slic_segment_ha` (default: 1.8 ha for Southern/Eastern Europe `PT`/`ES`/`IT`/`GR`/`PL`; 3.0 ha for Western Europe `NL`/`FR`/`DE`).
* **High-elasticity boundary tracking**: Compactness parameter (`--slic_compactness 0.05`) allows superpixel boundaries to conform precisely to physical parcel boundaries, hedgerows, and drainage canals.
* **Buffered tile processing**: Rasters are segmented in 2048 x 2048 blocks with a **64-pixel halo buffer**.
* **Disjoint Set Union (Union-Find) tile seam stitching (`_stitch_tile_seams`)**: Directly solves rectilinear grid edge artifacts by running an automated Union-Find boundary stitching pass across horizontal and vertical block seams (`x = 2048, 4096...` and `y = 2048, 4096...`). Adjacent segment labels on tile boundaries are seamlessly merged if they belong to the same continuous field parcel with consistent spectral properties.
* **Region Adjacency Graph (RAG) spectral fusion (`--enable_slic_rag`)**: Optional experimental post-segmentation fusion pass merging adjacent superpixels whose spectral distance falls below `--slic_rag_thresh` (default: 0.02).
* **32-bit integer raster**: Segments are output as a `UInt32` GeoTIFF (`*_segmentation_slic.tif`) where every discrete parcel receives a globally unique non-zero integer identifier (`segment_id`).

#### Mode 2: Meta AI SAM (Segment Anything Model)
* Adapts Meta AI Vision Transformers (`vit_h`, `vit_l`, `vit_b`) pre-trained on over 1 billion segmentation masks.
* Automatically distributes a structured prompt grid across composite rasters.
* Implements bilateral edge filtering and distance transform hole-filling to close internal parcel gaps and produce smooth field geometries.

#### Mode 3: LPIS cadastre (official European agricultural parcels)
* Ingests official vector databases (`.gpkg` or `.shp`) provided by national paying agencies (e.g., BRP in the Netherlands, ISIP in Portugal, ARiMR in Poland).
* Uses high-speed spatial bounding-box filtering (`pyogrio` with R-Tree spatial indexing) to extract intersecting parcels.
* Reprojects and rasterizes vectors into a 10.0 m `UInt32` grid in `EPSG:3857` and strictly masks with the multimodal data footprint.

---

## Execution commands

### 1. Interactive setup wizard (recommended)
```bash
python 2_classifier/run_classifier.py
```

### 2. SOTA multimodal classification (`[S1 + S2]`)
```bash
# SLIC superpixels (fast, fully automated):
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode slic --stage A

# Run classification for an entire country across all orbits:
python 2_classifier/run_classifier.py --country PT --classifier mlpxgb_presto --seg_mode slic --stage A

# Official cadastral parcel vectors (LPIS):
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode lpis --lpis_vector auxiliary_files/shapefiles_samples/PT/lpis.shp --stage A

# Meta AI SAM deep learning segmentation:
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier mlpxgb_presto --seg_mode sam --stage A
```

### 3. Single-radar classification (`[S1 only]`)
```bash
python 2_classifier/run_classifier.py --track PT/orbit_52 --classifier presto_s1 --seg_mode slic --stage A
```

### 4. Running specific single stages (1 to 8)
```bash
# Run only data footprint generation (Stage 1):
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 1

# Run only segmentation & composite creation (Stage 2):
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 2

# Run only sample point partitioning (Stage 3):
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 3

# Run only multimodal feature extraction (Stage 4):
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 4

# Run only model training (Stage 5):
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 5

# Run only vectorized inference with Bayesian priors (Stage 6):
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 6

# Run only cropland masking (Stage 7):
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 7

# Run only accuracy metrics and export Excel report (Stage 8):
python 2_classifier/run_classifier.py --track PT/orbit_52 --stage 8
```

### 5. Phase 4: Nationwide multi-orbit merge
```bash
# Merge SLIC classifications into a seamless national map with SOTA ensemble:
python 2_classifier/run_merge.py --country PT --classifier mlpxgb_presto --seg_mode slic

# Merge LPIS cadastral classifications for the Netherlands:
python 2_classifier/run_merge.py --country NL --classifier mlpxgb_presto --seg_mode lpis

# Merge Meta AI SAM classifications using confidence blending:
python 2_classifier/run_merge.py --country PT --classifier mlpxgb_presto --seg_mode sam --method confidence
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
| `--nisar_raster` | string | `None` | Optional explicit path override to NISAR L-band BigTIFF raster (`*_NISAR_GCOV_L_timeseries.tif`). Automatically falls back to `S1 + S2` if absent. |
| `--lpis_vector` | string | `None` | Optional explicit path to official LPIS parcel vector file (`.shp`, `.gpkg`). |
| `--slic_segment_ha` | float | `None` | Target superpixel parcel area in hectares for SLIC (adaptive: 1.8 ha for PT/ES/PL, 3.0 ha for NL/FR/DE). |
| `--slic_compactness` | float | `0.05` | SLIC boundary compactness (default: 0.05, elastic adherence to landscape boundaries). |
| `--slic_rag_thresh` | float | `0.02` | Region Adjacency Graph (RAG) spectral fusion distance threshold for SLIC. |
| `--enable_slic_rag` | flag | `False` | Enable experimental Region Adjacency Graph (RAG) spectral fusion pass for SLIC. |
| `--no_slic_rag` | flag | `False` | Explicitly disable Region Adjacency Graph (RAG) spectral fusion pass for SLIC. |
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
| **1** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_data_footprint.tif` | `Byte`, 1 b. | Binary mask (1=valid, 0=NoData) of S1, S2, and NUTS boundary intersection. |
| **2** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_sar_seasonal_composite.tif` | `Float32`, 1 b. | **Primary Seasonal SAR Composite**: April–August multi-temporal speckle-reduced backscatter. |
| **2** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_summed_composite.tif` | `Float32`, 1 b. | **Summed Composite**: Multi-temporal sum across all SAR bands (high-SNR fallback). |
| **2** | `0_segmentation/` | `{COUNTRY}_{ORBIT}_segmentation_{MODE}.tif` | `UInt32`, 1 b. | Object segmentation raster with unique integer `segment_id` per parcel. |
| **3** | `1_samples_and_features/` | `{COUNTRY}_{ORBIT}_learn_{MODE}.shp` | Vector Points | Stratified 70% training sample points with spatial attributes. |
| **3** | `1_samples_and_features/` | `{COUNTRY}_{ORBIT}_control_{MODE}.shp` | Vector Points | Stratified 30% independent validation sample points. |
| **4** | `1_samples_and_features/` | `{COUNTRY}_{ORBIT}_mlpxgb_presto_learn_features_{MODE}.csv` | Tabular CSV | Extracted S1 statistics, S2 reflectances, physical indices, and 128d Presto tokens (~450+ feats). |
| **5** | `2_models/` | `{COUNTRY}_{ORBIT}_mlpxgb_presto_model_{MODE}.pkl` | Pickle Checkpoint | Serialized PyTorch Deep MLP weights + XGBoost GBDT trees + label encoder + scaler. |
| **6** | `3_maps/` | `{COUNTRY}_{ORBIT}_classified_mlpxgb_presto_{MODE}.tif` | `UInt16`, 1 b. | Raw pixel crop classification raster (pixel value = numeric `crop_id`). |
| **6** | `3_maps/` | `{COUNTRY}_{ORBIT}_confidence_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 b. | Winning ensemble prediction probability confidence map (`0.0` to `1.0`). |
| **6** | `3_maps/` | `{COUNTRY}_{ORBIT}_entropy_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 b. | Normalized Shannon entropy spatial uncertainty map (`0.0` = certain, `1.0` = high confusion). |
| **6** | `3_maps/` | `{COUNTRY}_{ORBIT}_margin_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 b. | Prediction margin map (`P_first - P_second`, margin between top-2 crop candidates). |
| **7** | `3_maps/` | `{COUNTRY}_{ORBIT}_classified_masked_mlpxgb_presto_{MODE}.tif` | `UInt16`, 1 b. | **Final Track Crop Map**: Cropland-masked & footprint-clipped GeoTIFF with sieve filter. |
| **7** | `3_maps/` | `{COUNTRY}_{ORBIT}_confidence_masked_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 b. | **Final Track Confidence Map**: Cropland-masked confidence GeoTIFF. |
| **7** | `3_maps/` | `{COUNTRY}_{ORBIT}_entropy_masked_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 b. | **Final Track Entropy Map**: Cropland-masked spatial entropy uncertainty GeoTIFF. |
| **7** | `3_maps/` | `{COUNTRY}_{ORBIT}_margin_masked_mlpxgb_presto_{MODE}.tif` | `Float32`, 1 b. | **Final Track Margin Map**: Cropland-masked prediction margin GeoTIFF. |
| **8** | `4_reports/` | `report_{COUNTRY}_{ORBIT}_mlpxgb_presto_{MODE}.xlsx` | Styled Excel | Detailed validation report: Overall Accuracy, Cohen's Kappa, Confusion Matrix, F1-scores. |
| **Merge** | `national_products/` | `{COUNTRY}_national_crop_map{suffix}.tif` | `Int32`, 1 b. | **Seamless National Crop Map**: Confidence-blended multi-orbit BigTIFF. |
| **Merge** | `national_products/` | `{COUNTRY}_final_metrics{suffix}.xlsx` | Styled Excel | Aggregated nationwide statistical validation report. |

