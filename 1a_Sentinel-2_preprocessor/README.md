# Sentinel-2 optical preprocessor toolbox

This toolbox provides an automated processing pipeline for **Sentinel-2 multispectral optical** Level-2A (Surface Reflectance / Bottom-Of-Atmosphere) imagery. It constructs cloud-free, regular 10-day synthetic optical time-series composites matching 1:1 with the Sentinel-1 SAR spatial grid.

---

## Scientific methodology & optical remote sensing

### 1. Bottom-of-atmosphere (BOA) surface reflectance
Sentinel-2 Level-2A products provide physically corrected surface reflectances (BOA) by removing Rayleigh scattering, aerosol optical depth, and atmospheric gas absorption (ozone and water vapor), providing calibrated ground surface reflectances.

### 2. Spectral bands and agricultural biophysical indicators
The pipeline utilizes 9 spectral bands at 10 m and 20 m spatial resolutions (resampled to 10.0 m in `EPSG:3857`):
* **Visible bands (`B02` Blue 490 nm, `B03` Green 560 nm, `B04` Red 665 nm)**: Sensitive to photosynthetic chlorophyll absorption.
* **RedEdge bands (`B05` 705 nm, `B06` 740 nm, `B07` 783 nm)**: Capture the steep reflectance transition edge; highly sensitive to canopy nitrogen, leaf chlorophyll concentration, and early senescence.
* **Narrow NIR (`B8A` 865 nm)**: Measures internal leaf mesophyll cellular scattering, avoiding atmospheric water vapor absorption present in broad `B08`.
* **Shortwave Infrared (`B11` 1610 nm, `B12` 2190 nm)**: Sensitive to foliar water content and dry matter accumulation.

### 3. Narrow-band agricultural indices
The pipeline calculates key vegetation and moisture indices:
* **Normalized Difference Vegetation Index (NDVI)**:
  `NDVI = (B8A - B04) / (B8A + B04)`
* **Red-Edge Chlorophyll Indices (NDRE1 & NDRE2)**:
  `NDRE1 = (B06 - B05) / (B06 + B05)`
  `NDRE2 = (B07 - B05) / (B07 + B05)`
* **Normalized Difference Water Index (NDWI / NDII)**:
  `NDWI = (B8A - B11) / (B8A + B11)`

### 4. Standardized agricultural DOY time series & regional phenological profiles
Satellite observations across years and orbits have variable revisit dates due to cloud cover. The pipeline solves this by interpolating observations into **14 standardized 10-day agricultural reference dates (Day of Year - DOY)** using regional phenological profiles tailored to European biomes:
* **Mediterranean profile (`PT`, `ES`, `IT`, `EL`, `CY`, etc.)**:
  DOYs: `[45, 65, 85, 105, 120, 135, 150, 165, 180, 200, 220, 240, 260, 280]`
  Captures early February green-up (DOY 45 ~ Feb 14), early cereal heading (DOY 105-135), summer drought senescence (DOY 180-220), and autumn emergence (DOY 260-280).
* **Continental & Central European profile (`PL`, `NL`, `DE`, `FR`, etc.)**:
  DOYs: `[80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]`
  Tailored to spring green-up (DOY 80 ~ March 21), peak summer canopy closure (June-July), and late harvest (September-October).
* **Northern European profile (`SE`, `FI`, `EE`, `LT`, `LV`, `NO`)**:
  DOYs: `[105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300]`
  Adjusted for late snowmelt, short intense boreal growing seasons, and late autumn harvests.

### 5. Whittaker penalized least-squares smoother (`--enable_whittaker`)
Residual cloud spikes, shadow dips, and atmospheric fluctuations can distort temporal NDVI trajectories. The pipeline incorporates a vectorized, robust Whittaker smoother across the DOY sequence. It balances observation fidelity with a curvature smoothness penalty (`lambda = 15.0`), suppressing cloud and shadow artifacts while preserving true crop phenological transitions.

### 6. High-throughput ZSTD compression & temporal stack naming
* **Zstandard (ZSTD) level 3 with horizontal predictor (`PREDICTOR=2`)**: Multi-temporal BigTIFF mosaics utilize high-speed ZSTD compression, yielding 35% to 45% smaller file sizes than classic DEFLATE with significantly faster decompression rates during machine learning feature extraction.
* **Standardized temporal acquisition signature**: Output optical stacks include full temporal acquisition boundaries matching the Sentinel-1 SAR standard:
  `{COUNTRY}_{END_DATE}_{START_DATE}_S2_timeseries.tif` (e.g. `PT_20251007_20241015_S2_timeseries.tif`) with an instant hardlink alias to `{COUNTRY}_S2_timeseries.tif` for complete backward compatibility.

---

## Processing flowchart & optical pipeline architecture

```
+----------------------------------------------------------------------------------------------------+
|                         SENTINEL-2 OPTICAL PREPROCESSING PIPELINE FLOWCHART                        |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Input data ingestion: shared country S2 pool]           |
                     | - Destination: workingDirs/{COUNTRY}/S2/                 |
                     | - Copernicus CDSE OData API (Automated L2A download)     |
                     | - CreoDIAS Local Archive (Fast L2A SAFE extraction)      |
                     | -> All country MGRS tiles downloaded once per country    |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 1: Granule extraction and SCL cloud masking]      |
                     | 1. Extract 9 spectral bands (B02-B12) at 10m/20m         |
                     | 2. Query Scene Classification Layer (SCL)                |
                     | 3. Mask invalid pixels: Clouds (8,9), Cirrus (10),       |
                     |    Cloud Shadows (3), Snow (11), Saturated/NoData (0,1)  |
                     | 4. Export clean masked GeoTIFFs to {TILE}_tif/           |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 2: 14-DOY synthetic time-series interpolation]    |
                     | 1. Parallel multi-core time-series interpolation         |
                     | 2. Linear temporal spline across valid clear DOYs        |
                     | 3. Interpolate 9 bands + dynamic NDVI for 14 DOYs        |
                     | 4. Saved to workingDirs/{COUNTRY}/S2/{TILE}/_synthetic_s2|
                     | -> Computed once per country tile (eliminates redundancy)|
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 3: Per-orbit SAR grid alignment & BigTIFF stacking|
                     | 1. Discovers synthetic tiles from shared country pool    |
                     | 2. Spatial alignment to Sentinel-1 SAR reference grid    |
                     |    (Exact sub-pixel match: dX = 0.000m, dY = 0.000m)     |
                     | 3. Clip to target orbit footprint in EPSG:3857 at 10.0m  |
                     | 4. Export 126-band BigTIFF with DEFLATE/ZSTD compression |
                     | 5. Build 6 multi-scale pyramid overview levels           |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Output product: 1_input_stacks/]                        |
                     | {COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif                |
                     | (126-band multi-temporal optical BigTIFF, EPSG:3857, 10m)|
                     +----------------------------+-----------------------------+
```

---

## Processing stages in detail

### Country-level shared repository design (`workingDirs/{COUNTRY}/S2/`)
Sentinel-2 optical observations follow fixed geographic MGRS tiling grids (`29SMA`, `31UDR`, etc.) that cover an entire nation regardless of radar satellite track geometry. 

To eliminate hundreds of gigabytes of duplicate downloads and redundant time-series computations across overlapping Sentinel-1 SAR orbits:
* **Stage 1 & Stage 2 operate at the country level**: Data is ingested and interpolated into `workingDirs/{COUNTRY}/S2/` **exactly once per nation**.
* **Stage 3 operates per orbit**: Stacking warps and clips from the shared national pool directly into each orbit track's `1_input_stacks/{COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif`.

### Stage 1: Granule extraction and SCL cloud masking (`s2_download_cdse.py`, `s2_extract_creodias.py`)
* Extracts Sentinel-2 L2A BOA surface reflectance granules directly into the shared repository `workingDirs/{COUNTRY}/S2/{TILE}_tif/`.
* **Native multi-resolution extraction (10 m + 20 m)**: Automatically extracts visible bands (`B02`, `B03`, `B04`) from `R10m` at their full native 10 m spatial resolution (`10980 x 10980` px) when available, while extracting Red-Edge, NIR, and SWIR bands (`B05`, `B06`, `B07`, `B8A`, `B11`, `B12`) and `SCL` from `R20m` (`5490 x 5490` px).
* Evaluates the Scene Classification Layer (SCL) and applies strict pixel-level masking:
  * **Retained classes (valid)**: `2` (Dark area pixels), `4` (Vegetation), `5` (Bare soil), `6` (Water), `7` (Unclassified).
  * **Masked classes (invalid)**: `0` (No data), `1` (Saturated / defective), `3` (Cloud shadows), `8` (Cloud medium probability), `9` (Cloud high probability), `10` (Thin cirrus), `11` (Snow / ice).
* **Incremental skip**: Scans local directories before issuing download requests; already converted granules are skipped instantly.

### Stage 2: 14-DOY synthetic time-series interpolation (`s2_time_series.py`)
* Implements multi-core parallel temporal interpolation in pure Python/NumPy across `workingDirs/{COUNTRY}/S2/{TILE}/_synthetic_s2/`.
* **Resolution-aware dynamic SCL resampling & bounded caching**: Dynamically adapts to each spectral band's native grid (`10980 x 10980` for 10 m bands or `5490 x 5490` for 20 m bands), resampling the 20 m `SCL` cloud mask on-the-fly with a bounded LRU cache (`<= 24` masks) to keep system RAM consumption minimal.
* For each pixel, reconstructs missing cloud-covered observations using forward/backward temporal linear interpolation between nearest cloud-free dates across the agricultural calendar, followed by optional Whittaker penalized least-squares temporal smoothing (`--enable_whittaker`).
* Automatically skips tiles that already have all 14 DOYs completed.

### Stage 3: SAR grid matching and BigTIFF stacking (`s2_mosaic_stack.py`)
* **Shared pool sourcing**: Automatically locates synthetic tiles in `workingDirs/{COUNTRY}/S2/` (with fallback to legacy track folders).
* **Sub-pixel geometric co-registration**: Warps optical mosaics to the exact spatial extent, bounding box, and pixel grid of the target Sentinel-1 SAR stack (`dX = 0.000 m, dY = 0.000 m` at 10.0 m resolution), guaranteeing zero spatial shift during machine learning feature fusion.
* **BigTIFF & pyramid generation**: Compiles all 126 layers into a single compressed BigTIFF (`COMPRESS=DEFLATE`, `TILED=YES`) and builds external pyramid overviews (`[2, 4, 8, 16, 32, 64]`) for smooth visualization.

---

## File and directory structure

* **`run_s2_preprocessor.py`**: Master CLI runner and interactive terminal wizard.
* **`config_s2.json`**: Active configuration containing CDSE credentials, spectral bands, DOYs, and working directories.
* **`config_s2.example.json`**: Template configuration file.
* **`modules/`**:
  * `s2_download_cdse.py`: Automated retrieval and SCL cloud masking from CDSE API.
  * `s2_extract_creodias.py`: Direct extraction from local CreoDIAS archive.
  * `s2_time_series.py`: 14-DOY synthetic time-series interpolation.
  * `s2_mosaic_stack.py`: SAR grid alignment, 126-band BigTIFF creation, and pyramid overview generation.
  * `s2_pipeline.py`: Object-oriented pipeline orchestrator.

---

## Execution commands

### 1. Interactive wizard mode (zero-argument launch)
```bash
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py
```

### 2. Country-wide automated execution (CreoDIAS local archive vs CDSE API)
```bash
# Full automated run for Portugal from local CreoDIAS /eodata archive (8 parallel threads):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT --source creodias -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Full automated run for Portugal downloading from Copernicus CDSE API:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT --source cdse -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Full automated run for the Netherlands (auto-detects local CreoDIAS or CDSE):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country NL --source auto -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Full automated run for Poland with Whittaker temporal smoothing:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PL --source creodias -s 2024-10-15 -e 2025-09-15 --enable_whittaker --stage A --threads 8
```

### 3. Single orbit track & individual stage execution
```bash
# Process single orbit track end-to-end:
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --track PT/orbit_52 --source creodias -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Stage 1 only (granule download/extraction and SCL cloud masking):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT --source cdse -s 2024-10-15 -e 2025-09-15 --stage 1 --threads 8

# Stage 2 only (14-DOY synthetic time-series interpolation across country MGRS tiles):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --country PT --stage 2 --threads 8

# Stage 3 only (per-orbit SAR grid alignment, 126-band BigTIFF stacking & pyramid overviews):
python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --track PT/orbit_52 --stage 3 --threads 8
```

---

## Command-line arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-t, --track` | string | `None` | Satellite track identifier (e.g. `PT/orbit_52`, `NL/orbit_88`). |
| `-c, --country` | string | `None` | Country code (e.g. `PT`, `NL`, `PL`, `ES`, `FR`, `DE`). |
| `-o, --orbit` | int | `None` | Specific relative orbit number (e.g. `52`, `125`, `88`). |
| `-s, --start_date` | string | `None` | Start date of agricultural season (`YYYY-MM-DD`, required for Stage 1/A). |
| `-e, --end_date` | string | `None` | End date of agricultural season (`YYYY-MM-DD`, required for Stage 1/A). |
| `--cloud_cover` | float | `80.0` | Maximum allowable scene cloud cover percentage (0.0 to 100.0). |
| `--source` | choice | `auto` | Data source: `auto` (auto-detect local repo), `creodias` (local archive), or `cdse` (Copernicus API). |
| `--threads` | int | `8` | Number of parallel worker threads for multi-temporal interpolation. |
| `--stage` | string | `None` | Stage to execute: `A` (all stages), `1` (download/extract), `2` (time-series), `3` (mosaic & stack). |
| `--doys` | int list | `auto` | List of target DOYs (default: auto-detected regional profile). |
| `--enable_whittaker` | flag | `True` | Enable penalized least-squares Whittaker smoothing across DOYs (default: True). |
| `--no_whittaker` | flag | `False` | Disable Whittaker smoothing (use raw linear interpolation). |
| `--whittaker_lambda` | float | `15.0` | Smoothing penalty parameter lambda for Whittaker filter (default: 15.0). |
| `--overwrite` | flag | `False` | Force recomputation of already existing intermediate and final stacks. |

---

## Transient disk space & cleanup guidelines

Sentinel-2 preprocessing creates intermediate country tile extractions and mosaics:
* **`workingDirs/{COUNTRY}/S2/`** (~100 to 150 GB per country): Extracted tiles and DOY synthetic mosaics.
* **`1_input_stacks/`** (~120 to 180 GB per orbit): Final 126-band BigTIFF stack.

> [!TIP]
> Once Stage 3 completes and `{COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif` is verified in `1_input_stacks/`, intermediate uncompressed files in `_temp_processing/` can be safely deleted to recover disk space.
