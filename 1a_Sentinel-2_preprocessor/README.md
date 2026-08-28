# Sentinel-2 optical preprocessor toolbox

This toolbox provides an automated processing pipeline for **Sentinel-2 multispectral optical** Level-2A (Surface Reflectance / Bottom-Of-Atmosphere) imagery. It constructs cloud-free, regular 10-day synthetic optical time-series composites matching 1:1 with the Sentinel-1 SAR spatial grid.

---

## Scientific methodology & optical remote sensing

### 1. Bottom-Of-Atmosphere (BOA) Surface Reflectance
Sentinel-2 L2A data provides physically corrected surface reflectances $\rho_{\text{BOA}}(\lambda)$ by removing Rayleigh scattering, aerosol optical depth, and ozone/water absorption:

$$\rho_{\text{BOA}}(\lambda) = \frac{\pi \cdot (L_{\text{TOA}}(\lambda) - L_{\text{path}}(\lambda))}{\tau_v(\lambda) \cdot [E_0(\lambda) \cdot \cos \theta_s \cdot \tau_s(\lambda) + E_{\text{down}}(\lambda)]}$$

### 2. Spectral Bands & Agricultural Bio-Physical Indicators
The pipeline utilizes 9 spectral bands at 10 m and 20 m spatial resolutions (resampled to 10.0 m in `EPSG:3857`):
* **Visible bands (`B02` Blue 490 nm, `B03` Green 560 nm, `B04` Red 665 nm)**: Sensitive to photosynthetic chlorophyll $a$ and $b$ absorption.
* **RedEdge bands (`B05` 705 nm, `B06` 740 nm, `B07` 783 nm)**: Capture the steep reflectance transition edge; highly sensitive to canopy nitrogen, leaf chlorophyll concentration, and early senescence.
* **Narrow NIR (`B8A` 865 nm)**: Measures internal leaf mesophyll cellular scattering, avoiding atmospheric water vapor absorption present in broad `B08`.
* **Shortwave Infrared (`B11` 1610 nm, `B12` 2190 nm)**: Sensitive to foliar water content and dry matter accumulation.

### 3. Narrow-Band Agricultural Indices
* **Normalized Difference Vegetation Index (NDVI)**:
  $$\text{NDVI} = \frac{\text{B8A} - \text{B04}}{\text{B8A} + \text{B04}}$$
* **Red-Edge Chlorophyll Indices (NDRE1 & NDRE2)**:
  $$\text{NDRE1} = \frac{\text{B06} - \text{B05}}{\text{B06} + \text{B05}}, \quad \text{NDRE2} = \frac{\text{B07} - \text{B05}}{\text{B07} + \text{B05}}$$
* **Normalized Difference Water Index (NDWI / NDII)**:
  $$\text{NDWI} = \frac{\text{B8A} - \text{B11}}{\text{B8A} + \text{B11}}$$

### 4. Standardized Agricultural DOY Time-Series
Satellite observations across years and orbits have variable revisit dates due to cloud cover. The pipeline solves this by interpolating all observations into **14 standardized 10-day agricultural reference dates (Day of Year - DOY)**:
$$\text{DOYs} = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]$$
* DOY 80 (March 21): Early spring emergence / winter crop green-up.
* DOY 105–175 (April–June): Peak vegetative growth and canopy closure.
* DOY 189–231 (July–August): Flowering, grain filling, and harvest of summer crops.
* DOY 252–287 (September–October): Late harvest and autumn emergence.

---

## Processing flowchart & optical pipeline architecture

```
+----------------------------------------------------------------------------------------------------+
|                         SENTINEL-2 OPTICAL PREPROCESSING PIPELINE FLOWCHART                        |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Input Data Ingestion: Shared Country S2 Pool]           |
                     | - Destination: workingDirs/{COUNTRY}/S2/                 |
                     | - Copernicus CDSE OData API (Automated L2A download)     |
                     | - CreoDIAS Local Archive (Fast L2A SAFE extraction)      |
                     | -> All country MGRS tiles downloaded ONCE per country    |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 1: Granule Extraction & SCL Cloud Masking]        |
                     | 1. Extract 9 spectral bands (B02-B12) at 10m/20m         |
                     | 2. Query Scene Classification Layer (SCL)                |
                     | 3. Mask invalid pixels: Clouds (8,9), Cirrus (10),       |
                     |    Cloud Shadows (3), Snow (11), Saturated/NoData (0,1)  |
                     | 4. Export clean masked GeoTIFFs to {TILE}_tif/           |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 2: 14-DOY Synthetic Time-Series Interpolation]    |
                     | 1. Parallel multi-core time-series interpolation         |
                     | 2. Linear temporal spline across valid clear DOYs        |
                     | 3. Interpolate 9 bands + dynamic NDVI for 14 DOYs        |
                     | 4. Saved to workingDirs/{COUNTRY}/S2/{TILE}/_synthetic_s2|
                     | -> Computed ONCE per country tile (eliminates redundancy)|
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 3: Per-Orbit SAR Grid Alignment & BigTIFF Stacking|
                     | 1. Discovers synthetic tiles from shared country pool    |
                     | 2. Spatial alignment to Sentinel-1 SAR reference grid    |
                     |    (Exact sub-pixel match: dX = 0.000m, dY = 0.000m)     |
                     | 3. Clip to target orbit footprint in EPSG:3857 at 10.0m  |
                     | 4. Export 126-band BigTIFF with DEFLATE compression      |
                     | 5. Build 6 multi-scale pyramid overview levels           |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Output Product: 1_input_stacks/]                        |
                     | {COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif                |
                     | (126-band multi-temporal optical BigTIFF, EPSG:3857, 10m)|
                     +----------------------------------------------------------+
```

---

## Processing stages in detail

### Country-level shared repository design (`workingDirs/{COUNTRY}/S2/`)
Sentinel-2 optical observations follow fixed geographic MGRS tiling grids (`29SMA`, `31UDR`, etc.) that cover an entire nation regardless of radar satellite track geometry. 

To eliminate hundreds of gigabytes of duplicate downloads and redundant time-series computations across overlapping Sentinel-1 SAR orbits:
* **Stage 1 & Stage 2 operate at the country level**: Data is ingested and interpolated into `workingDirs/{COUNTRY}/S2/` **exactly once per nation**.
* **Stage 3 operates per orbit**: Stacking warps and clips from the shared national pool directly into each orbit track's `1_input_stacks/{COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif`.

### Stage 1: Granule extraction & SCL cloud masking (`s2_download_cdse.py`, `s2_extract_creodias.py`)
* Extracts Sentinel-2 L2A BOA surface reflectance granules directly into the shared repository `workingDirs/{COUNTRY}/S2/{TILE}_tif/`.
* Evaluates the Scene Classification Layer (SCL) and applies strict pixel-level masking:
  * **Retained classes (valid)**: `4` (Vegetation), `5` (Bare soil), `6` (Water), `7` (Unclassified).
  * **Masked classes (invalid)**: `0` (No data), `1` (Saturated / defective), `2` (Dark features), `3` (Cloud shadows), `8` (Cloud medium probability), `9` (Cloud high probability), `10` (Thin cirrus), `11` (Snow / ice).
* **Incremental skip**: Scans local directories before issuing download requests; already converted granules are skipped instantly.

### Stage 2: 14-DOY synthetic time-series interpolation (`s2_time_series.py`)
* Implements multi-core parallel temporal interpolation in pure Python/NumPy across `workingDirs/{COUNTRY}/S2/{TILE}/_synthetic_s2/`.
* For each pixel, reconstructs missing cloud-covered observations using forward/backward temporal linear interpolation between nearest cloud-free dates across the agricultural calendar.
* Automatically skips tiles that already have all 14 DOYs completed.

### Stage 3: SAR grid matching & BigTIFF stacking (`s2_mosaic_stack.py`)
* **Shared pool sourcing**: Automatically locates synthetic tiles in `workingDirs/{COUNTRY}/S2/` (with fallback to legacy track folders).
* **Sub-pixel geometric co-registration**: Warps optical mosaics to the exact spatial extent, bounding box, and pixel grid of the target Sentinel-1 SAR stack ($\Delta X = 0.000\text{ m}, \Delta Y = 0.000\text{ m}$ at 10.0 m resolution), guaranteeing zero spatial shift during machine learning feature fusion.
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
```powershell
python run_s2_preprocessor.py
```

### 2. Country-wide automated execution
```powershell
# Full automated run for Portugal (orbit 52 and orbit 125, 8 parallel threads):
python run_s2_preprocessor.py --country PT -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Full automated run for the Netherlands:
python run_s2_preprocessor.py --country NL -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Full automated run for Poland:
python run_s2_preprocessor.py --country PL -s 2024-10-15 -e 2025-09-15 --stage A --threads 8
```

### 3. Single orbit track execution
```powershell
# Process single orbit track:
python run_s2_preprocessor.py --track PT/orbit_52 -s 2024-10-15 -e 2025-09-15 --stage A --threads 8

# Run specific single stage (e.g. Stage 3 mosaic and stack only):
python run_s2_preprocessor.py --track PT/orbit_52 --stage 3 --threads 8
```

---

## Command-line arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-t, --track` | string | `None` | Satellite track identifier (e.g. `PT/orbit_52`, `NL/orbit_88`). |
| `-c, --country` | string | `None` | Country code (e.g. `PT`, `NL`, `PL`, `ES`, `FR`, `DE`). |
| `-s, --start_date` | string | `2024-10-15` | Start date of agricultural season (`YYYY-MM-DD`). |
| `-e, --end_date` | string | `2025-09-15` | End date of agricultural season (`YYYY-MM-DD`). |
| `--cloud_cover` | int | `80` | Maximum allowable tile cloud cover percentage (0 to 100). |
| `--source` | choice | `cdse` | Data source: `cdse` (Copernicus API) or `creodias` (local archive). |
| `--threads` | int | `8` | Number of parallel worker threads for multi-temporal interpolation. |
| `--stage` | string | `None` | Stage to execute: `A` (all stages), `1` (download/extract), `2` (time-series), `3` (mosaic & stack). |
| `--overwrite` | flag | `False` | Force recomputation of already existing output files. |

---

## Transient disk space & cleanup guidelines

Sentinel-2 preprocessing creates intermediate daily mosaics and tile slices:
* **`_temp_processing/`** ($\sim 100\text{ to }150\text{ GB}$ per orbit): Extracted tiles and DOY mosaics.
* **`1_input_stacks/`** ($\sim 120\text{ to }180\text{ GB}$ per orbit): Final 126-band BigTIFF stack.

> [!TIP]
> Once Stage 3 completes and `*_S2_timeseries.tif` is verified, the entire `_temp_processing/` directory can be deleted to recover $\sim 130\text{ GB}$ of disk space.
