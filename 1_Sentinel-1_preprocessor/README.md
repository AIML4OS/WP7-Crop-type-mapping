# Sentinel-1 SAR preprocessor toolbox

This toolbox provides an automated, scientific-grade preprocessing pipeline for **Sentinel-1 C-band Synthetic Aperture Radar (SAR)** Level-1 Ground Range Detected (GRD) products. It converts raw satellite swaths into radiometrically calibrated, terrain-corrected, multi-temporal orthorectified backscatter stacks aligned to standardized European administrative boundaries (`EPSG:3857`, 10.0 m pixel resolution).

The toolbox supports **two data ingestion pathways** and **two SAR processing engines**:
* **Two data ingestion pathways (`--source`)**:
  1. **Copernicus Data Space Ecosystem API (`cdse`)**: Automated OData catalog discovery and `.SAFE.zip` product download.
  2. **Local CreoDIAS repository (`creodias` / `cog`)**: Direct high-speed reading from mounted `/eodata` (or local network drive) `.SAFE` and `.SAFE-COG` archives.
* **Two SAR processing engines (`--engine`)**:
  1. **Pure Python SAR engine (`python` — default)**: 100% standalone `Python` / `NumPy` / `SciPy` / `GDAL` implementation requiring **zero ESA SNAP installation**. Performs 2D thermal noise removal, border noise removal, `Sigma0` LUT calibration, Range-Doppler terrain correction with **Copernicus 30m Global DEM**, and multi-temporal Quegan speckle filtering directly in Python.
  2. **ESA SNAP GPT engine (`snap`)**: Legacy Graph Processing Tool (`gpt`) XML workflow updated to use the **Copernicus 30m Global DEM** with bilinear resampling.

---

## Scientific methodology & radar remote sensing

### 1. Radar backscatter mechanisms in agriculture
Sentinel-1 operates at C-band microwave frequency (5.405 GHz, wavelength ~5.55 cm). Unlike optical sensors, microwaves penetrate clouds, haze, and rain, interacting directly with the structural and dielectric properties of the vegetation canopy and soil:
* **VV polarization (vertical transmit / vertical receive)**: Dominantly sensitive to surface roughness, soil moisture, and vertical canopy elements (e.g. cereal stems in wheat and barley).
* **VH polarization (vertical transmit / horizontal receive)**: Cross-polarization characterized by high sensitivity to volume scattering within the crop canopy, serving as a direct indicator of biomass accumulation, leaf density, and canopy closure.
* **VH/VV polarization ratio & cross-ratio**: Normalizes soil moisture variations, highlighting crop phenological transitions, stem elongation, and heading phases.

### 2. Radar backscatter calibration and polarimetric indices
The pipeline performs radiometric calibration and prepares physical backscatter stacks for downstream agricultural polarimetric indices:
* **Calibration to physical backscatter (`Sigma0`)**: Converts raw digital numbers (`DN`) into normalized radar cross section (`Sigma0`) using 2D sensor look-up tables (LUT) and thermal noise vectors:
  `Sigma0(l, p) = max(DN(l, p)^2 - N_rg(l, p) * N_az(l, p), 1e-5) / Sigma_LUT(l, p)^2`
* **Decibel transformation (dB)**: Converts linear backscatter into decibels: `Sigma0_dB = 10 * log10(Sigma0)`, standardizing the dynamic range typically between `-28 dB` and `0 dB`.
* **Radar Vegetation Index (RVI)**: Measures canopy volume scattering relative to total backscatter: `RVI = 4 * VH / (VV + VH)`.
* **Polarimetric cross-ratio (CR)**: Ratio of cross-polarized to co-polarized backscatter: `CR [dB] = VH [dB] - VV [dB]`.

### 3. Multi-temporal Quegan speckle filtering & winter freeze filtering
* **Vectorized multi-temporal Quegan filter**: For a coregistered time series of `M` linear `Sigma0` acquisitions, the Pure Python engine applies the unbiased Quegan multi-temporal speckle filter in `2048 x 2048` blocks:
  `J_k(x, y) = Mean_W(I_k) * (1 / M) * sum_{i=1..M}(I_i / Mean_W(I_i))`
  This suppresses multiplicative granular speckle noise by a factor of `sqrt(M)` while strictly preserving 10 m agricultural field margins.
* **Winter freeze exclusion (`--exclude_winter`)**: During freezing temperatures, liquid water transitions to ice, causing a dielectric drop that lowers backscatter by 4 to 8 dB. The `--exclude_winter` flag filters out scenes between December 1 and February 14.

---

## Processing flowchart & dual-engine architecture

```
+----------------------------------------------------------------------------------------------------+
|                         SENTINEL-1 SAR PREPROCESSING PIPELINE FLOWCHART                            |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Input data ingestion: 2 supported pathways]             |
                     | 1. Copernicus CDSE OData API (--source cdse)             |
                     | 2. CreoDIAS Local Repository (--source creodias / cog)   |
                     +----------------------------+-----------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v (--engine python [DEFAULT])                     v (--engine snap)
+-------------------------------------------------+   +----------------------------------------------+
| [Pure Python SAR Engine (s1_python_processor)]  |   | [ESA SNAP GPT Engine]                        |
| Stage 1:                                        |   | Stage 1:                                     |
| - Parse annotation, calibration & noise XMLs    |   | - Apply Precise Orbit File (POEORB)          |
| - 2D Thermal Noise Removal (TNR) & Border mask  |   | - Thermal & GRD Border Noise Removal         |
| - Sigma0 LUT bivariate spline calibration       |   | - Radiometric Calibration to linear Sigma0   |
| - Range-Doppler Terrain Correction with         |   | - Daily SliceAssembly & Subset (.dim)        |
|   Copernicus 30m Global DEM onto locked 10.0m   |   | Stage 2:                                     |
|   EPSG:3857 grid + on-the-fly slice stitching   |   | - SNAP CreateStack & Cross-Correlation Warp  |
| Stage 2:                                        |   | - Lee Sigma + Median Speckle Filter          |
| - Multi-temporal master grid alignment          |   | - Range-Doppler TC (Copernicus 30m DEM)      |
| Stage 3:                                        |   | Stage 3:                                     |
| - Vectorized Multi-Temporal Quegan Despeckling  |   | - GDAL VRT assembly & NUTS2 cutline warp     |
| - Preserves full native radar time series       |   | - Export DEFLATE BigTIFF & pyramid overviews |
| - Linear-to-dB, NUTS2 clip & BigTIFF + pyramids |   |                                              |
+------------------------+------------------------+   +----------------------+-----------------------+
                         |                                                   |
                         +------------------------+--------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Output product: 1_input_stacks/]                        |
                     | {COUNTRY}_orbit_{ORBIT}_{END}_{START}_VH_VV.tif          |
                     | (Dual-pol multi-temporal BigTIFF, 10.0m, EPSG:3857)      |
                     +----------------------------------------------------------+
```

---

## Processing stages in detail

### Stage 1: Radiometric calibration, orthorectification & slice assembly
* **Pure Python engine (`--engine python`)**:
  * Directly parses `annotation/*.xml`, `annotation/calibration/calibration-*.xml`, and `annotation/calibration/noise-*.xml` from `.SAFE` or `.SAFE-COG` directories.
  * Subtracts 2D range and azimuth thermal noise vectors (`noiseRangeLut` and `noiseAzimuthLut`) and masks GRD range/azimuth border artifacts.
  * Evaluates 2D `RectBivariateSpline` `Sigma0` calibration LUTs in horizontal strips.
  * Orthorectifies each slice using exact Range-Doppler foreshortening equations and the **Copernicus 30m Global DEM (GLO-30)** directly onto a locked 10.0 m `EPSG:3857` grid, seamlessly stitching same-date slices into `slice_assembly/{YYYYMMDD}.tif`.
* **ESA SNAP engine (`--engine snap`)**:
  * Applies precise orbit ephemerides (`Sentinel Precise (Auto Download)`), thermal noise removal, border noise removal, `Sigma0` calibration, and `SliceAssembly` into BEAM-DIMAP `.dim` products.

### Stage 2: Multi-temporal coregistration (`s1_coregistration.py`)
* **Pure Python engine (`--engine python`)**:
  * Aligns all daily orthorectified GeoTIFFs in `slice_assembly/*.tif` onto the unified master `EPSG:3857` 10.0 m coordinate grid and exports `wrapped/python_stack_manifest.json` without generating 250 GB of intermediate `.dim` files.
* **ESA SNAP engine (`--engine snap`)**:
  * Runs SNAP `CreateStack` cross-correlation coregistration, multi-temporal speckle filtering, and Range-Doppler terrain correction with `Copernicus 30m Global DEM` and `BILINEAR_INTERPOLATION`.

### Stage 3: Despeckling, stacking & NUTS2 clipping (`s1_stack_clip.py`)
* Applies vectorized multi-temporal Quegan speckle filtering across all dates in linear power domain (window size configurable via `--filter_window`, default `7`).
* Preserves the **full native Sentinel-1 radar time series** across all acquired orbit passes (unlike Sentinel-2 which requires 14-DOY interpolation due to cloud cover).
* Converts linear `Sigma0` into decibels (`dB = 10 * log10(Sigma0)`), clips to the official Eurostat GISCO NUTS2 national boundary in `EPSG:3857` at 10.0 m resolution, exports a tiled BigTIFF (`COMPRESS=DEFLATE`, `TILED=YES`, `BIGTIFF=YES`), and builds 6 pyramid overview levels (`[2, 4, 8, 16, 32, 64]`).

---

## File and directory structure

* **`run_s1_preprocessor.py`**: Master CLI orchestrator and interactive setup wizard.
* **`config_s1.json`**: Active configuration file (CDSE credentials, working directories, local CreoDIAS paths).
* **`config_s1.example.json`**: Template configuration file.
* **`modules/`**:
  * `s1_python_processor.py`: Standalone Pure Python Sentinel-1 SAR calibration, Range-Doppler terrain correction (Copernicus 30m DEM), multi-temporal Quegan speckle filter, and BigTIFF stacker.
  * `s1_calibration_cdse.py`: Automated scene discovery, download, and Stage 1 processing from the Copernicus CDSE OData API.
  * `s1_calibration_creodias.py`: Stage 1 discovery and processing from local CreoDIAS `.SAFE` repositories (`/eodata/Sentinel-1/SAR/IW_GRDH_1S`).
  * `s1_calibration_cog.py`: Stage 1 discovery and processing from local CreoDIAS `.SAFE-COG` repositories (`/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG`).
  * `s1_coregistration.py`: Stage 2 multi-temporal coregistration (Pure Python grid alignment & SNAP `CreateStack`).
  * `s1_stack_clip.py`: Stage 3 multi-temporal despeckling, NUTS2 administrative clipping, and BigTIFF pyramid generation.

---

## Execution commands

### 1. Interactive wizard mode (zero-argument launch)
```bash
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py
```

### 2. Pure Python execution with local CreoDIAS repository (`--source creodias` / `cog`)
```bash
# Full automated Pure Python run for Portugal (all optimal greedy orbits, excluding winter):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PT --source creodias --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A --threads 4

# Single orbit track execution in Pure Python using CreoDIAS COG archive:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 --source cog --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A

# Full automated Pure Python run for Poland:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country PL --source creodias --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A --threads 8
```

### 3. Pure Python execution with Copernicus CDSE download (`--source cdse`)
```bash
# Country-wide CDSE download and Pure Python processing for the Netherlands:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --country NL --source cdse --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A --threads 4

# Single orbit track download from CDSE and Pure Python processing:
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 --source cdse --engine python -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A
```

### 4. Running individual stages
```bash
# Run Stage 1 only (calibration, Range-Doppler orthorectification & daily slice assembly):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 --source creodias --engine python -s 2024-10-15 -e 2025-09-15 --stage 1

# Run Stage 2 only (multi-temporal grid alignment):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 --engine python --stage 2

# Run Stage 3 only (full radar time-series stacking with 7x7 Multi-Temporal Quegan filter):
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 --engine python --filter_window 7 --stage 3
```

### 5. Running with the legacy ESA SNAP GPT engine (`--engine snap`)
```bash
python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --track PT/orbit_52 --source cog --engine snap -s 2024-10-15 -e 2025-09-15 --exclude_winter --stage A
```

---

## Command-line arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-t, --track` | string | `None` | Satellite track identifier (e.g. `PT/orbit_52`, `NL/orbit_37`, `PL/orbit_29`). |
| `-c, --country` | string | `None` | Country code (e.g. `PT`, `NL`, `PL`, `ES`, `FR`, `DE`). |
| `-o, --orbit` | int | `None` | Specific relative orbit number (e.g. `52`, `125`, `37`). |
| `-s, --start_date` | string | `None` | Start date of agricultural season (`YYYY-MM-DD`, required for Stage 1/A). |
| `-e, --end_date` | string | `None` | End date of agricultural season (`YYYY-MM-DD`, required for Stage 1/A). |
| `--source` | choice | `auto` | Data ingestion pathway: `auto` (detect local), `cog` (CreoDIAS COG), `creodias` (CreoDIAS SAFE), or `cdse` (Copernicus CDSE API). |
| `--engine` | choice | `python` | SAR processing engine: `python` (Pure Python without SNAP), `snap` (ESA SNAP GPT), or `auto`. |
| `--filter_window` | int | `7` | Window size (pixels) for the multi-temporal Quegan speckle filter in Python Stage 3. |
| `--exclude_winter` | flag | `False` | Exclude winter freeze observations between December 1 and February 14. |
| `--threads` | int | `4` | Worker threads for parallel processing (default: 4). |
| `--stage` | choice | `None` | Stage to execute: `A` (all stages), `1` (calibration), `2` (coregistration), `3` (stack & clip). |
| `--overwrite` | flag | `False` | Force recomputation of already existing output files. |

---

## Transient disk space & cleanup guidelines

* **Pure Python engine (`--engine python`)**:
  * **`slice_assembly/*.tif`** (~40 to 60 GB per orbit): Daily calibrated, Range-Doppler terrain-corrected linear `Sigma0` GeoTIFFs (`COMPRESS=DEFLATE`).
  * **`wrapped/python_stack_manifest.json`** (< 50 KB): Lightweight JSON coordinate alignment manifest (eliminates ~500 GB of SNAP `.dim`/`.data` intermediate files).
  * **`1_input_stacks/`** (~80 to 120 GB per orbit): Final full-time-series multi-temporal BigTIFF stack with `.ovr` pyramids.
* **ESA SNAP engine (`--engine snap`)**:
  * Produces intermediate BEAM-DIMAP `.dim`/`.data` folders in `calibrated/`, `slice_assembly/`, `wrapped/`, and `S1_final_preprocessing/` (~250 to 450 GB each; `wrapped/` and `S1_final_preprocessing/` are automatically cleaned after verification).
