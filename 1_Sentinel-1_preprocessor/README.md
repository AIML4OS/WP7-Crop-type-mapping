# Sentinel-1 SAR preprocessor toolbox

This toolbox provides an automated, scientific-grade preprocessing pipeline for **Sentinel-1 C-band Synthetic Aperture Radar (SAR)** Level-1 Ground Range Detected (GRD) products. It converts raw satellite swaths into radiometrically calibrated, terrain-corrected, multi-temporal orthorectified backscatter stacks aligned to standardized European administrative boundaries.

---

## Scientific methodology & radar physics

### 1. Radar backscatter mechanisms in agriculture
Sentinel-1 operates at C-band microwave frequency ($f = 5.405\text{ GHz}$, wavelength $\lambda \approx 5.55\text{ cm}$). Unlike optical sensors, microwaves penetrate clouds, haze, and rain, interacting directly with the structural and dielectric properties of the vegetation canopy and soil:
* **$VV$ polarization (vertical transmit / vertical receive)**: Dominantly sensitive to surface roughness, soil moisture, and vertical canopy elements (e.g. cereal stems in wheat and barley).
* **$VH$ polarization (vertical transmit / horizontal receive)**: Cross-polarization characterized by high sensitivity to volume scattering within the crop canopy, serving as a direct indicator of biomass accumulation, leaf density, and canopy closure.
* **$VH/VV$ polarization ratio & cross-ratio**: Normalizes soil moisture variations, highlighting crop phenological transitions, stem elongation, and heading phases.

### 2. Fundamental Radar Equations & Polarimetry
The received radar backscatter power $P_r$ from a distributed agricultural target is given by:

$$P_r = \frac{P_t G^2 \lambda^2 \sigma^0 A_{\text{ground}}}{(4\pi)^3 R^4}$$

* **Calibration to physical backscatter ($\sigma^0$)**:
  $$\sigma^0_i = \frac{|DN_i|^2}{A_i^2}$$
* **Radar Vegetation Index (RVI)**:
  $$\text{RVI} = \frac{4 \cdot \sigma^0_{VH}}{\sigma^0_{VV} + \sigma^0_{VH}}$$
* **Polarimetric Cross-Ratio (CR)**:
  $$\text{CR}_{\text{dB}} = \sigma^0_{VH,\text{dB}} - \sigma^0_{VV,\text{dB}} = 10 \log_{10}\left(\frac{\sigma^0_{VH}}{\sigma^0_{VV}}\right)$$

### 3. Multi-temporal decorrelation & winter freeze filtering
During freezing temperatures, water transitions to ice, causing a dramatic drop in the dielectric constant ($\varepsilon_{\text{water}} \approx 80 \rightarrow \varepsilon_{\text{ice}} \approx 3.15$). This leads to an artificial backscatter drop of $4\text{ to }8\text{ dB}$. The pipeline includes an optional `--exclude_winter` filter that removes scenes between December 1 and February 14 to avoid training distortions.

---

## Processing flowchart & SNAP graph architecture

```
+----------------------------------------------------------------------------------------------------+
|                         SENTINEL-1 SAR PREPROCESSING PIPELINE FLOWCHART                            |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Input Data Ingestion]                                   |
                     | - Copernicus CDSE OData API (Automated SAFE download)    |
                     | - CreoDIAS Local Archive (Fast COG / SAFE extraction)    |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 1: Radiometric Calibration & Slice Assembly]      |
                     | 1. Apply Orbit File (Precise POEORB vectors, sub-cm acc) |
                     | 2. Thermal Noise Removal (TNR cross-talk correction)     |
                     | 3. Border Noise Removal (BNR sampling artifact mask)     |
                     | 4. Radiometric Calibration: DN -> Sigma0 (linear scale)  |
                     | 5. Slice Assembly: Daily swath stitching along track     |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 2: Multi-Temporal Coregistration (CreateStack)]   |
                     | 1. Select optimal master acquisition (mid-season)        |
                     | 2. Sub-pixel cross-correlation geometric alignment       |
                     | 3. Assemble multi-temporal wrapped stack (all DOYs)      |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Stage 3: Terrain Correction, Stacking & Clipping]       |
                     | 1. Range Doppler Terrain Correction (Copernicus 30m DEM) |
                     | 2. Linear to Decibel conversion: dB = 10 * log10(Sigma0) |
                     | 3. Administrative boundary clipping (GISCO NUTS2, 10.0m) |
                     | 4. Export BigTIFF with DEFLATE & 6 pyramid overview lvls |
                     +----------------------------+-----------------------------+
                                                  |
                                                  v
                     +----------------------------------------------------------+
                     | [Output Product: 1_input_stacks/]                        |
                     | {COUNTRY}_orbit_{ORBIT}_{END}_{START}_VH_VV.tif          |
                     | (Dual-pol multi-temporal BigTIFF, 10m, EPSG:3857)        |
                     +----------------------------------------------------------+
```

---

## Processing stages in detail

### Stage 1: Radiometric calibration & slice assembly (`s1_calibration_*.py`)
* **Precise orbit state vectors (POEORB)**: Automatically fetched from ESA servers with 5 cm positioning accuracy, updating satellite velocity and state vectors.
* **Thermal noise removal (TNR)**: Subtracts thermal antenna noise patterns along range lines, critical for low-intensity cross-polarization ($VH$).
* **Border noise removal (BNR)**: Masks radiometric artifacts and low-intensity sampling invalidities along slice edges.
* **Calibration to $\sigma^0$**: Converts raw digital numbers ($DN$) into physical radar backscatter coefficient:
  $$\sigma^0 = \frac{|DN|^2}{A_i^2}$$
* **Slice assembly**: Merges consecutive along-track radar frames acquired on the same date into a seamless track strip.

### Stage 2: Multi-temporal coregistration (`s1_coregistration.py`)
* Utilizes ESA SNAP GPT `CreateStack` operator.
* Selects a stable mid-season master scene and aligns all multi-temporal slave acquisitions with sub-pixel precision using coarse-to-fine cross-correlation, eliminating geometric shifts between dates.

### Stage 3: Terrain correction, stacking & clipping (`s1_stack_clip.py`)
* **Range Doppler terrain correction**: Orthorectifies radar geometry distortions (foreshortening, layover, shadow) using the Copernicus 30 m global DEM and Earth gravitational model (EGM96).
* **Decibel transformation**: Converts backscatter into decibels: $\sigma^0_{\text{dB}} = 10 \log_{10}(\sigma^0)$, standardizing signal dynamic range (typically $-28\text{ dB}$ to $0\text{ dB}$).
* **BigTIFF generation & pyramids**: Exports tiled BigTIFF (`COMPRESS=DEFLATE`, `TILED=YES`) in `EPSG:3857` at an exact $10.0\text{ m}$ pixel resolution and generates 6 pyramid overview layers (`[2, 4, 8, 16, 32, 64]`) for instant rendering.

---

## File and directory structure

* **`run_s1_preprocessor.py`**: Master CLI runner and interactive terminal wizard.
* **`config_s1.json`**: Active configuration containing CDSE credentials, SNAP paths, and working directories.
* **`config_s1.example.json`**: Template configuration file.
* **`modules/`**:
  * `s1_calibration_creodias.py`: Fast calibration from local CreoDIAS COG repository.
  * `s1_calibration_cdse.py`: Automated retrieval and calibration from CDSE API.
  * `s1_coregistration.py`: Multi-temporal coregistration using ESA SNAP GPT (`CreateStack`).
  * `s1_stack_clip.py`: Range Doppler terrain correction, GDAL BigTIFF stacking, and regional clipping.

---

## Execution commands

### 1. Interactive wizard mode (zero-argument launch)
```powershell
python run_s1_preprocessor.py
```

### 2. Country-wide automated execution
```powershell
# Full automated run for Portugal (orbit 52 and orbit 125):
python run_s1_preprocessor.py --country PT -s 2024-10-15 -e 2025-09-15 --stage A

# Full automated run for the Netherlands:
python run_s1_preprocessor.py --country NL -s 2024-10-15 -e 2025-09-15 --stage A

# Full automated run for Poland with winter freeze exclusion:
python run_s1_preprocessor.py --country PL --exclude_winter --stage A
```

### 3. Single orbit track execution
```powershell
# Process single orbit track:
python run_s1_preprocessor.py --track PT/orbit_52 -s 2024-10-15 -e 2025-09-15 --stage A

# Run specific single stage (e.g. Stage 3 terrain correction & clipping only):
python run_s1_preprocessor.py --track PT/orbit_52 --stage 3
```

---

## Command-line arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-t, --track` | string | `None` | Satellite track identifier (e.g. `PT/orbit_52`, `NL/orbit_88`). |
| `-c, --country` | string | `None` | Country code (e.g. `PT`, `NL`, `PL`, `ES`, `FR`, `DE`). |
| `-s, --start_date` | string | `2024-10-15` | Start date of agricultural season (`YYYY-MM-DD`). |
| `-e, --end_date` | string | `2025-09-15` | End date of agricultural season (`YYYY-MM-DD`). |
| `--source` | choice | `cog` | Data source: `cog` (CreoDIAS local repository) or `cdse` (Copernicus API). |
| `--exclude_winter` | flag | `False` | Exclude winter freeze observations between December 1 and February 14. |
| `--stage` | string | `None` | Stage to execute: `A` (all stages), `1` (calibration), `2` (coregistration), `3` (stack & clip). |
| `--overwrite` | flag | `False` | Force recomputation of already existing output files. |

---

## Transient disk space & cleanup guidelines

Sentinel-1 preprocessing produces large intermediate products during calibration and stacking:
* **`calibrated/`** ($\sim 250\text{ GB}$ per orbit): Calibrated individual slices.
* **`slice_assembly/`** ($\sim 250\text{ GB}$ per orbit): Assembled daily tracks.
* **`wrapped/`** ($\sim 250\text{ GB}$ per orbit): Coregistered multi-temporal stack.
* **`1_input_stacks/`** ($\sim 80\text{ to }120\text{ GB}$ per orbit): Final compact BigTIFF stack.

> [!TIP]
> Once Stage 3 finishes and the final BigTIFF is created in `1_input_stacks/`, all temporary folders (`calibrated/`, `slice_assembly/`, `wrapped/`) can be safely deleted to free $\sim 750\text{ GB}$ of disk space per orbit.
