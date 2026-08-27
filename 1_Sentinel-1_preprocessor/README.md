# Sentinel-1 SAR preprocessor toolbox

This toolbox handles automated radiometric calibration, multi-temporal coregistration, terrain correction, BigTIFF stacking, and administrative boundary clipping of Copernicus **Sentinel-1 C-band Synthetic Aperture Radar (SAR)** Level-1 Ground Range Detected (GRD) products.

---

## Architecture overview

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

---

## File structure

* **`run_s1_preprocessor.py`**: Unified master CLI runner and interactive terminal wizard.
* **`config_s1.json`**: Active configuration containing CDSE credentials, SNAP paths, and working directories.
* **`config_s1.example.json`**: Template configuration file.
* **`modules/`**: Local processing modules:
  * `s1_calibration_creodias.py`: Fast calibration from local CreoDIAS COG repository.
  * `s1_calibration_cdse.py`: Automated retrieval and calibration from CDSE API.
  * `s1_coregistration.py`: Multi-temporal coregistration using ESA SNAP GPT (`CreateStack`).
  * `s1_stack_clip.py`: Range Doppler terrain correction, GDAL BigTIFF stacking, and regional clipping.

---

## Execution commands

### 1. Interactive wizard (recommended for beginners)
```powershell
python run_s1_preprocessor.py
```

### 2. Automated country-wide processing
```powershell
# Full automated run for Portugal:
python run_s1_preprocessor.py --country PT -s 2024-10-15 -e 2025-09-15 --stage A

# Full automated run for the Netherlands:
python run_s1_preprocessor.py --country NL -s 2024-10-15 -e 2025-09-15 --stage A

# Full automated run for Poland with winter freeze exclusion (01.12 - 14.02):
python run_s1_preprocessor.py --country PL --exclude_winter --stage A
```

### 3. Single orbit track execution
```powershell
# Process single orbit track:
python run_s1_preprocessor.py --track PT/orbit_52 -s 2024-10-15 -e 2025-09-15 --stage A

# Run specific single stage (e.g. Stage 3 only):
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
| `--exclude_winter` | flag | `False` | Exclude winter observations between December 1 and February 14. |
| `--stage` | string | `None` | Stage to execute: `A` (all stages), `1` (calibration), `2` (coregistration), `3` (stack & clip). |
| `--overwrite` | flag | `False` | Force recomputation of already existing output files. |

---

## Output products

The final orthorectified multi-temporal BigTIFF stack is saved in:
`workingDirs/{COUNTRY}/orbit_{ORBIT}/1_input_stacks/{COUNTRY}_orbit_{ORBIT}_{END_DATE}_{START_DATE}_VH_VV.tif`

* Coordinate reference system: `EPSG:3857` (Web Mercator)
* Pixel resolution: Exactly $10.0\text{ m} \times 10.0\text{ m}$
* Compression: `DEFLATE` with multi-scale pyramid overviews (`[2, 4, 8, 16, 32, 64]`).
