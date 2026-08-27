# Sentinel-2 optical preprocessor toolbox

This toolbox generates cloud-free, regular 10-day synthetic multispectral optical time-series composites matching 1:1 with the Sentinel-1 SAR pixel grid from Copernicus **Sentinel-2 Level-2A (Bottom-Of-Atmosphere / Surface Reflectance)** products.

---

## Architecture overview

```
[Copernicus CDSE API / CreoDIAS L2A Archive]
                     |
                     v
    +----------------------------------+
    |  Stage 1: Retrieval & Masking    |  --> S2 L2A tile download/extraction, Scene Classification
    |  (SCL Cloud & Shadow Filtering)  |      Layer (SCL) filtering for clouds, shadows, and snow
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  Stage 2: Synthetic DOY          |  --> Multi-temporal interpolation across 14 standardized DOYs:
    |  Time-Series Interpolation       |      [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]
    +----------------+-----------------+
                     |
                     v
    +----------------------------------+
    |  Stage 3: SAR Grid Alignment     |  --> Sub-pixel resampling to Sentinel-1 raster bounding box,
    |  & 126-Band BigTIFF Stacking     |      126-band BigTIFF creation, and 6 pyramid overviews
    +----------------------------------+
```

---

## File structure

* **`run_s2_preprocessor.py`**: Unified master CLI runner and interactive terminal wizard.
* **`config_s2.json`**: Active configuration containing CDSE credentials, spectral bands, DOYs, and working directories.
* **`config_s2.example.json`**: Template configuration file.
* **`modules/`**: Local processing modules:
  * `s2_download_cdse.py`: Automated retrieval and SCL cloud masking from CDSE API.
  * `s2_extract_creodias.py`: Direct extraction from local CreoDIAS archive (`Y:/Sentinel-2/MSI/L2A`).
  * `s2_time_series.py`: Pure Python multi-temporal interpolation across 14 standardized DOYs.
  * `s2_mosaic_stack.py`: Mosaicking, sub-pixel grid alignment to Sentinel-1, 126-band BigTIFF creation, and pyramid overview generation.
  * `s2_pipeline.py`: Object-oriented pipeline orchestrator.

---

## Standardized agricultural DOYs & spectral bands

* **14 reference dates**:
  $$\text{DOYs} = [80, 105, 119, 132, 146, 161, 175, 189, 203, 217, 231, 252, 273, 287]$$
  *(from late March to mid-October)*
* **9 spectral bands per DOY**:
  `B02` (Blue), `B03` (Green), `B04` (Red), `B05` (RedEdge 1), `B06` (RedEdge 2), `B07` (RedEdge 3), `B8A` (Narrow NIR), `B11` (SWIR 1), `B12` (SWIR 2).
* **Total layers**: $14\text{ dates} \times 9\text{ bands} = 126\text{ spectral bands}$ per orbit.

---

## Execution commands

### 1. Interactive wizard (recommended for beginners)
```powershell
python run_s2_preprocessor.py
```

### 2. Automated country-wide processing
```powershell
# Full automated run for Portugal (multi-core, 8 worker threads):
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

# Run specific single stage (e.g. Stage 3 only):
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
| `--source` | choice | `cdse` | Data source: `cdse` (Copernicus API) or `creodias` (local repository). |
| `--threads` | int | `8` | Number of parallel worker threads for multi-temporal interpolation. |
| `--stage` | string | `None` | Stage to execute: `A` (all stages), `1` (download/extract), `2` (time-series), `3` (mosaic & stack). |
| `--overwrite` | flag | `False` | Force recomputation of already existing output files. |

---

## Output products

The final optical multi-temporal BigTIFF stack is saved in:
`workingDirs/{COUNTRY}/orbit_{ORBIT}/1_input_stacks/{COUNTRY}_orbit_{ORBIT}_S2_timeseries.tif`

* Coordinate reference system: `EPSG:3857` (Web Mercator)
* Pixel resolution: Exactly $10.0\text{ m} \times 10.0\text{ m}$ (matched 1:1 to Sentinel-1)
* Band count: 126 spectral layers
* Compression: `DEFLATE` with multi-scale pyramid overviews (`[2, 4, 8, 16, 32, 64]`).
