# Phase 1b: NISAR L-band SAR preprocessor (`1b_NISAR_preprocessor`)

## Overview

The `1b_NISAR_preprocessor` toolbox provides automated discovery, download, polarimetric extraction, cross-track antenna pattern de-tapering, seamless 12-day orbital cycle mosaicking, and multi-temporal BigTIFF stacking for **NASA/ISRO NISAR Level-2 Geocoded Polarimetric Covariance (`L2 GCOV`)** L-band (wavelength ~24 cm, 1.257 GHz) radar imagery across any European Union (`EU-27`) country.

All output stacks are co-registered pixel-for-pixel to the **Sentinel-1** and **Sentinel-2** reference grid (`EPSG:3857` Web Mercator at `10.0 m` ground sampling distance) and stored in `workingDirs/<COUNTRY>/orbit_<ORBIT>/1_input_stacks/` for direct assimilation into the multimodal crop classifier (`2_classifier/run_classifier.py`). When NISAR data is not available for a given region or season, the classifier **automatically falls back to `Sentinel-1 + Sentinel-2`**.

---

## Architecture & 3-stage pipeline

```text
1b_NISAR_preprocessor/
├── run_nisar_preprocessor.py      # Master CLI & interactive setup wizard
├── config_nisar.json              # Active configuration (NASA Earthdata credentials, paths)
├── config_nisar.example.json      # Template configuration
└── modules/
    ├── __init__.py                # Package initializer
    ├── nisar_asf_search.py        # S1/S2 reference grid lock, NUTS loader, local & ASF DAAC search
    ├── nisar_extract_gcov.py      # HDF5 L2 GCOV extraction, border erosion, EDT fill & de-tapering
    ├── nisar_mosaic_stack.py      # Slit healing, cross-seam harmonization, C1 feathering & stacking
    └── nisar_pipeline.py          # 3-stage NISAR pipeline orchestrator (CalculateTrackNISAR)
```

### Stage 1: Granule discovery, HDF5 extraction & radiometric calibration (`--stage 1`)
* Queries local `/eodata/NISAR` archives (`--source local`), the NASA ASF DAAC API (`--source asf`), or both automatically (`--source auto`, default), grouping scenes into 12-day orbital repeat cycles.
* Extracts four physical L-band polarimetric layers per granule:
  1. `NISAR_HHHH_dB_<DDMonYYYY>` — Co-polarized L-band backscatter (`Gamma0_HH` in dB, `10 * log10(Gamma0_HH)`).
  2. `NISAR_HVHV_dB_<DDMonYYYY>` — Cross-polarized L-band volume scattering (`Gamma0_HV` in dB, `10 * log10(Gamma0_HV)`).
  3. `NISAR_RVI_L_<DDMonYYYY>` — L-band Radar Vegetation Index (`RVI_L = 4 * HV / (HH + HV)`, range `[0.0, 1.5]`).
  4. `NISAR_RATIO_<DDMonYYYY>` — Co-to-cross-polarization power ratio in dB (`10 * log10(HH / HV)`).
* Applies cross-track invalid margin erosion (`5 px`), Euclidean Distance Transform (`EDT`) pinhole imputation (`<= 64 px`), and 2D adaptive monotonic cross-track range-edge de-tapering.

### Stage 2: 12-day cycle mosaicking & `EPSG:3857` (`10.0 m`) alignment (`--stage 2`)
* Performs along-track 1D vertical slit healing (`9 x 1` closing kernel), cross-seam radiometric median harmonization, and C1-continuous distance-squared feathering across overlapping frames.
* Warps and clips each 12-day cycle mosaic to match the exact `EPSG:3857` (`10.0 m`) bounding box and pixel dimensions of the `Sentinel-1` / `Sentinel-2` reference stack (`1_input_stacks/`).

### Stage 3: Multi-temporal L-band BigTIFF stacking & pyramid overviews (`--stage 3`)
* Stacks all 12-day cycle mosaics into a single multi-temporal `DEFLATE`-compressed BigTIFF (`4 * T` bands) with `[2, 4, 8, 16, 32, 64]` external pyramid overviews (`.ovr`).
* Automatically warps any legacy cycle mosaics in native UTM projections on-the-fly to match the `EPSG:3857` (`10.0 m`) master grid.
* Outputs `{COUNTRY}_orbit_{ORBIT}_{END_DATE}_{START_DATE}_NISAR_GCOV_L_timeseries.tif` (with instant hardlink aliases `*_NISAR_GCOV.tif` and `*_NISAR_timeseries.tif`).

---

## Quick-start CLI examples

```bash
# 1. Launch interactive setup wizard (zero arguments):
python 1b_NISAR_preprocessor/run_nisar_preprocessor.py

# 2. Run full NISAR preprocessing for a single orbit track (aligned 1:1 with S1/S2 stack):
python 1b_NISAR_preprocessor/run_nisar_preprocessor.py --track NL/orbit_88 --source auto -s 2025-10-15 -e 2026-09-15 --stage A --threads 4

# 3. Run full NISAR preprocessing across all orbits of a country from NASA ASF DAAC:
python 1b_NISAR_preprocessor/run_nisar_preprocessor.py --country PL --source asf -s 2025-10-15 -e 2026-09-15 --stage A --threads 4

# 4. Run from local HDF5 granules only (offline mode):
python 1b_NISAR_preprocessor/run_nisar_preprocessor.py --track PT/orbit_52 --source local -s 2025-10-15 -e 2026-09-15 --stage A

# 5. Run only Stage 3 (Re-stack existing cycle mosaics into S1-aligned BigTIFF):
python 1b_NISAR_preprocessor/run_nisar_preprocessor.py --track PT/orbit_52 --stage 3
```

---

## CLI arguments reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--track` | Single target track (`COUNTRY/orbit_ID`, e.g., `PT/orbit_52`). | Interactive |
| `--country` | Process all orbits for an EU-27 country code (e.g., `PT`, `NL`, `PL`). | `None` |
| `--source` | Data source mode: `auto` (local first, then ASF DAAC), `asf`, or `local`. | `auto` |
| `-s`, `--start_date` | Start date (`YYYY-MM-DD`). | Config value |
| `-e`, `--end_date` | End date (`YYYY-MM-DD`). | Config value |
| `--stage` | Execution stage: `1` (Extract), `2` (Mosaic), `3` (Stack), `A` (All). | `A` |
| `--threads` | Number of parallel worker threads. | `4` |
| `--keep_h5` | Retain raw `.h5` files in `_temp_nisar_h5/` after extraction. | `False` |
| `--config` | Path to custom JSON configuration file. | `config_nisar.json` |


