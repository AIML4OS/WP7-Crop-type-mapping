# Phase 1b: NISAR L-band SAR preprocessor (`1b_NISAR_preprocessor`)

## Overview

The `1b_NISAR_preprocessor` toolbox provides automated discovery, download, polarimetric extraction, cross-track antenna pattern de-tapering, seamless 12-day orbital cycle mosaicking, and multi-temporal BigTIFF stacking for **NASA/ISRO NISAR Level-2 Geocoded Polarimetric Covariance (`L2 GCOV`)** L-band ($\lambda \approx 24\text{ cm}$) radar imagery across any European Union (`EU-27`) country.

All output stacks are co-registered pixel-for-pixel to the **Sentinel-1** and **Sentinel-2** reference grid (`EPSG:3857` Web Mercator at `10.0 m` ground sampling distance) and stored in `workingDirs/<COUNTRY>/orbit_<ORBIT>/1_input_stacks/` for direct assimilation into the multimodal crop classifier (`2_classifier/run_classifier.py`). When NISAR data is not available for a given region or season, the classifier automatically falls back to `Sentinel-1 + Sentinel-2`.

---

## Key polarimetric & radiometric processing features

1. **Four physical L-band polarimetric channels per acquisition cycle**:
   - `NISAR_HHHH_dB_<DDMonYYYY>` — Co-polarized L-band backscatter ($\gamma^0_{HH}$ in dB, $10 \log_{10}(\gamma^0_{HH})$).
   - `NISAR_HVHV_dB_<DDMonYYYY>` — Cross-polarized L-band volume scattering ($\gamma^0_{HV}$ in dB, $10 \log_{10}(\gamma^0_{HV})$).
   - `NISAR_RVI_L_<DDMonYYYY>` — L-band Radar Vegetation Index ($4 \cdot \gamma^0_{HV} / (\gamma^0_{HH} + \gamma^0_{HV}) \in [0.0, 1.5]$).
   - `NISAR_RATIO_<DDMonYYYY>` — Co-to-cross-polarization power ratio ($\gamma^0_{HH} / \gamma^0_{HV} \in [0.0, 50.0]$).
2. **2D adaptive cross-track range-edge de-tapering**:
   - Removes antenna pattern roll-off along West and East swath edges while leaving top/bottom along-track frame borders untouched.
3. **Along-track slit healing & $C^1$ distance-squared feathering**:
   - Heals diagonal frame boundary slits via 1D vertical morphological closing and blends overlapping orbital tracks within each 12-day NISAR cycle using inverse-distance-squared weighting.
4. **Zero-dependency resilience**:
   - Uses `asf_search` and `h5py` when installed, and automatically falls back to the NASA ASF Search GeoJSON REST API and GDAL's native HDF5 subdataset driver if optional packages are absent.

---

## Quick-start CLI examples

```bash
# 1. Launch interactive setup wizard (zero arguments):
python 1b_NISAR_preprocessor/run_nisar_preprocessor.py

# 2. Run full NISAR preprocessing for a single orbit track (e.g., Netherlands Orbit 88):
python 1b_NISAR_preprocessor/run_nisar_preprocessor.py --track NL/orbit_88 -s 2025-10-15 -e 2026-09-15 --stage A

# 3. Run full NISAR preprocessing for all orbits of a country (e.g., Poland or Portugal):
python 1b_NISAR_preprocessor/run_nisar_preprocessor.py --country PL -s 2025-10-15 -e 2026-09-15 --stage A

# 4. Run only Stage 3 (Re-stack existing cycle mosaics into S1-aligned BigTIFF):
python 1b_NISAR_preprocessor/run_nisar_preprocessor.py --track PT/orbit_52 --stage 3
```
