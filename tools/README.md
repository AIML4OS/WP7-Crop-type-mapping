# Data preparation and auxiliary tools (`tools/`)

This directory provides standalone command-line scripts to prepare auxiliary datasets required by the Sentinel-1, Sentinel-2, NISAR, and machine learning classification pipelines.

---

## Tool inventory

```text
tools/
├── 1_download_nuts_boundaries.py       # Downloads GISCO NUTS administrative boundaries
├── 2_prepare_classification_samples.py # Extracts ground truth point samples from LPIS parcel databases
└── 3_build_raster_overviews.py         # Universal GDAL multi-scale pyramid overviews builder
```

> [!NOTE]
> **Integrated pipeline capabilities**:
> - **Cropland masking**: With `--seg_mode lpis`, the classifier automatically queries and rasterizes LPIS parcels on-the-fly for the current orbit footprint during Stage 2, setting non-agricultural pixels to NoData (`0`). Standalone static rasterization is no longer required.
> - **Dynamic Bayesian priors**: The classifier (`classifier_mlpxgb_presto.py`) dynamically computes per-orbit Bayesian class priors (`compute_dynamic_bayesian_priors`) from local training samples and crop area multipliers, automatically zeroing absent crops to eliminate hallucinations. Synchronized `priors.json` is exported automatically by `2_prepare_classification_samples.py`.

---

## 1. Download GISCO NUTS boundaries (`tools/1_download_nuts_boundaries.py`)

Downloads official NUTS0 (country), NUTS1 (major socio-economic regions), and NUTS2 (basic regions for regional policies) boundary shapefiles and GeoJSON layers directly from Eurostat GISCO web services into `auxiliary_files/shapefiles_nuts/`.

### Execution examples:
```powershell
# Download boundaries for a specific country:
python tools/1_download_nuts_boundaries.py -c PT
python tools/1_download_nuts_boundaries.py -c NL
python tools/1_download_nuts_boundaries.py -c PL

# Download boundaries for all European countries:
python tools/1_download_nuts_boundaries.py --all
```

---

## 2. Generate classification training samples (`tools/2_prepare_classification_samples.py`)

Extracts inside-field point ground truth samples (`samples.shp`) and synchronized crop class priors (`priors.json`) from national parcel polygon datasets (e.g. Dutch BRP, Polish ARiMR, Portuguese ISIP).

### Key capabilities:
- **Streaming SQL extraction**: Uses `pyogrio` to query massive multi-gigabyte GeoPackages (e.g. 2.8 GB / 2.33 million parcels) in seconds without memory overflow.
- **Adaptive negative buffering**: Applies 10 m $\rightarrow$ 5 m negative buffers to ensure sampling points are situated well within field interiors, eliminating boundary mixed pixels.
- **Multi-sensor class mapping**: Includes pre-configured crop mappings (e.g. 18-class Dutch BRP taxonomy optimized for Sentinel-1 C-band, Sentinel-2 optical, and NISAR L-band SAR).
- **Target coordinate system**: Outputs points in `EPSG:3857` (Web Mercator) with standard `crop_id`, `crop_name`, and `geometry` attributes.

### Execution examples:
```powershell
# Extract 18 crop classes from Dutch BRP 2026 GeoPackage:
python tools/2_prepare_classification_samples.py -c NL --input auxiliary_files/shapefiles_samples/NL/gewaspercelen_concept_2026.gpkg --crop_col gewas --max_samples_per_class 2000

# Extract points from Polish ARiMR parcel database:
python tools/2_prepare_classification_samples.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME --max_samples_per_class 3000

# Extract points from Portuguese ISIP parcel database:
python tools/2_prepare_classification_samples.py -c PT --input path/to/isip.shp --crop_col CATEGORIA --min_area_ha 0.3
```

---

## 3. Multi-scale pyramid overviews builder (`tools/3_build_raster_overviews.py`)

Builds external GDAL multi-scale pyramid overview layers (`[2, 4, 8, 16, 32, 64]`) with DEFLATE/LZW compression. Enables instant rendering, zooming, and panning in QGIS, ArcGIS Pro, and web map viewers without loading 100+ GB rasters into RAM.

### Execution examples:
```powershell
# Build overviews for a single GeoTIFF:
python tools/3_build_raster_overviews.py -i workingDirs/NL/orbit_37/processed_raster/NL_orbit_37_S2_timeseries.tif

# Build overviews for all rasters in a directory:
python tools/3_build_raster_overviews.py -d workingDirs/NL/orbit_37/processed_raster/

# Build overviews for all national products and orbit stacks for a country:
python tools/3_build_raster_overviews.py -c NL
```
