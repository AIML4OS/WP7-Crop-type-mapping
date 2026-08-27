# Data preparation and auxiliary tools (`tools/`)

This directory provides standalone command-line scripts to prepare auxiliary datasets required by the Sentinel-1, Sentinel-2, and machine learning classification pipelines.

---

## Tool inventory

```text
tools/
├── 1_download_nuts_boundaries.py    # Downloads GISCO NUTS administrative boundaries
├── 2_build_agricultural_mask.py     # Generates standardized high-resolution cropland masks
├── 3_prepare_classification_samples.py # Extracts ground truth point samples from parcel databases
├── 4_generate_crop_priors.py        # Computes statistical crop acreage priors for Bayesian calibration
└── 5_build_raster_overviews.py      # Universal GDAL multi-scale pyramid overviews builder
```

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

## 2. Build agricultural cropland mask (`tools/2_build_agricultural_mask.py`)

Generates standardized binary cropland masks (`EPSG:3857`, 10 m pixel resolution) saved in `auxiliary_files/raster_files/AgriMasks/{COUNTRY}/`. Used in Stage 6 of the classifier to suppress non-agricultural areas (forests, urban areas, water bodies).

### Execution examples:
```powershell
# From LPIS cadastral parcel vectors:
python tools/2_build_agricultural_mask.py -c NL --lpis path/to/brp.gpkg
python tools/2_build_agricultural_mask.py -c PT --lpis path/to/isip.gpkg
python tools/2_build_agricultural_mask.py -c PL --lpis path/to/arimr.shp

# From Copernicus CLMS / HRL raster layers:
python tools/2_build_agricultural_mask.py -c NL
```

---

## 3. Generate classification training samples (`tools/3_prepare_classification_samples.py`)

Extracts inside-field point ground truth samples (`samples.shp`) from national parcel polygon datasets, ensuring negative buffers to prevent field boundary contamination.

### Execution examples:
```powershell
# Extract points from Dutch BRP parcel database:
python tools/3_prepare_classification_samples.py -c NL --input path/to/brp.gpkg --crop_col GEWAS --min_area_ha 0.2

# Extract points from Polish ARiMR parcel database:
python tools/3_prepare_classification_samples.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME --max_samples_per_class 3000

# Extract points from Portuguese ISIP parcel database:
python tools/3_prepare_classification_samples.py -c PT --input path/to/isip.gpkg --crop_col OCUP_SOLO --min_area_ha 0.2
```

---

## 4. Compute Bayesian crop acreage priors (`tools/4_generate_crop_priors.py`)

Calculates real-world statistical crop area proportions from national parcel registries and exports `auxiliary_files/shapefiles_samples/{COUNTRY}/priors.json`. Used in Stage 5 to calibrate machine learning posterior probabilities against real agricultural acreage.

### Execution examples:
```powershell
python tools/4_generate_crop_priors.py -c NL --input path/to/brp.gpkg --crop_col GEWAS
python tools/4_generate_crop_priors.py -c PT --input path/to/isip.gpkg --crop_col OCUP_SOLO
python tools/4_generate_crop_priors.py -c PL --input path/to/arimr.shp --crop_col CROP_NAME
```

---

## 5. Multi-scale pyramid overviews builder (`tools/5_build_raster_overviews.py`)

Builds external GDAL multi-scale pyramid overview layers (`[2, 4, 8, 16, 32, 64]`) with DEFLATE/LZW compression. Enables instant rendering, zooming, and panning in QGIS, ArcGIS Pro, and web map viewers without loading 100+ GB rasters into RAM.

### Execution examples:
```powershell
# Build overviews for a single GeoTIFF:
python tools/5_build_raster_overviews.py -i workingDirs/PT/orbit_52/1_input_stacks/PT_orbit_52_S2_timeseries.tif

# Build overviews for all rasters in a directory:
python tools/5_build_raster_overviews.py -d workingDirs/PT/orbit_52/1_input_stacks/

# Build overviews for all national products and orbit stacks for a country:
python tools/5_build_raster_overviews.py -c PT
```
