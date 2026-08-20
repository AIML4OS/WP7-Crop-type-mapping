# AI Agent Guidelines & Engineering Standards (AGENTS.md)

This document specifies mandatory development guidelines, architectural rules, coding standards, and testing protocols for all AI coding agents and developers working on the **AIML CropMapper Cloud** repository.

---

## 🚫 1. Protected Directories (Strictly Read-Only)

> [!IMPORTANT]
> **DO NOT modify, edit, rename, or delete any files located in `Archive_scripts/` or `scratch/` directories.**

The following directories are frozen historical archives and reference implementations:
* `1_Sentinel-1_preprocessor/Archive_scripts/`
* `1a_Sentinel-2_preprocessor/Archive_scripts/`
* `2_classifier/Archive_scripts/`
* `scratch/`

These scripts must remain completely untouched for backward reference, auditability, and scientific reproducibility.

---

## 🏗️ 2. Architectural & Directory Structure Standards

1. **Toolbox Roots Must Remain Clean**:
   * The root of each toolbox directory (`1_Sentinel-1_preprocessor/`, `1a_Sentinel-2_preprocessor/`, `2_classifier/`) must contain **only**:
     * Unified master runner scripts (`run_s1_preprocessor.py`, `run_s2_preprocessor.py`, `run_classifier.py`, `run_merge.py`).
     * Active configuration files (`config_s1.json`, `config_s2.json`, `config_*.example.json`).
     * The `modules/` directory and the `Archive_scripts/` directory.

2. **Internal Module Placement (`modules/`)**:
   * All internal processing, extraction, calibration, modeling, and merging scripts must be placed inside the local `modules/` subdirectory.
   * File names must follow clean, semantic Python naming conventions (`snake_case`) without numeric step prefixes:
     * `s1_calibration_creodias.py`, `s1_calibration_cdse.py`, `s1_coregistration.py`, `s1_stack_clip.py`
     * `s2_extract_creodias.py`, `s2_download_cdse.py`, `s2_time_series.py`, `s2_mosaic_stack.py`, `s2_pipeline.py`
     * `classifier_mlpxgb_presto.py`, `classifier_presto_s1.py`, `classifier_otb.py`, `multi_orbit_merger.py`, `presto_model.py`

---

## 💻 3. CLI & Interactive Wizard Standards

1. **Dual Execution Modes**:
   * **Zero-argument invocation**: Running `python run_*.py` without arguments must launch a user-friendly interactive setup wizard guiding the user step-by-step through track discovery, segmentation mode, classifier model, and stage selection.
   * **Direct CLI invocation**: Every script must support full non-interactive execution with explicit CLI flags (e.g., `--track`, `--country`, `--classifier`, `--seg_mode`, `--source`, `--stage`).

2. **Documentation & Example Headers**:
   * Every `run_*.py` script must maintain a comprehensive, copy-pasteable `#` execution examples block in English at the top of the file docstring.
   * All user-facing documentation, comments, log messages, and CLI help strings must be written in clear English.

---

## 🛰️ 4. Geospatial Data Handling & BigTIFF Standards

1. **GDAL Dataset Creation & Compression**:
   * Always use standard BigTIFF creation options for multi-band satellite rasters:
     ```python
     options = ['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES']
     ```
   * Ensure coordinate reference system is set to `EPSG:3857` (Web Mercator) and pixel resolution is exactly 10.0 m.
   * Always close GDAL dataset handles (`ds = None` or `del ds`) to flush buffers to disk and prevent file lock conflicts on Windows.

2. **Multi-Scale Pyramid Overviews**:
   * All final optical and SAR BigTIFF stacks must have external pyramid overviews built (`[2, 4, 8, 16, 32, 64]`) with LZW/DEFLATE compression for instant QGIS/ArcGIS rendering.

---

## ⚡ 5. Concurrency, Multiprocessing & OpenMP Safety

1. **Preventing OpenMP / MKL Thread Conflicts**:
   * When using `concurrent.futures.ProcessPoolExecutor` or `multiprocessing`, ensure internal OpenMP/MKL multi-threading is constrained to prevent CPU thread thrashing and deadlocks:
     ```python
     os.environ["OMP_NUM_THREADS"] = "1"
     os.environ["MKL_NUM_THREADS"] = "1"
     os.environ["OPENBLAS_NUM_THREADS"] = "1"
     os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
     os.environ["NUMEXPR_NUM_THREADS"] = "1"
     ```

---

## 🧠 6. Machine Learning Model Serialization (Pickle/Joblib)

1. **Namespace & Module Aliases for Deserialization**:
   * When loading models with `joblib.load()` or `pickle.load()`, Python looks up custom class definitions in `__main__` or the module path where they were pickled.
   * Always register custom model classes (e.g., `EnsembleClassifier`, `TorchMLPClassifier`, `TorchANNClassifier`) in `sys.modules['__main__']` and provide module aliases:
     ```python
     import sys
     main_mod = sys.modules.get('__main__')
     if main_mod:
         setattr(main_mod, 'EnsembleClassifier', EnsembleClassifier)
         setattr(main_mod, 'TorchMLPClassifier', TorchMLPClassifier)
     sys.modules['1_classify_MLPXGB_presto_hybrid_S1S2'] = sys.modules[__name__]
     sys.modules['classifier_mlpxgb_presto'] = sys.modules[__name__]
     ```

---

## 🧪 7. Verification, Compilation & Testing Protocol

Before committing or concluding any code modification, agents MUST execute the following verification steps:

1. **Syntax & Bytecode Compilation**:
   * Verify all modified Python files compile cleanly:
     ```powershell
     python -m py_compile 1_Sentinel-1_preprocessor/run_s1_preprocessor.py 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py 2_classifier/run_classifier.py 2_classifier/run_merge.py
     ```

2. **Import & Entry Point Sanity Check**:
   * Test that entry points and modules import without broken dependencies or missing symbols:
     ```powershell
     python 2_classifier/run_classifier.py --help
     python 1_Sentinel-1_preprocessor/run_s1_preprocessor.py --help
     python 1a_Sentinel-2_preprocessor/run_s2_preprocessor.py --help
     ```

---

## 📦 8. Git & Remote Repository Sync Protocol

1. **Dual Remote Synchronization**:
   * Always push changes to both configured remotes:
     ```powershell
     git push origin AIML_CropMapper
     git push personal AIML_CropMapper
     ```
2. **Data Exclusion (.gitignore Compliance)**:
   * Never commit large raster files (`.tif`, `.SAFE`, `.zip`), model checkpoints (`.pkl`, `.pt`), or local working directory contents (`workingDir/`, `eodata/`).
