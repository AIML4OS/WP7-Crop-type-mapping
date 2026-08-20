# AI Agent Guidelines & Project Rules (AGENTS.md)

This document specifies mandatory development guidelines, architectural rules, and file protection policies for all AI coding agents working on the **AIML CropMapper Cloud** repository.

---

## 🚫 1. Protected Directories (Strictly Read-Only)

> [!IMPORTANT]
> **DO NOT modify, edit, rename, or delete any files located in `Archive_scripts/` directories.**

The following folders are frozen historical archives and reference implementations:
* `1_Sentinel-1_preprocessor/Archive_scripts/`
* `1a_Sentinel-2_preprocessor/Archive_scripts/`
* `2_classifier/Archive_scripts/`
* `scratch/`

These scripts must remain unchanged for backward reference, auditability, and reproducibility.

---

## 🏗️ 2. Architectural & Directory Structure Standards

1. **Toolbox Roots Must Remain Clean**:
   * The root of each toolbox directory (`1_Sentinel-1_preprocessor/`, `1a_Sentinel-2_preprocessor/`, `2_classifier/`) must contain **only**:
     * Unified master runner scripts (`run_s1_preprocessor.py`, `run_s2_preprocessor.py`, `run_classifier.py`, `run_merge.py`).
     * Active configuration files (`config_s1.json`, `config_s2.json`, `config_*.example.json`).
     * The `modules/` directory and the `Archive_scripts/` directory.

2. **Internal Module Placement (`modules/`)**:
   * All internal processing, extraction, calibration, modeling, and merging scripts must be placed inside the local `modules/` subdirectory.
   * File names must follow clean, semantic Python naming conventions (`snake_case`) without numeric step prefixes (e.g., `s1_calibration_creodias.py`, `s2_time_series.py`, `classifier_mlpxgb_presto.py`, `multi_orbit_merger.py`).

---

## 💻 3. CLI & Interactive Menu Standards

1. **Dual Execution Modes**:
   * **Zero-argument invocation**: Running `python run_*.py` without arguments must launch a user-friendly interactive setup wizard guiding the user through track discovery, segmentation mode, classifier model, and stage selection.
   * **Direct CLI invocation**: Every script must support full non-interactive execution with explicit CLI flags (e.g., `--track`, `--country`, `--classifier`, `--seg_mode`, `--source`, `--stage`).

2. **Documentation & Example Headers**:
   * Every `run_*.py` script must maintain a comprehensive, copy-pasteable `#` execution examples block in English at the top of the file docstring.
   * All user-facing documentation, comments, and CLI help messages must be written in clear English.

---

## 🧪 4. Code Quality, Serialization & Testing

1. **Serialization / Pickle Compatibility**:
   * When modifying or loading ML models with `joblib` or `pickle`, ensure classes (e.g., `EnsembleClassifier`, `TorchMLPClassifier`) are registered in `sys.modules['__main__']` and module aliases to guarantee unpickling across different entry points.

2. **Syntax & Compilation Verification**:
   * Always verify that all modified `.py` files compile cleanly without syntax errors:
     ```powershell
     python -m py_compile <modified_files>
     ```
