#!/usr/bin/env bash
# AIML CropMapper - Linux Environment Variable Configuration Script

export SNAP_GPT_EXE="/usr/local/bin/gpt"
export SNAP_AUXDATA_PATH="/data/snap_auxdata"
export AIML_WORKING_DIR="/data/projects/WP7-Crop-type-mapping/workingDirs"
export AIML_AUX_DIR="/data/projects/WP7-Crop-type-mapping/auxiliary_files"
export S1_REPO_PATH="/eodata/Sentinel-1/SAR/IW_GRDH_1S"
export S2_REPO_PATH="/eodata/Sentinel-2/MSI/L2A"
export KMP_DUPLICATE_LIB_OK="TRUE"
export OMP_NUM_THREADS="8"

echo "[OK] AIML CropMapper Linux environment variables configured."
