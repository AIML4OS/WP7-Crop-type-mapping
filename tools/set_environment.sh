#!/usr/bin/env bash
# AIML CropMapper - Linux Environment Variable Configuration Script

export SNAP_GPT_EXE="/usr/local/esa-snap/bin/gpt"
export SNAP_AUXDATA_PATH="$HOME/.snap/auxdata"
export AIML_WORKING_DIR="$HOME/AIML_CropMapper_Cloud/workingDir"
export AIML_AUX_DIR="$HOME/AIML_CropMapper_Cloud/auxiliary_files"
export S1_REPO_PATH="/eodata/Sentinel-1/SAR/IW_GRDH_1S"
export S2_REPO_PATH="/eodata/Sentinel-2/MSI/L2A"
export KMP_DUPLICATE_LIB_OK="TRUE"
export OMP_NUM_THREADS="4"

echo "[OK] AIML CropMapper Linux environment variables configured."
