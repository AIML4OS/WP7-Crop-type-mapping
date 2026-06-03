#!/bin/bash
# ============================================================================
# AIML CropMapper Pipeline - Linux Environment Setup Script
# ============================================================================
# Edit the paths below to match your actual system installation directories.
# Run this script using `source set_env.sh` before running the Python pipeline.

# Set the exact path to SNAP gpt
export SNAP_GPT_EXE="/usr/local/esa-snap/bin/gpt"

# Path to SNAP auxiliary files (where orbit files are cached)
export SNAP_AUXDATA_PATH="$HOME/.snap/auxdata"

# Path to the raw Sentinel-1 GRD SAFE repository directory
export S1_REPO_PATH="/mnt/sentinel1/SAR/IW_GRDH_1S"

# Output workspace directory for intermediate and final rasters
export AIML_WORKING_DIR="/home/user/AIML_CropMapper_Cloud/workingDir"

# Path to project's auxiliary files directory
export AIML_AUX_DIR="/home/user/AIML_CropMapper_Cloud/auxiliary_files"

# CDSE Credentials for S1 Downloader (Optional)
# export CDSE_USERNAME="your_username@email.com"
# export CDSE_PASSWORD="your_password"

echo "[INFO] AIML Environment Variables configured for Linux."
echo "SNAP_GPT_EXE: $SNAP_GPT_EXE"
echo "AIML_WORKING_DIR: $AIML_WORKING_DIR"
