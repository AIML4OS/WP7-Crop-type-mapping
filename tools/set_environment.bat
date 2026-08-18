@echo off
REM AIML CropMapper - Windows Environment Variable Configuration Script

REM 1. SNAP GPT Executable Path
set SNAP_GPT_EXE=D:/Program Files/esa-snap/bin/gpt.exe

REM 2. SNAP AuxData Directory
set SNAP_AUXDATA_PATH=C:/Users/Administrator/.snap/auxdata

REM 3. Workspace Working Directory
set AIML_WORKING_DIR=D:/AIML_CropMapper_Cloud/workingDir

REM 4. Auxiliary Files Directory
set AIML_AUX_DIR=D:/AIML_CropMapper_Cloud/auxiliary_files

REM 5. Sentinel Repository Paths (CREODIAS / Local)
set S1_REPO_PATH=Y:/Sentinel-1/SAR/IW_GRDH_1S
set S2_REPO_PATH=Y:/Sentinel-2/MSI/L2A

REM 6. OpenMP / PyTorch Concurrency Settings
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=4

echo [OK] AIML CropMapper environment variables configured successfully.
