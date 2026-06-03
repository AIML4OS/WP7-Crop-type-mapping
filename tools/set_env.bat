@echo off
REM ============================================================================
REM AIML CropMapper Pipeline - Windows Environment Setup Script
REM ============================================================================
REM Edit the paths below to match your actual system installation directories.
REM Run this script before running the Python pipeline.

REM Set the exact path to SNAP gpt.exe
set SNAP_GPT_EXE=D:\Program Files\esa-snap\bin\gpt.exe

REM Path to SNAP auxiliary files (where orbit files are cached)
set SNAP_AUXDATA_PATH=C:\Users\Administrator\.snap\auxdata

REM Path to the raw Sentinel-1 GRD SAFE repository directory
set S1_REPO_PATH=Y:\Sentinel-1\SAR\IW_GRDH_1S

REM Output workspace directory for intermediate and final rasters
set AIML_WORKING_DIR=D:\AIML_CropMapper_Cloud\workingDir

REM Path to project's auxiliary files directory
set AIML_AUX_DIR=D:\AIML_CropMapper_Cloud\auxiliary_files

REM CDSE Credentials for S1 Downloader (Optional)
REM set CDSE_USERNAME=your_username@email.com
REM set CDSE_PASSWORD=your_password

echo [INFO] AIML Environment Variables configured for Windows.
echo SNAP_GPT_EXE: %SNAP_GPT_EXE%
echo AIML_WORKING_DIR: %AIML_WORKING_DIR%
