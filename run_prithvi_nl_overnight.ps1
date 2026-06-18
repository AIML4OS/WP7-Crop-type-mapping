# run_prithvi_nl_overnight.ps1
# Runs the Prithvi-SAR classification pipeline for both NL orbits sequentially.
# Usage:
#   powershell -File run_prithvi_nl_overnight.ps1 -seg_mode sam
#   powershell -File run_prithvi_nl_overnight.ps1 -seg_mode slic
#   powershell -File run_prithvi_nl_overnight.ps1 -seg_mode lpis

param (
    [string]$seg_mode = "sam"
)

$python_path = "C:\Users\Administrator\miniforge3\envs\aiml_env\python.exe"

Write-Output "=== Starting Prithvi-SAR Overnight Processing for Netherlands ==="
Write-Output "Selected Segmentation Mode: $seg_mode"
Get-Date

Write-Output "`n[1/2] Processing NL/orbit_88..."
& $python_path 2_classifier/1_classify_prithvi_sar.py --track NL/orbit_88 --stage A --seg_mode $seg_mode

Write-Output "`n[2/2] Processing NL/orbit_161..."
& $python_path 2_classifier/1_classify_prithvi_sar.py --track NL/orbit_161 --stage A --seg_mode $seg_mode

Write-Output "`n=== Processing Completed! ==="
Get-Date
