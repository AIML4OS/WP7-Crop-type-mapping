# AIML CropMapper - Creodias Y: Drive Mounting Script
# This script runs rclone in the background to mount EODATA:eodata to Y:

# 1. Terminate any existing rclone mount processes
$oldProcesses = Get-Process rclone -ErrorAction SilentlyContinue
if ($oldProcesses) {
    Write-Host "[INFO] Stopping existing rclone mount process..."
    $oldProcesses | Stop-Process -Force
    Start-Sleep -Seconds 2
}

# 2. Define rclone parameters
$rclonePath = "C:\rclone\rclone.exe"
$remotePath = "EODATA:eodata"
$driveLetter = "Y:"

if (Test-Path $rclonePath) {
    Write-Host "[INFO] Mounting $remotePath to $driveLetter in the background..."
    
    # Start process completely detached with hidden window
    Start-Process -FilePath $rclonePath -ArgumentList "mount", "--read-only", $remotePath, $driveLetter -WindowStyle Hidden
    
    # Wait for mount to initialize
    Start-Sleep -Seconds 3
    
    if (Test-Path "$driveLetter\") {
        Write-Host "[SUCCESS] Drive $driveLetter mounted successfully!"
        Get-ChildItem "$driveLetter\" | Select-Object -First 5
    } else {
        Write-Error "[ERROR] Failed to mount drive $driveLetter. Check rclone configuration."
    }
} else {
    Write-Error "[ERROR] rclone.exe not found at $rclonePath"
}
