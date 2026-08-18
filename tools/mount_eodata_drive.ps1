# AIML CropMapper - CREODIAS EODATA Y: Drive Mounting Script for Windows (PowerShell)
# Mounts CloudFerro EODATA repository to drive Y: using rclone

# 1. Terminate any existing rclone mount processes
$oldProcesses = Get-Process rclone -ErrorAction SilentlyContinue
if ($oldProcesses) {
    Write-Host "[INFO] Stopping existing rclone mount process..."
    $oldProcesses | Stop-Process -Force
    Start-Sleep -Seconds 2
}

# 2. Define rclone parameters
$rcloneCandidates = @("C:\rclone\rclone.exe", "rclone.exe", "$env:USERPROFILE\rclone\rclone.exe")
$rclonePath = $null
foreach ($cand in $rcloneCandidates) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) {
        $rclonePath = $cand
        break
    } elseif (Test-Path $cand) {
        $rclonePath = $cand
        break
    }
}

$remotePath = "EODATA:eodata"
$driveLetter = "Y:"

if ($rclonePath) {
    Write-Host "[INFO] Mounting $remotePath to $driveLetter in background using $rclonePath..."
    Start-Process -FilePath $rclonePath -ArgumentList "mount", "--read-only", "--vfs-cache-mode", "minimal", $remotePath, $driveLetter -WindowStyle Hidden
    Start-Sleep -Seconds 3

    if (Test-Path "$driveLetter\") {
        Write-Host "[SUCCESS] Drive $driveLetter mounted successfully!"
        Get-ChildItem "$driveLetter\" | Select-Object -First 5
    } else {
        Write-Warning "[WARNING] Checking drive $driveLetter... Please wait a moment for rclone to complete handshake."
    }
} else {
    Write-Error "[ERROR] rclone executable not found. Please install rclone from https://rclone.org/."
}
