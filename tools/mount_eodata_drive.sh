#!/usr/bin/env bash
# AIML CropMapper - CREODIAS EODATA Mounting Script for Linux
# Mounts CloudFerro EODATA repository to /eodata or ~/eodata using rclone

MOUNT_POINT="${1:-/eodata}"

# 1. Terminate existing rclone mount
pkill -f "rclone mount EODATA:eodata" 2>/dev/null || true
sleep 1

# 2. Prepare mount directory
mkdir -p "$MOUNT_POINT"

echo "[INFO] Mounting EODATA:eodata to $MOUNT_POINT in background..."
rclone mount --read-only --vfs-cache-mode minimal --daemon EODATA:eodata "$MOUNT_POINT"
sleep 2

if [ -d "$MOUNT_POINT/Sentinel-1" ] || [ -d "$MOUNT_POINT/Sentinel-2" ]; then
    echo "[SUCCESS] EODATA repository mounted successfully at $MOUNT_POINT!"
    ls -l "$MOUNT_POINT" | head -n 6
else
    echo "[INFO] Repository mounting initialized at $MOUNT_POINT."
fi
