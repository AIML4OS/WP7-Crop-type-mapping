#!/usr/bin/env python
"""
run_merge.py - Phase 4 Multi-Orbit National Mosaic & Seamless Merging.

Merges classification maps and confidence rasters from overlapping Sentinel-1/Sentinel-2
orbits into a unified national GeoTIFF map.

Usage examples:
  python run_merge.py --country NL --seg_mode slic
  python run_merge.py --country PL --seg_mode slic
"""

import argparse
import sys
from pathlib import Path

# Ensure local imports work cleanly
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import importlib
merge_mod = importlib.import_module("2_merge_classifications")


def main():
    parser = argparse.ArgumentParser(description="Multi-Orbit National Mosaic & Seamless Merging.")
    parser.add_argument('-c', '--country', required=True, help="Country code, e.g. NL, PL, FR, PT")
    parser.add_argument('--seg_mode', default='slic', choices=['slic', 'sam', 'lpis'], help="Segmentation mode (default: slic)")
    parser.add_argument('--method', default='confidence', choices=['confidence', 'priority', 'majority'], help="Blending method")

    args = parser.parse_args()
    merge_mod.run_merge_for_country(
        country=args.country.upper(),
        seg_mode=args.seg_mode
    )


if __name__ == '__main__':
    main()
