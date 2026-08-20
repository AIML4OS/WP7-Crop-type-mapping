#!/usr/bin/env python
"""
================================================================================
AIML CropMapper Cloud - Phase 4 Multi-Orbit National Mosaic & Merging
================================================================================
Merges classified rasters and probability confidence maps from overlapping
satellite tracks into a single seamless national GeoTIFF crop map.

Features:
  - Multi-track Confidence Blending: Assigns overlapping pixels to the orbit with higher confidence.
  - Morphological Sieve Filtering: Eliminates isolated single-pixel noise clumps (<10 pixels).
  - Cropland Masking: Automatically clips results to the official national agricultural mask.
  - Multi-scale Pyramids: Builds external .ovr overviews for instant GIS rendering.

Execution Examples:
  # 1. Merge SLIC classification maps for the Netherlands:
  python run_merge.py --country NL --seg_mode slic

  # 2. Merge LPIS cadastral classification maps for Poland:
  python run_merge.py --country PL --seg_mode lpis

  # 3. Merge Meta AI SAM classification maps for Portugal:
  python run_merge.py --country PT --seg_mode sam
================================================================================
"""

import argparse
import sys
from pathlib import Path

# Ensure local and modules imports work cleanly
script_dir = Path(__file__).resolve().parent
modules_dir = script_dir / "modules"
for p in [script_dir, modules_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import importlib
merge_mod = importlib.import_module("2_merge_classifications")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Multi-Orbit National Mosaic & Seamless Merging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_merge.py --country NL --seg_mode slic
  python run_merge.py --country PL --seg_mode lpis
"""
    )
    parser.add_argument('-c', '--country', required=True, help="Country code (e.g. NL, PL, FR, PT, ES, DE)")
    parser.add_argument('--seg_mode', default='slic', choices=['slic', 'sam', 'lpis'], help="Segmentation mode: 'slic', 'sam', 'lpis' (default: slic)")
    parser.add_argument('--method', default='confidence', choices=['confidence', 'priority', 'majority'], help="Blending method across overlapping tracks (default: confidence)")

    args = parser.parse_args()
    merge_mod.run_merge_for_country(
        country=args.country.upper(),
        seg_mode=args.seg_mode
    )


if __name__ == '__main__':
    main()
