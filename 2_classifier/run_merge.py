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
import os
import sys
from pathlib import Path

# Ensure local and modules imports work cleanly
script_dir = Path(__file__).resolve().parent
modules_dir = script_dir / "modules"
for p in [script_dir, modules_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import importlib
merge_mod = importlib.import_module("multi_orbit_merger")


def interactive_merge_wizard():
    print("""
================================================================================
       AIML CropMapper Cloud - Phase 4 National Multi-Orbit Merger Wizard
================================================================================
 Merges classified orbit tracks into a seamless national GeoTIFF crop map.
================================================================================""")
    project_root = script_dir.parent
    base_dir = Path(os.environ.get("AIML_WORKING_DIR", str(project_root / "workingDirs")))
    countries = []
    if base_dir.exists():
        for d in sorted(base_dir.iterdir()):
            if d.is_dir() and len(d.name) in (2, 3) and any(d.glob("orbit_*")):
                countries.append(d.name)
    default_country = countries[0] if countries else "NL"
    c_str = "/".join(countries) if countries else "NL/PL/FR/PT/ES/DE"
    country = input(f" Enter country code ({c_str}) [{default_country}]: ").strip().upper() or default_country
    seg_mode = input(" Enter segmentation mode (slic/sam/lpis) [slic]: ").strip().lower() or 'slic'
    method = input(" Enter blending method (confidence/priority/majority) [confidence]: ").strip().lower() or 'confidence'
    suffix = f"_mlpxgb_presto_{seg_mode}"
    merge_mod.run_merge_for_country(
        country=country,
        seg_mode=seg_mode,
        suffix=suffix,
        method=method
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Multi-Orbit National Mosaic & Seamless Merging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_merge.py
  python run_merge.py --country NL --seg_mode slic
  python run_merge.py --country PL --seg_mode lpis
"""
    )
    parser.add_argument('-c', '--country', default=None, help="Country code (e.g. NL, PL, FR, PT, ES, DE)")
    parser.add_argument(
        '--classifier',
        default='auto',
        choices=['auto', 'mlpxgb_presto', 'mlpxgb_presto_s1s2', 's1s2', 'presto_s1', 'mlp', 'xgb', 'otb'],
        help="Classifier model: 'mlpxgb_presto' [Default SOTA], 'mlpxgb_presto_s1s2', 'presto_s1', 'mlp', 'xgb', 'otb', 'auto' (default: auto)"
    )
    parser.add_argument('--seg_mode', default='slic', choices=['slic', 'sam', 'lpis'], help="Segmentation mode: 'slic', 'sam', 'lpis' (default: slic)")
    parser.add_argument('--method', default='confidence', choices=['confidence', 'priority', 'majority'], help="Blending method across overlapping tracks (default: confidence)")

    args = parser.parse_args()
    if not args.country:
        interactive_merge_wizard()
        return

    suffix = ""
    if args.classifier in ['mlpxgb_presto', 'auto']:
        suffix = f"_mlpxgb_presto_{args.seg_mode}"
    elif args.classifier in ['mlpxgb_presto_s1s2', 's1s2']:
        suffix = f"_mlpxgb_presto_s1s2_{args.seg_mode}"
    elif args.classifier == 'presto_s1':
        suffix = f"_presto_{args.seg_mode}"
    elif args.classifier == 'mlp':
        suffix = f"_mlp_{args.seg_mode}"
    elif args.classifier == 'xgb':
        suffix = f"_xgb_{args.seg_mode}"
    elif args.classifier == 'otb':
        suffix = f"_{args.seg_mode}"

    merge_mod.run_merge_for_country(
        country=args.country.upper(),
        seg_mode=args.seg_mode,
        suffix=suffix,
        method=args.method
    )


if __name__ == '__main__':
    main()
