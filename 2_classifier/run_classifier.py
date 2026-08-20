#!/usr/bin/env python
"""
run_classifier.py - Unified Multimodal Crop Classifier & National Merger.

Provides a standardized English CLI and interactive menu for:
  Stage 0: Multimodal Data Footprint Generation (S1 & S2 intersection)
  Stage 1: Object-based Segmentation (SLIC Superpixels / SAM AI / LPIS Cadastre)
  Stage 2: Stratified Train/Validation Sample Split
  Stage 3: Multimodal Feature Extraction (S1 Sigma0 + S2 Optical + Presto/Prithvi Embeddings)
  Stage 4: Unified Fusion Ensemble Training (Deep PyTorch MLP + XGBoost GBDT)
  Stage 5: Tile-based Object Inference with Bayesian Priors
  Stage 6: Agricultural Area Masking (LPIS / Global Land Cover)
  Stage 7: Accuracy Assessment & Excel Metrics Report (F1, Kappa, OA)
  Stage 8: Phase 4 Multi-orbit National Mosaic & Seamless Merging (Merge)

Usage examples:
  # 1. Run all stages for a single orbit (SLIC + Multimodal MLP+XGBoost+Presto):
  python run_classifier.py --track NL/orbit_88 --stage A

  # 2. Run all stages with Segment Anything (SAM) vision model:
  python run_classifier.py --track NL/orbit_88 --seg_mode sam --stage A

  # 3. Merge all classified orbits for an entire country into a national map:
  python run_classifier.py --country NL --stage merge

  # 4. Interactive menu:
  python run_classifier.py --track NL/orbit_88
"""

import argparse
import importlib
import logging
import os
import pathlib
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Ensure local imports work cleanly
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDir"))


def run_pipeline(
    track: str,
    seg_mode: str = 'slic',
    classifier_model: str = 'mlpxgb_presto',
    stage: Optional[str] = None,
    mlp_weight: float = 0.65,
    s1_override: Optional[str] = None,
    s2_override: Optional[str] = None,
    lpis_vector: Optional[str] = None
):
    norm_track = track.replace('\\', '/')
    country = norm_track.split('/')[0].upper() if '/' in norm_track else track.upper()

    # Route to specialized engines if requested
    if classifier_model == 'otb':
        otb_mod = importlib.import_module("1_classify_otb")
        # Instantiate and run OTB classifier
        pipeline = otb_mod.ProcessingPipeline(track=norm_track, seg_mode=seg_mode)
        if stage == 'A' or stage is None:
            pipeline.run_all()
        return

    if classifier_model == 'presto_s1':
        s1_ann_mod = importlib.import_module("1_classify_ann_presto_hybrid")
        pipeline = s1_ann_mod.ProcessingPipeline(track=norm_track, seg_mode=seg_mode)
        if stage == 'A' or stage is None:
            pipeline.run_all()
        return

    # Primary multimodal SOTA engine (S1 + S2 + Presto + MLP + XGBoost)
    s1s2_mod = importlib.import_module("1_classify_MLPXGB_presto_hybrid_S1S2")
    pipeline = s1s2_mod.ProcessingPipelineS1S2(
        track=norm_track,
        seg_mode=seg_mode,
        mlp_weight=mlp_weight,
        s1_override=s1_override,
        s2_override=s2_override,
        lpis_vector=lpis_vector
    )

    if stage is None:
        interactive_menu(pipeline, country, norm_track)
    else:
        choice = stage.strip().upper()
        if choice == 'A':
            pipeline.stage_0_generate_footprint(False)
            pipeline.stage_1_segmentation(False)
            pipeline.stage_2_split_samples(False)
            pipeline.stage_3_selection(False)
            pipeline.stage_4_train_classifier(False)
            pipeline.stage_5_classify_vector(True)
            pipeline.stage_6_mask_classification(True)
            pipeline.stage_7_calculate_metrics()
        elif choice == '0': pipeline.stage_0_generate_footprint(True)
        elif choice == '1': pipeline.stage_1_segmentation(True)
        elif choice == '2': pipeline.stage_2_split_samples(True)
        elif choice == '3': pipeline.stage_3_selection(True)
        elif choice == '4': pipeline.stage_4_train_classifier(True)
        elif choice == '5': pipeline.stage_5_classify_vector(True)
        elif choice == '6': pipeline.stage_6_mask_classification(True)
        elif choice == '7': pipeline.stage_7_calculate_metrics()
        elif choice in ['8', 'MERGE']:
            merge_mod = importlib.import_module("2_merge_classifications")
            merge_mod.run_merge_for_country(country, seg_mode=seg_mode)


def run_merge_for_country(country: str, seg_mode: str = 'slic'):
    """Runs multi-orbit national mosaic and merging."""
    logging.info(f"\n============================================================")
    logging.info(f" [Phase 4] Multi-Orbit National Mosaic & Merging for {country.upper()}")
    logging.info(f" Segmentation mode: {seg_mode.upper()}")
    logging.info(f"============================================================")
    merge_mod = importlib.import_module("2_merge_classifications")
    merge_mod.run_merge_for_country(country.upper(), seg_mode=seg_mode)


def interactive_menu(pipeline, country: str, track: str):
    while True:
        menu_text = f"""
============================================================
 Multimodal Crop Classifier (AIML CropMapper Cloud)
 Track            : {track}
 Segmentation     : [{pipeline.seg_mode.upper()}]
 Fusion Ensemble  : [{pipeline.mlp_weight:.2f} Deep MLP + {1.0-pipeline.mlp_weight:.2f} XGBoost + Presto]
============================================================
 [0] Stage 0: Generate multimodal data footprint (S1 & S2)
 [1] Stage 1: Object-based segmentation ({pipeline.seg_mode.upper()})
 [2] Stage 2: Stratified train/validation sample split
 [3] Stage 3: Multimodal feature extraction (S1 + S2 + Presto)
 [4] Stage 4: Train unified fusion ensemble (Deep MLP + XGB)
 [5] Stage 5: Object-based inference with Bayesian priors
 [6] Stage 6: Apply agricultural area masks
 [7] Stage 7: Calculate accuracy metrics & export Excel report
 -----------------------------------------------------------
 [8] Phase 4: Multi-orbit national mosaic & seamless merge
 [A] Run all classification stages automatically (0 -> 7)
 [Q] Quit
============================================================
 Enter choice: """
        try:
            choice = input(menu_text).strip().upper()
            if choice == '0': pipeline.stage_0_generate_footprint(True)
            elif choice == '1': pipeline.stage_1_segmentation(True)
            elif choice == '2': pipeline.stage_2_split_samples(True)
            elif choice == '3': pipeline.stage_3_selection(True)
            elif choice == '4': pipeline.stage_4_train_classifier(True)
            elif choice == '5': pipeline.stage_5_classify_vector(True)
            elif choice == '6': pipeline.stage_6_mask_classification(True)
            elif choice == '7': pipeline.stage_7_calculate_metrics()
            elif choice == '8': run_merge_for_country(country, pipeline.seg_mode)
            elif choice == 'A':
                pipeline.stage_0_generate_footprint(False)
                pipeline.stage_1_segmentation(False)
                pipeline.stage_2_split_samples(False)
                pipeline.stage_3_selection(False)
                pipeline.stage_4_train_classifier(False)
                pipeline.stage_5_classify_vector(True)
                pipeline.stage_6_mask_classification(True)
                pipeline.stage_7_calculate_metrics()
            elif choice == 'Q': break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting classifier.")
            break


def main():
    parser = argparse.ArgumentParser(description="Unified Multimodal Crop Classifier & National Merger.")
    parser.add_argument('-t', '--track', default=None, help="Track identifier, e.g. NL/orbit_88, PL/orbit_22")
    parser.add_argument('-c', '--country', default=None, help="Country code, e.g. NL, PL, FR, PT")
    parser.add_argument('--stage', default=None, help="Stage to execute: 'A' (all), '0'..'7', '8' or 'merge'")
    parser.add_argument('--seg_mode', default='slic', choices=['slic', 'sam', 'lpis'], help="Segmentation mode (default: slic)")
    parser.add_argument('--classifier', default='mlpxgb_presto', choices=['mlpxgb_presto', 'presto_s1', 'otb', 'mlp', 'xgb'], help="Classifier model (default: mlpxgb_presto)")
    parser.add_argument('--mlp_weight', type=float, default=0.65, help="Weight of MLP in fusion ensemble (0.0 to 1.0, default: 0.65)")
    parser.add_argument('--s1_raster', default=None, help="Override path to Sentinel-1 Sigma0 GeoTIFF raster")
    parser.add_argument('--s2_raster', default=None, help="Override path to Sentinel-2 Multi-temporal GeoTIFF raster")
    parser.add_argument('--lpis_vector', default=None, help="Path to official LPIS parcel vector file (.shp, .gpkg)")

    args = parser.parse_args()

    if args.stage and args.stage.upper() in ['8', 'MERGE']:
        country = args.country
        if not country and args.track:
            country = args.track.replace('\\', '/').split('/')[0].upper()
        if not country:
            parser.error("--country or --track is required for merging.")
        run_merge_for_country(country, args.seg_mode)
        return

    if not args.track:
        if args.country:
            # Run for all orbits in country sequentially
            country_dir = BASE_DIR / args.country.upper()
            if country_dir.exists():
                orbits = [d.name for d in country_dir.glob("orbit_*") if d.is_dir()]
                for orb_name in orbits:
                    track_path = f"{args.country.upper()}/{orb_name}"
                    run_pipeline(
                        track=track_path,
                        seg_mode=args.seg_mode,
                        classifier_model=args.classifier,
                        stage=args.stage,
                        mlp_weight=args.mlp_weight,
                        s1_override=args.s1_raster,
                        s2_override=args.s2_raster,
                        lpis_vector=args.lpis_vector
                    )
                return
        parser.error("Either --track (-t) or --country (-c) must be specified.")

    run_pipeline(
        track=args.track,
        seg_mode=args.seg_mode,
        classifier_model=args.classifier,
        stage=args.stage,
        mlp_weight=args.mlp_weight,
        s1_override=args.s1_raster,
        s2_override=args.s2_raster,
        lpis_vector=args.lpis_vector
    )


if __name__ == '__main__':
    main()
