#!/usr/bin/env python
"""
================================================================================
AIML CropMapper Cloud - Multimodal Crop Classifier & National Merger
================================================================================
Unified master orchestrator for object-based multimodal crop classification.

Features:
  - Multimodal Data Footprint: Computes exact spatial intersection of S1 SAR & S2 Optical stacks.
  - Segmentation Modes (--seg_mode):
      * 'slic': Fast buffered Simple Linear Iterative Clustering superpixels.
      * 'sam' : Meta AI Segment Anything vision foundation model with edge refinement.
      * 'lpis': Official cadastral agricultural parcel vectors (.shp, .gpkg).
  - Feature Extraction:
      * Multimodal SAR statistics + Optical multi-temporal reflectances.
      * NASA Harvest Presto 128-d multi-temporal geospatial foundation embeddings.
  - Classification Architectures (--classifier):
      * 'mlpxgb_presto' (Default): Soft-voting ensemble of Deep PyTorch MLP + XGBoost GBDT + Presto.
      * 'presto_s1'              : Single-radar S1-only Presto ANN model.
      * 'otb'                    : Orfeo ToolBox machine learning models (Random Forest / SVM).
      * 'mlp'                    : Pure Deep PyTorch MLP classifier.
      * 'xgb'                    : Pure XGBoost GBDT classifier.
  - Post-Processing & Assessment:
      * Bayesian prior calibration against official crop acreage statistics.
      * Morphological sieve noise removal and agricultural cropland masking.
      * Automated validation metrics export to Excel (OA, Kappa, F1-scores, Confusion Matrix).
  - Multi-Orbit National Mosaic (Phase 4):
      * Confidence-weighted seamless blending across overlapping satellite tracks.

Execution Examples:
  # 1. Interactive setup wizard (simply run with zero arguments):
  python run_classifier.py

  # 2. Full automated pipeline (SLIC + Multimodal Deep MLP + XGBoost + Presto):
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode slic --stage A

  # 3. Full automated pipeline using Meta AI SAM deep vision segmentation:
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode sam --stage A

  # 4. Full automated pipeline using official LPIS cadastral parcel vectors:
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode lpis --lpis_vector path/to/parcels.gpkg --stage A

  # 5. Single-radar S1-only Presto ANN classification:
  python run_classifier.py --track NL/orbit_88 --classifier presto_s1 --seg_mode slic --stage A

  # 6. Orfeo ToolBox (Random Forest / SVM) machine learning classification:
  python run_classifier.py --track NL/orbit_88 --classifier otb --seg_mode slic --stage A

  # 7. Sequential automated classification for all orbits in a country:
  python run_classifier.py --country NL --classifier mlpxgb_presto --seg_mode slic --stage A

  # 8. Run only individual stages:
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 0  # Multimodal footprint only
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 1  # Segmentation only
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 2  # Stratified sample split only
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 3  # Feature extraction & Presto embeddings
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 4  # Model training (MLP + XGBoost)
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 5  # Tile-based object inference & Bayesian priors
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 6  # Agricultural cropland masking
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 7  # Accuracy assessment & Excel report export

  # 9. Merge all classified orbits for a country into a seamless national map (Phase 4):
  python run_classifier.py --country NL --seg_mode slic --stage merge
================================================================================
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

# Ensure local and modules imports work cleanly
script_dir = Path(__file__).resolve().parent
modules_dir = script_dir / "modules"
for p in [script_dir, modules_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDirs"))

# Register unpickling compatibility in __main__
try:
    from modules.classifier_mlpxgb_presto import EnsembleClassifier, TorchMLPClassifier
    import modules.classifier_mlpxgb_presto as _cls_mod
    sys.modules['1_classify_MLPXGB_presto_hybrid_S1S2'] = _cls_mod
    sys.modules['classifier_mlpxgb_presto'] = _cls_mod
except Exception:
    pass


def discover_available_tracks() -> List[str]:
    """Scans workingDirs and workingDir to find all available country/orbit directories."""
    tracks = set()
    candidate_bases = [BASE_DIR, Path(r"D:/AIML_CropMapper_Cloud/workingDir")]
    for b in candidate_bases:
        if b.exists():
            for country_dir in sorted(b.iterdir()):
                if country_dir.is_dir() and len(country_dir.name) in [2, 3]:
                    for orbit_dir in sorted(country_dir.glob("orbit_*")):
                        if orbit_dir.is_dir():
                            tracks.add(f"{country_dir.name}/{orbit_dir.name}")
    return sorted(list(tracks))


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

    logging.info(f"\n============================================================")
    logging.info(f" Multimodal Crop Classifier (AIML CropMapper Cloud)")
    logging.info(f" Track            : {norm_track}")
    logging.info(f" Segmentation     : [{seg_mode.upper()}]")
    logging.info(f" Classifier Model : [{classifier_model.upper()}]")
    logging.info(f"============================================================")

    # Route to specialized engines if requested
    if classifier_model == 'otb':
        otb_mod = importlib.import_module("classifier_otb")
        pipeline = otb_mod.ProcessingPipeline(track=norm_track, seg_mode=seg_mode)
        if stage == 'A' or stage is None:
            pipeline.run_all()
        return

    if classifier_model == 'presto_s1':
        s1_ann_mod = importlib.import_module("classifier_presto_s1")
        pipeline = s1_ann_mod.ProcessingPipeline(track=norm_track, seg_mode=seg_mode)
        if stage == 'A' or stage is None:
            pipeline.run_all()
        return

    # Handle pure MLP or pure XGBoost via weight adjustment
    if classifier_model == 'mlp':
        mlp_weight = 1.0
    elif classifier_model == 'xgb':
        mlp_weight = 0.0

    # Primary multimodal SOTA engine (S1 + S2 + Presto + MLP + XGBoost)
    s1s2_mod = importlib.import_module("classifier_mlpxgb_presto")
    pipeline = s1s2_mod.ProcessingPipelineS1S2(
        track=norm_track,
        seg_mode=seg_mode,
        mlp_weight=mlp_weight,
        s1_override=s1_override,
        s2_override=s2_override,
        lpis_vector=lpis_vector
    )

    if stage is None:
        interactive_menu(pipeline, country, norm_track, classifier_model)
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
            merge_mod = importlib.import_module("multi_orbit_merger")
            merge_mod.run_merge_for_country(country, seg_mode=seg_mode)


def run_merge_for_country(country: str, seg_mode: str = 'slic'):
    """Runs multi-orbit national mosaic and merging."""
    logging.info(f"\n============================================================")
    logging.info(f" [Phase 4] Multi-Orbit National Mosaic & Merging for {country.upper()}")
    logging.info(f" Segmentation mode: {seg_mode.upper()}")
    logging.info(f"============================================================")
    merge_mod = importlib.import_module("multi_orbit_merger")
    merge_mod.run_merge_for_country(country.upper(), seg_mode=seg_mode)


def interactive_setup_wizard():
    """Interactive CLI wizard shown when run_classifier.py is called without arguments."""
    print("""
============================================================
 AIML CropMapper Cloud - Multimodal Classification Wizard
============================================================""")

    # Step 1: Select Track
    tracks = discover_available_tracks()
    selected_track = None
    selected_country = None

    if tracks:
        print(" Discovered tracks in working directory:")
        for idx, t in enumerate(tracks, 1):
            print(f"  [{idx}] {t}")
        print("  [C] Enter custom track (e.g. PL/orbit_22)")
        print("  [N] Process entire country (e.g. NL, PL, FR)")
        print("  [Q] Quit")
        choice = input("\n Select track or option [1-%d/C/N/Q] (default: 1): " % len(tracks)).strip().upper()
        if choice == 'Q': return
        elif choice == 'C':
            selected_track = input(" Enter track identifier (e.g. NL/orbit_88): ").strip()
        elif choice == 'N':
            selected_country = input(" Enter country code (e.g. NL, PL, FR): ").strip().upper()
        elif choice.isdigit() and 1 <= int(choice) <= len(tracks):
            selected_track = tracks[int(choice) - 1]
        else:
            selected_track = tracks[0]
    else:
        selected_track = input(" Enter track identifier (e.g. NL/orbit_88) or country (e.g. NL): ").strip()
        if '/' not in selected_track:
            selected_country = selected_track.upper()
            selected_track = None

    # Step 2: Select Segmentation Mode
    print("""
============================================================
 Select Segmentation Mode:
  [1] SLIC Superpixels (Recommended SOTA, fast & scalable)
  [2] Meta AI SAM (Segment Anything foundation model)
  [3] Official LPIS Cadastral Parcels (.shp / .gpkg)
============================================================""")
    seg_choice = input(" Enter choice [1-3] (default: 1): ").strip()
    seg_modes = {'1': 'slic', '2': 'sam', '3': 'lpis'}
    seg_mode = seg_modes.get(seg_choice, 'slic')
    lpis_vector = None
    if seg_mode == 'lpis':
        lpis_vector = input(" Enter path to LPIS parcel vector file (.shp / .gpkg): ").strip()

    # Step 3: Select Classifier Architecture
    print("""
============================================================
 Select Classifier Model:
  [1] Multimodal Fusion (Deep MLP + XGBoost + Presto) [SOTA]
  [2] Single-Radar Presto ANN (S1 SAR only)
  [3] Orfeo ToolBox Machine Learning (Random Forest / SVM)
  [4] Pure PyTorch Deep MLP
  [5] Pure XGBoost GBDT
============================================================""")
    cls_choice = input(" Enter choice [1-5] (default: 1): ").strip()
    cls_models = {'1': 'mlpxgb_presto', '2': 'presto_s1', '3': 'otb', '4': 'mlp', '5': 'xgb'}
    classifier_model = cls_models.get(cls_choice, 'mlpxgb_presto')

    # Step 4: Select Execution Mode
    print("""
============================================================
 Select Execution Mode:
  [A] Run all stages automatically (Stage 0 -> 7)
  [I] Enter interactive stage-by-stage menu
  [0-7] Run specific single stage
  [8] Phase 4: Multi-orbit national merge
============================================================""")
    stage_choice = input(" Enter choice [A/I/0-8] (default: I): ").strip().upper()
    if stage_choice == '' or stage_choice == 'I':
        stage_choice = None

    if selected_country and not selected_track:
        if stage_choice and stage_choice in ['8', 'MERGE']:
            run_merge_for_country(selected_country, seg_mode)
        else:
            country_dir = BASE_DIR / selected_country
            if country_dir.exists():
                orbits = [d.name for d in country_dir.glob("orbit_*") if d.is_dir()]
                for orb_name in orbits:
                    track_path = f"{selected_country}/{orb_name}"
                    run_pipeline(
                        track=track_path,
                        seg_mode=seg_mode,
                        classifier_model=classifier_model,
                        stage=stage_choice,
                        lpis_vector=lpis_vector
                    )
    else:
        run_pipeline(
            track=selected_track,
            seg_mode=seg_mode,
            classifier_model=classifier_model,
            stage=stage_choice,
            lpis_vector=lpis_vector
        )


def interactive_menu(pipeline, country: str, track: str, classifier_model: str = 'mlpxgb_presto'):
    seg_modes = ['slic', 'sam', 'lpis']
    cls_models = ['mlpxgb_presto', 'presto_s1', 'otb', 'mlp', 'xgb']

    while True:
        menu_text = f"""
============================================================
 Multimodal Crop Classifier (AIML CropMapper Cloud)
 Track            : {track}
 Segmentation     : [{pipeline.seg_mode.upper()}]
 Classifier Model : [{classifier_model.upper()}]
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
 [M] Change segmentation mode (Current: {pipeline.seg_mode.upper()})
 [C] Change classifier model (Current: {classifier_model.upper()})
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
            elif choice == 'M':
                idx = (seg_modes.index(pipeline.seg_mode) + 1) % len(seg_modes)
                pipeline.seg_mode = seg_modes[idx]
                print(f"\n    Segmentation mode switched to: {pipeline.seg_mode.upper()}")
            elif choice == 'C':
                idx = (cls_models.index(classifier_model) + 1) % len(cls_models)
                classifier_model = cls_models[idx]
                print(f"\n    Classifier model switched to: {classifier_model.upper()}")
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
    parser = argparse.ArgumentParser(
        description="Unified Multimodal Crop Classifier & National Merger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive setup wizard (zero arguments):
  python run_classifier.py

  # Multimodal Deep MLP + XGBoost + Presto (Recommended SOTA):
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode slic --stage A

  # Vision Foundation Model Segmentation (Meta AI SAM):
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode sam --stage A

  # Cadastral Parcels Segmentation (LPIS):
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode lpis --lpis_vector path/to/parcels.gpkg --stage A

  # Single-radar S1-only Presto ANN:
  python run_classifier.py --track NL/orbit_88 --classifier presto_s1 --seg_mode slic --stage A

  # Orfeo ToolBox Machine Learning:
  python run_classifier.py --track NL/orbit_88 --classifier otb --seg_mode slic --stage A

  # Multi-orbit National Merging:
  python run_classifier.py --country NL --seg_mode slic --stage merge
"""
    )
    parser.add_argument('-t', '--track', default=None, help="Track identifier (e.g. NL/orbit_88, PL/orbit_22)")
    parser.add_argument('-c', '--country', default=None, help="Country code (e.g. NL, PL, FR, PT, ES, DE)")
    parser.add_argument('--stage', default=None, help="Stage to execute: 'A' (all 0-7), '0'..'7', '8' or 'merge'")
    parser.add_argument('--classifier', default='mlpxgb_presto', choices=['mlpxgb_presto', 'presto_s1', 'otb', 'mlp', 'xgb'], help="Classifier model: 'mlpxgb_presto' (default), 'presto_s1', 'otb', 'mlp', 'xgb'")
    parser.add_argument('--seg_mode', default='slic', choices=['slic', 'sam', 'lpis'], help="Segmentation mode: 'slic' (superpixels), 'sam' (Meta AI), 'lpis' (cadastre) (default: slic)")
    parser.add_argument('--mlp_weight', type=float, default=0.65, help="Weight of MLP in fusion ensemble (0.0 to 1.0, default: 0.65)")
    parser.add_argument('--s1_raster', default=None, help="Override path to Sentinel-1 Sigma0 GeoTIFF raster")
    parser.add_argument('--s2_raster', default=None, help="Override path to Sentinel-2 Multi-temporal GeoTIFF raster")
    parser.add_argument('--lpis_vector', default=None, help="Path to official LPIS parcel vector file (.shp, .gpkg)")

    args = parser.parse_args()

    # If run with zero arguments, open the interactive setup wizard!
    if not args.track and not args.country and not args.stage:
        interactive_setup_wizard()
        return

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
