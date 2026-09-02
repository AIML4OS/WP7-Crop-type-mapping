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
      * 'mlpxgb_presto' (Default): SOTA Dual-Tier Soft-Voting Ensemble (Deep PyTorch MLP + XGBoost GBDT + Presto).
      * 'mlp'                    : Pure Deep PyTorch MLP classifier [S1 + S2].
      * 'xgb'                    : Pure XGBoost GBDT classifier [S1 + S2].
      * 'presto_s1' (Archived)   : Single-radar S1-only Presto ANN model.
      * 'otb' (Archived)         : Orfeo ToolBox machine learning models.
  - Post-Processing & Assessment:
      * Bayesian prior calibration against official crop acreage statistics.
      * Morphological sieve noise removal and agricultural cropland masking.
      * Automated validation metrics export to Excel (OA, Kappa, F1-scores, Confusion Matrix).
  - Multi-Orbit National Mosaic (Phase 4):
      * Confidence-weighted seamless blending across overlapping satellite tracks.

Execution Examples:
  # 1. Interactive setup wizard (simply run with zero arguments):
  python run_classifier.py

  # 2. Full automated pipeline (SLIC + Multimodal Deep MLP + XGBoost + Presto [SOTA]):
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode slic --stage A

  # 3. Full automated pipeline using Meta AI SAM deep vision segmentation:
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode sam --stage A

  # 4. Full automated pipeline using official LPIS cadastral parcel vectors:
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode lpis --lpis_vector path/to/parcels.gpkg --stage A

  # 5. Pure PyTorch Deep MLP classifier:
  python run_classifier.py --track NL/orbit_88 --classifier mlp --seg_mode slic --stage A

  # 6. Pure XGBoost GBDT classifier:
  python run_classifier.py --track NL/orbit_88 --classifier xgb --seg_mode slic --stage A

  # 7. Sequential automated classification for all orbits in a country:
  python run_classifier.py --country NL --classifier mlpxgb_presto --seg_mode slic --stage A

  # 8. Run only individual stages:
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 1  # Multimodal footprint only
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 2  # Segmentation only
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 3  # Stratified sample split only
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 4  # Feature extraction & Presto embeddings
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 5  # Model training (MLP + XGBoost)
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 6  # Tile-based object inference & Bayesian priors
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 7  # Agricultural cropland masking
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --stage 8  # Accuracy assessment & Excel report export

  # 9. Merge all classified orbits for a country into a seamless national map (Phase 4):
  python run_merge.py --country NL --seg_mode slic
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

# Ensure local, modules and Archive_scripts imports work cleanly
script_dir = Path(__file__).resolve().parent
modules_dir = script_dir / "modules"
archive_dir = script_dir / "Archive_scripts"
for p in [script_dir, modules_dir, archive_dir]:
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

    # Route to specialized / archived engines if requested
    if classifier_model == 'otb':
        logging.warning("[DEPRECATED] 'otb' model has been archived to Archive_scripts/ and the standalone OTB binaries have been retired.")
        logging.warning("[RECOMMENDED] Use 'mlpxgb_presto' for state-of-the-art multimodal deep learning classification.")
        try:
            otb_mod = importlib.import_module("classifier_otb")
            pipeline = otb_mod.ProcessingPipeline(track=norm_track, seg_mode=seg_mode)
            if stage == 'A' or stage is None:
                pipeline.run_all()
        except Exception as e:
            logging.error(f"Failed to run archived OTB classifier: {e}")
        return

    if classifier_model == 'presto_s1':
        logging.warning("[DEPRECATED] 'presto_s1' SAR-only model has been archived to Archive_scripts/.")
        logging.warning("[RECOMMENDED] Use 'mlpxgb_presto' for multimodal SAR+Optical classification.")
        try:
            s1_ann_mod = importlib.import_module("classifier_presto_s1")
            pipeline = s1_ann_mod.ProcessingPipeline(track=norm_track, seg_mode=seg_mode)
            if stage == 'A' or stage is None:
                pipeline.run_all()
        except Exception as e:
            logging.error(f"Failed to run archived Presto-S1 classifier: {e}")
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
            if hasattr(pipeline, 'run_all'):
                pipeline.run_all()
            else:
                pipeline.stage_1_generate_footprint(False)
                pipeline.stage_2_segmentation(False)
                pipeline.stage_3_split_samples(False)
                pipeline.stage_4_selection(False)
                pipeline.stage_5_train_classifier(False)
                pipeline.stage_6_classify_vector(True)
                pipeline.stage_7_mask_classification(True)
                pipeline.stage_8_calculate_metrics()
        elif choice in ['1', '0']: pipeline.stage_1_generate_footprint(True)
        elif choice == '2': pipeline.stage_2_segmentation(True)
        elif choice == '3': pipeline.stage_3_split_samples(True)
        elif choice == '4': pipeline.stage_4_selection(True)
        elif choice == '5': pipeline.stage_5_train_classifier(True)
        elif choice == '6': pipeline.stage_6_classify_vector(True)
        elif choice == '7': pipeline.stage_7_mask_classification(True)
        elif choice == '8': pipeline.stage_8_calculate_metrics()
        else:
            logging.error(f"Unknown stage '{stage}'. Use 'A' (all) or '1'..'8'.")


def interactive_setup_wizard():
    print("""
================================================================================
       AIML CropMapper Cloud - Interactive Setup Wizard (Phase 3 & 4)
================================================================================
 Welcome! This wizard will guide you through setting up crop classification
 and national multi-orbit merging step-by-step.
================================================================================""")

    available_tracks = discover_available_tracks()
    countries = sorted(list(set(t.split('/')[0] for t in available_tracks)))

    # Step 1: Track / Country Discovery
    print("\n Available Countries & Tracks:")
    for c in countries:
        c_tracks = [t for t in available_tracks if t.startswith(f"{c}/")]
        print(f"   * {c}: {len(c_tracks)} orbit(s) ({', '.join(c_tracks)})")

    selected_track = None
    selected_country = None

    print("\n Selection options:")
    print("  [1] Select a specific satellite orbit track (e.g. PT/orbit_45, NL/orbit_88)")
    print("  [2] Select an entire country to process all its orbits sequentially")
    print("  [3] Merge classified tracks into national mosaic (Phase 4)")
    mode_choice = input(" Enter choice [1-3] (default: 1): ").strip()

    if mode_choice == '2':
        selected_country = input(f" Enter country code ({'/'.join(countries)}): ").strip().upper()
    elif mode_choice == '3':
        sel_c = input(f" Enter country code to merge ({'/'.join(countries)}): ").strip().upper()
        sel_s = input(" Enter segmentation mode used for classification (slic/sam/lpis) [slic]: ").strip().lower() or 'slic'
        merger_mod = importlib.import_module("multi_orbit_merger")
        merger = merger_mod.MultiOrbitMerger(country=sel_c, seg_mode=sel_s)
        merger.run_national_mosaic()
        return
    else:
        print("\n Discovered Tracks:")
        for idx, tr in enumerate(available_tracks, start=1):
            print(f"   [{idx}] {tr}")
        tr_choice = input(f" Enter track number [1-{len(available_tracks)}] or custom track: ").strip()
        if tr_choice.isdigit() and 1 <= int(tr_choice) <= len(available_tracks):
            selected_track = available_tracks[int(tr_choice) - 1]
        elif tr_choice:
            selected_track = tr_choice
        else:
            selected_track = available_tracks[0] if available_tracks else "NL/orbit_88"

    # Step 2: Select Segmentation Mode
    print("""
============================================================
 Select Segmentation Architecture:
  [1] SLIC Superpixels (Fast, edge-constrained, default)
  [2] Meta AI SAM (Segment Anything foundation vision model)
  [3] LPIS Cadastral Parcels (Official agricultural vector data)
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
 Select Classifier Architecture:
  [1] Multimodal Dual-Tier Fusion (Deep MLP + XGBoost + Presto) [S1 + S2] [SOTA - Recommended]
  [2] Pure PyTorch Deep MLP [S1 + S2]
  [3] Pure XGBoost GBDT [S1 + S2]
============================================================""")
    cls_choice = input(" Enter choice [1-3] (default: 1): ").strip()
    cls_models = {'1': 'mlpxgb_presto', '2': 'mlp', '3': 'xgb'}
    classifier_model = cls_models.get(cls_choice, 'mlpxgb_presto')

    # Step 4: Select Execution Mode
    print("""
============================================================
 Select Execution Mode:
  [A] Run all stages automatically (Stage 1 -> 8)
  [I] Enter interactive stage-by-stage menu

 Or select a specific stage directly:
  [1] Stage 1: Generate multimodal data footprint (S1 & S2)
  [2] Stage 2: Object-based segmentation (SLIC / SAM / LPIS)
  [3] Stage 3: Stratified train/validation sample split
  [4] Stage 4: Multimodal feature extraction (S1 + S2 + Presto)
  [5] Stage 5: Train unified fusion ensemble (Deep MLP + XGB)
  [6] Stage 6: Object-based inference with Bayesian priors
  [7] Stage 7: Apply agricultural area masks
  [8] Stage 8: Calculate accuracy metrics & export Excel report
============================================================""")
    stage_choice = input(" Enter choice [A/I/1-8] (default: A): ").strip().upper()
    if stage_choice == '' or stage_choice == 'I':
        stage_choice = None

    if selected_country and not selected_track:
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
    cls_models = ['mlpxgb_presto', 'mlp', 'xgb']

    cls_labels = {
        'mlpxgb_presto': 'MLPXGB_PRESTO [S1 + S2 SOTA]',
        'mlp': 'PYTORCH_MLP [S1 + S2]',
        'xgb': 'XGBOOST [S1 + S2]'
    }

    while True:
        cls_name = cls_labels.get(classifier_model, classifier_model.upper())
        menu_text = f"""
============================================================
 Multimodal Crop Classifier (AIML CropMapper Cloud)
 Track            : {track}
 Segmentation     : [{pipeline.seg_mode.upper()}]
 Classifier Model : [{cls_name}]
============================================================
 [1] Stage 1: Generate multimodal data footprint (S1 & S2)
 [2] Stage 2: Object-based segmentation ({pipeline.seg_mode.upper()})
 [3] Stage 3: Stratified train/validation sample split
 [4] Stage 4: Multimodal feature extraction (S1 + S2 + Presto)
 [5] Stage 5: Train unified fusion ensemble (Deep MLP + XGB)
 [6] Stage 6: Object-based inference with Bayesian priors
 [7] Stage 7: Apply agricultural area masks
 [8] Stage 8: Calculate accuracy metrics & export Excel report
 -----------------------------------------------------------
 [M] Change segmentation mode (Current: {pipeline.seg_mode.upper()})
 [C] Change classifier model (Current: {classifier_model.upper()})
 [A] Run all classification stages automatically (1 -> 8)
 [Q] Quit
============================================================
 Enter choice: """
        try:
            choice = input(menu_text).strip().upper()
            if choice in ['1', '0']: pipeline.stage_1_generate_footprint(True)
            elif choice == '2': pipeline.stage_2_segmentation(True)
            elif choice == '3': pipeline.stage_3_split_samples(True)
            elif choice == '4': pipeline.stage_4_selection(True)
            elif choice == '5': pipeline.stage_5_train_classifier(True)
            elif choice == '6': pipeline.stage_6_classify_vector(True)
            elif choice == '7': pipeline.stage_7_mask_classification(True)
            elif choice == '8': pipeline.stage_8_calculate_metrics()
            elif choice == 'M':
                idx = (seg_modes.index(pipeline.seg_mode) + 1) % len(seg_modes)
                pipeline.seg_mode = seg_modes[idx]
                print(f"\n    Segmentation mode switched to: {pipeline.seg_mode.upper()}")
            elif choice == 'C':
                idx = (cls_models.index(classifier_model) + 1) % len(cls_models) if classifier_model in cls_models else 0
                classifier_model = cls_models[idx]
                print(f"\n    Classifier model switched to: {classifier_model.upper()}")
            elif choice == 'A':
                if hasattr(pipeline, 'run_all'):
                    pipeline.run_all()
                else:
                    pipeline.stage_1_generate_footprint(False)
                    pipeline.stage_2_segmentation(False)
                    pipeline.stage_3_split_samples(False)
                    pipeline.stage_4_selection(False)
                    pipeline.stage_5_train_classifier(False)
                    pipeline.stage_6_classify_vector(True)
                    pipeline.stage_7_mask_classification(True)
                    pipeline.stage_8_calculate_metrics()
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

  # Pure PyTorch Deep MLP classifier:
  python run_classifier.py --track NL/orbit_88 --classifier mlp --seg_mode slic --stage A

  # Pure XGBoost GBDT classifier:
  python run_classifier.py --track NL/orbit_88 --classifier xgb --seg_mode slic --stage A

  # Multi-orbit National Merging (Phase 4):
  python run_merge.py --country NL --seg_mode slic
"""
    )
    parser.add_argument('-t', '--track', default=None, help="Track identifier (e.g. NL/orbit_88, PL/orbit_22)")
    parser.add_argument('-c', '--country', default=None, help="Country code (e.g. NL, PL, FR, PT, ES, DE)")
    parser.add_argument('--stage', default=None, help="Stage to execute: 'A' (all 1-8), or single stage '1'..'8' (legacy '0'..'7' supported)")
    parser.add_argument('--classifier', default='mlpxgb_presto',
                        choices=['mlpxgb_presto', 'mlp', 'xgb', 'presto_s1', 'otb'],
                        help="Classifier model: 'mlpxgb_presto' [S1+S2 SOTA] (default), 'mlp' [S1+S2], 'xgb' [S1+S2] (archived: 'presto_s1', 'otb')")
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
