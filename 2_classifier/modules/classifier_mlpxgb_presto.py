#!/usr/bin/env python
"""
classifier_mlpxgb_presto.py - Multimodal Sentinel-1 (Sigma0) + Sentinel-2 Enhanced Crop Classification
using NASA Harvest Presto Joint Foundation Embeddings, Red-Edge & SAR Physical Features,
High-Throughput Vectorized Chunk I/O, Outlier Noise Filtering, and Spatial Uncertainty Estimation (Entropy & Margin).

Pipeline Overview:
  Stage 1: Generate Multimodal Data Footprint (S1 + S2 valid data intersection)
  Stage 2: Multimodal Image Segmentation (SLIC / SAM / LPIS)
  Stage 3: Sample Point Split (70% learn / 30% control)
  Stage 4: Enhanced Feature Extraction (Vectorized Chunk I/O + Joint Presto + Red-Edge/SAR Physical Features)
  Stage 5: Train Unified MLP + XGBoost Fusion Ensemble (with Outlier Label Cleaning)
  Stage 6: Object-Based Tile Inference with Bayesian Priors & Spatial Uncertainty (Shannon Entropy & Margin)
  Stage 7: Apply Agricultural & Footprint Masks (4 rasters: class, conf, entropy, margin + Overviews)
  Stage 8: Calculate Out-of-Bag Validation Metrics & Generate Styled Excel Report (.xlsx)

Execution examples:
  # Mode 1: SLIC Superpixel Segmentation (Fast, no external vector required):
  python run_classifier.py --track NL/orbit_88 --classifier mlpxgb_presto --seg_mode slic --stage A

  # Mode 2: Official Cadastral LPIS Parcel Segmentation:
  python run_classifier.py --track PT/orbit_147 --classifier mlpxgb_presto --seg_mode lpis --stage A
  python run_classifier.py --track PL/orbit_12 --classifier mlpxgb_presto --seg_mode lpis --stage A
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
from pathlib import Path
import subprocess
import sys
import shutil
import shlex
import json
import math
import re
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Tuple

# Add classifier directory and project root to sys.path to allow importing single_file_presto
classifier_dir = str(Path(__file__).resolve().parent)
project_root = str(Path(__file__).resolve().parent.parent)
for p in [classifier_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal, ogr, osr, gdalconst
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
import joblib
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not found. Install torch for Presto embeddings and MLP.")

# XGBoost / GBDT
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier

# Scikit-image for SLIC
try:
    from skimage.segmentation import felzenszwalb, slic
    from skimage.util import img_as_float
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Global project directories
base_dir = Path(os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDirs"))
aux_dir = Path(os.environ.get("AIML_AUX_DIR", r"D:/AIML_CropMapper_Cloud/auxiliary_files"))
presto_dir = aux_dir / "Presto_models"

TOTAL_STAGES = 8


def get_gdal_creation_options(predictor: int = 2, zstd_level: int = 3) -> list:
    """Returns high-performance BigTIFF creation options with ZSTD compression (falling back to DEFLATE if needed)."""
    return ['COMPRESS=ZSTD', f'PREDICTOR={predictor}', f'ZSTD_LEVEL={zstd_level}', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS']


# =====================================================================
# 1. HELPERS: DATE PARSING & PRIORS
# =====================================================================

def parse_month_from_description(desc: str) -> int:
    """Parses month index (0-11) from raster band descriptions."""
    months_map = {
        'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
        'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
    }
    match = re.search(r'_(?:\d+)?([a-zA-Z]{3})\d{4}_', str(desc))
    if match:
        mon_str = match.group(1).lower()
        return months_map.get(mon_str, 0)
    return 0


def get_crop_aggregation(country: str, learn_shp_path: Optional[Path]) -> dict:
    return {}


def _match_crop_keyword(keywords: list, text: str) -> bool:
    """Case-insensitive token/keyword matcher with word-boundary protection for short tokens."""
    import re
    text_lower = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        if len(kw_lower) <= 4:
            # Word boundary with optional plural 's'
            pat = r'\b' + re.escape(kw_lower) + r's?\b'
            if re.search(pat, text_lower):
                return True
        else:
            if kw_lower in text_lower:
                return True
    return False


def get_crop_area_multiplier(cid: int, crop_name: str = "", expected_classes: int = 16) -> float:
    """
    Returns the typical physical parcel size in square meters (m^2) for a given crop.
    Directly incorporates the multi-crop area table from FullImageClassification.py,
    supporting multi-lingual keywords (Polish, English, Portuguese, Dutch, French, German, Spanish, Italian),
    broad aggregated classes, and adaptive default fallbacks based on classification taxonomy depth.
    """
    name = (crop_name or "").lower().strip()

    # 1. Broad aggregated classes (from FullImageClassification.py MR / WOSU configs)
    if _match_crop_keyword(['zboza ozime', 'uprawy ozime', 'oleiste', 'gorczycowate'], name):
        return 70000.0
    if _match_crop_keyword(['zboza jare', 'uprawy jare', 'straczkowe', 'warzywa i okopowe', 'warzywa_okopowe'], name):
        return 40000.0
    if _match_crop_keyword(['owoce i krzewy', 'owoce_krzewy'], name):
        return 20000.0
    if _match_crop_keyword(['zboza'], name):
        return 100000.0

    # 2. Multi-lingual Keyword Matching across EN, PT, PL, NL, FR, DE, ES, IT
    # Tier 1: Large broadacre & grasslands (50,000 - 70,000 m^2)
    if _match_crop_keyword([
        'grass', 'pasture', 'pastagem', 'prado', 'fallow', 'pousio', 'trawa', 'trawiast',
        'ugor', 'braak', 'blijvend', 'tijdelijk', 'prairie', 'weide', 'clover', 'trevo',
        'koniczyna', 'klaver', 'lucerne', 'luzerna', 'lucerna', 'luzerne', 'pastos', 'barbecho', 'tiuz'
    ], name):
        return 70000.0
    if _match_crop_keyword(['maize', 'milho', 'kukurydza', 'corn', 'mais', 'maiz'], name):
        return 70000.0
    if _match_crop_keyword(['wheat', 'trigo', 'pszenica', 'tarwe', 'ble', 'weizen'], name):
        return 70000.0
    if _match_crop_keyword(['rapeseed', 'canola', 'colza', 'rzepak', 'koolzaad', 'raps'], name):
        return 60000.0
    if _match_crop_keyword(['triticale', 'pszenzyto', 'koorn'], name):
        return 50000.0
    if _match_crop_keyword(['rice', 'arroz', 'ryz', 'riz', 'reis', 'riso'], name):
        return 50000.0
    if _match_crop_keyword(['sunflower', 'girassol', 'slonecznik', 'tournesol', 'zonnebloem', 'sonnenblume', 'girasole'], name):
        return 50000.0

    # Tier 2: Standard cereals & field crops (30,000 - 40,000 m^2)
    if _match_crop_keyword(['barley', 'cevada', 'jeczmien', 'gerst', 'orge', 'gerste', 'cebada', 'orzo'], name):
        return 40000.0
    if _match_crop_keyword(['rye', 'centeio', 'zyto', 'rogge', 'seigle', 'roggen', 'centeno', 'segale'], name):
        return 40000.0
    if _match_crop_keyword(['oats', 'aveia', 'owies', 'haver', 'avoine', 'hafer', 'avena'], name):
        return 40000.0
    if _match_crop_keyword(['sorghum', 'sorgo'], name):
        return 40000.0
    if _match_crop_keyword(['sugar beet', 'beet', 'beterraba', 'burak', 'suikerbiet', 'betterave', 'zuckerruebe', 'remolacha', 'barbabietola'], name):
        return 40000.0
    if _match_crop_keyword(['mieszanki zbozowe'], name):
        return 40000.0
    if _match_crop_keyword(['cotton', 'algodao', 'algodon', 'coton'], name):
        return 40000.0
    if _match_crop_keyword(['olive', 'olival', 'oliva', 'oliven', 'olivier', 'olivo'], name):
        return 30000.0
    if _match_crop_keyword(['lubin', 'lupin', 'tremo'], name):
        return 30000.0
    if _match_crop_keyword(['pea', 'ervilha', 'groch', 'erwt', 'pois', 'erbse', 'guisante', 'pisello'], name):
        return 30000.0

    # Tier 3: Orchards, vineyards & tubers (20,000 - 25,000 m^2)
    if _match_crop_keyword(['vineyard', 'vine', 'vinha', 'vinhedo', 'winnica', 'wijngaard', 'vigne', 'weinberg', 'vinedo', 'vigneto'], name):
        return 25000.0
    if _match_crop_keyword(['citrus', 'citrinos', 'orange', 'lemon', 'laranja', 'limao', 'citricos'], name):
        return 25000.0
    if _match_crop_keyword(['orchard', 'fruit', 'nut', 'pomar', 'fruto', 'sad', 'jablon', 'sliwa', 'wisnia', 'appel', 'peer', 'verger', 'obst', 'frutales', 'arvores'], name):
        return 25000.0
    if _match_crop_keyword(['potato', 'batata', 'ziemniak', 'aardappel', 'pomme de terre', 'kartoffel', 'patata'], name):
        return 20000.0

    # Tier 4: Medium & niche field crops (10,000 - 15,000 m^2)
    if _match_crop_keyword(['soybean', 'soja', 'soya'], name):
        return 15000.0
    if _match_crop_keyword(['gryka', 'buckwheat', 'sarrasin', 'buchweizen'], name):
        return 15000.0
    if _match_crop_keyword(['hemp', 'konopie', 'cannabis', 'chanvre', 'canamo'], name):
        return 15000.0
    if _match_crop_keyword(['flax', 'linseed', 'vlas', 'lin', 'len', 'lino'], name):
        return 10000.0
    if _match_crop_keyword(['bean', 'feijao', 'fasola', 'boon', 'haricot', 'bohne', 'alubia', 'fagiolo'], name):
        return 10000.0
    if _match_crop_keyword(['porzeczka', 'currant'], name):
        return 10000.0
    if _match_crop_keyword(['nursery', 'ornamental', 'viveiro', 'szkolka', 'sier', 'boomkwekerij', 'pepiniere', 'baumschule', 'vivero'], name):
        return 10000.0
    if _match_crop_keyword(['vegetable', 'hortalica', 'legume', 'leguminosa', 'warzyw', 'groente', 'maraichage', 'gemuese', 'hortalizas', 'verdura'], name):
        return 10000.0

    # Tier 5: Horticultural, vegetable & intensive crops (2,000 - 5,000 m^2)
    if _match_crop_keyword(['onion', 'cebola', 'cebula', 'ui', 'uien', 'oignon', 'zwiebel', 'cebolla', 'cipolla'], name):
        return 5000.0
    if _match_crop_keyword(['strawberry', 'morango', 'truskawka', 'aardbei', 'fraise', 'erdbeere', 'fresa', 'fragola'], name):
        return 5000.0
    if _match_crop_keyword(['cabbage', 'brassica', 'couve', 'kapusta', 'kool', 'chou', 'kohl', 'col', 'cavolo'], name):
        return 5000.0
    if _match_crop_keyword(['flower bulbs', 'bloembollen', 'tulpen', 'tulipan', 'bulb', 'bulbos'], name):
        return 5000.0
    if _match_crop_keyword(['chicory', 'witlof', 'cichorei', 'cykoria', 'endive'], name):
        return 5000.0
    if _match_crop_keyword(['gorczyca', 'mustard', 'moutarde', 'senf'], name):
        return 5000.0
    if _match_crop_keyword(['bob', 'bobik', 'faba'], name):
        return 5000.0
    if _match_crop_keyword(['melon', 'sandia', 'melancia', 'arbuz'], name):
        return 5000.0
    if _match_crop_keyword(['pumpkin', 'squash', 'dynia', 'abobora'], name):
        return 5000.0
    if _match_crop_keyword(['asparagus', 'asperge', 'szparagi', 'esparrago'], name):
        return 3000.0
    if _match_crop_keyword(['carrot', 'cenoura', 'marchew', 'peen', 'carotte', 'moehre', 'zanahoria', 'carota'], name):
        return 3000.0
    if _match_crop_keyword(['tobacco', 'tabaco', 'tyton', 'tabak', 'tabac'], name):
        return 3000.0
    if _match_crop_keyword(['garlic', 'czosnek', 'alho', 'ail', 'knoblauch', 'ajo'], name):
        return 3000.0
    if _match_crop_keyword(['berry', 'aronia', 'blueberry', 'raspberry', 'mirtilo', 'framboesa', 'amora', 'borowka', 'malina', 'beere', 'arandano'], name):
        return 3000.0
    if _match_crop_keyword(['tomato', 'tomate', 'pomidor', 'tomaat'], name):
        return 2000.0
    if _match_crop_keyword(['cucumber', 'pepino', 'ogorek', 'komkommer', 'concombre', 'gurke', 'cetriolo'], name):
        return 2000.0
    if _match_crop_keyword(['pepper', 'papryka', 'pimento', 'poivron', 'pimiento'], name):
        return 2000.0
    if _match_crop_keyword(['leszczyna', 'hazelnut'], name):
        return 2000.0

    # 3. Country-Specific Class ID Fallbacks (e.g. Standard 37 Classes in Poland)
    area_thresholds_pl = {
        1: 3000, 2: 5000, 3: 5000, 4: 40000, 5: 5000, 6: 10000, 7: 5000, 8: 30000, 9: 15000,
        10: 20000, 11: 30000, 12: 40000, 13: 5000, 14: 70000, 15: 2000, 16: 30000, 17: 5000,
        18: 3000, 19: 40000, 20: 2000, 21: 40000, 22: 2000, 23: 10000, 24: 25000, 25: 70000,
        26: 10000, 27: 50000, 28: 2000, 29: 60000, 30: 3000, 31: 15000, 32: 70000, 33: 5000,
        34: 3000, 35: 5000, 36: 20000, 37: 40000
    }
    if cid in area_thresholds_pl:
        return float(area_thresholds_pl[cid])

    # 4. Adaptive default agricultural fallback based on taxonomy depth (from FullImageClassification.py line 120)
    return 50000.0 if expected_classes < 10 else 10000.0


def compute_dynamic_bayesian_priors(
    classes: np.ndarray,
    class_counts: dict,
    total_samples: int,
    id_to_name: Optional[dict] = None
) -> np.ndarray:
    """
    Computes intelligent dynamic Bayesian priors combining:
      1. Physical parcel area distribution P_true = Count_c * Area_threshold_c.
      2. Inversion of training loss bias P_train = sqrt(Total / (N * Count_c)).
      3. Power 0.7 exponential smoothing.
      4. Strict zeroing of non-existent classes in the orbit (Count_c == 0 -> 0.0).
    """
    if id_to_name is None:
        id_to_name = {}

    n_classes = len(classes)
    if n_classes == 0 or total_samples == 0:
        return np.ones(n_classes, dtype=np.float32) / max(n_classes, 1)

    # 1. P_true: physical parcel surface area distribution
    area_multipliers = np.array([
        get_crop_area_multiplier(int(c), id_to_name.get(int(c), ""), expected_classes=n_classes)
        for c in classes
    ], dtype=np.float64)

    counts_arr = np.array([float(class_counts.get(c, 0)) for c in classes], dtype=np.float64)
    true_area = counts_arr * area_multipliers
    sum_true_area = np.sum(true_area)
    if sum_true_area > 0:
        p_true = true_area / sum_true_area
    else:
        p_true = np.ones(n_classes, dtype=np.float64) / n_classes

    # 2. P_train: exact training bias injected by square-root class loss weights
    train_bias = np.zeros(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        cnt = float(class_counts.get(c, 0))
        if cnt > 0:
            train_bias[i] = math.sqrt(total_samples / (n_classes * cnt))
        else:
            train_bias[i] = 0.0

    sum_train_bias = np.sum(train_bias)
    if sum_train_bias > 0:
        p_train = train_bias / sum_train_bias
    else:
        p_train = np.ones(n_classes, dtype=np.float64) / n_classes

    # 3. Bayesian Correction Factor with SATMIROL 0.7 smoothing
    w_bayes = np.zeros(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if class_counts.get(c, 0) == 0:
            w_bayes[i] = 0.0
        else:
            ratio = p_true[i] / (p_train[i] + 1e-9)
            w_bayes[i] = math.pow(ratio, 0.7)

    # 4. Clipping and strict zeroing
    w_bayes = np.clip(w_bayes, 0.01, 10.0)
    for i, c in enumerate(classes):
        if class_counts.get(c, 0) == 0:
            w_bayes[i] = 0.0

    sum_w = np.sum(w_bayes)
    if sum_w > 0:
        priors_arr = (w_bayes / sum_w).astype(np.float32)
    else:
        priors_arr = (np.ones(n_classes, dtype=np.float32) / n_classes)

    return priors_arr


def _get_priors_for_country(country: str, learn_shp_path: Optional[Path], classes: np.ndarray, class_counts: dict, total_samples: int, priors_file_override: Optional[Path] = None) -> np.ndarray:
    """Backward compatibility wrapper redirecting to compute_dynamic_bayesian_priors."""
    id_to_name = {}
    if learn_shp_path and os.path.exists(learn_shp_path):
        try:
            gdf = gpd.read_file(str(learn_shp_path), engine="pyogrio")
            if 'crop_id' in gdf.columns and 'crop_name' in gdf.columns:
                id_to_name = dict(zip(gdf['crop_id'].astype(int), gdf['crop_name'].astype(str)))
        except Exception:
            pass
    return compute_dynamic_bayesian_priors(classes, class_counts, total_samples, id_to_name)


def _calculate_class_weights(y_data: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    classes_in_data = np.unique(y_data)
    total_samples = len(y_data)
    n_classes = len(classes_in_data)
    weight_vector = np.ones(len(all_classes), dtype=np.float32)
    
    for c in classes_in_data:
        count = np.sum(y_data == c)
        if count > 0:
            weight = total_samples / (n_classes * count)
            idx = np.where(all_classes == c)[0][0]
            weight_vector[idx] = math.sqrt(weight)
            
    return weight_vector


# =====================================================================


def compute_vegetation_and_sar_indices(
    s1_means: Optional[np.ndarray],
    s2_means: Optional[np.ndarray],
    num_dates_s1: int,
    num_dates_s2: int,
    s1_stds: Optional[np.ndarray] = None,
    s2_stds: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes physiological red-edge (NDRE1, NDRE2), optical (NDVI, NDWI), polarimetric SAR (RVI, CR),
    temporal rate-of-change (Delta-VH), and intra-object texture/variance across multi-temporal observations.
    """
    N = s1_means.shape[0] if s1_means is not None else (s2_means.shape[0] if s2_means is not None else 0)
    if N == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    feats = []
    feat_names = []

    if s2_means is not None and num_dates_s2 > 0:
        ndre1_list, ndre2_list, ndvi_list, ndwi_list = [], [], [], []
        for d in range(num_dates_s2):
            base_idx = d * 9
            b4 = s2_means[:, base_idx + 2] / 10000.0
            b5 = s2_means[:, base_idx + 3] / 10000.0
            b6 = s2_means[:, base_idx + 4] / 10000.0
            b8a = s2_means[:, base_idx + 6] / 10000.0
            b11 = s2_means[:, base_idx + 7] / 10000.0

            ndre1 = (b8a - b5) / (b8a + b5 + 1e-6)
            ndre2 = (b6 - b5) / (b6 + b5 + 1e-6)
            ndvi = (b8a - b4) / (b8a + b4 + 1e-6)
            ndwi = (b8a - b11) / (b8a + b11 + 1e-6)

            ndre1_list.append(ndre1)
            ndre2_list.append(ndre2)
            ndvi_list.append(ndvi)
            ndwi_list.append(ndwi)
            feat_names.extend([f"ndre1_d{d}", f"ndre2_d{d}", f"ndvi_d{d}", f"ndwi_d{d}"])

        ndre1_arr = np.column_stack(ndre1_list)
        ndre2_arr = np.column_stack(ndre2_list)
        ndvi_arr = np.column_stack(ndvi_list)
        ndwi_arr = np.column_stack(ndwi_list)

        s2_temporal_stats = np.column_stack([
            np.max(ndre1_arr, axis=1), np.min(ndre1_arr, axis=1), np.max(ndre1_arr, axis=1) - np.min(ndre1_arr, axis=1), np.mean(ndre1_arr, axis=1),
            np.max(ndre2_arr, axis=1), np.mean(ndre2_arr, axis=1),
            np.max(ndvi_arr, axis=1), np.min(ndvi_arr, axis=1), np.max(ndvi_arr, axis=1) - np.min(ndvi_arr, axis=1), np.mean(ndvi_arr, axis=1),
            np.max(ndwi_arr, axis=1), np.mean(ndwi_arr, axis=1)
        ])
        feat_names.extend([
            "ndre1_max", "ndre1_min", "ndre1_amp", "ndre1_mean",
            "ndre2_max", "ndre2_mean",
            "ndvi_max", "ndvi_min", "ndvi_amp", "ndvi_mean",
            "ndwi_max", "ndwi_mean"
        ])
        feats.append(np.hstack([ndre1_arr, ndre2_arr, ndvi_arr, ndwi_arr, s2_temporal_stats]))

        # Intra-object optical texture / variance if provided
        if s2_stds is not None and s2_stds.shape[1] >= num_dates_s2 * 9:
            s2_std_mean = np.mean(s2_stds, axis=1).reshape(-1, 1)
            s2_std_max = np.max(s2_stds, axis=1).reshape(-1, 1)
            # Estimate of NDVI intra-parcel variance from red and NIR channels
            ndvi_intra_std = (np.mean(s2_stds[:, 2::9] + s2_stds[:, 6::9], axis=1) / 10000.0).reshape(-1, 1)
            feats.append(np.hstack([s2_std_mean, s2_std_max, ndvi_intra_std]))
            feat_names.extend(["s2_intra_std_mean", "s2_intra_std_max", "ndvi_intra_std"])

    if s1_means is not None and num_dates_s1 > 0:
        rvi_list, cr_list = [], []
        for d in range(num_dates_s1):
            vh_db = s1_means[:, d]
            vv_db = s1_means[:, num_dates_s1 + d]

            vh_lin = np.power(10.0, np.clip(vh_db, -35.0, 10.0) / 10.0)
            vv_lin = np.power(10.0, np.clip(vv_db, -35.0, 10.0) / 10.0)

            rvi = (4.0 * vh_lin) / (vv_lin + vh_lin + 1e-6)
            cr = vh_db - vv_db

            rvi_list.append(rvi)
            cr_list.append(cr)
            feat_names.extend([f"rvi_d{d}", f"cr_d{d}"])

        rvi_arr = np.column_stack(rvi_list)
        cr_arr = np.column_stack(cr_list)

        s1_temporal_stats = np.column_stack([
            np.max(rvi_arr, axis=1), np.min(rvi_arr, axis=1), np.max(rvi_arr, axis=1) - np.min(rvi_arr, axis=1), np.mean(rvi_arr, axis=1),
            np.max(cr_arr, axis=1), np.min(cr_arr, axis=1), np.mean(cr_arr, axis=1)
        ])
        feat_names.extend([
            "rvi_max", "rvi_min", "rvi_amp", "rvi_mean",
            "cr_max", "cr_min", "cr_mean"
        ])
        feats.append(np.hstack([rvi_arr, cr_arr, s1_temporal_stats]))

        # Physical SAR phenological dynamics: Delta-VH Spring vs Summer rate-of-change
        if num_dates_s1 >= 4:
            spring_d = max(0, min(int(num_dates_s1 * 0.25), num_dates_s1 - 1))
            summer_d = max(0, min(int(num_dates_s1 * 0.60), num_dates_s1 - 1))
            delta_vh = (s1_means[:, summer_d] - s1_means[:, spring_d]).reshape(-1, 1)
            delta_vv = (s1_means[:, summer_d + num_dates_s1] - s1_means[:, spring_d + num_dates_s1]).reshape(-1, 1)
            vh_amp = (np.max(s1_means[:, :num_dates_s1], axis=1) - np.min(s1_means[:, :num_dates_s1], axis=1)).reshape(-1, 1)
            feats.append(np.hstack([delta_vh, delta_vv, vh_amp]))
            feat_names.extend(["s1_delta_vh_spring_summer", "s1_delta_vv_spring_summer", "s1_vh_temporal_amp"])

        # Intra-object SAR texture / relative standard deviation
        if s1_stds is not None and s1_stds.shape[1] >= num_dates_s1 * 2:
            vh_intra_std_mean = np.mean(s1_stds[:, :num_dates_s1], axis=1).reshape(-1, 1)
            vv_intra_std_mean = np.mean(s1_stds[:, num_dates_s1:num_dates_s1 * 2], axis=1).reshape(-1, 1)
            vh_intra_std_max = np.max(s1_stds[:, :num_dates_s1], axis=1).reshape(-1, 1)
            feats.append(np.hstack([vh_intra_std_mean, vv_intra_std_mean, vh_intra_std_max]))
            feat_names.extend(["s1_vh_intra_std_mean", "s1_vv_intra_std_mean", "s1_vh_intra_std_max"])

    arr_res = np.hstack(feats).astype(np.float32) if feats else np.zeros((N, 0), dtype=np.float32)
    return arr_res, feat_names


def _clean_label_noise(X: np.ndarray, y: np.ndarray, classes: np.ndarray, prune_pct: float = 0.02) -> np.ndarray:
    """
    Filters out extreme multivariate feature outliers per crop class to sanitize training boundaries.
    """
    keep_mask = np.ones(len(y), dtype=bool)
    pruned_count = 0
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < 40:
            continue
        X_c = X[idx]
        centroid = np.median(X_c, axis=0)
        dist = np.linalg.norm(X_c - centroid, axis=1)
        cutoff = np.percentile(dist, 100.0 * (1.0 - prune_pct))
        outliers = idx[dist > cutoff]
        keep_mask[outliers] = False
        pruned_count += len(outliers)
    if pruned_count > 0:
        print(f"    [LABEL FILTER] Pruned {pruned_count} extreme outlier samples ({prune_pct*100:.1f}% per class) to sanitize boundaries.")
    return keep_mask
# 2. UNIFIED MLP + XGBOOST FUSION ENSEMBLE CLASSIFIER
# =====================================================================

class TorchMLPClassifier:
    """PyTorch Deep Neural Network Classifier for Multimodal Crop Classification."""
    def __init__(self, hidden_layer_sizes=(512, 256, 128), max_iter=200, batch_size=256, lr=0.001, class_weights=None, all_classes=None, temperature: float = 1.2):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.class_weights = class_weights
        self.all_classes = all_classes
        self.temperature = max(float(temperature), 0.1)
        self.device = torch.device('cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu')
        self.model = None
        self.le = None
        self.classes_ = None

    def fit(self, X, y):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.le = LabelEncoder()
        if self.all_classes is not None:
            self.le.fit(self.all_classes)
        else:
            self.le.fit(y)

        y_enc = self.le.transform(y)
        self.classes_ = self.le.classes_

        input_dim = X.shape[1]
        output_dim = len(self.classes_)

        layers = []
        in_dim = input_dim
        for h_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers).to(self.device)

        if self.class_weights is not None and len(self.class_weights) == output_dim:
            weights_tensor = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_enc, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_iter):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.model.eval()
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=min(len(X), 4096), shuffle=False)

        probs_list = []
        with torch.no_grad():
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                scaled_logits = logits / self.temperature
                probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()
                probs_list.append(probs)

        return np.vstack(probs_list)

    def predict(self, X):
        probs = self.predict_proba(X)
        preds_enc = np.argmax(probs, axis=1)
        return self.le.inverse_transform(preds_enc)


class EnsembleClassifier:
    """
    Unified Fusion Ensemble combining Deep PyTorch MLP and XGBoost GBDT via Soft Voting.
    """
    def __init__(self, mlp_model, xgb_model, weight_mlp=0.65):
        self.mlp_model = mlp_model
        self.xgb_model = xgb_model
        self.weight_mlp = weight_mlp
        self.classes_ = None
        self.xgb_classes_ = None
        self.imputer = SimpleImputer(strategy='mean')

    def fit(self, X, y):
        print("  [Fusion 1/2] Training PyTorch Deep MLP Model...")
        self.mlp_model.fit(X, y)
        self.classes_ = self.mlp_model.classes_

        print("  [Fusion 2/2] Training XGBoost Gradient Boosted Trees Model...")
        X_imputed = self.imputer.fit_transform(X)

        le = getattr(self.mlp_model, 'le', None)
        if le is None:
            le = LabelEncoder()
            le.fit(y)
        y_enc = le.transform(y)

        self.xgb_classes_ = np.unique(y_enc)
        xgb_le = LabelEncoder()
        xgb_le.fit(self.xgb_classes_)
        y_xgb = xgb_le.transform(y_enc)

        self.xgb_model.fit(X_imputed, y_xgb)
        print("  [Fusion Complete] Unified MLP + XGBoost Ensemble successfully fitted.")
        return self

    def predict_proba(self, X):
        p_mlp = self.mlp_model.predict_proba(X)
        X_imputed = self.imputer.transform(X)
        p_xgb_raw = self.xgb_model.predict_proba(X_imputed)

        p_xgb = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float32)
        p_xgb[:, self.xgb_classes_] = p_xgb_raw

        return self.weight_mlp * p_mlp + (1.0 - self.weight_mlp) * p_xgb

    def predict(self, X):
        p = self.predict_proba(X)
        preds_enc = np.argmax(p, axis=1)
        le = getattr(self.mlp_model, 'le', None)
        if le is not None:
            return le.inverse_transform(preds_enc)
        return self.classes_[preds_enc]

# =====================================================================
# Register model classes in __main__ and legacy namespaces for unpickling
# =====================================================================
import sys
current_mod = sys.modules.get(__name__)
if current_mod:
    sys.modules['1_classify_MLPXGB_presto_hybrid_S1S2'] = current_mod
    sys.modules['classifier_mlpxgb_presto'] = current_mod
    sys.modules['classifier_mlpxgb_presto_S1S2'] = current_mod

main_mod = sys.modules.get('__main__')
if main_mod:
    setattr(main_mod, 'EnsembleClassifier', EnsembleClassifier)
    setattr(main_mod, 'TorchMLPClassifier', TorchMLPClassifier)


# =====================================================================
# 3. MULTIMODAL PRESTO EMBEDDINGS (S1 + S2)
# =====================================================================


class PrestoMultimodalExtractor:
    """Computes 128-dimensional Presto foundation embeddings for multi-temporal S1 and S2 series, including true joint multimodal fusion."""
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.weights_path = presto_dir / "default_model.pt"
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Presto model weights not found at {self.weights_path}")
        
        try:
            import presto_model as single_file_presto
        except ImportError:
            import single_file_presto
        self.model = single_file_presto.Presto.construct(max_sequence_length=36)
        state_dict = torch.load(self.weights_path, map_location=self.device)
        state_dict.pop('encoder.pos_embed', None)
        state_dict.pop('decoder.pos_embed', None)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def get_joint_embeddings(
        self,
        batch_s1_vv_vh: Optional[torch.Tensor],
        batch_s2_9bands: Optional[torch.Tensor],
        batch_latlons: torch.Tensor,
        months_tensor: torch.Tensor
    ) -> np.ndarray:
        """Computes true multimodal joint cross-attention embeddings across S1 + S2 simultaneously."""
        if batch_s2_9bands is not None:
            B, T, _ = batch_s2_9bands.shape
        elif batch_s1_vv_vh is not None:
            B, T, _ = batch_s1_vv_vh.shape
        else:
            raise ValueError("At least one modality (S1 or S2) must be present.")

        x = torch.zeros(B, T, 17, dtype=torch.float32, device=self.device)
        mask = torch.ones(B, T, 17, dtype=torch.float32, device=self.device)

        if batch_s1_vv_vh is not None:
            t_s1 = min(T, batch_s1_vv_vh.shape[1])
            x[:, :t_s1, 0] = batch_s1_vv_vh[:, :t_s1, 0].to(self.device)
            x[:, :t_s1, 1] = batch_s1_vv_vh[:, :t_s1, 1].to(self.device)
            mask[:, :t_s1, 0:2] = 0.0

        if batch_s2_9bands is not None:
            t_s2 = min(T, batch_s2_9bands.shape[1])
            b2 = batch_s2_9bands[:, :t_s2, 0].to(self.device)
            b3 = batch_s2_9bands[:, :t_s2, 1].to(self.device)
            b4 = batch_s2_9bands[:, :t_s2, 2].to(self.device)
            b5 = batch_s2_9bands[:, :t_s2, 3].to(self.device)
            b6 = batch_s2_9bands[:, :t_s2, 4].to(self.device)
            b7 = batch_s2_9bands[:, :t_s2, 5].to(self.device)
            b8a = batch_s2_9bands[:, :t_s2, 6].to(self.device)
            b11 = batch_s2_9bands[:, :t_s2, 7].to(self.device)
            b12 = batch_s2_9bands[:, :t_s2, 8].to(self.device)
            ndvi = (b8a - b4) / (b8a + b4 + 1e-6)

            x[:, :t_s2, 2] = b2
            x[:, :t_s2, 3] = b3
            x[:, :t_s2, 4] = b4
            x[:, :t_s2, 5] = b5
            x[:, :t_s2, 6] = b6
            x[:, :t_s2, 7] = b7
            x[:, :t_s2, 9] = b8a
            x[:, :t_s2, 10] = b11
            x[:, :t_s2, 11] = b12
            x[:, :t_s2, 16] = ndvi

            mask[:, :t_s2, [2, 3, 4, 5, 6, 7, 9, 10, 11, 16]] = 0.0

        dw = torch.ones(B, T, dtype=torch.long, device=self.device) * 9
        if months_tensor.ndim == 1:
            if len(months_tensor) >= T:
                m_slice = months_tensor[:T]
            else:
                pad = torch.zeros(T - len(months_tensor), dtype=torch.long, device=self.device)
                m_slice = torch.cat([months_tensor, pad])
            month = m_slice.unsqueeze(0).expand(B, -1)
        else:
            month = months_tensor

        with torch.no_grad():
            features = self.model.encoder(
                x=x,
                dynamic_world=dw,
                latlons=batch_latlons.to(self.device),
                mask=mask,
                month=month,
                eval_task=True
            )
        return features.cpu().numpy()

    def get_s1_embeddings(self, batch_s1_vv_vh: torch.Tensor, batch_latlons: torch.Tensor, months_tensor: torch.Tensor) -> np.ndarray:
        B, T, _ = batch_s1_vv_vh.shape
        x = torch.zeros(B, T, 17, dtype=torch.float32, device=self.device)
        x[:, :, 0] = batch_s1_vv_vh[:, :, 0] # VV
        x[:, :, 1] = batch_s1_vv_vh[:, :, 1] # VH

        mask = torch.ones(B, T, 17, dtype=torch.float32, device=self.device)
        mask[:, :, 0:2] = 0.0

        dw = torch.ones(B, T, dtype=torch.long, device=self.device) * 9
        month = months_tensor.unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            features = self.model.encoder(
                x=x,
                dynamic_world=dw,
                latlons=batch_latlons.to(self.device),
                mask=mask,
                month=month,
                eval_task=True
            )
        return features.cpu().numpy()

    def get_s2_embeddings(self, batch_s2_9bands: torch.Tensor, batch_latlons: torch.Tensor, months_tensor: torch.Tensor) -> np.ndarray:
        B, T, _ = batch_s2_9bands.shape
        x = torch.zeros(B, T, 17, dtype=torch.float32, device=self.device)

        b2 = batch_s2_9bands[:, :, 0]
        b3 = batch_s2_9bands[:, :, 1]
        b4 = batch_s2_9bands[:, :, 2]
        b5 = batch_s2_9bands[:, :, 3]
        b6 = batch_s2_9bands[:, :, 4]
        b7 = batch_s2_9bands[:, :, 5]
        b8a = batch_s2_9bands[:, :, 6]
        b11 = batch_s2_9bands[:, :, 7]
        b12 = batch_s2_9bands[:, :, 8]

        ndvi = (b8a - b4) / (b8a + b4 + 1e-6)

        x[:, :, 2] = b2
        x[:, :, 3] = b3
        x[:, :, 4] = b4
        x[:, :, 5] = b5
        x[:, :, 6] = b6
        x[:, :, 7] = b7
        x[:, :, 9] = b8a
        x[:, :, 10] = b11
        x[:, :, 11] = b12
        x[:, :, 16] = ndvi

        mask = torch.ones(B, T, 17, dtype=torch.float32, device=self.device)
        mask[:, :, [2, 3, 4, 5, 6, 7, 9, 10, 11, 16]] = 0.0

        dw = torch.ones(B, T, dtype=torch.long, device=self.device) * 9
        month = months_tensor.unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            features = self.model.encoder(
                x=x,
                dynamic_world=dw,
                latlons=batch_latlons.to(self.device),
                mask=mask,
                month=month,
                eval_task=True
            )
        return features.cpu().numpy()

# =====================================================================
# SAM WORKER (MULTIPROCESSING)
# =====================================================================

def sam_worker(tile_info, ras_path, footprint_path, params):
    try:
        import os
        import sys
        import time
        from osgeo import gdal
        import numpy as np
        import torch
        torch.set_num_threads(1)  # Force single thread in worker
        import cv2
        cv2.setNumThreads(0)
        
        from samgeo import SamGeo
        from skimage.util import img_as_float
        from scipy.ndimage import distance_transform_edt
        import scipy.ndimage as ndimage
        
        x, y, xsize_valid, ysize_valid, x_start_buf, y_start_buf, xsize_buf, ysize_buf, buffer = tile_info
        print(f"    [Worker x={x}, y={y}] Started reading tile data...", flush=True)
        
        ds = gdal.Open(ras_path, gdal.GA_ReadOnly)
        nbands = ds.RasterCount
        
        img_list = []
        for b in range(1, nbands + 1):
            band = ds.GetRasterBand(b)
            arr = band.ReadAsArray(x_start_buf, y_start_buf, xsize_buf, ysize_buf)
            if arr is None:
                print(f"    [Worker x={x}, y={y}] Failed to read band {b}!", flush=True)
                return x, y, None, None
            arr = np.nan_to_num(arr)
            img_list.append(arr)
            
        img = np.dstack(img_list)
        
        ds_foot = None
        if footprint_path and os.path.exists(footprint_path):
            ds_foot = gdal.Open(footprint_path, gdal.GA_ReadOnly)
            valid_mask_buf = ds_foot.GetRasterBand(1).ReadAsArray(x_start_buf, y_start_buf, xsize_buf, ysize_buf) > 0
            valid_mask = valid_mask_buf
        else:
            valid_mask = np.sum(np.abs(img), axis=2) > 0
            
        if not np.any(valid_mask):
            print(f"    [Worker x={x}, y={y}] Tile contains no active pixels.", flush=True)
            return x, y, None, None
            
        print(f"    [Worker x={x}, y={y}] Loading SAM model...", flush=True)
        t_model_start = time.time()
        sam_geo = SamGeo(
            model_type=params.get('sam_model_type', 'vit_h'),
            checkpoint=params.get('sam_checkpoint', None),
            device=params.get('sam_device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            sam_kwargs={
                "points_per_side": params.get('points_per_side', 16),
                "pred_iou_thresh": params.get('pred_iou_thresh', 0.45),
                "stability_score_thresh": params.get('stability_score_thresh', 0.50),
                "crop_n_layers": params.get('crop_n_layers', 0),
                "crop_n_points_downscale_factor": params.get('crop_n_points_downscale_factor', 1),
                "min_mask_region_area": params.get('min_mask_region_area', 20),
                "box_nms_thresh": params.get('box_nms_thresh', 0.6)
            }
        )
        print(f"    [Worker x={x}, y={y}] SAM model loaded in {time.time() - t_model_start:.2f}s.", flush=True)
        
        img_8bit = np.zeros(img.shape, dtype=np.uint8)
        valid_pixels = valid_mask[:, :, np.newaxis]
        
        if np.any(valid_pixels):
            p2, p98 = np.percentile(img[valid_pixels], (2, 98))
            img_clip = np.clip(img, p2, p98)
            if p98 > p2:
                img_8bit[valid_pixels] = ((img_clip[valid_pixels] - p2) / (p98 - p2) * 255).astype(np.uint8)
                
            img_chan = np.ascontiguousarray(img_8bit[:, :, 0])
            img_smoothed = cv2.bilateralFilter(img_chan, d=9, sigmaColor=12, sigmaSpace=30)
            img_8bit[:, :, 0] = img_smoothed
            
            clahe_limit = params.get('clahe_limit', 0.0)
            if clahe_limit > 0.0:
                clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8,8))
                img_clahe = clahe.apply(img_8bit[:, :, 0])
                img_8bit[:, :, 0] = img_clahe
                
        if img_8bit.shape[2] == 1:
            img_rgb = np.repeat(img_8bit, 3, axis=2)
        else:
            img_rgb = img_8bit[:, :, :3]
            if img_rgb.shape[2] < 3:
                img_rgb = np.pad(img_rgb, ((0,0),(0,0),(0, 3-img_rgb.shape[2])), mode='constant')
                
        print(f"    [Worker x={x}, y={y}] Running SAM generate (points_per_side={params.get('points_per_side', 16)})...", flush=True)
        t_gen_start = time.time()
        sam_geo.generate(
            source=img_rgb,
            output=None,
            foreground=False,
            unique=True,
            min_size=10,
            max_size=100000
        )
        print(f"    [Worker x={x}, y={y}] SAM generate finished in {time.time() - t_gen_start:.2f}s.", flush=True)
        segments_buf = sam_geo.objects.astype(np.int32)
        
        zero_mask_buf = (segments_buf == 0) & valid_mask
        if np.any(zero_mask_buf) and np.any(segments_buf > 0):
            _, indices = distance_transform_edt(segments_buf == 0, return_indices=True)
            segments_buf[zero_mask_buf] = segments_buf[tuple(indices)][zero_mask_buf]
            
        segments_buf[~valid_mask] = 0
        
        median_size = params.get('median_size', 3)
        if median_size > 0:
            segments_buf = ndimage.median_filter(segments_buf, size=median_size)
            segments_buf[~valid_mask] = 0
            
        y_offset = y - y_start_buf
        x_offset = x - x_start_buf
        segments_buf[~valid_mask] = 0
        
        valid_mask_crop = valid_mask[y_offset : y_offset + ysize_valid, x_offset : x_offset + xsize_valid]
        segments = segments_buf[y_offset : y_offset + ysize_valid, x_offset : x_offset + xsize_valid]
        segments[~valid_mask_crop] = 0
        
        print(f"    [Worker x={x}, y={y}] Tile completed successfully.", flush=True)
        return x, y, segments, valid_mask_crop
    except Exception as e:
        print(f"Error in worker process for tile (x={tile_info[0]}, y={tile_info[1]}): {e}", flush=True)
        return tile_info[0], tile_info[1], None, None


def slic_worker(tile_info, ras_path, footprint_path, params):
    """
    Independent multi-core worker process for tiled SLIC superpixel segmentation.
    Executes in parallel across available CPU cores.
    """
    try:
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        from osgeo import gdal
        import numpy as np
        from skimage.segmentation import slic
        from skimage.util import img_as_float

        x, y, xsize_valid, ysize_valid, x_start_buf, y_start_buf, xsize_buf, ysize_buf, buffer = tile_info

        ds = gdal.Open(str(ras_path), gdal.GA_ReadOnly)
        if not ds:
            return x, y, None, None
        nbands = ds.RasterCount
        img_list = []
        for b in range(1, nbands + 1):
            band = ds.GetRasterBand(b)
            arr = band.ReadAsArray(x_start_buf, y_start_buf, xsize_buf, ysize_buf)
            if arr is None:
                return x, y, None, None
            arr = np.nan_to_num(arr)
            img_list.append(arr)
        ds = None
        img = np.dstack(img_list)

        if footprint_path and os.path.exists(footprint_path):
            ds_foot = gdal.Open(str(footprint_path), gdal.GA_ReadOnly)
            arr_foot = ds_foot.GetRasterBand(1).ReadAsArray(x_start_buf, y_start_buf, xsize_buf, ysize_buf)
            ds_foot = None
            valid_mask = arr_foot > 0 if arr_foot is not None else (np.sum(np.abs(img), axis=2) > 0)
        else:
            valid_mask = np.sum(np.abs(img), axis=2) > 0

        if not np.any(valid_mask):
            return x, y, None, None

        # Multi-band robust 2-98% percentile min-max scaling per channel
        norm_bands = []
        for b_arr in img_list:
            v_pix = b_arr[valid_mask]
            if len(v_pix) > 0:
                p2 = float(np.percentile(v_pix, 2.0))
                p98 = float(np.percentile(v_pix, 98.0))
                if (p98 - p2) > 1e-6:
                    scaled = np.clip((b_arr - p2) / (p98 - p2), 0.0, 1.0)
                else:
                    scaled = np.zeros_like(b_arr, dtype=np.float64)
            else:
                scaled = np.zeros_like(b_arr, dtype=np.float64)
            norm_bands.append(scaled.astype(np.float64))

        img_norm = np.dstack(norm_bands)
        tile_size = params.get('tile_size', 2048)
        active_pixels = int(np.sum(valid_mask))
        if active_pixels < 20:
            return x, y, None, None
        pixels_per_segment = params.get('pixels_per_segment', 250.0)
        n_segments_tile = max(5, int(active_pixels / max(10.0, pixels_per_segment)))
        n_segments_tile = min(n_segments_tile, 40000)

        # Pass mask=None to avoid scikit-image allocating an (N, N) distance matrix in _get_mask_centroids,
        # which causes 136 GiB RAM allocations when N > 50,000. Grid centroids use O(1) RAM.
        segments_buf = slic(
            img_norm,
            n_segments=n_segments_tile,
            compactness=params.get('compactness', 0.05),
            sigma=params.get('slic_sigma', 1.5),
            start_label=1,
            mask=None,
            enforce_connectivity=True,
            min_size_factor=params.get('min_size_factor', 0.2)
        )
        segments_buf[~valid_mask] = 0

        # Vectorized Region Adjacency Graph (RAG) spectral fusion (disabled by default):
        # When enabled, merges adjacent sub-parcel slivers with identical signatures without exceeding max parcel area.
        enable_rag = params.get('enable_rag', False)
        rag_thresh = float(params.get('rag_thresh', 0.02))
        max_rag_px = int(params.get('max_rag_parcel_ha', 15.0) * 100.0)
        if enable_rag and rag_thresh > 0:
            u_labels, inv = np.unique(segments_buf, return_inverse=True)
            if len(u_labels) > 1:
                flat_inv = inv.ravel()
                n_lbl = len(u_labels)
                counts = np.bincount(flat_inv)
                means = np.zeros((n_lbl, img_norm.shape[2]), dtype=np.float32)
                for c in range(img_norm.shape[2]):
                    means[:, c] = np.bincount(flat_inv, weights=img_norm[:, :, c].ravel()) / np.maximum(1, counts)

                # Extract adjacency pairs across horizontal and vertical neighbor pixels
                h_left = segments_buf[:, :-1].reshape(-1)
                h_right = segments_buf[:, 1:].reshape(-1)
                v_top = segments_buf[:-1, :].reshape(-1)
                v_bottom = segments_buf[1:, :].reshape(-1)

                pairs_h = np.column_stack([h_left, h_right])
                pairs_v = np.column_stack([v_top, v_bottom])
                all_pairs = np.vstack([pairs_h, pairs_v])

                valid_pairs = (all_pairs[:, 0] > 0) & (all_pairs[:, 1] > 0) & (all_pairs[:, 0] != all_pairs[:, 1])
                if np.any(valid_pairs):
                    edges = np.unique(np.sort(all_pairs[valid_pairs], axis=1), axis=0)

                    lbl_map = np.zeros(segments_buf.max() + 1, dtype=np.int32)
                    lbl_map[u_labels] = np.arange(n_lbl, dtype=np.int32)
                    idx1 = lbl_map[edges[:, 0]]
                    idx2 = lbl_map[edges[:, 1]]

                    diff = means[idx1] - means[idx2]
                    dists = np.linalg.norm(diff, axis=1)

                    size1 = counts[idx1]
                    size2 = counts[idx2]
                    merge_mask = (dists < rag_thresh) & ((size1 + size2) <= max_rag_px)
                    merge_edges = edges[merge_mask]
                    if len(merge_edges) > 0:
                        parent = {}

                        def find(i):
                            path = []
                            while parent.get(i, i) != i:
                                path.append(i)
                                i = parent[i]
                            for node in path:
                                parent[node] = i
                            return i

                        def union(i, j):
                            ri = find(i)
                            rj = find(j)
                            if ri != rj:
                                parent[rj] = ri

                        for e in merge_edges:
                            union(int(e[0]), int(e[1]))

                        remap = np.arange(segments_buf.max() + 1, dtype=np.int32)
                        for k in parent.keys():
                            remap[k] = find(k)
                        segments_buf = remap[segments_buf]

        y_offset = y - y_start_buf
        x_offset = x - x_start_buf
        valid_mask_crop = valid_mask[y_offset : y_offset + ysize_valid, x_offset : x_offset + xsize_valid]
        segments = segments_buf[y_offset : y_offset + ysize_valid, x_offset : x_offset + xsize_valid]
        segments[~valid_mask_crop] = 0

        return x, y, segments, valid_mask_crop
    except Exception as e:
        print(f"    [SLIC Worker Error x={tile_info[0]}, y={tile_info[1]}]: {e}", flush=True)
        return tile_info[0], tile_info[1], None, None


# =====================================================================
# 4. MULTIMODAL PROCESSING PIPELINE (STAGES 0 - 7)
# =====================================================================

class ProcessingPipelineS1S2:
    def __init__(
        self,
        track: str,
        seg_mode: str = 'slic',
        mlp_weight: float = 0.65,
        s1_override: Optional[str] = None,
        s2_override: Optional[str] = None,
        lpis_vector: Optional[str] = None,
        slic_segment_ha: Optional[float] = None,
        slic_compactness: float = 0.05,
        slic_rag_thresh: float = 0.02,
        enable_slic_rag: bool = False,
        overwrite: bool = False
    ):
        self.track = track
        self.seg_mode = seg_mode.lower()
        self.mlp_weight = mlp_weight
        self.lpis_vector_override = lpis_vector
        self.overwrite = bool(overwrite)
        
        norm_track = track.replace('\\', '/')
        self.country = norm_track.split('/')[0].upper() if '/' in norm_track else track.upper()
        self.total_stages = TOTAL_STAGES

        # Regional adaptive scale: default parcel scale matching cadastral dimensions
        if slic_segment_ha is not None:
            self.slic_segment_ha = float(slic_segment_ha)
        else:
            if self.country in ['PT', 'ES', 'IT', 'GR', 'PL']:
                self.slic_segment_ha = 1.8  # ~180 pixels at 10m (~134m x 134m parcel size)
            else:
                self.slic_segment_ha = 3.0  # ~300 pixels at 10m (~173m x 173m) for NL/FR/DE
        self.slic_compactness = float(slic_compactness)
        self.slic_rag_thresh = float(slic_rag_thresh)
        self.enable_slic_rag = bool(enable_slic_rag)

        self.sanitized_track = norm_track.replace('/', '_')
        if not self.sanitized_track.startswith(self.country + "_"):
            self.file_prefix = f"{self.country}_{self.sanitized_track}"
        else:
            self.file_prefix = self.sanitized_track

        # Define directories
        # Define directories (New sequential structure in workingDirs/)
        self.base_dir = base_dir
        self.aux_dir = aux_dir
        self.proc_dir = self.base_dir / self.track / '1_input_stacks'
        self.out_dir = self.base_dir / self.track / '2_classification'
        self.seg_dir = self.out_dir / '0_segmentation'
        self.samples_dir = self.out_dir / '1_samples_and_features'
        self.model_dir = self.out_dir / '2_models'
        self.class_dir = self.out_dir / '3_maps'
        self.reports_dir = self.out_dir / '4_reports'

        self._ensure_directories()

        # Resolve Sentinel-1 and Sentinel-2 rasters
        self.s1_ras = Path(s1_override) if s1_override else self._resolve_s1_raster()
        self.s2_ras = Path(s2_override) if s2_override else self._resolve_s2_raster()

        print(f"============================================================")
        print(f" Multimodal S1 (Sigma0) + S2 Crop Classifier (Unified MLP+XGB Fusion)")
        print(f" Track: {self.track} ({self.country})")
        print(f" S1 SAR Raster: {self.s1_ras.name if self.s1_ras else 'None'}")
        print(f" S2 Optical Raster: {self.s2_ras.name if self.s2_ras else 'None'}")
        print(f" Segmentation: {self.seg_mode.upper()} | Fusion weights: {self.mlp_weight:.2f} MLP + {1.0-self.mlp_weight:.2f} XGB")
        print(f"============================================================")

        # Resolve Samples & Output Paths
        self.sample_shp = self._resolve_samples_shp()
        
        self.suffix = f"_mlpxgb_presto_{self.seg_mode}"

        # Standard canonical targets in workingDirs/
        self.footprint_mask = self.seg_dir / f"{self.file_prefix}_data_footprint.tif"
        self.seg_tif = self.seg_dir / f"{self.file_prefix}_segmentation_{self.seg_mode}.tif"
        self.learn_shp = self.samples_dir / f"{self.file_prefix}_learn_{self.seg_mode}.shp"
        self.control_shp = self.samples_dir / f"{self.file_prefix}_control_{self.seg_mode}.shp"
        self.sel_csv = self.samples_dir / f"{self.file_prefix}_mlpxgb_presto_learn_features_{self.seg_mode}.csv"
        if not self.sel_csv.exists() and not self.overwrite:
            alt_csv = self.samples_dir / f"{self.file_prefix}_mlpxgb_presto_s1s2_learn_features_{self.seg_mode}.csv"
            if alt_csv.exists():
                self.sel_csv = alt_csv

        self.model_pkl = self.model_dir / f"{self.file_prefix}_mlpxgb_presto_model_{self.seg_mode}.pkl"
        if not self.model_pkl.exists() and not self.overwrite:
            alt_model = self.model_dir / f"{self.file_prefix}_mlpxgb_presto_s1s2_model_{self.seg_mode}.pkl"
            if alt_model.exists():
                self.model_pkl = alt_model

        self.class_tif = self.class_dir / f"{self.file_prefix}_classified{self.suffix}.tif"
        self.conf_tif = self.class_dir / f"{self.file_prefix}_confidence{self.suffix}.tif"
        self.entropy_tif = self.class_dir / f"{self.file_prefix}_entropy{self.suffix}.tif"
        self.margin_tif = self.class_dir / f"{self.file_prefix}_margin{self.suffix}.tif"
        self.masked_class = self.class_dir / f"{self.file_prefix}_classified_masked{self.suffix}.tif"
        self.masked_conf = self.class_dir / f"{self.file_prefix}_confidence_masked{self.suffix}.tif"
        self.masked_entropy = self.class_dir / f"{self.file_prefix}_entropy_masked{self.suffix}.tif"
        self.masked_margin = self.class_dir / f"{self.file_prefix}_margin_masked{self.suffix}.tif"
        self.metrics_fp = self.reports_dir / f"report_{self.file_prefix}{self.suffix}.xlsx"

        # Fallback to previously generated s1s2 rasters if running post-processing stages
        if not self.masked_class.exists():
            alt_masked = self.class_dir / f"{self.file_prefix}_classified_masked_mlpxgb_presto_s1s2_{self.seg_mode}.tif"
            if alt_masked.exists():
                self.masked_class = alt_masked
                self.masked_conf = self.class_dir / f"{self.file_prefix}_confidence_masked_mlpxgb_presto_s1s2_{self.seg_mode}.tif"
                self.masked_entropy = self.class_dir / f"{self.file_prefix}_entropy_masked_mlpxgb_presto_s1s2_{self.seg_mode}.tif"
                self.masked_margin = self.class_dir / f"{self.file_prefix}_margin_masked_mlpxgb_presto_s1s2_{self.seg_mode}.tif"
        # Fallback to legacy workingDir/ only if input stacks exist in legacy location and not in workingDirs
        if not self.proc_dir.exists():
            legacy_samples = [
                self.base_dir / self.track / 'classification_results' / 'samples' / f"{self.file_prefix}_learn_{self.seg_mode}.shp",
                self.base_dir / self.track / 'classification_results' / 'samples' / f"learn_{self.seg_mode}.shp",
                Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'classification_results' / 'samples' / f"{self.file_prefix}_learn_{self.seg_mode}.shp",
                Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'classification_results' / 'samples' / f"learn_{self.seg_mode}.shp",
            ]
            for c in legacy_samples:
                if c.exists():
                    self.learn_shp = c
                    break

            legacy_controls = [
                self.base_dir / self.track / 'classification_results' / 'samples' / f"{self.file_prefix}_control_{self.seg_mode}.shp",
                self.base_dir / self.track / 'classification_results' / 'samples' / f"control_{self.seg_mode}.shp",
                Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'classification_results' / 'samples' / f"{self.file_prefix}_control_{self.seg_mode}.shp",
                Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'classification_results' / 'samples' / f"control_{self.seg_mode}.shp",
            ]
            for c in legacy_controls:
                if c.exists():
                    self.control_shp = c
                    break

            legacy_footprints = [
                self.base_dir / self.track / 'classification_results' / 'segmentation' / f"{self.file_prefix}_data_footprint.tif",
                Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'classification_results' / 'segmentation' / f"{self.file_prefix}_data_footprint.tif",
            ]
            for c in legacy_footprints:
                if c.exists():
                    self.footprint_mask = c
                    break

            legacy_segs = [
                self.base_dir / self.track / 'classification_results' / 'segmentation' / f"{self.file_prefix}_segmentation_{self.seg_mode}.tif",
                Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'classification_results' / 'segmentation' / f"{self.file_prefix}_segmentation_{self.seg_mode}.tif",
            ]
            for c in legacy_segs:
                if c.exists():
                    self.seg_tif = c
                    break

            legacy_classes = [
                self.base_dir / self.track / 'classification_results' / 'classification' / f"{self.file_prefix}_classified{self.suffix}.tif",
                Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'classification_results' / 'classification' / f"{self.file_prefix}_classified{self.suffix}.tif",
            ]
            for c in legacy_classes:
                if c.exists():
                    self.class_tif = c
                    self.masked_class = c.parent / f"{self.file_prefix}_classified_masked{self.suffix}.tif"
                    break

            legacy_confs = [
                self.base_dir / self.track / 'classification_results' / 'classification' / f"{self.file_prefix}_confidence{self.suffix}.tif",
                Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'classification_results' / 'classification' / f"{self.file_prefix}_confidence{self.suffix}.tif",
            ]
            for c in legacy_confs:
                if c.exists():
                    self.conf_tif = c
                    self.masked_conf = c.parent / f"{self.file_prefix}_confidence_masked{self.suffix}.tif"
                    break

            if self.class_tif.parent != self.class_dir:
                self.metrics_fp = self.class_tif.parent.parent / f"{self.file_prefix}_metrics{self.suffix}.xlsx"

        self.agri_mask = self._resolve_agri_mask()

        # Segmentation params
        self.stage1_params = {
            'sam_model_type': 'vit_h',
            'sam_checkpoint': self._resolve_sam_checkpoint(),
            'sam_device': 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu',
            'tile_size': 2048,
            'buffer': 128,
            'points_per_side': 16,
            'pred_iou_thresh': 0.45,
            'stability_score_thresh': 0.50,
            'min_mask_region_area': 20,
            'box_nms_thresh': 0.6,
            'clahe_limit': 0.0,
            'median_size': 3
        }

    def _ensure_directories(self):
        for d in [self.seg_dir, self.samples_dir, self.model_dir, self.class_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _resolve_sam_checkpoint(self) -> Optional[str]:
        sam_dir = self.aux_dir / "SAM_models"
        for name in ['sam_vit_h_4b8939.pth', 'sam_vit_l_0b3195.pth', 'sam_vit_b_01ec64.pth']:
            p = sam_dir / name
            if p.exists():
                return str(p)
        return None

    def _resolve_s1_raster(self) -> Optional[Path]:
        candidate_dirs = [
            self.base_dir / self.track / '1_input_stacks',
            self.base_dir / self.track / 'processed_raster',
            Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / '1_input_stacks',
            Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'processed_raster'
        ]
        patterns = [f"*{self.sanitized_track}*_VH_VV*.tif", f"*_VH_VV*.tif", f"*{self.country}*_VH_VV*.tif", f"*{self.sanitized_track}*Sigma0*.tif", f"*Sigma0*.tif"]
        for c_dir in candidate_dirs:
            if c_dir.exists():
                for pat in patterns:
                    matches = list(c_dir.glob(pat))
                    if matches:
                        return matches[0]
        return None

    def _resolve_s2_raster(self) -> Optional[Path]:
        candidate_dirs = [
            self.base_dir / self.track / '1_input_stacks',
            self.base_dir / self.country / 'S2',
            self.base_dir / self.country / '1_input_stacks',
            self.base_dir / self.track / 'processed_raster',
            Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / '1_input_stacks',
            Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.country / 'S2',
            Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.track / 'processed_raster'
        ]
        patterns = [
            f"*{self.sanitized_track}*S2*.tif",
            f"{self.country}_S2_timeseries*.tif",
            f"*{self.country}*S2_timeseries*.tif",
            f"*S2_timeseries*.tif",
            f"*{self.country}*S2*.tif",
            f"*S2*.tif"
        ]
        for c_dir in candidate_dirs:
            if c_dir.exists():
                for pat in patterns:
                    matches = [f for f in c_dir.glob(pat) if f.is_file() and not f.name.endswith(".tmp.tif") and not f.name.endswith(".ovr")]
                    if matches:
                        return matches[0]

        # Cross-orbit discovery fallback: search other orbit folders of the same country
        country_dirs = [self.base_dir / self.country, Path(r"D:/AIML_CropMapper_Cloud/workingDir") / self.country]
        for c_root in country_dirs:
            if not c_root.exists():
                continue
            cross_candidates = list(c_root.glob("orbit_*/1_input_stacks/*S2*.tif"))
            for cand in cross_candidates:
                if cand.exists() and cand.stat().st_size > 100 * 1024 * 1024 and not cand.name.endswith(".tmp.tif") and not cand.name.endswith(".ovr"):
                    try:
                        ds_c = gdal.Open(str(cand))
                        if ds_c and ds_c.RasterCount >= 126:
                            dest_dir = self.base_dir / self.track / '1_input_stacks'
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest_s2 = dest_dir / f"{self.file_prefix}_S2_timeseries.tif"
                            if not dest_s2.exists():
                                try:
                                    os.link(str(cand), str(dest_s2))
                                    print(f"    [OPTIMIZATION] Reused country S2 stack from {cand.name} via instant hardlink to {dest_s2.name}!")
                                except Exception:
                                    import shutil
                                    shutil.copy2(str(cand), str(dest_s2))
                                    print(f"    [OPTIMIZATION] Reused country S2 stack from {cand.name} via copy to {dest_s2.name}!")
                            return dest_s2 if dest_s2.exists() else cand
                    except Exception:
                        continue

        return None

    def _resolve_samples_shp(self) -> Optional[Path]:
        samples_base = self.aux_dir / 'shapefiles_samples'
        candidates = [
            samples_base / self.file_prefix / "samples.shp",
            samples_base / self.country / "samples.shp",
            samples_base / f"{self.country}_{self.sanitized_track}" / "samples.shp"
        ]
        for c in candidates:
            if c.exists():
                return c
        shps = list(samples_base.glob(f"**/*{self.country}*/**/samples.shp"))
        return shps[0] if shps else None

    def _resolve_agri_mask(self) -> Optional[Path]:
        agrimasks_dir = self.aux_dir / "raster_files" / "AgriMasks" / self.country
        candidates = [
            agrimasks_dir / f"{self.country}_agri_mask_allcrops_epsg3857.tif",
            agrimasks_dir / f"{self.country}_agri_mask_3class_epsg3857.tif",
            self.aux_dir / "raster_files" / f"{self.country}_arable.tif",
            self.aux_dir / "raster_files" / f"{self.country}_agri_mask.tif",
            self.aux_dir / "raster_files" / "EU_arable_areas_mask_3857.tif",
            self.base_dir / self.track / "agri_mask.tif"
        ]
        for c in candidates:
            if c.exists():
                return c
        if agrimasks_dir.exists():
            tifs = list(agrimasks_dir.glob("*.tif"))
            if tifs: return tifs[0]
        return None

    def _resolve_lpis_vector(self) -> Optional[Path]:
        if hasattr(self, 'lpis_vector_override') and self.lpis_vector_override:
            p = Path(self.lpis_vector_override)
            if p.exists():
                return p

        agrimasks_dir = self.aux_dir / "raster_files" / "AgriMasks" / self.country
        samples_dir = self.aux_dir / "shapefiles_samples" / self.country
        candidates = [
            agrimasks_dir / "brpgewaspercelen_definitief_2025.gpkg",
            agrimasks_dir / "lpis.gpkg",
            agrimasks_dir / "lpis.shp",
            samples_dir / "lpis.gpkg",
            samples_dir / "lpis.shp",
            samples_dir / "parcels.gpkg",
            samples_dir / "parcels.shp",
            samples_dir / "samples_all.shp",
            samples_dir / "samples_all.gpkg",
            samples_dir / "samples.shp",
            self.aux_dir / "shapefiles_samples" / f"{self.country}_{self.sanitized_track}" / "lpis.shp",
            self.base_dir / self.track / "lpis.shp",
            self.base_dir / self.track / "parcels.gpkg"
        ]
        for c in candidates:
            if c.exists():
                return c
        if agrimasks_dir.exists():
            gpkgs = list(agrimasks_dir.glob("*.gpkg"))
            if gpkgs: return gpkgs[0]
            shps = list(agrimasks_dir.glob("*.shp"))
            if shps: return shps[0]
        return None

    def _create_seasonal_sar_composite(self) -> Path:
        """
        Creates a high-SNR 1-channel seasonal SAR composite for SLIC superpixel segmentation.
        Focuses on peak crop growing months (April to August) where canopy structural contrast
        between agricultural fields is maximal and permanent cadastral boundaries are sharpest.
        Averages across multi-temporal Sentinel-1 acquisitions to eliminate radar speckle noise.
        """
        composite_tif = self.seg_dir / f"{self.file_prefix}_sar_seasonal_composite.tif"
        if composite_tif.exists() and composite_tif.stat().st_size > 1024:
            return composite_tif

        ref_ras = self.s1_ras if (self.s1_ras and self.s1_ras.exists()) else self.s2_ras
        if not ref_ras or not ref_ras.exists():
            raise FileNotFoundError("Neither Sentinel-1 nor Sentinel-2 raster is available for segmentation composite.")

        ds = gdal.Open(str(ref_ras), gdal.GA_ReadOnly)
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        nbands = ds.RasterCount
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()

        # Identify target bands for seasonal peak growth (April to August)
        target_months = ['apr', 'may', 'jun', 'jul', 'aug']
        matched_bands = []
        if self.s1_ras and ref_ras == self.s1_ras:
            for b in range(1, nbands + 1):
                desc = str(ds.GetRasterBand(b).GetDescription()).lower()
                if any(m in desc for m in target_months):
                    matched_bands.append(b)

        if len(matched_bands) >= 4:
            selected_bands = matched_bands
            print(f"    [INFO] Generating seasonal SAR composite using {len(selected_bands)} April-August bands...")
        else:
            selected_bands = list(range(1, nbands + 1))
            print(f"    [INFO] Generating temporal SAR composite using all {len(selected_bands)} bands...")

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            str(composite_tif), cols, rows, 1, gdal.GDT_Float32,
            options=get_gdal_creation_options(predictor=3)
        )
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(0.0)

        tile_size = 4096
        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                xsize = min(tile_size, cols - x)
                ysize = min(tile_size, rows - y)

                sum_arr = np.zeros((ysize, xsize), dtype=np.float32)
                cnt_arr = np.zeros((ysize, xsize), dtype=np.int16)

                for b in selected_bands:
                    band = ds.GetRasterBand(b)
                    arr = band.ReadAsArray(x, y, xsize, ysize)
                    if arr is None:
                        continue
                    nodata = band.GetNoDataValue()
                    if nodata is not None:
                        valid = (arr != nodata) & (~np.isnan(arr)) & (arr != 0)
                    else:
                        valid = (~np.isnan(arr)) & (arr != 0)
                    sum_arr[valid] += arr[valid]
                    cnt_arr[valid] += 1

                mask = cnt_arr > 0
                out_block = np.zeros((ysize, xsize), dtype=np.float32)
                out_block[mask] = sum_arr[mask] / cnt_arr[mask]
                out_band.WriteArray(out_block, x, y)

        out_ds.FlushCache()
        out_ds = None
        ds = None
        print(f"    [OK] Seasonal SAR composite created successfully: {composite_tif.name}")
        return composite_tif

    # Backward compatibility alias
    _create_multimodal_pheno_composite = _create_seasonal_sar_composite

    def _create_summed_composite(self, ref_ras: Path) -> Path:
        print("    [INFO] Creating high-SNR summed composite for segmentation...")
        composite_tif = self.seg_dir / f"{self.file_prefix}_summed_composite.tif"
        if composite_tif.exists():
            return composite_tif

        ds = gdal.Open(str(ref_ras))
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        nbands = ds.RasterCount
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(str(composite_tif), cols, rows, 1, gdal.GDT_Float32,
                               options=get_gdal_creation_options(predictor=3))
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(0)

        tile_size = 4096
        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                xsize = min(tile_size, cols - x)
                ysize = min(tile_size, rows - y)

                sum_arr = np.zeros((ysize, xsize), dtype=np.float32)
                valid_mask = np.zeros((ysize, xsize), dtype=bool)

                for b in range(1, nbands + 1):
                    band = ds.GetRasterBand(b)
                    arr = band.ReadAsArray(x, y, xsize, ysize)
                    nodata = band.GetNoDataValue()
                    if nodata is not None:
                        mask = (arr != nodata) & (~np.isnan(arr)) & (arr != 0)
                    else:
                        mask = (~np.isnan(arr)) & (arr != 0)
                    sum_arr[mask] += arr[mask]
                    valid_mask |= mask

                sum_arr[~valid_mask] = 0
                out_band.WriteArray(sum_arr, x, y)

        out_ds.FlushCache()
        out_ds = None
        ds = None
        return composite_tif

    def _stitch_tile_seams(self, seg_ds, tile_size: int, cols: int, rows: int, comp_path: Path):
        """
        Fast Disjoint Set Union (Union-Find) boundary stitching pass across tile borders.
        Merges adjacent segment labels on vertical (x=2048, 4096...) and horizontal (y=2048, 4096...)
        tile seams if they belong to the same continuous field with consistent spectral NDVI.
        """
        print("    [TILE STITCHING] Eliminating tile seam artifacts across block boundaries...")
        seg_band = seg_ds.GetRasterBand(1)
        ds_comp = gdal.Open(str(comp_path), gdal.GA_ReadOnly) if comp_path.exists() else None
        ndvi_band = ds_comp.GetRasterBand(1) if ds_comp else None

        parent = {}

        def find(i):
            path = []
            while parent.get(i, i) != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        merges_count = 0

        # Dynamically determine seam difference threshold (0.8 dB for SAR dB composites, 0.05 for normalized [0, 1])
        seam_thresh = 0.05
        if ndvi_band:
            sample_data = ndvi_band.ReadAsArray(cols // 4, rows // 4, min(2048, cols // 2), min(2048, rows // 2))
            if sample_data is not None:
                v = sample_data[(sample_data != 0) & (~np.isnan(sample_data))]
                if len(v) > 0 and np.nanmean(v) < 0:
                    seam_thresh = 0.8

        # 1. Check vertical boundaries (along x = tile_size, 2*tile_size, ...)
        for x in range(tile_size, cols, tile_size):
            seg_col = seg_band.ReadAsArray(x - 1, 0, 2, rows)
            if seg_col is None:
                continue
            left_ids = seg_col[:, 0]
            right_ids = seg_col[:, 1]
            valid = (left_ids > 0) & (right_ids > 0) & (left_ids != right_ids)

            if np.any(valid):
                if ndvi_band:
                    comp_col = ndvi_band.ReadAsArray(x - 1, 0, 2, rows)
                    diff = np.abs(comp_col[:, 0] - comp_col[:, 1])
                    valid = valid & (diff < seam_thresh)

                cand_left = left_ids[valid]
                cand_right = right_ids[valid]
                pairs, counts = np.unique(np.column_stack([cand_left, cand_right]), axis=0, return_counts=True)
                for (lid, rid), cnt in zip(pairs, counts):
                    if cnt >= 8:
                        union(int(lid), int(rid))
                        merges_count += 1

        # 2. Check horizontal boundaries (along y = tile_size, 2*tile_size, ...)
        for y in range(tile_size, rows, tile_size):
            seg_row = seg_band.ReadAsArray(0, y - 1, cols, 2)
            if seg_row is None:
                continue
            top_ids = seg_row[0, :]
            bot_ids = seg_row[1, :]
            valid = (top_ids > 0) & (bot_ids > 0) & (top_ids != bot_ids)

            if np.any(valid):
                if ndvi_band:
                    comp_row = ndvi_band.ReadAsArray(0, y - 1, cols, 2)
                    diff = np.abs(comp_row[0, :] - comp_row[1, :])
                    valid = valid & (diff < seam_thresh)

                cand_top = top_ids[valid]
                cand_bot = bot_ids[valid]
                pairs, counts = np.unique(np.column_stack([cand_top, cand_bot]), axis=0, return_counts=True)
                for (tid, bid), cnt in zip(pairs, counts):
                    if cnt >= 8:
                        union(int(tid), int(bid))
                        merges_count += 1

        if ds_comp:
            ds_comp = None

        if merges_count == 0 or not parent:
            print("    [TILE STITCHING] No boundary seam merges required.")
            return

        # Flatten parent mappings
        remap = {}
        for k in parent.keys():
            root = find(k)
            if root != k:
                remap[int(k)] = int(root)

        print(f"    [TILE STITCHING] Merging {len(remap):,} boundary-fractured segment halves into unified parcels...")
        chunk_sz = 4096
        remap_keys_set = set(remap.keys())
        for y in range(0, rows, chunk_sz):
            for x in range(0, cols, chunk_sz):
                xs = min(chunk_sz, cols - x)
                ys = min(chunk_sz, rows - y)
                block = seg_band.ReadAsArray(x, y, xs, ys)
                if block is None:
                    continue
                u_ids = np.unique(block)
                intersect_keys = [k for k in u_ids if k in remap_keys_set]
                if intersect_keys:
                    for old_id in intersect_keys:
                        block[block == old_id] = remap[old_id]
                    seg_band.WriteArray(block, x, y)

        seg_band.FlushCache()
        print(f"    [TILE STITCHING COMPLETE] Tile seams successfully stitched.")

    # --- Stage 1: Footprint ---
    def stage_1_generate_footprint(self, force_recompute=False):
        stage = 1
        if self.footprint_mask.exists() and self.footprint_mask.stat().st_size > 1024 and not force_recompute:
            print(f"[Stage {stage}/{self.total_stages}] Footprint already exists ({self.footprint_mask.name}), skipping.")
            return

        print(f"[Stage {stage}/{self.total_stages}] Generating Multimodal Data Footprint (S1 SAR & S2 Optical Intersection)...")
        ref_ras = self.s1_ras if self.s1_ras else self.s2_ras
        if not ref_ras or not ref_ras.exists():
            raise FileNotFoundError("Neither S1 nor S2 raster found.")

        ds_s1 = gdal.Open(str(self.s1_ras)) if self.s1_ras else None
        ds_s2 = gdal.Open(str(self.s2_ras)) if self.s2_ras else None
        ref_ds = ds_s1 if ds_s1 else ds_s2

        cols, rows = ref_ds.RasterXSize, ref_ds.RasterYSize
        gt, proj = ref_ds.GetGeoTransform(), ref_ds.GetProjection()

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            str(self.footprint_mask), cols, rows, 1, gdal.GDT_Byte,
            options=get_gdal_creation_options(predictor=2)
        )
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)

        tile_size = 4096
        total_blocks = math.ceil(cols / tile_size) * math.ceil(rows / tile_size)
        done_blocks = 0

        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                xsize = min(tile_size, cols - x)
                ysize = min(tile_size, rows - y)

                mask = np.ones((ysize, xsize), dtype=bool)
                if ds_s1:
                    b1_s1 = ds_s1.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                    mask = mask & (b1_s1 != 0) & (b1_s1 != -9999) & (~np.isnan(b1_s1))
                if ds_s2:
                    b1_s2 = ds_s2.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                    mask = mask & (b1_s2 > 0) & (~np.isnan(b1_s2))

                out_ds.GetRasterBand(1).WriteArray(mask.astype(np.uint8), x, y)
                done_blocks += 1
                if done_blocks % 10 == 0 or done_blocks == total_blocks:
                    pct = (done_blocks / total_blocks) * 100.0
                    print(f"    [FOOTPRINT PROGRESS] {done_blocks}/{total_blocks} blocks completed ({pct:.1f}%)", flush=True)

        out_ds.FlushCache()
        out_ds = None
        ds_s1 = None
        ds_s2 = None
        print(f"    Multimodal intersection footprint saved to {self.footprint_mask}")

    # --- Stage 2: Segmentation (LPIS / SAM / SLIC) ---
    def stage_2_segmentation(self, force_recompute=False):
        stage = 2
        if self.seg_tif.exists() and not force_recompute:
            print(f"[Stage {stage}/{self.total_stages}] Segmentation raster exists ({self.seg_tif.name}), skipping.")
            return

        print(f"[Stage {stage}/{self.total_stages}] Running Multimodal Image Segmentation ({self.seg_mode.upper()})...")
        ref_ras = self.s1_ras if self.s1_ras else self.s2_ras
        if not ref_ras or not ref_ras.exists():
            raise FileNotFoundError("Reference raster not found for segmentation.")

        ds = gdal.Open(str(ref_ras))
        cols, rows = ds.RasterXSize, ds.RasterYSize
        gt, proj = ds.GetGeoTransform(), ds.GetProjection()

        if self.seg_mode == 'lpis':
            lpis_file = self._resolve_lpis_vector()
            if lpis_file and lpis_file.exists():
                print(f"    Loading official LPIS parcel vector from: {lpis_file}...")
                minx = gt[0]
                maxy = gt[3]
                maxx = minx + cols * gt[1]
                miny = maxy + rows * gt[5]

                try:
                    import pyogrio
                    info = pyogrio.read_info(str(lpis_file))
                    lpis_crs = info.get('crs')
                except Exception:
                    gdf_temp = gpd.read_file(str(lpis_file), rows=1)
                    lpis_crs = gdf_temp.crs.to_string() if gdf_temp.crs else None
                    info = {'fid_column': None}

                srs_target = osr.SpatialReference()
                srs_target.ImportFromWkt(proj)
                target_epsg = srs_target.GetAttrValue("AUTHORITY", 1) or "3857"

                from pyproj import Transformer
                transformer = Transformer.from_crs(f"EPSG:{target_epsg}", lpis_crs, always_xy=True)
                p1 = transformer.transform(minx, miny)
                p2 = transformer.transform(maxx, maxy)
                lpis_bbox = (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))

                print(f"    Querying LPIS with spatial filter bbox: {lpis_bbox}")
                try:
                    import pyogrio
                    gdf = pyogrio.read_dataframe(str(lpis_file), bbox=lpis_bbox)
                except Exception:
                    gdf = gpd.read_file(str(lpis_file), bbox=lpis_bbox)

                print(f"    Loaded {len(gdf)} intersecting parcels. Reprojecting to EPSG:{target_epsg}...")
                gdf_target = gdf.to_crs(f"EPSG:{target_epsg}")

                fid_col = info.get('fid_column') if isinstance(info, dict) else None
                if fid_col and fid_col in gdf_target.columns:
                    id_col = fid_col
                elif 'id' in gdf_target.columns:
                    id_col = 'id'
                elif 'id_0' in gdf_target.columns:
                    id_col = 'id_0'
                else:
                    id_col = None

                if id_col is None:
                    gdf_target['lpis_id'] = np.arange(1, len(gdf_target) + 1)
                    id_col = 'lpis_id'
                else:
                    gdf_target[id_col] = pd.to_numeric(gdf_target[id_col], errors='coerce').fillna(0).astype(np.int32)
                    if gdf_target[id_col].sum() == 0 or gdf_target[id_col].nunique() < len(gdf_target):
                        gdf_target['lpis_id'] = np.arange(1, len(gdf_target) + 1)
                        id_col = 'lpis_id'

                temp_gpkg = self.seg_dir / f"temp_lpis_{self.sanitized_track}.gpkg"
                gdf_target.geometry = gdf_target.geometry.force_2d()
                gdf_target.to_file(str(temp_gpkg), driver="GPKG", engine="pyogrio")

                driver = gdal.GetDriverByName("GTiff")
                ds_out = driver.Create(str(self.seg_tif), cols, rows, 1, gdal.GDT_Int32,
                                       options=get_gdal_creation_options(predictor=2))
                ds_out.SetGeoTransform(gt)
                ds_out.SetProjection(proj)

                band = ds_out.GetRasterBand(1)
                band.SetNoDataValue(0)
                band.Fill(0)

                print(f"    Rasterizing parcels to {self.seg_tif.name} (burning column '{id_col}')...")
                gdal.Rasterize(ds_out, str(temp_gpkg), attribute=id_col, callback=gdal.TermProgress_nocb)

                # Strictly mask with multimodal footprint (S1 + S2 intersection)
                if self.footprint_mask.exists():
                    print(f"    Masking LPIS segmentation with multimodal footprint ({self.footprint_mask.name})...")
                    ds_foot = gdal.Open(str(self.footprint_mask))
                    tile_size = 4096
                    for y in range(0, rows, tile_size):
                        for x in range(0, cols, tile_size):
                            xsize = min(tile_size, cols - x)
                            ysize = min(tile_size, rows - y)
                            seg_block = band.ReadAsArray(x, y, xsize, ysize)
                            foot_block = ds_foot.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                            if np.any(foot_block == 0):
                                seg_block[foot_block == 0] = 0
                                band.WriteArray(seg_block, x, y)
                    ds_foot = None

                ds_out.FlushCache()
                ds_out = None
                if os.path.exists(temp_gpkg):
                    try: os.remove(temp_gpkg)
                    except: pass
                print(f"    [OK] LPIS segmentation raster created and masked with footprint: {self.seg_tif}")
                return
            else:
                print(f"    [WARNING] LPIS vector dataset not found. Falling back to SLIC.")

        # Create high-SNR seasonal SAR composite (April-August peak crop growth)
        try:
            comp_ras = self._create_seasonal_sar_composite()
        except Exception as e:
            print(f"    [WARNING] Seasonal SAR composite failed ({e}), falling back to summed composite.")
            try:
                comp_ras = self._create_summed_composite(ref_ras)
            except Exception:
                comp_ras = ref_ras

        if self.seg_mode == 'sam':
            self._run_python_segmentation_tiled(comp_ras, self.stage1_params, 'python_sam')
        else:
            pixels_per_seg = max(10, int((self.slic_segment_ha * 10000.0) / 100.0))
            tile_sz = 2048
            buf_sz = 64
            max_tile_pixels = (tile_sz + 2 * buf_sz) ** 2
            n_segments_tile = max(100, int(max_tile_pixels / pixels_per_seg))
            rag_info = f" | RAG Fusion: thresh={self.slic_rag_thresh}" if self.enable_slic_rag else " | RAG Fusion: Disabled"
            print(f"    [SLIC TUNING] Target parcel size: {self.slic_segment_ha:.2f} ha (~{pixels_per_seg} px) | Compactness: {self.slic_compactness:.2f}{rag_info}")
            slic_params = {
                'tile_size': tile_sz,
                'buffer': buf_sz,
                'n_segments': n_segments_tile,
                'pixels_per_segment': pixels_per_seg,
                'compactness': self.slic_compactness,
                'slic_sigma': 1.5,
                'min_size_factor': 0.2,
                'enable_rag': self.enable_slic_rag,
                'rag_thresh': self.slic_rag_thresh,
                'max_rag_parcel_ha': 15.0
            }
            self._run_python_segmentation_tiled(comp_ras, slic_params, 'python_slic')

    def _run_python_segmentation_tiled(self, ras_path: Path, params: dict, method: str):
        is_sam = (method == 'python_sam')
        method_label = "SAM" if is_sam else "SLIC"
        print(f"    Running Tiled Python Segmentation ({method_label})...")
        ds = gdal.Open(str(ras_path))
        ds_foot = gdal.Open(str(self.footprint_mask)) if self.footprint_mask.exists() else None

        cols = ds.RasterXSize
        rows = ds.RasterYSize
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(str(self.seg_tif), cols, rows, 1, gdal.GDT_Int32,
                               options=get_gdal_creation_options(predictor=2))
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(0)

        tile_size = params.get('tile_size', 2048)
        buffer = params.get('buffer', 64 if not is_sam else 128)
        global_seg_id = 1

        from concurrent.futures import ProcessPoolExecutor, as_completed
        tile_tasks = []
        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                xsize_valid = min(tile_size, cols - x)
                ysize_valid = min(tile_size, rows - y)
                x_start_buf = max(0, x - buffer)
                y_start_buf = max(0, y - buffer)
                x_end_buf = min(cols, x + xsize_valid + buffer)
                y_end_buf = min(rows, y + ysize_valid + buffer)
                xsize_buf = x_end_buf - x_start_buf
                ysize_buf = y_end_buf - y_start_buf

                if ds_foot:
                    band_foot = ds_foot.GetRasterBand(1)
                    arr_foot = band_foot.ReadAsArray(x_start_buf, y_start_buf, xsize_buf, ysize_buf)
                    valid_mask = arr_foot > 0 if arr_foot is not None else None
                else:
                    band = ds.GetRasterBand(1)
                    arr = band.ReadAsArray(x_start_buf, y_start_buf, xsize_buf, ysize_buf)
                    valid_mask = np.nan_to_num(arr) > 0 if arr is not None else None

                if valid_mask is not None and np.any(valid_mask):
                    tile_tasks.append((x, y, xsize_valid, ysize_valid, x_start_buf, y_start_buf, xsize_buf, ysize_buf, buffer))

        total_tasks = len(tile_tasks)
        print(f"    Total active tiles to process with {method_label}: {total_tasks}")

        if is_sam:
            max_workers = min(8, os.cpu_count() or 4)
            worker_fn = sam_worker
        else:
            # Parallel multi-core SLIC: use available CPU cores (leaving 2 for system responsiveness)
            max_workers = max(1, min(14, (os.cpu_count() or 4) - 2))
            worker_fn = slic_worker
            print(f"    [MULTI-CORE SLIC] Parallelizing across {max_workers} CPU worker processes.")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(worker_fn, task, str(ras_path), str(self.footprint_mask) if self.footprint_mask.exists() else None, params): task
                for task in tile_tasks
            }
            completed_count = 0
            for future in as_completed(futures):
                x, y, segments, valid_mask_crop = future.result()
                completed_count += 1
                pct = (completed_count / total_tasks) * 100.0
                print(f"    [{method_label} PROGRESS] Tile {completed_count}/{total_tasks} ({pct:.1f}%) finished (x={x}, y={y}) | Total segments: {global_seg_id-1:,}", flush=True)

                if segments is not None:
                    seg_valid_mask = segments > 0
                    unique_segs = np.unique(segments[seg_valid_mask])
                    if len(unique_segs) > 0:
                        max_seg = segments.max()
                        mapping = np.zeros(max_seg + 1, dtype=np.int32)
                        mapping[unique_segs] = np.arange(global_seg_id, global_seg_id + len(unique_segs))
                        segments = mapping[segments]
                        segments[~valid_mask_crop] = 0
                        global_seg_id += len(unique_segs)
                    else:
                        segments[~valid_mask_crop] = 0
                    out_band.WriteArray(segments.astype(np.int32), x, y)
                else:
                    for task in tile_tasks:
                        if task[0] == x and task[1] == y:
                            xsize_valid, ysize_valid = task[2], task[3]
                            out_band.WriteArray(np.zeros((ysize_valid, xsize_valid), dtype=np.int32), x, y)
                            break

        # Stitch tile seams to eliminate boundary-cut parcel fracturing
        if not is_sam:
            self._stitch_tile_seams(out_ds, tile_size, cols, rows, ras_path)
            print("    [SIEVE FILTER] Absorbing isolated micro-slivers (< 30 px) into adjacent parcels...")
            gdal.SieveFilter(out_band, None, out_band, 30, 8, callback=None)

        out_band.FlushCache()
        out_ds = None
        ds = None
        ds_foot = None
        print(f"    Segmentation completed: {self.seg_tif}")

    # --- Stage 3: Sample Point Split (70/30) ---
    def stage_3_split_samples(self, force_recompute=False, learn_frac=0.7, random_state=42):
        stage = 3
        if self.learn_shp.exists() and self.control_shp.exists() and not force_recompute:
            print(f"[Stage {stage}/{self.total_stages}] Train/Validation sample split exists, skipping.")
            return

        print(f"[Stage {stage}/{self.total_stages}] Performing Stratified Train/Control Split...")
        if not self.sample_shp or not self.sample_shp.exists():
            raise FileNotFoundError(f"Sample shapefile not found at {self.sample_shp}")

        gdf = gpd.read_file(str(self.sample_shp), engine="pyogrio")
        crop_col = 'crop_id' if 'crop_id' in gdf.columns else 'code'
        gdf['crop_id'] = gdf[crop_col].astype(int)

        train_dfs, val_dfs = [], []
        for cid, group in gdf.groupby('crop_id'):
            if len(group) == 1:
                train_dfs.append(group)
            else:
                train_g = group.sample(frac=learn_frac, random_state=random_state)
                val_g = group.drop(train_g.index)
                train_dfs.append(train_g)
                val_dfs.append(val_g)

        gdf_train = pd.concat(train_dfs)
        gdf_val = pd.concat(val_dfs)

        gdf_train.to_file(str(self.learn_shp), engine="pyogrio")
        gdf_val.to_file(str(self.control_shp), engine="pyogrio")
        print(f"    Split {len(gdf)} points -> {len(gdf_train)} train, {len(gdf_val)} validation.")

    # --- Stage 4: Multimodal Feature Extraction ---
    # --- Stage 4: Multimodal Feature Extraction (Enhanced Vectorized Chunk I/O) ---
    def stage_4_selection(self, force_recompute=False):
        stage = 4
        if self.sel_csv.exists() and not force_recompute:
            try:
                df_test = pd.read_csv(self.sel_csv, nrows=5)
                n_feats = df_test.shape[1] - 2
                if self.s1_ras and self.s2_ras and n_feats < 400:
                    print(f"[Stage {stage}/{self.total_stages}] Existing CSV has only {n_feats} features (expected ~450+ multimodal features). Re-extracting...")
                else:
                    print(f"[Stage {stage}/{self.total_stages}] Multimodal features already extracted ({n_feats} features in {self.sel_csv.name}), skipping.")
                    return
            except Exception:
                pass

        print(f"[Stage {stage}/{self.total_stages}] Extracting Multimodal Features (S1 Sigma0 + S2 Optical + Joint Presto + Physical Indices)...")
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        extractor = PrestoMultimodalExtractor(device=device)

        gdf = gpd.read_file(str(self.learn_shp), engine="pyogrio")
        seg_ds = gdal.Open(str(self.seg_tif))
        seg_band = seg_ds.GetRasterBand(1)
        cols, rows = seg_ds.RasterXSize, seg_ds.RasterYSize
        gt = seg_ds.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)
        proj = seg_ds.GetProjection()

        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        if proj and gdf.crs:
            from pyproj import CRS
            target_crs = CRS.from_wkt(proj)
            gdf_proj = gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf
        else:
            gdf_proj = gdf

        if hasattr(gdf_proj.geometry, 'x'):
            xs, ys = gdf_proj.geometry.x.values, gdf_proj.geometry.y.values
        else:
            xs, ys = gdf_proj.geometry.centroid.x.values, gdf_proj.geometry.centroid.y.values

        pxs = (inv_gt[0] + inv_gt[1] * xs + inv_gt[2] * ys).astype(int)
        pys = (inv_gt[3] + inv_gt[4] * xs + inv_gt[5] * ys).astype(int)
        crop_ids = gdf['crop_id'].values

        target_segments = {}
        segment_coords = {}
        for idx, (px, py, cid) in enumerate(zip(pxs, pys, crop_ids)):
            if 0 <= px < cols and 0 <= py < rows:
                sid_arr = seg_band.ReadAsArray(px, py, 1, 1)
                sid = int(sid_arr[0, 0]) if sid_arr is not None else 0
                if sid > 0:
                    target_segments[sid] = cid
                    geom_wgs = gdf_wgs84.iloc[idx].geometry
                    centroid_wgs = geom_wgs.centroid if hasattr(geom_wgs, 'centroid') else geom_wgs
                    segment_coords[sid] = (centroid_wgs.y, centroid_wgs.x)

        print(f"    Found {len(target_segments)} unique training segments.")
        target_sids = set(target_segments.keys())

        ds_s1 = gdal.Open(str(self.s1_ras)) if self.s1_ras else None
        ds_s2 = gdal.Open(str(self.s2_ras)) if self.s2_ras else None

        nbands_s1 = ds_s1.RasterCount if ds_s1 else 0
        nbands_s2 = ds_s2.RasterCount if ds_s2 else 0
        num_dates_s1 = nbands_s1 // 2
        num_dates_s2 = nbands_s2 // 9

        months_s1 = [parse_month_from_description(ds_s1.GetRasterBand(b).GetDescription()) for b in range(1, num_dates_s1 + 1)] if ds_s1 else [0]
        months_s2 = [parse_month_from_description(ds_s2.GetRasterBand(b).GetDescription()) for b in range(1, num_dates_s2 + 1)] if ds_s2 else [0]

        month_tensor_s1 = torch.tensor(months_s1, dtype=torch.long, device=device)
        month_tensor_s2 = torch.tensor(months_s2, dtype=torch.long, device=device)
        month_tensor_joint = torch.tensor(months_s2 if num_dates_s2 >= num_dates_s1 else months_s1, dtype=torch.long, device=device)

        print("    Accumulating segment statistics using fast vectorized chunked I/O...")
        t_io_start = time.time()
        tile_size = 2048
        accum_counts = {sid: 0 for sid in target_sids}
        accum_s1_sums = {sid: np.zeros(nbands_s1, dtype=np.float64) for sid in target_sids} if ds_s1 else {}
        accum_s1_sq_sums = {sid: np.zeros(nbands_s1, dtype=np.float64) for sid in target_sids} if ds_s1 else {}
        accum_s2_sums = {sid: np.zeros(nbands_s2, dtype=np.float64) for sid in target_sids} if ds_s2 else {}
        accum_s2_sq_sums = {sid: np.zeros(nbands_s2, dtype=np.float64) for sid in target_sids} if ds_s2 else {}
        target_sids_set = set(target_sids)

        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                xsize = min(tile_size, cols - x)
                ysize = min(tile_size, rows - y)

                sub_seg = seg_band.ReadAsArray(x, y, xsize, ysize)
                if sub_seg is None:
                    continue
                u_in_tile, inv_arr = np.unique(sub_seg, return_inverse=True)
                present_pairs = [(int(s), idx) for idx, s in enumerate(u_in_tile) if s in target_sids_set]
                if not present_pairs:
                    continue

                flat_inv = inv_arr.ravel()
                counts = np.bincount(flat_inv)
                for sid, idx in present_pairs:
                    accum_counts[sid] += int(counts[idx])

                if ds_s1:
                    s1_tile = np.nan_to_num(ds_s1.ReadAsArray(x, y, xsize, ysize).astype(np.float32))
                    if s1_tile.ndim == 2: s1_tile = s1_tile[np.newaxis, ...]
                    # Convert SAR dB to linear power scale for unbiased physical aggregation
                    s1_tile_lin = np.power(10.0, np.clip(s1_tile, -45.0, 15.0) / 10.0)
                    for b in range(nbands_s1):
                        b_lin = s1_tile_lin[b].ravel()
                        sums = np.bincount(flat_inv, weights=b_lin)
                        sq_sums = np.bincount(flat_inv, weights=b_lin ** 2)
                        for sid, idx in present_pairs:
                            accum_s1_sums[sid][b] += float(sums[idx])
                            accum_s1_sq_sums[sid][b] += float(sq_sums[idx])

                if ds_s2:
                    s2_tile = np.nan_to_num(ds_s2.ReadAsArray(x, y, xsize, ysize).astype(np.float32))
                    if s2_tile.ndim == 2: s2_tile = s2_tile[np.newaxis, ...]
                    for b in range(nbands_s2):
                        b_vals = s2_tile[b].ravel()
                        sums = np.bincount(flat_inv, weights=b_vals)
                        sq_sums = np.bincount(flat_inv, weights=b_vals ** 2)
                        for sid, idx in present_pairs:
                            accum_s2_sums[sid][b] += float(sums[idx])
                            accum_s2_sq_sums[sid][b] += float(sq_sums[idx])

        print(f"    Vectorized chunk I/O completed in {time.time() - t_io_start:.1f}s.")

        valid_sids = [sid for sid in target_sids if accum_counts[sid] > 0]
        s1_means_list = []
        s1_stds_list = []
        s2_means_list = []
        s2_stds_list = []
        batch_records = []
        batch_s1_profiles = []
        batch_s2_profiles = []
        batch_latlons = []

        for sid in valid_sids:
            cid = target_segments[sid]
            cnt = accum_counts[sid]
            lat, lon = segment_coords.get(sid, (52.0, 5.0))
            record = {'crop_id': cid, 'seg_id': sid}

            s1_mean = None
            if ds_s1:
                # Unbiased mean in dB from linear power accumulation
                mean_lin = (accum_s1_sums[sid] / cnt).astype(np.float64)
                var_lin = (accum_s1_sq_sums[sid] / cnt) - (mean_lin ** 2)
                std_lin = np.sqrt(np.maximum(var_lin, 0.0))
                # Convert linear mean back to physical dB
                s1_mean = (10.0 * np.log10(np.maximum(mean_lin, 1e-4))).astype(np.float32)
                # Scale-invariant texture (CV = std / mean)
                s1_std = (std_lin / (mean_lin + 1e-6)).astype(np.float32)
                s1_means_list.append(s1_mean)
                s1_stds_list.append(s1_std)

                for b_i, val in enumerate(s1_mean):
                    record[f's1_b{b_i}'] = float(val)

                s1_prof = np.zeros((num_dates_s1, 2), dtype=np.float32)
                for d in range(num_dates_s1):
                    s1_prof[d, 0] = (s1_mean[num_dates_s1 + d] + 25.0) / 25.0
                    s1_prof[d, 1] = (s1_mean[d] + 25.0) / 25.0
                batch_s1_profiles.append(s1_prof)

            s2_mean = None
            if ds_s2:
                s2_mean = (accum_s2_sums[sid] / cnt).astype(np.float32)
                var_s2 = (accum_s2_sq_sums[sid] / cnt) - (s2_mean.astype(np.float64) ** 2)
                s2_std = np.sqrt(np.maximum(var_s2, 0.0)).astype(np.float32)
                s2_means_list.append(s2_mean)
                s2_stds_list.append(s2_std)

                for b_i, val in enumerate(s2_mean):
                    record[f's2_b{b_i}'] = float(val)

                s2_prof = np.zeros((num_dates_s2, 9), dtype=np.float32)
                for d in range(num_dates_s2):
                    for band_idx in range(9):
                        s2_prof[d, band_idx] = s2_mean[d * 9 + band_idx] / 10000.0
                batch_s2_profiles.append(s2_prof)

            batch_records.append(record)
            batch_latlons.append([lat, lon])

        # Physical indices with intra-object variance / texture features
        s1_means_arr = np.array(s1_means_list, dtype=np.float32) if s1_means_list else None
        s1_stds_arr = np.array(s1_stds_list, dtype=np.float32) if s1_stds_list else None
        s2_means_arr = np.array(s2_means_list, dtype=np.float32) if s2_means_list else None
        s2_stds_arr = np.array(s2_stds_list, dtype=np.float32) if s2_stds_list else None

        phys_indices_arr, phys_names = compute_vegetation_and_sar_indices(
            s1_means_arr, s2_means_arr, num_dates_s1, num_dates_s2,
            s1_stds=s1_stds_arr, s2_stds=s2_stds_arr
        )
        print(f"    Computed {phys_indices_arr.shape[1]} physical Red-Edge, SAR phenological & texture features.")
        for r_idx, rec in enumerate(batch_records):
            for f_i, f_name in enumerate(phys_names):
                rec[f_name] = float(phys_indices_arr[r_idx, f_i])

        print(f"    Computing Presto Joint Multimodal embeddings in parallel batches (total {len(batch_records)} valid segments)...")
        batch_size = 256
        n_batches = math.ceil(len(batch_records) / batch_size) if batch_records else 0
        for b_idx in range(n_batches):
            start_i = b_idx * batch_size
            end_i = min(len(batch_records), (b_idx + 1) * batch_size)
            b_ll = torch.tensor(batch_latlons[start_i:end_i], dtype=torch.float32, device=device)

            b_s1_t = torch.from_numpy(np.stack(batch_s1_profiles[start_i:end_i], axis=0)).to(device) if batch_s1_profiles else None
            b_s2_t = torch.from_numpy(np.stack(batch_s2_profiles[start_i:end_i], axis=0)).to(device) if batch_s2_profiles else None

            embs_joint = extractor.get_joint_embeddings(b_s1_t, b_s2_t, b_ll, month_tensor_joint)
            for rec_i, emb in enumerate(embs_joint):
                for f_i, f_val in enumerate(emb):
                    batch_records[start_i + rec_i][f'presto_joint_{f_i}'] = float(f_val)

            if b_s1_t is not None:
                embs_s1 = extractor.get_s1_embeddings(b_s1_t, b_ll, month_tensor_s1)
                for rec_i, emb in enumerate(embs_s1):
                    for f_i, f_val in enumerate(emb):
                        batch_records[start_i + rec_i][f'presto_s1_{f_i}'] = float(f_val)

            if b_s2_t is not None:
                embs_s2 = extractor.get_s2_embeddings(b_s2_t, b_ll, month_tensor_s2)
                for rec_i, emb in enumerate(embs_s2):
                    for f_i, f_val in enumerate(emb):
                        batch_records[start_i + rec_i][f'presto_s2_{f_i}'] = float(f_val)

            sys.stdout.write(f"\r    Presto batch progress: {end_i}/{len(batch_records)} segments ({(end_i/len(batch_records)*100):.1f}%)...  ")
            sys.stdout.flush()

        df = pd.DataFrame(batch_records)
        df.to_csv(self.sel_csv, index=False)
        print(f"\n    Multimodal features saved to {self.sel_csv} ({df.shape[1] - 2} features).\n")
    # --- Stage 5: Train Unified MLP + XGBoost Fusion Ensemble (with Outlier Cleaning) ---
    def stage_5_train_classifier(self, force_recompute=False, **kwargs):
        stage = 5
        if self.model_pkl.exists() and not force_recompute:
            print(f"[Stage {stage}/{self.total_stages}] Fusion Ensemble model exists ({self.model_pkl.name}), skipping.")
            return

        print(f"[Stage {stage}/{self.total_stages}] Training Unified Multimodal Fusion Ensemble (PyTorch MLP + XGBoost with Label Noise Filtering)...")

        df = pd.read_csv(self.sel_csv)
        y = df['crop_id'].values
        X = df.drop(columns=['crop_id', 'seg_id']).values

        all_classes = np.unique(y)
        clean_mask = _clean_label_noise(X, y, all_classes, prune_pct=0.02)
        X = X[clean_mask]
        y = y[clean_mask]

        class_weights = _calculate_class_weights(y, all_classes)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        mlp = TorchMLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            max_iter=200,
            batch_size=256,
            lr=0.001,
            class_weights=class_weights,
            all_classes=all_classes
        )

        n_jobs = int(os.environ.get("OMP_NUM_THREADS", 4))
        if HAS_XGBOOST:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=350,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method='hist',
                random_state=42,
                n_jobs=n_jobs,
                eval_metric='mlogloss'
            )
        else:
            print("    [INFO] XGBoost not found in environment; using sklearn.ensemble.HistGradientBoostingClassifier.")
            xgb_clf = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.08,
                max_depth=7,
                random_state=42
            )

        ensemble = EnsembleClassifier(mlp_model=mlp, xgb_model=xgb_clf, weight_mlp=self.mlp_weight)
        ensemble.fit(X_scaled, y)

        joblib.dump({'model': ensemble, 'scaler': scaler, 'features': df.drop(columns=['crop_id', 'seg_id']).columns.tolist()}, self.model_pkl)
        print(f"    [OK] Model successfully trained and serialized: {self.model_pkl}\n")
    # --- Stage 6: Tile-based Object Inference with Bayesian Priors & Spatial Uncertainty ---
    def stage_6_classify_vector(self, force_recompute=False):
        stage = 6
        if self.class_tif.exists() and self.entropy_tif.exists() and not force_recompute:
            print(f"[Stage {stage}/{self.total_stages}] Classification rasters exist, skipping.")
            return

        print(f"[Stage {stage}/{self.total_stages}] Running Vectorized Tile-based Inference with Bayesian Priors & Spatial Uncertainty...")
        main_mod = sys.modules.get('__main__')
        if main_mod:
            setattr(main_mod, 'EnsembleClassifier', EnsembleClassifier)
            setattr(main_mod, 'TorchMLPClassifier', TorchMLPClassifier)
        sys.modules['1_classify_MLPXGB_presto_hybrid_S1S2'] = sys.modules[__name__]
        sys.modules['classifier_mlpxgb_presto'] = sys.modules[__name__]
        sys.modules['classifier_mlpxgb_presto_S1S2'] = sys.modules[__name__]

        data = joblib.load(self.model_pkl)
        clf = data['model']
        scaler = data['scaler']

        ref_ras = self.s1_ras if self.s1_ras else self.s2_ras
        ds_info = gdal.Open(str(ref_ras))
        cols, rows = ds_info.RasterXSize, ds_info.RasterYSize
        gt, proj = ds_info.GetGeoTransform(), ds_info.GetProjection()

        driver = gdal.GetDriverByName('GTiff')
        ds_cls = driver.Create(str(self.class_tif), cols, rows, 1, gdal.GDT_Int32, options=get_gdal_creation_options(predictor=2))
        ds_cls.SetGeoTransform(gt)
        ds_cls.SetProjection(proj)

        ds_conf = driver.Create(str(self.conf_tif), cols, rows, 1, gdal.GDT_Float32, options=get_gdal_creation_options(predictor=3))
        ds_conf.SetGeoTransform(gt)
        ds_conf.SetProjection(proj)

        ds_ent = driver.Create(str(self.entropy_tif), cols, rows, 1, gdal.GDT_Float32, options=get_gdal_creation_options(predictor=3))
        ds_ent.SetGeoTransform(gt)
        ds_ent.SetProjection(proj)

        ds_mar = driver.Create(str(self.margin_tif), cols, rows, 1, gdal.GDT_Float32, options=get_gdal_creation_options(predictor=3))
        ds_mar.SetGeoTransform(gt)
        ds_mar.SetProjection(proj)

        df_learn = pd.read_csv(self.sel_csv)
        classes = clf.classes_
        class_counts = df_learn['crop_id'].value_counts().to_dict()

        id_to_name = {}
        shp_to_check = self.learn_shp if (hasattr(self, 'learn_shp') and self.learn_shp and self.learn_shp.exists()) else self.sample_shp
        if shp_to_check and shp_to_check.exists():
            try:
                gdf_n = gpd.read_file(str(shp_to_check), engine="pyogrio")
                col_map = {c.lower(): c for c in gdf_n.columns}
                id_col_key = next((k for k in ['crop_id', 'crop_ids', 'code', 'id', 'class_id'] if k in col_map), None)
                name_col_key = next((k for k in ['crop_name', 'crop_names', 'crop_type', 'label', 'name', 'class_name', 'crop', 'nom'] if k in col_map), None)
                if id_col_key and name_col_key:
                    id_to_name = dict(zip(gdf_n[col_map[id_col_key]].astype(int), gdf_n[col_map[name_col_key]].astype(str)))
            except Exception as e:
                print(f"    [WARNING] Could not read crop names from shapefile: {e}")

        country_priors_file = self.aux_dir / "shapefiles_samples" / self.country / "priors.json"
        if country_priors_file.exists():
            try:
                with open(country_priors_file, 'r', encoding='utf-8') as pf:
                    priors_data = json.load(pf)
                    for p_idx, p_name in enumerate(priors_data.keys(), start=1):
                        if p_idx not in id_to_name:
                            id_to_name[p_idx] = p_name.title()
            except Exception:
                pass

        priors_arr = compute_dynamic_bayesian_priors(
            classes=classes,
            class_counts=class_counts,
            total_samples=len(df_learn),
            id_to_name=id_to_name
        )
        print("    Dynamic Bayesian class priors calculated:")
        for c_val, p_val in zip(classes, priors_arr):
            c_name = id_to_name.get(int(c_val), f"Class {c_val}")
            c_cnt = class_counts.get(c_val, 0)
            print(f"      - Class {int(c_val):>2} ({c_name:<26}): samples={c_cnt:>5} -> prior={p_val * 100:5.2f}%")

        seg_ds = gdal.Open(str(self.seg_tif))
        foot_ds = gdal.Open(str(self.footprint_mask))

        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        extractor = PrestoMultimodalExtractor(device=device)

        ds_s1 = gdal.Open(str(self.s1_ras)) if self.s1_ras else None
        ds_s2 = gdal.Open(str(self.s2_ras)) if self.s2_ras else None
        nbands_s1 = ds_s1.RasterCount if ds_s1 else 0
        nbands_s2 = ds_s2.RasterCount if ds_s2 else 0
        num_dates_s1 = nbands_s1 // 2
        num_dates_s2 = nbands_s2 // 9

        months_s1 = [parse_month_from_description(ds_s1.GetRasterBand(b).GetDescription()) for b in range(1, num_dates_s1 + 1)] if ds_s1 else [0]
        months_s2 = [parse_month_from_description(ds_s2.GetRasterBand(b).GetDescription()) for b in range(1, num_dates_s2 + 1)] if ds_s2 else [0]
        month_tensor_s1 = torch.tensor(months_s1, dtype=torch.long, device=device)
        month_tensor_s2 = torch.tensor(months_s2, dtype=torch.long, device=device)
        month_tensor_joint = torch.tensor(months_s2 if num_dates_s2 >= num_dates_s1 else months_s1, dtype=torch.long, device=device)

        from scipy import ndimage
        srs_ras = osr.SpatialReference()
        srs_ras.ImportFromWkt(proj)
        ras_epsg = srs_ras.GetAttrValue("AUTHORITY", 1) or "3857"
        from pyproj import Transformer
        transformer_to_wgs84 = Transformer.from_crs(f"EPSG:{ras_epsg}", "EPSG:4326", always_xy=True)

        tile_size = 2048
        tile_coords = []
        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                xsize = min(tile_size, cols - x)
                ysize = min(tile_size, rows - y)
                tile_coords.append((x, y, xsize, ysize))

        total_tiles = len(tile_coords)
        total_segments_classified = 0
        t_infer_start = time.time()

        print(f"    Starting asynchronous tile-based inference with Producer-Consumer GPU pipeline ({total_tiles} tiles of {tile_size}x{tile_size} px)...")

        batch_queue = queue.Queue(maxsize=2)
        producer_error = [None]

        def _tile_producer_worker():
            try:
                for idx, (x, y, xsize, ysize) in enumerate(tile_coords):
                    if producer_error[0] is not None:
                        break
                    tile_cnt = idx + 1
                    sub_seg = seg_ds.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                    foot_arr = foot_ds.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)

                    u_all, inv_arr = np.unique(sub_seg, return_inverse=True)
                    has_zero = (len(u_all) > 0 and u_all[0] == 0)
                    offset = 1 if has_zero else 0
                    u_sids = u_all[offset:]

                    if len(u_sids) == 0:
                        batch_queue.put({
                            'empty': True,
                            'x': x, 'y': y, 'xsize': xsize, 'ysize': ysize,
                            'tile_cnt': tile_cnt
                        })
                        continue

                    flat_inv = inv_arr.ravel()
                    counts = np.bincount(flat_inv)
                    valid_counts = np.maximum(counts[offset:], 1)

                    # Vectorized centroids via local bincount
                    yy, xx = np.indices((ysize, xsize), dtype=np.float32)
                    sum_y = np.bincount(flat_inv, weights=yy.ravel())[offset:]
                    sum_x = np.bincount(flat_inv, weights=xx.ravel())[offset:]
                    cy_arr = (sum_y / valid_counts) + y
                    cx_arr = (sum_x / valid_counts) + x
                    mx_arr = gt[0] + cx_arr * gt[1] + cy_arr * gt[2]
                    my_arr = gt[3] + cx_arr * gt[4] + cy_arr * gt[5]
                    lons, lats = transformer_to_wgs84.transform(mx_arr, my_arr)
                    latlons = np.column_stack([lats, lons]).astype(np.float32)
                    b_ll = torch.from_numpy(latlons)

                    s1_means = None
                    s1_stds = None
                    b_s1 = None
                    if ds_s1:
                        s1_tile = np.nan_to_num(ds_s1.ReadAsArray(x, y, xsize, ysize).astype(np.float32))
                        if s1_tile.ndim == 2: s1_tile = s1_tile[np.newaxis, ...]
                        s1_tile_lin = np.power(10.0, np.clip(s1_tile, -45.0, 15.0) / 10.0)
                        s1_means = np.zeros((len(u_sids), nbands_s1), dtype=np.float32)
                        s1_stds = np.zeros((len(u_sids), nbands_s1), dtype=np.float32)
                        for b in range(nbands_s1):
                            b_lin = s1_tile_lin[b].ravel()
                            sums_lin = np.bincount(flat_inv, weights=b_lin)
                            sq_sums_lin = np.bincount(flat_inv, weights=b_lin ** 2)
                            m_lin = sums_lin[offset:] / valid_counts
                            v_lin = (sq_sums_lin[offset:] / valid_counts) - (m_lin ** 2)
                            std_lin = np.sqrt(np.maximum(v_lin, 0.0))
                            s1_means[:, b] = (10.0 * np.log10(np.maximum(m_lin, 1e-4))).astype(np.float32)
                            s1_stds[:, b] = (std_lin / (m_lin + 1e-6)).astype(np.float32)

                        s1_profiles = np.zeros((len(u_sids), num_dates_s1, 2), dtype=np.float32)
                        for d in range(num_dates_s1):
                            s1_profiles[:, d, 0] = (s1_means[:, num_dates_s1 + d] + 25.0) / 25.0
                            s1_profiles[:, d, 1] = (s1_means[:, d] + 25.0) / 25.0
                        b_s1 = torch.from_numpy(s1_profiles)

                    s2_means = None
                    s2_stds = None
                    b_s2 = None
                    if ds_s2:
                        s2_tile = np.nan_to_num(ds_s2.ReadAsArray(x, y, xsize, ysize).astype(np.float32))
                        if s2_tile.ndim == 2: s2_tile = s2_tile[np.newaxis, ...]
                        s2_means = np.zeros((len(u_sids), nbands_s2), dtype=np.float32)
                        s2_stds = np.zeros((len(u_sids), nbands_s2), dtype=np.float32)
                        for b in range(nbands_s2):
                            b_vals = s2_tile[b].ravel()
                            sums = np.bincount(flat_inv, weights=b_vals)
                            sq_sums = np.bincount(flat_inv, weights=b_vals ** 2)
                            m_s2 = sums[offset:] / valid_counts
                            v_s2 = (sq_sums[offset:] / valid_counts) - (m_s2 ** 2)
                            s2_means[:, b] = m_s2
                            s2_stds[:, b] = np.sqrt(np.maximum(v_s2, 0.0)).astype(np.float32)

                        s2_profiles = np.zeros((len(u_sids), num_dates_s2, 9), dtype=np.float32)
                        for d in range(num_dates_s2):
                            for band_idx in range(9):
                                s2_profiles[:, d, band_idx] = s2_means[:, d * 9 + band_idx] / 10000.0
                        b_s2 = torch.from_numpy(s2_profiles)

                    # Physical indices with intra-object variance / texture features
                    phys_indices, _ = compute_vegetation_and_sar_indices(
                        s1_means, s2_means, num_dates_s1, num_dates_s2,
                        s1_stds=s1_stds, s2_stds=s2_stds
                    )

                    batch_queue.put({
                        'empty': False,
                        'x': x, 'y': y, 'xsize': xsize, 'ysize': ysize,
                        'tile_cnt': tile_cnt,
                        'u_all': u_all, 'inv_arr': inv_arr, 'offset': offset,
                        'u_sids': u_sids, 'foot_arr': foot_arr,
                        'b_ll': b_ll, 'b_s1': b_s1, 'b_s2': b_s2,
                        's1_means': s1_means, 's2_means': s2_means,
                        'phys_indices': phys_indices
                    })
                batch_queue.put(None)
            except Exception as e:
                producer_error[0] = e
                batch_queue.put(None)

        producer_thread = threading.Thread(target=_tile_producer_worker, daemon=True)
        producer_thread.start()

        while True:
            item = batch_queue.get()
            if item is None:
                batch_queue.task_done()
                break

            x = item['x']
            y = item['y']
            xsize = item['xsize']
            ysize = item['ysize']
            tile_cnt = item['tile_cnt']

            if item['empty']:
                if tile_cnt % 25 == 0 or tile_cnt == total_tiles:
                    elapsed = time.time() - t_infer_start
                    rate = tile_cnt / elapsed if elapsed > 0 else 0
                    eta_sec = (total_tiles - tile_cnt) / rate if rate > 0 else 0
                    eta_str = f"{int(eta_sec//60)}m {int(eta_sec%60):02d}s"
                    sys.stdout.write(
                        f"\r    [INFERENCE] Tile {tile_cnt}/{total_tiles} ({(tile_cnt/total_tiles*100):.1f}%) | "
                        f"Objects: {total_segments_classified:,} | Time: {int(elapsed//60)}m {int(elapsed%60):02d}s | ETA: {eta_str}  "
                    )
                    sys.stdout.flush()
                batch_queue.task_done()
                continue

            u_all = item['u_all']
            inv_arr = item['inv_arr']
            offset = item['offset']
            u_sids = item['u_sids']
            foot_arr = item['foot_arr']
            s1_means = item['s1_means']
            s2_means = item['s2_means']
            phys_indices = item['phys_indices']

            b_ll = item['b_ll'].to(device, non_blocking=True)
            b_s1 = item['b_s1'].to(device, non_blocking=True) if item['b_s1'] is not None else None
            b_s2 = item['b_s2'].to(device, non_blocking=True) if item['b_s2'] is not None else None

            embs_s1 = extractor.get_s1_embeddings(b_s1, b_ll, month_tensor_s1) if b_s1 is not None else None
            embs_s2 = extractor.get_s2_embeddings(b_s2, b_ll, month_tensor_s2) if b_s2 is not None else None
            embs_joint = extractor.get_joint_embeddings(b_s1, b_s2, b_ll, month_tensor_joint)

            feat_blocks = []
            if s1_means is not None: feat_blocks.append(s1_means)
            if s2_means is not None: feat_blocks.append(s2_means)
            if phys_indices.shape[1] > 0: feat_blocks.append(phys_indices)
            if embs_joint is not None: feat_blocks.append(embs_joint)
            if embs_s1 is not None: feat_blocks.append(embs_s1)
            if embs_s2 is not None: feat_blocks.append(embs_s2)

            X_tile = np.hstack(feat_blocks)
            X_tile_scaled = scaler.transform(X_tile)
            raw_probs = clf.predict_proba(X_tile_scaled)

            corr_probs = raw_probs * priors_arr
            corr_probs = corr_probs / np.sum(corr_probs, axis=1, keepdims=True)

            preds = clf.classes_[np.argmax(corr_probs, axis=1)]
            confs = np.max(corr_probs, axis=1)

            # Normalized Shannon Entropy
            n_classes = len(clf.classes_)
            log_k = np.log(max(n_classes, 2))
            entropies = -np.sum(corr_probs * np.log(corr_probs + 1e-12), axis=1) / log_k
            entropies = np.clip(entropies, 0.0, 1.0)

            # Confidence Margin
            if n_classes > 1:
                part_probs = np.partition(corr_probs, -2, axis=1)
                margins = part_probs[:, -1] - part_probs[:, -2]
            else:
                margins = np.ones_like(confs)

            # Fast O(1) Local Compact LUT Remapping
            lut_pred = np.zeros(len(u_all), dtype=np.int32)
            lut_conf = np.zeros(len(u_all), dtype=np.float32)
            lut_ent = np.zeros(len(u_all), dtype=np.float32)
            lut_mar = np.zeros(len(u_all), dtype=np.float32)

            lut_pred[offset:] = preds
            lut_conf[offset:] = confs
            lut_ent[offset:] = entropies
            lut_mar[offset:] = margins

            pred_arr = lut_pred[inv_arr]
            prob_arr = lut_conf[inv_arr]
            ent_arr = lut_ent[inv_arr]
            mar_arr = lut_mar[inv_arr]

            pred_arr[foot_arr == 0] = 0
            prob_arr[foot_arr == 0] = 0
            ent_arr[foot_arr == 0] = 0
            mar_arr[foot_arr == 0] = 0

            ds_cls.GetRasterBand(1).WriteArray(pred_arr, x, y)
            ds_conf.GetRasterBand(1).WriteArray(prob_arr, x, y)
            ds_ent.GetRasterBand(1).WriteArray(ent_arr, x, y)
            ds_mar.GetRasterBand(1).WriteArray(mar_arr, x, y)

            total_segments_classified += len(u_sids)

            elapsed = time.time() - t_infer_start
            rate = tile_cnt / elapsed if elapsed > 0 else 0
            eta_sec = (total_tiles - tile_cnt) / rate if rate > 0 else 0
            eta_str = f"{int(eta_sec//60)}m {int(eta_sec%60):02d}s"
            sys.stdout.write(
                f"\r    [INFERENCE] Tile {tile_cnt}/{total_tiles} ({(tile_cnt/total_tiles*100):.1f}%) | "
                f"Objects: {total_segments_classified:,} | Time: {int(elapsed//60)}m {int(elapsed%60):02d}s | ETA: {eta_str}  "
            )
            sys.stdout.flush()

            batch_queue.task_done()

        producer_thread.join()
        if producer_error[0] is not None:
            raise producer_error[0]

        ds_cls.FlushCache()
        ds_conf.FlushCache()
        ds_ent.FlushCache()
        ds_mar.FlushCache()
        ds_cls = None
        ds_conf = None
        ds_ent = None
        ds_mar = None
        total_time_min = (time.time() - t_infer_start) / 60.0
        print(f"\n    [INFERENCE COMPLETE] Successfully classified {total_segments_classified:,} objects across {total_tiles} tiles in {total_time_min:.1f} minutes.")
        print(f"    Raw classification saved: {self.class_tif}")
        print(f"    Raw entropy saved:        {self.entropy_tif}")
        print(f"    Raw margin saved:         {self.margin_tif}\n")
    # --- Stage 7: Apply Agricultural & Footprint Masks ---
    def stage_7_mask_classification(self, force_recompute=False):
        stage = 7
        if self.masked_class.exists() and self.masked_conf.exists() and self.masked_entropy.exists() and not force_recompute:
            print(f"[Stage {stage}/{self.total_stages}] Masked outputs already exist, skipping.")
            return

        print(f"[Stage {stage}/{self.total_stages}] Applying Agricultural & Footprint Masks to Classification, Confidence, Entropy, Margin...")
        ref_ras = self.class_tif if self.class_tif.exists() else (self.s1_ras if self.s1_ras else self.s2_ras)
        if not ref_ras or not ref_ras.exists():
            print("ERROR: Reference raster or classification output not found.")
            return

        if not self.class_tif.exists() or not self.conf_tif.exists():
            print("ERROR: Classification outputs not found. Run Stage 6 first.")
            return

        ds_ref = gdal.Open(str(ref_ras))
        cols = ds_ref.RasterXSize
        rows = ds_ref.RasterYSize
        gt = ds_ref.GetGeoTransform()
        proj = ds_ref.GetProjection()
        ds_ref = None

        mask_tif = self.agri_mask
        ds_mask = None
        temp_mask_vrt = None
        if mask_tif and mask_tif.exists():
            print(f"    Warping country agricultural mask to match classification raster bounds...")
            minx = gt[0]
            maxy = gt[3]
            maxx = minx + gt[1] * cols
            miny = maxy + gt[5] * rows

            temp_mask_vrt = str(self.masked_class).replace('.tif', '_mask_temp.vrt')
            mask_opts = gdal.WarpOptions(
                format='VRT',
                outputBounds=(minx, miny, maxx, maxy),
                width=cols,
                height=rows,
                dstSRS=proj,
                resampleAlg=gdal.GRA_NearestNeighbour
            )
            ds_mask = gdal.Warp(temp_mask_vrt, str(mask_tif), options=mask_opts)
            if not ds_mask:
                print(f"    [WARNING] Failed to warp agricultural mask ({mask_tif}). Continuing without it.")
            else:
                print(f"    Applying country agricultural mask: {mask_tif.name}")
        else:
            print("    [INFO] Arable mask not found or not configured. Masking with data footprint only.")

        ds_foot = gdal.Open(str(self.footprint_mask)) if (self.footprint_mask and self.footprint_mask.exists()) else None
        ds_cls = gdal.Open(str(self.class_tif))
        ds_conf = gdal.Open(str(self.conf_tif))
        ds_ent = gdal.Open(str(self.entropy_tif)) if self.entropy_tif.exists() else None
        ds_mar = gdal.Open(str(self.margin_tif)) if self.margin_tif.exists() else None

        driver = gdal.GetDriverByName('GTiff')
        out_cls = driver.Create(str(self.masked_class), cols, rows, 1, gdal.GDT_Int32,
                                options=get_gdal_creation_options(predictor=2))
        out_cls.SetGeoTransform(gt)
        out_cls.SetProjection(proj)
        out_cls.GetRasterBand(1).SetNoDataValue(0)

        out_conf = driver.Create(str(self.masked_conf), cols, rows, 1, gdal.GDT_Float32,
                                 options=get_gdal_creation_options(predictor=3))
        out_conf.SetGeoTransform(gt)
        out_conf.SetProjection(proj)
        out_conf.GetRasterBand(1).SetNoDataValue(0)

        out_ent = driver.Create(str(self.masked_entropy), cols, rows, 1, gdal.GDT_Float32,
                                options=get_gdal_creation_options(predictor=3))
        out_ent.SetGeoTransform(gt)
        out_ent.SetProjection(proj)
        out_ent.GetRasterBand(1).SetNoDataValue(0)

        out_mar = driver.Create(str(self.masked_margin), cols, rows, 1, gdal.GDT_Float32,
                                options=get_gdal_creation_options(predictor=3))
        out_mar.SetGeoTransform(gt)
        out_mar.SetProjection(proj)
        out_mar.GetRasterBand(1).SetNoDataValue(0)

        tile_size = 4096
        total_tiles = math.ceil(cols / tile_size) * math.ceil(rows / tile_size)
        tile_cnt = 0

        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                xsize = min(tile_size, cols - x)
                ysize = min(tile_size, rows - y)

                cls_arr = ds_cls.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                conf_arr = ds_conf.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                ent_arr = ds_ent.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize) if ds_ent else np.zeros((ysize, xsize), dtype=np.float32)
                mar_arr = ds_mar.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize) if ds_mar else np.zeros((ysize, xsize), dtype=np.float32)

                combined_mask = np.ones((ysize, xsize), dtype=bool)
                if ds_foot:
                    foot_arr = ds_foot.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                    combined_mask = combined_mask & (foot_arr > 0)

                if ds_mask:
                    mask_arr = ds_mask.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                    combined_mask = combined_mask & (mask_arr > 0)

                cls_arr[~combined_mask] = 0
                conf_arr[~combined_mask] = 0.0
                ent_arr[~combined_mask] = 0.0
                mar_arr[~combined_mask] = 0.0

                out_cls.GetRasterBand(1).WriteArray(cls_arr, x, y)
                out_conf.GetRasterBand(1).WriteArray(conf_arr, x, y)
                out_ent.GetRasterBand(1).WriteArray(ent_arr, x, y)
                out_mar.GetRasterBand(1).WriteArray(mar_arr, x, y)

                tile_cnt += 1
                if tile_cnt % 20 == 0 or tile_cnt == total_tiles:
                    pct = (tile_cnt / total_tiles) * 100.0
                    print(f"    [MASKING PROGRESS] {tile_cnt}/{total_tiles} tiles ({pct:.1f}%)", flush=True)

        out_cls.GetRasterBand(1).FlushCache()
        out_conf.GetRasterBand(1).FlushCache()
        out_ent.GetRasterBand(1).FlushCache()
        out_mar.GetRasterBand(1).FlushCache()

        # Build multi-scale pyramids
        print("    Building multi-scale pyramid overviews [2, 4, 8, 16, 32, 64]...")
        try:
            out_cls.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64])
            out_conf.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32, 64])
            out_ent.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32, 64])
            out_mar.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32, 64])
        except Exception:
            pass

        out_cls = None
        out_conf = None
        out_ent = None
        out_mar = None
        ds_cls = None
        ds_conf = None
        ds_ent = None
        ds_mar = None
        ds_foot = None
        ds_mask = None
        print(f"    [MASKING COMPLETE] Masked products created:")
        print(f"      - {self.masked_class}")
        print(f"      - {self.masked_conf}")
        print(f"      - {self.masked_entropy}")
        print(f"      - {self.masked_margin}\n")
    def stage_8_calculate_metrics(self):
        stage = 8
        print(f"[Stage {stage}/{self.total_stages}] Calculating Out-of-Bag Validation Metrics and Generating Excel Report...")
        eval_shp = self.control_shp if (self.control_shp and self.control_shp.exists()) else self.learn_shp
        if not eval_shp or not eval_shp.exists():
            print(f"    [WARNING] No validation shapefile found ({eval_shp}). Skipping point evaluation.")
            return

        target_tif = self.masked_class if self.masked_class.exists() else self.class_tif
        if not target_tif or not target_tif.exists():
            print(f"    [ERROR] No classified raster found ({target_tif}). Run Stage 5 & 6 first.")
            return

        gdf_val = gpd.read_file(str(eval_shp), engine="pyogrio")
        crop_col = 'crop_id' if 'crop_id' in gdf_val.columns else 'code'
        gdf_val['crop_id'] = gdf_val[crop_col].astype(int)

        ds = gdal.Open(str(target_tif))
        gt = ds.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)
        cols, rows = ds.RasterXSize, ds.RasterYSize
        cls_arr = ds.GetRasterBand(1).ReadAsArray()
        ras_proj = ds.GetProjection()

        if ras_proj and gdf_val.crs:
            from pyproj import CRS
            target_crs = CRS.from_wkt(ras_proj)
            if gdf_val.crs != target_crs:
                gdf_val_proj = gdf_val.to_crs(target_crs)
            else:
                gdf_val_proj = gdf_val
        else:
            gdf_val_proj = gdf_val

        if hasattr(gdf_val_proj.geometry, 'x'):
            xs = gdf_val_proj.geometry.x.values
            ys = gdf_val_proj.geometry.y.values
        else:
            xs = gdf_val_proj.geometry.centroid.x.values
            ys = gdf_val_proj.geometry.centroid.y.values

        pxs = (inv_gt[0] + inv_gt[1] * xs + inv_gt[2] * ys).astype(int)
        pys = (inv_gt[3] + inv_gt[4] * xs + inv_gt[5] * ys).astype(int)

        y_true = []
        y_pred = []
        for cid, px, py in zip(gdf_val['crop_id'].values, pxs, pys):
            if 0 <= px < cols and 0 <= py < rows:
                pred = cls_arr[py, px]
                if pred > 0:
                    y_true.append(cid)
                    y_pred.append(pred)

        if not y_true:
            print("    [WARNING] No validation points fell on valid classified pixels.")
            return

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        all_classes = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        oa = float(np.trace(cm) / np.sum(cm))

        # Cohen's Kappa
        p_o = oa
        p_e = float(np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (np.sum(cm) ** 2))
        kappa = float((p_o - p_e) / (1.0 - p_e + 1e-9))

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=all_classes, zero_division=0)

        # Resolve English crop names mapping
        crop_name_map = {}
        id_cols = [c for c in ['crop_id', 'crop_ids', 'code', 'id', 'class_id'] if c in gdf_val.columns]
        name_cols = [c for c in ['crop_name', 'crop_names', 'crop_type', 'label', 'name', 'class_name', 'crop', 'nom'] if c in gdf_val.columns]
        if id_cols and name_cols:
            for _, row in gdf_val[[id_cols[0], name_cols[0]]].drop_duplicates().iterrows():
                try:
                    cid = int(row[id_cols[0]])
                    cname = str(row[name_cols[0]]).strip()
                    if cname and cname.lower() not in ['none', 'nan', '']:
                        crop_name_map[cid] = cname
                except Exception:
                    pass

        # Fallback to master sample_shp if any crop name missing
        if self.sample_shp and self.sample_shp.exists():
            try:
                gdf_samp = gpd.read_file(str(self.sample_shp), engine="pyogrio")
                id_cols_s = [c for c in ['crop_id', 'crop_ids', 'code', 'id', 'class_id'] if c in gdf_samp.columns]
                name_cols_s = [c for c in ['crop_name', 'crop_names', 'crop_type', 'label', 'name', 'class_name', 'crop', 'nom'] if c in gdf_samp.columns]
                if id_cols_s and name_cols_s:
                    for _, row in gdf_samp[[id_cols_s[0], name_cols_s[0]]].drop_duplicates().iterrows():
                        try:
                            cid = int(row[id_cols_s[0]])
                            cname = str(row[name_cols_s[0]]).strip()
                            if cid not in crop_name_map and cname and cname.lower() not in ['none', 'nan', '']:
                                crop_name_map[cid] = cname
                        except Exception:
                            pass
            except Exception:
                pass

        # Fallback to country priors.json if available
        country_priors_file = self.aux_dir / "shapefiles_samples" / self.country / "priors.json"
        if country_priors_file.exists():
            try:
                with open(country_priors_file, 'r', encoding='utf-8') as pf:
                    priors_data = json.load(pf)
                    for p_idx, p_name in enumerate(priors_data.keys(), start=1):
                        if p_idx not in crop_name_map:
                            crop_name_map[p_idx] = p_name.title()
            except Exception:
                pass

        # Excel Export
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Validation Metrics"

        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        sub_font = Font(name="Calibri", size=11, bold=True, color="1F497D")
        bold_font = Font(name="Calibri", size=11, bold=True)
        regular_font = Font(name="Calibri", size=11)
        align_left = Alignment(horizontal='left', vertical='center')
        align_center = Alignment(horizontal='center', vertical='center')
        thin_border = Border(
            left=Side(style='thin', color='D9D9D9'), right=Side(style='thin', color='D9D9D9'),
            top=Side(style='thin', color='D9D9D9'), bottom=Side(style='thin', color='D9D9D9')
        )

        ws.cell(row=1, column=1, value=f"Crop Classification Accuracy Report: {self.track}").font = Font(name="Calibri", size=14, bold=True, color="1F497D")
        ws.cell(row=2, column=1, value=f"Track: {self.track} ({self.country}) | Segmentation: {self.seg_mode.upper()} | Model: Unified PyTorch MLP + XGBoost Fusion Ensemble").font = bold_font
        ws.cell(row=3, column=1, value="Data: Multimodal Sentinel-1 SAR (Sigma0 VH/VV) + Sentinel-2 MSI (B02-B12) + NASA Harvest Presto Embeddings").font = Font(name="Calibri", size=10, italic=True, color="595959")

        # Summary table
        ws.cell(row=5, column=1, value="Metric").fill = header_fill
        ws.cell(row=5, column=1).font = header_font
        ws.cell(row=5, column=1).alignment = align_left
        ws.cell(row=5, column=2, value="Value").fill = header_fill
        ws.cell(row=5, column=2).font = header_font
        ws.cell(row=5, column=2).alignment = align_left

        ws.cell(row=6, column=1, value="Overall Accuracy (OA)").font = regular_font
        ws.cell(row=6, column=1).border = thin_border
        ws.cell(row=6, column=2, value=f"{oa * 100:.1f}%").font = bold_font
        ws.cell(row=6, column=2).alignment = align_left
        ws.cell(row=6, column=2).border = thin_border

        ws.cell(row=7, column=1, value="Cohen's Kappa").font = regular_font
        ws.cell(row=7, column=1).border = thin_border
        ws.cell(row=7, column=2, value=f"{kappa:.4f}").font = bold_font
        ws.cell(row=7, column=2).alignment = align_left
        ws.cell(row=7, column=2).border = thin_border

        ws.cell(row=8, column=1, value="Validation Samples Count").font = regular_font
        ws.cell(row=8, column=1).border = thin_border
        ws.cell(row=8, column=2, value=f"{len(y_true):,}".replace(',', ' ')).font = regular_font
        ws.cell(row=8, column=2).alignment = align_left
        ws.cell(row=8, column=2).border = thin_border

        # Per-class table
        r = 10
        ws.cell(row=r, column=1, value="Per-Class Classification Accuracy").font = sub_font
        r += 1
        headers_pc = ["Class ID", "Crop Name", "Precision (User Acc)", "Recall (Prod Acc)", "F1-Score", "Validation Samples"]
        for c_idx, h_text in enumerate(headers_pc, start=1):
            cell = ws.cell(row=r, column=c_idx, value=h_text)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_left

        for idx, cid in enumerate(all_classes):
            r += 1
            c_name = crop_name_map.get(int(cid), f"Class {cid}")
            sample_cnt = int(np.sum(y_true == cid))
            
            c1 = ws.cell(row=r, column=1, value=int(cid))
            c1.font = regular_font
            c1.alignment = align_left
            c1.border = thin_border
            
            c2 = ws.cell(row=r, column=2, value=c_name)
            c2.font = regular_font
            c2.alignment = align_left
            c2.border = thin_border
            
            c3 = ws.cell(row=r, column=3, value=f"{precision[idx] * 100:.1f}%")
            c3.font = regular_font
            c3.alignment = align_left
            c3.border = thin_border
            
            c4 = ws.cell(row=r, column=4, value=f"{recall[idx] * 100:.1f}%")
            c4.font = regular_font
            c4.alignment = align_left
            c4.border = thin_border
            
            c5 = ws.cell(row=r, column=5, value=f"{f1[idx] * 100:.1f}%")
            c5.font = regular_font
            c5.alignment = align_left
            c5.border = thin_border
            
            c6 = ws.cell(row=r, column=6, value=f"{sample_cnt:,}".replace(',', ' '))
            c6.font = regular_font
            c6.alignment = align_left
            c6.border = thin_border

        # Confusion Matrix table
        r += 3
        ws.cell(row=r, column=1, value="Confusion Matrix (Rows: Ground Truth, Cols: Prediction)").font = sub_font
        r += 1
        c_top = ws.cell(row=r, column=1, value="True \\ Pred")
        c_top.fill = header_fill
        c_top.font = header_font
        c_top.alignment = align_left
        for c_idx, cid in enumerate(all_classes):
            c_name = crop_name_map.get(int(cid), str(cid))
            cell = ws.cell(row=r, column=c_idx + 2, value=f"{int(cid)}: {c_name}")
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = align_center

        for row_idx, true_cid in enumerate(all_classes):
            r += 1
            t_name = crop_name_map.get(int(true_cid), str(true_cid))
            c_row_hdr = ws.cell(row=r, column=1, value=f"{int(true_cid)}: {t_name}")
            c_row_hdr.font = bold_font
            c_row_hdr.alignment = align_left
            c_row_hdr.border = thin_border
            for col_idx, pred_cid in enumerate(all_classes):
                val = int(cm[row_idx, col_idx])
                cell = ws.cell(row=r, column=col_idx + 2, value=val)
                cell.font = regular_font
                cell.border = thin_border
                cell.alignment = align_center

        # Area estimation table
        unique_cls, counts = np.unique(cls_arr, return_counts=True)
        valid_cls_mask = unique_cls > 0
        if np.any(valid_cls_mask):
            unique_cls = unique_cls[valid_cls_mask]
            counts = counts[valid_cls_mask]
            
            resx = abs(gt[1])
            resy = abs(gt[5])
            pixel_ha = (resx * resy) / 10000.0
            total_ha = float(np.sum(counts) * pixel_ha)

            r += 3
            ws.cell(row=r, column=1, value="Classified Agricultural Area Statistics").font = sub_font
            r += 1
            headers_area = ["Class ID", "Crop Name", "Area (ha)", "Area (%)"]
            for c_idx, h_text in enumerate(headers_area, start=1):
                cell = ws.cell(row=r, column=c_idx, value=h_text)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = align_left

            for u_id, u_count in zip(unique_cls, counts):
                r += 1
                c_name = crop_name_map.get(int(u_id), f"Class {u_id}")
                c_ha = int(round(float(u_count * pixel_ha)))
                c_pct = (float(u_count * pixel_ha) / total_ha * 100.0) if total_ha > 0 else 0.0
                
                c1 = ws.cell(row=r, column=1, value=int(u_id))
                c1.font = regular_font
                c1.alignment = align_left
                c1.border = thin_border
                
                c2 = ws.cell(row=r, column=2, value=c_name)
                c2.font = regular_font
                c2.alignment = align_left
                c2.border = thin_border
                
                c3 = ws.cell(row=r, column=3, value=f"{c_ha:,}".replace(',', ' '))
                c3.font = regular_font
                c3.alignment = align_left
                c3.border = thin_border
                
                c4 = ws.cell(row=r, column=4, value=f"{c_pct:.1f}%")
                c4.font = regular_font
                c4.alignment = align_left
                c4.border = thin_border

            # Total row
            r += 1
            tot_ha_int = int(round(total_ha))
            c1 = ws.cell(row=r, column=1, value="Total")
            c1.font = bold_font
            c1.alignment = align_left
            c1.border = thin_border
            
            c2 = ws.cell(row=r, column=2, value="All Agricultural Crops")
            c2.font = bold_font
            c2.alignment = align_left
            c2.border = thin_border
            
            c3 = ws.cell(row=r, column=3, value=f"{tot_ha_int:,}".replace(',', ' '))
            c3.font = bold_font
            c3.alignment = align_left
            c3.border = thin_border
            
            c4 = ws.cell(row=r, column=4, value="100.0%")
            c4.font = bold_font
            c4.alignment = align_left
            c4.border = thin_border

        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            ws.column_dimensions[col_letter].width = max(max_len + 4, 16)

        wb.save(str(self.metrics_fp))
        print(f"    [OK] Metrics report saved to: {self.metrics_fp}")
        print(f"    Validation Overall Accuracy (OA): {oa * 100:.1f}% | Kappa: {kappa:.4f}\n")

    def run_all(self):
        """Executes all classification stages (1 through 8) sequentially."""
        force = getattr(self, 'overwrite', False)
        self.stage_1_generate_footprint(False)
        self.stage_2_segmentation(False)
        self.stage_3_split_samples(False)
        self.stage_4_selection(force)
        self.stage_5_train_classifier(force)
        self.stage_6_classify_vector(True)
        self.stage_7_mask_classification(True)
        self.stage_8_calculate_metrics()

    # Backward compatibility aliases for legacy stage indexing (0-7 vs 1-8)
    stage_0_generate_footprint = stage_1_generate_footprint
    stage_1_segmentation = stage_2_segmentation
    stage_2_split_samples = stage_3_split_samples
    stage_3_selection = stage_4_selection
    stage_4_train_classifier = stage_5_train_classifier
    stage_5_classify_vector = stage_6_classify_vector
    stage_6_mask_classification = stage_7_mask_classification
    stage_7_calculate_metrics = stage_8_calculate_metrics


# =====================================================================
# 5. CLI & INTERACTIVE MENU
# =====================================================================

def main_menu(pipeline):
    while True:
        menu = f"""
    --- Multimodal Crop Classification Pipeline (Unified MLP + XGBoost Fusion) ---
    Track: {pipeline.track} ({pipeline.country})
    Model: Unified PyTorch MLP + XGBoost Fusion Ensemble (weight: {pipeline.mlp_weight:.2f})
    Segmentation: {pipeline.seg_mode.upper()}

    [1] Stage 1: Generate Data Footprint (S1 + S2)
    [2] Stage 2: Multimodal Segmentation ({pipeline.seg_mode.upper()})
    [3] Stage 3: Prepare Point Split (70/30)
    [4] Stage 4: Extract Multimodal Features (S1+S2+Presto)
    [5] Stage 5: Train Unified MLP + XGBoost Fusion Ensemble
    [6] Stage 6: Run Object-Based Inference with Bayesian Priors
    [7] Stage 7: Apply Agricultural & Footprint Mask
    [8] Stage 8: Calculate Validation Metrics (.xlsx)

    [A] Run All Stages (1 -> 8)
    [Q] Quit

    Enter choice: """
        try:
            choice = input(menu).strip().upper()
            if choice in ['1', '0']: pipeline.stage_1_generate_footprint(True)
            elif choice == '2': pipeline.stage_2_segmentation(True)
            elif choice == '3': pipeline.stage_3_split_samples(True)
            elif choice == '4': pipeline.stage_4_selection(True)
            elif choice == '5': pipeline.stage_5_train_classifier(True)
            elif choice == '6': pipeline.stage_6_classify_vector(True)
            elif choice == '7': pipeline.stage_7_mask_classification(True)
            elif choice == '8': pipeline.stage_8_calculate_metrics()
            elif choice == 'A':
                pipeline.run_all()
            elif choice == 'Q': break
        except (KeyboardInterrupt, EOFError):
            break


def main():
    parser = argparse.ArgumentParser(description="Multimodal S1 (Sigma0) + S2 Crop Classification with Unified MLP + XGBoost Fusion Ensemble.")
    parser.add_argument('-t', '--track', default=None, help="Track/orbit identifier, e.g. NL/orbit_88, PT/orbit_147, PL/orbit_12")
    parser.add_argument('-c', '--country', default=None, help="Country code, e.g. PT, NL, PL (processes all orbits sequentially)")
    parser.add_argument('--stage', default=None, help="Stage to run: 'A' (all 1-8), or single stage '1'..'8' (legacy '0'..'7' supported)")
    parser.add_argument('--seg_mode', default='slic', choices=['sam', 'slic', 'lpis'], help="Segmentation mode (default: slic)")
    parser.add_argument('--mlp_weight', type=float, default=0.65, help="Weight of MLP in fusion ensemble (0.0 to 1.0, default: 0.65)")
    parser.add_argument('--s1_raster', default=None, help="Override path to Sentinel-1 Sigma0 VH/VV GeoTIFF raster")
    parser.add_argument('--s2_raster', default=None, help="Override path to Sentinel-2 Multi-temporal GeoTIFF raster")
    parser.add_argument('--lpis_vector', default=None, help="Path to official LPIS cadastral parcel vector file (.shp, .gpkg) for --seg_mode lpis")
    parser.add_argument('--slic_segment_ha', type=float, default=None, help="Target superpixel parcel area in hectares for SLIC (default: adaptive, 2.5 ha for PT/ES/PL, 3.5 ha for NL/FR/DE)")
    parser.add_argument('--slic_compactness', type=float, default=3.0, help="SLIC superpixel boundary compactness (default: 3.0)")
    parser.add_argument('--slic_rag_thresh', type=float, default=0.10, help="Region Adjacency Graph (RAG) spectral fusion distance threshold for SLIC (default: 0.10)")
    parser.add_argument('--no_slic_rag', action='store_true', help="Disable Region Adjacency Graph (RAG) spectral fusion pass for SLIC")

    args = parser.parse_args()

    def _exec_pipeline(tr):
        pipeline = ProcessingPipelineS1S2(
            track=tr,
            seg_mode=args.seg_mode,
            mlp_weight=args.mlp_weight,
            s1_override=args.s1_raster,
            s2_override=args.s2_raster,
            lpis_vector=args.lpis_vector,
            slic_segment_ha=args.slic_segment_ha,
            slic_compactness=args.slic_compactness,
            slic_rag_thresh=args.slic_rag_thresh,
            enable_slic_rag=not args.no_slic_rag
        )

        if args.stage is None:
            main_menu(pipeline)
        else:
            choice = args.stage.strip().upper()
            if choice in ['A', 'ALL']:
                pipeline.run_all()
            elif choice in ['1', '0']: pipeline.stage_1_generate_footprint(True)
            elif choice == '2': pipeline.stage_2_segmentation(True)
            elif choice == '3': pipeline.stage_3_split_samples(True)
            elif choice == '4': pipeline.stage_4_selection(True)
            elif choice == '5': pipeline.stage_5_train_classifier(True)
            elif choice == '6': pipeline.stage_6_classify_vector(True)
            elif choice == '7': pipeline.stage_7_mask_classification(True)
            elif choice == '8': pipeline.stage_8_calculate_metrics()

    target_tracks = []
    country_candidate = args.country.upper() if args.country else None
    if not country_candidate and args.track:
        norm_t = args.track.replace('\\', '/')
        if '/' not in norm_t:
            country_candidate = norm_t.upper()

    if country_candidate:
        c_dir = base_dir / country_candidate
        if c_dir.exists():
            orbs = sorted([d.name for d in c_dir.glob("orbit_*") if d.is_dir()])
            if orbs:
                target_tracks = [f"{country_candidate}/{o}" for o in orbs]

    if not target_tracks:
        if args.track:
            target_tracks = [args.track.replace('\\', '/')]
        else:
            parser.error("Either --track (-t) or --country (-c) must be specified.")

    for tr in target_tracks:
        _exec_pipeline(tr)


if __name__ == '__main__':
    main()
