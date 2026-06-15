# Jak uruchomić skrypt:
# python 1_classify_prithvi_sar.py --track NL/orbit_88
# python 1_classify_prithvi_sar.py --track PT/orbit_161

import os
import argparse
from pathlib import Path
import subprocess
import sys
import shutil
import shlex
import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr, gdalconst
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
import openpyxl
from openpyxl.styles import Font
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import math

# Crop aggregation mapping for the Netherlands (NL) to reduce semantic confusion
CROP_AGGREGATION_NL = {
    2: 5,   # Clover -> Grassland
    7: 5,   # Lucerne -> Grassland
}

def get_crop_aggregation(country, learn_shp_path):
    aggregation = {}
    if country == 'NL' and learn_shp_path and os.path.exists(learn_shp_path):
        try:
            import geopandas as gpd
            gdf = gpd.read_file(str(learn_shp_path), engine="pyogrio")
            if 'crop_id' in gdf.columns and 'crop_name' in gdf.columns:
                id_to_name = dict(zip(gdf['crop_id'].astype(int), gdf['crop_name'].astype(str).str.lower()))
                
                # Find ID of grassland
                grassland_id = None
                for cid, name in id_to_name.items():
                    if "grassland" in name or "grass" in name:
                        grassland_id = cid
                        break
                
                if grassland_id is not None:
                    for cid, name in id_to_name.items():
                        if any(k in name for k in ["clover", "klaver", "lucerne", "luzerne"]):
                            aggregation[cid] = grassland_id
        except Exception as e:
            print(f"    [WARNING] Failed to dynamically construct NL crop aggregation: {e}")
            
    if not aggregation and country == 'NL':
        aggregation = CROP_AGGREGATION_NL
        
    return aggregation

def _get_priors_for_country(country, learn_shp_path, classes, class_counts, total_samples, priors_file_override=None):
    # Try custom JSON override first
    priors_json_path = None
    if priors_file_override and os.path.exists(priors_file_override):
        priors_json_path = Path(priors_file_override)
    else:
        # Resolve project root relative to learn_shp_path
        if learn_shp_path:
            p = Path(learn_shp_path).resolve()
            for parent in p.parents:
                aux_dir = parent / 'auxiliary_files'
                if aux_dir.exists():
                    aux_priors = aux_dir / 'shapefiles_samples' / country / 'priors.json'
                    if aux_priors.exists():
                        priors_json_path = aux_priors
                        break
            
            # Fallback to track_dir/priors.json
            if not priors_json_path:
                track_dir = Path(learn_shp_path).parent.parent
                track_priors = track_dir / f"priors_{country}.json"
                if not track_priors.exists():
                    track_priors = track_dir / "priors.json"
                if track_priors.exists():
                    priors_json_path = track_priors

    if priors_json_path and priors_json_path.exists():
        try:
            import json
            with open(priors_json_path, 'r') as f:
                custom_priors = json.load(f)
            print(f"    Loaded name-based priors from {priors_json_path}")
            
            # Read crop names from shapefile
            id_to_name = {}
            if learn_shp_path and os.path.exists(learn_shp_path):
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(str(learn_shp_path), engine="pyogrio")
                    if 'crop_id' in gdf.columns and 'crop_name' in gdf.columns:
                        id_to_name = dict(zip(gdf['crop_id'].astype(int), gdf['crop_name'].astype(str).str.lower()))
                except Exception as e:
                    print(f"    [WARNING] Could not read crop names from shapefile: {e}")
            
            # Build priors map dynamically
            raw_priors = {}
            sorted_keys = sorted(custom_priors.keys(), key=len, reverse=True)
            for cid, name in id_to_name.items():
                matched_val = 1e-5
                for key in sorted_keys:
                    if key.lower() in name or name in key.lower():
                        matched_val = float(custom_priors[key])
                        break
                raw_priors[cid] = matched_val
                
            for key, val in custom_priors.items():
                try:
                    cid = int(key)
                    raw_priors[cid] = float(val)
                except ValueError:
                    pass
            
            # Apply aggregation
            crop_aggregation = get_crop_aggregation(country, learn_shp_path)
            aggregated_priors = {}
            for cid, val in raw_priors.items():
                mapped_cid = crop_aggregation.get(cid, cid)
                aggregated_priors[mapped_cid] = aggregated_priors.get(mapped_cid, 0.0) + val
                
            p_true = np.array([aggregated_priors.get(c, 1e-5) for c in classes])
            p_true = p_true / np.sum(p_true)
            return p_true
        except Exception as e:
            print(f"    [WARNING] Failed to load name-based priors: {e}")

    # NL specific priors fallback
    if country == 'NL':
        real_priors_nl = {
            1: 0.0030, 2: 0.0015, 3: 0.0775, 4: 0.0057, 5: 0.7214,
            6: 0.0033, 7: 0.0056, 8: 0.0847, 9: 0.0060, 10: 0.0007,
            11: 0.0073, 12: 0.0087, 13: 0.0255, 14: 0.0023, 15: 0.0149,
            16: 0.0058, 17: 0.0044, 18: 0.0006, 19: 0.0034, 20: 0.0178
        }
        crop_aggregation = get_crop_aggregation(country, learn_shp_path)
        aggregated_priors = {}
        for cid, val in real_priors_nl.items():
            mapped_cid = crop_aggregation.get(cid, cid)
            aggregated_priors[mapped_cid] = aggregated_priors.get(mapped_cid, 0.0) + val
        p_true = np.array([aggregated_priors.get(c, 1e-5) for c in classes])
        p_true = p_true / np.sum(p_true)
        return p_true


    # PL & any other country - dynamic estimation using keyword-based field sizes
    id_to_name = {}
    if learn_shp_path and os.path.exists(learn_shp_path):
        try:
            import geopandas as gpd
            gdf = gpd.read_file(str(learn_shp_path), engine="pyogrio")
            if 'crop_id' in gdf.columns and 'crop_name' in gdf.columns:
                id_to_name = dict(zip(gdf['crop_id'].astype(int), gdf['crop_name'].astype(str)))
        except Exception as e:
            print(f"    [WARNING] Could not read crop names from shapefile: {e}")

    # Area threshold keyword matcher
    def get_area_multiplier(cid):
        name = id_to_name.get(cid, '').lower()
        if not name:
            # Fallback to standard 37 classes PL hardcoded values if crop_id maps to them (1 to 37)
            area_thresholds_pl = {
                1: 3000, 2: 5000, 3: 5000, 4: 40000, 5: 5000, 6: 10000, 7: 5000, 8: 30000, 9: 15000,
                10: 20000, 11: 30000, 12: 40000, 13: 5000, 14: 70000, 15: 2000, 16: 30000, 17: 5000,
                18: 3000, 19: 40000, 20: 2000, 21: 40000, 22: 2000, 23: 10000, 24: 25000, 25: 70000,
                26: 10000, 27: 50000, 28: 2000, 29: 60000, 30: 3000, 31: 15000, 32: 70000, 33: 5000,
                34: 3000, 35: 5000, 36: 20000, 37: 40000
            }
            return area_thresholds_pl.get(cid, 10000)
            
        # Large field crops (~40k - 100k m2)
        if any(k in name for k in ['grass', 'tiuz', 'grassland', 'pasture', 'trawa', 'trawiast', 'blijvend', 'tijdelijk', 'permanent', 'clover', 'klaver', 'lucerne', 'luzerne']):
            return 70000
        if any(k in name for k in ['maize', 'mais', 'kukurydza', 'corn']):
            return 70000
        if any(k in name for k in ['wheat', 'pszenica', 'tarwe']):
            return 70000
        if any(k in name for k in ['barley', 'jeczmien', 'gerst']):
            return 40000
        if any(k in name for k in ['rye', 'zyto', 'rogge']):
            return 40000
        if any(k in name for k in ['triticale', 'pszenzyto', 'koorn']):
            return 50000
        if any(k in name for k in ['oats', 'owies', 'haver']):
            return 40000
        if any(k in name for k in ['rapeseed', 'rzepak', 'koolzaad']):
            return 60000
        if any(k in name for k in ['sugar beet', 'burak', 'suikerbiet']):
            return 40000
        if any(k in name for k in ['fallow', 'braak', 'ugor']):
            return 40000
            
        # Medium/small (~10k - 30k m2)
        if any(k in name for k in ['potato', 'ziemniak', 'aardappel']):
            return 20000
        if any(k in name for k in ['orchard', 'fruit', 'sad', 'appel', 'peer', 'jablon', 'sliwa', 'wisnia']):
            return 20000
        if any(k in name for k in ['pea', 'groch', 'erwt']):
            return 30000
        if any(k in name for k in ['bean', 'fasola', 'boon']):
            return 10000
        if any(k in name for k in ['aronia', 'blueberry', 'borowka', 'currant', 'porzeczka', 'bessen']):
            return 10000
        if any(k in name for k in ['nursery', 'ornamental', 'szkolka', 'sier', 'boomkwekerij']):
            return 10000
            
        # Small / vegetable crops (<10k m2)
        if any(k in name for k in ['onion', 'cebula', 'ui']):
            return 5000
        if any(k in name for k in ['strawberry', 'truskawka', 'aardbei']):
            return 5000
        if any(k in name for k in ['cabbage', 'brassica', 'kapusta', 'kool']):
            return 5000
        if any(k in name for k in ['carrot', 'marchew', 'peen']):
            return 3000
        if any(k in name for k in ['tomato', 'pomidor', 'tomaat']):
            return 2000
        if any(k in name for k in ['cucumber', 'ogorek', 'komkommer']):
            return 2000
        if any(k in name for k in ['tobacco', 'tyton', 'tabak']):
            return 3000
        if any(k in name for k in ['raspberry', 'blackberry', 'malina', 'jezyna', 'framboos']):
            return 5000
        return 10000

    area_multipliers = np.array([get_area_multiplier(c) for c in classes])
    counts_arr = np.array([class_counts.get(c, 0) for c in classes])
    true_area_dist = counts_arr * area_multipliers
    p_true = true_area_dist / (np.sum(true_area_dist) + 1e-9)
    return p_true


# Try importing SAM
try:
    from samgeo import SamGeo
    from skimage.util import img_as_float
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

# Try importing PyTorch & Transformers for Prithvi-SAR
try:
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch or huggingface_hub not found. Please install them to use Prithvi-SAR.")

# Global Paths
base_dir = Path("D:/AIML_CropMapper_Cloud/workingDir")
aux_dir = Path("D:/AIML_CropMapper_Cloud/auxiliary_files")
prithvi_dir = aux_dir / "Prithvi_models"

TOTAL_STAGES = 8


def resolve_prithvi_model():
    """Ensures that prithvi_mae.py and Prithvi_100M.pt are available locally."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required but not installed.")
        
    prithvi_dir.mkdir(parents=True, exist_ok=True)
    mae_path = prithvi_dir / "prithvi_mae.py"
    weights_path = prithvi_dir / "Prithvi_100M.pt"
    
    repo_id = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
    
    if not mae_path.exists():
        print(f"[Prithvi Setup] Downloading prithvi_mae.py from HuggingFace ({repo_id})...")
        downloaded = hf_hub_download(repo_id, "prithvi_mae.py")
        shutil.copy(downloaded, mae_path)
        print(f"[Prithvi Setup] Saved to {mae_path}")
        
    if not weights_path.exists():
        print(f"[Prithvi Setup] Downloading Prithvi_100M.pt from HuggingFace ({repo_id})...")
        downloaded = hf_hub_download(repo_id, "Prithvi_100M.pt")
        shutil.copy(downloaded, weights_path)
        print(f"[Prithvi Setup] Saved to {weights_path}")
        
    # Append the directory to sys.path so we can import the architecture
    if str(prithvi_dir) not in sys.path:
        sys.path.insert(0, str(prithvi_dir))
        
    return mae_path, weights_path


def load_prithvi_encoder(weights_path):
    """Instantiates the PrithviViT encoder and loads the pre-trained weights."""
    from prithvi_mae import PrithviViT
    
    print("[Prithvi Loader] Instantiating PrithviViT model (100M config)...")
    # 100M configuration parameters from config.json
    model = PrithviViT(
        img_size=224,
        patch_size=(1, 16, 16),
        num_frames=3,
        in_chans=6,
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    print(f"[Prithvi Loader] Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Strip the 'encoder.' prefix to match PrithviViT's keys
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state_dict[k.replace("encoder.", "")] = v
            
    model.load_state_dict(encoder_state_dict, strict=True)
    model.eval()
    print("[Prithvi Loader] Model successfully loaded and ready for feature extraction.")
    return model


def prepare_segment_patch(raster_ds, segment_mask, bbox, target_size=(224, 224)):
    """
    Crops the segment bounding box, masks out non-segment pixels,
    and resamples to the target shape [C, T, H, W] for Prithvi.
    """
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    
    # Read raw multiband data in the segment bounding box
    nbands = raster_ds.RasterCount
    bands_data = []
    for b in range(1, nbands + 1):
        band = raster_ds.GetRasterBand(b)
        arr = band.ReadAsArray(int(x_min), int(y_min), int(w), int(h))
        arr = np.nan_to_num(arr)
        bands_data.append(arr)
    bands_arr = np.stack(bands_data, axis=0) # [C_raster, H_crop, W_crop]
    
    # Apply segment mask inside the crop (only keep pixels belonging to the segment)
    segment_mask_cropped = segment_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
    bands_arr = bands_arr * (segment_mask_cropped[None, :, :] > 0)
    
    # Prithvi-EO expects 6 channels, 3 frames, and 224x224 size.
    # Group bands into 3 temporal frames (early, mid, late season)
    num_dates = nbands // 2
    if num_dates < 1:
        num_dates = 1
        
    dates_grouped = []
    # Reorganize bands into dates
    for d in range(num_dates):
        # S1 stacked as VH, VV
        vh = bands_arr[d*2] if d*2 < nbands else bands_arr[0]
        vv = bands_arr[d*2+1] if d*2+1 < nbands else bands_arr[0]
        dates_grouped.append((vv, vh))
        
    # Replicate or average to get exactly 3 frames
    frames = []
    if len(dates_grouped) >= 3:
        # Divide into early, mid, late groups
        g1 = dates_grouped[:len(dates_grouped)//3]
        g2 = dates_grouped[len(dates_grouped)//3:2*len(dates_grouped)//3]
        g3 = dates_grouped[2*len(dates_grouped)//3:]
        
        for g in [g1, g2, g3]:
            vv_avg = np.mean([item[0] for item in g], axis=0)
            vh_avg = np.mean([item[1] for item in g], axis=0)
            frames.append((vv_avg, vh_avg))
    else:
        # Replicate available frames to get 3
        while len(dates_grouped) < 3:
            dates_grouped.append(dates_grouped[-1] if dates_grouped else (np.zeros((h, w)), np.zeros((h, w))))
        frames = dates_grouped[:3]
        
    # Replicate 2 SAR bands (VV, VH) into 6 bands expected by Prithvi: [VV, VH, VV, VH, VV, VH]
    # We normalize from decibels (dB) to [0.0, 1.0] range expected by Prithvi-EO (optical model)
    prithvi_tensor = np.zeros((6, 3, h, w), dtype=np.float32)
    for t in range(3):
        vv, vh = frames[t]
        
        # Min-max scale VV from [-25.0, 0.0] to [0.0, 1.0]
        vv_norm = np.clip((vv + 25.0) / 25.0, 0.0, 1.0)
        # Min-max scale VH from [-30.0, -5.0] to [0.0, 1.0]
        vh_norm = np.clip((vh + 30.0) / 25.0, 0.0, 1.0)
        
        prithvi_tensor[0, t] = vv_norm
        prithvi_tensor[1, t] = vh_norm
        prithvi_tensor[2, t] = vv_norm
        prithvi_tensor[3, t] = vh_norm
        prithvi_tensor[4, t] = vv_norm
        prithvi_tensor[5, t] = vh_norm
        
    # Resize the spatial dimensions to [224, 224] using bilinear interpolation
    import torch.nn.functional as F
    torch_tensor = torch.from_numpy(prithvi_tensor).unsqueeze(0) # [1, 6, 3, H, W]
    resized_tensor = F.interpolate(torch_tensor.view(1, 18, h, w), size=target_size, mode='bilinear', align_corners=False)
    resized_tensor = resized_tensor.view(1, 6, 3, target_size[0], target_size[1])
    
    return resized_tensor.squeeze(0) # [6, 3, 224, 224]


class ProcessingPipeline:
    def __init__(self, track):
        self.track = track
        normalized_track = track.replace('\\', '/')
        if '/' in normalized_track:
            self.country = normalized_track.split('/')[0].upper()
        elif len(track) == 2:
            self.country = track.upper()
        else:
            print(f"Error: Track '{track}' does not contain country code and is not a 2-letter country code.")
            sys.exit(1)

        self.total_stages = TOTAL_STAGES
        print(f"Initializing Prithvi-SAR pipeline for Track: {self.track}, Country: {self.country}")

        self.sanitized_track = self.track.replace('/', '_').replace('\\', '_')
        if self.sanitized_track.upper().startswith(self.country.upper() + "_"):
            self.file_prefix = self.sanitized_track
        else:
            self.file_prefix = f"{self.country}_{self.sanitized_track}"

        # --- 1. Define all paths ---
        self.base_dir = base_dir
        self.aux_dir = aux_dir
        self.proc_dir = self.base_dir / self.track / 'processed_raster'
        self.out_dir = self.base_dir / self.track / 'classification_results'
        self.samples_dir = self.out_dir / 'samples'
        self.model_dir = self.out_dir / 'train_model'
        self.seg_dir = self.out_dir / 'segmentation'
        self.class_dir = self.out_dir / 'classification'

        self._ensure_directories()

        # --- 2. Resolve input raster ---
        search_patterns = [
            f"{self.sanitized_track}_*_VH_VV*.tif",
            f"*_{self.sanitized_track}_*_VH_VV*.tif",
            f"*{self.sanitized_track}*.tif",
            f"{self.track}_*_VH_VV*.tif",
            f"*{self.track}*.tif",
        ]

        self.hdr = None
        if self.proc_dir.exists():
            for pattern in search_patterns:
                self.hdr = next(self.proc_dir.glob(pattern), None)
                if self.hdr:
                    break

            if not self.hdr:
                raise FileNotFoundError(f"No raster file (TIF) found for track {self.track} in {self.proc_dir}")

            self.ras = self.hdr
            print(f"Input raster found: {self.ras}")
        else:
            raise FileNotFoundError(f"Processing directory does not exist: {self.proc_dir}")

        # --- 3. Define all output file paths ---
        self.seg_tif = self.seg_dir / f"{self.file_prefix}_segmentation.tif"
        self.seg_shp = self.seg_dir / f"{self.file_prefix}_segmentation.sqlite"

        # Samples
        samples_base = self.aux_dir / 'shapefiles_samples'
        candidate_paths = [
            samples_base / self.file_prefix / "samples.shp",
            samples_base / f"{self.country}_{self.sanitized_track}" / "samples.shp",
            samples_base / f"{self.country}_{self.track}" / "samples.shp",
            samples_base / self.sanitized_track / "samples.shp",
            samples_base / self.track / "samples.shp",
            samples_base / self.country / "samples.shp",
            samples_base / "samples.shp"
        ]

        self.sample_shp = None
        for p in candidate_paths:
            if p.exists():
                self.sample_shp = p
                print(f"Training samples found at: {self.sample_shp}")
                break

        if not self.sample_shp:
            print(f"\nCRITICAL WARNING: Could not find 'samples.shp' inside {samples_base}")
            self.sample_shp = samples_base / self.file_prefix / "samples.shp"

        # Output paths
        self.learn_shp = self.samples_dir / 'learn.shp'
        self.control_shp = self.samples_dir / 'control.shp'
        self.sel_csv = self.samples_dir / f"{self.file_prefix}_prithvi_learn_features.csv"

        # Classification outputs
        self.class_tif = self.class_dir / f"{self.file_prefix}_prithvi_classified.tif"
        self.conf_tif = self.class_dir / f"{self.file_prefix}_prithvi_confidence_map.tif"

        self.footprint_mask = self.seg_dir / f"{self.file_prefix}_data_footprint.tif"
        self.masked_class = self.class_dir / f"{self.file_prefix}_prithvi_classified_masked.tif"
        self.masked_conf = self.class_dir / f"{self.file_prefix}_prithvi_confidence_masked.tif"
        self.metrics_fp = self.class_dir / f"{self.file_prefix}_prithvi_metrics.xlsx"

        self.agri_mask = self._resolve_agri_mask()
        self.stage1_params = {
            'method': 'python_sam',
            'tile_size': 2048,
            'buffer': 128,
            'sam_checkpoint': str(self.aux_dir / 'SAM_models' / 'sam_vit_h_4b8939.pth'),
            'sam_model_type': 'vit_h',
            'sam_device': 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu'
        }
        self.stage4_params = {
            'sk_hidden_sizes': '128,64',
            'sk_activation': 'relu',
            'sk_solver': 'adam',
            'sk_alpha': 0.0001,
            'sk_max_iter': 500,
            'balance_threshold': 1000
        }

    def _ensure_directories(self):
        for d in [self.out_dir, self.samples_dir, self.model_dir, self.seg_dir, self.class_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _resolve_agri_mask(self):
        raster_dir = self.aux_dir / 'raster_files'
        country_dir = raster_dir / 'AgriMasks' / self.country
        mask_3class = country_dir / f"{self.country}_agri_mask_3class_epsg3857.tif"
        mask_allcrops = country_dir / f"{self.country}_agri_mask_allcrops_epsg3857.tif"

        if mask_allcrops.exists():
            print(f"    [OK] Agricultural Mask selected: {mask_allcrops.name}")
            return mask_allcrops
        elif mask_3class.exists():
            print(f"    [OK] Agricultural Mask selected: {mask_3class.name}")
            return mask_3class
        else:
            print(f"    [WARNING] No agricultural mask found for country '{self.country}'.")
            return None

    # --- Stage 0: Robust Data Footprint ---
    def stage_0_generate_footprint(self, force_recompute=False):
        stage = 0
        if self.footprint_mask.exists() and not force_recompute:
            print("[Stage 0] Data footprint mask already exists, skipping.")
            return

        print(f"[Stage 0/{self.total_stages}] Generating robust data footprint mask from radar stack...")
        ds = gdal.Open(str(self.ras))
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        nbands = ds.RasterCount
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(str(self.footprint_mask), cols, rows, 1, gdal.GDT_Byte,
                               options=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'])
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)
        out_band = out_ds.GetRasterBand(1)

        tile_size = 2048
        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                xsize = min(tile_size, cols - x)
                ysize = min(tile_size, rows - y)

                valid_mask = np.ones((ysize, xsize), dtype=bool)
                for b in range(1, min(nbands + 1, 5)):
                    arr = ds.GetRasterBand(b).ReadAsArray(x, y, xsize, ysize)
                    arr = np.nan_to_num(arr)
                    valid_mask &= (arr != 0) & (~np.isnan(arr))

                out_arr = np.where(valid_mask, 1, 0).astype(np.uint8)
                out_band.WriteArray(out_arr, x, y)

        out_band.FlushCache()
        out_ds = None
        ds = None
        print(f"    Footprint mask saved to {self.footprint_mask}\n")

    def _create_summed_composite(self):
        """Creates a single-band composite by summing the log-domain (dB) values of all SAR bands to reduce speckle while preserving low-backscatter crop contrast."""
        print("    [INFO] Creating a log-domain (dB) summed composite of all SAR bands...")

        gdal.SetCacheMax(4 * 1024 * 1024 * 1024)

        composite_tif = self.seg_dir / f"{self.file_prefix}_summed_composite.tif"

        if composite_tif.exists():
            print(f"    [INFO] Summed composite already exists at {composite_tif}.")
            return composite_tif

        ds = gdal.Open(str(self.ras))
        if not ds:
            raise RuntimeError(f"Could not open source raster {self.ras}")

        cols = ds.RasterXSize
        rows = ds.RasterYSize
        nbands = ds.RasterCount
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(str(composite_tif), cols, rows, 1, gdal.GDT_Float32,
                               options=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'])
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
                    
                    if arr is None:
                        raise RuntimeError(f"Failed to read block at x={x}, y={y} for band {b}.")
                        
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
        print(f"    [INFO] Summed composite saved to {composite_tif}")
        return composite_tif

    # --- Stage 1: Segmentation (SAM) ---
    def stage_1_segmentation(self, force_recompute=False):
        stage = 1
        if self.seg_tif.exists() and not force_recompute:
            print(f"[Stage {stage}/{self.total_stages}] Segmentation Raster already exists, skipping.\n")
            return

        print(f"[Stage {stage}/{self.total_stages}] Generating object segmentation using SAM...")
        if not HAS_SAM:
            print("ERROR: segment-geospatial (samgeo) is not installed. Standalone SAM segmentation requires it.")
            return

        self._ensure_directories()
        
        # Create summed composite first
        original_ras = self.ras
        try:
            self.ras = self._create_summed_composite()
        except Exception as e:
            print(f"    [WARNING] Failed to create summed composite: {e}. Falling back to full stack.")
            
        self._run_python_segmentation_tiled(self.stage1_params, stage, 'python_sam')
        self.ras = original_ras

    def _run_python_segmentation_tiled(self, params, stage, method):
        print(f"    Running Tiled Python Segmentation ({method})...")
        try:
            ds = gdal.Open(str(self.ras))
            if not ds:
                raise RuntimeError("Could not open raster")

            # Open Data Footprint Mask (Stage 0 output) if it exists
            ds_foot = None
            if self.footprint_mask.exists():
                ds_foot = gdal.Open(str(self.footprint_mask))

            cols = ds.RasterXSize
            rows = ds.RasterYSize
            nbands = ds.RasterCount
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()

            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(str(self.seg_tif), cols, rows, 1, gdal.GDT_Int32,
                                   options=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'])
            out_ds.SetGeoTransform(gt)
            out_ds.SetProjection(proj)
            out_band = out_ds.GetRasterBand(1)
            out_band.SetNoDataValue(0)

            tile_size = params.get('tile_size', 2048)
            buffer = params.get('buffer', 128)
            global_seg_id = 1

            sam_geo = None
            if method == 'python_sam':
                print(f"    Loading SAM-Geo model ({params['sam_model_type']}) to {params['sam_device']}...")
                try:
                    sam_geo = SamGeo(
                        model_type=params['sam_model_type'],
                        checkpoint=params['sam_checkpoint'],
                        device=params['sam_device'],
                        sam_kwargs={
                            "points_per_side": 96,
                            "pred_iou_thresh": 0.55,
                            "stability_score_thresh": 0.55,
                            "crop_n_layers": 1,
                            "crop_n_points_downscale_factor": 2,
                            "min_mask_region_area": 10
                        }
                    )
                except Exception as e:
                    print(f"    [ERROR] Failed to load SAM-Geo model: {e}")
                    print("    Please ensure you have installed segment-geospatial and have the proper checkpoint.")
                    return

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

                    print(f"    Processing Tile: x={x}, y={y} (buffered {xsize_buf}x{ysize_buf})")

                    img_list = []
                    for b in range(1, nbands + 1):
                        band = ds.GetRasterBand(b)
                        arr = band.ReadAsArray(x_start_buf, y_start_buf, xsize_buf, ysize_buf)
                        if arr is None:
                            img_list = None
                            break
                        arr = np.nan_to_num(arr)
                        img_list.append(arr)

                    if img_list is None:
                        continue

                    img = np.dstack(img_list)

                    # Use footprint mask if available, otherwise fallback to sum > 0
                    if ds_foot:
                        valid_mask_buf = ds_foot.GetRasterBand(1).ReadAsArray(x_start_buf, y_start_buf, xsize_buf, ysize_buf) > 0
                        valid_mask = valid_mask_buf
                    else:
                        valid_mask = np.sum(np.abs(img), axis=2) > 0

                    if not np.any(valid_mask):
                        continue

                    img_norm = img_as_float(img)

                    if method == 'python_sam':
                        import cv2
                        from scipy.ndimage import distance_transform_edt
                        
                        # Convert float32 1-band to 8-bit RGB for SAM
                        img_8bit = np.zeros(img.shape, dtype=np.uint8)
                        valid_pixels = valid_mask[:, :, np.newaxis]
                        
                        if np.any(valid_pixels):
                            p2, p98 = np.percentile(img[valid_pixels], (2, 98))
                            img_clip = np.clip(img, p2, p98)
                            
                            # Avoid division by zero
                            if p98 > p2:
                                img_8bit[valid_pixels] = ((img_clip[valid_pixels] - p2) / (p98 - p2) * 255).astype(np.uint8)
                                
                            # Apply CLAHE to enhance contrast in darker regions (SAR data is very skewed)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            img_clahe = clahe.apply(img_8bit[:, :, 0])
                            img_8bit[:, :, 0] = img_clahe
                            
                            # Apply bilateral filter to smooth out speckle noise while preserving sharp boundaries
                            print("    [SAM-Geo] Applying bilateral filter for edge-preserving speckle smoothing...")
                            img_chan = np.ascontiguousarray(img_8bit[:, :, 0])
                            img_smoothed = cv2.bilateralFilter(img_chan, d=9, sigmaColor=50, sigmaSpace=50)
                            img_8bit[:, :, 0] = img_smoothed
                            
                        # SAM requires 3 channel RGB
                        if img_8bit.shape[2] == 1:
                            img_rgb = np.repeat(img_8bit, 3, axis=2)
                        else:
                            img_rgb = img_8bit[:, :, :3]
                            if img_rgb.shape[2] < 3:
                                img_rgb = np.pad(img_rgb, ((0,0),(0,0),(0, 3-img_rgb.shape[2])), mode='constant')
                                
                        sam_geo.generate(
                            source=img_rgb,
                            output=None,
                            foreground=False,
                            unique=True,
                            min_size=10,
                            max_size=100000
                        )
                        segments_buf = sam_geo.objects.astype(np.int32)
                                    
                        # Fill empty spaces (NoData) with nearest segment (distance transform)
                        zero_mask_buf = (segments_buf == 0) & valid_mask
                        if np.any(zero_mask_buf) and np.any(segments_buf > 0):
                            _, indices = distance_transform_edt(segments_buf == 0, return_indices=True)
                            segments_buf[zero_mask_buf] = segments_buf[tuple(indices)][zero_mask_buf]

                    y_offset = y - y_start_buf
                    x_offset = x - x_start_buf
                    
                    # Mask out segments outside footprint
                    segments_buf[~valid_mask] = 0

                    # Get the valid mask for the unbuffered tile
                    valid_mask_crop = valid_mask[y_offset : y_offset + ysize_valid, x_offset : x_offset + xsize_valid]
                    segments = segments_buf[y_offset : y_offset + ysize_valid, x_offset : x_offset + xsize_valid]
                    
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

            out_ds.FlushCache()
            out_ds = None
            if ds_foot:
                ds_foot = None
            print(f"    Segmentation Raster saved to {self.seg_tif}\n")

        except Exception as e:
            print(f"ERROR in Python segmentation: {e}")
            raise

    # --- Stage 2: Prepare Point Split ---
    def stage_2_prepare_points(self, force_recompute=False):
        stage = 2
        if self.learn_shp.exists() and self.control_shp.exists() and not force_recompute:
            print("[Stage 2] Split samples already exist, skipping.")
            return

        print(f"[Stage 2/{self.total_stages}] Preparing Point Samples split...")
        if not self.sample_shp.exists():
            print(f"ERROR: Base samples shapefile not found at {self.sample_shp}")
            return

        gdf = gpd.read_file(str(self.sample_shp), engine="pyogrio")
        if 'crop_id' not in gdf.columns:
            print("ERROR: Column 'crop_id' not found in samples.")
            return

        gdf_shuffled = gdf.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(gdf_shuffled) * 0.7)
        gdf_learn = gdf_shuffled.iloc[:split_idx]
        gdf_control = gdf_shuffled.iloc[split_idx:]

        gdf_learn.to_file(str(self.learn_shp), driver="ESRI Shapefile", engine="pyogrio")
        gdf_control.to_file(str(self.control_shp), driver="ESRI Shapefile", engine="pyogrio")
        print(f"    Saved learn points ({len(gdf_learn)}) and control points ({len(gdf_control)})\n")

    # --- Stage 3: Feature Extraction (Prithvi-SAR) ---
    def stage_3_selection(self):
        stage = 3
        if self.sel_csv.exists():
            print(f"[Stage {stage}] Prithvi-SAR features already extracted, skipping.")
            return

        print(f"[Stage {stage}/{self.total_stages}] Extracting deep PRITHVI-SAR features for training segments...")
        
        # Setup and load model
        mae_script, weights_path = resolve_prithvi_model()
        model = load_prithvi_encoder(weights_path)
        
        if not self.learn_shp.exists():
            print("ERROR: Learn samples not found.")
            return

        gdf = gpd.read_file(str(self.learn_shp), engine="pyogrio")
        
        ds = gdal.Open(str(self.ras))
        gt = ds.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)
        raster_proj = ds.GetProjection()
        cols = ds.RasterXSize
        rows = ds.RasterYSize

        if raster_proj and gdf.crs:
            from pyproj import CRS
            target_crs = CRS.from_wkt(raster_proj)
            if gdf.crs != target_crs:
                print("    Reprojecting samples to Match Raster CRS...")
                gdf = gdf.to_crs(target_crs)

        seg_ds = gdal.Open(str(self.seg_tif))
        seg_band = seg_ds.GetRasterBand(1)
        seg_array = seg_band.ReadAsArray()
        
        # Find which segments contain training points
        xs = gdf.geometry.x.values
        ys = gdf.geometry.y.values
        pxs = (inv_gt[0] + inv_gt[1] * xs + inv_gt[2] * ys).astype(int)
        pys = (inv_gt[3] + inv_gt[4] * xs + inv_gt[5] * ys).astype(int)
        crop_ids = gdf['crop_id'].values

        target_segments = {}
        for px, py, crop_id in zip(pxs, pys, crop_ids):
            if 0 <= px < cols and 0 <= py < rows:
                seg_id = seg_array[py, px]
                if seg_id > 0:
                    target_segments[seg_id] = crop_id

        if not target_segments:
            print("ERROR: No valid samples found overlapping the raster.")
            return

        print(f"    Found {len(target_segments)} unique segments for training.")
        
        # Compute bounding boxes for these segments
        print("    Computing segment bounding boxes...")
        from scipy.ndimage import find_objects
        slices = find_objects(seg_array)
        
        features_list = []
        seg_ids_list = []
        crop_ids_list = []
        
        count = 0
        total = len(target_segments)
        
        # We process segments batch by batch
        batch_size = 16
        batch_tensors = []
        batch_seg_ids = []
        
        for seg_id, crop_id in target_segments.items():
            if seg_id - 1 >= len(slices) or slices[seg_id - 1] is None:
                continue
                
            sl = slices[seg_id - 1]
            y_min, y_max = sl[0].start, sl[0].stop
            x_min, x_max = sl[1].start, sl[1].stop
            
            # Ensure bounding box is at least 16x16
            if x_max - x_min < 16:
                x_max = min(cols, x_min + 16)
                x_min = max(0, x_max - 16)
            if y_max - y_min < 16:
                y_max = min(rows, y_min + 16)
                y_min = max(0, y_max - 16)
                
            bbox = (x_min, y_min, x_max, y_max)
            
            try:
                # Prepare tensor shape [6, 3, 224, 224]
                tensor = prepare_segment_patch(ds, seg_array, bbox)
                batch_tensors.append(tensor)
                batch_seg_ids.append(seg_id)
            except Exception as e:
                # print(f"Error preparing patch for segment {seg_id}: {e}")
                continue
                
            if len(batch_tensors) == batch_size or (count + len(batch_tensors) == total):
                # Run Prithvi inference
                input_batch = torch.stack(batch_tensors, dim=0) # [B, 6, 3, 224, 224]
                with torch.no_grad():
                    out = model(input_batch)
                    # Out is a tuple, first item is embeddings: [B, 148, 768]
                    embeddings = out[0]
                    # Mean pool over tokens -> [B, 768]
                    pooled = embeddings.mean(dim=1).numpy()
                    
                for idx, pooled_feat in enumerate(pooled):
                    curr_seg_id = batch_seg_ids[idx]
                    features_list.append(pooled_feat)
                    seg_ids_list.append(curr_seg_id)
                    crop_ids_list.append(target_segments[curr_seg_id])
                    
                count += len(batch_tensors)
                sys.stdout.write(f"\r      Extracted features for {count}/{total} segments...  ")
                sys.stdout.flush()
                
                batch_tensors = []
                batch_seg_ids = []
                
        print("\n    Aggregation complete. Saving features...")
        
        feature_data = {'crop_id': crop_ids_list, 'seg_id': seg_ids_list}
        features_arr = np.array(features_list)
        for f_idx in range(768):
            feature_data[f'feat{f_idx}'] = features_arr[:, f_idx]
            
        df_final = pd.DataFrame(feature_data)
        df_final.to_csv(self.sel_csv, index=False)
        print(f"    Prithvi-SAR Features saved to {self.sel_csv}\n")
        
        ds = None
        seg_ds = None

    # --- Stage 4: Train Classifier ---
    def stage_4_train_classifier(self, **kwargs):
        self._ensure_directories()
        params = self.stage4_params.copy()
        params.update(kwargs)
        stage = 4

        if not self.sel_csv.exists():
            print("ERROR: Feature CSV not found.")
            return

        model_fn = self.model_dir / f"{self.file_prefix}_prithvi_model.pkl"
        print(f"[Stage {stage}/{self.total_stages}] Training Classifier (ANN) on Prithvi-SAR Embeddings...")

        df = pd.read_csv(self.sel_csv)
        feat_cols = [c for c in df.columns if c.startswith('feat')]
        self.feat_cols = feat_cols

        print("    Balancing classes (Capped Oversampling)...")
        threshold = params.get('balance_threshold', 1000)
        df_balanced = pd.DataFrame()
        for crop_id in df['crop_id'].unique():
            df_class = df[df['crop_id'] == crop_id]
            count = len(df_class)

            if count < threshold:
                df_resampled = resample(df_class, replace=True, n_samples=threshold, random_state=42)
                df_balanced = pd.concat([df_balanced, df_resampled])
            else:
                df_balanced = pd.concat([df_balanced, df_class])

        print(f"    Original samples: {len(df)}. Balanced samples: {len(df_balanced)}")

        X = df_balanced[feat_cols].values
        y = df_balanced['crop_id'].values

        if self.country == 'NL':
            print("    Applying crop aggregation for Netherlands training labels...")
            crop_aggregation = get_crop_aggregation(self.country, self.learn_shp)
            y = np.array([crop_aggregation.get(val, val) for val in y])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        hidden_sizes = tuple(map(int, str(params['sk_hidden_sizes']).split(',')))
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation=params['sk_activation'],
            solver=params['sk_solver'],
            alpha=params['sk_alpha'],
            max_iter=params['sk_max_iter'],
            random_state=42,
            verbose=True
        )

        clf.fit(X_scaled, y)

        joblib.dump({'model': clf, 'scaler': scaler, 'feats': feat_cols}, model_fn)
        print(f"Model saved to {model_fn}")

        y_pred = clf.predict(X_scaled)
        labels = sorted(list(set(y)))
        cm = confusion_matrix(y, y_pred, labels=labels)
        print("\n--- Training Confusion Matrix ---")
        print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
        print("\n")

    # --- Stage 5: Tiled Inference (Prithvi-SAR + ANN) ---
    def stage_5_classify_vector(self, force_recompute=False):
        self._ensure_directories()
        stage = 5

        model_file = self.model_dir / f"{self.file_prefix}_prithvi_model.pkl"
        if not model_file.exists():
            print("ERROR: Model not found.")
            return

        if self.class_tif.exists() and not force_recompute:
            print(f"[Stage {stage}] Classification Raster exists, skipping.")
            return

        print(f"[Stage {stage}/{self.total_stages}] Running Object-Based Inference using Prithvi-SAR...")

        data = joblib.load(model_file)
        clf = data['model']
        scaler = data['scaler']
        
        # Load Prithvi-SAR
        mae_script, weights_path = resolve_prithvi_model()
        model = load_prithvi_encoder(weights_path)
        
        # --- Calculate Bayesian Priors ---
        print("    Calculating Bayesian priors for Prithvi inference...")
        df_learn = pd.read_csv(self.sel_csv)
        y_train = df_learn['crop_id'].values
        if self.country == 'NL':
            print("    Applying crop aggregation for Netherlands priors...")
            crop_aggregation = get_crop_aggregation(self.country, self.learn_shp)
            y_train = np.array([crop_aggregation.get(val, val) for val in y_train])

        class_counts = pd.Series(y_train).value_counts().to_dict()
        classes = clf.classes_
        total_samples = len(df_learn)
        
        p_true = _get_priors_for_country(
            self.country, self.learn_shp, classes, class_counts, total_samples, 
            priors_file_override=self.base_dir / "priors.json"
        )
        
        threshold = 1000
        balanced_counts = {}
        for c in classes:
            count = len(df_learn[df_learn['crop_id'] == c])
            balanced_counts[c] = max(count, threshold)
        total_balanced = sum(balanced_counts.values())
        p_train = np.array([balanced_counts[c] / total_balanced for c in classes])
        
        correction = p_true / (p_train + 1e-9)
        correction = np.power(correction, 0.7)
        # Cap extreme multipliers to prevent division by zero or extreme noise, but keep bounds wide enough
        # so that true class priors (like Grassland with 72%) can be correctly reflected.
        correction = np.clip(correction, 0.01, 10.0)
        priors_arr = correction / np.sum(correction)
        # ---------------------------------

        ds_stack_info = gdal.Open(str(self.ras))
        cols = ds_stack_info.RasterXSize
        rows = ds_stack_info.RasterYSize
        gt = ds_stack_info.GetGeoTransform()
        proj = ds_stack_info.GetProjection()
        ds_stack_info = None

        driver = gdal.GetDriverByName('GTiff')
        ds_cls = driver.Create(str(self.class_tif), cols, rows, 1, gdal.GDT_Int32,
                               options=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'])
        ds_cls.SetGeoTransform(gt)
        ds_cls.SetProjection(proj)
        ds_cls.GetRasterBand(1).SetNoDataValue(0)

        ds_conf = driver.Create(str(self.conf_tif), cols, rows, 1, gdal.GDT_Float32,
                                options=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'])
        ds_conf.SetGeoTransform(gt)
        ds_conf.SetProjection(proj)
        ds_conf.GetRasterBand(1).SetNoDataValue(0)

        tile_size = 2048
        write_lock = threading.Lock()

        # Load entire segmentation database
        seg_ds = gdal.Open(str(self.seg_tif))
        seg_band = seg_ds.GetRasterBand(1)
        seg_array = seg_band.ReadAsArray()
        seg_ds = None
        
        print("    Computing full segmentation bounding boxes...")
        from scipy.ndimage import find_objects
        slices = find_objects(seg_array)

        def process_tile(x, y):
            xsize = min(tile_size, cols - x)
            ysize = min(tile_size, rows - y)

            ds_stack = gdal.Open(str(self.ras))
            ds_foot = gdal.Open(str(self.footprint_mask))

            try:
                # Sub-segmentation
                sub_seg = seg_array[y:y+ysize, x:x+xsize]
                foot_arr = ds_foot.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                
                unique_ids = np.unique(sub_seg)
                unique_ids = unique_ids[unique_ids > 0]
                
                if len(unique_ids) == 0:
                    return

                # Prepare patches batch by batch
                features = []
                valid_ids = []
                batch_tensors = []
                batch_ids = []
                batch_size = 16
                
                for sid in unique_ids:
                    if sid - 1 >= len(slices) or slices[sid - 1] is None:
                        continue
                    sl = slices[sid - 1]
                    y_min, y_max = sl[0].start, sl[0].stop
                    x_min, x_max = sl[1].start, sl[1].stop
                    
                    if x_max - x_min < 16:
                        x_max = min(cols, x_min + 16)
                        x_min = max(0, x_max - 16)
                    if y_max - y_min < 16:
                        y_max = min(rows, y_min + 16)
                        y_min = max(0, y_max - 16)
                        
                    bbox = (x_min, y_min, x_max, y_max)
                    
                    try:
                        tensor = prepare_segment_patch(ds_stack, seg_array, bbox)
                        batch_tensors.append(tensor)
                        batch_ids.append(sid)
                    except:
                        continue
                        
                    if len(batch_tensors) == batch_size:
                        input_batch = torch.stack(batch_tensors, dim=0)
                        with torch.no_grad():
                            out = model(input_batch)
                            pooled = out[0].mean(dim=1).numpy()
                        for idx, p_f in enumerate(pooled):
                            features.append(p_f)
                            valid_ids.append(batch_ids[idx])
                        batch_tensors = []
                        batch_ids = []
                        
                if batch_tensors:
                    input_batch = torch.stack(batch_tensors, dim=0)
                    with torch.no_grad():
                        out = model(input_batch)
                        pooled = out[0].mean(dim=1).numpy()
                    for idx, p_f in enumerate(pooled):
                        features.append(p_f)
                        valid_ids.append(batch_ids[idx])

                if not valid_ids:
                    return

                features_arr = np.stack(features, axis=0)
                X_scaled = scaler.transform(features_arr)
                
                # Classify
                raw_probs = clf.predict_proba(X_scaled)
                
                # Apply Bayesian Prior Correction
                corrected_probs = raw_probs * priors_arr
                corrected_probs = corrected_probs / np.sum(corrected_probs, axis=1, keepdims=True)
                
                preds = clf.classes_[np.argmax(corrected_probs, axis=1)]
                max_probs = np.max(corrected_probs, axis=1)

                id_to_pred = {sid: int(pred) for sid, pred in zip(valid_ids, preds)}
                id_to_prob = {sid: float(prob) for sid, prob in zip(valid_ids, max_probs)}

                # Create output arrays
                pred_arr = np.zeros_like(sub_seg, dtype=np.int32)
                prob_arr = np.zeros_like(sub_seg, dtype=np.float32)

                for sid in unique_ids:
                    if sid in id_to_pred:
                        mask = (sub_seg == sid)
                        pred_arr[mask] = id_to_pred[sid]
                        prob_arr[mask] = id_to_prob[sid]

                # Force footprint masking
                pred_arr[foot_arr == 0] = 0
                prob_arr[foot_arr == 0] = 0

                with write_lock:
                    ds_cls.GetRasterBand(1).WriteArray(pred_arr, x, y)
                    ds_conf.GetRasterBand(1).WriteArray(prob_arr, x, y)

            except Exception as e:
                # print(f"Error processing tile x={x}, y={y}: {e}")
                pass
            finally:
                ds_stack = None
                ds_foot = None

        # Build task list
        tiles = []
        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                tiles.append((x, y))

        print(f"    Processing {len(tiles)} tiles in parallel (Threads: 4)...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(lambda t: process_tile(*t), tiles)

        ds_cls.GetRasterBand(1).FlushCache()
        ds_conf.GetRasterBand(1).FlushCache()
        ds_cls = None
        ds_conf = None
        print("    Classification and Confidence map generated successfully!\n")

    # --- Stage 6: Apply Masks ---
    def stage_6_mask_classification(self, force_recompute=False):
        stage = 6
        if self.masked_class.exists() and self.masked_conf.exists() and not force_recompute:
            print("[Stage 6] Masked outputs already exist, skipping.")
            return

        print(f"[Stage 6/{self.total_stages}] Applying Arable & Data Footprint Mask...")
        ds_stack = gdal.Open(str(self.ras))
        if not ds_stack:
            raise RuntimeError(f"Could not open source raster {self.ras}")
        cols = ds_stack.RasterXSize
        rows = ds_stack.RasterYSize
        gt = ds_stack.GetGeoTransform()
        proj = ds_stack.GetProjection()

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
                raise RuntimeError("Failed to warp the arable mask.")
            print(f"    Applying country agricultural mask: {mask_tif.name}")
        else:
            print("    WARNING: Arable mask not found. Only applying data footprint mask.")

        ds_foot = gdal.Open(str(self.footprint_mask))
        ds_cls = gdal.Open(str(self.class_tif))
        ds_conf = gdal.Open(str(self.conf_tif))

        driver = gdal.GetDriverByName('GTiff')
        out_cls = driver.Create(str(self.masked_class), cols, rows, 1, gdal.GDT_Int32,
                                options=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'])
        out_cls.SetGeoTransform(gt)
        out_cls.SetProjection(proj)
        out_cls.GetRasterBand(1).SetNoDataValue(0)

        out_conf = driver.Create(str(self.masked_conf), cols, rows, 1, gdal.GDT_Float32,
                                 options=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES'])
        out_conf.SetGeoTransform(gt)
        out_conf.SetProjection(proj)
        out_conf.GetRasterBand(1).SetNoDataValue(0)

        tile_size = 2048
        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                xsize = min(tile_size, cols - x)
                ysize = min(tile_size, rows - y)

                cls_arr = ds_cls.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                conf_arr = ds_conf.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)
                foot_arr = ds_foot.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)

                mask_arr = np.ones((ysize, xsize), dtype=np.uint8)
                if ds_mask:
                    mask_arr = ds_mask.GetRasterBand(1).ReadAsArray(x, y, xsize, ysize)

                # Combine masks
                combined_mask = (foot_arr > 0) & (mask_arr > 0)
                cls_arr[~combined_mask] = 0
                conf_arr[~combined_mask] = 0.0

                out_cls.GetRasterBand(1).WriteArray(cls_arr, x, y)
                out_conf.GetRasterBand(1).WriteArray(conf_arr, x, y)

        out_cls.GetRasterBand(1).FlushCache()
        out_conf.GetRasterBand(1).FlushCache()
        out_cls = None
        out_conf = None
        ds_mask = None
        ds_foot = None
        ds_cls = None
        ds_conf = None
        ds_stack = None
        
        if temp_mask_vrt and os.path.exists(temp_mask_vrt):
            try:
                os.remove(temp_mask_vrt)
            except Exception as e:
                print(f"Warning: Could not remove temp VRT: {e}")
                
        print("    Masking stage complete.\n")

    # --- Stage 7: Validation Metrics ---
    def stage_7_calculate_metrics(self):
        stage = 7
        print(f"[Stage {stage}/{self.total_stages}] Calculating Validation Metrics (OOB Control points)...")

        if not self.control_shp.exists():
            print("ERROR: Control samples not found.")
            return

        gdf = gpd.read_file(str(self.control_shp), engine="pyogrio")
        ds = gdal.Open(str(self.masked_class))
        gt = ds.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)
        raster_proj = ds.GetProjection()

        if raster_proj and gdf.crs:
            from pyproj import CRS
            target_crs = CRS.from_wkt(raster_proj)
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)

        xs = gdf.geometry.x.values
        ys = gdf.geometry.y.values
        pxs = (inv_gt[0] + inv_gt[1] * xs + inv_gt[2] * ys).astype(int)
        pys = (inv_gt[3] + inv_gt[4] * xs + inv_gt[5] * ys).astype(int)
        
        true_labels = gdf['crop_id'].values
        if self.country == 'NL':
            print("    Applying crop aggregation for Netherlands validation labels...")
            crop_aggregation = get_crop_aggregation(self.country, self.control_shp)
            true_labels = np.array([crop_aggregation.get(val, val) for val in true_labels])

        pred_labels = []
        valid_indices = []

        band = ds.GetRasterBand(1)
        for i, (px, py) in enumerate(zip(pxs, pys)):
            if 0 <= px < ds.RasterXSize and 0 <= py < ds.RasterYSize:
                val = int(band.ReadAsArray(int(px), int(py), 1, 1)[0, 0])
                if val > 0:
                    if self.country == 'NL':
                        crop_aggregation = get_crop_aggregation(self.country, self.control_shp)
                        val = crop_aggregation.get(val, val)
                    pred_labels.append(val)
                    valid_indices.append(i)

        ds = None
        true_labels = true_labels[valid_indices]

        if len(pred_labels) == 0:
            print("ERROR: No valid control points overlap the classified raster mask.")
            return

        print(f"    Extracted {len(pred_labels)} validation points overlapping the mask.")
        
        # Calculate scores
        labels = sorted(list(set(true_labels) | set(pred_labels)))
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, labels=labels, zero_division=0)

        # Write to Excel
        wb = openpyxl.Workbook()
        ws_summary = wb.active
        ws_summary.title = "Summary Metrics"
        ws_summary.append(["Class ID", "Precision", "Recall", "F1-Score", "Support"])
        
        for idx, lbl in enumerate(labels):
            ws_summary.append([lbl, precision[idx], recall[idx], f1[idx], int(support[idx])])

        total_correct = np.sum(np.diag(cm))
        total_samples = np.sum(cm)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        ws_summary.append([])
        ws_summary.append(["Overall Accuracy", accuracy])

        ws_cm = wb.create_sheet(title="Confusion Matrix")
        ws_cm.append(["True \\ Pred"] + labels)
        for idx, row in enumerate(cm):
            ws_cm.append([labels[idx]] + list(map(int, row)))

        wb.save(self.metrics_fp)
        print(f"    Metrics successfully saved to {self.metrics_fp}")
        print(f"    Overall Accuracy: {accuracy:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Object-Based Crop Classification using NASA-IBM Prithvi-SAR")
    parser.add_argument('--track', required=True, help="Track name, e.g. NL/orbit_88 or PT/orbit_161")
    args = parser.parse_args()

    pipeline = ProcessingPipeline(args.track)
    
    print("\n--- PRITHVI-SAR CLASSIFICATION MENU ---")
    print("  [1] Stage 0: Generate Data Footprint")
    print("  [2] Stage 1: Segmentation (SAM)")
    print("  [3] Stage 2: Prepare Point Split")
    print("  [4] Stage 3: Extract Prithvi-SAR Features")
    print("  [5] Stage 4: Train ANN Classifier")
    print("  [6] Stage 5: Run Inference (Object-based)")
    print("  [7] Stage 6: Apply Agricultural Mask")
    print("  [8] Stage 7: Calculate Validation Metrics")
    
    print("\n  [A] Run All Stages (Stages 0-7)")
    print("  [Q] Quit")
    
    choice = input("\nEnter choice: ").strip().upper()
    if choice == 'A':
        pipeline.stage_0_generate_footprint()
        pipeline.stage_1_segmentation()
        pipeline.stage_2_prepare_points()
        pipeline.stage_3_selection()
        pipeline.stage_4_train_classifier()
        pipeline.stage_5_classify_vector(force_recompute=True)
        pipeline.stage_6_mask_classification(force_recompute=True)
        pipeline.stage_7_calculate_metrics()
    elif choice == '1':
        pipeline.stage_0_generate_footprint(force_recompute=True)
    elif choice == '2':
        pipeline.stage_1_segmentation(force_recompute=True)
    elif choice == '3':
        pipeline.stage_2_prepare_points(force_recompute=True)
    elif choice == '4':
        pipeline.stage_3_selection()
    elif choice == '5':
        pipeline.stage_4_train_classifier()
    elif choice == '6':
        pipeline.stage_5_classify_vector(force_recompute=True)
    elif choice == '7':
        pipeline.stage_6_mask_classification(force_recompute=True)
    elif choice == '8':
        pipeline.stage_7_calculate_metrics()
    elif choice == 'Q':
        sys.exit(0)
    else:
        print("Invalid choice. Exiting.")


if __name__ == '__main__':
    main()
