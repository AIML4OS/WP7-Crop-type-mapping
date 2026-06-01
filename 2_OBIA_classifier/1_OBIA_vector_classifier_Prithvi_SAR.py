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

track_regions = {
    'P1': 'AT', 'P1a': 'AT',
    'P2': 'IE', 'P2a': 'IE',
    'P3': 'NL',
    'P4': 'PT', 'P4a': 'PT'
}
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
    prithvi_tensor = np.zeros((6, 3, h, w), dtype=np.float32)
    for t in range(3):
        vv, vh = frames[t]
        prithvi_tensor[0, t] = vv
        prithvi_tensor[1, t] = vh
        prithvi_tensor[2, t] = vv
        prithvi_tensor[3, t] = vh
        prithvi_tensor[4, t] = vv
        prithvi_tensor[5, t] = vh
        
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
        else:
            self.country = track_regions.get(track)
            if not self.country:
                if len(track) == 2:
                    self.country = track.upper()
                else:
                    print(f"Error: Track '{track}' not defined in track_regions configuration.")
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

    # --- Stage 1 & 2: Reuse points from samples.shp ---
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
                preds = clf.predict(X_scaled)
                probs = clf.predict_proba(X_scaled)
                max_probs = np.max(probs, axis=1)

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
        if mask_tif and mask_tif.exists():
            ds_mask = gdal.Open(str(mask_tif))
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
        pred_labels = []
        valid_indices = []

        band = ds.GetRasterBand(1)
        for i, (px, py) in enumerate(zip(pxs, pys)):
            if 0 <= px < ds.RasterXSize and 0 <= py < ds.RasterYSize:
                val = int(band.ReadAsArray(int(px), int(py), 1, 1)[0, 0])
                if val > 0:
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
    print("  [1] Run Full Classification Pipeline (Stages 0-7)")
    print("  [2] Stage 0: Generate Data Footprint")
    print("  [3] Stage 2: Prepare Point Split")
    print("  [4] Stage 3: Extract Prithvi-SAR Features")
    print("  [5] Stage 4: Train ANN Classifier")
    print("  [6] Stage 5: Run Inference (Object-based)")
    print("  [7] Stage 6: Apply Agricultural Mask")
    print("  [8] Stage 7: Calculate Validation Metrics")
    
    choice = input("\nEnter choice (1-8): ").strip()
    if choice == '1':
        pipeline.stage_0_generate_footprint()
        pipeline.stage_2_prepare_points()
        pipeline.stage_3_selection()
        pipeline.stage_4_train_classifier()
        pipeline.stage_5_classify_vector()
        pipeline.stage_6_mask_classification()
        pipeline.stage_7_calculate_metrics()
    elif choice == '2':
        pipeline.stage_0_generate_footprint(force_recompute=True)
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
    else:
        print("Invalid choice. Exiting.")


if __name__ == '__main__':
    main()
