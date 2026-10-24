"""
================================================================================
AIML CropMapper Cloud - NISAR L2 GCOV Download & Polarimetric Extraction
================================================================================
Downloads NISAR L2 GCOV HDF5 granules (or reads pre-staged local HDF5 archives)
and extracts 4 physical L-band polarimetric channels per scene:
  1. HHHH_gamma0.tif (Co-polarized L-band gamma0 backscatter)
  2. HVHV_gamma0.tif (Cross-polarized L-band gamma0 backscatter)
  3. RVI_L.tif       (L-band Radar Vegetation Index: 4*HV / (HH + HV))
  4. ratio_HH_HV.tif (Co/Cross-pol power ratio: HH / HV)

Includes cross-track horizontal border trimming, internal pinhole EDT healing,
and 2D along-track adaptive monotonic range-edge de-tapering to eliminate
antenna pattern roll-off at swath edges. Supports both `h5py` and native `GDAL`
HDF5 driver fallback.
================================================================================
"""

import http.cookiejar
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage
from osgeo import gdal, osr

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    h5py = None
    HAS_H5PY = False


class DownloadAndExtractGCOV:
    """
    Downloads NISAR L2 GCOV products and extracts polarimetric layers
    (HHHH, HVHV, RVI_L, ratio_HH_HV) into Cloud-Optimized/BigTIFF GeoTIFFs.
    """

    def __init__(
        self,
        product: Dict,
        temp_dir: Path,
        session=None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        keep_intermediate_h5: bool = False,
        border_trim_pixels: int = 8
    ):
        self.product = product
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.session = session
        self.username = username
        self.password = password
        self.keep_intermediate_h5 = keep_intermediate_h5
        self.border_trim_pixels = int(border_trim_pixels)

    @staticmethod
    def _detaper_range_edges(arr: np.ndarray, W: int = 265, min_gain: float = 0.015) -> np.ndarray:
        """
        Eliminates cross-track (range) antenna pattern roll-off at the West and East swath edges
        using 2D along-track adaptive monotonic de-tapering before polarimetric ratio calculation.
        Leaves along-track (azimuth) top/bottom frame edges untouched.
        """
        from scipy.ndimage import uniform_filter1d

        out = arr.copy()
        rows, cols = out.shape
        ref_w = 160
        if rows < 100 or cols < W + ref_w + 20:
            return out

        for side in ("west", "east"):
            mask = out > 1e-6
            has_valid = np.any(mask, axis=1)
            first_x = np.argmax(mask, axis=1)
            last_x = cols - 1 - np.argmax(mask[:, ::-1], axis=1)
            span = last_x - first_x

            ex_raw = first_x if side == "west" else last_x
            valid_mask = has_valid & (span > W + ref_w + 10) & (ex_raw > 5) & (ex_raw < cols - 5)
            edge_x = np.where(valid_mask, ex_raw, -1).astype(np.int32)

            valid_r = np.where(edge_x >= 0)[0]
            if len(valid_r) < 120:
                continue

            ex_smooth = uniform_filter1d(edge_x[valid_r].astype(np.float32), size=160)
            dx_dr = np.gradient(ex_smooth, valid_r.astype(np.float32))
            is_range = (np.abs(dx_dr) > 0.15) & (np.abs(dx_dr) < 0.85)
            range_r = valid_r[is_range]
            if len(range_r) < 100:
                continue

            ex_r = edge_x[range_r]
            if side == "west":
                in_bounds = (ex_r + W + ref_w) < cols
            else:
                in_bounds = (ex_r - W - ref_w) >= 0
            range_r = range_r[in_bounds]
            ex_r = ex_r[in_bounds]
            if len(range_r) < 100:
                continue

            d_idx = np.arange(W + ref_w, dtype=np.int32)[None, :]
            if side == "west":
                strips = out[range_r[:, None], ex_r[:, None] + d_idx]
            else:
                strips = out[range_r[:, None], ex_r[:, None] - d_idx]

            ref_slices = strips[:, W: W + ref_w]
            ref_vals = np.where(ref_slices > 1e-5, ref_slices, np.nan)
            valid_ref_cnt = np.sum(~np.isnan(ref_vals), axis=1)
            ref_med = np.nanmedian(ref_vals, axis=1)
            good_rows = (valid_ref_cnt > 40) & (ref_med > 1e-5)
            if np.sum(good_rows) < 80:
                continue

            ratio_mat = np.full((rows, W), np.nan, dtype=np.float32)
            ratio_mat[range_r[good_rows], :] = strips[good_rows, :W] / ref_med[good_rows, None]

            win_half = 320
            step = 100
            ctrl_rows = []
            ctrl_g = []
            for r_center in range(0, rows, step):
                r0 = max(0, r_center - win_half)
                r1 = min(rows, r_center + win_half)
                sub = ratio_mat[r0:r1, :]
                if np.max(np.sum(~np.isnan(sub), axis=0)) >= 40:
                    g_loc = np.nanmedian(sub, axis=0)
                    g_loc = np.where(np.isnan(g_loc), 1.0, g_loc)
                    g_loc = np.maximum.accumulate(g_loc)
                    g_loc = np.clip(g_loc, 1e-4, 1.0)
                    ctrl_rows.append(r_center)
                    ctrl_g.append(g_loc)

            if not ctrl_rows:
                continue

            ctrl_rows_arr = np.array(ctrl_rows, dtype=np.float32)
            ctrl_g_arr = np.stack(ctrl_g, axis=0)
            r_float = range_r.astype(np.float32)
            g_sub = np.empty((len(range_r), W), dtype=np.float32)
            for d in range(W):
                g_sub[:, d] = np.interp(r_float, ctrl_rows_arr, ctrl_g_arr[:, d])
            g_sub = np.maximum.accumulate(g_sub, axis=1)

            d_w = np.arange(W, dtype=np.int32)[None, :]
            if side == "west":
                cols_idx = ex_r[:, None] + d_w
            else:
                cols_idx = ex_r[:, None] - d_w

            cur_vals = out[range_r[:, None], cols_idx]
            low_snr = (g_sub < min_gain) & (d_w < 12)
            corrected = np.where(low_snr, 0.0, cur_vals / g_sub)
            out[range_r[:, None], cols_idx] = corrected

        return np.clip(out, 0.0, 10.0)

    def _save_geotiff(self, array: np.ndarray, output_path: Path, x_coords: np.ndarray, y_coords: np.ndarray, epsg: int):
        rows, cols = array.shape
        x_res = (x_coords[-1] - x_coords[0]) / max(len(x_coords) - 1, 1)
        y_res = (y_coords[-1] - y_coords[0]) / max(len(y_coords) - 1, 1)

        top_left_x = x_coords[0] - (x_res / 2.0)
        top_left_y = y_coords[0] - (y_res / 2.0)

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(
            str(output_path),
            cols,
            rows,
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"]
        )
        ds.SetGeoTransform((float(top_left_x), float(x_res), 0.0, float(top_left_y), 0.0, float(y_res)))

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ds.SetProjection(srs.ExportToWkt())

        band = ds.GetRasterBand(1)
        band.WriteArray(array)
        band.SetNoDataValue(0.0)
        band.FlushCache()
        ds = None

    def _download_h5_fallback(self, url: str, dest_path: Path) -> Optional[Path]:
        """Downloads HDF5 archive from NASA Earthdata URS using standard HTTP Basic Auth + CookieJar."""
        try:
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            if self.username and self.password:
                password_mgr.add_password(None, "https://urs.earthdata.nasa.gov", self.username, self.password)
                password_mgr.add_password(None, url, self.username, self.password)
            cookie_jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.HTTPBasicAuthHandler(password_mgr),
                urllib.request.HTTPCookieProcessor(cookie_jar)
            )
            req = urllib.request.Request(url, headers={"User-Agent": "AIML-CropMapper-Cloud/2.0"})
            with opener.open(req, timeout=300) as resp, open(dest_path, "wb") as out_f:
                shutil.copyfileobj(resp, out_f, length=16 * 1024 * 1024)
            return dest_path if dest_path.exists() and dest_path.stat().st_size > 1024 else None
        except Exception as e:
            logging.error(f"HTTP download failed for {url}: {e}")
            return None

    def _read_h5_grids(self, h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Reads (raw_hh, raw_hv, x_coords, y_coords, epsg) from NISAR L2 GCOV HDF5 file.
        Uses `h5py` if installed, or falls back to GDAL's native HDF5 subdataset driver.
        """
        if HAS_H5PY:
            with h5py.File(h5_path, "r") as h5_file:
                grid_group = None
                for candidate_path in [
                    "science/LSAR/GCOV/grids/frequencyA",
                    "science/LSAR/GCOV/grids/frequencyB",
                    "GCOV/grids/frequencyA",
                    "grids/frequencyA"
                ]:
                    if candidate_path in h5_file:
                        grid_group = h5_file[candidate_path]
                        break
                if grid_group is None:
                    raise ValueError(f"Could not locate GCOV frequency grid group in {h5_path.name}")

                x_coords = grid_group["xCoordinates"][:]
                y_coords = grid_group["yCoordinates"][:]

                epsg = 4326
                if "projection" in grid_group:
                    proj_val = grid_group["projection"][()]
                    if isinstance(proj_val, (int, np.integer)):
                        epsg = int(proj_val)
                    elif hasattr(grid_group["projection"], "attrs") and "epsg_code" in grid_group["projection"].attrs:
                        epsg = int(grid_group["projection"].attrs["epsg_code"])

                hh_name = "HHHH" if "HHHH" in grid_group else ("VVVV" if "VVVV" in grid_group else None)
                hv_name = "HVHV" if "HVHV" in grid_group else ("VHVH" if "VHVH" in grid_group else None)
                if hh_name is None or hv_name is None:
                    raise KeyError(f"Could not find dual-pol channels (HHHH/HVHV or VVVV/VHVH) in {h5_path.name}")

                raw_hh = grid_group[hh_name][:].astype(np.float32)
                raw_hv = grid_group[hv_name][:].astype(np.float32)
                return raw_hh, raw_hv, x_coords, y_coords, epsg

        # GDAL HDF5 Subdataset Fallback
        ds_hh = None
        ds_hv = None
        for freq in ["frequencyA", "frequencyB"]:
            for hh_c, hv_c in [("HHHH", "HVHV"), ("VVVV", "VHVH")]:
                sub_hh = f'HDF5:"{h5_path}"://science/LSAR/GCOV/grids/{freq}/{hh_c}'
                sub_hv = f'HDF5:"{h5_path}"://science/LSAR/GCOV/grids/{freq}/{hv_c}'
                ds_hh = gdal.Open(sub_hh, gdal.GA_ReadOnly)
                ds_hv = gdal.Open(sub_hv, gdal.GA_ReadOnly)
                if ds_hh is not None and ds_hv is not None:
                    break
            if ds_hh is not None and ds_hv is not None:
                break

        if ds_hh is None or ds_hv is None:
            raise RuntimeError(f"Neither h5py nor GDAL HDF5 driver could open GCOV grids in {h5_path.name}")

        raw_hh = ds_hh.ReadAsArray().astype(np.float32)
        raw_hv = ds_hv.ReadAsArray().astype(np.float32)
        gt = ds_hh.GetGeoTransform()
        cols, rows = ds_hh.RasterXSize, ds_hh.RasterYSize
        proj_wkt = ds_hh.GetProjection()
        ds_hh = None
        ds_hv = None

        epsg = 3857
        if proj_wkt:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj_wkt)
            epsg = int(srs.GetAttrValue("AUTHORITY", 1) or 3857)

        x_coords = gt[0] + (np.arange(cols, dtype=np.float64) + 0.5) * gt[1]
        y_coords = gt[3] + (np.arange(rows, dtype=np.float64) + 0.5) * gt[5]
        return raw_hh, raw_hv, x_coords, y_coords, epsg

    def process(self) -> List[Path]:
        title = self.product.get("title", "nisar_scene")
        asf_prod = self.product.get("asf_product")
        local_h5 = self.product.get("local_h5")

        scene_out_dir = self.temp_dir / title
        expected_layers = ["HHHH_gamma0.tif", "HVHV_gamma0.tif", "RVI_L.tif", "ratio_HH_HV.tif"]
        existing_tifs = [scene_out_dir / l for l in expected_layers if (scene_out_dir / l).exists()]
        if len(existing_tifs) == len(expected_layers):
            logging.info(f"Using already extracted polarimetric layers for {title}")
            return existing_tifs

        h5_path = None
        is_local_archive = False

        try:
            if local_h5 and Path(local_h5).exists():
                h5_path = Path(local_h5)
                is_local_archive = True
                logging.info(f"Reading local NISAR HDF5 archive: {h5_path.name}")
            else:
                logging.info(f"Downloading NISAR granule: {title}")
                if asf_prod is not None:
                    download_results = asf_prod.download(path=str(self.temp_dir), session=self.session)
                    if download_results:
                        h5_path = Path(download_results)
                if (h5_path is None or not h5_path.exists()) and self.product.get("url"):
                    target_h5 = self.temp_dir / f"{title}.h5"
                    h5_path = self._download_h5_fallback(self.product["url"], target_h5)
                if h5_path is None or not h5_path.exists():
                    candidates = list(self.temp_dir.glob(f"*{title}*.h5"))
                    if candidates:
                        h5_path = candidates[0]

            if not h5_path or not h5_path.exists():
                logging.error(f"Failed to locate or download H5 file for {title}")
                return []

            raw_hh, raw_hv, x_coords, y_coords, epsg = self._read_h5_grids(h5_path)

            swath_footprint = (~np.isnan(raw_hh)) & (~np.isnan(raw_hv))
            if np.any(swath_footprint):
                swath_filled = ndimage.binary_fill_holes(swath_footprint)
                if self.border_trim_pixels > 0:
                    horiz_struct = np.ones((1, 3), dtype=bool)
                    swath_boundary = ndimage.binary_erosion(
                        swath_filled, structure=horiz_struct, iterations=self.border_trim_pixels
                    )
                else:
                    swath_boundary = swath_filled

                valid_signal = swath_boundary & (~np.isnan(raw_hh)) & (~np.isnan(raw_hv)) & ((raw_hh > 1e-7) | (raw_hv > 1e-7))
                arr_hh = np.where(valid_signal, np.maximum(raw_hh, 1e-6), 0.0).astype(np.float32)
                arr_hv = np.where(valid_signal, np.maximum(raw_hv, 1e-6), 0.0).astype(np.float32)

                nan_internal = swath_boundary & (np.isnan(raw_hh) | np.isnan(raw_hv))
                if np.any(nan_internal) and np.any(valid_signal):
                    fill_idx = ndimage.distance_transform_edt(nan_internal, return_distances=False, return_indices=True)
                    arr_hh[nan_internal] = arr_hh[tuple(fill_idx)][nan_internal]
                    arr_hv[nan_internal] = arr_hv[tuple(fill_idx)][nan_internal]
            else:
                arr_hh = np.zeros_like(raw_hh, dtype=np.float32)
                arr_hv = np.zeros_like(raw_hv, dtype=np.float32)

            arr_hh = np.clip(np.nan_to_num(arr_hh, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 10.0)
            arr_hv = np.clip(np.nan_to_num(arr_hv, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 10.0)

            arr_hh = self._detaper_range_edges(arr_hh, W=265, min_gain=0.015)
            arr_hv = self._detaper_range_edges(arr_hv, W=265, min_gain=0.015)

            sum_pol = arr_hh + arr_hv
            arr_rvi = np.zeros_like(arr_hh, dtype=np.float32)
            mask_valid = sum_pol > 1e-6
            arr_rvi[mask_valid] = (4.0 * arr_hv[mask_valid]) / sum_pol[mask_valid]
            arr_rvi = np.clip(arr_rvi, 0.0, 1.5)

            arr_ratio = np.zeros_like(arr_hh, dtype=np.float32)
            mask_hv = arr_hv > 1e-6
            arr_ratio[mask_hv] = arr_hh[mask_hv] / arr_hv[mask_hv]
            arr_ratio = np.clip(arr_ratio, 0.0, 50.0)

            scene_out_dir.mkdir(parents=True, exist_ok=True)
            layers = [
                ("HHHH_gamma0.tif", arr_hh),
                ("HVHV_gamma0.tif", arr_hv),
                ("RVI_L.tif", arr_rvi),
                ("ratio_HH_HV.tif", arr_ratio)
            ]

            out_tifs = []
            for layer_name, layer_arr in layers:
                out_tif = scene_out_dir / layer_name
                self._save_geotiff(layer_arr, out_tif, x_coords, y_coords, epsg)
                out_tifs.append(out_tif)

            if not is_local_archive and not self.keep_intermediate_h5 and h5_path and h5_path.exists():
                try:
                    h5_path.unlink()
                except Exception:
                    pass

            return out_tifs
        except Exception as e:
            logging.error(f"Error extracting NISAR GCOV layers for {title}: {e}")
            return []
