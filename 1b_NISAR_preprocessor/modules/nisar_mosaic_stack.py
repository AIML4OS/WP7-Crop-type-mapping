"""
================================================================================
AIML CropMapper Cloud - NISAR Seamless Multi-Track Mosaicking & BigTIFF Stacker
================================================================================
Performs:
  1. Along-track frame slit healing (1D vertical morphological closing).
  2. 2D along-track adaptive range-edge detection & antenna roll-off de-tapering.
  3. Cross-seam radiometric harmonization & texture-preserving gap bridging.
  4. C^1 inverse-distance-squared feathering across overlapping swaths.
  5. Reprojection and clipping to EPSG:3857 at 10.0 m resolution aligned
     pixel-for-pixel to the Sentinel-1 & Sentinel-2 reference grid.
  6. Multi-temporal BigTIFF stacking (`COMPRESS=DEFLATE, TILED=YES, BIGTIFF=YES`)
     with external pyramid overviews `[2, 4, 8, 16, 32, 64]` and descriptive
     band metadata (`NISAR_HHHH_dB_19Jun2026`, `NISAR_HVHV_dB_19Jun2026`,
     `NISAR_RVI_L_19Jun2026`, `NISAR_RATIO_19Jun2026`).
================================================================================
"""

import datetime
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal, ogr, osr
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    uniform_filter1d
)


class MosaicNISAR:
    """
    Mosaics and clips NISAR polarimetric layer tiles into a target EPSG:3857 (10.0 m)
    Sentinel-1 reference grid (or country/belt polygon grid).
    """

    BAND_SPECS = [
        ("HHHH_gamma0.tif", ["HHHH_gamma0.tif", "*_HHHH*.tif"]),
        ("HVHV_gamma0.tif", ["HVHV_gamma0.tif", "*_HVHV*.tif"]),
        ("RVI_L.tif", ["RVI_L.tif", "*_RVI*.tif"]),
        ("ratio_HH_HV.tif", ["ratio_HH_HV.tif", "*_ratio*.tif"])
    ]

    def __init__(
        self,
        input_scene_dirs: List[Path],
        output_mosaic_dir: Path,
        target_grid: Dict,
        cutline_shp: Optional[Path] = None,
        country_code: Optional[str] = None,
        pixel_resolution: float = 10.0,
        target_epsg: int = 3857,
        blend_width_pixels: int = 350
    ):
        self.input_scene_dirs = [Path(p) for p in input_scene_dirs]
        self.output_mosaic_dir = Path(output_mosaic_dir)
        self.target_grid = target_grid
        self.cutline_shp = Path(cutline_shp) if cutline_shp else None
        self.country_code = country_code.upper() if country_code else None
        self.pixel_resolution = float(target_grid.get("x_res", pixel_resolution))
        self.target_epsg = int(target_grid.get("epsg", target_epsg))
        self.blend_width_pixels = int(blend_width_pixels)
        self.output_mosaic_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_track_id(scene_dir: Path) -> str:
        parts = scene_dir.name.split("_")
        if len(parts) >= 6 and parts[5].isdigit():
            return f"track_{int(parts[5]):03d}"
        return scene_dir.name

    def _find_band_file(self, scene_dir: Path, candidate_patterns: List[str]) -> Optional[Path]:
        for pat in candidate_patterns:
            if "*" in pat:
                matches = list(scene_dir.glob(pat))
                if matches:
                    return matches[0]
            else:
                cand = scene_dir / pat
                if cand.exists():
                    return cand
        return None

    @staticmethod
    def _detaper_and_find_range_edges(
        arr: np.ndarray,
        is_power_band: bool = True,
        W: int = 380,
        min_gain: float = 0.015
    ) -> Dict[str, np.ndarray]:
        out = arr.copy()
        rows, cols = out.shape
        ref_w = 160
        edges = {
            "west": np.full(rows, -1, dtype=np.int32),
            "east": np.full(rows, -1, dtype=np.int32)
        }

        if rows < 80 or cols < W + ref_w + 20:
            return {"arr": out, "west": edges["west"], "east": edges["east"]}

        for side in ("west", "east"):
            mask = out > 1e-6
            has_valid = np.any(mask, axis=1)
            first_x = np.argmax(mask, axis=1)
            last_x = cols - 1 - np.argmax(mask[:, ::-1], axis=1)
            span = last_x - first_x

            ex_raw = first_x if side == "west" else last_x
            valid_mask = has_valid & (span > W + ref_w + 20) & (ex_raw > 10) & (ex_raw < cols - 10)
            edge_x = np.where(valid_mask, ex_raw, -1).astype(np.int32)

            valid_r = np.where(edge_x >= 0)[0]
            if len(valid_r) < 80:
                continue

            smooth_sz = min(160, max(20, len(valid_r) // 3))
            ex_smooth = uniform_filter1d(edge_x[valid_r].astype(np.float32), size=smooth_sz)
            dx_dr = np.gradient(ex_smooth, valid_r.astype(np.float32))
            is_range = (np.abs(dx_dr) > 0.15) & (np.abs(dx_dr) < 0.85)
            for r_idx, ok in zip(valid_r, is_range):
                if not ok:
                    edge_x[r_idx] = -1

            range_r = np.where(edge_x >= 0)[0]
            if len(range_r) < 60:
                continue

            if not is_power_band:
                trim_px = 24
                ex_r = edge_x[range_r]
                if side == "west":
                    in_b = (ex_r + trim_px) < cols
                    rr, er = range_r[in_b], ex_r[in_b]
                    if len(rr) > 0:
                        d_t = np.arange(trim_px, dtype=np.int32)[None, :]
                        out[rr[:, None], er[:, None] + d_t] = 0.0
                        edge_x[rr] = er + trim_px
                else:
                    in_b = (ex_r - trim_px) >= 0
                    rr, er = range_r[in_b], ex_r[in_b]
                    if len(rr) > 0:
                        d_t = np.arange(trim_px, dtype=np.int32)[None, :]
                        out[rr[:, None], er[:, None] - d_t] = 0.0
                        edge_x[rr] = er - trim_px
                edges[side] = edge_x
                continue

            ex_r = edge_x[range_r]
            in_bounds = ((ex_r + W + ref_w) < cols) if side == "west" else ((ex_r - W - ref_w) >= 0)
            rr, er = range_r[in_bounds], ex_r[in_bounds]
            if len(rr) < 60:
                edges[side] = edge_x
                continue

            d_idx = np.arange(W + ref_w, dtype=np.int32)[None, :]
            strips = out[rr[:, None], er[:, None] + d_idx] if side == "west" else out[rr[:, None], er[:, None] - d_idx]

            ref_slices = strips[:, W: W + ref_w]
            ref_vals = np.where(ref_slices > 1e-5, ref_slices, np.nan)
            valid_ref_cnt = np.sum(~np.isnan(ref_vals), axis=1)
            ref_med = np.nanmedian(ref_vals, axis=1)
            good_rows = (valid_ref_cnt > 40) & (ref_med > 1e-5)
            if np.sum(good_rows) < 40:
                edges[side] = edge_x
                continue

            ratio_mat = np.full((rows, W), np.nan, dtype=np.float32)
            ratio_mat[rr[good_rows], :] = strips[good_rows, :W] / ref_med[good_rows, None]

            win_half = 320
            step = 100
            ctrl_rows, ctrl_g = [], []
            for r_center in range(0, rows, step):
                r0 = max(0, r_center - win_half)
                r1 = min(rows, r_center + win_half)
                sub = ratio_mat[r0:r1, :]
                if np.max(np.sum(~np.isnan(sub), axis=0)) >= 30:
                    g_loc = np.nanmedian(sub, axis=0)
                    g_loc = np.where(np.isnan(g_loc), 1.0, g_loc)
                    g_loc = np.clip(np.maximum.accumulate(g_loc), 1e-4, 1.0)
                    ctrl_rows.append(r_center)
                    ctrl_g.append(g_loc)

            if not ctrl_rows:
                edges[side] = edge_x
                continue

            ctrl_rows_arr = np.array(ctrl_rows, dtype=np.float32)
            ctrl_g_arr = np.stack(ctrl_g, axis=0)
            r_float = rr.astype(np.float32)
            g_sub = np.empty((len(rr), W), dtype=np.float32)
            for d in range(W):
                g_sub[:, d] = np.interp(r_float, ctrl_rows_arr, ctrl_g_arr[:, d])
            g_sub = np.maximum.accumulate(g_sub, axis=1)

            d_w = np.arange(W, dtype=np.int32)[None, :]
            cols_idx = (er[:, None] + d_w) if side == "west" else (er[:, None] - d_w)

            cur_vals = out[rr[:, None], cols_idx]
            low_snr = (g_sub < min_gain) & (d_w < 24)
            corrected = np.where(low_snr, 0.0, cur_vals / g_sub)
            out[rr[:, None], cols_idx] = np.clip(corrected, 0.0, 10.0)

            trim_counts = np.sum(low_snr, axis=1)
            edge_x[rr] = (er + trim_counts) if side == "west" else (er - trim_counts)
            edges[side] = edge_x

        return {"arr": out, "west": edges["west"], "east": edges["east"]}

    @staticmethod
    def _harmonize_and_bridge_tracks(
        track_data: List[Dict[str, np.ndarray]],
        rows: int,
        cols: int,
        max_val: float = 10.0
    ):
        n_tracks = len(track_data)
        if n_tracks < 2:
            return

        for i in range(n_tracks):
            for j in range(n_tracks):
                if i == j:
                    continue
                ex_east = track_data[i]["east"]
                ex_west = track_data[j]["west"]
                both_valid = (ex_east > 0) & (ex_west > 0)
                if np.sum(both_valid) < 40:
                    continue

                ov = ex_east[both_valid] - ex_west[both_valid]
                if np.median(ov) < -160 or np.median(ov) > 1200:
                    continue

                arr_l = track_data[i]["arr"]
                arr_r = track_data[j]["arr"]

                m_left = np.full(rows, np.nan, dtype=np.float32)
                m_right = np.full(rows, np.nan, dtype=np.float32)
                valid_rows = np.where(both_valid)[0]

                for r in valid_rows:
                    l_end = ex_east[r]
                    r_start = ex_west[r]
                    if l_end - r_start > 1400 or l_end - r_start < -160:
                        continue
                    l_sl = arr_l[r, max(0, l_end - 360): max(0, l_end - 20)]
                    r_sl = arr_r[r, min(cols, r_start + 20): min(cols, r_start + 360)]
                    lv = l_sl[l_sl > 1e-5]
                    rv = r_sl[r_sl > 1e-5]
                    if len(lv) > 40 and len(rv) > 40:
                        m_left[r] = np.median(lv)
                        m_right[r] = np.median(rv)

                valid_m = ~np.isnan(m_left)
                if np.sum(valid_m) >= 30:
                    all_r = np.arange(rows, dtype=np.float32)
                    m_l_s = uniform_filter1d(np.interp(all_r, all_r[valid_m], m_left[valid_m]), size=480)
                    m_r_s = uniform_filter1d(np.interp(all_r, all_r[valid_m], m_right[valid_m]), size=480)
                    m_mid = np.sqrt(m_l_s * m_r_s)

                    H = 640
                    for r in valid_rows:
                        l_end = ex_east[r]
                        r_start = ex_west[r]
                        if l_end - r_start > 1400 or l_end - r_start < -160:
                            continue
                        gain_l = np.clip(m_mid[r] / max(m_l_s[r], 1e-5), 0.70, 1.45)
                        gain_r = np.clip(m_mid[r] / max(m_r_s[r], 1e-5), 0.70, 1.45)

                        x1_l = l_end + 1
                        x0_l = max(0, x1_l - H)
                        if x1_l > x0_l + 10:
                            u = np.linspace(0.0, 1.0, x1_l - x0_l, dtype=np.float32)
                            w_ramp = 0.5 - 0.5 * np.cos(np.pi * u)
                            arr_l[r, x0_l:x1_l] = np.clip(
                                arr_l[r, x0_l:x1_l] * (1.0 + (gain_l - 1.0) * w_ramp), 0.0, max_val
                            )

                        x0_r = r_start
                        x1_r = min(cols, x0_r + H)
                        if x1_r > x0_r + 10:
                            u = np.linspace(0.0, 1.0, x1_r - x0_r, dtype=np.float32)
                            w_ramp = 0.5 - 0.5 * np.cos(np.pi * u)
                            arr_r[r, x0_r:x1_r] = np.clip(
                                arr_r[r, x0_r:x1_r] * (gain_r + (1.0 - gain_r) * w_ramp), 0.0, max_val
                            )

                bridge_margin = 64
                for r in valid_rows:
                    l_end = ex_east[r]
                    r_start = ex_west[r]
                    ov_r = l_end - r_start
                    if -140 <= ov_r < bridge_margin:
                        needed = (bridge_margin - ov_r) // 2 + 16
                        src_l = arr_l[r, max(0, l_end - needed - 8): max(0, l_end - 8)][::-1]
                        if len(src_l) > 0 and l_end + 1 + len(src_l) < cols:
                            arr_l[r, l_end + 1: l_end + 1 + len(src_l)] = src_l
                        src_r = arr_r[r, min(cols, r_start + 8): min(cols, r_start + 8 + needed)][::-1]
                        if len(src_r) > 0 and r_start - len(src_r) >= 0:
                            arr_r[r, r_start - len(src_r): r_start] = src_r

    @staticmethod
    def _heal_along_track_slits(arr: np.ndarray, max_gap: int = 64) -> np.ndarray:
        valid = arr > 1e-6
        if not np.any(valid):
            return arr

        closed = binary_closing(valid, structure=np.ones((max_gap, 1), dtype=bool))
        slits = closed & (~valid)
        if not np.any(slits):
            return arr

        ys, xs = np.where(slits)
        pad = max_gap + 16
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(arr.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(arr.shape[1], int(xs.max()) + pad + 1)

        out = arr.copy()
        sub_out = out[y0:y1, x0:x1]
        sub_valid = valid[y0:y1, x0:x1]
        sub_closed = closed[y0:y1, x0:x1]
        sub_clean = binary_erosion(sub_valid, structure=np.ones((7, 1), dtype=bool))
        sub_fill = sub_closed & (~sub_clean)

        if np.any(sub_clean) and np.any(sub_fill):
            idx = distance_transform_edt(~sub_clean, return_distances=False, return_indices=True)
            sub_out[sub_fill] = sub_out[tuple(idx)][sub_fill]

        return out

    def _blend_block(
        self,
        track_arrs: List[np.ndarray],
        rows: int,
        cols: int,
        is_power_band: bool = True,
        max_val: float = 10.0
    ) -> np.ndarray:
        healed_tracks = [self._heal_along_track_slits(arr, max_gap=64) for arr in track_arrs]
        track_data = [
            self._detaper_and_find_range_edges(arr, is_power_band=is_power_band, W=380, min_gain=0.015)
            for arr in healed_tracks
        ]
        self._harmonize_and_bridge_tracks(track_data, rows, cols, max_val=max_val)

        clean_arrs = [td["arr"] for td in track_data]
        clean_masks = [(a > 1e-6) for a in clean_arrs]

        sum_masks = sum(m.astype(np.uint8) for m in clean_masks)
        out = np.zeros((rows, cols), dtype=np.float32)

        for arr, m in zip(clean_arrs, clean_masks):
            single = m & (sum_masks == 1)
            out[single] = arr[single]

        overlap = sum_masks > 1
        if np.any(overlap):
            ys, xs = np.where(overlap)
            pad = self.blend_width_pixels
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(cols, int(xs.max()) + pad + 1)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(rows, int(ys.max()) + pad + 1)

            sub_overlap = overlap[y0:y1, x0:x1]
            sub_arrs = [arr[y0:y1, x0:x1] for arr in clean_arrs]
            sub_masks = [m[y0:y1, x0:x1] for m in clean_masks]

            sub_dists = []
            for sm in sub_masks:
                if np.any(sm):
                    fp = binary_fill_holes(sm)
                    d = distance_transform_edt(fp).astype(np.float32)
                else:
                    d = np.zeros_like(sm, dtype=np.float32)
                sub_dists.append(d)

            w_list = [d * d for d in sub_dists]
            w_sum = sum(w_list)
            sub_blended = np.zeros_like(sub_arrs[0], dtype=np.float32)
            valid_w = w_sum > 0
            if np.any(valid_w):
                num = sum(a * w for a, w in zip(sub_arrs, w_list))
                sub_blended[valid_w] = num[valid_w] / w_sum[valid_w]

            out[y0:y1, x0:x1][sub_overlap] = sub_blended[sub_overlap]

        out = self._heal_along_track_slits(out, max_gap=64)
        valid_out = out > 1e-6
        if np.any(valid_out):
            filled_out = binary_fill_holes(valid_out)
            internal_holes = filled_out & (~valid_out)
            if np.any(internal_holes):
                ys, xs = np.where(internal_holes)
                pad = 64
                y0 = max(0, int(ys.min()) - pad)
                y1 = min(rows, int(ys.max()) + pad + 1)
                x0 = max(0, int(xs.min()) - pad)
                x1 = min(cols, int(xs.max()) + pad + 1)
                sub_out = out[y0:y1, x0:x1]
                sub_valid = valid_out[y0:y1, x0:x1]
                sub_filled = filled_out[y0:y1, x0:x1]
                sub_clean = binary_erosion(sub_valid, iterations=4)
                sub_need = sub_filled & (~sub_clean)
                if np.any(sub_clean) and np.any(sub_need):
                    idx = distance_transform_edt(~sub_clean, return_distances=False, return_indices=True)
                    sub_out[sub_need] = sub_out[tuple(idx)][sub_need]

        return np.clip(out, 0.0, max_val)

    def _rasterize_aoi_mask(self, cols: int, rows: int, gt: tuple, proj: str) -> Optional[gdal.Dataset]:
        if not self.cutline_shp or not self.cutline_shp.exists():
            return None
        mem_drv = gdal.GetDriverByName("MEM")
        mask_ds = mem_drv.Create("", cols, rows, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(gt)
        mask_ds.SetProjection(proj)
        band = mask_ds.GetRasterBand(1)
        band.Fill(0)

        shp_ds = ogr.Open(str(self.cutline_shp))
        if shp_ds is not None:
            layer = shp_ds.GetLayer()
            if self.country_code and "NUTS_RG" in self.cutline_shp.name:
                layer.SetAttributeFilter(f"CNTR_CODE = '{self.country_code}'")
            gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
            shp_ds = None
        band.FlushCache()
        return mask_ds

    def run_mosaic(self) -> List[Path]:
        target_extent = self.target_grid["extent"]
        cols, rows = self.target_grid["size"]
        dst_srs_str = f"EPSG:{self.target_epsg}"

        warp_unclipped_options = dict(
            dstSRS=dst_srs_str,
            outputBounds=target_extent,
            width=cols,
            height=rows,
            srcNodata=0.0,
            dstNodata=0.0,
            multithread=True,
            warpOptions=["NUM_THREADS=ALL_CPUS", "UNIFIED_SRC_NODATA=YES"],
            resampleAlg="bilinear"
        )

        scenes_by_track: Dict[str, List[Path]] = {}
        for s_dir in self.input_scene_dirs:
            t_id = self._get_track_id(s_dir)
            scenes_by_track.setdefault(t_id, []).append(s_dir)

        gt = (
            target_extent[0],
            (target_extent[2] - target_extent[0]) / max(cols, 1),
            0.0,
            target_extent[3],
            0.0,
            -(target_extent[3] - target_extent[1]) / max(rows, 1)
        )

        srs_tgt = osr.SpatialReference()
        srs_tgt.ImportFromEPSG(self.target_epsg)
        srs_tgt.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        proj = srs_tgt.ExportToWkt()

        aoi_mask_ds = self._rasterize_aoi_mask(cols, rows, gt, proj)
        aoi_mask_band = aoi_mask_ds.GetRasterBand(1) if aoi_mask_ds else None

        created_files = []
        for out_band_name, candidate_patterns in self.BAND_SPECS:
            is_power_band = out_band_name in ("HHHH_gamma0.tif", "HVHV_gamma0.tif")
            max_val = 10.0 if is_power_band else (1.5 if out_band_name == "RVI_L.tif" else 50.0)

            track_sources: Dict[str, List[str]] = {}
            all_sources = []
            for t_id, s_dirs in scenes_by_track.items():
                t_files = []
                for s_dir in s_dirs:
                    f = self._find_band_file(s_dir, candidate_patterns)
                    if f and f.exists():
                        t_files.append(str(f))
                        all_sources.append(str(f))
                if t_files:
                    track_sources[t_id] = t_files

            if not track_sources:
                continue

            output_file = self.output_mosaic_dir / out_band_name
            temp_vrts = []
            track_datasets = []
            out_ds = None
            out_band = None
            try:
                vrt_opts = gdal.WarpOptions(**warp_unclipped_options, format="VRT")
                for t_id, sources in sorted(track_sources.items()):
                    frame_vrts = []
                    frame_warp_opts = gdal.WarpOptions(dstSRS=dst_srs_str, format="VRT", srcNodata=0.0, dstNodata=0.0)
                    for i, s_file in enumerate(sources):
                        f_vrt_p = self.output_mosaic_dir / f"_tmp_{t_id}_{out_band_name}_f{i}.vrt"
                        gdal.Warp(str(f_vrt_p), s_file, options=frame_warp_opts)
                        frame_vrts.append(str(f_vrt_p))
                        temp_vrts.append(f_vrt_p)

                    raw_vrt_path = self.output_mosaic_dir / f"_tmp_{t_id}_{out_band_name}_raw.vrt"
                    gdal.BuildVRT(str(raw_vrt_path), frame_vrts, srcNodata=0.0, VRTNodata=0.0)
                    temp_vrts.append(raw_vrt_path)

                    vrt_path = self.output_mosaic_dir / f"_tmp_{t_id}_{out_band_name}.vrt"
                    t_ds = gdal.Warp(str(vrt_path), str(raw_vrt_path), options=vrt_opts)
                    if t_ds is not None:
                        temp_vrts.append(vrt_path)
                        track_datasets.append(t_ds)

                driver = gdal.GetDriverByName("GTiff")
                out_ds = driver.Create(
                    str(output_file),
                    cols, rows, 1,
                    gdal.GDT_Float32,
                    options=["COMPRESS=DEFLATE", "BIGTIFF=YES", "TILED=YES", "NUM_THREADS=ALL_CPUS"]
                )
                out_ds.SetGeoTransform(gt)
                out_ds.SetProjection(proj)
                out_band = out_ds.GetRasterBand(1)
                out_band.SetNoDataValue(0.0)

                block_height = 4096
                halo = 512
                for y in range(0, rows, block_height):
                    cur_rows = min(block_height, rows - y)
                    y_read0 = max(0, y - halo)
                    y_read1 = min(rows, y + cur_rows + halo)
                    read_rows = y_read1 - y_read0
                    core_start = y - y_read0
                    core_end = core_start + cur_rows

                    track_arrs = []
                    for t_ds in track_datasets:
                        b_arr = t_ds.GetRasterBand(1).ReadAsArray(0, y_read0, cols, read_rows)
                        if b_arr is None:
                            b_arr = np.zeros((read_rows, cols), dtype=np.float32)
                        else:
                            b_arr = np.nan_to_num(b_arr, nan=0.0, posinf=0.0, neginf=0.0)
                        track_arrs.append(b_arr)

                    blended_ext = self._blend_block(
                        track_arrs, read_rows, cols,
                        is_power_band=is_power_band, max_val=max_val
                    )
                    blended_core = blended_ext[core_start:core_end, :]

                    if aoi_mask_band is not None:
                        b_mask = aoi_mask_band.ReadAsArray(0, y, cols, cur_rows)
                        if b_mask is not None:
                            blended_core[b_mask == 0] = 0.0

                    out_band.WriteArray(blended_core, 0, y)

                out_band.FlushCache()
                out_band = None
                out_ds = None
                created_files.append(output_file)
            except Exception as e:
                logging.error(f"Seamless NISAR mosaicking failed for {out_band_name} ({e}), running direct GDAL Warp fallback...")
                out_band = None
                out_ds = None
                fallback_opts = gdal.WarpOptions(
                    **warp_unclipped_options,
                    creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES", "TILED=YES"]
                )
                gdal.Warp(str(output_file), all_sources, options=fallback_opts)
                if output_file.exists():
                    created_files.append(output_file)
            finally:
                for ds in track_datasets:
                    ds = None
                for vrt_p in temp_vrts:
                    try:
                        if vrt_p.exists():
                            vrt_p.unlink()
                    except Exception:
                        pass

        aoi_mask_band = None
        aoi_mask_ds = None
        return created_files


def stack_nisar_timeseries(
    mosaic_base_dir: Path,
    output_stack_tif: Path,
    target_grid: Optional[Dict] = None,
    convert_gamma0_to_db: bool = True,
    build_overviews: bool = True,
    cleanup_mosaics: bool = False
) -> Optional[Path]:
    """
    Stacks all chronological NISAR cycle/date folders (`mosaic_base_dir/YYYYMMDD/`)
    into a single multi-temporal BigTIFF co-registered with Sentinel-1 and Sentinel-2.

    Band layout per date `t` (4 bands per date, total `4 * T_nisar` bands):
      - `NISAR_HHHH_dB_<DDMonYYYY>` (dB scale `10 * log10(gamma0_HH)`, NoData = 0.0)
      - `NISAR_HVHV_dB_<DDMonYYYY>` (dB scale `10 * log10(gamma0_HV)`, NoData = 0.0)
      - `NISAR_RVI_L_<DDMonYYYY>`   (L-band RVI in `[0.0, 1.5]`, NoData = 0.0)
      - `NISAR_RATIO_<DDMonYYYY>`   (HH/HV ratio in `[0.0, 50.0]`, NoData = 0.0)
    """
    if not mosaic_base_dir.exists():
        logging.warning(f"NISAR mosaic directory does not exist: {mosaic_base_dir}")
        return None

    date_dirs = sorted([
        d for d in mosaic_base_dir.iterdir()
        if d.is_dir() and len(d.name) == 8 and d.name.isdigit() and (d / "HHHH_gamma0.tif").exists()
    ])
    if not date_dirs:
        logging.warning(f"No valid NISAR date mosaic directories found in {mosaic_base_dir}")
        return None

    first_tif = date_dirs[0] / "HHHH_gamma0.tif"
    ds0 = gdal.Open(str(first_tif), gdal.GA_ReadOnly)
    if ds0 is None:
        return None
    cols = ds0.RasterXSize
    rows = ds0.RasterYSize
    gt = ds0.GetGeoTransform()
    proj = ds0.GetProjection()
    ds0 = None

    if target_grid and "size" in target_grid and "extent" in target_grid:
        t_cols, t_rows = target_grid["size"]
        t_ext = target_grid["extent"]
        t_epsg = int(target_grid.get("epsg", 3857))
        if t_cols > 0 and t_rows > 0:
            cols, rows = int(t_cols), int(t_rows)
            gt = (
                float(t_ext[0]),
                (float(t_ext[2]) - float(t_ext[0])) / max(cols, 1),
                0.0,
                float(t_ext[3]),
                0.0,
                -(float(t_ext[3]) - float(t_ext[1])) / max(rows, 1)
            )
            srs_tgt = osr.SpatialReference()
            srs_tgt.ImportFromEPSG(t_epsg)
            srs_tgt.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            proj = srs_tgt.ExportToWkt()

    layer_defs = [
        ("HHHH_gamma0.tif", "NISAR_HHHH_dB", True),
        ("HVHV_gamma0.tif", "NISAR_HVHV_dB", True),
        ("RVI_L.tif", "NISAR_RVI_L", False),
        ("ratio_HH_HV.tif", "NISAR_RATIO", False)
    ]

    total_bands = len(date_dirs) * len(layer_defs)
    output_stack_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_tif = output_stack_tif.with_suffix(".tmp.tif")
    if tmp_tif.exists():
        try:
            tmp_tif.unlink()
        except Exception:
            pass

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(tmp_tif),
        cols,
        rows,
        total_bands,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]
    )
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    band_idx = 1
    tile_h = 4096
    for d_dir in date_dirs:
        try:
            dt_obj = datetime.datetime.strptime(d_dir.name, "%Y%m%d").date()
            date_label = dt_obj.strftime("%d%b%Y")
        except Exception:
            date_label = d_dir.name

        for file_name, prefix, is_power in layer_defs:
            src_file = d_dir / file_name
            out_band = out_ds.GetRasterBand(band_idx)
            desc = f"{prefix}_{date_label}"
            out_band.SetDescription(desc)
            out_band.SetNoDataValue(0.0)

            if src_file.exists():
                src_ds = gdal.Open(str(src_file), gdal.GA_ReadOnly)
                if src_ds is not None:
                    s_cols, s_rows = src_ds.RasterXSize, src_ds.RasterYSize
                    s_gt = src_ds.GetGeoTransform()
                    same_grid = (
                        s_cols == cols and s_rows == rows and
                        abs(s_gt[0] - gt[0]) <= 5.0 and abs(s_gt[3] - gt[3]) <= 5.0
                    )
                    warped_ds = None
                    if not same_grid:
                        min_x = gt[0]
                        max_y = gt[3]
                        max_x = min_x + cols * gt[1]
                        min_y = max_y + rows * gt[5]
                        w_opts = gdal.WarpOptions(
                            format="VRT",
                            outputBounds=(min_x, min_y, max_x, max_y),
                            width=cols,
                            height=rows,
                            dstSRS=proj,
                            srcNodata=0.0,
                            dstNodata=0.0,
                            resampleAlg=gdal.GRA_Bilinear
                        )
                        warped_ds = gdal.Warp("", str(src_file), options=w_opts)
                    active_ds = warped_ds if warped_ds is not None else src_ds
                    src_b = active_ds.GetRasterBand(1)
                    for y in range(0, rows, tile_h):
                        cur_h = min(tile_h, rows - y)
                        arr = src_b.ReadAsArray(0, y, cols, cur_h)
                        if arr is None:
                            continue
                        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                        if is_power and convert_gamma0_to_db:
                            valid = arr > 1e-6
                            db_arr = np.zeros_like(arr, dtype=np.float32)
                            db_arr[valid] = np.clip(10.0 * np.log10(np.maximum(arr[valid], 1e-6)), -35.0, 10.0)
                            # Shift exact 0.0 dB to 0.001 dB so 0.0 strictly represents NoData
                            zero_db = valid & (np.abs(db_arr) < 1e-4)
                            db_arr[zero_db] = 1e-3
                            arr = db_arr
                        out_band.WriteArray(arr, 0, y)
                    warped_ds = None
                    src_ds = None
            out_band.FlushCache()
            band_idx += 1

    out_ds.FlushCache()
    if build_overviews:
        logging.info("Building multi-scale pyramid overviews [2, 4, 8, 16, 32, 64] on NISAR stack...")
        gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
        out_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32, 64])
    out_ds = None

    if output_stack_tif.exists():
        try:
            output_stack_tif.unlink()
        except Exception:
            pass
    tmp_tif.rename(output_stack_tif)

    if cleanup_mosaics and output_stack_tif.exists() and output_stack_tif.stat().st_size > 1024 * 1024:
        try:
            shutil.rmtree(mosaic_base_dir, ignore_errors=True)
        except Exception:
            pass

    logging.info(f"SUCCESS: NISAR Multi-Temporal Stack created at {output_stack_tif} ({total_bands} bands across {len(date_dirs)} dates)!")
    return output_stack_tif

