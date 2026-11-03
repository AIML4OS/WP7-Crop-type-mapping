"""
================================================================================
AIML CropMapper Cloud - NISAR End-to-End Processing Pipeline Coordinator
================================================================================
Coordinates the 3-stage NISAR L-band SAR (L2 GCOV) preprocessing workflow:
  - Stage 1: NASA ASF DAAC / Local `/eodata/NISAR` discovery, orbital cycle
             grouping, minimal-seam track selection, and HDF5 polarimetric
             layer extraction (`HHHH`, `HVHV`, `RVI_L`, `ratio_HH_HV`).
  - Stage 2: Seamless multi-track mosaicking (along-track slit healing, 2D
             adaptive range-edge de-tapering, cross-seam harmonization, and
             C^1 distance-squared feathering) in EPSG:3857 at 10.0 m.
  - Stage 3: Pixel-for-pixel co-registration to the Sentinel-1 & Sentinel-2
             reference grid and multi-temporal BigTIFF stacking with pyramid
             overviews in `workingDirs/<COUNTRY>/orbit_<ORBIT>/1_input_stacks/`.
================================================================================
"""

import datetime
from itertools import combinations
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from osgeo import ogr, osr

from nisar_asf_search import (
    AUX_DIR,
    BASE_DIR,
    FindNisarASF,
    get_s1_reference_grid,
    load_country_or_belt_geometry
)
from nisar_extract_gcov import DownloadAndExtractGCOV
from nisar_mosaic_stack import MosaicNISAR, stack_nisar_timeseries

try:
    import asf_search as asf
    HAS_ASF = True
except ImportError:
    asf = None
    HAS_ASF = False


class CalculateTrackNISAR:
    """
    Manages NISAR L2 GCOV download, extraction, cycle mosaicking, and BigTIFF
    stacking for a single Sentinel-1 orbit track or country AOI.
    """

    def __init__(
        self,
        list_to_download: List[Dict],
        target_grid: Dict,
        track_dir: Path,
        country_code: str,
        configuration: dict,
        session=None,
        cutline_shp: Optional[Path] = None
    ):
        self.configuration = configuration
        self.target_grid = target_grid
        self.track_dir = Path(track_dir)
        self.country_code = country_code.upper()
        self.session = session
        self.cutline_shp = cutline_shp

        self.dirs = {
            "temp": self.track_dir / "temp_nisar_extracted",
            "mosaic": self.track_dir / "nisar_cycle_mosaics",
            "stacks": self.track_dir / "1_input_stacks"
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self.images_by_period = self._group_images(list_to_download)

    def _get_aoi_geom_4326(self) -> Optional[ogr.Geometry]:
        if "geom_4326" in self.target_grid and self.target_grid["geom_4326"] is not None:
            return self.target_grid["geom_4326"].Clone()
        wkt = self.target_grid.get("wkt_4326")
        if wkt:
            return ogr.CreateGeometryFromWkt(wkt)
        return None

    def _filter_cycle_tracks(self, items: List, min_aoi_cov: float = 95.0) -> List[Dict]:
        """
        Selects the minimal, optimal set of tracks that covers the target AOI polygon.
        Strictly enforces topological contiguity (overlapping swaths within or immediately
        adjacent to the AOI) so disjoint or gap-separated tracks are never merged into an
        incomplete mosaic.
        Prioritizes:
        1. Highest AOI coverage (rounded to 2 decimal places).
        2. Fewest tracks (minimizes spatial seams and track boundaries).
        3. Shortest date span (maximizes temporal and phenological consistency).
        """
        if not items:
            return []

        by_track = {}
        for img_date, img in items:
            t = img.get("path_number")
            if t is None:
                parts = img.get("title", "").split("_")
                t = int(parts[5]) if len(parts) >= 6 and parts[5].isdigit() else str(t)
            by_track.setdefault(t, []).append((img_date, img))

        if len(by_track) <= 1:
            return [it[1] for it in items]

        try:
            aoi_geom = self._get_aoi_geom_4326()
            if aoi_geom is None or aoi_geom.GetArea() <= 0:
                best_t = max(by_track.keys(), key=lambda k: len(by_track[k]))
                return [it[1] for it in by_track[best_t]]
            aoi_area = aoi_geom.GetArea()
            # Buffer AOI by ~10 km (0.1 deg) to capture track intersections at/near AOI borders
            aoi_buf = aoi_geom.Buffer(0.1)

            t_geom = {}
            t_geom_aoi = {}
            t_dates = {}
            for t, t_list in by_track.items():
                u = ogr.Geometry(ogr.wkbMultiPolygon)
                dates = []
                for img_date, img in t_list:
                    asf_p = img.get("asf_product")
                    geom_dict = asf_p.geometry if (asf_p and hasattr(asf_p, "geometry")) else img.get("geometry")
                    if geom_dict:
                        g = ogr.CreateGeometryFromJson(json.dumps(geom_dict))
                        if g:
                            g_shrunk = g.Buffer(-0.035)
                            if g_shrunk and not g_shrunk.IsEmpty():
                                g = g_shrunk
                            u = u.Union(g)
                    dates.append(img_date)
                if u and not u.IsEmpty():
                    u = u.Buffer(0.0)
                t_geom[t] = u
                t_geom_aoi[t] = u.Intersection(aoi_buf) if (u and not u.IsEmpty() and aoi_buf) else ogr.Geometry(ogr.wkbMultiPolygon)
                t_dates[t] = (min(dates), max(dates)) if dates else (datetime.date.min, datetime.date.min)

            def _is_contiguous(c_tracks: Tuple) -> bool:
                if len(c_tracks) <= 1:
                    return True
                # Tracks must intersect each other within or immediately adjacent to the AOI (aoi_buf)
                visited = {c_tracks[0]}
                queue = [c_tracks[0]]
                while queue:
                    curr = queue.pop(0)
                    for other in c_tracks:
                        if other not in visited:
                            g_curr = t_geom_aoi.get(curr)
                            g_other = t_geom_aoi.get(other)
                            if g_curr and g_other and not g_curr.IsEmpty() and not g_other.IsEmpty() and g_curr.Intersects(g_other):
                                inter = g_curr.Intersection(g_other)
                                if inter and not inter.IsEmpty() and inter.GetArea() > 1e-5:
                                    visited.add(other)
                                    queue.append(other)
                return len(visited) == len(c_tracks)

            tracks = list(by_track.keys())
            combos = []
            for r in range(1, len(tracks) + 1):
                for combo in combinations(tracks, r):
                    if not _is_contiguous(combo):
                        continue
                    u_combo = ogr.Geometry(ogr.wkbMultiPolygon)
                    combo_dates = []
                    for t in combo:
                        u_combo = u_combo.Union(t_geom[t])
                        combo_dates.extend([t_dates[t][0], t_dates[t][1]])
                    inter = aoi_geom.Intersection(u_combo)
                    cov = (inter.GetArea() / aoi_area) * 100.0 if inter else 0.0
                    span_days = (max(combo_dates) - min(combo_dates)).days if combo_dates else 0
                    combos.append((round(cov, 2), len(combo), span_days, cov, combo))

            if not combos:
                single_tracks = []
                for t in tracks:
                    inter = aoi_geom.Intersection(t_geom[t])
                    cov = (inter.GetArea() / aoi_area) * 100.0 if inter else 0.0
                    single_tracks.append((cov, t))
                single_tracks.sort(key=lambda x: -x[0])
                best_t = single_tracks[0][1] if single_tracks else tracks[0]
                return [it[1] for it in by_track[best_t]]

            # Criteria ranking (adopting calculate_belts_nisar.py):
            # 1) Maximum coverage rounded to 2 decimal places (-x[0], descending)
            # 2) Fewest tracks (x[1], ascending): minimizes swath boundary seams
            # 3) Shortest date span (x[2], ascending): maximizes temporal/phenological consistency
            combos.sort(key=lambda x: (-x[0], x[1], x[2]))
            chosen = combos[0][4]

            kept_imgs = []
            for t in chosen:
                kept_imgs.extend([it[1] for it in by_track[t]])

            dropped_tracks = [str(t) for t in tracks if t not in chosen]
            if dropped_tracks:
                logging.info(
                    f"Selected optimal contiguous tracks: {list(chosen)} (coverage: {combos[0][0]:.2f}%) "
                    f"for {self.track_dir.name}. Filtered out tracks: {dropped_tracks}"
                )

            return kept_imgs
        except Exception as e:
            logging.warning(f"Could not calculate optimal NISAR track selection ({e}); falling back to single best track.")
            try:
                best_t = max(by_track.keys(), key=lambda k: len(by_track[k]))
                return [it[1] for it in by_track[best_t]]
            except Exception:
                return [it[1] for it in items]

    def _calculate_items_aoi_coverage(self, images: List[Dict]) -> float:
        if not images:
            return 0.0
        try:
            aoi_geom = self._get_aoi_geom_4326()
            if aoi_geom is None or aoi_geom.GetArea() <= 0:
                return 0.0
            aoi_area = aoi_geom.GetArea()

            u = ogr.Geometry(ogr.wkbMultiPolygon)
            has_any_geom = False
            for img in images:
                asf_p = img.get("asf_product")
                geom_dict = asf_p.geometry if (asf_p and hasattr(asf_p, "geometry")) else img.get("geometry")
                if geom_dict:
                    g = ogr.CreateGeometryFromJson(json.dumps(geom_dict))
                    if g:
                        has_any_geom = True
                        g_shrunk = g.Buffer(-0.035)
                        if g_shrunk and not g_shrunk.IsEmpty():
                            g = g_shrunk
                        u = u.Union(g)
            if not has_any_geom:
                return 0.0
            u = u.Buffer(0.0)
            inter = aoi_geom.Intersection(u)
            return (inter.GetArea() / aoi_area) * 100.0 if inter else 0.0
        except Exception as e:
            logging.warning(f"Could not calculate AOI coverage ({e}); reporting 0.0% coverage.")
            return 0.0

    def _group_images(self, raw_list: List[Dict]) -> Dict[str, Dict]:
        mode = str(self.configuration.get("grouping_mode", "cycle")).lower()
        cycle_days = int(self.configuration.get("cycle_days", 12))
        min_aoi_cov = float(
            self.configuration.get(
                "min_aoi_coverage_pct",
                self.configuration.get("min_belt_coverage_pct", 95.0)
            )
        )

        dedup_map = {}
        for image in raw_list:
            title = image.get("title", "")
            parts = title.split("_")
            if len(parts) >= 8 and parts[4].isdigit() and parts[5].isdigit() and parts[7].isdigit():
                granule_key = (int(parts[4]), int(parts[5]), int(parts[7]))
                prev = dedup_map.get(granule_key)
                if prev is None or title > prev.get("title", ""):
                    dedup_map[granule_key] = image
            else:
                dedup_map[title] = image

        valid_items = []
        for image in dedup_map.values():
            start_time = image.get("start_time")
            if not start_time:
                continue
            try:
                if isinstance(start_time, str):
                    date_str = start_time.split("T")[0]
                    img_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                elif isinstance(start_time, (datetime.date, datetime.datetime)):
                    img_date = start_time if isinstance(start_time, datetime.date) else start_time.date()
                else:
                    continue
                valid_items.append((img_date, image))
            except Exception:
                continue

        if not valid_items:
            return {}

        valid_items.sort(key=lambda x: (x[0], str(x[1].get("start_time", ""))))
        grouped_result = {}

        if mode == "cycle":
            cycle_buckets = {}
            for img_date, img in valid_items:
                title = img.get("title", "")
                parts = title.split("_")
                if len(parts) >= 6 and parts[4].isdigit():
                    c_idx = int(parts[4])
                else:
                    min_date = valid_items[0][0]
                    c_idx = (img_date - min_date).days // cycle_days
                cycle_buckets.setdefault(c_idx, []).append((img_date, img))

            for c_idx, items in sorted(cycle_buckets.items()):
                filtered_imgs = self._filter_cycle_tracks(items, min_aoi_cov)
                if not filtered_imgs:
                    logging.warning(f"Skipping Cycle {c_idx}: no contiguous tracks could be selected.")
                    continue

                cov = self._calculate_items_aoi_coverage(filtered_imgs)

                # Use acquisition dates of filtered images only
                filtered_dates = []
                for img in filtered_imgs:
                    st = img.get("start_time", "")
                    if isinstance(st, str) and "T" in st:
                        d = datetime.datetime.strptime(st.split("T")[0], "%Y-%m-%d").date()
                    elif isinstance(st, (datetime.date, datetime.datetime)):
                        d = st if isinstance(st, datetime.date) else st.date()
                    else:
                        continue
                    filtered_dates.append(d)

                filtered_dates = sorted(list(set(filtered_dates)))
                if not filtered_dates:
                    filtered_dates = sorted(list(set(it[0] for it in items)))

                folder_key = filtered_dates[0].strftime("%Y%m%d")
                dates_str = ", ".join(d.strftime("%Y-%m-%d") for d in filtered_dates)

                if cov < min_aoi_cov:
                    logging.warning(
                        f"Skipping Cycle {c_idx} [Folder {folder_key}]: AOI coverage is only {cov:.1f}% "
                        f"(below required {min_aoi_cov:.1f}% threshold for classification)."
                    )
                    continue

                grouped_result[folder_key] = {
                    "images": filtered_imgs,
                    "cycle_num": c_idx,
                    "coverage_pct": cov,
                    "description": f"Cycle {c_idx} [Folder {folder_key}] (coverage: {cov:.1f}%, dates: {dates_str})"
                }
        elif mode == "dekada":
            dekada_buckets = {}
            for img_date, img in valid_items:
                d_day = 1 if img_date.day <= 10 else (11 if img_date.day <= 20 else 21)
                dekada_key = f"{img_date.year:04d}{img_date.month:02d}{d_day:02d}"
                dekada_buckets.setdefault(dekada_key, []).append((img_date, img))

            for d_key, items in sorted(dekada_buckets.items()):
                imgs = [it[1] for it in items]
                cov = self._calculate_items_aoi_coverage(imgs)
                if cov < min_aoi_cov:
                    logging.warning(
                        f"Skipping Dekada {d_key}: AOI coverage is only {cov:.1f}% "
                        f"(below required {min_aoi_cov:.1f}% threshold for classification)."
                    )
                    continue
                unique_dates = sorted(list(set(it[0].strftime("%Y-%m-%d") for it in items)))
                grouped_result[d_key] = {
                    "images": imgs,
                    "coverage_pct": cov,
                    "description": f"Dekada {d_key} (coverage: {cov:.1f}%, dates: {', '.join(unique_dates)})"
                }
        else:
            date_buckets = {}
            for img_date, img in valid_items:
                d_key = img_date.strftime("%Y%m%d")
                date_buckets.setdefault(d_key, []).append(img)
            for d_key, imgs in sorted(date_buckets.items()):
                grouped_result[d_key] = {
                    "images": imgs,
                    "description": f"Date {d_key}"
                }

        logging.info(f"Grouped NISAR scenes into {len(grouped_result)} {mode} periods for {self.track_dir.name}")
        return grouped_result

    def _cleanup_intermediate_data(self):
        temp_dir = self.dirs["temp"]
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception:
                    pass

    def stage_1_and_2_extract_and_mosaic(self, stage: str = "A"):
        stage_u = (stage or "A").strip().upper()
        keep_h5 = bool(self.configuration.get("keep_intermediate_h5", False))
        pixel_res = float(self.configuration.get("pixel_resolution_meters", 10.0))
        target_epsg = int(self.configuration.get("target_epsg", 3857))
        border_trim_px = int(self.configuration.get("border_trim_pixels", 8))
        blend_width_px = int(self.configuration.get("blend_width_pixels", 350))
        overwrite = bool(self.configuration.get("overwrite_existing", False))
        username = self.configuration.get("earthdata_username")
        password = self.configuration.get("earthdata_password")

        token = getattr(self.session, "token", None) or self.configuration.get("earthdata_token")
        cookie_jar = getattr(self.session, "cookie_jar", None) or (AUX_DIR / "urs_cookies.txt")

        for period_key, period_info in self.images_by_period.items():
            date_str = period_key
            images = period_info["images"]
            desc = period_info.get("description", date_str)
            mosaic_out_dir = self.dirs["mosaic"] / date_str
            period_temp_dir = self.dirs["temp"] / date_str
            period_temp_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"=== Processing NISAR Period: {desc} ({len(images)} granules) ===")

            if stage_u != "1" and not overwrite and mosaic_out_dir.exists() and (mosaic_out_dir / "HHHH_gamma0.tif").exists():
                logging.info(f"Mosaic for {date_str} already exists in {mosaic_out_dir.name}. Skipping.")
                continue

            extracted_scene_dirs = []
            if stage_u == "2" and period_temp_dir.exists():
                existing_scenes = [
                    d for d in sorted(period_temp_dir.iterdir())
                    if d.is_dir() and (d / "HHHH_gamma0.tif").exists()
                ]
                if existing_scenes:
                    extracted_scene_dirs = existing_scenes

            if not extracted_scene_dirs:
                threads = int(self.configuration.get("threads", 4))
                if len(images) > 1 and threads > 1:
                    import time
                    from concurrent.futures import ThreadPoolExecutor

                    def _extract_single(item):
                        idx, img = item
                        if idx > 0:
                            time.sleep(min(idx * 3.5, 12.0))
                        try:
                            extractor = DownloadAndExtractGCOV(
                                product=img,
                                temp_dir=period_temp_dir,
                                session=self.session,
                                cookie_jar=cookie_jar,
                                token=token,
                                username=username,
                                password=password,
                                keep_intermediate_h5=keep_h5,
                                border_trim_pixels=border_trim_px
                            )
                            tifs = extractor.process()
                            return tifs[0].parent if tifs else None
                        except Exception as e:
                            logging.error(f"Scene extraction exception for {img.get('title')}: {e}")
                            return None

                    max_workers = min(len(images), threads)
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        results = list(executor.map(_extract_single, enumerate(images)))
                    extracted_scene_dirs = [r for r in results if r is not None]
                else:
                    for img in images:
                        try:
                            extractor = DownloadAndExtractGCOV(
                                product=img,
                                temp_dir=period_temp_dir,
                                session=self.session,
                                cookie_jar=cookie_jar,
                                token=token,
                                username=username,
                                password=password,
                                keep_intermediate_h5=keep_h5,
                                border_trim_pixels=border_trim_px
                            )
                            tifs = extractor.process()
                            if tifs:
                                extracted_scene_dirs.append(tifs[0].parent)
                        except Exception as e:
                            logging.error(f"Scene extraction exception for {img.get('title')}: {e}")

            if stage_u == "1":
                continue

            if extracted_scene_dirs:
                mosaic = MosaicNISAR(
                    input_scene_dirs=extracted_scene_dirs,
                    output_mosaic_dir=mosaic_out_dir,
                    target_grid=self.target_grid,
                    cutline_shp=self.cutline_shp,
                    country_code=self.country_code,
                    pixel_resolution=pixel_res,
                    target_epsg=target_epsg,
                    blend_width_pixels=blend_width_px
                )
                mosaic.run_mosaic()
                shutil.rmtree(period_temp_dir, ignore_errors=True)
            else:
                logging.warning(f"No valid scenes extracted for {desc}. Skipping mosaic.")

    def stage_3_stack_timeseries(self, start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None) -> Optional[Path]:
        date_dirs = sorted([
            d.name for d in self.dirs["mosaic"].iterdir()
            if d.is_dir() and len(d.name) == 8 and d.name.isdigit() and (d / "HHHH_gamma0.tif").exists()
        ]) if self.dirs["mosaic"].exists() else []

        if not date_dirs:
            logging.warning(f"No NISAR cycle mosaics available to stack in {self.dirs['mosaic']}")
            return None

        s_str = date_dirs[0] if not start_date else start_date.strftime("%Y%m%d")
        e_str = date_dirs[-1] if not end_date else end_date.strftime("%Y%m%d")

        if self.track_dir.name.startswith("orbit_"):
            prefix = f"{self.country_code}_{self.track_dir.name}"
        else:
            prefix = self.country_code

        out_stack_tif = self.dirs["stacks"] / f"{prefix}_{e_str}_{s_str}_NISAR_GCOV_L_timeseries.tif"
        legacy_gcov_tif = self.dirs["stacks"] / f"{prefix}_{e_str}_{s_str}_NISAR_GCOV.tif"
        canonical_tif = self.dirs["stacks"] / f"{prefix}_NISAR_timeseries.tif"
        cleanup_mosaics = bool(self.configuration.get("cleanup_cycle_mosaics", False))
        calib = str(self.configuration.get("calibration", "gamma0_db")).lower()
        convert_to_db = bool(self.configuration.get("convert_to_db", True)) and ("linear" not in calib)

        stacked = stack_nisar_timeseries(
            mosaic_base_dir=self.dirs["mosaic"],
            output_stack_tif=out_stack_tif,
            target_grid=self.target_grid,
            convert_gamma0_to_db=convert_to_db,
            build_overviews=True,
            cleanup_mosaics=cleanup_mosaics
        )

        if stacked and stacked.exists():
            # Create canonical relative symlink in 1_input_stacks/ (e.g. NL_orbit_37_NISAR_timeseries.tif -> ...)
            # Using relative symlink avoids confusing users with duplicated file sizes and saves disk space
            if canonical_tif.resolve() != stacked.resolve():
                try:
                    if canonical_tif.is_symlink() or canonical_tif.exists():
                        canonical_tif.unlink()
                    canonical_tif.symlink_to(stacked.name)
                except Exception as e:
                    logging.warning(f"Could not create canonical symlink {canonical_tif}: {e}")

            # Clean up any legacy hardlink aliases that may confuse directory listings
            sym_date_tif = self.dirs["stacks"] / f"{prefix}_{s_str}_{e_str}_NISAR_GCOV_L_timeseries.tif"
            for stale_alias in (legacy_gcov_tif, sym_date_tif):
                if stale_alias.exists() and stale_alias.resolve() == stacked.resolve() and not stale_alias.is_symlink():
                    try:
                        stale_alias.unlink()
                    except Exception:
                        pass

            # Only link to shared country NISAR pool `workingDirs/<COUNTRY>/NISAR/<COUNTRY>_NISAR_timeseries.tif`
            # when this stack covers the full country (not clipped to a single orbit_X).
            if not self.track_dir.name.startswith("orbit_"):
                shared_nisar_dir = BASE_DIR / self.country_code / "NISAR"
                shared_nisar_dir.mkdir(parents=True, exist_ok=True)
                shared_tif = shared_nisar_dir / f"{self.country_code}_NISAR_timeseries.tif"
                if shared_tif.resolve() != stacked.resolve():
                    try:
                        if shared_tif.is_symlink() or shared_tif.exists():
                            shared_tif.unlink()
                        rel_target = os.path.relpath(stacked, shared_nisar_dir)
                        shared_tif.symlink_to(rel_target)
                    except Exception as e:
                        logging.warning(f"Could not create shared NISAR symlink {shared_tif}: {e}")

        return stacked

