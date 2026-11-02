#!/usr/bin/env python
"""
================================================================================
AIML CropMapper Cloud - Phase 1b: Unified NISAR L-Band SAR Preprocessor
================================================================================
End-to-end operational pipeline for NASA/ISRO NISAR Level-2 Geocoded Polarimetric
Covariance (L2 GCOV) time-series extraction, cross-track antenna de-tapering,
seamless multi-track orbital cycle mosaicking, S1/S2 EPSG:3857 (10.0 m) grid
co-registration, and BigTIFF stacking across any European country (EU-27).

Key Capabilities:
  - Dual Data Source Discovery: Local `/eodata/NISAR` HDF5 repository fast-path
    and NASA ASF DAAC search (`asf_search` or built-in ASF REST API fallback).
  - Physical L-Band Polarimetry: Extracts HHHH (dB), HVHV (dB), RVI_L (4*HV/(HH+HV)),
    and HH/HV Ratio with 2D adaptive cross-track range-edge de-tapering, along-track
    slit healing, and C^1 inverse-distance-squared feathering.
  - S1 & S2 Pixel-for-Pixel Compatibility: Automatically matches the exact
    EPSG:3857 (10.0 m) pixel grid of Sentinel-1 & Sentinel-2 stacks in
    `workingDirs/<COUNTRY>/orbit_<ORBIT>/1_input_stacks/`.

Execution Examples:
  # 1. Interactive CLI setup wizard (zero arguments):
  python run_nisar_preprocessor.py

  # 2. Full pipeline for a specific orbit track (e.g. Netherlands Orbit 88):
  python run_nisar_preprocessor.py --track NL/orbit_88 -s 2025-10-15 -e 2026-09-15 --stage A

  # 3. Full pipeline for all discovered orbits of a country (e.g. Poland or Portugal):
  python run_nisar_preprocessor.py --country PL -s 2025-10-15 -e 2026-09-15 --stage A

  # 4. Run only Stage 3 (Re-stack existing cycle mosaics into S1-aligned BigTIFF):
  python run_nisar_preprocessor.py --track PT/orbit_52 --stage 3
================================================================================
"""

import argparse
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Constrain OpenMP/MKL threads before importing scientific libraries
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

script_dir = Path(__file__).resolve().parent
modules_dir = script_dir / "modules"
for p in [script_dir, modules_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nisar_asf_search import (
    AUX_DIR,
    BASE_DIR,
    FindNisarASF,
    discover_country_orbits,
    get_country_geometry,
    get_country_shapefile,
    get_s1_orbit_extent_geometry,
    get_s1_reference_grid,
    load_country_or_belt_geometry,
    prepare_orbit_cutline_shapefile,
    resolve_target_grid_and_cutline
)
from nisar_pipeline import CalculateTrackNISAR

try:
    import asf_search as asf
    HAS_ASF = True
except ImportError:
    asf = None
    HAS_ASF = False


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_nisar_config(config_path: Optional[Path] = None) -> dict:
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.extend([
        script_dir / "config_nisar.json",
        script_dir / "config_nisar.example.json",
        script_dir.parent / "docs" / "1b_NISAR_toolbox" / "config.json"
    ])
    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return {
        "product_type": "GCOV",
        "flight_direction": "DESCENDING",
        "target_epsg": 3857,
        "pixel_resolution_meters": 10.0,
        "grouping_mode": "cycle",
        "cycle_days": 12,
        "border_trim_pixels": 8,
        "blend_width_pixels": 350,
        "min_aoi_coverage_pct": 50.0,
        "min_track_coverage_pct": 35.0
    }


class NISARPreprocessorPipeline:
    """Unified 3-stage NISAR L2 GCOV Preprocessor compatible with S1 and S2."""

    def __init__(
        self,
        country: str,
        orbit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        config_file: Optional[Path] = None,
        grouping_mode: Optional[str] = None,
        source: str = "auto",
        overwrite: bool = False,
        threads: int = 4,
        keep_h5: bool = False
    ):
        self.country = country.upper()
        self.orbit = int(orbit) if orbit is not None else None
        self.track = f"{self.country}/orbit_{self.orbit}" if self.orbit else self.country
        self.track_dir = (BASE_DIR / self.country / f"orbit_{self.orbit}") if self.orbit else (BASE_DIR / self.country)
        self.config = load_nisar_config(config_file)
        self.source = (source or "auto").lower()
        if grouping_mode:
            self.config["grouping_mode"] = grouping_mode
        if overwrite:
            self.config["overwrite_existing"] = True
        if keep_h5:
            self.config["keep_intermediate_h5"] = True
        if threads:
            self.config["threads"] = int(threads)

        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else datetime.date(2025, 2, 1)
        self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else datetime.date(2025, 10, 15)

    def _authenticate_earthdata(self):
        token = os.environ.get("EARTHDATA_TOKEN") or self.config.get("earthdata_token")
        username = os.environ.get("EARTHDATA_USERNAME") or self.config.get("earthdata_username")
        password = os.environ.get("EARTHDATA_PASSWORD") or self.config.get("earthdata_password")

        extended_domains = [
            "asf.alaska.edu",
            "earthdata.nasa.gov",
            "earthdatacloud.nasa.gov",
            "nasa.gov",
            "alaska.edu"
        ]

        if not HAS_ASF:
            logging.warning("asf_search package not installed. Cannot initialize NASA Earthdata session.")
            return None

        # 1. Direct Bearer token authentication
        if token and "YOUR_" not in token:
            try:
                logging.info("Authenticating with NASA Earthdata using provided Bearer token...")
                session = asf.ASFSession(auth_domains=extended_domains)
                session.auth_with_token(token)
                session.token = token
                self.config["earthdata_token"] = token
                logging.info("Successfully authenticated NASA Earthdata session with Bearer token.")
                return session
            except Exception as e:
                logging.warning(f"NASA Earthdata token authentication failed: {e}")

        # 2. Username/Password authentication
        if username and password and "YOUR_" not in username:
            # Step 2a: Retrieve EDL Bearer token via /api/users/find_or_create_token
            # This directly sets Authorization: Bearer <token> on trusted domains and avoids
            # the known cumulus.asf.alaska.edu asf-urs cookie HTTP 400 error.
            try:
                from requests.auth import HTTPBasicAuth
                init_session = asf.ASFSession()
                init_session.auth = HTTPBasicAuth(username, password)
                retrieved_token = init_session._get_urs_access_token()
                init_session.auth = None
                if retrieved_token:
                    session = asf.ASFSession(auth_domains=extended_domains)
                    session.auth_with_token(retrieved_token)
                    session.token = retrieved_token
                    self.config["earthdata_token"] = retrieved_token
                    logging.info(f"Successfully authenticated NASA Earthdata as user: {username} (EDL Bearer token active)")
                    return session
            except Exception as e:
                logging.warning(f"Direct EDL Bearer token retrieval encountered issue ({e}); attempting auth_with_creds...")

            # Step 2b: Standard auth_with_creds fallback
            try:
                logging.info(f"Authenticating with NASA Earthdata as user: {username}")
                session = asf.ASFSession(auth_domains=extended_domains)
                session.auth_with_creds(username, password)
                return session
            except Exception as e:
                logging.warning(f"NASA Earthdata asf_search session auth warning: {e}")

        return None

    def _resolve_target_grid_and_cutline(self):
        target_epsg = int(self.config.get("target_epsg", 3857))
        pixel_res = float(self.config.get("pixel_resolution_meters", 10.0))
        return resolve_target_grid_and_cutline(
            country_code=self.country,
            orbit_num=self.orbit,
            track_dir=self.track_dir,
            target_epsg=target_epsg,
            pixel_res=pixel_res,
            aux_dir=AUX_DIR
        )

    def run(self, stage: str = "A"):
        logging.info(f"\n============================================================")
        logging.info(f" AIML CropMapper Cloud - NISAR L-Band SAR Preprocessor")
        logging.info(f" Target Track : {self.track}")
        logging.info(f" Time Window  : {self.start_date} -> {self.end_date}")
        logging.info(f" Stage        : {stage}")
        logging.info(f"============================================================")

        target_grid, cutline_shp = self._resolve_target_grid_and_cutline()
        stage_u = stage.strip().upper()

        if stage_u in ("A", "1", "2"):
            session = self._authenticate_earthdata() if self.source in ("asf", "auto") else None
            finder = FindNisarASF()
            local_repo = Path(self.config.get("local_nisar_repository", "/eodata/NISAR"))
            products = finder.search_products(
                geometry_wkt=target_grid["wkt_4326"],
                start_date=self.start_date,
                end_date=self.end_date,
                product_type=self.config.get("product_type", "GCOV"),
                flight_direction=self.config.get("flight_direction", "DESCENDING"),
                session=session,
                local_dir=local_repo if local_repo.exists() else None,
                source=self.source
            )
            if not products:
                logging.warning(
                    f"No NISAR L2 GCOV scenes found for {self.track} between {self.start_date} and {self.end_date}. "
                    f"Downstream classifier will automatically fall back to Sentinel-1 + Sentinel-2."
                )
                return

            processor = CalculateTrackNISAR(
                list_to_download=products,
                target_grid=target_grid,
                track_dir=self.track_dir,
                country_code=self.country,
                configuration=self.config,
                session=session,
                cutline_shp=cutline_shp
            )
            processor.stage_1_and_2_extract_and_mosaic(stage=stage_u)
            if stage_u == "A":
                processor.stage_3_stack_timeseries(self.start_date, self.end_date)
        elif stage_u == "3":
            processor = CalculateTrackNISAR(
                list_to_download=[],
                target_grid=target_grid,
                track_dir=self.track_dir,
                country_code=self.country,
                configuration=self.config,
                session=None,
                cutline_shp=cutline_shp
            )
            processor.stage_3_stack_timeseries(self.start_date, self.end_date)


def interactive_wizard():
    print("""
================================================================================
     AIML CropMapper Cloud - NISAR L-Band SAR Preprocessor Setup Wizard
================================================================================
 Welcome! This wizard configures NISAR L2 GCOV polarimetric extraction,
 seamless cycle mosaicking, and EPSG:3857 (10.0 m) S1/S2 grid stacking.
================================================================================""")
    country = input(" Enter 2-letter Country Code (e.g. NL, PT, PL, FR, DE) [PL]: ").strip().upper() or "PL"
    orbits = discover_country_orbits(country)
    orbit_val = None
    if orbits:
        print(f" Discovered existing S1 orbits for {country}: {orbits}")
        orb_in = input(f" Enter single orbit number or press Enter for ALL orbits: ").strip()
        if orb_in.isdigit():
            orbit_val = int(orb_in)
    s_date = input(" Enter Start Date (YYYY-MM-DD) [2025-10-15]: ").strip() or "2025-10-15"
    e_date = input(" Enter End Date   (YYYY-MM-DD) [2026-09-15]: ").strip() or "2026-09-15"
    source = input(" Enter Data Source ('auto', 'asf', 'local') [auto]: ").strip().lower() or "auto"
    stage = input(" Enter Stage ('A' for All, '1' Extract, '2' Mosaic, '3' Stack) [A]: ").strip().upper() or "A"
    threads_in = input(" Enter Worker Threads [4]: ").strip()
    threads_val = int(threads_in) if threads_in.isdigit() else 4

    if orbit_val is None and orbits:
        for o_num in orbits:
            pipe = NISARPreprocessorPipeline(
                country=country,
                orbit=o_num,
                start_date=s_date,
                end_date=e_date,
                source=source,
                threads=threads_val
            )
            pipe.run(stage=stage)
    else:
        pipe = NISARPreprocessorPipeline(
            country=country,
            orbit=orbit_val,
            start_date=s_date,
            end_date=e_date,
            source=source,
            threads=threads_val
        )
        pipe.run(stage=stage)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Unified NISAR L-Band SAR (L2 GCOV) Preprocessor for AIML CropMapper Cloud.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-t", "--track", default=None, help="Track identifier (e.g. NL/orbit_88, PT/orbit_52, NL_orbit_110)")
    parser.add_argument("-c", "--country", default=None, help="2-letter EU country code (e.g. NL, PT, PL, FR, DE)")
    parser.add_argument("-o", "--orbit", type=int, default=None, help="Relative S1 reference orbit number")
    parser.add_argument("-s", "--start_date", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("-e", "--end_date", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--source", default="auto", choices=["asf", "local", "auto"], help="Data source: 'asf' (NASA ASF DAAC), 'local' (/eodata/NISAR), or 'auto' (default)")
    parser.add_argument("--stage", default="A", choices=["A", "1", "2", "3", "a"], help="Stage to run: 'A' (all), '1' (extract), '2' (mosaic), '3' (stack)")
    parser.add_argument("--grouping_mode", default=None, choices=["cycle", "dekada", "date"], help="Temporal grouping mode (default: cycle [12-day NISAR orbital cycle])")
    parser.add_argument("--threads", type=int, default=4, help="Number of worker threads (default: 4)")
    parser.add_argument("--keep_h5", action="store_true", help="Retain raw .h5 files after extraction")
    parser.add_argument("-j", "--config", default=None, help="Path to config_nisar.json")
    parser.add_argument("-f", "--overwrite", action="store_true", help="Force overwrite of existing mosaics and stacks")

    args = parser.parse_args()

    if not args.track and not args.country:
        interactive_wizard()
        return

    country = args.country.upper() if args.country else None
    orbit = args.orbit
    if args.track:
        norm_t = args.track.replace("\\", "/").strip("/")
        m = re.search(r"(?:^|/)([A-Za-z]{2})[/\-_](?:orbit[_-]?)?(\d+)(?:/|$)", norm_t, re.IGNORECASE)
        if m:
            if not country:
                country = m.group(1).upper()
            if orbit is None:
                orbit = int(m.group(2))
        else:
            parts = [p for p in norm_t.split("/") if p]
            if len(parts) >= 2:
                m_orb = re.search(r"(\d+)", parts[-1])
                if m_orb and orbit is None:
                    orbit = int(m_orb.group(1))
                if not country and len(parts[-2]) == 2 and parts[-2].isalpha():
                    country = parts[-2].upper()
            elif len(parts) == 1:
                m_orb = re.search(r"([A-Za-z]{2})_orbit_(\d+)", parts[0], re.IGNORECASE)
                if m_orb:
                    if not country:
                        country = m_orb.group(1).upper()
                    if orbit is None:
                        orbit = int(m_orb.group(2))
                else:
                    m_orb2 = re.search(r"(\d+)", parts[0])
                    if m_orb2 and orbit is None:
                        orbit = int(m_orb2.group(1))
                    if not country:
                        m_c = re.search(r"^([A-Za-z]{2})$", parts[0])
                        if m_c:
                            country = m_c.group(1).upper()

    if not country:
        parser.error("Either --track (-t) or --country (-c) must be specified.")

    cfg_path = Path(args.config) if args.config else None

    if orbit is not None:
        pipe = NISARPreprocessorPipeline(
            country=country,
            orbit=orbit,
            start_date=args.start_date,
            end_date=args.end_date,
            config_file=cfg_path,
            grouping_mode=args.grouping_mode,
            source=args.source,
            overwrite=args.overwrite,
            threads=args.threads,
            keep_h5=args.keep_h5
        )
        pipe.run(stage=args.stage)
    else:
        orbits = discover_country_orbits(country)
        if orbits:
            for o_num in orbits:
                pipe = NISARPreprocessorPipeline(
                    country=country,
                    orbit=o_num,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    config_file=cfg_path,
                    grouping_mode=args.grouping_mode,
                    source=args.source,
                    overwrite=args.overwrite,
                    threads=args.threads,
                    keep_h5=args.keep_h5
                )
                pipe.run(stage=args.stage)
        else:
            pipe = NISARPreprocessorPipeline(
                country=country,
                orbit=None,
                start_date=args.start_date,
                end_date=args.end_date,
                config_file=cfg_path,
                grouping_mode=args.grouping_mode,
                source=args.source,
                overwrite=args.overwrite,
                threads=args.threads,
                keep_h5=args.keep_h5
            )
            pipe.run(stage=args.stage)


if __name__ == "__main__":
    main()

