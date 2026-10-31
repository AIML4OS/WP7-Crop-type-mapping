import argparse
import datetime
import logging
import pathlib
import sys
import re
import subprocess
import shutil
import requests
import zipfile
import json
import time
import threading
from collections import defaultdict
from osgeo import ogr, osr
import os

# ================= CONFIGURATION =================
# How to run the script for a selected country:
# python 1c_slice_calibration_cdse.py -s 2024-10-15 -e 2024-11-30 -c PL
# python 1c_slice_calibration_cdse.py -s 2024-10-15 -e 2024-11-30 -c FR

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
GPT_EXE = os.environ.get("SNAP_GPT_EXE", r"D:/Program Files/esa-snap/bin/gpt.exe")
AUXDATA_PATH = os.environ.get("SNAP_AUXDATA_PATH", r"C:/Users/Administrator/.snap/auxdata")
WORKING_DIR = os.environ.get("AIML_WORKING_DIR", str(_PROJECT_ROOT / "workingDirs"))
AUX_DIR = os.environ.get("AIML_AUX_DIR", str(_PROJECT_ROOT / "auxiliary_files"))

# CDSE Credentials
CDSE_USERNAME = os.environ.get("CDSE_USERNAME")
CDSE_PASSWORD = os.environ.get("CDSE_PASSWORD")

# Auto-load from JSON configuration files
try:
    _root_dir = pathlib.Path(__file__).resolve().parent.parent
    _cfg_paths = [
        _root_dir / "config_s1.json",
        _root_dir / "config_cdse.json",
        _root_dir / "config.json",
        pathlib.Path(__file__).resolve().parent / "config_s1.json"
    ]
    for _cp in _cfg_paths:
        if _cp.exists():
            with open(_cp, 'r', encoding='utf-8') as _f:
                _data = json.load(_f)
                if not CDSE_USERNAME:
                    CDSE_USERNAME = _data.get("cdse", {}).get("username") or _data.get("username")
                if not CDSE_PASSWORD:
                    CDSE_PASSWORD = _data.get("cdse", {}).get("password") or _data.get("password")
                if "paths" in _data:
                    GPT_EXE = _data["paths"].get("snap_gpt_exe", GPT_EXE)
                    AUXDATA_PATH = _data["paths"].get("snap_auxdata_path", AUXDATA_PATH)
                    if "AIML_WORKING_DIR" not in os.environ:
                        WORKING_DIR = _data["paths"].get("working_dir", WORKING_DIR)
                    if "AIML_AUX_DIR" not in os.environ:
                        AUX_DIR = _data["paths"].get("aux_dir", AUX_DIR)
except Exception:
    pass

# ================= XML TEMPLATES =================

# Standard Calibration (S1A/S1B)
CALIBRATION_TEMPLATE = r'''<graph id="Graph">
  <version>1.0</version>
{read_nodes}{tnr_nodes}{aof_nodes}{bnr_nodes}{calib_nodes}
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_file}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>'''

SLICE_ASSEMBLY_TEMPLATE = r'''<graph id="Graph">
  <version>1.0</version>
{read_nodes}
  <node id="SliceAssembly">
    <operator>SliceAssembly</operator>
    <sources>
{slice_sources}    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations/>
    </parameters>
  </node>
  <node id="Subset">
    <operator>Subset</operator>
    <sources>
      <sourceProduct refid="SliceAssembly"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <geoRegion>{geoRegion}</geoRegion>
      <subSamplingX>1</subSamplingX>
      <subSamplingY>1</subSamplingY>
      <fullSwath>false</fullSwath>
      <copyMetadata>false</copyMetadata>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Subset"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_file}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
  <applicationData id="Presentation">
    <Description/>
{app_data}  </applicationData>
</graph>'''

SINGLE_SLICE_TEMPLATE = r'''<graph id="Graph">
  <version>1.0</version>
{read_nodes}
  <node id="Subset">
    <operator>Subset</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <geoRegion>{geoRegion}</geoRegion>
      <subSamplingX>1</subSamplingX>
      <subSamplingY>1</subSamplingY>
      <fullSwath>false</fullSwath>
      <copyMetadata>false</copyMetadata>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Subset"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_file}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>'''

TNR_NODE = r'''  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VH,VV</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
      <outputNoise>false</outputNoise>
      <reIntroduceThermalNoise>false</reIntroduceThermalNoise>
    </parameters>
  </node>'''

AOF_NODE = r'''  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="ThermalNoiseRemoval"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>'''

BNR_NODE = r'''  <node id="Remove-GRD-Border-Noise">
    <operator>Remove-GRD-Border-Noise</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VH,VV</selectedPolarisations>
      <borderLimit>500</borderLimit>
      <trimThreshold>0.5</trimThreshold>
    </parameters>
  </node>'''

CALIB_NODE = r'''  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="Remove-GRD-Border-Noise"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <auxFile/>
      <outputImageScaleInDb>false</outputImageScaleInDb>
      <createGammaBand>false</createGammaBand>
      <createBetaBand>false</createBetaBand>
      <selectedPolarisations>VH,VV</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
    </parameters>
  </node>'''


# Path to NUTS shapefiles
SHAPEFILES_DIR = pathlib.Path(os.environ.get("AIML_AUX_DIR", AUX_DIR)) / "shapefiles_nuts"


def load_shapefile_geometry_ogr(shp_path, target_epsg=4326):
    """Loads all geometries from a shapefile, transforms them to the target EPSG code, and returns their union."""
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(str(shp_path))
    if not ds:
        return None
    layer = ds.GetLayer()

    src_srs = layer.GetSpatialRef()
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(target_epsg)

    coord_trans = None
    if src_srs and not src_srs.IsSame(dst_srs):
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        coord_trans = osr.CoordinateTransformation(src_srs, dst_srs)

    union_geom = None
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom:
            cloned_geom = geom.Clone()
            if coord_trans:
                cloned_geom.Transform(coord_trans)
            if union_geom is None:
                union_geom = cloned_geom
            else:
                union_geom = union_geom.Union(cloned_geom)
    ds = None
    return union_geom


def get_country_geometry(country_code: str):
    """Dynamically locates and loads the NUTS2 shapefile for the given country code."""
    shp_path = SHAPEFILES_DIR / country_code / f"NUTS2_{country_code}.shp"
    if not shp_path.exists():
        logging.warning(f"Country shapefile not found at {shp_path}. Searching for any .shp in folder...")
        country_dir = SHAPEFILES_DIR / country_code
        if country_dir.exists():
            shp_files = list(country_dir.glob("*.shp"))
            if shp_files:
                shp_path = shp_files[0]
            else:
                logging.error(f"No shapefile found in {country_dir}")
                return None
        else:
            logging.error(f"Country directory {country_dir} does not exist.")
            return None

    logging.info(f"Loading country geometry from {shp_path}...")
    return load_shapefile_geometry_ogr(shp_path)


# ================= LOGIC: CDSE AUTHENTICATION =================

class CDSETokenManager:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.token = None
        self.token_fetch_time = 0.0
        self._lock = threading.Lock()

    def get_token(self):
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        res = requests.post(token_url, data=data, timeout=30)
        if res.status_code == 200:
            self.token = res.json()["access_token"]
            self.token_fetch_time = time.time()
            return self.token
        else:
            raise Exception(f"Failed to authenticate with CDSE Keycloak: {res.status_code} - {res.text}")

    def get_headers(self, force_refresh=False):
        with self._lock:
            now = time.time()
            # CDSE access tokens expire after 600s; proactively refresh after 480s
            if not self.token or force_refresh or (now - self.token_fetch_time) > 480.0:
                self.get_token()
            return {"Authorization": f"Bearer {self.token}"}


# ================= LOGIC: DOWNLOADER =================

def download_product(token_manager, product_id, dest_zip_path):
    download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = token_manager.get_headers()
    
    for attempt in range(2):
        res = requests.get(download_url, headers=headers, allow_redirects=False)
        if res.status_code == 401:
            logging.info("Token expired during download start. Refreshing...")
            headers = token_manager.get_headers(force_refresh=True)
            continue
            
        if res.status_code in [301, 302, 303, 307, 308]:
            redirect_url = res.headers["Location"]
            res = requests.get(redirect_url, headers=headers, stream=True)
            
        res.raise_for_status()
        break
    else:
        raise Exception(f"Failed to start download: {res.status_code}")
        
    total_size = int(res.headers.get('content-length', 0))
    chunk_size = 1024 * 1024 # 1MB chunks
    downloaded = 0
    
    logging.info(f"Downloading product {product_id} to {dest_zip_path.name} ({total_size / (1024*1024):.1f} MB)...")
    with open(dest_zip_path, 'wb') as f:
        for chunk in res.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = int(100 * downloaded / total_size)
                    sys.stdout.write(f"\r    Download progress: {percent}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)")
                    sys.stdout.flush()
    sys.stdout.write("\n")
    logging.info("Download completed.")


def extract_zip(zip_path, extract_dir):
    logging.info(f"Extracting {zip_path.name} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    logging.info("Extraction completed.")


# ================= LOGIC: CDSE OPTIMIZER =================

class CDSECountryOrbitOptimizer:
    def __init__(self, token_manager, country_geom: ogr.Geometry):
        self.token_manager = token_manager
        self.country_geom = country_geom

    def _solve_set_cover(self, discovered_orbits, min_coverage_pct=0.005):
        if not discovered_orbits:
            return [], 0.0

        orbit_intersections = {}
        for orbit_num, footprints in discovered_orbits.items():
            orbit_union = None
            for fp in footprints:
                if orbit_union is None:
                    orbit_union = fp.Clone()
                else:
                    orbit_union = orbit_union.Union(fp)

            intersection = orbit_union.Intersection(self.country_geom)
            if intersection and not intersection.IsEmpty():
                orbit_intersections[orbit_num] = {
                    'geom': intersection,
                    'area': intersection.GetArea()
                }

        if not orbit_intersections:
            return [], 0.0

        selected_orbits = []
        target_geom = None
        for orbit_data in orbit_intersections.values():
            if target_geom is None:
                target_geom = orbit_data['geom'].Clone()
            else:
                target_geom = target_geom.Union(orbit_data['geom'])

        total_target_area = target_geom.GetArea()
        remaining_target = target_geom.Clone()

        while remaining_target and not remaining_target.IsEmpty():
            best_orbit = None
            best_new_area = 0.0

            for orbit_num, data in orbit_intersections.items():
                if orbit_num in selected_orbits:
                    continue
                new_overlap = data['geom'].Intersection(remaining_target)
                if new_overlap and not new_overlap.IsEmpty():
                    new_area = new_overlap.GetArea()
                    if new_area > best_new_area:
                        best_new_area = new_area
                        best_orbit = orbit_num

            min_significant_area = min_coverage_pct * total_target_area
            if best_orbit is None or best_new_area < min_significant_area:
                break

            selected_orbits.append(best_orbit)
            remaining_target = remaining_target.Difference(orbit_intersections[best_orbit]['geom'])

        final_coverage_geom = None
        for o_num in selected_orbits:
            o_geom = orbit_intersections[o_num]['geom']
            if final_coverage_geom is None:
                final_coverage_geom = o_geom.Clone()
            else:
                final_coverage_geom = final_coverage_geom.Union(o_geom)

        final_coverage_area = final_coverage_geom.GetArea() if final_coverage_geom else 0.0
        return selected_orbits, final_coverage_area

    def discover_and_optimize(self, start_date: datetime.date, search_days=12, country_code=None, min_coverage_pct=0.005):
        logging.info(f"Starting CDSE orbit discovery for a {search_days}-day window starting from {start_date}...")
        
        simplified_geom = self.country_geom.Simplify(0.05)
        wkt_geom = simplified_geom.ExportToWkt()
        end_date = start_date + datetime.timedelta(days=search_days)

        filter_str = (
            "Collection/Name eq 'SENTINEL-1' "
            "and contains(Name, 'IW_GRDH_1S') "
            "and not contains(Name, '_COG.SAFE') "
            f"and ContentDate/Start ge {start_date.isoformat()}T00:00:00.000Z "
            f"and ContentDate/Start le {end_date.isoformat()}T23:59:59.999Z "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_geom}')"
        )

        search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        params = {
            "$filter": filter_str,
            "$expand": "Attributes",
            "$top": 1000
        }

        headers = self.token_manager.get_headers()
        res = requests.get(search_url, headers=headers, params=params)
        if res.status_code == 401:
            headers = self.token_manager.get_headers(force_refresh=True)
            res = requests.get(search_url, headers=headers, params=params)

        if res.status_code != 200:
            logging.error(f"CDSE Orbit Discovery failed: {res.status_code} - {res.text}")
            return [], None

        products = res.json().get("value", [])
        logging.info(f"Discovered {len(products)} candidate scenes in CDSE.")

        discovered_asc = defaultdict(list)
        discovered_dsc = defaultdict(list)

        for p in products:
            rel_orbit = None
            orbit_dir = None
            for attr in p.get("Attributes", []):
                if attr.get("Name") == "relativeOrbitNumber":
                    rel_orbit = int(attr.get("Value"))
                elif attr.get("Name") == "orbitDirection":
                    orbit_dir = attr.get("Value")

            if rel_orbit is None:
                continue

            geo_footprint = p.get("GeoFootprint")
            if not geo_footprint:
                continue
            try:
                footprint = ogr.CreateGeometryFromJson(json.dumps(geo_footprint))
            except Exception as e:
                logging.warning(f"Failed to parse geometry for product {p['Name']}: {e}")
                continue

            if not footprint:
                continue

            if footprint.Intersects(self.country_geom):
                if orbit_dir == 'ASCENDING':
                    discovered_asc[rel_orbit].append(footprint)
                elif orbit_dir == 'DESCENDING':
                    discovered_dsc[rel_orbit].append(footprint)
                else:
                    discovered_asc[rel_orbit].append(footprint)

        logging.info(f"Discovered orbits in CDSE: {len(discovered_asc)} ASCENDING, {len(discovered_dsc)} DESCENDING.")

        logging.info("Optimizing coverage for ASCENDING passes...")
        selected_asc, area_asc = self._solve_set_cover(discovered_asc, min_coverage_pct=min_coverage_pct)
        logging.info(f"  ASCENDING: Selected {len(selected_asc)} orbits covering {area_asc:.4f} sq. degrees.")

        logging.info("Optimizing coverage for DESCENDING passes...")
        selected_dsc, area_dsc = self._solve_set_cover(discovered_dsc, min_coverage_pct=min_coverage_pct)
        logging.info(f"  DESCENDING: Selected {len(selected_dsc)} orbits covering {area_dsc:.4f} sq. degrees.")

        if not selected_asc and not selected_dsc:
            logging.error("No overlapping Sentinel-1 orbits discovered in CDSE!")
            return [], None
        elif not selected_asc:
            choose_asc = False
        elif not selected_dsc:
            choose_asc = True
        else:
            diff_area = area_asc - area_dsc
            if abs(diff_area) < 0.05:
                if len(selected_asc) <= len(selected_dsc):
                    choose_asc = True
                else:
                    choose_asc = False
            else:
                choose_asc = diff_area > 0

        if choose_asc:
            logging.info(f"Selected ASCENDING pass direction (Orbits: {selected_asc})")
            return selected_asc, 'ASCENDING'
        else:
            logging.info(f"Selected DESCENDING pass direction (Orbits: {selected_dsc})")
            return selected_dsc, 'DESCENDING'


# ================= LOGIC: CDSE FINDER / CACHING DOWNLOADER =================

def safe_exists(p) -> bool:
    """Safely check if path exists, catching OSError (including Errno 5 / EIO on FUSE/s3fs mounts)."""
    if p is None:
        return False
    try:
        return pathlib.Path(p).exists()
    except OSError:
        return False
    except Exception:
        return False


def safe_is_dir(p) -> bool:
    """Safely check if path is a directory, catching OSError (including Errno 5 / EIO on FUSE/s3fs mounts)."""
    if p is None:
        return False
    try:
        return pathlib.Path(p).is_dir()
    except OSError:
        return False
    except Exception:
        return False


def safe_file_size(p) -> int:
    """Safely get file size in bytes, catching OSError (including Errno 5 / EIO on FUSE/s3fs mounts)."""
    if p is None:
        return 0
    try:
        return pathlib.Path(p).stat().st_size
    except OSError:
        return 0
    except Exception:
        return 0


def _is_valid_eodata_safe(safe_dir: pathlib.Path) -> bool:
    """Verify that a candidate /eodata SAFE folder exists, is a directory, and has an accessible manifest."""
    try:
        if not safe_is_dir(safe_dir):
            return False
        # Manifest check confirms directory is actually accessible and not an unreadable FUSE mount error
        manifest_std = safe_dir / "manifest.safe"
        manifest_upper = safe_dir / "manifest.SAFE"
        return safe_exists(manifest_std) or safe_exists(manifest_upper)
    except OSError:
        return False
    except Exception:
        return False


class CDSESentinel1Finder:
    def __init__(self, token_manager, download_dir: pathlib.Path):
        self.token_manager = token_manager
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def find_products_by_orbit(self, orbit_num: int, target_geom: ogr.Geometry,
                               start_date: datetime.date, end_date: datetime.date,
                               working_dir: pathlib.Path = None, country_code: str = None,
                               pass_direction: str = None, exclude_winter: bool = False):
        """Query CDSE for S1 products matching spatial bounds and download them locally on-demand."""
        simplified_geom = target_geom.Simplify(0.05)
        wkt_geom = simplified_geom.ExportToWkt()

        filter_str = (
            "Collection/Name eq 'SENTINEL-1' "
            "and contains(Name, 'IW_GRDH_1S') "
            "and not contains(Name, '_COG.SAFE') "
            f"and ContentDate/Start ge {start_date.isoformat()}T00:00:00.000Z "
            f"and ContentDate/Start le {end_date.isoformat()}T23:59:59.999Z "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_geom}') "
            f"and Attributes/OData.CSC.IntegerAttribute/any(att:att/Name eq 'relativeOrbitNumber' and att/Value eq {orbit_num})"
        )
        if pass_direction:
            filter_str += f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'orbitDirection' and att/Value eq '{pass_direction}')"

        search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        params = {
            "$filter": filter_str,
            "$expand": "Attributes",
            "$top": 1000
        }

        logging.info(f"Querying CDSE for products of orbit {orbit_num}...")
        headers = self.token_manager.get_headers()
        res = requests.get(search_url, headers=headers, params=params)
        if res.status_code == 401:
            headers = self.token_manager.get_headers(force_refresh=True)
            res = requests.get(search_url, headers=headers, params=params)

        if res.status_code != 200:
            logging.error(f"CDSE Product query failed: {res.status_code} - {res.text}")
            return

        products = res.json().get("value", [])
        logging.info(f"Found {len(products)} products in CDSE for orbit {orbit_num}.")

        # Group products by acquisition date (UTC date) and compute reference coverage
        date_groups = defaultdict(list)
        date_cov_areas = {}
        ref_cov_area = 0.0
        for p in products:
            start_time_str = p["ContentDate"]["Start"]
            date_str = start_time_str.split("T")[0]
            prod_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            date_groups[prod_date].append(p)

        for prod_date, p_list in date_groups.items():
            day_union = None
            for p in p_list:
                fp_str = p.get("Footprint", "") or ""
                if "POLYGON" in fp_str:
                    try:
                        wkt = fp_str.split(";", 1)[-1].rstrip("'")
                        g = ogr.CreateGeometryFromWkt(wkt)
                        if g is not None and g.Intersects(target_geom):
                            day_union = g.Clone() if day_union is None else day_union.Union(g)
                    except Exception:
                        pass
            if day_union is not None:
                try:
                    inter = day_union.Intersection(target_geom)
                    if inter is not None:
                        cov = inter.GetArea()
                        date_cov_areas[prod_date] = cov
                        ref_cov_area = max(ref_cov_area, cov)
                except Exception:
                    pass

        # Iterate through the dates in chronological order
        for prod_date in sorted(date_groups.keys()):
            if exclude_winter:
                m, d = prod_date.month, prod_date.day
                if m == 12 or m == 1 or (m == 2 and d <= 14):
                    logging.info(f"Skipping winter date: {prod_date} (exclude_winter=True)")
                    continue

            # Check pass completeness (no missing middle slice / >= 85% reference swath coverage)
            parsed_times = []
            for p in date_groups[prod_date]:
                parts = p.get("Name", "").split("_")
                if len(parts) >= 7:
                    try:
                        t_start = datetime.datetime.strptime(parts[4], "%Y%m%dT%H%M%S")
                        t_stop = datetime.datetime.strptime(parts[5], "%Y%m%dT%H%M%S")
                        parsed_times.append((t_start, t_stop, p.get("Name", "")))
                    except ValueError:
                        pass
            parsed_times.sort(key=lambda x: x[0])
            has_mid_gap = False
            for i in range(len(parsed_times) - 1):
                prev_start, prev_stop, prev_name = parsed_times[i]
                curr_start, curr_stop, curr_name = parsed_times[i + 1]
                gap_sec = (curr_start - prev_stop).total_seconds()
                if 5.0 < gap_sec < 600.0:
                    logging.warning(
                        f"   [SKIP DATE {prod_date}] Incomplete pass: missing middle slice detected "
                        f"({gap_sec:.0f}s gap between {prev_name} and {curr_name})."
                    )
                    has_mid_gap = True
                    break
            if has_mid_gap:
                continue

            day_cov = date_cov_areas.get(prod_date, 0.0)
            if ref_cov_area > 0.0 and day_cov > 0.0 and day_cov < 0.85 * ref_cov_area:
                pct = (day_cov / ref_cov_area) * 100.0
                logging.warning(
                    f"   [SKIP DATE {prod_date}] Incomplete pass: daily slices cover only {pct:.1f}% "
                    f"of expected orbit swath over AOI (< 85% threshold; missing slice)."
                )
                continue

            if working_dir and country_code:
                date_str_formatted = prod_date.strftime("%Y%m%d")
                final_output_dir = working_dir / country_code / f"orbit_{orbit_num}" / "slice_assembly"
                if safe_is_dir(final_output_dir):
                    existing_dims = [
                        f for f in final_output_dir.glob(f"{date_str_formatted}_{country_code}_orbit_{orbit_num}_*.dim")
                        if safe_is_dir(f.with_suffix('.data'))
                    ]
                    existing_tifs = [
                        f for f in final_output_dir.glob(f"{date_str_formatted}_{country_code}_orbit_{orbit_num}_*.tif")
                        if safe_file_size(f) > 1024
                    ]
                    # Also check pure Python Stage 1 output format (date_str.tif)
                    py_tif = final_output_dir / f"{date_str_formatted}.tif"
                    if safe_file_size(py_tif) > 1024:
                        existing_tifs.append(py_tif)
                    existing_finals = existing_dims + existing_tifs
                    if existing_finals:
                        logging.info(f"Skipping download/processing for {date_str_formatted}: Final output already exists ({existing_finals[0].name})")
                        continue

            local_safe_paths = []
            for p in date_groups[prod_date]:
                p_name = p["Name"]
                if p_name.endswith(".SAFE"):
                    safe_folder_name = p_name
                else:
                    safe_folder_name = p_name + ".SAFE"

                # CDSE / CreoDIAS local /eodata fast-path: if running inside CDSE cloud with /eodata mounted,
                # read directly from /eodata instead of downloading over HTTP
                eodata_cand_dirs = [
                    pathlib.Path(f"/eodata/Sentinel-1/SAR/IW_GRDH_1S/{prod_date.year:04d}/{prod_date.month:02d}/{prod_date.day:02d}") / safe_folder_name,
                    pathlib.Path(f"/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/{prod_date.year:04d}/{prod_date.month:02d}/{prod_date.day:02d}") / safe_folder_name.replace(".SAFE", "_COG.SAFE"),
                ]
                eodata_hit = next((ep for ep in eodata_cand_dirs if _is_valid_eodata_safe(ep)), None)
                if eodata_hit is not None:
                    logging.info(f"Using mounted CDSE /eodata SAFE folder directly: {eodata_hit}")
                    local_safe_paths.append(eodata_hit)
                    continue

                local_safe_path = self.download_dir / safe_folder_name
                
                # Check if already downloaded and extracted (verify manifest exists and is complete)
                has_cached_safe = safe_is_dir(local_safe_path) and (
                    safe_exists(local_safe_path / "manifest.safe") or safe_exists(local_safe_path / "manifest.SAFE")
                )

                if has_cached_safe:
                    logging.info(f"Using cached SAFE folder: {local_safe_path}")
                    local_safe_paths.append(local_safe_path)
                else:
                    zip_name = safe_folder_name.replace(".SAFE", ".zip")
                    zip_path = self.download_dir / zip_name
                    
                    try:
                        download_product(self.token_manager, p["Id"], zip_path)
                        # Extract zip
                        extract_zip(zip_path, self.download_dir)
                        # Delete zip
                        if safe_exists(zip_path):
                            zip_path.unlink()
                        local_safe_paths.append(local_safe_path)
                    except Exception as ex:
                        logging.error(f"Failed to download/extract {p_name}: {ex}")
                        if safe_exists(zip_path):
                            zip_path.unlink()
                        if safe_exists(local_safe_path):
                            shutil.rmtree(local_safe_path, ignore_errors=True)
                        continue

            if local_safe_paths:
                yield prod_date, local_safe_paths


# ================= LOGIC: SNAP PROCESSOR =================

def run_calibration_stage(track_name, safe_paths, working_dir):
    track_dir = working_dir / track_name
    calibrated_dir = track_dir / "calibrated"
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    processed_dims = []
    total = len(safe_paths)
    
    for idx, scene_path in enumerate(safe_paths, 1):
        stem = scene_path.stem
        output_dim = calibrated_dir / f"{stem}_Cal.dim"

        # CHECK IF CALIBRATED FILE EXISTS - SKIP IF SO
        if output_dim.exists() and output_dim.with_suffix('.data').is_dir():
            logging.info(f"Skipping existing calibration: {stem}")
            processed_dims.append(output_dim)
            continue

        input_file = scene_path / "manifest.safe"
        try:
            if not input_file.exists():
                input_file = scene_path / "manifest.SAFE"
            if not input_file.exists():
                logging.error(f"SKIPPING {stem}: manifest.safe not found in {scene_path}")
                continue
        except OSError as e:
            logging.error(f"SKIPPING {stem}: I/O Error checking manifest: {e}")
            continue

        xml_file = track_dir / f"{stem}_calibration.xml"

        read_node = (
            f"  <node id=\"Read\">\n"
            f"    <operator>Read</operator>\n"
            f"    <sources/>\n"
            f"    <parameters class=\"com.bc.ceres.binding.dom.XppDomElement\">\n"
            f"      <file>{str(input_file)}</file>\n"
            f"    </parameters>\n"
            f"  </node>\n"
        )

        xml_content = CALIBRATION_TEMPLATE.format(
            read_nodes=read_node,
            tnr_nodes=TNR_NODE,
            aof_nodes=AOF_NODE,
            bnr_nodes=BNR_NODE,
            calib_nodes=CALIB_NODE,
            output_file=str(output_dim)
        )

        xml_file.write_text(xml_content, encoding='utf-8')

        logging.info(f"[{track_name}] Calibrating {idx}/{total}: {stem}")
        cmd = [GPT_EXE, f"-DAuxDataPath={AUXDATA_PATH}", "-q", "4", str(xml_file)]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            output_data_dir = output_dim.with_suffix('.data')
            img_files = list(output_data_dir.glob("*.img")) if output_data_dir.is_dir() else []
            if output_dim.exists() and output_data_dir.is_dir() and len(img_files) > 0 and all(f.stat().st_size > 1024 for f in img_files):
                processed_dims.append(output_dim)
            else:
                logging.error(f"Calibration FAILED for {stem}. Corrupted or empty product created. Purging.")
                output_dim.unlink(missing_ok=True)
                shutil.rmtree(output_data_dir, ignore_errors=True)

            xml_file.unlink(missing_ok=True)
        except (subprocess.CalledProcessError, Exception) as e:
            err_msg = getattr(e, 'stderr', str(e))
            logging.error(f"Error calibrating {stem}:\n{err_msg}")
            output_dim.unlink(missing_ok=True)
            shutil.rmtree(output_dim.with_suffix('.data'), ignore_errors=True)
            xml_file.unlink(missing_ok=True)

    return processed_dims


def run_slice_assembly_stage(track_name, calibrated_dims, working_dir, roi_wkt=None):
    track_dir = working_dir / track_name
    slice_folder = track_dir / "slice_assembly"
    slice_folder.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for dim_path in calibrated_dims:
        data_dir = dim_path.with_suffix(".data")
        if not dim_path.exists() or not data_dir.exists():
            logging.warning(f"Incomplete BEAM-DIMAP product skipped: {dim_path.name} (Missing .data folder)")
            continue

        try:
            parts = dim_path.stem.split('_')
            date_str = next((p[:8] for p in parts if len(p) >= 8 and p[:8].isdigit()), "00000000")
            groups[date_str].append(dim_path)
        except Exception as e:
            logging.warning(f"Could not parse date from {dim_path.name}: {e}")

    for date_str, files in groups.items():
        if date_str == "00000000":
            continue

        if not files:
            logging.warning(f"[{track_name}] No valid calibrated files for {date_str}. Skipping.")
            continue

        files.sort(key=lambda p: p.name)
        sensor = files[0].stem.split('_')[0]
        sanitized_track = track_name.replace('/', '_')
        out_dim = slice_folder / f"{date_str}_{sanitized_track}_IW_GRDH_{sensor}.dim"

        # CHECK IF FINAL SLICE EXISTS AND IS VALID - SKIP IF SO
        data_dir = out_dim.with_suffix('.data')
        if out_dim.exists() and data_dir.is_dir():
            img_files = list(data_dir.glob("*.img"))
            if len(img_files) > 0 and all(f.stat().st_size > 1024 for f in img_files):
                logging.info(f"[{track_name}] Slice {date_str} exists and is valid, skipping.")
                continue
            else:
                logging.warning(f"[{track_name}] Found damaged or incomplete slice for {date_str}. Purging and recomputing.")
                out_dim.unlink(missing_ok=True)
                shutil.rmtree(data_dir, ignore_errors=True)

        xml_file = track_dir / f"stage2_slice_{date_str}.xml"
        if not roi_wkt:
            logging.error("No ROI WKT geometry provided for cropping. Skipping.")
            continue

        if len(files) > 1:
            logging.info(f"[{track_name}] Assembling & Cropping date {date_str} ({len(files)} slices)")
            read_nodes = []
            slice_sources = []
            app_data = []

            for idx, fpath in enumerate(files):
                node_id = "Read" if idx == 0 else f"Read{idx + 1}"
                read_nodes.append(
                    f"  <node id=\"{node_id}\">\n"
                    f"    <operator>Read</operator>\n"
                    f"    <sources/>\n"
                    f"    <parameters class=\"com.bc.ceres.binding.dom.XppDomElement\">\n"
                    f"      <file>{str(fpath)}</file>\n"
                    f"    </parameters>\n"
                    f"  </node>\n"
                )
                tag = f"sourceProduct.{idx + 1}"
                slice_sources.append(f"      <{tag} refid=\"{node_id}\"/>\n")
                app_data.append(
                    f"    <node id=\"{node_id}\"><displayPosition x=\"41.0\" y=\"{51 + 60 * idx}\"/></node>\n")

            xml_content = SLICE_ASSEMBLY_TEMPLATE.format(
                read_nodes=''.join(read_nodes),
                slice_sources=''.join(slice_sources),
                geoRegion=roi_wkt,
                app_data=''.join(app_data),
                output_file=str(out_dim)
            )

        else:
            logging.info(f"[{track_name}] Single slice found for {date_str}. Skipping Assembly, running Subset.")
            read_nodes = (
                f"  <node id=\"Read\">\n"
                f"    <operator>Read</operator>\n"
                f"    <sources/>\n"
                f"    <parameters class=\"com.bc.ceres.binding.dom.XppDomElement\">\n"
                f"      <file>{str(files[0])}</file>\n"
                f"    </parameters>\n"
                f"  </node>\n"
            )
            xml_content = SINGLE_SLICE_TEMPLATE.format(
                read_nodes=read_nodes,
                geoRegion=roi_wkt,
                output_file=str(out_dim)
            )

        xml_file.write_text(xml_content, encoding='utf-8')
        cmd = [GPT_EXE, f"-DAuxDataPath={AUXDATA_PATH}", "-q", "4", str(xml_file)]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            xml_file.unlink(missing_ok=True)
            out_data_dir = out_dim.with_suffix('.data')
            img_files = list(out_data_dir.glob("*.img")) if out_data_dir.is_dir() else []
            if out_dim.exists() and out_data_dir.is_dir() and len(img_files) > 0 and all(f.stat().st_size > 1024 for f in img_files):
                logging.info(f"Successfully created {out_dim.name}")
            else:
                logging.error(f"SliceAssembly FAILED or produced empty files for {date_str}. Purging corrupted output.")
                out_dim.unlink(missing_ok=True)
                shutil.rmtree(out_data_dir, ignore_errors=True)
        except (subprocess.CalledProcessError, Exception) as e:
            err_msg = getattr(e, 'stderr', str(e))
            logging.error(f"Error processing {date_str} (SNAP SliceAssembly failed):\n{err_msg}")
            out_dim.unlink(missing_ok=True)
            shutil.rmtree(out_dim.with_suffix('.data'), ignore_errors=True)
            xml_file.unlink(missing_ok=True)

    calibrated_dir = track_dir / "calibrated"
    if calibrated_dir.exists():
        logging.info(f"Clean up disabled by user. Keeping intermediate files in: {calibrated_dir}")


def process_orbit_cdse(
    country: str,
    orbit: int,
    start_date: str,
    end_date: str,
    exclude_winter: bool = False,
    download_dir: str = None,
    max_workers: int = 4,
    engine: str = "python",
    overwrite: bool = False,
):
    """Programmatic API to query, download, calibrate, and slice a Sentinel-1 orbit from CDSE."""
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date() if isinstance(start_date, str) else start_date
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date
    work_dir = pathlib.Path(WORKING_DIR)
    dl_dir = pathlib.Path(download_dir) if download_dir else work_dir / "S1_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    if not CDSE_USERNAME or not CDSE_PASSWORD:
        raise ValueError("CDSE_USERNAME or CDSE_PASSWORD environment variables are not set.")

    token_manager = CDSETokenManager(CDSE_USERNAME, CDSE_PASSWORD)
    country_code = country.upper()
    country_geom = get_country_geometry(country_code)
    if not country_geom:
        raise FileNotFoundError(f"Could not load boundary geometry for country {country_code}")

    env = country_geom.GetEnvelope()
    roi_wkt = f"POLYGON (({env[0]} {env[2]}, {env[1]} {env[2]}, {env[1]} {env[3]}, {env[0]} {env[3]}, {env[0]} {env[2]}))"

    use_python = str(engine).lower() == "python" or (
        str(engine).lower() == "auto" and not safe_exists(GPT_EXE)
    )
    if use_python:
        try:
            from modules.s1_python_processor import process_date_slices_python, CopernicusDEMManager
        except ImportError:
            from s1_python_processor import process_date_slices_python, CopernicusDEMManager
        dem_mgr = CopernicusDEMManager()

    finder = CDSESentinel1Finder(token_manager, dl_dir)
    track_name = f"{country_code}/orbit_{orbit}"

    logging.info(
        f"--- STARTING CDSE ORBIT: {orbit} (Country: {country_code}, Engine: {'Python' if use_python else 'SNAP'}, "
        f"Range: {start} to {end}, Exclude Winter: {exclude_winter}) ---"
    )
    for date_obj, found_safes in finder.find_products_by_orbit(
        orbit, country_geom, start, end, working_dir=work_dir, country_code=country_code, exclude_winter=exclude_winter
    ):
        date_str = date_obj.strftime("%Y%m%d")
        logging.info(f"Processing {len(found_safes)} products for date {date_obj}")
        if use_python:
            process_date_slices_python(
                track_name=track_name,
                date_str=date_str,
                safe_dirs=found_safes,
                work_dir=work_dir,
                country_geom_4326=country_geom,
                dem_manager=dem_mgr,
                overwrite=overwrite,
            )
        else:
            calibrated_files = run_calibration_stage(track_name, found_safes, work_dir)
            if not calibrated_files:
                continue
            run_slice_assembly_stage(track_name, calibrated_files, work_dir, roi_wkt=roi_wkt)
    logging.info(f"--- FINISHED CDSE ORBIT: {orbit} (Country: {country_code}) ---")


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(description="Sentinel-1 Find & Process (Copernicus Data Space Ecosystem)")
    parser.add_argument('-s', '--start_date', required=True, help="Start date YYYY-MM-DD")
    parser.add_argument('-e', '--end_date', required=True, help="End date YYYY-MM-DD")
    parser.add_argument('-c', '--country', required=True, help="Country code (e.g. PL, AT, FR...) for automatic orbit selection.")
    parser.add_argument('-o', '--orbit', type=int, help="Specify a single relative orbit number to process (skips auto-discovery).")
    parser.add_argument('-d', '--download_dir', help="Directory to cache downloaded S1 ZIP and SAFE products.")
    parser.add_argument('--engine', choices=['python', 'snap', 'auto'], default='python', help="Processing engine: 'python' (pure Python, default) or 'snap' (ESA SNAP GPT).")
    parser.add_argument('--exclude_winter', action='store_true', help="Exclude winter freeze scenes (Dec 1 - Feb 14).")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing output files.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    use_python = args.engine == 'python' or (args.engine == 'auto' and not pathlib.Path(GPT_EXE).exists())
    if not use_python:
        gpt_path = pathlib.Path(GPT_EXE)
        if not gpt_path.exists():
            logging.error(f"CRITICAL ERROR: SNAP gpt.exe not found at: {gpt_path}. Use --engine python instead.")
            sys.exit(1)

    try:
        start = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError:
        logging.error("Invalid date format. Use YYYY-MM-DD.")
        sys.exit(1)

    work_dir = pathlib.Path(WORKING_DIR)
    download_dir = pathlib.Path(args.download_dir) if args.download_dir else work_dir / "S1_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    if not CDSE_USERNAME or not CDSE_PASSWORD:
        logging.error("CRITICAL ERROR: CDSE_USERNAME or CDSE_PASSWORD environment variables are not set.")
        sys.exit(1)

    logging.info(f"Initialising CDSE Token Manager (User: {CDSE_USERNAME})...")
    token_manager = CDSETokenManager(CDSE_USERNAME, CDSE_PASSWORD)

    country_code = args.country.upper()
    country_geom = get_country_geometry(country_code)
    if not country_geom:
        logging.error(f"Could not load boundary geometry for country {country_code}")
        sys.exit(1)

    if args.orbit:
        selected_orbits = [args.orbit]
    else:
        optimizer = CDSECountryOrbitOptimizer(token_manager, country_geom)
        selected_orbits, _ = optimizer.discover_and_optimize(start, country_code=country_code)

    if not selected_orbits:
        logging.error(f"No optimal orbits found for country {country_code} in CDSE.")
        sys.exit(1)

    for orbit_num in selected_orbits:
        process_orbit_cdse(
            country=country_code,
            orbit=orbit_num,
            start_date=start,
            end_date=end,
            exclude_winter=args.exclude_winter,
            download_dir=str(download_dir),
            engine="python" if use_python else "snap",
            overwrite=args.overwrite,
        )


if __name__ == '__main__':
    main()
