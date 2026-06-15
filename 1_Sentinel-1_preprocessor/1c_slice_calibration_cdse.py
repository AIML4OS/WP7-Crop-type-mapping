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
from collections import defaultdict
from osgeo import ogr, osr
import os

# ================= CONFIGURATION =================
# How to run the script for a selected country:
# python 1c_slice_calibration_cdse.py -s 2024-10-15 -e 2024-11-30 -c PL
# python 1c_slice_calibration_cdse.py -s 2024-10-15 -e 2024-11-30 -c FR

GPT_EXE = os.environ.get("SNAP_GPT_EXE", r"D:/Program Files/esa-snap/bin/gpt.exe")

# Path to SNAP AuxData
AUXDATA_PATH = os.environ.get("SNAP_AUXDATA_PATH", r"C:/Users/Administrator/.snap/auxdata")

# Directory where processing results (calibrated/sliced) will be saved
WORKING_DIR = os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDir")

# CDSE Credentials
CDSE_USERNAME = os.environ.get("CDSE_USERNAME")
CDSE_PASSWORD = os.environ.get("CDSE_PASSWORD")

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
      <orbitType>Sentinel Restituted (Auto Download)</orbitType>
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
SHAPEFILES_DIR = pathlib.Path(os.environ.get("AIML_AUX_DIR", "D:/AIML_CropMapper_Cloud/auxiliary_files")) / "shapefiles_nuts"


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

    def get_token(self):
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        res = requests.post(token_url, data=data)
        if res.status_code == 200:
            self.token = res.json()["access_token"]
            return self.token
        else:
            raise Exception(f"Failed to authenticate with CDSE Keycloak: {res.status_code} - {res.text}")

    def get_headers(self, force_refresh=False):
        if not self.token or force_refresh:
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

class CDSESentinel1Finder:
    def __init__(self, token_manager, download_dir: pathlib.Path):
        self.token_manager = token_manager
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def find_products_by_orbit(self, orbit_num: int, target_geom: ogr.Geometry,
                               start_date: datetime.date, end_date: datetime.date,
                               working_dir: pathlib.Path = None, country_code: str = None,
                               pass_direction: str = None):
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

        # Group products by acquisition date (UTC date)
        date_groups = defaultdict(list)
        for p in products:
            start_time_str = p["ContentDate"]["Start"]
            date_str = start_time_str.split("T")[0]
            prod_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            date_groups[prod_date].append(p)

        # Iterate through the dates in chronological order
        for prod_date in sorted(date_groups.keys()):
            if working_dir and country_code:
                date_str_formatted = prod_date.strftime("%Y%m%d")
                final_output_dir = working_dir / country_code / f"orbit_{orbit_num}" / "slice_assembly"
                if final_output_dir.exists():
                    existing_finals = [f for f in final_output_dir.glob(f"{date_str_formatted}_{country_code}_orbit_{orbit_num}_*.dim") if
                                       f.with_suffix('.data').is_dir()]
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

                local_safe_path = self.download_dir / safe_folder_name
                
                # Check if already downloaded and extracted
                if local_safe_path.exists() and any(local_safe_path.iterdir()):
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
                        if zip_path.exists():
                            zip_path.unlink()
                        local_safe_paths.append(local_safe_path)
                    except Exception as ex:
                        logging.error(f"Failed to download/extract {p_name}: {ex}")
                        if zip_path.exists():
                            zip_path.unlink()
                        if local_safe_path.exists():
                            shutil.rmtree(local_safe_path)
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
            if output_dim.exists() and output_data_dir.is_dir():
                processed_dims.append(output_dim)
            else:
                logging.error(f"Calibration FAILED for {stem}. Output file or data directory not created: {output_dim}")

            xml_file.unlink(missing_ok=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error calibrating {stem}:\n{e.stderr}")

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

        # CHECK IF FINAL SLICE EXISTS - SKIP IF SO
        if out_dim.exists() and out_dim.with_suffix('.data').is_dir():
            logging.info(f"[{track_name}] Slice {date_str} exists, skipping.")
            continue

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
            logging.info(f"Successfully created {out_dim.name}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing {date_str}:\n{e.stderr}")

    calibrated_dir = track_dir / "calibrated"
    if calibrated_dir.exists():
        logging.info(f"Clean up disabled by user. Keeping intermediate files in: {calibrated_dir}")


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(description="Sentinel-1 Find & Process (Copernicus Data Space Ecosystem)")
    parser.add_argument('-s', '--start_date', required=True, help="Start date YYYY-MM-DD")
    parser.add_argument('-e', '--end_date', required=True, help="End date YYYY-MM-DD")
    parser.add_argument('-c', '--country', required=True, help="Country code (e.g. PL, AT, FR...) for automatic orbit selection.")
    parser.add_argument('-d', '--download_dir', help="Directory to cache downloaded S1 ZIP and SAFE products.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    gpt_path = pathlib.Path(GPT_EXE)
    if not gpt_path.exists():
        logging.error(f"CRITICAL ERROR: SNAP gpt.exe not found at: {gpt_path}")
        sys.exit(1)

    try:
        start = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError:
        logging.error("Invalid date format. Use YYYY-MM-DD.")
        sys.exit(1)

    work_dir = pathlib.Path(WORKING_DIR)
    
    # Resolve download directory
    if args.download_dir:
        download_dir = pathlib.Path(args.download_dir)
    else:
        download_dir = work_dir / "S1_downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    if not CDSE_USERNAME or not CDSE_PASSWORD:
        logging.error("CRITICAL ERROR: CDSE_USERNAME or CDSE_PASSWORD environment variables are not set. Please export/set them before running this script.")
        sys.exit(1)

    logging.info(f"Initialising CDSE Token Manager (User: {CDSE_USERNAME})...")
    token_manager = CDSETokenManager(CDSE_USERNAME, CDSE_PASSWORD)
    
    country_code = args.country.upper()
    country_geom = get_country_geometry(country_code)
    if not country_geom:
        logging.error(f"Could not load boundary geometry for country {country_code}")
        sys.exit(1)

    # Optimize orbit selection dynamically using CDSE
    optimizer = CDSECountryOrbitOptimizer(token_manager, country_geom)
    selected_orbits, selected_pass = optimizer.discover_and_optimize(start, country_code=country_code)

    if not selected_orbits:
        logging.error(f"No optimal orbits found for country {country_code} in CDSE.")
        sys.exit(1)

    # Get simplified bounding polygon for SNAP Subset
    env = country_geom.GetEnvelope()  # (minX, maxX, minY, maxY)
    roi_wkt = f"POLYGON (({env[0]} {env[2]}, {env[1]} {env[2]}, {env[1]} {env[3]}, {env[0]} {env[3]}, {env[0]} {env[2]}))"

    finder = CDSESentinel1Finder(token_manager, download_dir)

    for orbit_num in selected_orbits:
        logging.info(f"--- STARTING ORBIT: {orbit_num} (Country: {country_code}) ---")
        track_name = f"{country_code}/orbit_{orbit_num}"

        # 1. FIND, DOWNLOAD, EXTRACT LOOP
        for date_obj, found_safes in finder.find_products_by_orbit(
            orbit_num, country_geom, start, end, working_dir=work_dir, country_code=country_code, pass_direction=selected_pass
        ):
            logging.info(f"Processing {len(found_safes)} products for date {date_obj}")

            # 2. CALIBRATE
            calibrated_files = run_calibration_stage(track_name, found_safes, work_dir)

            if not calibrated_files:
                logging.warning(f"No files were successfully calibrated for {date_obj}. Skipping Assembly.")
                continue

            # 3. SLICE ASSEMBLY & SUBSET
            run_slice_assembly_stage(track_name, calibrated_files, work_dir, roi_wkt=roi_wkt)

        logging.info(f"--- FINISHED ORBIT: {orbit_num} (Country: {country_code}) ---")


if __name__ == '__main__':
    main()
