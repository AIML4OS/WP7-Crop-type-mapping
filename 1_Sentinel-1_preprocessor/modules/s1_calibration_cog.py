import argparse
import datetime
import logging
import pathlib
import sys
import re
import subprocess
import shutil
import time
from collections import defaultdict
from osgeo import ogr, osr

import os

# ================= CONFIGURATION =================
# How to run the script for a selected country (e.g., Poland - PL, France - FR, Austria - AT):
# python 1b_slice_calibration_cog.py -s 2024-10-15 -e 2024-11-30 -c PL
# python 1b_slice_calibration_cog.py -s 2024-10-15 -e 2024-11-30 -c FR
#
# How to run the script for a specific orbit (e.g., 161 for Netherlands):
# python 1b_slice_calibration_cog.py -s 2024-10-15 -e 2025-09-15 -c NL -o 161

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
GPT_EXE = os.environ.get("SNAP_GPT_EXE", r"D:/Program Files/esa-snap/bin/gpt.exe")

# Path to SNAP AuxData
AUXDATA_PATH = os.environ.get("SNAP_AUXDATA_PATH", r"C:/Users/Administrator/.snap/auxdata")


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


def safe_list_safes(day_path: pathlib.Path, max_retries: int = 2) -> list[pathlib.Path]:
    """Safely and efficiently list *.SAFE directories in day_path, handling FUSE/s3fs I/O errors gracefully."""
    candidates = [day_path]
    day_str = str(day_path)
    if "IW_GRDH_1S-COG" in day_str:
        candidates.append(pathlib.Path(day_str.replace("IW_GRDH_1S-COG", "IW_GRDH_1S")))
    elif "IW_GRDH_1S" in day_str:
        candidates.append(pathlib.Path(day_str.replace("IW_GRDH_1S", "IW_GRDH_1S-COG")))

    last_err = None
    for cand in candidates:
        for attempt in range(max_retries + 1):
            try:
                entries = os.listdir(cand)
                safes = [cand / name for name in entries if name.endswith(".SAFE")]
                if safes or cand == day_path:
                    return safes
            except FileNotFoundError as e:
                last_err = e
                break
            except OSError as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(0.3)
                    continue
                break
            except Exception:
                break
    if last_err is not None:
        logging.warning(f"Could not read directory {day_path} (FUSE/s3fs I/O error or unavailable): {last_err}")
    return []



# Repository where raw .SAFE data is stored
_DEFAULT_S1_COG_REPO = "/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG" if (os.name != "nt" or safe_exists("/eodata")) else r"Y:\Sentinel-1\SAR\IW_GRDH_1S-COG"
LOCAL_REPO_PATH = os.environ.get("S1_COG_REPO_PATH", _DEFAULT_S1_COG_REPO)
# Directory where processing results (calibrated/sliced) will be saved
WORKING_DIR = os.environ.get("AIML_WORKING_DIR", str(_PROJECT_ROOT / "workingDirs"))
AUX_DIR = os.environ.get("AIML_AUX_DIR", str(_PROJECT_ROOT / "auxiliary_files"))


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


class CountryOrbitOptimizer:
    def __init__(self, repo_path: pathlib.Path, country_geom: ogr.Geometry):
        self.repo_path = repo_path
        self.country_geom = country_geom
        self.finder = LocalSentinel1Finder(repo_path)

    def _solve_set_cover(self, discovered_orbits, min_coverage_pct=0.005):
        if not discovered_orbits:
            return [], 0.0

        # Compute union footprint for each orbit and its intersection area
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

        # Greedy Set Cover Solver
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

            # Stop if the best candidate adds no significant new coverage (e.g. less than 0.5%)
            min_significant_area = min_coverage_pct * total_target_area
            if best_orbit is None or best_new_area < min_significant_area:
                break

            selected_orbits.append(best_orbit)
            remaining_target = remaining_target.Difference(orbit_intersections[best_orbit]['geom'])

        # Calculate final coverage area
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
        """Scans a period in the S1 repository to separate ascending/descending orbits,
           optimizes both using Set Cover, and selects the direction with the best coverage and minimal orbits."""
        logging.info(f"Starting orbit discovery for a {search_days}-day window starting from {start_date}...")
        
        # Define candidate orbits for common countries to speed up S3 scan
        candidate_orbits = None
        if country_code:
            country_orbits = {
                'NL': [15, 37, 88, 110, 139, 161],
                'PL': [22, 29, 73, 95, 102, 124, 146, 168, 175],
                'IE': [30, 74, 103, 132, 147],
                'FR': [8, 30, 37, 59, 81, 88, 103, 110, 132, 139, 153, 161],
                'AT': [22, 29, 73, 95, 102, 124, 146, 168],
                'PT': [45, 147, 23, 125],
                'ES': [8, 81, 88, 153, 161, 45, 147, 23, 125],
                'DE': [22, 29, 73, 95, 102, 124, 146, 168, 175],
                'IT': [44, 117, 146, 168]
            }
            candidate_orbits = country_orbits.get(country_code.upper())
            if candidate_orbits:
                logging.info(f"Using pre-filtered candidate orbits for {country_code}: {candidate_orbits}")

        discovered_asc = defaultdict(list)
        discovered_dsc = defaultdict(list)

        current_date = start_date
        end_search_date = start_date + datetime.timedelta(days=search_days)

        while current_date < end_search_date:
            day_path = self.repo_path / str(current_date.year) / f"{current_date.month:02d}" / f"{current_date.day:02d}"
            safe_dirs = safe_list_safes(day_path)
            for safe_dir in safe_dirs:
                try:
                    orbit_num = self.finder._get_relative_orbit(safe_dir)
                    if orbit_num is None:
                        continue
                        
                    # Filter by candidate list if available (avoids slow manifest read)
                    if candidate_orbits is not None and orbit_num not in candidate_orbits:
                        continue

                    footprint = self.finder._get_safe_footprint(safe_dir)
                    if footprint is None:
                        continue

                    # Check intersection with country geometry
                    if footprint.Intersects(self.country_geom):
                        pass_dir = self.finder._get_pass_direction(safe_dir)
                        if pass_dir == 'ASCENDING':
                            discovered_asc[orbit_num].append(footprint)
                        elif pass_dir == 'DESCENDING':
                            discovered_dsc[orbit_num].append(footprint)
                        else:
                            discovered_asc[orbit_num].append(footprint)
                except Exception as ex:
                    logging.warning(f"Error checking candidate SAFE {safe_dir.name}: {ex}")
                    continue

            current_date += datetime.timedelta(days=1)

        logging.info(f"Discovered orbits: {len(discovered_asc)} ASCENDING, {len(discovered_dsc)} DESCENDING.")

        logging.info("Optimizing coverage for ASCENDING passes...")
        selected_asc, area_asc = self._solve_set_cover(discovered_asc, min_coverage_pct=min_coverage_pct)
        logging.info(f"  ASCENDING: Selected {len(selected_asc)} orbits covering {area_asc:.4f} sq. degrees.")

        logging.info("Optimizing coverage for DESCENDING passes...")
        selected_dsc, area_dsc = self._solve_set_cover(discovered_dsc, min_coverage_pct=min_coverage_pct)
        logging.info(f"  DESCENDING: Selected {len(selected_dsc)} orbits covering {area_dsc:.4f} sq. degrees.")

        # Choose the best direction based on coverage first, then minimal orbits
        if not selected_asc and not selected_dsc:
            logging.error("No overlapping Sentinel-1 orbits discovered in the repository!")
            return [], None
        elif not selected_asc:
            choose_asc = False
        elif not selected_dsc:
            choose_asc = True
        else:
            diff_area = area_asc - area_dsc
            if abs(diff_area) < 0.05:
                # Coverage is nearly identical, choose the direction with fewer orbits
                if len(selected_asc) <= len(selected_dsc):
                    choose_asc = True
                else:
                    choose_asc = False
            else:
                # Choose the one with larger coverage area
                choose_asc = diff_area > 0

        if choose_asc:
            logging.info(f"Selected ASCENDING pass direction (Orbits: {selected_asc})")
            return selected_asc, 'ASCENDING'
        else:
            logging.info(f"Selected DESCENDING pass direction (Orbits: {selected_dsc})")
            return selected_dsc, 'DESCENDING'


# ================= LOGIC: FINDER =================

class LocalSentinel1Finder:
    def __init__(self, repo_path: pathlib.Path):
        self.repo_path = repo_path
        self._cache_path = None
        self._cache_content = None

    def _read_manifest(self, safe_path: pathlib.Path):
        """Reads manifest.safe with a 1-item cache to prevent redundant remote S3/rclone reads."""
        if self._cache_path == safe_path:
            return self._cache_content
            
        self._cache_path = safe_path
        self._cache_content = None
        
        manifest = safe_path / "manifest.safe"
        for attempt in range(2):
            try:
                if not safe_exists(manifest):
                    manifest = safe_path / "manifest.SAFE"
                if safe_exists(manifest):
                    with open(manifest, 'r', encoding='utf-8') as f:
                        self._cache_content = f.read()
                    break
                else:
                    logging.warning(f"Manifest not found for {safe_path.name}")
                    break
            except Exception as e:
                if attempt == 0:
                    time.sleep(0.3)
                    continue
                logging.error(f"SKIP: Disk I/O Error reading manifest for {safe_path.name}: {e}")
            
        return self._cache_content

    def _get_safe_footprint(self, safe_path: pathlib.Path):
        """Reads manifest.safe to find the footprint."""
        try:
            content = self._read_manifest(safe_path)
            if not content:
                return None

            match = re.search(r'<gml:coordinates>(.*?)</gml:coordinates>', content, re.DOTALL)
            if not match:
                match = re.search(r'<coordinates>(.*?)</coordinates>', content, re.DOTALL)

            if match:
                coord_str = match.group(1).strip()
                points = []
                for pair in coord_str.split():
                    if ',' in pair:
                        lat, lon = pair.split(',')
                        points.append(f"{lon} {lat}")

                if points:
                    if points[0] != points[-1]:
                        points.append(points[0])
                    wkt_string = f"POLYGON (({', '.join(points)}))"
                    try:
                        return ogr.CreateGeometryFromWkt(wkt_string)
                    except Exception as e:
                        logging.error(f"OGR Creation Error for {safe_path.name}. WKT invalid: {wkt_string[:50]}...")
                        return None
            else:
                logging.warning(f"No coordinates tag found in manifest for {safe_path.name}")
        except Exception as e:
            logging.warning(f"Failed to parse footprint for {safe_path.name}: {e}")
        return None

    def _estimate_relative_orbit_from_name(self, safe_name: str):
        """Estimates the relative orbit number from the filename to avoid slow disk read."""
        parts = safe_name.split('_')
        if len(parts) >= 7:
            platform = parts[0]
            try:
                abs_orbit = int(parts[6])
                if platform == 'S1A':
                    return ((abs_orbit - 73) % 175) + 1
                elif platform == 'S1B':
                    return ((abs_orbit - 27) % 175) + 1
                elif platform == 'S1C':
                    # ESA June 2026 orbital re-phasing maneuver (2026-06-09 to 2026-06-23)
                    # moved S1C into the S1D plane, shifting its relative orbit offset from 172 to 99.
                    if abs_orbit <= 8010:
                        return ((abs_orbit - 172) % 175) + 1
                    elif abs_orbit >= 8240:
                        return ((abs_orbit - 99) % 175) + 1
                    return None
                elif platform == 'S1D':
                    return ((abs_orbit - 42) % 175) + 1
            except ValueError:
                pass
        return None

    def _get_relative_orbit_from_manifest(self, safe_path: pathlib.Path):
        """Reads manifest.safe directly to extract the authoritative relativeOrbitNumber."""
        try:
            content = self._read_manifest(safe_path)
            if not content:
                return None

            match = re.search(r':relativeOrbitNumber\s+type="start">(\d+)<', content)
            if match:
                return int(match.group(1))
        except Exception as e:
            logging.warning(f"Failed to parse relative orbit for {safe_path.name}: {e}")
        return None

    def _get_relative_orbit(self, safe_path: pathlib.Path):
        """Reads manifest.safe to find the relative orbit number (with fast filename-based pre-check)."""
        estimated = self._estimate_relative_orbit_from_name(safe_path.name)
        if estimated is not None:
            return estimated
        return self._get_relative_orbit_from_manifest(safe_path)

    def _get_pass_direction(self, safe_path: pathlib.Path):
        """Reads manifest.safe to find the pass direction (ASCENDING or DESCENDING)."""
        try:
            content = self._read_manifest(safe_path)
            if not content:
                return None

            match = re.search(r':pass>(ASCENDING|DESCENDING)<', content)
            if match:
                return match.group(1)
        except Exception as e:
            logging.warning(f"Failed to parse pass direction for {safe_path.name}: {e}")
        return None

    def _query_odata_orbit_products(
        self,
        orbit_num: int,
        target_geom: ogr.Geometry,
        start_date: datetime.date,
        end_date: datetime.date,
        pass_direction: str = None,
    ):
        """
        Query the CDSE OData catalog for exact S3Path locations of Sentinel-1 slices
        matching orbit_num and target_geom envelope. Avoids calling os.listdir on
        1,000-entry s3fs /eodata directories (which triggers s3fs readdir_multi_head
        I/O errors). Returns None if the OData API is unreachable.
        """
        if not str(self.repo_path).startswith("/eodata"):
            return None
        try:
            import json
            import urllib.parse
            import urllib.request

            min_lon, max_lon, min_lat, max_lat = target_geom.GetEnvelope()
            bbox_wkt = (
                f"POLYGON(({min_lon:.4f} {min_lat:.4f},{max_lon:.4f} {min_lat:.4f},"
                f"{max_lon:.4f} {max_lat:.4f},{min_lon:.4f} {max_lat:.4f},{min_lon:.4f} {min_lat:.4f}))"
            )
            filter_parts = [
                "Collection/Name eq 'SENTINEL-1'",
                "contains(Name,'IW_GRDH_1S')",
                f"Attributes/OData.CSC.IntegerAttribute/any(att:att/Name eq 'relativeOrbitNumber' and att/Value eq {int(orbit_num)})",
                f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}')",
                f"ContentDate/Start ge {start_date.isoformat()}T00:00:00.000Z",
                f"ContentDate/Start le {end_date.isoformat()}T23:59:59.999Z",
            ]
            if pass_direction:
                filter_parts.append(
                    f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'orbitDirection' and att/Value eq '{pass_direction.upper()}')"
                )
            filter_str = " and ".join(filter_parts)

            by_date_slice = defaultdict(lambda: defaultdict(dict))
            skip = 0
            while True:
                params = urllib.parse.urlencode(
                    {
                        "$filter": filter_str,
                        "$top": 1000,
                        "$skip": skip,
                        "$orderby": "ContentDate/Start asc",
                    }
                )
                url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?{params}"
                req = urllib.request.Request(url, headers={"User-Agent": "AIML-CropMapper/3.0"})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    items = json.loads(resp.read().decode("utf-8")).get("value", [])
                for item in items:
                    name = item.get("Name", "")
                    s3p = item.get("S3Path", "")
                    if not name.endswith(".SAFE") or not s3p.startswith("/eodata"):
                        continue
                    parts = name.split("_")
                    if len(parts) < 7:
                        continue
                    d_str = parts[4][:8]
                    slice_key = f"{parts[0]}_{parts[4]}_{parts[5]}_{parts[6]}"
                    dt = datetime.date(int(d_str[:4]), int(d_str[4:6]), int(d_str[6:8]))
                    kind = "cog" if "_COG" in name else "std"
                    by_date_slice[dt][slice_key][kind] = pathlib.Path(s3p)
                if len(items) < 1000:
                    break
                skip += 1000
            return by_date_slice
        except Exception as e:
            logging.debug(f"CDSE OData catalog pre-lookup unavailable, falling back to directory scan: {e}")
            return None

    def _resolve_odata_date_safes(self, slices_dict: dict) -> list[pathlib.Path]:
        """Resolve exact SAFE directories on /eodata for a single date, preferring COG or STD based on repo_path."""
        prefer_cog = "COG" in str(self.repo_path)
        order = ("cog", "std") if prefer_cog else ("std", "cog")
        chosen = []
        for _slice_key, kinds in sorted(slices_dict.items()):
            picked = None
            for k in order:
                cand = kinds.get(k)
                if cand is not None and (safe_exists(cand / "manifest.safe") or safe_exists(cand / "manifest.SAFE")):
                    picked = cand
                    break
            if picked is None:
                picked = kinds.get(order[0]) or kinds.get(order[1])
            if picked is not None:
                chosen.append(picked)
        return chosen

    def find_products_by_orbit(self, orbit_num: int, target_geom: ogr.Geometry,
                               start_date: datetime.date, end_date: datetime.date,
                               working_dir: pathlib.Path = None, country_code: str = None,
                               pass_direction: str = None, exclude_winter: bool = False,
                               overwrite: bool = False):
        """Finds Sentinel-1 SAFE directories for a given relative orbit that intersect the country's geometry."""
        last_orbit_match_date = None
        scan_next_day = False
        current_date = start_date
        odata_by_date = self._query_odata_orbit_products(
            orbit_num=orbit_num,
            target_geom=target_geom,
            start_date=start_date,
            end_date=end_date,
            pass_direction=pass_direction,
        )

        while current_date <= end_date:
            if exclude_winter:
                m, d = current_date.month, current_date.day
                if m == 12 or m == 1 or (m == 2 and d <= 14):
                    next_year = current_date.year if (m == 1 or (m == 2 and d <= 14)) else current_date.year + 1
                    target_date = datetime.date(next_year, 2, 15)
                    logging.info(f"[INFO] Skipping winter dormancy period ({current_date.isoformat()} to {datetime.date(next_year, 2, 14).isoformat()}). Resuming scan at {target_date.isoformat()}...")
                    current_date = target_date
                    continue

            should_scan = False
            if odata_by_date is not None:
                if current_date in odata_by_date:
                    # Skip 1-day-apart S1D commissioning duplicates prior to the June 2026 constellation handover
                    if (
                        last_orbit_match_date is not None
                        and (current_date - last_orbit_match_date).days == 1
                        and current_date < datetime.date(2026, 6, 10)
                    ):
                        should_scan = False
                    else:
                        should_scan = True
            elif last_orbit_match_date is None or scan_next_day:
                should_scan = True
            else:
                days_since_match = (current_date - last_orbit_match_date).days
                if days_since_match % 6 == 0:
                    should_scan = True
                elif current_date >= datetime.date(2026, 6, 10) and (days_since_match % 6 in (1, 5)):
                    # Handle June 2026 S1A -> S1C/S1D +1 day constellation plane handover
                    should_scan = True

            if should_scan:
                day_products = []

                # --- CHECK IF FINAL SLICED PRODUCT ALREADY EXISTS ---
                if not overwrite and working_dir and country_code:
                    date_str = current_date.strftime("%Y%m%d")
                    final_output_dir = working_dir / country_code / f"orbit_{orbit_num}" / "slice_assembly"
                    if safe_is_dir(final_output_dir):
                        # We use country_code and orbit_num in the final dim name
                        existing_dims = [f for f in final_output_dir.glob(f"{date_str}_{country_code}_orbit_{orbit_num}_*.dim") if
                                           safe_is_dir(f.with_suffix('.data'))]
                        existing_tifs = [f for f in final_output_dir.glob(f"{date_str}_{country_code}_orbit_{orbit_num}_*.tif") if
                                           safe_file_size(f) > 1024]
                        py_tif = final_output_dir / f"{date_str}.tif"
                        if safe_file_size(py_tif) > 1024:
                            existing_tifs.append(py_tif)
                        existing_finals = existing_dims + existing_tifs
                        if existing_finals:
                            logging.info(
                                f"Skipping search for {date_str}: Final output already exists ({existing_finals[0].name})")
                            if last_orbit_match_date is None:
                                logging.info(
                                    f"   First orbit match for orbit {orbit_num} found on {current_date} (existing output). Switching to 6-day scan cycle.")
                            last_orbit_match_date = current_date
                            scan_next_day = False
                            current_date += datetime.timedelta(days=1)
                            continue
                # ----------------------------------------------------

                if odata_by_date is not None:
                    slices_for_day = odata_by_date.get(current_date)
                    safe_dirs = self._resolve_odata_date_safes(slices_for_day) if slices_for_day else []
                else:
                    day_path = self.repo_path / str(
                        current_date.year) / f"{current_date.month:02d}" / f"{current_date.day:02d}"
                    safe_dirs = safe_list_safes(day_path)

                for safe_dir in safe_dirs:
                    try:
                        # 1. Check Orbit (Fast filename parse first to avoid slow disk read)
                        fast_orbit = self._estimate_relative_orbit_from_name(safe_dir.name)
                        if fast_orbit is not None and fast_orbit != orbit_num:
                            continue

                        # Always verify authoritative orbit from manifest (cached for footprint below)
                        manifest_orbit = self._get_relative_orbit_from_manifest(safe_dir)
                        if manifest_orbit is not None and manifest_orbit != orbit_num:
                            continue
                        if fast_orbit is None and manifest_orbit is None:
                            continue

                        # 2. Check Pass Direction
                        if pass_direction:
                            parsed_pass = self._get_pass_direction(safe_dir)
                            if parsed_pass != pass_direction:
                                continue

                        # 3. Check Geometry
                        prod_geom = self._get_safe_footprint(safe_dir)
                        if not prod_geom:
                            logging.warning(f"   [SKIP] Geometry parse failed: {safe_dir.name}")
                            continue

                        if prod_geom.Intersects(target_geom):
                            logging.info(f"   -> MATCH: {safe_dir.name}")
                            day_products.append(safe_dir)
                    except Exception as ex:
                        logging.warning(f"Error inspecting product {safe_dir.name}: {ex}")
                        continue

                if day_products:
                    # If this is the first find, set the base date for the 6-day cycle.
                    if last_orbit_match_date is None:
                        logging.info(
                            f"   First orbit match for orbit {orbit_num} found on {current_date}. Switching to 6-day scan cycle.")
                    last_orbit_match_date = current_date
                    scan_next_day = False

                    yield current_date, day_products
                else:
                    # If a scheduled 6-day step yielded no match, also check +1 day (e.g. constellation shift)
                    scan_next_day = (
                        last_orbit_match_date is not None
                        and (current_date - last_orbit_match_date).days % 6 == 0
                    )

            current_date += datetime.timedelta(days=1)


# ================= LOGIC: PROCESSOR =================

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

            # --- CRITICAL FIX: Verify output exists and is non-empty ---
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
        # --- VALIDATE DIM/DATA PAIR ---
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

        # --- FIX: Ensure we have files to process ---
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

    # DISABLED CLEANUP TO ALLOW RE-RUNS
    calibrated_dir = track_dir / "calibrated"
    if calibrated_dir.exists():
        logging.info(f"Clean up disabled by user. Keeping intermediate files in: {calibrated_dir}")


def process_orbit_cog(
    country: str,
    orbit: int,
    start_date: str,
    end_date: str,
    exclude_winter: bool = False,
    source_repo: str = None,
    max_workers: int = 4,
    engine: str = "python",
    overwrite: bool = False,
):
    """Programmatic API to calibrate and slice a specific Sentinel-1 COG orbit."""
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date() if isinstance(start_date, str) else start_date
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date
    repo = pathlib.Path(source_repo if source_repo else LOCAL_REPO_PATH)
    work_dir = pathlib.Path(WORKING_DIR)

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

    finder = LocalSentinel1Finder(repo)
    track_name = f"{country_code}/orbit_{orbit}"

    logging.info(
        f"--- STARTING COG ORBIT: {orbit} (Country: {country_code}, Engine: {'Python' if use_python else 'SNAP'}, "
        f"Range: {start} to {end}, Exclude Winter: {exclude_winter}) ---"
    )
    for date_obj, found_safes in finder.find_products_by_orbit(
        orbit, country_geom, start, end, working_dir=work_dir, country_code=country_code, exclude_winter=exclude_winter,
        overwrite=overwrite
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
    logging.info(f"--- FINISHED COG ORBIT: {orbit} (Country: {country_code}) ---")

process_orbit = process_orbit_cog


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(description="Sentinel-1 Find & Process (Local CreoDIAS COG Repository)")
    parser.add_argument('-s', '--start_date', required=True, help="Start date YYYY-MM-DD")
    parser.add_argument('-e', '--end_date', required=True, help="End date YYYY-MM-DD")
    parser.add_argument('-c', '--country', required=True, help="Country code (e.g. AT, IE, NL, PT...) for automatic orbit selection.")
    parser.add_argument('-o', '--orbit', type=int, help="Specify a single relative orbit number to process (skips auto-discovery).")
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

    repo = pathlib.Path(LOCAL_REPO_PATH)
    if not repo.exists():
        logging.error(f"Local repository not found at {repo}")
        sys.exit(1)

    country_code = args.country.upper()
    country_geom = get_country_geometry(country_code)
    if not country_geom:
        logging.error(f"Could not load boundary geometry for country {country_code}")
        sys.exit(1)

    if args.orbit:
        selected_orbits = [args.orbit]
    else:
        optimizer = CountryOrbitOptimizer(repo, country_geom)
        selected_orbits, _ = optimizer.discover_and_optimize(start, country_code=country_code)

    if not selected_orbits:
        logging.error(f"No optimal orbits found for country {country_code}.")
        sys.exit(1)

    for orbit_num in selected_orbits:
        process_orbit_cog(
            country=country_code,
            orbit=orbit_num,
            start_date=start,
            end_date=end,
            exclude_winter=args.exclude_winter,
            source_repo=str(repo),
            engine="python" if use_python else "snap",
            overwrite=args.overwrite,
        )


if __name__ == '__main__':
    main()