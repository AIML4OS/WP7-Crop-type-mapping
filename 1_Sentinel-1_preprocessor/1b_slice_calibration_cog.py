import argparse
import datetime
import logging
import pathlib
import sys
import re
import subprocess
import shutil
from collections import defaultdict
from osgeo import ogr, osr

import os

# ================= CONFIGURATION =================
# Jak uruchomić skrypt dla wybranego kraju (np. Polski - PL, Francji - FR, Austrii - AT):
# python 1b_slice_calibration_cog.py -s 2024-10-15 -e 2024-11-30 -c PL
# python 1b_slice_calibration_cog.py -s 2024-10-15 -e 2024-11-30 -c FR

GPT_EXE = os.environ.get("SNAP_GPT_EXE", r"D:/Program Files/esa-snap/bin/gpt.exe")

# Path to SNAP AuxData
AUXDATA_PATH = os.environ.get("SNAP_AUXDATA_PATH", r"C:/Users/Administrator/.snap/auxdata")

# Repository where raw .SAFE data is stored
LOCAL_REPO_PATH = os.environ.get("S1_REPO_PATH", r"Y:\Sentinel-1\SAR\IW_GRDH_1S-COG")
# Directory where processing results (calibrated/sliced) will be saved
WORKING_DIR = os.environ.get("AIML_WORKING_DIR", r"D:/AIML_CropMapper_Cloud/workingDir")


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
                'AT': [22, 29, 73, 95, 102, 124, 146, 168]
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
            if day_path.exists() and any(day_path.iterdir()):
                for safe_dir in day_path.glob("*.SAFE"):
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
        try:
            if not manifest.exists():
                manifest = safe_path / "manifest.SAFE"
            if manifest.exists():
                with open(manifest, 'r', encoding='utf-8') as f:
                    self._cache_content = f.read()
            else:
                logging.warning(f"Manifest not found for {safe_path.name}")
        except Exception as e:
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
                elif platform == 'S1B' or platform == 'S1C' or platform == 'S1D':
                    return ((abs_orbit - 27) % 175) + 1
            except ValueError:
                pass
        return None

    def _get_relative_orbit(self, safe_path: pathlib.Path):
        """Reads manifest.safe to find the relative orbit number (with fast filename-based fallback)."""
        # Fast path: check filename estimation first
        estimated = self._estimate_relative_orbit_from_name(safe_path.name)
        if estimated is not None:
            return estimated

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

    def find_products_by_orbit(self, orbit_num: int, target_geom: ogr.Geometry,
                               start_date: datetime.date, end_date: datetime.date,
                               working_dir: pathlib.Path = None, country_code: str = None,
                               pass_direction: str = None):
        """Finds Sentinel-1 SAFE directories for a given relative orbit that intersect the country's geometry."""
        first_orbit_match_date = None
        current_date = start_date

        while current_date <= end_date:
            should_scan = False
            if first_orbit_match_date is None:
                # Scan every day until the first match
                should_scan = True
            else:
                # After first match, only scan on the 6-day cycle
                if (current_date - first_orbit_match_date).days % 6 == 0:
                    should_scan = True

            if should_scan:
                day_products = []

                # --- CHECK IF FINAL SLICED PRODUCT ALREADY EXISTS ---
                if working_dir and country_code:
                    date_str = current_date.strftime("%Y%m%d")
                    final_output_dir = working_dir / country_code / f"orbit_{orbit_num}" / "slice_assembly"
                    if final_output_dir.exists():
                        # We use country_code and orbit_num in the final dim name
                        existing_finals = [f for f in final_output_dir.glob(f"{date_str}_{country_code}_orbit_{orbit_num}_*.dim") if
                                           f.with_suffix('.data').is_dir()]
                        if existing_finals:
                            logging.info(
                                f"Skipping search for {date_str}: Final output already exists ({existing_finals[0].name})")
                            current_date += datetime.timedelta(days=1)
                            continue
                # ----------------------------------------------------

                day_path = self.repo_path / str(
                    current_date.year) / f"{current_date.month:02d}" / f"{current_date.day:02d}"

                try:
                    if day_path.exists() and any(day_path.iterdir()):
                        scan_msg = f"Scanning {day_path} for orbit {orbit_num}"
                        logging.info(scan_msg)

                        for safe_dir in day_path.glob("*.SAFE"):
                            # 1. Check Orbit
                            parsed_orbit = self._get_relative_orbit(safe_dir)
                            if parsed_orbit != orbit_num:
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
                except OSError as e:
                    logging.error(f"SKIP DATE {current_date}: I/O Error accessing folder {day_path}: {e}")

                if day_products:
                    # If this is the first find, set the base date for the 6-day cycle.
                    if first_orbit_match_date is None:
                        logging.info(
                            f"   First orbit match for orbit {orbit_num} found on {current_date}. Switching to 6-day scan cycle.")
                        first_orbit_match_date = current_date

                    yield current_date, day_products

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

            # --- CRITICAL FIX: Verify output exists ---
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

    # DISABLED CLEANUP TO ALLOW RE-RUNS
    calibrated_dir = track_dir / "calibrated"
    if calibrated_dir.exists():
        logging.info(f"Clean up disabled by user. Keeping intermediate files in: {calibrated_dir}")


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(description="Sentinel-1 Find & Process (Local Y: Drive - COG)")
    parser.add_argument('-s', '--start_date', required=True, help="Start date YYYY-MM-DD")
    parser.add_argument('-e', '--end_date', required=True, help="End date YYYY-MM-DD")
    parser.add_argument('-c', '--country', required=True, help="Country code (e.g. AT, IE, NL, PT...) for automatic orbit selection.")

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

    repo = pathlib.Path(LOCAL_REPO_PATH)
    work_dir = pathlib.Path(WORKING_DIR)

    if not repo.exists():
        logging.error(f"Local repository not found at {repo}")
        sys.exit(1)

    finder = LocalSentinel1Finder(repo)

    country_code = args.country.upper()
    country_geom = get_country_geometry(country_code)
    if not country_geom:
        logging.error(f"Could not load boundary geometry for country {country_code}")
        sys.exit(1)

    # Optimize orbit selection dynamically
    optimizer = CountryOrbitOptimizer(repo, country_geom)
    selected_orbits, selected_pass = optimizer.discover_and_optimize(start, country_code=country_code)

    if not selected_orbits:
        logging.error(f"No optimal orbits found for country {country_code}.")
        sys.exit(1)

    # Get simplified bounding polygon for SNAP Subset
    env = country_geom.GetEnvelope()  # (minX, maxX, minY, maxY)
    roi_wkt = f"POLYGON (({env[0]} {env[2]}, {env[1]} {env[2]}, {env[1]} {env[3]}, {env[0]} {env[3]}, {env[0]} {env[2]}))"

    for orbit_num in selected_orbits:
        logging.info(f"--- STARTING ORBIT: {orbit_num} (Country: {country_code}) ---")
        track_name = f"{country_code}/orbit_{orbit_num}"

        # 1. FIND & PROCESS LOOP
        for date_obj, found_safes in finder.find_products_by_orbit(
            orbit_num, country_geom, start, end, working_dir=work_dir, country_code=country_code, pass_direction=selected_pass
        ):
            logging.info(f"Processing {len(found_safes)} products for date {date_obj}")

            # 2. CALIBRATE
            calibrated_files = run_calibration_stage(track_name, found_safes, work_dir)

            if not calibrated_files:
                logging.warning(f"No files were successfully calibrated for {date_obj}. Skipping Assembly.")
                continue

            # 3. SLICE ASSEMBLY & SUBSET (passing dynamic roi_wkt)
            run_slice_assembly_stage(track_name, calibrated_files, work_dir, roi_wkt=roi_wkt)

        logging.info(f"--- FINISHED ORBIT: {orbit_num} (Country: {country_code}) ---")


if __name__ == '__main__':
    main()