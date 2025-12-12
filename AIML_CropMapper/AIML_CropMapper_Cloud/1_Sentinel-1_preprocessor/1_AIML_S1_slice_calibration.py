import argparse
import datetime
import logging
import pathlib
import sys
import re
import subprocess
import shutil
from collections import defaultdict
from osgeo import ogr

# ================= CONFIGURATION =================
# python 1_AIML_S1_slice_calibration.py -s 2024-10-15 -e 2024-11-30 -t P1

# TODO: USER - VERIFY THIS PATH IS CORRECT
GPT_EXE = r"D:/Program Files/esa-snap/bin/gpt.exe"

# Path to SNAP AuxData
AUXDATA_PATH = r"C:/Users/Administrator/.snap/auxdata"

# Repository where raw .SAFE data is stored
LOCAL_REPO_PATH = r"Y:\Sentinel-1\SAR\IW_GRDH_1S"
# Directory where processing results (calibrated/sliced) will be saved
WORKING_DIR = r"D:/AIML/WP7-Crop-type-mapping/AIML_CropMapper/workingDir"

# Track Cycle Definitions (Reference dates for 6-day repeat cycle)
BASE_DATES = {
    'P1': datetime.date(2019, 3, 11), 'P1a': datetime.date(2019, 3, 12),
    'P2': datetime.date(2019, 3, 13), 'P3': datetime.date(2019, 3, 19),
    'P4': datetime.date(2019, 3, 16), 'P4a': datetime.date(2019, 3, 17)
}

# Geo-region polygons (WKT)
GEO_REGIONS = {
    "P1": "POLYGON ((14.440557479858398 47.45246505737305, 17.1320743560791 47.45246505737305, 17.1320743560791 49.17211456298828, 14.440557479858398 49.17211456298828, 14.440557479858398 47.45246505737305, 14.440557479858398 47.45246505737305))",
    "P1a": "POLYGON ((14.440557479858398 47.45246505737305, 17.1320743560791 47.45246505737305, 17.1320743560791 49.17211456298828, 14.440557479858398 49.17211456298828, 14.440557479858398 47.45246505737305, 14.440557479858398 47.45246505737305))",
    "P2": "POLYGON ((-8.08387447843179174 52.68196228582226581, -8.08387447843179174 54.11395092355871128, -5.99627779992083187 54.11395092355871128, -5.99627779992083187 52.68196228582226581, -8.08387447843179174 52.68196228582226581))",
    "P3": "POLYGON ((5.67898359033639721 52.34101116853894808, 5.67898359033639721 53.47158269585312951, 7.54043640634247758 53.47158269585312951, 7.54043640634247758 52.34101116853894808, 5.67898359033639721 52.34101116853894808))",
    "P4": "POLYGON ((-9.51702908045403007 38.7314903865104867, -9.51702908045403007 39.83872917391149571, -7.80898814257814 39.83872917391149571, -7.80898814257814 38.7314903865104867, -9.51702908045403007 38.7314903865104867, -9.51702908045403007 38.7314903865104867))",
    "P4a": "POLYGON ((-9.51702908045403007 38.7314903865104867, -9.51702908045403007 39.83872917391149571, -7.80898814257814 39.83872917391149571, -7.80898814257814 38.7314903865104867, -9.51702908045403007 38.7314903865104867, -9.51702908045403007 38.7314903865104867))"
}

# ================= XML TEMPLATES =================

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


# ================= LOGIC: FINDER =================

class LocalSentinel1Finder:
    def __init__(self, repo_path: pathlib.Path):
        self.repo_path = repo_path
        self.poly_geoms = {}
        for name, wkt in GEO_REGIONS.items():
            self.poly_geoms[name] = ogr.CreateGeometryFromWkt(wkt)

    def _get_safe_footprint(self, safe_path: pathlib.Path):
        """Reads manifest.safe to find the footprint."""
        try:
            manifest = safe_path / "manifest.safe"
            try:
                if not manifest.exists():
                    manifest = safe_path / "manifest.SAFE"
                if not manifest.exists():
                    return None
            except OSError as e:
                logging.error(f"SKIP: Disk I/O Error checking existence of {safe_path.name}: {e}")
                return None

            try:
                with open(manifest, 'r', encoding='utf-8') as f:
                    content = f.read()
            except OSError as e:
                logging.error(f"SKIP: Disk I/O Error reading {manifest}: {e}")
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
        except Exception as e:
            logging.warning(f"Failed to parse footprint for {safe_path.name}: {e}")
        return None

    def find_products(self, track_name: str, start_date: datetime.date, end_date: datetime.date,
                      working_dir: pathlib.Path = None):
        found_products = []
        base_date = BASE_DATES.get(track_name)
        geo_key = track_name
        target_geom = self.poly_geoms.get(geo_key)

        if not target_geom:
            logging.error(f"No geometry defined for {track_name}")
            return []

        current_date = start_date
        while current_date <= end_date:
            if (current_date - base_date).days % 6 == 0:

                # --- CHECK IF FINAL SLICED PRODUCT ALREADY EXISTS ---
                # This prevents searching Y: drive if work is already done.
                if working_dir:
                    date_str = current_date.strftime("%Y%m%d")
                    final_output_dir = working_dir / track_name / "slice_assembly"
                    # We check for any file matching the date and track pattern (S1A or S1C)
                    # Pattern example: 20250403_P1a_IW_GRDH_*.dim
                    if final_output_dir.exists():
                        existing_finals = list(final_output_dir.glob(f"{date_str}_{track_name}_*.dim"))
                        if existing_finals:
                            logging.info(
                                f"Skipping search for {date_str}: Final output already exists ({existing_finals[0].name})")
                            current_date += datetime.timedelta(days=1)
                            continue
                # ----------------------------------------------------

                day_path = self.repo_path / str(
                    current_date.year) / f"{current_date.month:02d}" / f"{current_date.day:02d}"

                try:
                    if day_path.exists():
                        logging.info(f"Scanning {day_path} for track {track_name}...")
                        for safe_dir in day_path.glob("*.SAFE"):
                            prod_geom = self._get_safe_footprint(safe_dir)
                            if prod_geom and prod_geom.Intersects(target_geom):
                                logging.info(f"   -> Found Match: {safe_dir.name}")
                                found_products.append(safe_dir)
                    else:
                        logging.debug(f"Skipping {current_date} (No folder)")
                except OSError as e:
                    logging.error(f"SKIP DATE {current_date}: I/O Error accessing folder {day_path}: {e}")

            current_date += datetime.timedelta(days=1)

        return found_products


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
        if output_dim.exists():
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
            processed_dims.append(output_dim)
            xml_file.unlink(missing_ok=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error calibrating {stem}:\n{e.stderr}")

    return processed_dims


def run_slice_assembly_stage(track_name, calibrated_dims, working_dir):
    track_dir = working_dir / track_name
    slice_folder = track_dir / "slice_assembly"
    slice_folder.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for dim_path in calibrated_dims:
        try:
            parts = dim_path.stem.split('_')
            date_str = next((p[:8] for p in parts if len(p) >= 8 and p[:8].isdigit()), "00000000")
            groups[date_str].append(dim_path)
        except Exception:
            logging.warning(f"Could not parse date from {dim_path.name}")

    for date_str, files in groups.items():
        if date_str == "00000000":
            continue

        files.sort(key=lambda p: p.name)
        sensor = files[0].stem.split('_')[0]
        out_dim = slice_folder / f"{date_str}_{track_name}_IW_GRDH_{sensor}.dim"

        # CHECK IF FINAL SLICE EXISTS - SKIP IF SO
        if out_dim.exists():
            logging.info(f"[{track_name}] Slice {date_str} exists, skipping.")
            continue

        xml_file = track_dir / f"stage2_slice_{date_str}.xml"
        roi_wkt = GEO_REGIONS.get(track_name, "")

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
    parser = argparse.ArgumentParser(description="Sentinel-1 Find & Process (Local Y: Drive)")
    parser.add_argument('-s', '--start_date', required=True, help="Start date YYYY-MM-DD")
    parser.add_argument('-e', '--end_date', required=True, help="End date YYYY-MM-DD")
    parser.add_argument('-t', '--track', action='append', choices=list(BASE_DATES.keys()),
                        help="Track(s) to process. If omitted, all are processed.")

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
    tracks_to_process = args.track or list(BASE_DATES.keys())

    for track in tracks_to_process:
        logging.info(f"--- STARTING TRACK: {track} ---")

        # 1. FIND (Updated to pass work_dir for intelligent skipping)
        found_safes = finder.find_products(track, start, end, working_dir=work_dir)

        if not found_safes:
            logging.warning(f"No new products found for {track} in range {start} to {end}.")
            continue

        logging.info(f"Found {len(found_safes)} products to process.")

        # 2. CALIBRATE
        calibrated_files = run_calibration_stage(track, found_safes, work_dir)

        if not calibrated_files:
            logging.warning("No files were successfully calibrated. Skipping Assembly.")
            continue

        # 3. SLICE ASSEMBLY & SUBSET
        run_slice_assembly_stage(track, calibrated_files, work_dir)

        logging.info(f"--- FINISHED TRACK: {track} ---")


if __name__ == '__main__':
    main()