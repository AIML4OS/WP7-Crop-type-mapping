#!/usr/bin/env python
import os
import re
import subprocess
import pathlib
import argparse
import logging
import shutil
import sys
from datetime import datetime

# python 2_AIML_S1_coregistration.py -t P1 P1a P2 P2a P3 P4 P4a
# python 2_AIML_S1_coregistration.py -t P2


# ================= CONFIGURATION =================
# Using your CURRENT working paths
GPT_EXE = r"D:/Program Files/esa-snap/bin/gpt.exe"
BASE_DIR = r"D:/AIML_CropMapper_Cloud/workingDir"
AUXDATA_PATH = r"C:/Users/Administrator/.snap/auxdata"

# Correct EPSG:3857 WKT for SNAP (From your old script)
MAP_PROJECTION = '''PROJCS["WGS 84 / Pseudo-Mercator",
  GEOGCS["WGS 84",
    DATUM["World Geodetic System 1984",
      SPHEROID["WGS 84",6378137.0,298.257223563,AUTHORITY["EPSG","7030"]],
      AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0.0,AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.017453292519943295],
    AXIS["Geodetic longitude",EAST],
    AXIS["Geodetic latitude",NORTH],
    AUTHORITY["EPSG","4326"]],
  PROJECTION["Popular Visualisation Pseudo Mercator",AUTHORITY["EPSG","1024"]],
  PARAMETER["semi_minor",6378137.0],
  PARAMETER["latitude_of_origin",0.0],
  PARAMETER["central_meridian",0.0],
  PARAMETER["scale_factor",1.0],
  PARAMETER["false_easting",0.0],
  PARAMETER["false_northing",0.0],
  UNIT["m",1.0],
  AXIS["Easting",EAST],
  AXIS["Northing",NORTH],
  AUTHORITY["EPSG","3857"]]'''


# ================= HELPERS =================

def format_date(date_str):
    """Converts YYYYMMDD to DDMonYYYY (e.g., 20241022 -> 22Oct2024) for SNAP band naming."""
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.strftime("%d%b%Y")
    except Exception:
        return date_str


def extract_date(filename):
    """Finds the first 8-digit number in the filename."""
    m = re.search(r'(\d{8})', os.path.basename(filename))
    return m.group(1) if m else "00000000"


def run_gpt(xml_path):
    """Executes SNAP GPT."""
    cmd = [GPT_EXE, f"-DAuxDataPath={AUXDATA_PATH}", '-q', '4', str(xml_path)]
    logging.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("GPT Processing Failed!")
        logging.error(e.stderr)
        return False


# ================= XML TEMPLATES =================

# Stage 1: Coregistration (Wrap)
# Uses ProductSet-Reader -> CreateStack -> Cross-Correlation -> Warp
TEMPLATE_WRAP = r'''<graph id="Graph">
  <version>1.0</version>
  <node id="ProductSet-Reader">
    <operator>ProductSet-Reader</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <fileList>{file_list}</fileList>
    </parameters>
  </node>
  <node id="CreateStack">
    <operator>CreateStack</operator>
    <sources>
      <sourceProduct refid="ProductSet-Reader"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <masterBands/>
      <sourceBands/>
      <resamplingType>NONE</resamplingType>
      <extent>Master</extent>
      <initialOffsetMethod>Product Geolocation</initialOffsetMethod>
    </parameters>
  </node>
  <node id="Cross-Correlation">
    <operator>Cross-Correlation</operator>
    <sources>
      <sourceProduct refid="CreateStack"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <numGCPtoGenerate>2000</numGCPtoGenerate>
      <coarseRegistrationWindowWidth>128</coarseRegistrationWindowWidth>
      <coarseRegistrationWindowHeight>128</coarseRegistrationWindowHeight>
      <rowInterpFactor>2</rowInterpFactor>
      <columnInterpFactor>2</columnInterpFactor>
      <maxIteration>10</maxIteration>
      <gcpTolerance>0.5</gcpTolerance>
      <applyFineRegistration>true</applyFineRegistration>
      <inSAROptimized>true</inSAROptimized>
      <fineRegistrationWindowWidth>32</fineRegistrationWindowWidth>
      <fineRegistrationWindowHeight>32</fineRegistrationWindowHeight>
      <fineRegistrationWindowAccAzimuth>16</fineRegistrationWindowAccAzimuth>
      <fineRegistrationWindowAccRange>16</fineRegistrationWindowAccRange>
      <fineRegistrationOversampling>16</fineRegistrationOversampling>
      <coherenceWindowSize>3</coherenceWindowSize>
      <coherenceThreshold>0.6</coherenceThreshold>
      <useSlidingWindow>false</useSlidingWindow>
      <computeOffset>true</computeOffset>
      <onlyGCPsOnLand>false</onlyGCPsOnLand>
    </parameters>
  </node>
  <node id="Warp">
    <operator>Warp</operator>
    <sources>
      <sourceProduct refid="Cross-Correlation"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <rmsThreshold>0.05</rmsThreshold>
      <warpPolynomialOrder>1</warpPolynomialOrder>
      <interpolationMethod>Cubic convolution (6 points)</interpolationMethod>
      <demRefinement>false</demRefinement>
      <demName>SRTM 3Sec</demName>
      <excludeMaster>false</excludeMaster>
    </parameters>
  </node>
  <node id="WriteWrap">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Warp"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{wrapped_file}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>'''

# Stage 2 & 3: EXACT PIPELINE FROM OLD SCRIPT
# Read -> MTSF -> TC -> Lin2dB -> Speckle(Median) -> Write
TEMPLATE_FILTER_TC_FULL = r'''<graph id="Graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <useAdvancedOptions>true</useAdvancedOptions>
      <file>{wrapped_file}</file>
      <copyMetadata>true</copyMetadata>
      <bandNames>{band_names}</bandNames>
      <maskNames/>
    </parameters>
  </node>
  <node id="Multi-Temporal-Speckle-Filter">
    <operator>Multi-Temporal-Speckle-Filter</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands>{band_names}</sourceBands>
      <filter>Lee Sigma</filter>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Multi-Temporal-Speckle-Filter"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>SRTM 3Sec</demName>
      <externalDEMFile/>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <externalDEMApplyEGM>true</externalDEMApplyEGM>
      <demResamplingMethod>NEAREST_NEIGHBOUR</demResamplingMethod>
      <imgResamplingMethod>NEAREST_NEIGHBOUR</imgResamplingMethod>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <pixelSpacingInDegree>8.983152841195215E-5</pixelSpacingInDegree>
      <mapProjection>{map_projection}</mapProjection>
    </parameters>
  </node>
  <node id="LinearToFromdB">
    <operator>LinearToFromdB</operator>
    <sources>
        <sourceProduct refid="Terrain-Correction"/>
    </sources>
  </node>
  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources>
        <sourceProduct refid="LinearToFromdB"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <filter>Median</filter>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="Speckle-Filter"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_file}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>'''


# ================= LOGIC =================

def process_wrap(files_with_dates, wrapped_file, track_dir):
    logging.info(">>> PROGRESS: [33%] Starting Stage 1/3: Coregistration (Wrapping) <<<")
    xml_path = track_dir / 'wrap.xml'

    # Use ProductSet-Reader logic from old script
    file_list_str = ','.join(str(f[0]).replace('\\', '/') for f in files_with_dates)

    xml_content = TEMPLATE_WRAP.format(
        file_list=file_list_str,
        wrapped_file=str(wrapped_file).replace('\\', '/')
    )

    xml_path.write_text(xml_content, encoding='utf-8')
    success = run_gpt(xml_path)

    if success:
        xml_path.unlink(missing_ok=True)
    return success


def process_polarization(wrapped_file, output_file, track_dir, band_names, pol):
    if pol == 'VH':
        logging.info(">>> PROGRESS: [66%] Starting Stage 2/3: Processing VH Polarization <<<")
    else:
        logging.info(">>> PROGRESS: [100%] Starting Stage 3/3: Processing VV Polarization <<<")

    xml_path = track_dir / f'{pol.lower()}.xml'

    # Using the full old-script pipeline
    xml_content = TEMPLATE_FILTER_TC_FULL.format(
        wrapped_file=str(wrapped_file).replace('\\', '/'),
        band_names=band_names,
        map_projection=MAP_PROJECTION,
        output_file=str(output_file).replace('\\', '/')
    )

    xml_path.write_text(xml_content, encoding='utf-8')
    success = run_gpt(xml_path)

    if success:
        xml_path.unlink(missing_ok=True)
    return success


def run_full_processing(selected_tracks, overwrite=False):
    work_dir = pathlib.Path(BASE_DIR)

    for track in selected_tracks:
        logging.info(f"\n=== Processing Track: {track} ===")
        track_dir = work_dir / track
        slice_folder = track_dir / 'slice_assembly'

        if not slice_folder.exists():
            logging.warning(f"No slice_assembly folder for {track}. Skipping.")
            continue

        input_files = sorted(slice_folder.glob('*.dim'))

        if len(input_files) < 2:
            logging.warning(f"Not enough files in {track}. Needs at least 2.")
            continue

        files_with_dates = []
        for f in input_files:
            d_str = extract_date(f.name)
            files_with_dates.append((f, d_str))

        files_with_dates.sort(key=lambda x: x[1])

        first_date = files_with_dates[0][1]
        last_date = files_with_dates[-1][1]

        # Band Naming Logic (Old Script: 2*i-1 and 2*i)
        vh_bands = []
        vv_bands = []

        for i, (f, d_str) in enumerate(files_with_dates):
            snap_date = format_date(d_str)
            if i == 0:
                vh_bands.append(f"Sigma0_VH_mst_{snap_date}")
                vv_bands.append(f"Sigma0_VV_mst_{snap_date}")
            else:
                vh_idx = 2 * i - 1
                vv_idx = 2 * i
                vh_bands.append(f"Sigma0_VH_slv{vh_idx}_{snap_date}")
                vv_bands.append(f"Sigma0_VV_slv{vv_idx}_{snap_date}")

        vh_band_str = ','.join(vh_bands)
        vv_band_str = ','.join(vv_bands)

        wrapped_folder = track_dir / 'wrapped'
        wrapped_folder.mkdir(exist_ok=True)
        wrapped_file = wrapped_folder / f"wrapped_{first_date}_{last_date}.dim"

        out_folder = track_dir / 'S1_final_preprocessing'
        out_folder.mkdir(exist_ok=True)
        output_vh = out_folder / f"{last_date}_{first_date}_VH.dim"
        output_vv = out_folder / f"{last_date}_{first_date}_VV.dim"

        # --- STAGE 1: WRAPPING ---
        run_wrap = True
        if wrapped_file.exists():
            if overwrite:
                logging.info("Overwrite active: Deleting old wrapped file.")
                try:
                    if wrapped_file.exists(): wrapped_file.unlink()
                    data_dir = wrapped_file.with_suffix(".data")
                    if data_dir.exists(): shutil.rmtree(data_dir)
                except Exception as e:
                    logging.error(f"Failed to delete old file: {e}")
            else:
                logging.info("Wrapped file exists. Skipping Stage 1.")
                run_wrap = False

        if run_wrap:
            if not process_wrap(files_with_dates, wrapped_file, track_dir):
                logging.error("Wrapping failed. Stopping.")
                continue

        # --- STAGE 2: VH ---
        if not process_polarization(wrapped_file, output_vh, track_dir, vh_band_str, 'VH'):
            logging.error("VH Processing failed.")

        # --- STAGE 3: VV ---
        if not process_polarization(wrapped_file, output_vv, track_dir, vv_band_str, 'VV'):
            logging.error("VV Processing failed.")

        logging.info(f"Finished {track}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--track', nargs='+', choices=['P1', 'P2', 'P3', 'P4', 'P1a', 'P4a'], required=True)
    parser.add_argument('--overwrite', action='store_true', help="Force recreation of the wrapped stack file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    if not pathlib.Path(GPT_EXE).exists():
        logging.error(f"GPT not found at {GPT_EXE}")
        sys.exit(1)

    run_full_processing(args.track, args.overwrite)


if __name__ == '__main__':
    main()