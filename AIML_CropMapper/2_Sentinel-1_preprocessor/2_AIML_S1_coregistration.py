#!/usr/bin/env python
import os
import re
import glob
import subprocess
import pathlib
import argparse
from datetime import datetime

# --- Configuration ---
gpt_exe = r"D:\Program Files\esa-snap\bin\gpt.exe"
base_dir = r"D:\AIML_CropMapper\AIML_CropMapper\workingDir"
auxdata_path = r"C:\Users\slesinskip\.snap\auxdata"

# --- Correct EPSG:3857 WKT for SNAP ---
map_projection = '''PROJCS["WGS 84 / Pseudo-Mercator",
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

#########################################
# Helper: Convert YYYYMMDD to DDMonYYYY
#########################################
def format_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.strftime("%d%b%Y")
    except Exception:
        return date_str

#########################################
# Date extraction from filename
#########################################
def extract_date(filename):
    m = re.search(r'(\d{8})', os.path.basename(filename))
    return m.group(1) if m else "00000000"

#########################################
# XML Templates
#########################################
template_wrap = r'''<graph id="Graph">
  <version>1.0</version>
  <!-- Read input products -->
  <node id="ProductSet-Reader">
    <operator>ProductSet-Reader</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <fileList>{file_list}</fileList>
    </parameters>
  </node>
  <!-- Create Stack -->
  <node id="CreateStack">
    <operator>CreateStack</operator>
    <sources>
      <sourceProduct.2 refid="ProductSet-Reader"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <masterBands/>
      <sourceBands/>
      <resamplingType>NONE</resamplingType>
      <extent>Master</extent>
      <initialOffsetMethod>Product Geolocation</initialOffsetMethod>
    </parameters>
  </node>
  <!-- Cross-Correlation -->
  <node id="Cross-Correlation">
    <operator>Cross-Correlation</operator>
    <sources>
      <sourceProduct refid="CreateStack"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <!-- parameters omitted for brevity -->
    </parameters>
  </node>
  <!-- Warp -->
  <node id="Warp">
    <operator>Warp</operator>
    <sources>
      <sourceProduct refid="Cross-Correlation"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <!-- parameters omitted for brevity -->
    </parameters>
  </node>
  <!-- Write wrapped product -->
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

template_vh = r'''<graph id="Graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <useAdvancedOptions>true</useAdvancedOptions>
      <file>{wrapped_file}</file>
      <copyMetadata>true</copyMetadata>
      <bandNames>{vh_bandNames}</bandNames>
      <maskNames/>
    </parameters>
  </node>
  <node id="Multi-Temporal-Speckle-Filter">
    <operator>Multi-Temporal-Speckle-Filter</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands>{vh_bandNames}</sourceBands>
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
    <sources><sourceProduct refid="Terrain-Correction"/></sources>
  </node>
  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources><sourceProduct refid="LinearToFromdB"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <filter>Median</filter>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="Speckle-Filter"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_vh}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>'''

template_vv = r'''<graph id="Graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <useAdvancedOptions>true</useAdvancedOptions>
      <file>{wrapped_file}</file>
      <copyMetadata>true</copyMetadata>
      <bandNames>{vv_bandNames}</bandNames>
      <maskNames/>
    </parameters>
  </node>
  <node id="Multi-Temporal-Speckle-Filter">
    <operator>Multi-Temporal-Speckle-Filter</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <sourceBands>{vv_bandNames}</sourceBands>
      <filter>Lee Sigma</filter>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="Multi-Temporal-Speckle-Filter"/></sources>
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
    <sources><sourceProduct refid="Terrain-Correction"/></sources>
  </node>
  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources><sourceProduct refid="LinearToFromdB"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <filter>Median</filter>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="Speckle-Filter"/></sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>{output_vv}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>'''

#########################################
# Processing functions and main
#########################################
def run_process(cmd, xml_filename):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Error during processing:")
        print(result.stderr)
        return False
    os.remove(xml_filename)
    return True

def process_stage(xml_content, xml_filename):
    with open(xml_filename, 'w') as f:
        f.write(xml_content)
    cmd = [gpt_exe, f"-DAuxDataPath={auxdata_path}", '-q', '4', xml_filename]
    return run_process(cmd, xml_filename)

def process_wrap(file_list, wrapped_file, track_dir):
    print("Stage 1/3 (33%) - Wrapping")
    return process_stage(template_wrap.format(
        file_list=file_list,
        wrapped_file=wrapped_file
    ), track_dir / 'wrap.xml')

def process_vh(wrapped_file, output_vh, track_dir, vh_bandNames):
    print("Stage 2/3 (66%) - VH processing")
    return process_stage(template_vh.format(
        wrapped_file=wrapped_file,
        vh_bandNames=vh_bandNames,
        map_projection=map_projection,
        output_vh=output_vh
    ), track_dir / 'vh.xml')

def process_vv(wrapped_file, output_vv, track_dir, vv_bandNames):
    print("Stage 3/3 (100%) - VV processing")
    return process_stage(template_vv.format(
        wrapped_file=wrapped_file,
        vv_bandNames=vv_bandNames,
        map_projection=map_projection,
        output_vv=output_vv
    ), track_dir / 'vv.xml')

def cleanup_track(track_dir):
    wrapped_folder = track_dir / 'wrapped'
    if wrapped_folder.exists():
        for f in wrapped_folder.glob('*'):
            f.unlink()
        wrapped_folder.rmdir()
    for xml in track_dir.glob('*.xml'):
        xml.unlink()

def run_full_processing(selected_tracks):
    total = len(selected_tracks)
    for idx, track in enumerate(selected_tracks, 1):
        print(f"\nProcessing track {idx}/{total}: {track}")
        track_dir = pathlib.Path(base_dir) / track
        slice_folder = track_dir / 'slice_assembly'
        if not slice_folder.exists():
            print(f"No slice_assembly folder for track {track}")
            continue
        input_files = sorted(slice_folder.glob('*.dim'))
        if len(input_files) < 3:
            print(f"Not enough files in {track}")
            continue
        files_with_dates = [(f, int(extract_date(f.name))) for f in input_files]
        files_with_dates.sort(key=lambda x: x[1])
        first_date, last_date = str(files_with_dates[0][1]), str(files_with_dates[-1][1])
        vh_bandNames = ','.join(
            f"Sigma0_VH_{'mst' if i==0 else 'slv'+str(2*i-1)}_{format_date(str(d).zfill(8))}"
            for i, (f, d) in enumerate(files_with_dates)
        )
        vv_bandNames = ','.join(
            f"Sigma0_VV_{'mst' if i==0 else 'slv'+str(2*i)}_{format_date(str(d).zfill(8))}"
            for i, (f, d) in enumerate(files_with_dates)
        )
        wrapped_folder = track_dir / 'wrapped'
        wrapped_folder.mkdir(exist_ok=True)
        wrapped_file = str(wrapped_folder / f"wrapped_{first_date}_{last_date}.dim")
        out_folder = track_dir / 'S1_final_preprocessing'
        out_folder.mkdir(exist_ok=True)
        output_vh = str(out_folder / f"{last_date}_{first_date}_VH.dim")
        output_vv = str(out_folder / f"{last_date}_{first_date}_VV.dim")

        if process_wrap(','.join(str(f) for f, _ in files_with_dates), wrapped_file, track_dir):
            if process_vh(wrapped_file, output_vh, track_dir, vh_bandNames) \
               and process_vv(wrapped_file, output_vv, track_dir, vv_bandNames):
                cleanup_track(track_dir)
                print(f"Finished {track}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', nargs='+', choices=['P1','P2','P3','P4','P1a','P4a'], required=True)
    args = parser.parse_args()
    run_full_processing(args.track)

if __name__ == '__main__':
    main()
