#!/usr/bin/env python
import os
import glob
import subprocess
import pathlib
import shutil
from collections import defaultdict
import argparse

# --- Configuration ---
# Path to SNAP Graph Processing Tool and data directories
gpt_exe = r"D:\Program Files\esa-snap\bin\gpt.exe"
base_dir = r"D:\AIML_CropMapper\\AIML_CropMapper\workingDir"
auxdata_path = r"C:\Users\slesinskip\.snap\auxdata"

# Geo-region polygons for each track
geo_regions = {
    "P1":  "POLYGON ((14.440557479858398 47.45246505737305, 17.1320743560791 47.45246505737305, 17.1320743560791 49.17211456298828, 14.440557479858398 49.17211456298828, 14.440557479858398 47.45246505737305, 14.440557479858398 47.45246505737305))",
    "P1a": "POLYGON ((14.440557479858398 47.45246505737305, 17.1320743560791 47.45246505737305, 17.1320743560791 49.17211456298828, 14.440557479858398 49.17211456298828, 14.440557479858398 47.45246505737305, 14.440557479858398 47.45246505737305))",
    "P2":  "POLYGON ((-8.08387447843179174 52.68196228582226581, -8.08387447843179174 54.11395092355871128, -5.99627779992083187 54.11395092355871128, -5.99627779992083187 52.68196228582226581, -8.08387447843179174 52.68196228582226581))",
    "P3":  "POLYGON ((5.67898359033639721 52.34101116853894808, 5.67898359033639721 53.47158269585312951, 7.54043640634247758 53.47158269585312951, 7.54043640634247758 52.34101116853894808, 5.67898359033639721 52.34101116853894808))",
    "P4":  "POLYGON ((-9.51702908045403007 38.7314903865104867, -9.51702908045403007 39.83872917391149571, -7.80898814257814 39.83872917391149571, -7.80898814257814 38.7314903865104867, -9.51702908045403007 38.7314903865104867))",
    "P4a": "POLYGON ((-9.51702908045403007 38.7314903865104867, -9.51702908045403007 39.83872917391149571, -7.80898814257814 39.83872917391149571, -7.80898814257814 38.7314903865104867, -9.51702908045403007 38.7314903865104867))"
}

# --- XML Graph templates ---
calibration_template = r'''<graph id="Graph">
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

slice_template = r'''<graph id="Graph">
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

# --- Processing node snippets ---
tnr_node = r'''  <node id="ThermalNoiseRemoval">
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
aof_node = r'''  <node id="Apply-Orbit-File">
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
bnr_node = r'''  <node id="Remove-GRD-Border-Noise">
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
calib_node = r'''  <node id="Calibration">
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

# --- Utility functions ---
def clean_filename(stem):
    return stem.replace(".SAFE", "")

def extract_date_from_parts(parts):
    for part in parts:
        if len(part) >= 8 and part[:8].isdigit():
            return part[:8]
    return "00000000"

# --- Calibration stage ---
def run_calibration_stage(selected_tracks):
    for track_folder in selected_tracks:
        track_dir = pathlib.Path(base_dir) / track_folder
        zip_dir = track_dir / "zip"
        if not zip_dir.exists():
            print(f"[{track_folder}] no zip folder, skipping")
            continue
        calibrated_dir = track_dir / "calibrated"
        calibrated_dir.mkdir(exist_ok=True)

        scenes = sorted(zip_dir.glob("*.zip"))
        for idx, scene in enumerate(scenes, 1):
            stem = clean_filename(scene.stem)
            date = extract_date_from_parts(stem.split('_'))
            output_dim = calibrated_dir / f"{stem}_Cal.dim"
            xml_file = track_dir / f"{stem}_calibration.xml"

            read_node = (
                f"  <node id=\"Read\">\n"
                f"    <operator>Read</operator>\n"
                f"    <sources/>\n"
                f"    <parameters class=\"com.bc.ceres.binding.dom.XppDomElement\">\n"
                f"      <file>{scene}</file>\n"
                f"    </parameters>\n"
                f"  </node>\n"
            )
            xml = calibration_template.format(
                read_nodes=read_node,
                tnr_nodes=tnr_node,
                aof_nodes=aof_node,
                bnr_nodes=bnr_node,
                calib_nodes=calib_node,
                output_file=output_dim
            )
            xml_file.write_text(xml, encoding='utf-8')
            print(f"[{track_folder}] Calibrating {stem} ({idx}/{len(scenes)})")
            res = subprocess.run([
                gpt_exe, f"-DAuxDataPath={auxdata_path}", "-q", "4", str(xml_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                print(f"Error calibrating {stem}: {res.stderr}")
            else:
                print(f"Calibrated -> {output_dim}")
                xml_file.unlink(missing_ok=True)

# --- Slice Assembly stage ---
def run_slice_assembly_stage(selected_tracks):
    for track_folder in selected_tracks:
        track_dir = pathlib.Path(base_dir) / track_folder
        calibrated_dir = track_dir / "calibrated"
        if not calibrated_dir.exists():
            print(f"[{track_folder}] no calibrated data, skipping slice stage")
            continue

        dims = sorted(calibrated_dir.glob("*_Cal.dim"))
        groups = defaultdict(list)
        for dim in dims:
            date = extract_date_from_parts(dim.stem.split('_'))
            groups[date].append(str(dim))

        slice_folder = track_dir / "slice_assembly"
        slice_folder.mkdir(exist_ok=True)

        for date, files in groups.items():
            sensor = pathlib.Path(files[0]).stem.split('_')[-1]
            out_dim = slice_folder / f"{date}_{track_folder}_IW_GRDH_{sensor}.dim"
            if out_dim.exists():
                print(f"[{track_folder}] slice {date} exists, skipping")
                continue

            read_nodes = []
            slice_sources = []
            app_data = []
            for idx, fpath in enumerate(files):
                node_id = "Read" if idx == 0 else f"Read{idx+1}"
                read_nodes.append(
                    f"  <node id=\"{node_id}\">\n"
                    f"    <operator>Read</operator>\n"
                    f"    <sources/>\n"
                    f"    <parameters class=\"com.bc.ceres.binding.dom.XppDomElement\">\n"
                    f"      <file>{fpath}</file>\n"
                    f"    </parameters>\n"
                    f"  </node>\n"
                )
                # Use indexed sourceProduct tags so SNAP can find each source.
                tag = f"sourceProduct.{idx+1}"
                slice_sources.append(f"      <{tag} refid=\"{node_id}\"/>\n")
                app_data.append(
                    f"    <node id=\"{node_id}\"><displayPosition x=\"41.0\" y=\"{51+60*idx}\"/></node>\n"
                )

            xml_file = track_dir / f"stage2_slice_assembly_{date}.xml"
            xml = slice_template.format(
                read_nodes=''.join(read_nodes),
                slice_sources=''.join(slice_sources),
                geoRegion=geo_regions.get(track_folder, "POLYGON EMPTY"),
                app_data=''.join(app_data),
                output_file=out_dim
            )
            xml_file.write_text(xml, encoding='utf-8')
            print(f"[{track_folder}] assembling slice {date}")
            try:
                subprocess.run([
                    gpt_exe, f"-DAuxDataPath={auxdata_path}", "-q", "4", str(xml_file)
                ], check=True)
                print(f"Sliced -> {out_dim}")
                xml_file.unlink(missing_ok=True)
            except subprocess.CalledProcessError as err:
                print(f"Error slicing {date}: {err}")

        shutil.rmtree(calibrated_dir)
        print(f"Cleaned calibrated data for {track_folder}")

# --- Main entry point ---
def main():
    parser = argparse.ArgumentParser(
        description="Run preprocessing (calibration + slice assembly) for Sentinel-1 tracks"
    )
    parser.add_argument(
        "-t", "--track", nargs='+', choices=list(geo_regions.keys()),
        help="Tracks to process. Default: all"
    )
    args = parser.parse_args()
    tracks = args.track if args.track else list(geo_regions.keys())
    print(f"Selected tracks: {tracks}")
    print("Starting calibration stage...")
    run_calibration_stage(tracks)
    print("Calibration complete. Starting slice assembly...")
    run_slice_assembly_stage(tracks)
    print("All done.")

if __name__ == "__main__":
    main()
