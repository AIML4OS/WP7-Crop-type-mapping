import os
import argparse
import shutil
from pathlib import Path
import re
import sys
from datetime import datetime
from osgeo import gdal, gdalconst, ogr, osr

# How to run the script:
# python 3_stack_clip.py -t PL/orbit_12
# python 3_stack_clip.py -t FR/orbit_8 FR/orbit_81 FR/orbit_110

# ================= CONFIGURATION =================
# Update these paths to match your system
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = Path(os.environ.get("AIML_WORKING_DIR", str(_PROJECT_ROOT / "workingDirs")))
WORKING_DIR = BASE_DIR
SHAPEFILES_DIR = Path(os.environ.get("AIML_AUX_DIR", str(_PROJECT_ROOT / "auxiliary_files"))) / "shapefiles_nuts"



STRIP_PATTERN = re.compile(r"_(mst|slv\d+)_")


# ================= LOGIC =================

def make_progress(label):
    def callback(complete, message, unknown):
        percent = int(complete * 100)
        sys.stdout.write(f"\r    {label} progress: {percent}%")
        sys.stdout.flush()
        return 1

    return callback


def extract_date_range(name: str) -> str:
    """Extracts YYYYMMDD_YYYYMMDD from folder name."""
    m = re.search(r"(\d{8}_\d{8})", name)
    return m.group(1) if m else name


def extract_band_date(stem: str) -> datetime:
    """Extracts date (DDMonYYYY) from filename."""
    m = re.search(r"_(\d{2}[A-Za-z]{3}\d{4})(?:_|$)", stem)
    if not m:
        return datetime.min
    try:
        return datetime.strptime(m.group(1), "%d%b%Y")
    except ValueError:
        return datetime.min


def reproject_shapefile(src_shp, dst_shp, target_epsg=3857, force_src_epsg=None):
    """Reprojects a shapefile to the target EPSG code."""
    driver = ogr.GetDriverByName('ESRI Shapefile')
    src_ds = driver.Open(str(src_shp))
    if not src_ds:
        print(f"Failed to open shapefile: {src_shp}")
        return False

    src_layer = src_ds.GetLayer()
    src_srs = src_layer.GetSpatialRef()
    if not src_srs:
        if force_src_epsg:
            src_srs = osr.SpatialReference()
            src_srs.ImportFromEPSG(force_src_epsg)
        else:
            print("Shapefile has no spatial reference. Cannot reproject.")
            return False

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(src_srs, target_srs)

    if Path(dst_shp).exists():
        driver.DeleteDataSource(str(dst_shp))

    dst_ds = driver.CreateDataSource(str(dst_shp))
    dst_layer = dst_ds.CreateLayer(src_layer.GetName(), target_srs, src_layer.GetGeomType())

    layer_defn = src_layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(i)
        dst_layer.CreateField(field_defn)

    for feature in src_layer:
        geom = feature.GetGeometryRef()
        if geom:
            geom.Transform(transform)
        new_feature = ogr.Feature(dst_layer.GetLayerDefn())
        new_feature.SetFrom(feature)
        new_feature.SetGeometry(geom)
        dst_layer.CreateFeature(new_feature)
        new_feature = None

    dst_ds = None
    src_ds = None
    return True


def stack_and_clip(track: str):
    track_dir = BASE_DIR / track
    candidate_finals = [
        track_dir / '_temp_processing' / 's1_sar' / '3_coregistered',
        track_dir / 'S1_final_preprocessing',
    ]
    if os.name == 'nt':
        candidate_finals.append(Path(r"D:/AIML_CropMapper_Cloud/workingDir") / track / 'S1_final_preprocessing')
    final_dir = candidate_finals[0]
    for c in candidate_finals:
        if c.exists() and (list(c.glob('*_VH.data')) or list(c.glob('*_VV.data'))):
            final_dir = c
            break

    out_dir = BASE_DIR / track / '1_input_stacks'
    out_dir.mkdir(parents=True, exist_ok=True)

    vh_folder = next(final_dir.glob('*_VH.data'), None)
    vv_folder = next(final_dir.glob('*_VV.data'), None)

    if not vh_folder or not vv_folder:
        print(f"Skipping {track}: missing VH or VV .data folder in {final_dir}")
        return

    vh_imgs = [p for p in vh_folder.glob('*.img') if extract_band_date(p.stem) != datetime.min]
    vv_imgs = [p for p in vv_folder.glob('*.img') if extract_band_date(p.stem) != datetime.min]

    vh_imgs.sort(key=lambda p: extract_band_date(p.stem))
    vv_imgs.sort(key=lambda p: extract_band_date(p.stem))

    if not vh_imgs or not vv_imgs:
        print(f"Skipping {track}: no valid .img files found.")
        return

    if len(vh_imgs) != len(vv_imgs):
        print(f"Error {track}: Mismatch in band counts (VH: {len(vh_imgs)}, VV: {len(vv_imgs)})")
        return

    dr = extract_date_range(vh_folder.parent.name if vh_folder.parent.name.count('_') > 1 else vh_folder.name)
    if not re.search(r"\d{8}_\d{8}", dr):
        d_start = extract_band_date(vh_imgs[0].stem).strftime("%Y%m%d")
        d_end = extract_band_date(vh_imgs[-1].stem).strftime("%Y%m%d")
        dr = f"{d_end}_{d_start}"

    sanitized_track = track.replace('/', '_').replace('\\', '_')
    vrt_file = out_dir / f"{sanitized_track}_{dr}_temp_stack.vrt"
    print(f"Creating Virtual Stack for {track} ({dr}): {len(vh_imgs) + len(vv_imgs)} bands")

    input_files = [str(p) for p in vh_imgs] + [str(p) for p in vv_imgs]
    vrt_options = gdal.BuildVRTOptions(separate=True)
    ds_vrt = gdal.BuildVRT(str(vrt_file), input_files, options=vrt_options)

    for i, img_path in enumerate(vh_imgs):
        band = ds_vrt.GetRasterBand(i + 1)
        desc = STRIP_PATTERN.sub("_", img_path.stem)
        band.SetDescription(desc)

    offset = len(vh_imgs)
    for i, img_path in enumerate(vv_imgs):
        band = ds_vrt.GetRasterBand(offset + i + 1)
        desc = STRIP_PATTERN.sub("_", img_path.stem)
        band.SetDescription(desc)

    # Force Projection to EPSG:3857 if missing
    if not ds_vrt.GetProjection():
        ds_vrt.SetProjection("EPSG:3857")

    ds_vrt.FlushCache()
    ds_vrt = None
    print("    VRT created.")

    # --- CLIP FROM VRT ---
    if '/' in track or '\\' in track:
        normalized_track = track.replace('\\', '/')
        regions = [normalized_track.split('/')[0].upper()]
    elif len(track) == 2:
        regions = [track.upper()]
    else:
        print(f"Cannot resolve region for track '{track}', skipping clip.")
        return

    # CHANGED: Removed PREDICTOR=2 to fix SNAP compatibility, added NUM_THREADS for parallel compression
    creation_options = ['COMPRESS=DEFLATE', 'BIGTIFF=YES', 'TILED=YES', 'NUM_THREADS=ALL_CPUS']
    for region in regions:
        sanitized_track = track.replace('/', '_').replace('\\', '_')
        if sanitized_track.startswith(f"{region}_"):
            out_file = out_dir / f"{sanitized_track}_{dr}_VH_VV.tif"
        else:
            out_file = out_dir / f"{region}_{sanitized_track}_{dr}_VH_VV.tif"
        shp_path = SHAPEFILES_DIR / region / f"NUTS2_{region}.shp"
        if not shp_path.exists():
            geojson_path = SHAPEFILES_DIR / "NUTS_RG_01M_2021_3857.geojson"
            if geojson_path.exists():
                try:
                    shp_path.parent.mkdir(parents=True, exist_ok=True)
                    src_ds = ogr.Open(str(geojson_path), 0)
                    if src_ds is not None:
                        src_lyr = src_ds.GetLayer()
                        src_lyr.SetAttributeFilter(f"CNTR_CODE = '{region}'")
                        drv = ogr.GetDriverByName("ESRI Shapefile")
                        dst_ds = drv.CreateDataSource(str(shp_path))
                        dst_lyr = dst_ds.CreateLayer(f"NUTS2_{region}", src_lyr.GetSpatialRef(), ogr.wkbMultiPolygon)
                        for feat in src_lyr:
                            geom = feat.GetGeometryRef()
                            if geom is not None:
                                out_feat = ogr.Feature(dst_lyr.GetLayerDefn())
                                out_feat.SetGeometry(geom.Clone())
                                dst_lyr.CreateFeature(out_feat)
                                out_feat = None
                        dst_ds = None
                        src_ds = None
                except Exception as e:
                    print(f"    WARNING: Could not extract {region} from GISCO GeoJSON: {e}")
        if not shp_path.exists():
            print(f"    WARNING: Shapefile not found: {shp_path}")
            continue

        # --- REPROJECT SHAPEFILE ---
        temp_shp = out_dir / f"temp_cutline_{region}.shp"
        print(f"    Reprojecting shapefile to EPSG:3857...")

        # FIX: Force source EPSG for Ireland if the .prj is wrong
        force_epsg = 29902 if region == 'IE' else None

        if not reproject_shapefile(shp_path, temp_shp, 3857, force_src_epsg=force_epsg):
            print("    Error reprojecting shapefile. Skipping.")
            continue

        print(f"    Clipping {track} to {region} (Directly from VRT)...")

        warp_opts = gdal.WarpOptions(
            format='GTiff',
            creationOptions=creation_options,
            cutlineDSName=str(temp_shp),
            cropToCutline=True,
            srcNodata=0.0,
            dstNodata=0.0,
            dstSRS='EPSG:3857',
            multithread=True,
            callback=make_progress(f"Clipping {region}")
        )

        ds_out = gdal.Warp(str(out_file), str(vrt_file), options=warp_opts)

        if ds_out:
            print("\n    Building internal overviews (pyramids) for instant QGIS loading...")
            gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
            gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
            ds_out.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64], callback=make_progress("Building pyramids"))
            print()
            ds_out.FlushCache()
            ds_out = None
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"    Clipping complete. Output size: {size_mb:.2f} MB")
            if size_mb < 100:
                print("    WARNING: Output file is suspiciously small. Check projection overlap!")
            else:
                # Auto-cleanup raw S1_final_preprocessing folder once compressed BigTIFF is verified
                s1_final_dir = track_dir / 'S1_final_preprocessing'
                if s1_final_dir.exists():
                    print(f"    Auto-cleanup: removing S1_final_preprocessing for {track} to free disk space (~300-450 GB)...")
                    shutil.rmtree(str(s1_final_dir), ignore_errors=True)
        else:
            print("\n    Error: GDAL Warp failed.")

        # Cleanup temp shapefile
        if temp_shp.exists():
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                f = temp_shp.with_suffix(ext)
                if f.exists(): os.remove(f)

    # Cleanup VRT
    if vrt_file.exists():
        vrt_file.unlink()


def process_track(
    track: str,
    overwrite: bool = False,
    engine: str = "python",
    filter_window: int = 7,
):
    """Programmatic API for single track stacking & clipping."""
    track_clean = track.replace("\\", "/")
    parts = [p for p in track_clean.split("/") if p]
    if len(parts) >= 2 and parts[-1].startswith("orbit_"):
        country = parts[-2].upper()
        orbit = int(parts[-1].replace("orbit_", ""))
        return process_orbit_stack_and_clip(
            country=country,
            orbit=orbit,
            overwrite=overwrite,
            engine=engine,
            filter_window=filter_window,
        )
    return stack_and_clip(track)


def process_orbit_stack_and_clip(
    country: str,
    orbit: int,
    overwrite: bool = False,
    engine: str = "python",
    filter_window: int = 7,
):
    """Programmatic API for Stage 3 despeckling, NUTS2 clipping, and BigTIFF stacking."""
    c = country.upper()
    track = f"{c}/orbit_{orbit}"
    track_dir = pathlib.Path(WORKING_DIR) / track
    manifest_path = track_dir / "wrapped" / "python_stack_manifest.json"
    slice_dir = track_dir / "slice_assembly"
    s1_final_dir = track_dir / "S1_final_preprocessing"

    has_python_inputs = manifest_path.exists() or (slice_dir.exists() and len(list(slice_dir.glob("*.tif"))) > 0)
    has_snap_inputs = s1_final_dir.exists() and len(list(s1_final_dir.glob("*.dim"))) > 0

    use_python = (
        str(engine).lower() == "python"
        or (str(engine).lower() == "auto" and has_python_inputs)
    )
    if use_python and not (has_snap_inputs and not has_python_inputs):
        try:
            from modules.s1_python_processor import process_orbit_stack_and_clip_python
        except ImportError:
            from s1_python_processor import process_orbit_stack_and_clip_python
        return process_orbit_stack_and_clip_python(
            country=c,
            orbit=orbit,
            work_dir=pathlib.Path(WORKING_DIR),
            overwrite=overwrite,
            filter_window=filter_window,
        )
    else:
        return stack_and_clip(track)


def main():
    parser = argparse.ArgumentParser(description="Sentinel-1 Stage 3 Despeckling, Stacking & NUTS2 Clipping")
    parser.add_argument('-t', '--track', nargs='+', required=True, help="Track path(s) to process, e.g. PL/orbit_29")
    parser.add_argument('--engine', choices=['python', 'snap', 'auto'], default='python', help="Processing engine: 'python' (pure Python, default) or 'snap' (ESA SNAP GPT).")
    parser.add_argument('--filter_window', type=int, default=7, help="Multi-temporal Quegan speckle filter window size (default: 7).")
    parser.add_argument('--overwrite', action='store_true', help="Force recreation of the final BigTIFF stack.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    for track in sorted(list(set(args.track))):
        print(f"\n=== Processing {track} (Engine: {args.engine}) ===")
        process_track(
            track,
            overwrite=args.overwrite,
            engine=args.engine,
            filter_window=args.filter_window,
        )


if __name__ == '__main__':
    main()