import os
import argparse
from pathlib import Path
import re
import sys
from datetime import datetime
from osgeo import gdal, gdalconst, ogr, osr

# python 3_AIML_S1_stack_clip.py -t P1

# ================= CONFIGURATION =================
# Update these paths to match your system
BASE_DIR = Path("D:/AIML_CropMapper_Cloud/workingDir")
SHAPEFILES_DIR = Path("D:/AIML_CropMapper_Cloud/auxiliary_files/shapefiles_nuts")

# Mapping for automatic sub-track selection
GROUP_MAP = {"P1": "P1a", "P4": "P4a", "P2": "P2a"}

# Regions to clip for each track
TRACK_REGIONS_MAP = {
    'P1': ['AT'], 'P1a': ['AT'],
    'P2': ['IE'], 'P2a': ['IE'],
    'P3': ['NL'],
    'P4': ['PT'], 'P4a': ['PT']
}

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
    
    # Determine Source SRS
    if force_src_epsg:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(force_src_epsg)
        print(f"    [Override] Forcing source SRS to EPSG:{force_src_epsg}")
    else:
        src_srs = src_layer.GetSpatialRef()
    
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(target_epsg)
    
    # Create output
    if os.path.exists(str(dst_shp)):
        driver.DeleteDataSource(str(dst_shp))
        
    dst_ds = driver.CreateDataSource(str(dst_shp))
    dst_layer = dst_ds.CreateLayer(src_layer.GetName(), geom_type=src_layer.GetGeomType(), srs=dst_srs)
    
    # Coordinate Transformation
    coord_trans = osr.CoordinateTransformation(src_srs, dst_srs)
    
    # Copy features
    src_layer.ResetReading()
    for feature in src_layer:
        geom = feature.GetGeometryRef()
        geom.Transform(coord_trans)
        
        new_feature = ogr.Feature(dst_layer.GetLayerDefn())
        new_feature.SetGeometry(geom)
        dst_layer.CreateFeature(new_feature)
        new_feature = None
        
    dst_ds = None
    src_ds = None
    return True

def stack_and_clip(track: str):
    final_dir = BASE_DIR / track / 'S1_final_preprocessing'
    out_dir = BASE_DIR / track / 'processed_raster'
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

    # --- VIRTUAL STACKING ---
    vrt_file = out_dir / f"{track}_{dr}_temp_stack.vrt"
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
    regions = TRACK_REGIONS_MAP.get(track, [])
    if not regions:
        print(f"No regions defined for {track}, skipping clip.")

    # CHANGED: Removed PREDICTOR=2 to fix SNAP compatibility
    creation_options = ['COMPRESS=DEFLATE', 'BIGTIFF=YES', 'TILED=YES']

    for region in regions:
        shp_path = SHAPEFILES_DIR / region / f"NUTS2_{region}.shp"
        out_file = out_dir / f"{region}_{track}_{dr}_VH_VV.tif"

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
            dstNodata=0,
            dstSRS='EPSG:3857',
            callback=make_progress(f"Clipping {region}")
        )
        
        ds_out = gdal.Warp(str(out_file), str(vrt_file), options=warp_opts)
        
        if ds_out:
            ds_out = None
            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"\n    Clipping complete. Output size: {size_mb:.2f} MB")
            if size_mb < 100:
                print("    WARNING: Output file is suspiciously small. Check projection overlap!")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--track', nargs='+', required=True)
    args = parser.parse_args()

    sel = set(args.track)
    for t in list(sel):
        if t in GROUP_MAP:
            sel.add(GROUP_MAP[t])

    for track in ['P1', 'P1a', 'P2', 'P2a', 'P3', 'P4', 'P4a']:
        if track in sel:
            print(f"\n=== Processing {track} ===")
            stack_and_clip(track)

if __name__ == '__main__':
    main()