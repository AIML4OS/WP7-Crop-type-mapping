import os
import argparse
from pathlib import Path
import re
from datetime import datetime
from osgeo import gdal, gdalconst

# Configuration
base_dir = Path("D:/AIML_CropMapper/AIML_CropMapper/workingDir")
shapefiles_dir = Path("D:/AIML_CropMapper/auxiliary_files/shapefiles_nuts")
group_map = {"P1": "P1a", "P4": "P4a"}
strip_pattern = re.compile(r"_(mst|slv\d+)_")
track_regions_map = {
    'P1': ['AU'], 'P1a': ['AU'],
    'P2': ['IR'], 'P3': ['NL'],
    'P4': ['PT'], 'P4a': ['PT']
}

# GDAL progress callback
def make_progress(label):
    def callback(complete, message, unknown):
        percent = int(complete * 100)
        print(f"    {label} progress: {percent}%", end="\r")
        return 1
    return callback

# Extract date range from folder name (YYYYMMDD_YYYYMMDD)
def extract_date_range(name: str) -> str:
    m = re.search(r"(\d{8}_\d{8})", name)
    return m.group(1) if m else name

# Extract single-date from filename for sorting
def extract_band_date(stem: str) -> datetime:
    m = re.search(r"_(\d{2}[A-Za-z]{3}\d{4})_", stem)
    if not m:
        return datetime.min
    try:
        return datetime.strptime(m.group(1), "%d%b%Y")
    except ValueError:
        return datetime.min

# Main processing: stack and clip using GDAL
def stack_and_clip(track: str):
    final_dir = base_dir / track / 'S1_final_preprocessing'
    out_dir = base_dir / track / 'processed_raster'
    out_dir.mkdir(parents=True, exist_ok=True)

    vh_folder = next(final_dir.glob('*_VH.data'), None)
    vv_folder = next(final_dir.glob('*_VV.data'), None)
    if not vh_folder or not vv_folder:
        print(f"Skipping {track}: missing VH or VV data.")
        return

    vh_imgs = sorted(vh_folder.glob('*.img'), key=lambda p: extract_band_date(p.stem))
    vv_imgs = sorted(vv_folder.glob('*.img'), key=lambda p: extract_band_date(p.stem))
    if not vh_imgs or not vv_imgs:
        print(f"Skipping {track}: no images to process.")
        return

    dr = extract_date_range(vh_folder.name)
    stack_file = out_dir / f"{track}_{dr}_VH_VV_stack.img"
    print(f"Stacking {track} ({dr}): {len(vh_imgs)+len(vv_imgs)} bands")

    # Open first dataset to get georeference and projection
    ds0 = gdal.Open(str(vh_imgs[0]), gdalconst.GA_ReadOnly)
    cols = ds0.RasterXSize
    rows = ds0.RasterYSize
    proj = ds0.GetProjection()
    geo = ds0.GetGeoTransform()
    ds0 = None

    driver = gdal.GetDriverByName('ENVI')
    stack_ds = driver.Create(str(stack_file), cols, rows, len(vh_imgs)+len(vv_imgs), gdalconst.GDT_Float32)
    stack_ds.SetProjection(proj)
    stack_ds.SetGeoTransform(geo)

    band = 1
    for img_list in (vh_imgs, vv_imgs):
        for img in img_list:
            ds = gdal.Open(str(img), gdalconst.GA_ReadOnly)
            data = ds.GetRasterBand(1).ReadAsArray()
            stack_ds.GetRasterBand(band).WriteArray(data)
            desc = strip_pattern.sub("_", img.stem)
            stack_ds.GetRasterBand(band).SetDescription(desc)
            print(f"    Writing band {band}/{len(vh_imgs)+len(vv_imgs)}", end="\r")
            band += 1
            ds = None
    stack_ds.FlushCache()
    stack_ds = None
    print("    Stacking complete.")

    # Clip to each region
    for region in track_regions_map.get(track, []):
        out_file = out_dir / f"{region}_{track}_{dr}_VH_VV.img"
        print(f"Clipping {track} to {region} ({dr})...")
        ds = gdal.Warp(
            str(out_file), str(stack_file),
            format='ENVI', cutlineDSName=str(shapefiles_dir/region/f"NUTS2_{region}.shp"),
            cropToCutline=True, dstNodata=0,
            callback=make_progress("Clipping")
        )
        ds = None
        print("    Clipping complete.")

    # Cleanup intermediate stack files
    stack_file.unlink(missing_ok=True)
    stack_file.with_suffix('.hdr').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', nargs='+', required=True)
    args = parser.parse_args()

    sel = set(args.track)
    for t in list(sel):
        if t in group_map:
            sel.add(group_map[t])

    for track in ['P1', 'P1a', 'P2', 'P3', 'P4', 'P4a']:
        if track in sel:
            print(f"=== Processing {track} ===")
            stack_and_clip(track)

if __name__ == '__main__':
    main()