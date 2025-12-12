import os
import argparse
from pathlib import Path
import re
import sys
from datetime import datetime
from osgeo import gdal, gdalconst

# python 3_AIML_S1_stack_clip.py -t P1

# ================= CONFIGURATION =================
# Update these paths to match your system
BASE_DIR = Path("D:/AIML/WP7-Crop-type-mapping/AIML_CropMapper/workingDir")
SHAPEFILES_DIR = Path("D:/AIML/WP7-Crop-type-mapping/AIML_CropMapper/auxiliary_files/shapefiles_nuts")

# Mapping for automatic sub-track selection
GROUP_MAP = {"P1": "P1a", "P4": "P4a"}

# Regions to clip for each track
TRACK_REGIONS_MAP = {
    'P1': ['AT'], 'P1a': ['AU'],  # Changed AU to AT (Austria ISO code is AT, verify your shapefiles)
    'P2': ['IR'], 'P3': ['NL'],  # Changed IR to IE (Ireland ISO code is IE)
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
    """
    Extracts date (DDMonYYYY) from filename.
    Fixed regex to allow date at end of string (no trailing underscore required).
    Example: Sigma0_VH_mst_22Oct2024 -> 22Oct2024
    """
    # Regex explanation:
    # _ : starts with underscore
    # (\d{2}[A-Za-z]{3}\d{4}) : Capture Group 1 (e.g. 22Oct2024)
    # (?:_|$) : Non-capturing group matching EITHER an underscore OR end of string
    m = re.search(r"_(\d{2}[A-Za-z]{3}\d{4})(?:_|$)", stem)
    if not m:
        return datetime.min
    try:
        return datetime.strptime(m.group(1), "%d%b%Y")
    except ValueError:
        return datetime.min


def stack_and_clip(track: str):
    final_dir = BASE_DIR / track / 'S1_final_preprocessing'
    out_dir = BASE_DIR / track / 'processed_raster'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find input data folders (.data)
    # Note: This picks the first one found. If you have multiple runs, ensure the folder is clean.
    vh_folder = next(final_dir.glob('*_VH.data'), None)
    vv_folder = next(final_dir.glob('*_VV.data'), None)

    if not vh_folder or not vv_folder:
        print(f"Skipping {track}: missing VH or VV .data folder in {final_dir}")
        return

    # Gather .img files
    vh_imgs = list(vh_folder.glob('*.img'))
    vv_imgs = list(vv_folder.glob('*.img'))

    # Filter out internal/ancillary images (like projected incidence angle if present)
    # We only want Sigma0 bands which contain the date
    vh_imgs = [p for p in vh_imgs if extract_band_date(p.stem) != datetime.min]
    vv_imgs = [p for p in vv_imgs if extract_band_date(p.stem) != datetime.min]

    # Sort by date
    vh_imgs.sort(key=lambda p: extract_band_date(p.stem))
    vv_imgs.sort(key=lambda p: extract_band_date(p.stem))

    if not vh_imgs or not vv_imgs:
        print(f"Skipping {track}: no valid .img files found.")
        return

    # SAFETY CHECK: Ensure dates match
    if len(vh_imgs) != len(vv_imgs):
        print(f"Error {track}: Mismatch in band counts (VH: {len(vh_imgs)}, VV: {len(vv_imgs)})")
        return

    for i, (vh, vv) in enumerate(zip(vh_imgs, vv_imgs)):
        d_vh = extract_band_date(vh.stem)
        d_vv = extract_band_date(vv.stem)
        if d_vh != d_vv:
            print(f"Error {track}: Date mismatch at index {i}. VH: {d_vh}, VV: {d_vv}")
            return

    dr = extract_date_range(vh_folder.parent.name if vh_folder.parent.name.count('_') > 1 else vh_folder.name)

    # If extracting date range from folder name fails (e.g. it was renamed), make one up
    if not re.search(r"\d{8}_\d{8}", dr):
        d_start = extract_band_date(vh_imgs[0].stem).strftime("%Y%m%d")
        d_end = extract_band_date(vh_imgs[-1].stem).strftime("%Y%m%d")
        dr = f"{d_end}_{d_start}"  # Often usually Last_First in your previous naming

    stack_file = out_dir / f"{track}_{dr}_VH_VV_stack.img"
    print(f"Stacking {track} ({dr}): {len(vh_imgs) + len(vv_imgs)} bands")

    # Open first dataset to get georeference
    ds0 = gdal.Open(str(vh_imgs[0]), gdalconst.GA_ReadOnly)
    if not ds0:
        print(f"Could not open {vh_imgs[0]}")
        return

    cols = ds0.RasterXSize
    rows = ds0.RasterYSize
    proj = ds0.GetProjection()
    geo = ds0.GetGeoTransform()
    ds0 = None

    # Create Stack
    driver = gdal.GetDriverByName('ENVI')
    stack_ds = driver.Create(str(stack_file), cols, rows, len(vh_imgs) + len(vv_imgs), gdalconst.GDT_Float32)
    stack_ds.SetProjection(proj)
    stack_ds.SetGeoTransform(geo)

    # Write Data (Interleaved: Band1_VH, Band1_VV, Band2_VH, Band2_VV... or Sequential?)
    # Your script did Sequential (All VH then All VV).
    # Sticking to your logic: All VH then All VV.

    band_idx = 1

    # Write VH
    print("    Writing VH bands...")
    for img in vh_imgs:
        ds = gdal.Open(str(img), gdalconst.GA_ReadOnly)
        data = ds.GetRasterBand(1).ReadAsArray()
        stack_ds.GetRasterBand(band_idx).WriteArray(data)

        # Clean description
        desc = STRIP_PATTERN.sub("_", img.stem)
        stack_ds.GetRasterBand(band_idx).SetDescription(desc)

        band_idx += 1
        ds = None

    # Write VV
    print("    Writing VV bands...")
    for img in vv_imgs:
        ds = gdal.Open(str(img), gdalconst.GA_ReadOnly)
        data = ds.GetRasterBand(1).ReadAsArray()
        stack_ds.GetRasterBand(band_idx).WriteArray(data)

        desc = STRIP_PATTERN.sub("_", img.stem)
        stack_ds.GetRasterBand(band_idx).SetDescription(desc)

        band_idx += 1
        ds = None

    stack_ds.FlushCache()
    stack_ds = None
    print("\n    Stacking complete.")

    # Clip to regions
    regions = TRACK_REGIONS_MAP.get(track, [])
    if not regions:
        print(f"No regions defined for {track}, skipping clip.")

    for region in regions:
        shp_path = SHAPEFILES_DIR / region / f"NUTS2_{region}.shp"
        out_file = out_dir / f"{region}_{track}_{dr}_VH_VV.img"

        if not shp_path.exists():
            print(f"    WARNING: Shapefile not found: {shp_path}")
            continue

        print(f"    Clipping {track} to {region}...")

        # GDAL Warp for Clipping
        warp_opts = gdal.WarpOptions(
            format='ENVI',
            cutlineDSName=str(shp_path),
            cropToCutline=True,
            dstNodata=0,
            callback=make_progress(f"Clipping {region}")
        )

        ds = gdal.Warp(str(out_file), str(stack_file), options=warp_opts)
        ds = None  # Close file
        print("\n    Clipping complete.")

    # Cleanup intermediate stack
    # stack_file.unlink(missing_ok=True)
    # stack_file.with_suffix('.hdr').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--track', nargs='+', required=True)
    args = parser.parse_args()

    # Resolve Groups (P1 -> P1 and P1a)
    sel = set(args.track)
    for t in list(sel):
        if t in GROUP_MAP:
            sel.add(GROUP_MAP[t])

    # Process in order
    for track in ['P1', 'P1a', 'P2', 'P3', 'P4', 'P4a']:
        if track in sel:
            print(f"\n=== Processing {track} ===")
            stack_and_clip(track)


if __name__ == '__main__':
    main()