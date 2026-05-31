import os
import urllib.request
import zipfile
from pathlib import Path

# Paths
dest_dir = Path(r"d:\AIML_CropMapper_Cloud\auxiliary_files\raster_files\AgriMasks\PT")
dest_dir.mkdir(parents=True, exist_ok=True)

zip_path = dest_dir / "PT.zip"
csv_path = dest_dir / "pt_2021.csv"

# Download URLs
zip_url = "https://zenodo.org/records/10118572/files/PT.zip?download=1"
csv_url = "https://zenodo.org/records/10118572/files/pt_2021.csv?download=1"

print(f"Downloading {zip_url} to {zip_path}...")
urllib.request.urlretrieve(zip_url, zip_path)
print("Done downloading ZIP.")

print(f"Downloading {csv_url} to {csv_path}...")
urllib.request.urlretrieve(csv_url, csv_path)
print("Done downloading CSV.")

# Unzip PT.zip
print(f"Extracting {zip_path} to {dest_dir}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dest_dir)
print("Extraction complete.")
