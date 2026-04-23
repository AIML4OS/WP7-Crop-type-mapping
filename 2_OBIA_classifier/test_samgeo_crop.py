import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os

def test_samgeo():
    # Sprawdzenie czy pakiet jest zainstalowany
    try:
        from samgeo import SamGeo
    except ImportError:
        print("BŁĄD: Biblioteka 'segment-geospatial' nie jest zainstalowana!")
        print("Uruchom poniższą komendę w terminalu, aby ją zainstalować:")
        print("pip install segment-geospatial")
        return

    input_tif = r"D:\AIML_CropMapper_Cloud\workingDir\P2\classification_results\segmentation\IE_P2_summed_composite.tif"
    out_dir = r"D:\AIML_CropMapper_Cloud\workingDir\P2\classification_results\segmentation"
    os.makedirs(out_dir, exist_ok=True)
    
    crop_tif = os.path.join(out_dir, "test_samgeo_input_crop_2.tif")
    out_tif = os.path.join(out_dir, "test_samgeo_output_2.tif")
    out_vector = os.path.join(out_dir, "test_samgeo_output_2.gpkg")
    
    # Przycina obraz dla SAM-Geo - NOWY KAFEL do testów
    x_off = 30000
    y_off = 40000
    win_size = 1024
    
    from osgeo import gdal
    import numpy as np
    print(f"Otwieranie głównego pliku i przycinanie: {input_tif}")
    ds = gdal.Open(input_tif)
    if not ds:
        print("Błąd odczytu rastra wejściowego.")
        return
        
    arr = ds.GetRasterBand(1).ReadAsArray(x_off, y_off, win_size, win_size)
    driver = gdal.GetDriverByName('GTiff')
    gt = ds.GetGeoTransform()
    new_gt = list(gt)
    new_gt[0] = gt[0] + x_off * gt[1]
    new_gt[3] = gt[3] + y_off * gt[5]
    
    # Normalize the Float32 array to 0-255 uint8 (ignore outliers via percentiles)
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    arr_norm = np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)
    arr_uint8 = (arr_norm * 255).astype(np.uint8)
    
    import cv2
    # Apply CLAHE to enhance contrast in darker regions (SAR data is very skewed)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    arr_clahe = clahe.apply(arr_uint8)
    
    # SAM expects an RGB image, so we duplicate the single SAR band to 3 bands
    crop_ds = driver.Create(crop_tif, win_size, win_size, 3, gdal.GDT_Byte)
    crop_ds.SetGeoTransform(new_gt)
    crop_ds.SetProjection(ds.GetProjection())
    for i in range(1, 4):
        crop_ds.GetRasterBand(i).WriteArray(arr_clahe)
    crop_ds.FlushCache()
    crop_ds = None
    
    print(f"Otwieranie pliku: {input_tif}")
    
    # Inicjalizacja modelu SAM-Geo (używamy tego samego checkpointu co wcześniej)
    sam_checkpoint = r"D:\AIML_CropMapper_Cloud\auxiliary_files\SAM_models\sam_vit_l_0b3195.pth"
    print("Ładowanie modelu SAM (segment-geospatial)...")
    sam = SamGeo(
        model_type="vit_l",
        checkpoint=sam_checkpoint,
        sam_kwargs={
            "points_per_side": 96,             # Zwiększona gęstość punktów (więcej małych obiektów)
            "pred_iou_thresh": 0.55,           # Jeszcze niższy próg akceptacji, żeby wyciągnąć drobnicę
            "stability_score_thresh": 0.55,    # Niższy próg stabilności
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 10         # 10 pikseli - ekstremalna czułość na drobnicę (ziarnicę)
        }
    )
    
    # Generowanie maski (zapisuje od razu do pliku TIF z zachowaniem georeferencji)
    print(f"Rozpoczęcie automatycznej segmentacji wycinka: {crop_tif}")
    sam.generate(
        source=crop_tif, 
        output=out_tif, 
        foreground=False,  # WAŻNE: False, bo inaczej Otsu usuwa małe / ciemniejsze pola
        unique=True,       # Generuj unikalne ID dla każdego segmentu (nie tylko maskę binarną)
        min_size=10,       # Minimalny obszar segmentu
        max_size=100000    # Maksymalny obszar (chroni przed maską na cały obraz)
    )
    
    print(f"Pomyślnie wygenerowano raster segmentacji: {out_tif}")

    # Wypełnianie zer (pustych przestrzeni na lądzie/wodzie) jak w zwykłym SAM
    print("Wypełnianie pustych przestrzeni (NoData) najbliższymi segmentami...")
    from scipy.ndimage import distance_transform_edt
    out_ds = gdal.Open(out_tif, gdal.GA_Update)
    band = out_ds.GetRasterBand(1)
    out_arr = band.ReadAsArray()
    
    zero_mask = (out_arr == 0)
    if np.any(zero_mask):
        _, indices = distance_transform_edt(zero_mask, return_indices=True)
        filled_arr = out_arr[tuple(indices)]
        band.WriteArray(filled_arr)
        band.FlushCache()
    out_ds = None
    print("Zakończono wypełnianie pustych przestrzeni.")
    
    # Opcjonalnie: automatyczna konwersja do poligonów (Wektor)
    try:
        print("Konwersja do wektora (GeoPackage)...")
        sam.tiff_to_vector(out_tif, out_vector)
        print(f"Pomyślnie wygenerowano wektory: {out_vector}")
    except Exception as e:
        print(f"Nie udało się wyeksportować wektora (być może brakuje biblioteki geopandas): {e}")

    print("\nGotowe! Otwórz wygenerowane pliki w QGIS.")

if __name__ == '__main__':
    test_samgeo()
