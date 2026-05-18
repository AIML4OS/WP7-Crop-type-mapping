"""
build_agri_mask.py
==================
Rozpakowuje, mozaikuje, przycina i tworzy binarne maski terenow rolnych
z danych Copernicus HRL Crop Type 2023.

Generuje 2 warianty masek (obie BINARNE: 0=brak upraw, 1=uprawy, 255=NoData):

  Wariant A: maska pola uprawne 3-klasowe (jare / oziminy / rzepak ozimy)
             -> 1 tam gdzie jest jakakolwiek z tych 3 klas
  Wariant B: maska wszystkich upraw (wlacznie z trwalymi)
             -> 1 tam gdzie jest jakakolwiek uprawa

UWAGA: Obie maski sa BINARNE (0/1). Wartosc 0 NIE jest tagowana jako NoData.
       NoData = 255 (obszar poza zakresem danych zrodlowych).

Uzycie:
  python build_agri_mask.py --country IE
  python build_agri_mask.py --country IE --target_crs EPSG:3857
  python build_agri_mask.py --country IE --clip_shp sciezka/do/granicy.shp
  python build_agri_mask.py --country IE --no_clip

Struktura katalogow (dane wejsciowe):
  auxiliary_files/
    raster_files/
      AgriMasks/
        <COUNTRY>/
          Results/    <- zip-y z Copernicus CLMS

    shapefiles_nuts/
      <COUNTRY>/
        NUTS2_<COUNTRY>.shp  <- granica kraju do przycinania (auto-wykrywana)

Wyniki trafiaja do:
  auxiliary_files/
    raster_files/
      AgriMasks/
        <COUNTRY>/
          <COUNTRY>_agri_mask_3class_<CRS>.tif   <- Wariant A (binarna)
          <COUNTRY>_agri_mask_allcrops_<CRS>.tif <- Wariant B (binarna)
"""

import os
import sys
import zipfile
import argparse
import shutil
import tempfile
from pathlib import Path


# -----------------------------------------------------------------------
# KONFIGURACJA SCIEZEK BAZOWYCH
# Skrypt jest w: D:/AIML_CropMapper_Cloud/2_OBIA_classifier/
# -----------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent                          # 2_OBIA_classifier/
PROJECT_ROOT = SCRIPT_DIR.parent                              # D:/AIML_CropMapper_Cloud/
AUX_DIR      = PROJECT_ROOT / 'auxiliary_files'
AGRIMASKS_DIR = AUX_DIR / 'raster_files' / 'AgriMasks'
NUTS_DIR      = AUX_DIR / 'shapefiles_nuts'


# -----------------------------------------------------------------------
# DEFINICJA KLAS HRL CTY 2023
# Zrodlo: CLMS_HRLVLCC_CTY_R10.qml
# -----------------------------------------------------------------------

# Wariant A: klasy uwzglednianych upraw (1=jare, 2=oziminy, 3=rzepak ozimy)
# -> maska binarna: te piksele dadza wartosc 1 w wyniku
CLASS_3_INCLUDE = {
    # Oziminy
    1110,   # Wheat (pszenica - glownie ozima)
    1120,   # Barley (jeczmien - czesc ozima)
    1150,   # Other Cereals (inne zboza ozime)
    # Rzepak ozimy
    1430,   # Rapeseed
    # Uprawy jare i pozostale
    1130,   # Maize (kukurydza)
    1210,   # Fresh Vegetables
    1220,   # Dry Pulses (straczkowe)
    1310,   # Potatoes
    1320,   # Sugar Beet
    1410,   # Sunflower
    1420,   # Soybeans
    1440,   # Flax, cotton and hemp
    3100,   # Unclassified arable crop
}

# Wariant B: wszystkie uprawy (wlacznie z trwalymi)
ALL_CROPS_INCLUDE = {
    1110, 1120, 1130, 1140, 1150,   # Zboza
    1210, 1220,                     # Warzywa i straczkowe
    1310, 1320,                     # Okopowe
    1410, 1420, 1430, 1440,         # Oleiste i przemyslowe
    2100, 2200, 2310, 2320,         # Uprawy trwale (winorosla, oliwki, sady, orzechy)
    3100, 3200,                     # Nieklasyfikowane uprawy
}

NODATA_VAL = None   # Brak NoData - identycznie jak EU_arable_areas_mask_3857.tif
                    # Wartosc 0 = brak upraw / poza zakresem, 1 = uprawy


# -----------------------------------------------------------------------
# FUNKCJE
# -----------------------------------------------------------------------

def unzip_tiles(results_dir: Path, temp_dir: Path) -> list:
    """Rozpakowuje wszystkie zip-y z katalogu Results do temp_dir.
    Zwraca liste sciezek do wypakowanych plikow TIF."""
    zip_files = sorted(results_dir.glob('*.zip'))
    if not zip_files:
        print(f"  [UWAGA] Brak plikow ZIP w: {results_dir}")
        return []

    tif_files = []
    print(f"  Rozpakowywanie {len(zip_files)} plikow ZIP...")
    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path, 'r') as zf:
            tif_names = [
                n for n in zf.namelist()
                if n.lower().endswith('.tif') and not n.endswith('.aux.xml')
            ]
            for tif_name in tif_names:
                out_path = temp_dir / Path(tif_name).name
                if not out_path.exists():
                    zf.extract(tif_name, temp_dir)
                    extracted = temp_dir / tif_name
                    if extracted != out_path and extracted.exists():
                        shutil.move(str(extracted), str(out_path))
                tif_files.append(str(out_path))
        print(f"    OK: {zf_path.name}")

    # Sprawdz tez juz wypakowane TIF-y w Results (np. jeden juz byl rozpakowywany recznie)
    already_tifs = [
        str(p) for p in results_dir.glob('**/*.tif')
        if not p.name.endswith('.aux.xml')
    ]
    if already_tifs and not zip_files:
        print(f"  Znaleziono {len(already_tifs)} plikow TIF bezposrednio w katalogu Results.")
        return already_tifs

    return tif_files


def reclassify_to_binary(src_path: str, dst_path: str, include_set: set):
    """
    Reklasyfikuje raster do maski binarnej:
      1 -> piksel nalezacy do klasy w include_set (uprawy)
      0 -> brak upraw LUB poza zakresem danych (65535 w HRL CTY -> 0)

    BRAK wartosci NoData - identycznie jak EU_arable_areas_mask_3857.tif.
    Wartosc 0 jest poprawna wartoscia (brak upraw), nie NoData.
    """
    from osgeo import gdal
    import numpy as np

    ds = gdal.Open(src_path)
    if ds is None:
        print(f"  BLAD: Nie mozna otworzyc: {src_path}")
        return

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.int32)

    # Binarna maska: 1 = uprawa, 0 = wszystko inne (w tym 65535 = poza kafelkiem)
    out = np.zeros(data.shape, dtype=np.uint8)
    for val in include_set:
        out[data == val] = 1
    # 65535 (poza zakresem HRL CTY) -> pozostaje 0 (juz ustawione przez zeros)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        dst_path,
        ds.RasterXSize, ds.RasterYSize, 1,
        gdal.GDT_Byte,
        options=['COMPRESS=DEFLATE', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
    )
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(out)
    # NIE ustawiamy NoData - identycznie jak EU_arable_areas_mask_3857.tif
    out_ds.FlushCache()
    out_ds = None
    ds = None
    print(f"    Reklasyfikacja: {Path(src_path).name} -> {Path(dst_path).name}")


def _detect_shp_crs(shp_path: str) -> str:
    """
    Wykrywa prawdziwy CRS shapefile na podstawie zakresu wspolrzednych.
    Uzywane jako obejscie gdy .prj ma blednie zadeklarowany CRS.
    """
    from osgeo import ogr
    ds = ogr.Open(shp_path)
    if not ds:
        return None
    layer = ds.GetLayer()
    srs = layer.GetSpatialRef()
    ext = layer.GetExtent()  # (minX, maxX, minY, maxY)
    declared = srs.GetAuthorityCode(None) if srs else None

    # Sprawdz zasiag - EPSG:3857 ma wspolrzedne rzedu milionow (ok. +-20M)
    # EPSG:3035 ma wspolrzedne rzedu setek tysiecy / kilku milionow
    # EPSG:4326 ma wspolrzedne +-180 / +-90
    max_coord = max(abs(ext[0]), abs(ext[1]), abs(ext[2]), abs(ext[3]))
    if max_coord < 180:
        real_crs = 'EPSG:4326'
    elif max_coord < 10_000_000 and abs(ext[2]) > 1_000_000:
        real_crs = 'EPSG:3035'
    else:
        real_crs = 'EPSG:3857'

    if declared != real_crs.split(':')[1]:
        print(f"  [UWAGA] CRS w .prj: EPSG:{declared}, wykryty z zasiegu: {real_crs}")
        print(f"          Zasieg: {ext}, uzyje {real_crs} jako cutlineSRS")
    return real_crs


def mosaic_and_reproject(tif_files: list, output_path: str, target_crs: str,
                         clip_shp: str = None):
    """
    Mozaikuje kafelki i reprojekcjonuje do target_crs.
    Wynik: czysty raster binarny 0/1 BEZ zadnej wartosci NoData.
      0 = brak upraw,  1 = uprawy
    Uzywa gdal.Warp bezposrednio z lista plikow (bez VRT) aby uniknac
    problemu z zerowanymi wartosciami przy BuildVRT na Windows.
    """
    from osgeo import gdal

    print(f"  Mozaikowanie + reprojekcja {len(tif_files)} kafelkow do {target_crs}...")

    # gdal.Warp przyjmuje liste plikow bezposrednio - bez VRT
    ds = gdal.Warp(
        output_path,
        tif_files,          # lista reklasyfikowanych kafelkow w EPSG:3035
        format='GTiff',
        dstSRS=target_crs,
        resampleAlg=gdal.GRA_NearestNeighbour,
        creationOptions=['COMPRESS=DEFLATE', 'TILED=YES',
                         'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'BIGTIFF=YES'],
        multithread=True,
        warpOptions=['NUM_THREADS=ALL_CPUS'],
    )
    if ds is None:
        print(f"  BLAD: gdal.Warp nie powiodl sie!")
        return
    ds.FlushCache()
    ds = None

    print(f"  Zapisano: {output_path}")








def resolve_clip_shp(country_code: str) -> str | None:
    """Szuka pliku SHP z granica kraju w standardowej lokalizacji NUTS."""
    nuts_country_dir = NUTS_DIR / country_code
    if not nuts_country_dir.exists():
        return None
    # Szukaj: NUTS2_<COUNTRY>.shp lub jakiegokolwiek SHP
    candidates = [
        nuts_country_dir / f"NUTS2_{country_code}.shp",
        nuts_country_dir / f"NUTS1_{country_code}.shp",
        *list(nuts_country_dir.glob("*.shp")),
    ]
    for p in candidates:
        if p.exists():
            print(f"  Granica kraju: {p}")
            return str(p)
    return None


def build_mask_for_country(country_code: str, results_dir: Path, output_dir: Path,
                           target_crs: str, clip_shp: str = None, force: bool = False):
    """Glowna funkcja: buduje binarne maski dla jednego kraju."""
    print(f"\n{'='*60}")
    print(f" Przetwarzanie: {country_code}  ({target_crs})")
    print(f"{'='*60}")

    crs_tag = target_crs.replace(':', '').lower().replace('epsg', 'epsg')
    out_3class   = output_dir / f"{country_code}_agri_mask_3class_{crs_tag}.tif"
    out_allcrops = output_dir / f"{country_code}_agri_mask_allcrops_{crs_tag}.tif"

    if out_3class.exists() and out_allcrops.exists() and not force:
        print(f"  Maski juz istnieja. Uzyj --force aby wygenerowac ponownie.")
        print(f"  {out_3class}")
        print(f"  {out_allcrops}")
        return

    # Uzywamy stalego katalogu intermediate zamiast tempfile.TemporaryDirectory
    # Na Windows GDAL trzyma uchwyty do plikow otwarte i TemporaryDirectory
    # nie moze ich usunac - co powoduje ze pliki wyjsciowe sa puste/zerowe.
    temp_path = output_dir / "intermediate"
    temp_path.mkdir(parents=True, exist_ok=True)

    try:
        # [1] Rozpakuj
        print("\n[1/4] Rozpakowywanie ZIP-ow...")
        raw_tifs = unzip_tiles(results_dir, temp_path)
        if not raw_tifs:
            print("  BLAD: Brak plikow TIF po rozpakowaniu!")
            return

        # [2] Reklasyfikacja kazdego kafelka -> binarne 0/1
        print(f"\n[2/4] Reklasyfikacja {len(raw_tifs)} kafelkow (binarna 0/1)...")
        tifs_3class   = []
        tifs_allcrops = []

        for src_tif in raw_tifs:
            stem = Path(src_tif).stem
            dst_3class   = str(temp_path / f"{stem}_3class.tif")
            dst_allcrops = str(temp_path / f"{stem}_allcrops.tif")
            reclassify_to_binary(src_tif, dst_3class,   include_set=CLASS_3_INCLUDE)
            reclassify_to_binary(src_tif, dst_allcrops, include_set=ALL_CROPS_INCLUDE)
            tifs_3class.append(dst_3class)
            tifs_allcrops.append(dst_allcrops)

        # [3] Mozaik + reprojekcja + przycinanie
        print("\n[3/4] Mozaikowanie, reprojekcja, przycinanie - Wariant A (3-klasowy -> binarny)...")
        mosaic_and_reproject(tifs_3class,   str(out_3class),   target_crs, clip_shp)

        print("\n[3/4] Mozaikowanie, reprojekcja, przycinanie - Wariant B (wszystkie uprawy -> binarny)...")
        mosaic_and_reproject(tifs_allcrops, str(out_allcrops), target_crs, clip_shp)

    finally:
        # Usun katalog intermediate po zakonczeniu wszystkich operacji GDAL
        import gc
        gc.collect()   # wymusz zwolnienie uchwytow GDAL
        try:
            shutil.rmtree(str(temp_path))
            print(f"\n  Usunieto katalog intermediate: {temp_path}")
        except Exception as e:
            print(f"\n  [INFO] Nie mozna usunac intermediate (mozna usunac recznie): {e}")


    # [4] Podsumowanie
    print("\n[4/4] Gotowe!")
    for f, label in [(out_3class, "Wariant A (3-klasowy binarny)"),
                     (out_allcrops, "Wariant B (wszystkie uprawy binarny)")]:
        size_mb = f.stat().st_size / 1024**2 if f.exists() else 0
        print(f"  {label}: {f}  [{size_mb:.1f} MB]")

    print()
    print("  Legenda (obie maski binarne, brak wartosci NoData):")
    print("    0 = brak upraw / obszar poza danymi HRL CTY")
    print("    1 = uprawy")
    print("    (identyczna struktura jak EU_arable_areas_mask_3857.tif)")
    print()
    print("  Klasy uwzglednionie w Wariancie A (3-klasowy):")
    print("    jare:    kukurydza, warzywa, straczkowe, okopowe, oleiste, nieklasyfikowane")
    print("    oziminy: pszenica, jeczmien, inne zboza ozime")
    print("    rzepak:  Rapeseed (1430)")


# -----------------------------------------------------------------------
# PUNKT WEJSCIA
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Buduje binarne maski rolnicze z Copernicus HRL CTY 2023',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przyklady:
  python build_agri_mask.py --country IE
  python build_agri_mask.py --country IE --target_crs EPSG:3857
  python build_agri_mask.py --country IE --no_clip
  python build_agri_mask.py --country IE --clip_shp D:/moja/granica.shp
  python build_agri_mask.py --country PL --force
        """
    )
    parser.add_argument('--country', '-c', required=True,
                        help='Kod kraju (IE, AT, PL, DE, ...)')
    parser.add_argument('--target_crs', default='EPSG:3857',
                        help='Docelowy uklad wspolrzednych (domyslnie EPSG:3857)')
    parser.add_argument('--clip_shp', default=None,
                        help='Sciezka do pliku SHP do przycinania (nadpisuje auto-wykrywanie)')
    parser.add_argument('--no_clip', action='store_true',
                        help='Nie przycinaj do granic kraju')
    parser.add_argument('--results_dir', default=None,
                        help='Katalog z plikami ZIP (domyslnie AgriMasks/<COUNTRY>/Results/)')
    parser.add_argument('--output_dir', default=None,
                        help='Katalog wyjsciowy (domyslnie AgriMasks/<COUNTRY>/)')
    parser.add_argument('--force', action='store_true',
                        help='Nadpisz istniejace pliki wyjsciowe')
    args = parser.parse_args()

    country = args.country.upper()

    # Katalog z danymi ZIP
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = AGRIMASKS_DIR / country / 'Results'

    # Katalog wyjsciowy -> AgriMasks/<COUNTRY>/
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = AGRIMASKS_DIR / country

    if not results_dir.exists():
        print(f"BLAD: Katalog z danymi nie istnieje: {results_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Granica do przycinania
    clip_shp = None
    if args.no_clip:
        print("  Przycinanie wylaczone (--no_clip)")
    elif args.clip_shp:
        clip_shp = args.clip_shp
    else:
        clip_shp = resolve_clip_shp(country)
        if clip_shp is None:
            print(f"  [INFO] Brak pliku SHP granicy dla '{country}' w {NUTS_DIR / country}")
            print(f"  Maska NIE bedzie przycinana do granic kraju.")
            print(f"  Uzyj --clip_shp lub dodaj plik do: {NUTS_DIR / country / f'NUTS2_{country}.shp'}")

    build_mask_for_country(
        country_code=country,
        results_dir=results_dir,
        output_dir=output_dir,
        target_crs=args.target_crs,
        clip_shp=clip_shp,
        force=args.force,
    )


if __name__ == '__main__':
    main()
