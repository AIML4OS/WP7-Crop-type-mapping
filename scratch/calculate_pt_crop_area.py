import geopandas as gpd
import pandas as pd

# Load shapefile (only columns we need to save memory)
print("Reading shapefile columns...")
gdf = gpd.read_file(
    r"d:\AIML_CropMapper_Cloud\auxiliary_files\raster_files\AgriMasks\PT\PT_2021_EC21.shp",
    columns=['EC_trans_n', 'OSA_AREA']
)

# Total area in database
total_area = gdf['OSA_AREA'].sum()
print(f"Total area in Shapefile: {total_area:.2f} hectares")

# Set up class mappings based on EC_trans_n
grassland_crops = {
    'PERMANENT PASTURES', 'PASTURE WITH BUSHES', 'TEMPORARY MEADOWS', 'ryegrass',
    'lolium_ryegrass', 'temporary_grass', 'pasture_meadow_grassland_grass', 'clover',
    'lucerne', 'alfalfa_lucerne', 'COMMON BIRDSFOOT', 'legumes_harvested_green',
    'COMMON LAND PASTURE'
}
olives = {'OLIVE VALLEY'}
vineyards = {'VINEYARD'}
maize = {'CORN', 'MAIZE; POTATOES', 'MAIZE; OTHER VEGETABLES', 'CORN; OTHER VEGETABLES; POTATO',
         'MAIZE; BEANS', 'CORN; OATS', 'CORN; ANNUAL AND OTHER CULT. FORAGE ANNUALS',
         'PUMPKINS AND COURGETTES; CORN', 'MAIZE; POTATO; ONION', 'CORN; TEMPORARY MEADOWS',
         'MAIZE; POTATOES; OTHER VEGETABLES; OATS', 'MAIZE; BEANS; POTATOES',
         'MAIZE; BEANS; POTATOES; ANNUAL INTERCROPS AND OTHER CROPS. FORRAG. ANNUALS',
         'MAIZE; POTATO; ONION', 'MAIZE; VEGETABLES',
         'MAIZE; BEANS; OTHER VEGETABLES; POTATOES; INTERCROPPING AND OTHER ANNUAL CROPS. FORRAG. ANNUALS'}
rice = {'RICE', 'ELEGIBLE LANDSCAPE FEATRURES - RICE', 'NON ELEGIBLE LANDSCAPE FEATURES - RICE (COMP. MAA)'}
wheat = {'WHEAT', 'WHEAT; CORN', 'WHEAT; OUTRAS HORTÍCOLAS', 'WHEAT; LUPIN', 'WHEAT; MELON',
         'WHEAT; VEGETABLES', 'WHEAT; ANNUAL INTERCROPS AND OTHER FODDER CROPS FORRAG. ANNUALS',
         'WHEAT; CORN; OTHER VEGETABLES'}
barley = {'BARLEY', 'Barley; lolium'}
oats = {'OAT', 'OATS; BEANS', 'OATS; ANNUAL INTERCROPS AND OTHER FODDER CROPS FORRAG. ANNUALS',
        'POTATO; OATS', 'POTATOES; OATS', 'BATATA;AVEIA'}
rye = {'RYE', 'RYE; CORN', 'RYE; MAIZE; OTHER VEGETABLES; POTATOES', 'RYE; MAIZE; POTATO',
       'RYE; MAIZE; BEANS; POTATOES; OTHER VEGETABLES', 'CEREALS; SOY', 'RYE; OTHER VEGETABLES; POTATOES',
       'RYE; POTATO', 'RYE; ANNUAL INTERCROPS AND OTHER FODDER CROPS FORRAG. ANNUALS'}
triticale = {'TRITICALE'}
other_cereals = {'SORGHUM', 'OTHER CEREALS', 'OTHER CEREALS; CORN', 'OTHER CEREALS; OTHER VEGETABLES; POTATOES',
                 'OTHER CEREALS; BEANS', 'OTHER CEREALS; CABBAGE; OTHER VEGETABLES; ONION'}
potatoes = {'POTATO', 'SWEET POTATO', 'BATATA DOCE', 'BATATA', 'BATATA DOCE;MILHO',
            'SWEET POTATO; CORN', 'SWEET POTATO; CORN; COURGETTE', 'SWEET POTATO; POTATO',
            'SWEET POTATO; BEANS', 'BATATA;CEBOLA', 'BATATA;CONSOCIAÇÕES ANUAIS E OUTRAS CULT. FORRAG. ANUAIS',
            'POTATO; ONION', 'POTATO; ANNUAL AND OTHER CULT. FORAGE ANNUALS', 'POTATOES; OTHER VEGETABLES'}
beets = {'BEETROOT', 'BEETROOT_BEETS'}
legumes = {'BEAN', 'LUPINE', 'TREMOCILHA', 'Lupinus luteus', 'FAVA', 'ERVILHA', 'GRÃO DE BICO', 'PEA', 'CHICKPEA',
           'BEANS; BROAD BEAN', 'BEAN; PEA; OTHER VEGETABLES; BROAD BEAN', 'sweet_lupins', 'beans', 'peas', 'chickpeas',
           'LUPINE; BROAD BEAN', 'FEIJÃO;BATATA', 'BEANS; POTATOES', 'FEIJÃO;OUTRAS HORTÍCOLAS', 'BEANS; OTHER VEGETABLES',
           'FEIJÃO;BATATA;OUTRAS HORTÍCOLAS', 'BEAN; POTATO; OTHER VEGETABLES', 'FEIJÃO;NABO', 'BEANS; TURNIP',
           'FEIJÃO;CEBOLA', 'BEANS; ONION', 'COUVE;TREMOÇO;FAVA', 'CABBAGE; LUPIN; BEAN BEAN'}
vegetables = {'OTHER VEGETABLES', 'ABÓBORAS E ABOBORINHAS', 'PUMPKINS AND COURGETTES', 'COURGETTE', 'MELÃO', 'MELANCIA',
              'ALHO FRANCÊS', 'ALHO', 'ALFACE', 'NABO', 'NABIÇA', 'CENOURA', 'TOMATE', 'COUVE', 'GREEN CABBAGE',
              'PIMENTO', 'ESPINAFRE', 'pumpkin_squash_gourd', 'zucchini_courgette', 'fresh_vegetables', 'melon',
              'watermelon', 'garlic', 'onions', 'turnips', 'carrots_daucus', 'tomato', 'spinach', 'cress',
              'salads_lettuce_leaf_vegetables', 'brassica_oleracea_cabbage', 'piper_pepper',
              'PUMPKINS AND COURGETTES; POTATOES; OTHER VEGETABLES', 'PUMPKINS AND COURGETTES; POTATOES',
              'PUMPKINS AND COURGETTES; OTHER VEGETABLES', 'PUMPKINS AND COURGETTES; OATS', 'PUMPKINS AND PUMPKINS; CORN; OTHER VEGETABLES',
              'PUMPKINS AND COURGETTES; CORN; TURNIP; POTATO', 'PUMPKINS AND COURGETTES; CHICKPEA; CORN; BEAN; POTATO; OTHER VEGETABLES; ANNUAL AND OTHER CULT. FORAGE ANNUALS',
              'PUMPKINS AND COURGETTES; LUPIN', 'PUMPKINS AND COURGETTES; TEMPORARY MEADOWS', 'PUMPKINS AND COURGETTES; ONION',
              'CABBAGE; OATS', 'CABBAGE;POTATO;ONION', 'COUVE;BATATA;CEBOLA', 'TURNIP; TURNIP', 'NABIÇA;NABO',
              'OTHER VEGETABLES; POTATOES', 'OTHER VEGETABLES; BEAN', 'OTHER VEGETABLES; LUPINS',
              'OTHER VEGETABLES; POTATOES; ANNUAL CONSOCIATIONS AND OTHER CULT. FORAGE ANNUALS', 'OTHER VEGETABLES; ONION',
              'lolium; OTHER VEGETABLES', 'TEMPORARY MEADOWS; OTHER VEGETABLES', 'FALLOW; OTHER VEGETABLES',
              'MELON; OTHER VEGETABLES; ANNUAL AND OTHER CULT. FORAGE ANNUALS'}
fruits = {'PERA', 'MAÇÃ', 'PÊSSEGO', 'AMEIXA', 'MARMELO', 'CEREJA', 'GINJA', 'DAMASCO', 'KIWI', 'FIGO', 'FIGO DA INDIA',
          'MORANGO', 'FRAMBOESA', 'AMORA', 'MIRTILO', 'OUTRAS FRUTOS FRESCOS', 'OUTROS PEQUENOS FRUTOS', 'OUTROS FRUTOS SUB-TROPICAIS',
          'orchards_fruits', 'apples', 'pears', 'quinces', 'plums', 'apricots', 'cherry_cherries', 'blueberry',
          'blackberry', 'raspberry_raspberries', 'strawberries', 'fig', 'persimmon', 'kiwi', 'avocado',
          'sour cherry', 'plum', 'peach', 'cherry', 'apricot', 'fig_da_india', 'strawberry', 'raspberry',
          'blackberry', 'blueberry'}
citrus = {'LARANJA', 'LIMÃO', 'OUTROS CITRINOS', 'citrus_plantations'}
nuts = {'CASTANHA', 'AMENDOA', 'NOZ', 'PINHÃO', 'AMENDOIM', 'PISTACIOS', 'sweet_chestnuts', 'almond', 'nuts', 'pistachio',
        'CHESTNUT PLANTATION; OTHER HARDWOOD PLANTATION', 'POVOAMENTO CASTANHEIRO;POVOAMENTO OUTRAS FOLHOSAS',
        'POVOAMENTO CASTANHEIRO', 'CHESTNUT PLANTATIONS'}
fallow = {'POUSIO', 'fallow_land_not_crop', 'FALLOWING/ INTERCROPPING (INTERRUPTED CULTIVATEN TO MAKE SOIL MORE FERTILE)'}
forests = {'SOBREIRO', 'POVOAMENTO DE SOBREIROS', 'POVOAMENTO AZINHEIRAS', 'POVOAMENTO DE PINHEIRO MANSO',
           'POVOAMENTO DE EUCALIPTO', 'POVOAMENTO OUTRAS FOLHOSAS', 'POVOAMENTO OUTRAS RESINOSAS', 'POVOAMENTO F MISTO',
           'BOSQUETES', 'ACEIRO FLORESTAL', 'GALERIA RIPÍCOLA', 'OUTRAS SUPERFÍCIES FLORESTAIS', 'tree_wood_forest',
           'oak', 'eucalyptus', 'CORK OAK FOR CORK PRODUCTION', 'CORK OAK PLANTATION', 'PLANTATION OF OTHER HARDWOODS',
           'EVERGREEN OAK PLANTATION', 'PINE TREES PLANTATION', 'OTHER CONIFEROUS FORESTS', 'MIXED Forest',
           'EUCALYPTUS PLANTATION', 'BLACK OAK NEGRAL PLANTATION', 'FOREST FIREBREAKS', 'RIPARIAN GALLERY',
           'OTHER FOREST SURFACES', 'ELEGIBLE LANDSCAPE FEATURES - HEDGES AND WINDBREAKS', 'RIPARIAN GALLERY',
           'SETTLEMENT OF PINE TREES; OTHER FOREST SURFACES', 'OTHER FOREST SURFACES; CORK GROVE',
           'PLANTATION OF OTHER HARDWOODS; PLANTATION OF CORK OAK', 'SETTLEMENT OF OTHER HARDWOODS; SETTLEMENT OF OTHER RESIN',
           'PLANTATION OF OTHER HARDWOODS; PLANTATION OF OTHER RESIN; OTHER F MIXED PLANTATION', 'SETTLEMENT OTHER HARDWOOD; F MIXED SETTLEMENT',
           'POVOAMENTO OUTRAS FOLHOSAS;POVOAMENTO DE SOBREIROS', 'POVOAMENTO OUTRAS FOLHOSAS;POVOAMENTO OUTRAS RESINOSAS',
           'POVOAMENTO OUTRAS FOLHOSAS;POVOAMENTO OUTRAS RESINOSAS;POVOAMENTO F MISTO', 'POVOAMENTO OUTRAS FOLHOSAS;POVOAMENTO F MISTO',
           'POVOAMENTO DE PINHEIRO MANSO;OUTRAS SUPERFÍCIES FLORESTAIS', 'OUTRAS SUPERFÍCIES FLORESTAIS;POVOAMENTO DE SOBREIROS'}

# Combine all target crop sets
all_target_crops = (
    grassland_crops | olives | vineyards | maize | rice | wheat | barley | oats | rye | triticale |
    other_cereals | potatoes | beets | legumes | vegetables | fruits | citrus | nuts | fallow | forests
)

# Categorize gdf
def classify_crop(name):
    if not name or pd.isna(name) or name.strip() == '' or name == 'NOT KNOWN':
        return 'NOT KNOWN'
    if name in grassland_crops: return 'Grassland & Pastures'
    if name in olives: return 'Olive Groves'
    if name in vineyards: return 'Vineyards'
    if name in maize: return 'Maize'
    if name in rice: return 'Rice'
    if name in wheat: return 'Wheat'
    if name in barley: return 'Barley'
    if name in oats: return 'Oats'
    if name in rye: return 'Rye'
    if name in triticale: return 'Triticale'
    if name in other_cereals: return 'Other Cereals'
    if name in potatoes: return 'Potatoes'
    if name in beets: return 'Beets'
    if name in legumes: return 'Legumes & Pulses'
    if name in vegetables: return 'Vegetables'
    if name in fruits: return 'Orchards & Fruits'
    if name in citrus: return 'Citrus'
    if name in nuts: return 'Nuts'
    if name in fallow: return 'Fallow Land'
    if name in forests: return 'Forests & Woodlands'
    return 'OTHER UNCLASSIFIED'

gdf['Class'] = gdf['EC_trans_n'].apply(classify_crop)

# Calculate area statistics
area_stats = gdf.groupby('Class')['OSA_AREA'].agg(['sum', 'count'])
area_stats['sum_pct'] = (area_stats['sum'] / total_area) * 100

print("\nPortugal Crop Area Statistics:")
print(area_stats.sort_values(by='sum', ascending=False).to_string())

mapped_area = area_stats.loc[area_stats.index != 'NOT KNOWN', 'sum'].sum()
mapped_pct = (mapped_area / total_area) * 100
print(f"\nTotal Mapped Crop Area (our 20 classes): {mapped_area:.2f} ha ({mapped_pct:.2f}% of total shapefile area)")
print(f"Total Unmapped Area (NOT KNOWN / Others): {total_area - mapped_area:.2f} ha ({(100 - mapped_pct):.2f}%)")
