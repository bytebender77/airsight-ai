"""
Script 2: Download Weather (ERA5) — 2° global grid, 2015-2022
Run: python3 dl_2_weather.py
"""
import ee, pandas as pd, os, time

ee.Initialize(project='pm25-earth-engine')

GRID_STEP = 2.0
LAT_RANGE = (-60, 72)
LON_RANGE = (-180, 180)
START_DATE = '2015-01-01'
END_DATE = '2022-01-01'
CHUNK_SIZE = 1500
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, 'data', 'weather_global.csv')

if os.path.exists(OUTPUT):
    print("⏭️  Weather already exists, skipping."); exit()

features = []
lat = LAT_RANGE[0] + GRID_STEP / 2
while lat < LAT_RANGE[1]:
    lon = LON_RANGE[0] + GRID_STEP / 2
    while lon < LON_RANGE[1]:
        features.append(ee.Feature(ee.Geometry.Point([round(lon, 2), round(lat, 2)]),
                                   {'lat': round(lat, 2), 'lon': round(lon, 2)}))
        lon += GRID_STEP
    lat += GRID_STEP

n_pts = len(features)
chunks = [ee.FeatureCollection(features[i:i+CHUNK_SIZE]) for i in range(0, n_pts, CHUNK_SIZE)]
print(f"🌡️  Weather: {n_pts} points, {len(chunks)} chunks")

BANDS = ['temperature_2m', 'dewpoint_temperature_2m',
         'u_component_of_wind_10m', 'v_component_of_wind_10m', 'surface_pressure']

era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
    .filterDate(START_DATE, END_DATE).select(BANDS)
images = era5.toList(era5.size())
n = era5.size().getInfo()
print(f"   Found {n} images")

rows = []
t = time.time()
for i in range(n):
    img = ee.Image(images.get(i))
    d = img.date().format('YYYY-MM-dd').getInfo()
    if i % 10 == 0: print(f"   {i+1}/{n} ({d})... [{(time.time()-t)/60:.1f} min]")
    for c in chunks:
        try:
            r = img.reduceRegions(collection=c, reducer=ee.Reducer.mean(), scale=10000).getInfo()
            for f in r['features']:
                p = f['properties']
                row = {'date': d, 'lat': p['lat'], 'lon': p['lon']}
                for b in BANDS:
                    row[b] = p.get(b)
                rows.append(row)
        except Exception as e:
            print(f"   ⚠️ {e}")
            time.sleep(3)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"\n✅ Weather done: {len(df):,} rows | {(time.time()-t)/60:.1f} min")
