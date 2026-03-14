"""
Script 1: Download PM2.5 — 2° global grid, 2015-2022
Run: python3 dl_1_pm25.py
"""
import ee, pandas as pd, os, time, math

ee.Initialize(project='pm25-earth-engine')

GRID_STEP = 2.0
LAT_RANGE = (-60, 72)
LON_RANGE = (-180, 180)
START_DATE = '2015-01-01'
END_DATE = '2022-01-01'
CHUNK_SIZE = 1500
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, 'data', 'pm25_global.csv')

if os.path.exists(OUTPUT):
    print("⏭️  PM2.5 already exists, skipping."); exit()

# Build grid
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
print(f"🌍 PM2.5: {n_pts} points, {len(chunks)} chunks")

pm25 = ee.ImageCollection("projects/sat-io/open-datasets/GLOBAL-SATELLITE-PM25/MONTHLY") \
    .filterDate(START_DATE, END_DATE)
images = pm25.toList(pm25.size())
n = pm25.size().getInfo()
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
                rows.append({'date': d, 'lat': p['lat'], 'lon': p['lon'], 'pm25': p.get('mean')})
        except Exception as e:
            print(f"   ⚠️ {e}")
            time.sleep(3)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"\n✅ PM2.5 done: {len(df):,} rows | {df['pm25'].notna().sum():,} non-null | {(time.time()-t)/60:.1f} min")
