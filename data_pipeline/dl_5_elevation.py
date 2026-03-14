"""
Script 5: Download Elevation (Static) — 2° global grid
Run: python3 dl_5_elevation.py
Finishes in ~30 seconds
"""
import ee, pandas as pd, os, time

ee.Initialize(project='pm25-earth-engine')

GRID_STEP = 2.0
LAT_RANGE = (-60, 72)
LON_RANGE = (-180, 180)
CHUNK_SIZE = 1500
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, 'data', 'elevation_global.csv')

if os.path.exists(OUTPUT):
    print("⏭️  Elevation already exists, skipping."); exit()

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
print(f"⛰️  Elevation: {n_pts} points, {len(chunks)} chunks")

elev = ee.Image("USGS/SRTMGL1_003").select("elevation")
rows = []
t = time.time()
for ci, c in enumerate(chunks):
    print(f"   Chunk {ci+1}/{len(chunks)}...")
    r = elev.reduceRegions(collection=c, reducer=ee.Reducer.mean(), scale=5000).getInfo()
    for f in r['features']:
        p = f['properties']
        rows.append({'lat': p['lat'], 'lon': p['lon'], 'elevation': p.get('mean')})

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f"\n✅ Elevation done: {len(df):,} rows | {df['elevation'].notna().sum():,} non-null | {time.time()-t:.0f}s")
