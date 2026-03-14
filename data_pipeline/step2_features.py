"""
Step 2: Build daily features + lag variables for 24/48/72hr forecasting.
Input:  /pm25_daily/data/pm25_daily.csv
Uses:   existing monthly weather/AOD/cloud/elevation from pm25_2deg
Output: /pm25_daily/data/daily_features.csv

Run: python3 step2_features.py
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

DAILY_PM25  = '/Users/kunalkumargupta/Desktop/hackathon/pm25_daily/data/pm25_daily.csv'
MONTHLY_DIR = '/Users/kunalkumargupta/Desktop/hackathon/pm25_2deg/data'
OUTPUT      = '/Users/kunalkumargupta/Desktop/hackathon/pm25_daily/data/daily_features.csv'

print("📦 Loading daily PM2.5...")
df = pd.read_csv(DAILY_PM25, parse_dates=['date'])
print(f"   {len(df):,} rows")

# ─── Add date features ─────────────────────────────────────────
df['month']      = df['date'].dt.month
df['year']       = df['date'].dt.year
df['day_of_year']= df['date'].dt.dayofyear
df['day_of_month']= df['date'].dt.day
df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin']    = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos']    = np.cos(2 * np.pi * df['day_of_year'] / 365)

# ─── Merge monthly weather (approximate: assign month's weather to each day) ──
print("\n📊 Merging monthly weather features...")
weather = pd.read_csv(os.path.join(MONTHLY_DIR, 'weather_global.csv'), parse_dates=['date'])
aod     = pd.read_csv(os.path.join(MONTHLY_DIR, 'aod_global.csv'), parse_dates=['date'])
cloud   = pd.read_csv(os.path.join(MONTHLY_DIR, 'cloud_global.csv'), parse_dates=['date'])
elev    = pd.read_csv(os.path.join(MONTHLY_DIR, 'elevation_global.csv'))

# Add year/month to monthly tables for merging
for d in [weather, aod, cloud]:
    d['year']  = d['date'].dt.year
    d['month'] = d['date'].dt.month

# Merge on year+month+lat+lon
df = df.merge(weather.drop(columns='date'), on=['year','month','lat','lon'], how='left')
df = df.merge(aod.drop(columns='date'),     on=['year','month','lat','lon'], how='left')
df = df.merge(cloud.drop(columns='date'),   on=['year','month','lat','lon'], how='left')
df = df.merge(elev, on=['lat','lon'], how='left')
print(f"   After merge: {df.shape}")

# ─── Derive weather features ───────────────────────────────────
if 'temperature_2m' in df.columns:
    t_c = df['temperature_2m'] - 273.15
    d_c = df['dewpoint_temperature_2m'] - 273.15
    df['relative_humidity']   = 100 * np.exp((17.625*d_c)/(243.04+d_c)) / np.exp((17.625*t_c)/(243.04+t_c))
    df['temperature_celsius'] = t_c
    df['wind_speed']     = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)
    df['wind_direction'] = np.degrees(np.arctan2(df['v_component_of_wind_10m'], df['u_component_of_wind_10m']))
    print("   ✅ Derived weather features")

# ─── Daily Lag Features ────────────────────────────────────────
print("\n⚙️  Creating daily lag features per grid point...")
df = df.sort_values(['lat','lon','date'])

for lag in [1, 2, 3, 7]:
    df[f'pm25_lag_{lag}d']    = df.groupby(['lat','lon'])['pm25'].shift(lag)

for win in [3, 7, 14]:
    df[f'pm25_roll_{win}d']   = df.groupby(['lat','lon'])['pm25'].transform(
        lambda x: x.shift(1).rolling(win, min_periods=1).mean())

# ─── Target Variables (forecast horizons) ─────────────────────
print("🎯 Creating forecast targets (day+1, day+2, day+3)...")
df['target_24h'] = df.groupby(['lat','lon'])['pm25'].shift(-1)   # next day
df['target_48h'] = df.groupby(['lat','lon'])['pm25'].shift(-2)   # 2 days ahead
df['target_72h'] = df.groupby(['lat','lon'])['pm25'].shift(-3)   # 3 days ahead

# Drop rows with no lag features (first 7 days of each series)
df = df.dropna(subset=['pm25_lag_1d', 'target_24h'])
print(f"   Final rows: {len(df):,}")

df.to_csv(OUTPUT, index=False)
print(f"\n✅ Saved daily feature dataset: {OUTPUT}")
print(f"   Shape: {df.shape}")
print(f"\n   Sample columns: {list(df.columns)}")
