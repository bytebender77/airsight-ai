"""
Step 1: Interpolate monthly PM2.5 to daily using cubic spline.
Input:  /pm25_2deg/data/final_land_dataset.csv  (monthly, land-only)
Output: /pm25_daily/data/pm25_daily.csv         (daily per grid point)

Run: python3 step1_interpolate.py
Estimated time: 3-5 minutes
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import os, warnings
warnings.filterwarnings('ignore')

INPUT  = '/Users/kunalkumargupta/Desktop/hackathon/pm25_2deg/data/final_land_dataset.csv'
OUTPUT = '/Users/kunalkumargupta/Desktop/hackathon/pm25_daily/data/pm25_daily.csv'

print("📦 Loading monthly land dataset...")
df = pd.read_csv(INPUT, parse_dates=['date'])
print(f"   {len(df):,} rows | {df[['lat','lon']].drop_duplicates().shape[0]:,} unique grid points")

# We assign each monthly value to the 15th of the month (midpoint)
df['knot_date'] = pd.to_datetime({'year': df['date'].dt.year,
                                   'month': df['date'].dt.month,
                                   'day': 15})
df = df.sort_values(['lat', 'lon', 'knot_date'])

# Build complete daily date range
daily_dates = pd.date_range('2015-01-01', '2021-12-31', freq='D')

rows = []
locations = df[['lat','lon']].drop_duplicates().values
total = len(locations)
print(f"\n🔄 Interpolating {total:,} grid points to daily...")

for idx, (lat, lon) in enumerate(locations):
    if idx % 500 == 0:
        print(f"   [{idx:,}/{total:,}] lat={lat}, lon={lon}")

    sub = df[(df['lat'] == lat) & (df['lon'] == lon)].sort_values('knot_date')
    if len(sub) < 3:
        continue

    # Numeric time axis (days since first date)
    t0 = sub['knot_date'].min()
    t_knots = (sub['knot_date'] - t0).dt.days.values
    pm25_knots = sub['pm25'].values

    # Clamp negatives (PM2.5 can't be negative)
    pm25_knots = np.maximum(pm25_knots, 0)

    # Fit cubic spline through the monthly midpoints
    cs = CubicSpline(t_knots, pm25_knots, extrapolate=False)

    # Evaluate on every day in range
    t_start = (daily_dates[0] - t0).days
    t_end   = (daily_dates[-1] - t0).days

    # Only interpolate between first and last knot
    t_valid_mask = (daily_dates >= sub['knot_date'].min()) & (daily_dates <= sub['knot_date'].max())
    valid_dates  = daily_dates[t_valid_mask]
    t_daily      = (valid_dates - t0).days

    pm25_daily = cs(t_daily)
    pm25_daily = np.maximum(pm25_daily, 0)  # clamp negatives from spline overshoot

    for date, pm25_val in zip(valid_dates, pm25_daily):
        rows.append({'date': date.date(), 'lat': lat, 'lon': lon, 'pm25': round(float(pm25_val), 4)})

print(f"\n💾 Saving {len(rows):,} daily rows...")
result = pd.DataFrame(rows)
result.to_csv(OUTPUT, index=False)

print(f"\n✅ Done!")
print(f"   Rows: {len(result):,}")
print(f"   Date range: {result['date'].min()} → {result['date'].max()}")
print(f"   PM2.5 range: {result['pm25'].min():.2f} – {result['pm25'].max():.2f} µg/m³")
print(f"   File: {OUTPUT}")
