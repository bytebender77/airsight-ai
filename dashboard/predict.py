"""
predict.py — Live PM2.5 forecaster for judge evaluation
========================================================
Usage:
  python3 predict.py

The script asks for a lat/lon and recent PM2.5 values,
then outputs 24h / 48h / 72h predictions with health category.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATS = [
    (0,   5,   "Good ✅",                     "WHO Safe"),
    (5,   15,  "Moderate 🟡",                  "Acceptable"),
    (15,  25,  "Unhealthy for Sensitive 🟠",   "Limit outdoor activity"),
    (25,  50,  "Unhealthy 🔴",                 "Avoid prolonged outdoor exposure"),
    (50,  150, "Very Unhealthy 🟣",            "Stay indoors"),
    (150, 999, "Hazardous ☠️",                 "Emergency conditions"),
]

def get_cat(pm):
    for lo, hi, label, advice in CATS:
        if lo <= pm < hi:
            return label, advice
    return "Hazardous ☠️", "Emergency conditions"


def load_models():
    models = {}
    for h in ['24h', '48h', '72h']:
        path = os.path.join(BASE_DIR, f'pm25_model_{h}.json')
        if not os.path.exists(path):
            print(f"❌ Model not found: {path}")
            exit(1)
        m = xgb.XGBRegressor()
        m.load_model(path)
        models[h] = m
    return models

# Features the model expects (exact order matters!)
FEATURES = ['lat', 'lon', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'temperature_celsius', 'relative_humidity', 'wind_speed', 'wind_direction',
            'surface_pressure', 'aod', 'cloud_fraction', 'elevation',
            'pm25_lag_1d', 'pm25_lag_2d', 'pm25_lag_3d', 'pm25_lag_7d',
            'pm25_roll_3d', 'pm25_roll_7d', 'pm25_roll_14d']


def build_feature_row(lat, lon, month, day_of_year,
                      pm_today, pm_1d, pm_2d, pm_3d, pm_7d,
                      temp_c=22.0, humidity=55.0, wind_speed=3.0,
                      wind_dir=180.0, pressure=101325.0, aod=0.3,
                      cloud=0.4, elevation=200.0):
    """Build input feature row for the XGBoost models."""
    row = {
        'lat':               lat,
        'lon':               lon,
        'month_sin':         np.sin(2 * np.pi * month / 12),
        'month_cos':         np.cos(2 * np.pi * month / 12),
        'day_sin':           np.sin(2 * np.pi * day_of_year / 365),
        'day_cos':           np.cos(2 * np.pi * day_of_year / 365),
        'temperature_celsius': temp_c,
        'relative_humidity': humidity,
        'wind_speed':        wind_speed,
        'wind_direction':    wind_dir,
        'surface_pressure':  pressure,
        'aod':               aod,
        'cloud_fraction':    cloud,
        'elevation':         elevation,
        'pm25_lag_1d':       pm_1d,       # Yesterday
        'pm25_lag_2d':       pm_2d,       # 2 days ago
        'pm25_lag_3d':       pm_3d,       # 3 days ago
        'pm25_lag_7d':       pm_7d,       # 7 days ago
        'pm25_roll_3d':      np.mean([pm_today, pm_1d, pm_2d]),   # 3-day rolling
        'pm25_roll_7d':      np.mean([pm_today if i==0 else pm_1d for i in range(7)]),  # approx
        'pm25_roll_14d':     np.mean([pm_today if i==0 else pm_1d for i in range(14)]), # approx
    }
    return pd.DataFrame([row])[FEATURES]

# ─── Main ────────────────────────────────────────────────────────────
print("=" * 60)
print("🌍 PM2.5 Short-Term Forecast — AirSight AI")
print("   Model: XGBoost | Trained: 2015-2020 | Test R²: 0.979")
print("=" * 60)

models = load_models()

print("\n📍 Enter location:")
lat = float(input("   Latitude  (-60 to 72): "))
lon = float(input("   Longitude (-180 to 180): "))

print("\n📅 Enter date:")
month       = int(input("   Month (1-12): "))
day_of_year = int(input("   Day of year (1-365): "))

print("\n💨 Enter recent PM2.5 readings (µg/m³):")
pm_today = float(input("   Today (current):   "))
pm_1d    = float(input("   Yesterday:         "))
pm_2d    = float(input("   2 days ago:        "))
pm_3d    = float(input("   3 days ago:        "))
pm_7d    = float(input("   7 days ago:        "))

print("\n🌡️  Enter current weather (or press Enter for defaults):")
temp_raw = input("   Temperature (°C) [default 22]: ").strip()
hum_raw  = input("   Relative Humidity (%) [default 55]: ").strip()
wind_raw = input("   Wind Speed (m/s) [default 3]: ").strip()
pres_raw = input("   Surface Pressure (Pa) [default 101325]: ").strip()
aod_raw  = input("   AOD (Aerosol Optical Depth) [default 0.3]: ").strip()

temp_c    = float(temp_raw)  if temp_raw  else 22.0
humidity  = float(hum_raw)   if hum_raw   else 55.0
wind_s    = float(wind_raw)  if wind_raw  else 3.0
pressure  = float(pres_raw)  if pres_raw  else 101325.0
aod       = float(aod_raw)   if aod_raw   else 0.3

# Build input
X = build_feature_row(lat, lon, month, day_of_year,
                      pm_today, pm_1d, pm_2d, pm_3d, pm_7d,
                      temp_c, humidity, wind_s, 180.0, pressure, aod)

# Predict
print("\n" + "=" * 60)
print("🔮 FORECAST RESULTS")
print("=" * 60)
print(f"   Location: ({lat}°, {lon}°)  |  Month: {month}  |  Day: {day_of_year}")
print(f"   Current PM2.5: {pm_today:.1f} µg/m³")
print("-" * 60)

for horizon, key in [("24 hours (tomorrow)", "24h"),
                      ("48 hours (day after)", "48h"),
                      ("72 hours (3 days out)", "72h")]:
    pred = max(0.0, float(models[key].predict(X)[0]))
    cat, advice = get_cat(pred)
    print(f"\n  +{horizon}")
    print(f"    PM2.5:  {pred:.1f} µg/m³")
    print(f"    Status: {cat}")
    print(f"    Advice: {advice}")

print("\n" + "=" * 60)
print("⚕️  WHO Annual Standard: 5 µg/m³ | WHO 24h Standard: 15 µg/m³")
print("=" * 60)
