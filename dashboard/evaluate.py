"""
evaluate.py — Judge Evaluation Script for AirSight AI
=======================================================
Usage:
  python3 evaluate.py --file judge_data.csv

The judge provides a CSV with:
  Required columns (model inputs):
    lat, lon, month, day_of_year, pm_today, pm_1d, pm_2d, pm_3d, pm_7d

  Optional weather columns (defaults used if missing):
    temp_c, humidity, wind_speed, aod, pressure, elevation

  Required columns (ground truth for accuracy check):
    actual_24h, actual_48h, actual_72h

Output:
  - Per-row predictions vs actuals
  - R², RMSE, MAE for each horizon
  - Saved: evaluation_results.csv

Example CSV (judge_sample.csv is included):
  python3 evaluate.py --file judge_sample.csv
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

CATS = [
    (0,   5,   "Good ✅"),
    (5,   15,  "Moderate 🟡"),
    (15,  25,  "Unhealthy for Sensitive 🟠"),
    (25,  50,  "Unhealthy 🔴"),
    (50,  150, "Very Unhealthy 🟣"),
    (150, 999, "Hazardous ☠️"),
]
def get_cat(pm):
    for lo, hi, label in CATS:
        if lo <= pm < hi: return label
    return "Hazardous ☠️"

FEATURES = [
    'lat', 'lon', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
    'temperature_celsius', 'relative_humidity', 'wind_speed', 'wind_direction',
    'surface_pressure', 'aod', 'cloud_fraction', 'elevation',
    'pm25_lag_1d', 'pm25_lag_2d', 'pm25_lag_3d', 'pm25_lag_7d',
    'pm25_roll_3d', 'pm25_roll_7d', 'pm25_roll_14d'
]

def load_models():
    models = {}
    for h in ['24h', '48h', '72h']:
        path = os.path.join(MODEL_DIR, f'pm25_model_{h}.json')
        if not os.path.exists(path):
            print(f"❌ Model not found: {path}")
            sys.exit(1)
        m = xgb.XGBRegressor()
        m.load_model(path)
        models[h] = m
    return models

def build_features(row):
    pm0 = float(row['pm_today'])
    pm1 = float(row['pm_1d'])
    pm2 = float(row['pm_2d'])
    pm3 = float(row['pm_3d'])
    pm7 = float(row.get('pm_7d', pm3))
    month = int(row['month'])
    doy   = int(row['day_of_year'])
    return {
        'lat':                float(row['lat']),
        'lon':                float(row['lon']),
        'month_sin':          np.sin(2 * np.pi * month / 12),
        'month_cos':          np.cos(2 * np.pi * month / 12),
        'day_sin':            np.sin(2 * np.pi * doy / 365),
        'day_cos':            np.cos(2 * np.pi * doy / 365),
        'temperature_celsius': float(row.get('temp_c', 22)),
        'relative_humidity':  float(row.get('humidity', 55)),
        'wind_speed':         float(row.get('wind_speed', 3)),
        'wind_direction':     180.0,
        'surface_pressure':   float(row.get('pressure', 101325)),
        'aod':                float(row.get('aod', 0.3)),
        'cloud_fraction':     float(row.get('cloud', 0.4)),
        'elevation':          float(row.get('elevation', 200)),
        'pm25_lag_1d':        pm1,
        'pm25_lag_2d':        pm2,
        'pm25_lag_3d':        pm3,
        'pm25_lag_7d':        pm7,
        'pm25_roll_3d':       np.mean([pm0, pm1, pm2]),
        'pm25_roll_7d':       np.mean([pm0, pm1, pm2, pm3, pm7, pm7, pm7]),
        'pm25_roll_14d':      np.mean([pm0, pm1, pm2, pm3] + [pm7]*10),
    }

def main():
    parser = argparse.ArgumentParser(description='AirSight AI — Judge Evaluation')
    parser.add_argument('--file', required=True, help='Path to judge CSV file')
    parser.add_argument('--out',  default='evaluation_results.csv', help='Output file')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    print("=" * 65)
    print("🌍  AirSight AI — PM2.5 Forecast Evaluation")
    print("    Model: XGBoost | Trained: 2015-2020 | Test R²: 0.979")
    print("=" * 65)

    models = load_models()
    df = pd.read_csv(args.file)
    print(f"\n📂 Loaded: {args.file}  ({len(df)} rows)\n")

    has_actuals = all(c in df.columns for c in ['actual_24h', 'actual_48h', 'actual_72h'])
    if not has_actuals:
        print("ℹ️  No actual_* columns found — running prediction-only mode.\n")

    results = []
    for i, row in df.iterrows():
        feats = build_features(row)
        X = pd.DataFrame([feats])[FEATURES]
        preds = {h: max(0.0, float(models[h].predict(X)[0])) for h in ['24h', '48h', '72h']}

        result = {
            'lat':      row['lat'],
            'lon':      row['lon'],
            'pm_today': row['pm_today'],
            'pred_24h': round(preds['24h'], 2),
            'pred_48h': round(preds['48h'], 2),
            'pred_72h': round(preds['72h'], 2),
            'cat_24h':  get_cat(preds['24h']),
            'cat_48h':  get_cat(preds['48h']),
            'cat_72h':  get_cat(preds['72h']),
        }
        if has_actuals:
            result['actual_24h'] = row['actual_24h']
            result['actual_48h'] = row['actual_48h']
            result['actual_72h'] = row['actual_72h']
            result['err_24h'] = round(abs(preds['24h'] - float(row['actual_24h'])), 2)
            result['err_48h'] = round(abs(preds['48h'] - float(row['actual_48h'])), 2)
            result['err_72h'] = round(abs(preds['72h'] - float(row['actual_72h'])), 2)

        results.append(result)
        print(f"  Row {i+1:3d} | ({row['lat']:6.1f}, {row['lon']:7.1f}) | today={row['pm_today']:6.1f} "
              f"→ 24h: {preds['24h']:6.1f}  48h: {preds['48h']:6.1f}  72h: {preds['72h']:6.1f}"
              + (f"  |  err: {result['err_24h']:.1f}" if has_actuals else ""))

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out, index=False)
    print(f"\n✅ Results saved: {args.out}")

    if has_actuals:
        print("\n" + "=" * 65)
        print("📊 ACCURACY METRICS")
        print("=" * 65)
        for h in ['24h', '48h', '72h']:
            y_true = res_df[f'actual_{h}'].astype(float)
            y_pred = res_df[f'pred_{h}'].astype(float)
            r2   = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            print(f"\n  +{h} Forecast:")
            print(f"    R²   = {r2:.4f}  {'✅' if r2 > 0.85 else '⚠️'}")
            print(f"    RMSE = {rmse:.2f} µg/m³")
            print(f"    MAE  = {mae:.2f} µg/m³")
        print("\n" + "=" * 65)

if __name__ == '__main__':
    main()
