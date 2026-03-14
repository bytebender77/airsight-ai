"""
Step 3: Train 24/48/72hr forecast models using XGBoost.
Input:  /pm25_daily/data/daily_features.csv
Output: 3 model files + evaluation + feature importance

Run: python3 step3_train_forecast.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

INPUT   = '/Users/kunalkumargupta/Desktop/hackathon/pm25_daily/data/daily_features.csv'
OUT_DIR = '/Users/kunalkumargupta/Desktop/hackathon/pm25_daily'

FEATURES = [
    'lat', 'lon',
    'month_sin', 'month_cos', 'day_sin', 'day_cos',
    'temperature_celsius', 'relative_humidity', 'wind_speed', 'wind_direction', 'surface_pressure',
    'aod', 'cloud_fraction', 'elevation',
    'pm25_lag_1d', 'pm25_lag_2d', 'pm25_lag_3d', 'pm25_lag_7d',
    'pm25_roll_3d', 'pm25_roll_7d', 'pm25_roll_14d'
]

print("📦 Loading daily feature dataset...")
df = pd.read_csv(INPUT, parse_dates=['date'])
print(f"   {len(df):,} rows | {df['date'].min()} → {df['date'].max()}")

# Available features only
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"   Using {len(FEATURES)} features")

# ─── Time-based split: train on 2015-2020, test on 2021 ────────
train = df[df['year'] < 2021]
test  = df[df['year'] == 2021]
print(f"\n   Train: {len(train):,} | Test: {len(test):,}")

results = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
feat_imp_all = {}

for i, (horizon, target_col) in enumerate([
    ('24h', 'target_24h'),
    ('48h', 'target_48h'),
    ('72h', 'target_72h')
]):
    print(f"\n{'='*60}")
    print(f"🧠 Training {horizon} forecast model...")

    # Drop rows where target is NaN (last days of each series)
    tr = train.dropna(subset=[target_col])
    te = test.dropna(subset=[target_col])

    X_train, y_train = tr[FEATURES], tr[target_col]
    X_test,  y_test  = te[FEATURES], te[target_col]

    model = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=200)

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)

    results[horizon] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
    print(f"   R²:   {r2:.4f}")
    print(f"   RMSE: {rmse:.4f} µg/m³")
    print(f"   MAE:  {mae:.4f} µg/m³")

    model_path = os.path.join(OUT_DIR, f'pm25_model_{horizon}.json')
    model.save_model(model_path)
    print(f"   ✅ Saved: {model_path}")

    feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    feat_imp_all[horizon] = feat_imp

    # Scatter plot: predicted vs actual
    axes[i].scatter(y_test[:3000], preds[:3000], alpha=0.2, s=2, color='steelblue')
    lim = max(y_test[:3000].max(), preds[:3000].max())
    axes[i].plot([0, lim], [0, lim], 'r--', lw=1)
    axes[i].set_title(f'{horizon} Forecast\nR²={r2:.3f}  RMSE={rmse:.2f}')
    axes[i].set_xlabel('Actual PM2.5'); axes[i].set_ylabel('Predicted PM2.5')

plt.suptitle('PM2.5 Forecast: Predicted vs Actual (2021 Test Set)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'forecast_accuracy.png'), dpi=150)
print(f"\n✅ Saved: forecast_accuracy.png")

# ─── Final summary ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("📊 FORECAST MODEL SUMMARY")
print(f"{'='*60}")
print(f"{'Horizon':10s} {'R²':>8s} {'RMSE (µg/m³)':>14s} {'MAE (µg/m³)':>13s}")
print("-"*50)
for h, m in results.items():
    print(f"{h:10s} {m['R2']:8.4f} {m['RMSE']:14.4f} {m['MAE']:13.4f}")

# Feature importance for 24h model
feat_imp_all['24h'].head(15).sort_values().plot(
    kind='barh', figsize=(10, 6),
    title='Feature Importance — 24h PM2.5 Forecast Model'
)
plt.xlabel('Importance'); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_importance_daily.png'), dpi=150)
print(f"\n✅ Saved: feature_importance_daily.png")
print("\n🎉 All 3 forecast models ready!")
