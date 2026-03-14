"""
Script 6: Merge all datasets + Feature Engineering
Run AFTER all dl_1 through dl_5 are done.
Run: python3 merge_and_train.py
"""
import pandas as pd, numpy as np, os, xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ─── Load ────────────────────────────────────────────────────
print("📂 Loading datasets...")
pm25    = pd.read_csv(os.path.join(DATA_DIR, 'pm25_global.csv'))
weather = pd.read_csv(os.path.join(DATA_DIR, 'weather_global.csv'))
aod     = pd.read_csv(os.path.join(DATA_DIR, 'aod_global.csv'))
cloud   = pd.read_csv(os.path.join(DATA_DIR, 'cloud_global.csv'))
elev    = pd.read_csv(os.path.join(DATA_DIR, 'elevation_global.csv'))
for name, d in [('PM2.5',pm25),('Weather',weather),('AOD',aod),('Cloud',cloud),('Elev',elev)]:
    print(f"   ✅ {name}: {d.shape}")

# ─── Merge ───────────────────────────────────────────────────
print("\n🔗 Merging...")
df = pm25.merge(weather, on=['date','lat','lon'], how='left')
df = df.merge(aod, on=['date','lat','lon'], how='left')
df = df.merge(cloud, on=['date','lat','lon'], how='left')
df = df.merge(elev, on=['lat','lon'], how='left')
print(f"   Shape: {df.shape}")

# ─── Feature Engineering ─────────────────────────────────────
print("\n⚙️  Engineering features...")
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

if 'u_component_of_wind_10m' in df.columns:
    df['wind_speed'] = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)
    df['wind_direction'] = np.degrees(np.arctan2(df['v_component_of_wind_10m'], df['u_component_of_wind_10m']))

if 'temperature_2m' in df.columns:
    t_c = df['temperature_2m'] - 273.15
    d_c = df['dewpoint_temperature_2m'] - 273.15
    df['relative_humidity'] = 100 * np.exp((17.625*d_c)/(243.04+d_c)) / np.exp((17.625*t_c)/(243.04+t_c))
    df['temperature_celsius'] = t_c

df = df.sort_values(['lat','lon','date'])
for lag in [1,2,3]:
    df[f'pm25_lag{lag}'] = df.groupby(['lat','lon'])['pm25'].shift(lag)
df['pm25_rolling_3m'] = df.groupby(['lat','lon'])['pm25'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df = df.dropna(subset=['pm25'])

out = os.path.join(DATA_DIR, 'final_global_dataset.csv')
df.to_csv(out, index=False)
print(f"🎉 Merged dataset: {df.shape} → {out}")

# ─── Train Model ─────────────────────────────────────────────
print("\n🧠 Training XGBoost...")
FEATURES = [c for c in ['lat','lon','month_sin','month_cos','temperature_celsius',
    'relative_humidity','wind_speed','wind_direction','surface_pressure','aod',
    'cloud_fraction','elevation','pm25_lag1','pm25_lag2','pm25_lag3','pm25_rolling_3m'] if c in df.columns]

train = df[df['year'] < 2021]
test  = df[df['year'] == 2021]
print(f"   Train: {len(train):,} rows | Test: {len(test):,} rows")

model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
model.fit(train[FEATURES], train['pm25'], eval_set=[(test[FEATURES], test['pm25'])], verbose=100)

preds = model.predict(test[FEATURES])
print(f"\n📊 Results:")
print(f"   R²:   {r2_score(test['pm25'], preds):.4f}")
print(f"   RMSE: {mean_squared_error(test['pm25'], preds, squared=False):.4f}")
print(f"   MAE:  {mean_absolute_error(test['pm25'], preds):.4f}")

model.save_model(os.path.join(BASE_DIR, 'pm25_model_2deg.json'))

feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n🔝 Top Features:"); print(feat_imp.head(10))

plt.figure(figsize=(10,6))
feat_imp.head(15).plot(kind='barh')
plt.title('Feature Importance (Global PM2.5, 2° Grid, 2015-2022)')
plt.xlabel('Importance'); plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'feature_importance_2deg.png'))
print(f"\n✅ Model saved. Plot saved.")
