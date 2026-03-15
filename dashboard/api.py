"""
api.py — Flask prediction API for PM2.5 forecast dashboard
Run: python3 api.py
Serves on: http://localhost:5050
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import xgboost as xgb
import os

app = Flask(__name__)
CORS(app)

# Robust Path Handling for Local and Render.com
# Find project root (one level up from dashboard/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR     = os.path.join(PROJECT_ROOT, 'models')
CSV_PATH     = os.path.join(PROJECT_ROOT, 'data', 'final_land_dataset.csv')

# Load full dataset into memory at startup for fast /snapshot queries
print("📂 Loading full dataset for time slider...")
try:
    _DF = pd.read_csv(CSV_PATH, usecols=['lat','lon','year','month','pm25'])
    _DF['pm25'] = pd.to_numeric(_DF['pm25'], errors='coerce')
    _DF = _DF.dropna(subset=['pm25'])
    _DF['pm25'] = _DF['pm25'].clip(0, 999).round(2)
    # Available months meta
    _MONTHS = sorted(_DF[['year','month']].drop_duplicates().values.tolist())
    print(f"✅ Dataset loaded: {len(_DF)} rows, {len(_MONTHS)} months ({_MONTHS[0]} → {_MONTHS[-1]})")
except Exception as e:
    print(f"⚠️  Could not load full dataset ({e}). /snapshot will be unavailable.")
    _DF = None
    _MONTHS = []



CATS = [
    (0,   5,   "Good",                        "#22d3ee", "WHO Safe"),
    (5,   15,  "Moderate",                    "#facc15", "Acceptable"),
    (15,  25,  "Unhealthy for Sensitive",     "#fb923c", "Limit outdoor activity"),
    (25,  50,  "Unhealthy",                   "#f87171", "Avoid prolonged outdoor exposure"),
    (50,  150, "Very Unhealthy",              "#c084fc", "Stay indoors"),
    (150, 999, "Hazardous",                   "#ff4444", "Emergency conditions"),
]

def get_cat(pm):
    for lo, hi, label, color, advice in CATS:
        if lo <= pm < hi:
            return label, color, advice
    return "Hazardous", "#ff4444", "Emergency conditions"

# Load all 3 models on startup
MODELS = {}
for h in ['24h', '48h', '72h']:
    m = xgb.XGBRegressor()
    m.load_model(os.path.join(BASE_DIR, f'pm25_model_{h}.json'))
    MODELS[h] = m
print("✅ Models loaded: 24h, 48h, 72h")

FEATURES = ['lat', 'lon', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'temperature_celsius', 'relative_humidity', 'wind_speed', 'wind_direction',
            'surface_pressure', 'aod', 'cloud_fraction', 'elevation',
            'pm25_lag_1d', 'pm25_lag_2d', 'pm25_lag_3d', 'pm25_lag_7d',
            'pm25_roll_3d', 'pm25_roll_7d', 'pm25_roll_14d']

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    lat   = float(d['lat'])
    lon   = float(d['lon'])
    month = int(d.get('month', 12))
    doy   = int(d.get('day_of_year', 345))
    pm0   = float(d['pm_today'])
    pm1   = float(d['pm_1d'])
    pm2   = float(d['pm_2d'])
    pm3   = float(d['pm_3d'])
    pm7   = float(d.get('pm_7d', pm3))
    temp  = float(d.get('temp_c', 22))
    hum   = float(d.get('humidity', 55))
    wind  = float(d.get('wind_speed', 3))
    aod   = float(d.get('aod', 0.3))
    pres  = float(d.get('pressure', 101325))
    cloud = float(d.get('cloud', 0.4))
    elev  = float(d.get('elevation', 200))

    row = {
        'lat': lat, 'lon': lon,
        'month_sin':  np.sin(2 * np.pi * month / 12),
        'month_cos':  np.cos(2 * np.pi * month / 12),
        'day_sin':    np.sin(2 * np.pi * doy / 365),
        'day_cos':    np.cos(2 * np.pi * doy / 365),
        'temperature_celsius': temp,
        'relative_humidity':   hum,
        'wind_speed':          wind,
        'wind_direction':      180.0,
        'surface_pressure':    pres,
        'aod':         aod,
        'cloud_fraction': cloud,
        'elevation':   elev,
        'pm25_lag_1d': pm1,
        'pm25_lag_2d': pm2,
        'pm25_lag_3d': pm3,
        'pm25_lag_7d': pm7,
        'pm25_roll_3d':  np.mean([pm0, pm1, pm2]),
        'pm25_roll_7d':  np.mean([pm0, pm1, pm2, pm3, pm7, pm7, pm7]),
        'pm25_roll_14d': np.mean([pm0, pm1, pm2, pm3] + [pm7]*10),
    }
    X = pd.DataFrame([row])[FEATURES]

    results = {}
    for h, model in MODELS.items():
        pred = max(0.0, float(model.predict(X)[0]))
        lbl, col, adv = get_cat(pred)
        results[h] = {'pm25': round(pred, 1), 'label': lbl, 'color': col, 'advice': adv}

    return jsonify({'status': 'ok', 'forecasts': results, 'current': pm0})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    rows = request.json.get('rows', [])
    if not rows:
        return jsonify({'error': 'No rows provided'}), 400

    results = []
    for d in rows:
        lat   = float(d['lat']);   lon   = float(d['lon'])
        month = int(d.get('month', 12));  doy = int(d.get('day_of_year', 345))
        pm0 = float(d['pm_today']); pm1 = float(d['pm_1d'])
        pm2 = float(d['pm_2d']);    pm3 = float(d['pm_3d'])
        pm7 = float(d.get('pm_7d', pm3))
        row = {
            'lat': lat, 'lon': lon,
            'month_sin': np.sin(2*np.pi*month/12), 'month_cos': np.cos(2*np.pi*month/12),
            'day_sin':   np.sin(2*np.pi*doy/365),  'day_cos':   np.cos(2*np.pi*doy/365),
            'temperature_celsius': float(d.get('temp_c', 22)),
            'relative_humidity':   float(d.get('humidity', 55)),
            'wind_speed':          float(d.get('wind_speed', 3)),
            'wind_direction':      180.0,
            'surface_pressure':    float(d.get('pressure', 101325)),
            'aod':          float(d.get('aod', 0.3)),
            'cloud_fraction': float(d.get('cloud', 0.4)),
            'elevation':    float(d.get('elevation', 200)),
            'pm25_lag_1d': pm1, 'pm25_lag_2d': pm2, 'pm25_lag_3d': pm3, 'pm25_lag_7d': pm7,
            'pm25_roll_3d':  np.mean([pm0, pm1, pm2]),
            'pm25_roll_7d':  np.mean([pm0, pm1, pm2, pm3, pm7, pm7, pm7]),
            'pm25_roll_14d': np.mean([pm0, pm1, pm2, pm3] + [pm7]*10),
        }
        X = pd.DataFrame([row])[FEATURES]
        preds = {}
        for h, model in MODELS.items():
            pred = max(0.0, float(model.predict(X)[0]))
            lbl, col, adv = get_cat(pred)
            preds[h] = {'pm25': round(pred, 1), 'label': lbl, 'color': col}

        entry = {'lat': lat, 'lon': lon, 'pm_today': pm0, 'forecasts': preds}
        for h in ['24h','48h','72h']:
            k = f'actual_{h}'
            if k in d and d[k] not in (None, ''):
                actual = float(d[k])
                entry[f'actual_{h}'] = actual
                entry[f'err_{h}'] = round(abs(preds[h]['pm25'] - actual), 2)
        results.append(entry)

    # Compute accuracy metrics if actuals provided
    metrics = {}
    for h in ['24h','48h','72h']:
        ak = f'actual_{h}'
        rows_with_actual = [r for r in results if ak in r]
        if rows_with_actual:
            y_true = np.array([r[ak] for r in rows_with_actual])
            y_pred = np.array([r['forecasts'][h]['pm25'] for r in rows_with_actual])
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2   = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
            mae  = float(np.mean(np.abs(y_true - y_pred)))
            metrics[h] = {'r2': round(r2, 4), 'rmse': round(rmse, 2), 'mae': round(mae, 2), 'n': len(rows_with_actual)}

    return jsonify({'status': 'ok', 'results': results, 'metrics': metrics})

@app.route('/months')
def months():
    if _DF is None:
        return jsonify({'error': 'Dataset not loaded'}), 503
    return jsonify({'months': _MONTHS})  # [[year,month], ...]

@app.route('/snapshot')
def snapshot():
    if _DF is None:
        return jsonify({'error': 'Dataset not loaded'}), 503
    try:
        year  = int(request.args.get('year',  2021))
        month = int(request.args.get('month', 12))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid year/month'}), 400

    sub = _DF[(_DF['year'] == year) & (_DF['month'] == month)][['lat','lon','pm25']]
    if sub.empty:
        return jsonify({'error': f'No data for {year}-{month:02d}'}), 404

    points = sub.to_dict('records')
    mean_pm = round(float(sub['pm25'].mean()), 2)
    hazard  = int((sub['pm25'] > 50).sum())
    safe    = int((sub['pm25'] <= 5).sum())
    top10   = sub.nlargest(10, 'pm25').to_dict('records')
    return jsonify({
        'year': year, 'month': month,
        'snapshot': points,
        'stats': {'mean': mean_pm, 'hazard_zones': hazard, 'safe_zones': safe, 'total_points': len(points)},
        'top10': top10
    })

@app.route('/health')
def health():
    return jsonify({'status': 'running', 'models': list(MODELS.keys()), 'months_available': len(_MONTHS)})



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"🚀 PM2.5 Prediction API running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
