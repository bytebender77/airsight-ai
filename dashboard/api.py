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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

@app.route('/health')
def health():
    return jsonify({'status': 'running', 'models': list(MODELS.keys())})

if __name__ == '__main__':
    print("🚀 PM2.5 Prediction API running at http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
