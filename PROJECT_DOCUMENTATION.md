# 📖 AirSight AI: Complete Project Documentation

> A step-by-step technical walkthrough of how we built AirSight AI — from raw satellite data to a production-ready global air quality forecasting platform.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Phase 1: Data Extraction](#2-phase-1-data-extraction-google-earth-engine)
3. [Phase 2: Data Merging & Cleaning](#3-phase-2-data-merging--cleaning)
4. [Phase 3: Temporal Interpolation](#4-phase-3-temporal-interpolation-monthly--daily)
5. [Phase 4: Feature Engineering](#5-phase-4-feature-engineering-21-dimensions)
6. [Phase 5: Model Training](#6-phase-5-model-training-xgboost)
7. [Phase 6: API Development](#7-phase-6-api-development-flask)
8. [Phase 7: Dashboard Development](#8-phase-7-dashboard-development)
9. [Complete Data Flow Diagram](#9-complete-data-flow-diagram)
10. [Reproducing the Results](#10-reproducing-the-results)

---

## 1. Project Overview

### Problem Statement
Ground-based PM2.5 monitoring stations are expensive (~$10,000 each) and unevenly distributed globally. Most developing countries — where pollution is worst — have minimal monitoring infrastructure. This creates a critical "data gap" in public health awareness.

### Our Approach
We bypass the need for ground stations entirely by using **satellite-derived atmospheric data** from NASA and ECMWF, available globally. We train machine learning models on 7 years of historical data to forecast PM2.5 levels 24, 48, and 72 hours into the future for **any coordinate on Earth**.

### Architecture Overview
```
Google Earth Engine (5 datasets)
        ↓
Data Pipeline (Python scripts)
        ↓
Feature Engineering (21 dimensions)
        ↓
XGBoost Training (3 models)
        ↓
Flask API (real-time inference)
        ↓
Interactive Dashboard (Leaflet.js + Chart.js)
```

---

## 2. Phase 1: Data Extraction (Google Earth Engine)

### What is Google Earth Engine?
Google Earth Engine (GEE) is a cloud-based platform that hosts **petabytes** of satellite imagery and geospatial datasets. We use its Python API to extract data without downloading raw satellite images.

### Grid Design
We defined a **2° × 2° global grid** spanning:
- **Latitude:** -60° to 72° (covering all inhabited landmasses)
- **Longitude:** -180° to 180° (full global coverage)
- **Result:** ~3,554 grid points after filtering to land areas
- **Time Range:** January 2015 to December 2021

Each grid point is centered at the midpoint of its 2° cell. The `reduceRegions` function in GEE averages satellite values within each grid cell.

### Dataset 1: PM2.5 (`dl_1_pm25.py`)
- **Source:** `projects/sat-io/open-datasets/GLOBAL-SATELLITE-PM25/MONTHLY`
- **Description:** CAMS (Copernicus Atmosphere Monitoring Service) Global Reanalysis. This is the most authoritative satellite-derived PM2.5 dataset available.
- **Resolution:** Monthly aggregates
- **Output:** `pm25_global.csv` with columns: `date, lat, lon, pm25`
- **How it works:**
  1. The script builds a grid of `ee.Feature` points.
  2. For each monthly image in the collection, it calls `reduceRegions` to extract the mean PM2.5 value at each grid point.
  3. Points are processed in chunks of 1,500 to avoid GEE's compute limits.

### Dataset 2: Weather (`dl_2_weather.py`)
- **Source:** `ECMWF/ERA5_LAND/MONTHLY_AGGR`
- **Description:** ERA5-Land is the ECMWF's state-of-the-art global reanalysis dataset. It provides hourly estimates of atmospheric variables, aggregated to monthly means.
- **Variables Extracted:**
  - `temperature_2m` — Air temperature at 2 meters (Kelvin)
  - `dewpoint_temperature_2m` — Dewpoint temperature (Kelvin)
  - `u_component_of_wind_10m` — Eastward wind component at 10m (m/s)
  - `v_component_of_wind_10m` — Northward wind component at 10m (m/s)
  - `surface_pressure` — Surface atmospheric pressure (Pa)
- **Output:** `weather_global.csv`

### Dataset 3: Aerosol Optical Depth (`dl_3_aod.py`)
- **Source:** `MODIS/061/MOD08_M3`
- **Description:** MODIS (Moderate Resolution Imaging Spectroradiometer) on NASA's Terra satellite measures Aerosol Optical Depth — a measure of how much sunlight is blocked by particles in the atmosphere. AOD is the **best satellite-based proxy for ground-level PM2.5**.
- **Variable:** `Aerosol_Optical_Depth_Land_Ocean_Mean_Mean`
- **Output:** `aod_global.csv` with columns: `date, lat, lon, aod`

### Dataset 4: Cloud Cover (`dl_4_cloud.py`)
- **Source:** `MODIS/061/MOD08_M3`
- **Description:** Cloud fraction affects both pollution dispersion and satellite measurement accuracy. Heavy cloud cover traps pollutants near the surface.
- **Variable:** `Cloud_Fraction_Mean_Mean`
- **Output:** `cloud_global.csv` with columns: `date, lat, lon, cloud_fraction`

### Dataset 5: Elevation (`dl_5_elevation.py`)
- **Source:** `USGS/SRTMGL1_003`
- **Description:** SRTM (Shuttle Radar Topography Mission) provides global Digital Elevation Models. Elevation affects atmospheric pressure and pollution dispersion — valleys trap pollutants, while high-altitude areas generally have cleaner air.
- **Note:** This is a **static** dataset (doesn't change over time), so it only needs to be downloaded once.
- **Output:** `elevation_global.csv` with columns: `lat, lon, elevation`

### Summary of Raw Data
| Dataset | Source | Rows | Time Dependency |
|---------|--------|------|-----------------|
| PM2.5 | CAMS Reanalysis | ~298K | Monthly |
| Weather | ERA5-Land | ~298K | Monthly |
| AOD | MODIS Terra | ~298K | Monthly |
| Cloud | MODIS Terra | ~298K | Monthly |
| Elevation | SRTM | ~3.5K | Static |

---

## 3. Phase 2: Data Merging & Cleaning

### Script: `merge_and_train.py`

This script combines all 5 raw datasets into a single unified DataFrame.

**Merge Strategy:**
```python
df = pm25.merge(weather, on=['date','lat','lon'], how='left')
df = df.merge(aod,     on=['date','lat','lon'], how='left')
df = df.merge(cloud,   on=['date','lat','lon'], how='left')
df = df.merge(elev,    on=['lat','lon'],         how='left')  # static, no date
```

- **Join keys:** `date + lat + lon` for temporal datasets; `lat + lon` for elevation.
- **Join type:** `LEFT JOIN` — we keep all PM2.5 rows even if a weather variable is missing (handled later by imputation or the model's native missing-value support).
- **Output:** `final_global_dataset.csv`

---

## 4. Phase 3: Temporal Interpolation (Monthly → Daily)

### Script: `step1_interpolate.py`

### Why?
Satellite data is aggregated monthly, but air quality changes daily. To train a model that predicts "tomorrow's" PM2.5, we need daily resolution.

### Method: Cubic Spline Interpolation
```python
from scipy.interpolate import CubicSpline

# Assign each monthly value to the 15th (midpoint of month)
# Fit a smooth cubic spline through these "knot" points
cs = CubicSpline(t_knots, pm25_knots, extrapolate=False)

# Evaluate the spline at every single day
pm25_daily = cs(t_daily)
pm25_daily = np.maximum(pm25_daily, 0)  # PM2.5 can't be negative
```

**Key Design Decisions:**
1. Each monthly value is placed at the **15th of the month** (the temporal midpoint).
2. A **Cubic Spline** (smooth polynomial curve) is fitted through these monthly "knot" points.
3. The spline is **evaluated at every day** between the first and last knot, producing smooth daily estimates.
4. **Negative clamping:** Spline overshoot can produce negative values. We clamp these to 0 because PM2.5 cannot be negative in reality.

**Output:** `pm25_daily.csv` — daily PM2.5 for all grid points.

---

## 5. Phase 4: Feature Engineering (21 Dimensions)

### Script: `step2_features.py`

This is the most critical step. We transform the raw data into a format that the machine learning model can learn from.

### 5.1 Temporal Features
```python
df['month']       = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin']     = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos']     = np.cos(2 * np.pi * df['day_of_year'] / 365)
```
**Why cyclical encoding?** December (12) and January (1) are actually adjacent months, but numerically they're far apart. Sine/cosine encoding preserves this circular relationship, allowing the model to understand seasonal cycles.

### 5.2 Weather Variables (Derived)
```python
# Temperature: Convert Kelvin → Celsius
df['temperature_celsius'] = df['temperature_2m'] - 273.15

# Relative Humidity: Magnus formula from dewpoint
df['relative_humidity'] = 100 * exp((17.625*d_c)/(243.04+d_c)) / exp((17.625*t_c)/(243.04+t_c))

# Wind Speed: Pythagorean from u,v components
df['wind_speed'] = sqrt(u² + v²)

# Wind Direction: Arctangent of v/u
df['wind_direction'] = degrees(arctan2(v, u))
```

### 5.3 Lag Features (Historical Memory)
```python
for lag in [1, 2, 3, 7]:
    df[f'pm25_lag_{lag}d'] = df.groupby(['lat','lon'])['pm25'].shift(lag)
```
**Why lags?** Air pollution has **temporal inertia**. Yesterday's PM2.5 is the strongest predictor of today's. We provide 4 lookback windows:
- **1d lag:** "What was yesterday's pollution?"
- **2d, 3d lags:** "Is pollution trending up or down?"
- **7d lag:** "What was pollution like same day last week?" (weekly industrial activity patterns)

### 5.4 Rolling Averages (Trend Detection)
```python
for window in [3, 7, 14]:
    df[f'pm25_roll_{window}d'] = df.groupby(['lat','lon'])['pm25'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean())
```
Rolling averages smooth out daily noise and capture underlying trends. The **14-day rolling average** acts as a "baseline pollution" indicator for each location.

### 5.5 Target Variables
```python
df['target_24h'] = df.groupby(['lat','lon'])['pm25'].shift(-1)   # next day
df['target_48h'] = df.groupby(['lat','lon'])['pm25'].shift(-2)   # 2 days ahead
df['target_72h'] = df.groupby(['lat','lon'])['pm25'].shift(-3)   # 3 days ahead
```
The **negative shift** creates "future" labels. For each row, `target_24h` is **tomorrow's PM2.5** at the same location.

### Complete Feature List (21 Dimensions)
| # | Feature | Source | Type |
|---|---------|--------|------|
| 1 | `lat` | Grid | Spatial |
| 2 | `lon` | Grid | Spatial |
| 3 | `month_sin` | Date | Cyclical Temporal |
| 4 | `month_cos` | Date | Cyclical Temporal |
| 5 | `day_sin` | Date | Cyclical Temporal |
| 6 | `day_cos` | Date | Cyclical Temporal |
| 7 | `temperature_celsius` | ERA5 | Weather |
| 8 | `relative_humidity` | ERA5 (derived) | Weather |
| 9 | `wind_speed` | ERA5 (derived) | Weather |
| 10 | `wind_direction` | ERA5 (derived) | Weather |
| 11 | `surface_pressure` | ERA5 | Weather |
| 12 | `aod` | MODIS | Atmospheric |
| 13 | `cloud_fraction` | MODIS | Atmospheric |
| 14 | `elevation` | SRTM | Topographic |
| 15 | `pm25_lag_1d` | PM2.5 | Historical Lag |
| 16 | `pm25_lag_2d` | PM2.5 | Historical Lag |
| 17 | `pm25_lag_3d` | PM2.5 | Historical Lag |
| 18 | `pm25_lag_7d` | PM2.5 | Historical Lag |
| 19 | `pm25_roll_3d` | PM2.5 | Rolling Average |
| 20 | `pm25_roll_7d` | PM2.5 | Rolling Average |
| 21 | `pm25_roll_14d` | PM2.5 | Rolling Average |

**Output:** `daily_features.csv` — the final training-ready dataset.

---

## 6. Phase 5: Model Training (XGBoost)

### Script: `step3_train_forecast.py`

### Why XGBoost?
- **Best for tabular data:** XGBoost consistently outperforms neural networks on structured/tabular datasets. Our data is a table of 21 features, not images or sequences.
- **Handles missing values natively:** Some AOD/cloud readings are missing due to satellite orbits. XGBoost learns the optimal split direction for missing values automatically.
- **Fast inference:** < 10ms per prediction — critical for real-time dashboards.
- **Interpretable:** Feature importance reveals which variables drive predictions.

### Training Configuration
```python
model = xgb.XGBRegressor(
    n_estimators=800,       # 800 boosting rounds
    learning_rate=0.05,     # Step size for each tree
    max_depth=7,            # Tree depth (controls complexity)
    subsample=0.8,          # Use 80% of data per tree (prevents overfitting)
    colsample_bytree=0.8,   # Use 80% of features per tree
    n_jobs=-1,              # Use all CPU cores
    random_state=42         # Reproducibility
)
```

### Time-Series Split (Walk-Forward Validation)
```python
train = df[df['year'] < 2021]   # 2015–2020: Training
test  = df[df['year'] == 2021]  # 2021: Testing
```
**Why not random split?** In time-series data, random splitting causes **data leakage** — the model would see future data during training. We use a strict temporal cutoff: train on 6 years, test on the 7th.

### Three Separate Models
We train **3 independent models**, each optimized for a different forecast horizon:

| Model | Target | Horizon | R² | RMSE (µg/m³) | MAE (µg/m³) |
|-------|--------|---------|-----|------|-----|
| `pm25_model_24h.json` | `target_24h` | Tomorrow | 0.979 | ~4.2 | ~2.8 |
| `pm25_model_48h.json` | `target_48h` | Day after tomorrow | 0.968 | ~5.1 | ~3.4 |
| `pm25_model_72h.json` | `target_72h` | 3 days ahead | 0.961 | ~5.8 | ~3.9 |

**Key observations:**
- R² decreases slightly as we predict further into the future (expected behavior).
- Even the 72h model explains 96.1% of variance — highly reliable.
- RMSE increases modestly (4.2 → 5.8 µg/m³) across horizons.

### Top Feature Importance (24h Model)
The model heavily relies on:
1. **pm25_lag_1d** — Yesterday's PM2.5 (strongest predictor)
2. **pm25_roll_3d** — 3-day rolling average
3. **pm25_lag_2d** — PM2.5 from 2 days ago
4. **aod** — Aerosol Optical Depth (satellite-based pollution proxy)
5. **temperature_celsius** — Temperature affects atmospheric stability

This confirms our hypothesis: PM2.5 is primarily driven by **recent history** (lags) and **atmospheric conditions** (AOD, temperature).

### Model Export
Models are saved as `.json` files using XGBoost's native format:
```python
model.save_model('pm25_model_24h.json')
```
This allows instant loading in the production API without scikit-learn overhead.

---

## 7. Phase 6: API Development (Flask)

### Script: `dashboard/api.py`

### Endpoints

#### `POST /predict`
The core prediction endpoint. Accepts a JSON payload with location, date, and weather data:
```json
{
  "lat": 28.6, "lon": 77.2,
  "month": 12, "day_of_year": 345,
  "pm_today": 190, "pm_1d": 178, "pm_2d": 162,
  "pm_3d": 145, "pm_7d": 120,
  "temp_c": 12, "humidity": 75, "wind_speed": 2, "aod": 0.45
}
```
- Constructs a 21-feature vector from the input.
- Runs inference through all 3 XGBoost models.
- Returns predictions for 24h, 48h, and 72h.

#### `GET /snapshot?year=2021&month=12`
Returns PM2.5 data for a specific month from the preloaded dataset (298K rows loaded into RAM at startup). This powers the time slider on the dashboard.

#### `GET /months`
Returns the list of all available `[year, month]` pairs for the slider.

#### `POST /evaluate`
Batch prediction endpoint for the Judge Evaluation Console. Accepts an array of rows, runs predictions, and computes R², RMSE, MAE metrics if actual values are provided.

#### `GET /health`
Health check endpoint showing model status and available months.

---

## 8. Phase 7: Dashboard Development

### Interactive Map (`dashboard.html`)
- **Leaflet.js** renders ~3,500 colored dots on a dark-themed world map.
- **Color coding:** Green (safe) → Yellow → Red (hazardous) based on PM2.5 levels.
- **Pulsing dots:** Locations with PM2.5 ≥ 150 µg/m³ show a red pulsing animation.
- **Click interaction:** Clicking a dot shows a 12-month PM2.5 chart and populates the forecast form.

### Time Slider
- Allows scrubbing through **84 months** (Jan 2015 – Dec 2021).
- **Animate button** plays through months automatically.
- API fetches data on-demand for each month.

### AI Forecast Panel
- **City Search:** Photon geocoder (photon.komoot.io) for browser-safe location lookup.
- **3-Step Flow:** Satellite data → XGBoost → Forecast results.
- **Health Alerts:** Automatic banners when PM2.5 exceeds WHO guidelines (≥ 25 µg/m³).
- **Lung Equivalents:** Translates PM2.5 into "cigarettes per day" (1 cig ≈ 22 µg/m³ for 24h).

### Technical Specs Modal
- **⚙️ Specs button** in header opens a glassmorphic modal showing model architecture, features, and accuracy.

### Judge Evaluation Page (`evaluate.html`)
- Manual row entry or CSV upload.
- Color-coded results table (predicted vs actual).
- Live accuracy metrics: R², RMSE, MAE.

---

## 9. Complete Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    GOOGLE EARTH ENGINE                        │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ ┌───────┐ ┌──────────┐ │
│  │ CAMS     │ │ ERA5     │ │MODIS │ │MODIS  │ │ SRTM     │ │
│  │ PM2.5    │ │ Weather  │ │ AOD  │ │Cloud  │ │Elevation │ │
│  └────┬─────┘ └────┬─────┘ └──┬───┘ └───┬───┘ └────┬─────┘ │
└───────┼────────────┼──────────┼─────────┼──────────┼────────┘
        │            │          │         │          │
        v            v          v         v          v
  dl_1_pm25.py  dl_2_weather  dl_3_aod  dl_4_cloud  dl_5_elevation
        │            │          │         │          │
        v            v          v         v          v
    pm25_global  weather_     aod_      cloud_   elevation_
      .csv        global.csv  global    global     global.csv
        │            │          │         │          │
        └────────────┴──────────┴─────────┴──────────┘
                              │
                    merge_and_train.py
                              │
                              v
                   final_global_dataset.csv
                              │
                   step1_interpolate.py
                              │
                              v
                      pm25_daily.csv
                              │
                    step2_features.py
                              │
                              v
                    daily_features.csv  (21 features + 3 targets)
                              │
                   step3_train_forecast.py
                              │
                  ┌───────────┼───────────┐
                  v           v           v
          model_24h.json  model_48h.json  model_72h.json
                  │           │           │
                  └───────────┼───────────┘
                              │
                          api.py (Flask)
                              │
                    ┌─────────┼─────────┐
                    v                   v
             dashboard.html      evaluate.html
              (Leaflet.js)        (Judge Console)
```

---

## 10. Reproducing the Results

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step-by-Step Execution
```bash
# 1. Extract data from Google Earth Engine (requires GEE authentication)
cd data_pipeline
python3 dl_1_pm25.py
python3 dl_2_weather.py
python3 dl_3_aod.py
python3 dl_4_cloud.py
python3 dl_5_elevation.py

# 2. Merge all datasets
python3 merge_and_train.py

# 3. Interpolate to daily resolution
python3 step1_interpolate.py

# 4. Engineer features
python3 step2_features.py

# 5. Train models
python3 step3_train_forecast.py

# 6. Launch the dashboard
cd ../dashboard
bash start_demo.sh
```

### Expected Output
- 3 model files: `pm25_model_24h.json`, `pm25_model_48h.json`, `pm25_model_72h.json`
- Accuracy chart: `forecast_accuracy.png`
- Feature importance chart: `feature_importance_daily.png`
- Dashboard available at: `http://localhost:8502/dashboard.html`

---

*This document was prepared for the India Innovates 2026 Hackathon. All data sources are publicly available and freely accessible.*
