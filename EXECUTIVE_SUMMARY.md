# 📋 Executive Summary: AirSight AI

## The Problem
Air pollution kills **7 million people annually** (WHO). Yet most of the world lacks real-time PM2.5 monitoring — ground stations are expensive ($10,000+) and unevenly distributed. Developing nations, where pollution is highest, have the fewest sensors.

## Our Solution
**AirSight AI** is a global PM2.5 forecasting platform that uses **satellite data** (available everywhere) combined with **XGBoost machine learning** to predict air quality **24, 48, and 72 hours** into the future for any coordinate on Earth.

## Key Metrics
| Metric | Value |
|--------|-------|
| **Model Accuracy (R²)** | 0.979 (24h), 0.968 (48h), 0.961 (72h) |
| **Spatial Coverage** | 3,554 grid points across all landmasses |
| **Temporal Range** | 7 years (2015–2021), 84 months |
| **Data Points Processed** | 298,536 rows |
| **Data Sources** | 5 satellite datasets via Google Earth Engine |
| **Inference Speed** | < 10ms per prediction |
| **Features Engineered** | 21 input dimensions |

## How It Works (End-to-End)
```
Satellite Data (GEE) → Data Pipeline → Feature Engineering → XGBoost Training → Flask API → Interactive Dashboard
```

1. **Data Extraction:** We extract PM2.5, Weather (ERA5), Aerosol Optical Depth (MODIS), Cloud Cover, and Elevation data from Google Earth Engine across a 2°×2° global grid.
2. **Interpolation:** Monthly satellite snapshots are interpolated to daily resolution using Cubic Spline fitting.
3. **Feature Engineering:** We create 21 input features including temporal lags (1d, 2d, 3d, 7d), rolling averages (3d, 7d, 14d), cyclical time encodings, and derived weather metrics (wind speed, humidity from dewpoint).
4. **Model Training:** Three separate XGBoost models are trained for 24h, 48h, and 72h horizons using a Time-Series Split (train: 2015–2020, test: 2021).
5. **Serving:** A Flask API loads the 3 models and provides real-time predictions. A Leaflet.js dashboard visualizes the global data with time-slider browsing and on-demand forecasting.

## Impact & Innovation
- **No Ground Stations Needed:** Works anywhere satellites can see — rural villages, oceans, conflict zones.
- **Human-Readable Output:** PM2.5 levels translated into "Cigarette Equivalents" (1 cig ≈ 22 µg/m³ for 24h exposure).
- **Health Alert System:** Automatic warnings when forecasts exceed WHO guidelines.
- **Judge Verification:** A dedicated evaluation console lets judges test the model with their own data.

## Technology Stack
- **ML:** XGBoost 2.0, scikit-learn, NumPy, Pandas
- **Data:** Google Earth Engine (CAMS, ERA5, MODIS, SRTM)
- **Backend:** Flask + Gunicorn
- **Frontend:** Leaflet.js, Chart.js, Glassmorphism CSS
- **Deployment:** GitHub Pages (frontend) + Render.com (API)

---

*Built for India Innovates 2026 Hackathon*
