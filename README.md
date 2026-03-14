# 📡 AirSight AI: Global PM2.5 Forecasting Platform

> **Winner Prospect @ India Innovates 2026**  
> *Harnessing XGBoost and Monthly Satellite Snapshots to Predict Global Air Quality.*

![Version](https://img.shields.io/badge/Version-1.0.0-blue)
![Tech](https://img.shields.io/badge/Tech-XGBoost%20%7C%20Flask%20%7C%20Leaflet-green)
![Accuracy](https://img.shields.io/badge/Accuracy-R%C2%B2%200.979-orange)

---

## 🌟 The Vision
Air pollution is the "silent killer," yet real-time data is often missing in developing regions. **AirSight AI** bridges this gap by using historical satellite data (2015–2021) and advanced Machine Learning to provide:
1. **Historical Visibility:** A 7-year global timeline of PM2.5 trends.
2. **Future Foresight:** Precise 24h, 48h, and 72h predictions using local history and weather.
3. **Human Impact:** Translating complex PM2.5 numbers into "Cigarette Equivalents" for better public understanding.

---

## 🛠️ Technical Architecture

### 🧠 The Brain: XGBoost ML Engine
We trained three specialized XGBoost models (one for each forecast horizon) on 298,000+ data points across 3,554 global grid points.
- **Features (21 total):**
  - **Lags:** PM2.5 readings from 1d, 2d, 3d, and 7d ago.
  - **Temporal:** Month, Day of Year.
  - **Weather:** Temperature, Humidity, Wind Speed, Aerosol Optical Depth (AOD).
  - **Spatial:** Lat/Lon specific baseline pollution.
- **Performance:** 
  - **R² Score:** 0.979 (24h), 0.968 (48h), 0.961 (72h).
  - **Latency:** <10ms per prediction.

### 🌐 The Dashboard: Glassmorphism UI
- **Leaflet.js:** Custom dark-themed world map with pulsing "Hazardous" hotspots.
- **Chart.js:** Interactive time-series analysis for any clicked point.
- **Nominatim/Photon:** Real-time city geocoding.

---

## 🧪 Judge Evaluation Console
We built a dedicated **Evaluation Page** for judges to verify our claims. 
- Upload a CSV of unseen data.
- Run batch predictions.
- **Live Accuracy Check:** The tool automatically calculates R², RMSE, and MAE against provided actuals.

---

## 🚀 Quick Start (Demo Mode)

### 1. Prerequisites
```bash
pip install flask flask-cors pandas xgboost numpy
```

### 2. Launching the Demo
We provide a unified startup script to handle the API and Frontend:
```bash
cd dashboard
bash start_demo.sh
```
- **Dashboard:** `http://localhost:8502/dashboard.html`
- **Evaluation:** `http://localhost:8502/evaluate.html`

### 🔧 For Remote Demo (ngrok)
If you need to share the link with judges remotely:
1. `ngrok http 5050`
2. `bash start_demo.sh https://your-url.ngrok-free.app`

---

## 👥 Credits
Developed with ❤️ for **India Innovates 2026**.

---

## 📄 License
MIT License - Project developed for Hackathon purposes.
