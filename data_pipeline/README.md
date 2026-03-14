# 🚜 Data Pipeline: From Satellite to CSV

This directory contains the scripts responsible for extracting, cleaning, and preparing the global PM2.5 and meteorological dataset.

---

## 1. Extraction (Google Earth Engine)
We use the **Google Earth Engine (GEE) Python API** to extract monthly data from 2015 to 2021.
- **PM2.5 (`dl_1_pm25.py`):** Uses the CAMS (Copernicus Atmosphere Monitoring Service) Global Reanalysis dataset.
- **Weather (`dl_2_weather.py`):** Extracts Temperature, Humidity, and Wind Speed from the ERA5-Land Hourly dataset.
- **AOD (`dl_3_aod.py`):** Extracts Aerosol Optical Depth (AOD) from MODIS (MOD08_M3), a key proxy for PM2.5.
- **Grid:** Data is sampled across a global 2° x 2° grid (approx. 3,554 points covering landmasses).

## 2. Interpolation (`step1_interpolate.py`)
Satellite data often has gaps due to cloud cover or sensor orbits.
- **RBF Interpolation:** We use Radial Basis Function (RBF) interpolation from `scipy` to fill missing values across the global grid.
- **Ensuring Continuity:** This step ensures that every grid point has a complete continuous history for the 84-month period (7 years).

## 3. Feature Preparation (`step2_features.py`)
Before training, we transform the raw data into a supervised learning format.
- **Lag Generation:** We create "Time-Lags" for PM2.5 (1d, 2d, 3d, 7d). This allows the model to learn historical patterns.
- **Temporal Normalization:** Months and Days are converted into cyclical sine/cosine features (optional) or treated as discrete features to capture seasonal peaks.
- **Merging:** Weather variables are merged with PM2.5 lags based on coordinate and timestamp.

## 4. Final Output
The pipeline produces **`dashboard_data.json`** for the map and **`final_land_dataset.csv`** for model training.
- **Rows:** ~298,536
- **Attributes:** Lat, Lon, Year, Month, Day, PM2.5, Lags (1-7), Temp, Humidity, Wind, AOD.
