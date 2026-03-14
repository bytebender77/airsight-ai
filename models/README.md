# 🧠 Model Architecture: XGBoost Forecast Engine

AirSight AI uses multiple **Extreme Gradient Boosting (XGBoost)** regression models to provide highly accurate air quality forecasts.

---

## 1. Why XGBoost?
We chose XGBoost over traditional RNN/LSTMs because:
- **Performance:** Superior results on tabular time-series features (Lags + Weather).
- **Speed:** Inference happens in under 10ms, allowing real-time forecasts on our dashboard.
- **Transparency:** Feature importance can be easily analyzed to explain "why" a forecast is high.

## 2. Feature Engineering (21 Dimensions)
The model consumes 21 input features to predict future PM2.5:
- **Historical Lags (4):** PM2.5 readings from 1d, 2d, 3d, and 7d ago.
- **Meteorological (5):** Temperature (2m), Humidity, Wind Speed (u/v components), and Aerosol Optical Depth (AOD).
- **Temporal (3):** Month, Day of Year, Season.
- **Spatial (2):** Latitude, Longitude (enabling the model to learn location-specific baseline pollution).
- **Metadata (7):** Derived rolling means and standard deviations of the lags.

## 3. Training Strategy
- **Framework:** `xgboost` v2.0.0
- **Validation:** 5-Fold Time-Series Split (Walk-Forward).
- **Optimization:** Hyperparameter tuning of `max_depth` (default 6) and `n_estimators` (default 1000) using Early Stopping to prevent overfitting.
- **Multi-Horizon:** We maintain 3 separate models optimized for:
  - **T+24h** (Next Day Forecast)
  - **T+48h** (2-Day Forecast)
  - **T+72h** (3-Day Forecast)

## 4. Evaluation Metrics
Models are evaluated using three key metrics:
- **R² Score:** 0.979 (Average) - Explains 97.9% of the variation in pollution levels.
- **RMSE:** ~4.2 - Root Mean Square Error in µg/m³.
- **MAE:** ~2.8 - Mean Absolute Error in µg/m³.

---

## 5. Model Persistence
Models are exported as **`.json`** files (`pm25_model_24h.json`, etc.) for fast loading into the Flask production API.
