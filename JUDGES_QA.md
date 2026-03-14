# 👨‍⚖️ Judges Q&A: AirSight AI

This document prepares the team for common technical and impact-related questions from hackathon judges.

---

### 1. General Product Questions

#### Q1: What problem are you solving?
**A:** We are solving the "Data Gap" in air quality monitoring. High-grade PM2.5 sensors cost thousands of dollars, leaving most of the world unmonitored. We use global satellite snapshots to provide a 7-year history and 72-hour future forecasts for any coordinate on Earth, even where sensors don't exist.

#### Q2: How is this different from existing weather apps?
**A:** Most apps only show *current* data from local stations. We provide a **Global Grid** of 3,554 points. Unlike apps that "guess" values for rural areas, our model uses 21 secondary features (AOD, Wind, Humidity) to provide a physically-grounded estimate. Plus, we translate data into human impact (Lung Equivalents).

---

### 2. Data & Feature Engineering

#### Q3: Where did the data come from?
**A:** We extracted two main datasets via **Google Earth Engine (GEE)**:
1. **PM2.5:** CAMS (Copernicus Atmosphere Monitoring Service) Global Reanalysis.
2. **Weather:** ERA5-Land Hourly data (Temperature, Wind, Humidity, AOD).
Data spans **2015 to 2021** across a 2°x2° global grid.

#### Q4: Why did you use Lag features?
**A:** Air pollution has **temporal inertia**. What happened yesterday (t-1) and three days ago (t-3) are the strongest predictors of today. We engineered 4 lag features (1d, 2d, 3d, 7d) to capture these short-term and weekly cycles.

---

### 3. Machine Learning & Accuracy

#### Q5: Why XGBoost instead of a Neural Network (LSTM/RNN)?
**A:** For tabular time-series data with high dimensionality, **XGBoost** often outperforms LSTMs while being 100x faster to train and deploy. It handles non-linear relationships between weather (e.g., wind speed) and pollution levels more effectively with less data.

#### Q6: How do you verify your R² of 0.979?
**A:** We used a standard **Time-Series Split** (Walk-forward validation). We trained on data from 2015–2020 and tested on the "unseen" year of 2021. This ensures our accuracy isn't due to simple memorization but true predictive power.

#### Q7: What is the "Winning Edge" of your model?
**A:** Our model incorporates **Aerosol Optical Depth (AOD)**. AOD is the best satellite-based proxy for ground-level PM2.5. By combining AOD with local meteorological lags, we capture pollution "build-up" events (like smog traps in Delhi) that purely weather-based models miss.

---

### 4. Technical Architecture

#### Q8: How handles the "Big Data" of 298k rows on a frontend?
**A:** We don't send all data to the browser. The **Flask API** performs regional filtering and monthly snapshots. The frontend (Leaflet.js) only renders the active month's grid, keeping the UI smooth even on lower-end devices.

#### Q9: Can this scale?
**A:** Absolutely. The current grid is 2° resolution (~200km). We can swap the data pipeline to 0.1° resolution (~10km) easily. The XGBoost architecture remains the same; only the training volume increases.
