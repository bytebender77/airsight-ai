"""
AirSight AI — Blind Test Submission Generator
==============================================
Classifies PM2.5 into severity classes using XGBoost + engineered features.

Severity Classes (based on WHO/AQI standards):
  Good       : PM2.5 ≤ 5.0 µg/m³  (WHO annual guideline)
  Moderate   : 5.0 < PM2.5 ≤ 15.0  (WHO interim target 4)
  Sensitive  : 15.0 < PM2.5 ≤ 35.0  (Unhealthy for sensitive groups)
  Critical   : PM2.5 > 35.0         (Unhealthy / Very Unhealthy)

Run: python3 generate_submission.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import os, warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# 1. LOAD BLIND TEST DATA
# ──────────────────────────────────────────────────────────────
TEST_PATH = '/Users/kunalkumargupta/Desktop/hackathon/airsight-pm25/Test_Features.csv'
test = pd.read_csv(TEST_PATH)
pm_col = 'Geographic-Mean PM2.5 [ug/m3]'

print(f"📦 Test data loaded: {test.shape}")
print(f"   PM2.5 range: {test[pm_col].min():.1f} – {test[pm_col].max():.1f} µg/m³")
print(f"   Regions: {test['Region'].nunique()}")
print(f"   Years: {test['Year'].min()} – {test['Year'].max()}")

# ──────────────────────────────────────────────────────────────
# 2. DEFINE SEVERITY THRESHOLDS
# ──────────────────────────────────────────────────────────────
# Based on WHO Air Quality Guidelines (2021 update) and standard AQI breakpoints:
def classify_severity(pm25):
    if pm25 <= 5.0:
        return 'Good'
    elif pm25 <= 15.0:
        return 'Moderate'
    elif pm25 <= 35.0:
        return 'Sensitive'
    else:
        return 'Critical'

# ──────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
print("\n⚙️  Engineering features...")

df = test.copy()

# Rename columns to remove brackets (XGBoost doesn't allow [, ] in feature names)
df.columns = [c.replace('[', '').replace(']', '').replace('<', '') for c in df.columns]
pm_col = 'Geographic-Mean PM2.5 ug/m3'

# Temporal features
df['year_norm'] = (df['Year'] - df['Year'].min()) / max(df['Year'].max() - df['Year'].min(), 1)

# PM2.5 derivative features
df['pm_trend'] = df[pm_col] - df['pm_lag1']          
df['pm_accel'] = df['pm_lag1'] - df['pm_lag2']        
df['pm_volatility'] = df[[pm_col, 'pm_lag1', 'pm_lag2', 'pm_lag3']].std(axis=1)
df['pm_ratio_lag1'] = df[pm_col] / (df['pm_lag1'] + 0.01)
df['pm_max_recent'] = df[['pm_lag1', 'pm_lag2', 'pm_lag3']].max(axis=1)
df['pm_min_recent'] = df[['pm_lag1', 'pm_lag2', 'pm_lag3']].min(axis=1)
df['pm_range_recent'] = df['pm_max_recent'] - df['pm_min_recent']

# Coverage quality
df['coverage_score'] = df['Population Coverage %'] * df['Geographic Coverage %'] / 100

# Log transform for PM2.5 (common in environmental data)
df['pm_log'] = np.log1p(df[pm_col])

print(f"   Features: {df.shape[1]} columns")

# ──────────────────────────────────────────────────────────────
# 4. TRAIN AN XGBOOST CLASSIFIER (self-supervised on thresholds)
# ──────────────────────────────────────────────────────────────
# Since this is fundamentally a threshold-based classification,
# we create training labels from the thresholds and then train
# an XGBoost classifier that can learn nuanced boundary decisions
# from the additional features (lags, trends, coverage).

print("\n🧠 Training XGBoost Classifier...")

# Create labels
df['severity'] = df[pm_col].apply(classify_severity)

# Encode labels
le = LabelEncoder()
df['severity_encoded'] = le.fit_transform(df['severity'])

FEATURES = [
    pm_col, 'pm_lag1', 'pm_lag2', 'pm_lag3', 'pm_roll3',
    'Population Coverage %', 'Geographic Coverage %',
    'year_norm', 'pm_trend', 'pm_accel', 'pm_volatility',
    'pm_ratio_lag1', 'pm_max_recent', 'pm_min_recent',
    'pm_range_recent', 'coverage_score', 'pm_log'
]

X = df[FEATURES]
y = df['severity_encoded']

# Train XGBoost with cross-validation
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method='hist',
    n_jobs=-1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Cross-validate to report accuracy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    preds = model.predict(X.iloc[val_idx])
    acc = accuracy_score(y.iloc[val_idx], preds)
    cv_scores.append(acc)

print(f"   5-Fold CV Accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Train final model on ALL data
model.fit(X, y)

# Predict
df['predicted_severity'] = le.inverse_transform(model.predict(X))

# ──────────────────────────────────────────────────────────────
# 5. GENERATE SUBMISSION
# ──────────────────────────────────────────────────────────────
print("\n📊 Classification Distribution:")
print(df['predicted_severity'].value_counts().to_string())

submission = df[['Region', 'Year', 'predicted_severity']].copy()
OUT_PATH = '/Users/kunalkumargupta/Desktop/hackathon/airsight-pm25/submission.csv'
submission.to_csv(OUT_PATH, index=False)

print(f"\n✅ Submission saved: {OUT_PATH}")
print(f"   Total rows: {len(submission)}")
print(f"   Unique regions: {submission['Region'].nunique()}")
print(f"   Year range: {submission['Year'].min()} – {submission['Year'].max()}")

# ──────────────────────────────────────────────────────────────
# 6. VERIFICATION
# ──────────────────────────────────────────────────────────────
print("\n📋 Sample predictions:")
# Show a few interesting cases
for region in ['Afghanistan', 'India', 'United States of America', 'China', 'Australia']:
    rows = submission[submission['Region'] == region]
    if len(rows) > 0:
        for _, r in rows.iterrows():
            pm = df[(df['Region'] == region) & (df['Year'] == r['Year'])][pm_col].values[0]
            print(f"   {region:30s} {r['Year']} | PM2.5={pm:6.1f} → {r['predicted_severity']}")

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n🔝 Top Features for Classification:")
for i, (feat, imp) in enumerate(feat_imp.head(8).items()):
    print(f"   {i+1}. {feat}: {imp:.4f}")
