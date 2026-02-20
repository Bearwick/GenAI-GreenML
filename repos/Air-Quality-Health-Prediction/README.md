# 🏥 Air Quality Health Prediction

This project aims to predict **hospital visits due to air pollution** using regression models and classify whether a day is **high-risk or low-risk** using classification models, based on air quality features from the `AirQualityUCI.csv` dataset.

## 🧪 Features Used

- CO(GT) — Carbon Monoxide
- NOx(GT) — Nitric Oxides
- NO2(GT) — Nitrogen Dioxide
- C6H6(GT) — Benzene
- T — Temperature (°C)
- RH — Relative Humidity (%)

---

## 🧼 Preprocessing

- Combined and parsed `Date` and `Time` columns into a `Datetime` index.
- Dropped irrelevant columns.
- Converted pollutant values to numeric types.
- Resampled data to daily averages.
- Created a synthetic target variable:
- `Hospital_Visits` for regression.
- `Risk_Label` (high-risk if visits > median) for classification.

---

## 📈 Models Used

### Regression Model:
- **RandomForestRegressor**
- Predicts number of hospital visits.
- Performance metrics:
- **R² Score**
- **RMSE (Root Mean Squared Error)**

### Classification Model:
- **RandomForestClassifier**
- Predicts whether the day is high-risk (1) or low-risk (0).
- Performance metrics:
- **Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC Curve**

---

## 📊 Visualizations

- 📌 **Correlation Heatmap** — shows relationships between features and the target.
- 📌 **Regression Plot** — actual vs. predicted hospital visits.
- 📌 **Confusion Matrix** — classification performance on 0 vs. 1.
- 📌 **Feature Importance** — shows how much each pollutant contributes to predictions.

---

## ▶️ Running the Project

1. Make sure you have Python 3.9+ with required libraries.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt