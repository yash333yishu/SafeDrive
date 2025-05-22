# 🚘 SafeDrive – Intelligent Road Safety Recommender

**SafeDrive** is a Flask-based web application that predicts road safety levels based on real-time weather, traffic, driver condition, and location. It integrates machine learning with live APIs to recommend whether driving is safe, risky, or dangerous.

---

## 🌟 Features

- 🔐 Secure user login system
- 📍 Address autocomplete & current location detection
- 🚦 Real-time traffic & weather data integration (OpenWeatherMap, Google Maps)
- 🧠 Machine Learning prediction (Random Forest model)
- 📊 Visualizations: safety trends, confusion matrix, feature correlation
- 🧭 Route rendering with live traffic overlays
- 🧾 Emergency contacts, SOS button, driver health input
- 🧪 Model comparison with Logistic Regression, SVM
- 💬 Tailored recommendations for every risk level

---

## 🧠 Machine Learning

- Algorithm: `RandomForestClassifier` (best performing model)
- Features used:
  - Temperature
  - Visibility (simulated based on weather)
  - Traffic density (time-based logic)
  - Vehicle speed
  - Risk zone (based on GPS)
- Labels:
  - `0`: Safe
  - `1`: Caution
  - `2`: Danger

---

## 🔧 Tech Stack

| Layer         | Tools / Frameworks                      |
|---------------|------------------------------------------|
| Frontend      | HTML, TailwindCSS, JS, Google Maps API  |
| Backend       | Python Flask                            |
| Machine Learning | scikit-learn, pandas, joblib         |
| Data Storage  | SQLite (`road_safety.db`)               |
| API Integration | OpenWeatherMap, Google Maps API       |
| Hosting       | Render (Flask support)                  |

---

