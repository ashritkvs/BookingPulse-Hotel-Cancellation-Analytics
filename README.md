# 🏨 BookingPulse — Hotel Cancellation Intelligence Dashboard

**Built by:** Venkata Sai Ashrit Kommireddy  
**MS Data Science · Stony Brook University**

---

## What This Is

An end-to-end Streamlit dashboard for hotel booking cancellation prediction and revenue optimization. Upload the Hotel Booking Demand dataset and get instant EDA, a trained XGBoost model, SHAP explainability, and a live risk scorer — all in one place.

---

## Features

| Tab | What it shows |
|-----|--------------|
| 📊 EDA | Cancellation rates by hotel type, lead time distributions, country-level breakdown, monthly trends |
| 🤖 Model Performance | Confusion matrix, classification report, ROC-AUC, XGBoost feature importances |
| 🔍 SHAP Explainability | SHAP beeswarm plot with interpretation guide |
| 🎯 Risk Scorer | Enter any booking's details → get live cancellation probability + revenue team recommendation |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Upload the dataset
- Download `hotel_bookings.csv` from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Upload it via the sidebar file uploader

---

## Dataset

**Hotel Booking Demand** — 119,390 bookings across City Hotel and Resort Hotel, 2015–2017.  
Source: Antonio, Almeida & Nunes (2019), published on Kaggle.

---

## Tech Stack

- **Streamlit** — dashboard framework  
- **XGBoost** — cancellation prediction model  
- **SHAP** — model explainability  
- **Scikit-learn** — train/test split, metrics  
- **Pandas / NumPy** — data processing  
- **Matplotlib / Seaborn** — visualizations  
