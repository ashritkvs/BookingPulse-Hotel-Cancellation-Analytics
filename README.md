# 📊 BookingPulse: Hotel Booking Cancellation & Revenue Optimization

A complete end-to-end data analytics project that explores, predicts, and explains hotel booking cancellations. This project delivers actionable insights for hotel revenue teams to minimize lost revenue and optimize booking strategies.

---

## 📁 Dataset

- Source: [Hotel Booking Demand – Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- File: `hotel_bookings.csv`

---

## 🧠 Problem Statement

Hotel booking cancellations significantly impact revenue. The goal of this project is to:

- Predict which bookings are likely to cancel
- Identify factors influencing cancellations
- Provide actionable business recommendations

---

## 🔧 Tools & Technologies

| Category | Tools Used |
|----------|------------|
| Language | Python |
| Data Viz | Matplotlib, Seaborn, SHAP |
| Modeling | XGBoost |
| Evaluation | Confusion Matrix, Classification Report, ROC AUC |
| Platform | Google Colab |
| Version Control | Git, GitHub |

---

## 📊 Key Analyses

- Cancellation rate by hotel type, country, and season
- Lead time, deposit types, and booking channels analysis
- Total guest analysis and ADR impact on cancellations

---

## 📈 Machine Learning Pipeline

- **Model:** XGBoost Classifier  
- **Features:** Lead time, total guests, booking changes, ADR, etc.  
- **Explainability:** SHAP values to interpret feature importance  
- **Evaluation Metrics:** Accuracy, Precision, Recall, ROC-AUC

---

## 💡 Business Insights

- 📌 Longer lead time → Higher cancellation probability  
- 📌 City hotels show higher cancellations than resort hotels  
- 📌 Repeat guests are highly loyal and less likely to cancel  
- 📌 Higher ADR bookings are more prone to cancellation  
- 📌 Booking changes and deposit type significantly affect outcomes

---

## 📁 Project Structure

BookingPulse/
├── data/
│ └── hotel_bookings.csv
├── BookingPulse_Hotel_Booking_Cancellation_&_Revenue_Optimization.ipynb
├── README.md



---

## 📌 Future Enhancements

- Deploy as a Streamlit app for live cancellation predictions
- Add dashboard (Power BI / Tableau)
- Include advanced revenue simulations and pricing policies

---

