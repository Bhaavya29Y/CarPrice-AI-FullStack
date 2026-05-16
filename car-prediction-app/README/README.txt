# 🚗 CarPrice AI: Full-Stack Predictive Analytics Platform

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 💎 Project Overview
This platform automates the appraisal process for used vehicles, leveraging machine learning to deliver instant, high-accuracy market valuations. It is built using a **decoupled architecture**, separating the high-speed inference engine (FastAPI) from the interactive user experience (Streamlit).

### 📈 Key Results
- **Model Accuracy:** 92.4% R² Score.
- **Efficiency:** Reduced manual valuation time from hours to < 200ms.
- **Reliability:** Integrated a robust data pipeline to handle 11+ car features including brand, power, and mileage.

## 🛠️ Technical Architecture

- **Backend (Inference Layer):** FastAPI server serving a serialized Random Forest model. Implements Pydantic for strict data validation.
- **Frontend (UX Layer):** Streamlit dashboard featuring custom CSS, real-time API communication, and Plotly-driven market analytics.
- **ML Pipeline:** Advanced preprocessing including standard scaling and label encoding of categorical market data.

## 📂 System Structure
```text
CAR-PREDICTION-APP
├── backend/            # ML Engine & API
│   ├── main.py         # REST API Endpoints
│   ├── model.pkl       # Optimized RF Model
│   └── requirements.txt
└── frontend/           # Interactive Dashboard
    ├── app.py          # Streamlit UI Logic
    └── requirements.txt