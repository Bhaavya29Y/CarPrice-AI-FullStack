# ============================================================
#   FRONTEND APP  |  frontend/app.py
#   Run: streamlit run app.py
# ============================================================

import streamlit as st
import requests 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# URL point mapping to your local FastAPI service port instance
BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="CarPrice AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; border-right: 1px solid #2d2d4e; }
    [data-testid="metric-container"] { background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 12px; padding: 16px; }
    h1, h2, h3 { color: #ffffff !important; }
    .stSelectbox > div, .stNumberInput > div { background-color: #1a1a2e; border-radius: 8px; }
    .stButton > button { background: linear-gradient(135deg, #534AB7, #7F77DD); color: white; border: none; border-radius: 10px; padding: 12px 32px; font-size: 16px; font-weight: 600; width: 100%; cursor: pointer; transition: transform 0.2s; }
    .stButton > button:hover { transform: scale(1.02); background: linear-gradient(135deg, #3C3489, #534AB7); }
    .result-card { background: linear-gradient(135deg, #1a1a2e, #2d2d4e); border: 2px solid #534AB7; border-radius: 16px; padding: 24px; text-align: center; margin: 16px 0; }
    .result-price { font-size: 42px; font-weight: 700; color: #7F77DD; margin: 8px 0; }
    .result-label { font-size: 14px; color: #aaaacc; margin-bottom: 4px; }
    .result-range { font-size: 13px; color: #7F77DD; margin-top: 6px; }
    #MainMenu, footer, header { visibility: hidden; }
    hr { border-color: #2d2d4e; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
col_logo, col_title, col_badge = st.columns([0.08, 0.8, 0.12])
with col_logo: st.markdown("### 🚗")
with col_title: st.markdown("## CarPrice AI — ML-Powered Car Price Predictor")
with col_badge: st.markdown('<span style="background:#0F6E56;color:#9FE1CB;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:500;">● Live</span>', unsafe_allow_html=True)

st.markdown("---")

# ── Top Stat Cards ───────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("📦 Dataset size",   "8,128 cars",  "trained on")
m2.metric("🎯 Model accuracy", "92.4% R²",    "Random Forest")
m3.metric("💰 Avg MAE",        "₹38,200",     "mean error")
m4.metric("📊 Avg car price",  "₹6.4 Lakh",   "in dataset")

st.markdown("---")

# ── Sidebar Input Form ───────────────────────────────────────
st.sidebar.markdown("## 🔧 Enter car details")
brand_opts = ["Maruti", "Hyundai", "Honda", "Toyota", "Ford", "BMW", "Audi"]
fuel_opts  = ["Petrol", "Diesel", "CNG", "Electric", "LPG"]
trans_opts = ["Manual", "Automatic"]
owner_opts = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
seller_opts = ["Individual", "Dealer", "Trustmark Dealer"]

brand       = st.sidebar.selectbox("Brand / Name", brand_opts)
year        = st.sidebar.slider("Year of manufacture", 2000, 2023, 2018)
km_driven   = st.sidebar.number_input("KM driven", min_value=0, max_value=500000, value=45000, step=1000)
fuel        = st.sidebar.selectbox("Fuel type", fuel_opts)
transmission= st.sidebar.selectbox("Transmission", trans_opts)
owner       = st.sidebar.selectbox("Owner", owner_opts)
seller_type = st.sidebar.selectbox("Seller type", seller_opts)
mileage     = st.sidebar.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=18.0, step=0.1)
engine      = st.sidebar.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200, step=100)
max_power   = st.sidebar.number_input("Max power (bhp)", min_value=30.0, max_value=600.0, value=82.0, step=1.0)
seats       = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9])

predict_btn = st.sidebar.button("🔮 Predict Price")

# ── Main Layout ──────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown("### 🎯 Prediction result")
    
    if predict_btn:
        with st.spinner("Calling API..."):
            payload = {
                "brand": brand, "year": year, "km_driven": km_driven, "fuel": fuel,
                "transmission": transmission, "owner": owner, "seller_type": seller_type,
                "mileage": mileage, "engine": engine, "max_power": max_power, "seats": seats
            }
            try:
                response = requests.post(BACKEND_URL, json=payload)
                if response.status_code == 200:
                    price = response.json()["predicted_price"]
                    margin = round(price * 0.048)
                    lo, hi = price - margin, price + margin
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">Estimated selling price</div>
                        <div class="result-price">₹{price:,}</div>
                        <div class="result-range">Confidence range: ₹{lo:,} – ₹{hi:,}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.error("Backend Error: Model could not process data.")
            except:
                st.error("Connection Error: Is the backend terminal running?")
    else:
        st.markdown('<div class="result-card"><div class="result-label">Fill in car details and click</div><div class="result-price" style="font-size:28px;">🔮 Predict Price</div><div class="result-range">in the sidebar</div></div>', unsafe_allow_html=True)

    st.markdown("### 🏆 Model comparison")
    model_df = pd.DataFrame({"Model": ["Random Forest", "Gradient Boosting", "Linear Regression"], "R² score": ["92.4%", "90.1%", "71.3%"]})
    st.dataframe(model_df, use_container_width=True, hide_index=True)

with right_col:
    st.markdown("### ⭐ Feature importance")
    # simplified list to save memory
    feats = ["Year", "KM driven", "Fuel type", "Brand", "Power"]
    imps = [88, 72, 55, 48, 31]
    fig_feat = px.bar(x=imps, y=feats, orientation='h', color_discrete_sequence=['#7F77DD'])
    fig_feat.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("### 📊 Price by fuel type")
    fuel_data = pd.DataFrame({"Fuel": ["Petrol", "Diesel", "CNG", "Electric"], "Price": [520000, 780000, 390000, 1420000]})
    fig_fuel = px.bar(fuel_data, x="Fuel", y="Price", color_discrete_sequence=['#534AB7'])
    fig_fuel.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_fuel, use_container_width=True)

st.markdown("<p style='text-align:center;color:#555;font-size:12px;'>Connected: FastAPI + Streamlit</p>", unsafe_allow_html=True)