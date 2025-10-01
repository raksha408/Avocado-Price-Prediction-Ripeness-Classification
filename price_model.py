import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor, Pool
import os

# === Constants ===
INR_RATE = 82.0
IMAGE_PATH = r"C:\Users\sraks\OneDrive\Pictures\avocado_price pic.png"
DATASET_PATH = r"D:\Major\Programs\Avocado_Price_Prediction\Dataset\avocado.csv"

PLOT_PATHS = {
    "ExtraTree Regressor": [r"D:\Major\Avocado_Project\assets\plots\extratree_scatter.png"],
    "DecisionTree Regressor": [r"D:\Major\Avocado_Project\assets\plots\decisiontree_scatter.png"],
    "RandomForest": [r"D:\Major\Avocado_Project\assets\plots\randomforest_scatter.png"],
    "CatBoost": [r"D:\Major\Avocado_Project\assets\plots\catboost_scatter.png"],
}

# === Page Config ===
st.set_page_config(page_title="🥑 Avocado Price Prediction", page_icon="🥑", layout="centered")

# === Styling ===
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cartoonify&display=swap');
  .stApp, body { background-color: #e6f4ea !important; }
  #MainMenu, header { visibility: hidden !important; }
  h1.avocado-header {
    font-family: 'Cartoonify', cursive !important;
    font-size: 4rem !important;
    color: #568203 !important;
    text-align: center !important;
    margin: 0.5rem 0 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2) !important;
  }
  .avocado-subheader {
    font-family: 'Cartoonify', cursive !important;
    color: #33691e !important;
    font-size: 2rem !important;
    margin: 1rem 0 0.5rem !important;
  }
  input[type=number]::-webkit-outer-spin-button,
  input[type=number]::-webkit-inner-spin-button {
    -webkit-appearance: none !important; margin: 0 !important;
  }
  input[type=number] { -moz-appearance: textfield !important; }
  .stNumberInput button { display: none !important; }
  .stTextInput, .stDateInput, .stSelectbox, .stRadio, div.stButton > button {
    font-family: 'Cartoonify', cursive !important;
  }
  div.stButton {
    display: flex !important; justify-content: center !important; margin-top: 1rem !important;
  }
  div.stButton > button {
    background-color: #81c784 !important; color: #fff !important;
    border-radius: 0.75rem !important; padding: 0.75rem 1.5rem !important;
    font-size: 1.125rem !important; font-weight: bold !important;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1) !important;
    transition: background-color 0.3s ease, box-shadow 0.3s ease !important;
  }
  div.stButton > button:hover {
    background-color: #4caf50 !important; box-shadow: 4px 4px 10px rgba(0,0,0,0.15) !important;
  }
  .result-box {
    background-color: #dcedc8 !important; border: 2px solid #aed581 !important;
    border-radius: 1rem !important; padding: 1.25rem !important; margin-top: 1.25rem !important;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1) !important;
    font-family: 'Cartoonify', cursive !important; color: #33691e !important; font-size: 1.25rem !important;
  }
</style>
""", unsafe_allow_html=True)

# === Header ===
c1, c2 = st.columns([1, 3])
c1.image(IMAGE_PATH, use_container_width=True)
c2.markdown('<h1 class="avocado-header">Avocado Price Prediction</h1>', unsafe_allow_html=True)

# === Load dataset to get all regions ===
try:
    df = pd.read_csv(DATASET_PATH)
    all_regions = sorted(df['region'].unique().tolist())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    all_regions = ['West']  # fallback option

# === Load Models ===
MODELS = {
    "ExtraTree Regressor": joblib.load("avocado_price/models/ExtraTree/extratrees_after.joblib"),
    "DecisionTree Regressor": joblib.load("avocado_price/models/DecisionTree/decision_tree_best_model.joblib"),
    "RandomForest": joblib.load("avocado_price/models/RandomForest/final_model.pkl"),
    "CatBoost": CatBoostRegressor().load_model("avocado_price/models/Catboost/catboost_price_model_tuned.cbm"),
}
model_key = st.radio("Choose Model:", list(MODELS.keys()), index=0, horizontal=True)
model = MODELS[model_key]

# === Input Form ===
st.markdown('<div class="avocado-subheader">Enter Avocado Details</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    sel_date = st.date_input("Date")
    region   = st.selectbox("Region", all_regions)
    type_    = st.selectbox("Type", ['conventional', 'organic'])
with right:
    tv    = st.number_input("Total Volume", min_value=0.0, format="%.2f", key="tv")
    p4046 = st.number_input("4046",         min_value=0.0, format="%.2f", key="p4046")
    p4225 = st.number_input("4225",         min_value=0.0, format="%.2f", key="p4225")
    p4770 = st.number_input("4770",         min_value=0.0, format="%.2f", key="p4770")

# === Prediction Logic ===
if st.button("Predict Average Price"):
    # Check if all numerical values are 0.00
    if all(v == 0.0 for v in [tv, p4046, p4225, p4770]):
        usd, inr = 0.0, 0.0
    else:
        ord_d = sel_date.toordinal()
        bags = max(0.0, tv - (p4046 + p4225 + p4770))

        X = pd.DataFrame([{
            "Date": ord_d,
            "Total Volume": tv,
            "4046": p4046,
            "4225": p4225,
            "4770": p4770,
            "Total Bags": bags,
            "Small Bags": bags * 0.6,
            "Large Bags": bags * 0.3,
            "XLarge Bags": bags * 0.1,
            "type": type_,
            "year": sel_date.year,
            "region": region
        }])

        # Predict using model
        usd = model.predict(Pool(X, cat_features=["type", "region"]))[0] if model_key == "CatBoost" else model.predict(X)[0]
        inr = usd * INR_RATE

    # === Display Result ===
    st.markdown(
        f"""
        <div class="result-box">
          <h3>Prediction Result</h3>
          <p><b>Price (USD):</b> ${usd:.2f}</p>
          <p><b>Price (INR):</b> ₹{inr:,.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # === Show Scatter Plot for Selected Model ===
    st.markdown("Price analysis")
    for p in PLOT_PATHS.get(model_key, []):
        if os.path.exists(p):
            cols = st.columns([1, 2, 1])
            cols[1].image(p, width=400)
        else:
            st.warning(f"Plot not found: {p}")
