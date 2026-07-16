"""Streamlit frontend calling the local FastAPI /predict endpoint."""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.title("Berlin Secondary Sales Price Predictor")
st.caption("Linear regression: price_eur ~ area_m2")

area_m2 = st.number_input("Area (m²)", min_value=1.0, value=70.0, step=1.0)

if st.button("Predict price"):
    try:
        response = requests.get(API_URL, params={"area_m2": area_m2}, timeout=5)
        response.raise_for_status()
        price_eur = response.json()["price_eur"]
        st.metric("Predicted price", f"€{price_eur:,.0f}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}. Is it running? ({e})")
