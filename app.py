import streamlit as st
import numpy as np
import joblib

# Page setup
st.set_page_config(
    page_title="🔥 Fire Type Classifier",
    layout="centered",
    page_icon="🔥"
)

# Load model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load("best_fire_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# Title and intro
st.markdown(
    "<h1 style='text-align: center; color: orange;'>🔥 Fire Type Classification</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Predict the fire type using MODIS satellite data inputs.</p>",
    unsafe_allow_html=True
)
st.divider()

# Inputs in two columns
col1, col2 = st.columns(2)

with col1:
    brightness = st.number_input("🔥 Brightness", min_value=200.0, max_value=500.0, value=300.0)
    frp = st.number_input("🔥 Fire Radiative Power (FRP)", min_value=0.0, max_value=500.0, value=15.0)
    scan = st.number_input("📡 Scan", min_value=0.1, max_value=10.0, value=1.0)

with col2:
    bright_t31 = st.number_input("🌡 Brightness T31", min_value=200.0, max_value=500.0, value=290.0)
    track = st.number_input("🛰 Track", min_value=0.1, max_value=10.0, value=1.0)
    confidence = st.selectbox("✅ Confidence Level", ["low", "nominal", "high"])

# Map confidence to numeric value
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Combine and scale input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

# Predict and display result
st.divider()
if st.button("🚀 Predict Fire Type"):
    try:
        prediction = model.predict(scaled_input)[0]
        fire_types = {
            0: "🌿 Vegetation Fire",
            2: "🏞 Other Static Land Source",
            3: "🌊 Offshore Fire"
        }
        result = fire_types.get(prediction, "❓ Unknown")
        st.success(f"*Predicted Fire Type:* {result}")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
