import streamlit as st
import numpy as np
import joblib
import os
import gdown

# Page setup
st.set_page_config(
    page_title="🔥 Fire Type Classifier",
    layout="centered",
    page_icon="🔥"
)

# Paths
model_file = "best_fire_detection_model.pkl"
scaler_file = "scaler.pkl"

# Google Drive file ID
model_drive_id = "1K-CkomnFCIaZVmTCCGRVR1wG75_54EZU"

# Function to download model from Google Drive
def download_model_from_drive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# Load model and scaler
@st.cache_resource
def load_assets():
    # Download model from Drive if not found
    if not os.path.exists(model_file):
        st.info("Downloading model from Google Drive...")
        download_model_from_drive(model_drive_id, model_file)

    try:
        model = joblib.load(model_file)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None

    # Load scaler
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
    else:
        st.warning("Scaler file not found. Please upload 'scaler.pkl' to the app directory.")
        scaler = None

    return model, scaler

model, scaler = load_assets()

# UI
st.markdown(
    "<h1 style='text-align: center; color: orange;'>🔥 Fire Type Classification</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Predict the fire type using MODIS satellite data inputs.</p>",
    unsafe_allow_html=True
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    brightness = st.number_input("🔥 Brightness", min_value=200.0, max_value=500.0, value=300.0)
    frp = st.number_input("🔥 Fire Radiative Power (FRP)", min_value=0.0, max_value=500.0, value=15.0)
    scan = st.number_input("📡 Scan", min_value=0.1, max_value=10.0, value=1.0)

with col2:
    bright_t31 = st.number_input("🌡 Brightness T31", min_value=200.0, max_value=500.0, value=290.0)
    track = st.number_input("🛰 Track", min_value=0.1, max_value=10.0, value=1.0)
    confidence = st.selectbox("✅ Confidence Level", ["low", "nominal", "high"])

confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])

# Check if model and scaler are loaded
if model is not None and scaler is not None:
    scaled_input = scaler.transform(input_data)

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
            st.success(f"Predicted Fire Type: {result}")
        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")
else:
    st.warning("Model or scaler not loaded — prediction not available.")
