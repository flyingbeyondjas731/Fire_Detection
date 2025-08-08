import streamlit as st
import numpy as np
import joblib
import gdown
import os

# File IDs from Google Drive
model_file_id = "1K-CkomnFCIaZVmTCCGRVR1wG75_54EZU"
model_filename = "best_fire_detection_model.pkl"
scaler_filename = "scaler.pkl"

# Download model from Google Drive if not present
if not os.path.exists(model_filename):
    model_url = f"https://drive.google.com/uc?id={model_file_id}"
    gdown.download(model_url, model_filename, quiet=False)

# Load model
model = joblib.load(model_filename)

# Load scaler (must be present in the same folder or also use gdown if needed)
if os.path.exists(scaler_filename):
    scaler = joblib.load(scaler_filename)
else:
    st.warning("Scaler file not found. Please upload scaler.pkl to the same folder.")
    scaler = None

# Example Streamlit content
st.title("🔥 Fire Detection App")

st.write("This is a demo of a deployed fire detection model using Streamlit.")

# Optional input section (adjust as needed)
# if scaler is not None:
#     user_input = st.text_input("Enter sample values...")
#     processed_input = scaler.transform([user_input])
#     prediction = model.predict(processed_input)
#     st.write(f"Prediction: {prediction}")
# Set page title
st.set_page_config(page_title="Fire Type Classifier", layout="centered")

# App title and info
st.title("Fire Type Classification")
st.markdown("Predict fire type based on MODIS satellite readings.")

# User input fields for 6 features
brightness = st.number_input("Brightness", value=300.0)
bright_t31 = st.number_input("Brightness T31", value=290.0)
frp = st.number_input("Fire Radiative Power (FRP)", value=15.0)
scan = st.number_input("Scan", value=1.0)
track = st.number_input("Track", value=1.0)
confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"])

# Map confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Combine and scale input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

# Predict and display
if st.button("Predict Fire Type"):
    prediction = model.predict(scaled_input)[0]

    fire_types = {
        0: "Vegetation Fire",
        2: "Other Static Land Source",
        3: "Offshore Fire"
    }

    result = fire_types.get(prediction, "Unknown")
    st.success(f"**Predicted Fire Type:** {result}")
