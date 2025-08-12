import streamlit as st
import numpy as np
import joblib
import gdown
import os

# ====== Download model from Google Drive ======
# Replace with your file ID from Google Drive link
# Example: https://drive.google.com/file/d/1AbCdEfGhIJ/view?usp=sharing
# File ID = 1AbCdEfGhIJ
file_id = "1K-CkomnFCIaZVmTCCGRVR1wG75_54EZU"
model_filename = "best_fire_detection_model.pkl"

if not os.path.exists(model_filename):
    st.write("Downloading model from Google Drive... (only first run will be slow)")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_filename, quiet=False)

# ====== Load model and scaler ======
model = joblib.load(model_filename)
scaler = joblib.load("scaler.pkl")  # scaler file is already in project

# ====== Streamlit App UI ======
st.set_page_config(page_title="Fire Type Classifier", layout="centered")
st.title("Fire Type Classification")
st.markdown("Predict fire type based on MODIS satellite readings.")

brightness = st.number_input("Brightness", value=300.0)
bright_t31 = st.number_input("Brightness T31", value=290.0)
frp = st.number_input("Fire Radiative Power (FRP)", value=15.0)
scan = st.number_input("Scan", value=1.0)
track = st.number_input("Track", value=1.0)
confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"])

confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

if st.button("Predict Fire Type"):
    prediction = model.predict(scaled_input)[0]
    fire_types = {
        0: "Vegetation Fire",
        2: "Other Static Land Source",
        3: "Offshore Fire"
    }
    result = fire_types.get(prediction, "Unknown")
    st.success(f"**Predicted Fire Type:** {result}")
