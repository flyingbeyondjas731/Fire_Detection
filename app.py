import streamlit as st
import numpy as np
import joblib
import gdown
import os
import plotly.graph_objects as go
from datetime import datetime

# ===================== CUSTOM CSS =====================
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #e3f2fd, #fce4ec);
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            padding: 2rem;
            border-radius: 20px;
        }
        .stButton>button {
            background-color: #42a5f5;
            color: white;
            border-radius: 12px;
            padding: 0.6em 1.5em;
            font-size: 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1e88e5;
        }
        .prediction-card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ===================== MODEL LOADING =====================
file_id = "1K-CkomnFCIaZVmTCCGRVR1wG75_54EZU"
model_filename = "best_fire_detection_model.pkl"

if not os.path.exists(model_filename):
    st.write("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_filename, quiet=False)

model = joblib.load(model_filename)
scaler = joblib.load("scaler.pkl")

# ===================== APP CONFIG =====================
st.set_page_config(page_title="🔥 Fire Type Classifier", layout="wide")
st.title("🔥 Fire Type Classification Dashboard")
st.markdown("A stylish, interactive tool to predict **fire type** from MODIS satellite readings.")

# ===================== PRESET DATA =====================
example_inputs = {
    "Vegetation Fire": [320, 300, 25, 1.2, 1.1, "nominal"],
    "Other Static Land Source": [290, 280, 5, 1.0, 1.0, "low"],
    "Offshore Fire": [310, 295, 40, 1.5, 1.3, "high"]
}

logs = []

# ===================== SIDEBAR PRESETS =====================
st.sidebar.header("⚡ Quick Presets")
preset_choice = st.sidebar.selectbox("Choose Example", ["Custom"] + list(example_inputs.keys()))
if preset_choice != "Custom":
    brightness, bright_t31, frp, scan, track, confidence = example_inputs[preset_choice]
else:
    brightness, bright_t31, frp, scan, track, confidence = 300.0, 290.0, 15.0, 1.0, 1.0, "nominal"

# ===================== USER INPUT =====================
confidence_map = {"low": 0, "nominal": 1, "high": 2}

col1, col2, col3 = st.columns(3)
with col1:
    brightness = st.number_input("Brightness", value=brightness)
    frp = st.number_input("FRP (Fire Radiative Power)", value=frp)
with col2:
    bright_t31 = st.number_input("Brightness T31", value=bright_t31)
    scan = st.number_input("Scan", value=scan)
with col3:
    track = st.number_input("Track", value=track)
    confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"], index=list(confidence_map.keys()).index(confidence))

# ===================== PREDICTION =====================
if st.button("🚀 Predict Fire Type"):
    input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_map[confidence]]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    fire_types = {0: "🌿 Vegetation Fire", 2: "🏭 Other Static Land Source", 3: "🌊 Offshore Fire"}
    result = fire_types.get(prediction, "❓ Unknown")

    # Log entry
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Predicted: {result} | Inputs: {list(input_data[0])}")

    # Show prediction card
    st.markdown(f"<div class='prediction-card'><h3>Predicted Fire Type: {result}</h3></div>", unsafe_allow_html=True)

    # Radar Chart
    categories = ["Brightness", "Brightness T31", "FRP", "Scan", "Track", "Confidence"]
    values = [brightness, bright_t31, frp, scan, track, confidence_map[confidence]]
    fig_radar = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', name='Your Input'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, height=400)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Gauge Chart for FRP
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=frp,
        title={'text': "FRP Intensity"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

# ===================== LOGS PANEL =====================
with st.expander("📜 Prediction Logs"):
    for log in logs:
        st.write(log)
