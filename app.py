import streamlit as st
import numpy as np
import joblib
import gdown
import os
import plotly.graph_objects as go

# ====== CUSTOM CSS ======
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background: transparent;
}
.card {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}
h1, h2, h3 {
    color: #FFD700;
}
.stButton>button {
    background-color: #FFD700;
    color: black;
    border-radius: 12px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #FFC300;
}
</style>
""", unsafe_allow_html=True)

# ====== Download model ======
file_id = "1K-CkomnFCIaZVmTCCGRVR1wG75_54EZU"
model_filename = "best_fire_detection_model.pkl"

if not os.path.exists(model_filename):
    st.write("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_filename, quiet=False)

# ====== Load model & scaler ======
model = joblib.load(model_filename)
scaler = joblib.load("scaler.pkl")

# ====== App Title ======
st.title("🔥 Fire Type Classification")
st.markdown("Predict fire type from MODIS satellite readings — now with style!")

# ====== Preset Buttons ======
preset_values = {
    "Vegetation Fire": [330, 300, 20, 1.2, 1.0, "high"],
    "Static Land": [280, 275, 5, 0.5, 0.5, "low"],
    "Offshore Fire": [350, 320, 50, 2.0, 1.5, "nominal"]
}

preset_choice = st.radio("🔹 Quick Test Presets:", list(preset_values.keys()))
if st.button("Load Preset"):
    vals = preset_values[preset_choice]
    brightness, bright_t31, frp, scan, track, conf = vals
else:
    brightness, bright_t31, frp, scan, track, conf = 300.0, 290.0, 15.0, 1.0, 1.0, "low"

# ====== Inputs ======
st.markdown("<div class='card'>", unsafe_allow_html=True)
brightness = st.number_input("Brightness", value=brightness)
bright_t31 = st.number_input("Brightness T31", value=bright_t31)
frp = st.number_input("Fire Radiative Power (FRP)", value=frp)
scan = st.number_input("Scan", value=scan)
track = st.number_input("Track", value=track)
confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"], index=["low","nominal","high"].index(conf))
st.markdown("</div>", unsafe_allow_html=True)

# ====== Prediction ======
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

if st.button("🔮 Predict Fire Type"):
    prediction = int(model.predict(scaled_input)[0])
    fire_types = {0: "Vegetation Fire", 2: "Other Static Land Source", 3: "Offshore Fire"}
    result = fire_types.get(prediction, "Unknown")
    st.success(f"**Predicted Fire Type:** {result}")

    # Debug Info
    with st.expander("🔍 Debug Info"):
        st.write("Raw model output:", prediction)
        st.write("Scaled input:", scaled_input)

    # ====== Radar Chart ======
    fig = go.Figure()
    categories = ["Brightness", "Brightness T31", "FRP", "Scan", "Track", "Confidence"]
    fig.add_trace(go.Scatterpolar(
        r=[brightness, bright_t31, frp, scan, track, confidence_val],
        theta=categories,
        fill='toself',
        name='Input Features'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ====== FRP Gauge ======
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=frp,
        title={'text': "FRP Intensity"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "orange"}}
    ))
    gauge_fig.update_layout(template="plotly_dark")
    st.plotly_chart(gauge_fig, use_container_width=True)
