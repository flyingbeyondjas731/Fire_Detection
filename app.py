import streamlit as st
import numpy as np
import joblib
import gdown
import os
import base64
import plotly.graph_objects as go

# ====== Download model from Google Drive ======
file_id = "1K-CkomnFCIaZVmTCCGRVR1wG75_54EZU"
model_filename = "best_fire_detection_model.pkl"

if not os.path.exists(model_filename):
    st.write("Downloading model from Google Drive... (only first run will be slow)")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_filename, quiet=False)

# ====== Load model and scaler ======
model = joblib.load(model_filename)
scaler = joblib.load("scaler.pkl")  # scaler file is already in project

# ====== Set page config ======
st.set_page_config(page_title="Fire Type Classifier", layout="centered")

# ====== Encode background image ======
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# You need to save a background image file (e.g. 'forest_fire.jpg') in the same folder
background_image = "forest_fire.jpg"  # Change to your file name
if not os.path.exists(background_image):
    st.error("Background image missing! Please add a file named 'forest_fire.jpg' in the same folder.")
else:
    bg_base64 = get_base64(background_image)

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    # ====== Custom CSS for white & colorful text ======
custom_text_styles = """
<style>
/* Make all labels white */
label, .stSelectbox label, .stNumberInput label, .stRadio label {
    color: white !important;
    font-weight: bold;
    font-size: 1.1rem;
}

/* Change input box text color */
input, select, textarea {
    color: white !important;
    background-color: rgba(0,0,0,0.4) !important;
}

/* Change placeholder text color */
.stSelectbox div[role="listbox"], .stNumberInput input {
    color: white !important;
}

/* Title & description styles */
h1, h2, h3, h4, h5, h6, p {
    color: white !important;
}

/* Button styling */
.stButton button {
    background-color: rgba(255,69,0,0.9);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
}
.stButton button:hover {
    background-color: rgba(255,140,0,0.9);
    color: black;
}
</style>
"""
st.markdown(custom_text_styles, unsafe_allow_html=True)


# ====== Title ======
st.markdown("<h1 style='color:white;text-align:center;'>🔥 Fire Type Classification 🔥</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:white;text-align:center;'>Predict fire type based on MODIS satellite readings.</p>", unsafe_allow_html=True)

# ====== Presets ======
presets = {
    "Vegetation Fire": [320, 300, 25, 1.0, 1.0, 0],
    "Offshore Fire": [290, 280, 10, 0.5, 0.5, 2],
    "Static Land Source": [310, 295, 5, 0.8, 0.8, 1]
}

preset_choice = st.selectbox("Select a preset (optional)", ["Custom"] + list(presets.keys()))

if preset_choice != "Custom":
    brightness, bright_t31, frp, scan, track, confidence_val = presets[preset_choice]
else:
    brightness = st.number_input("Brightness", value=300.0)
    bright_t31 = st.number_input("Brightness T31", value=290.0)
    frp = st.number_input("Fire Radiative Power (FRP)", value=15.0)
    scan = st.number_input("Scan", value=1.0)
    track = st.number_input("Track", value=1.0)
    confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"])
    confidence_map = {"low": 0, "nominal": 1, "high": 2}
    confidence_val = confidence_map[confidence]

# ====== Prediction ======
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

fire_types = {
    0: "Vegetation Fire",
    1: "Other Static Land Source",
    2: "Offshore Fire"
}

if st.button("Predict Fire Type"):
    raw_pred = model.predict(scaled_input)[0]
    result = fire_types.get(raw_pred, f"Unknown ({raw_pred})")

    st.markdown(f"<div class='glass-card'><h2 style='color:white;'>Predicted Fire Type: {result}</h2></div>", unsafe_allow_html=True)

    # Debug info
    with st.expander("🔍 Debug Info"):
        st.write("Raw model output:", raw_pred)
        st.write("Scaled Input:", scaled_input)

    # ====== Radar Chart ======
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[brightness, bright_t31, frp, scan, track, confidence_val],
        theta=["Brightness", "Brightness T31", "FRP", "Scan", "Track", "Confidence"],
        fill='toself',
        name='Input Features',
        line_color="orange"
    ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        template="plotly_dark"
    )
    st.plotly_chart(radar, use_container_width=True)

    # ====== Gauge for FRP ======
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=frp,
        title={'text': "FRP Intensity"},
        gauge={'axis': {'range': [0, 50]},
               'bar': {'color': "red"}}
    ))
    gauge.update_layout(template="plotly_dark")
    st.plotly_chart(gauge, use_container_width=True)

