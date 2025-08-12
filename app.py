import streamlit as st
import numpy as np
import joblib
import gdown
import os

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="🔥 Fire Type Classifier",
    page_icon="🔥",
    layout="centered"
)

# ====== HEADER ======
st.title("🔥 Fire Type Classification")
st.markdown(
    """
    <style>
    .big-font { font-size:22px !important; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="big-font">Predict fire type from MODIS satellite readings</p>', unsafe_allow_html=True)
st.write("---")

# ====== Download model from Google Drive ======
file_id = "1K-CkomnFCIaZVmTCCGRVR1wG75_54EZU"
model_filename = "best_fire_detection_model.pkl"

if not os.path.exists(model_filename):
    with st.spinner("⏳ Downloading model from Google Drive... Please wait"):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_filename, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ====== Load model & scaler ======
model = joblib.load(model_filename)
scaler = joblib.load("scaler.pkl")  # Ensure this file is in the project

# ====== SIDEBAR INFO ======
st.sidebar.header("ℹ️ About the App")
st.sidebar.write("""
This app uses a trained ML model to predict fire types based on MODIS satellite readings.  
**Model Inputs:**
- Brightness  
- Brightness T31  
- Fire Radiative Power (FRP)  
- Scan & Track  
- Confidence Level
""")
st.sidebar.info("Created with ❤️ using Streamlit")

# ====== INPUT FORM ======
with st.form(key="fire_form"):
    col1, col2 = st.columns(2)
    with col1:
        brightness = st.number_input("🌞 Brightness", value=300.0, min_value=200.0, max_value=500.0, step=0.5)
        frp = st.number_input("🔥 Fire Radiative Power (FRP)", value=15.0, min_value=0.0, max_value=500.0, step=0.1)
        confidence = st.selectbox("📊 Confidence Level", ["low", "nominal", "high"])
    with col2:
        bright_t31 = st.number_input("🌡 Brightness T31", value=290.0, min_value=200.0, max_value=350.0, step=0.5)
        scan = st.number_input("📏 Scan", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
        track = st.number_input("🛤 Track", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

    submit_button = st.form_submit_button("🔍 Predict Fire Type")

# ====== PREDICTION ======
if submit_button:
    confidence_map = {"low": 0, "nominal": 1, "high": 2}
    confidence_val = confidence_map[confidence]

    input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    fire_types = {
        0: "🌿 Vegetation Fire",
        2: "🏭 Other Static Land Source",
        3: "🌊 Offshore Fire"
    }
    result = fire_types.get(prediction, "❓ Unknown")

    # ====== RESULT DISPLAY ======
    st.subheader("📌 Prediction Result")
    st.success(f"**Predicted Fire Type:** {result}")

    if "Vegetation" in result:
        st.info("🌱 Likely vegetation or forest fire detected.")
    elif "Offshore" in result:
        st.warning("🌊 Possible offshore fire detected — marine monitoring recommended.")
    else:
        st.error("🏭 Fire detected from other land-based sources.")
