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
    try:
        model = joblib.load("best_fire_detection_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model or scaler file not found. Please check your file paths.")
        st.stop()

model, scaler = load_assets()

st.title("🔥 Fire Type Classifier")
st.write("Enter fire detection feature values to predict the fire type.")

# Example input fields (replace with your dataset's features)
confidence = st.number_input("Confidence (%)", min_value=0, max_value=100, value=50)
latitude = st.number_input("Latitude", value=0.0, format="%.6f")
longitude = st.number_input("Longitude", value=0.0, format="%.6f")

if st.button("Predict Fire Type"):
    try:
        # Prepare input for model
        input_data = np.array([[confidence, latitude, longitude]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.success(f"🔥 Predicted Fire Type: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
