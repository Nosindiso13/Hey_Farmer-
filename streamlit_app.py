import streamlit as st
import requests # Will be removed later, kept for initial modification clarity
import json
import pandas as pd
import numpy as np
import pickle # Replaced joblib with pickle
import os
import io
from PIL import Image
import asyncio # Added for async operations

# Imports for Gemini API
import google.genai as genai # Corrected import
from google.colab import userdata

# Imports for Pest Detection
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# --- Configuration & Global Variables ---
MODEL_PATH = 'model_artifacts/xgboost_pipeline.joblib'

# --- Caching Models for Efficiency ---
@st.cache_resource
def load_crop_yield_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Crop Yield Model file not found at {MODEL_PATH}.")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            pipeline = pickle.load(f) # Replaced joblib.load with pickle.load
        return pipeline
    except Exception as e:
        st.error(f"Error loading Crop Yield model: {e}")
        return None

@st.cache_resource
def load_pest_detection_model():
    try:
        model = MobileNetV2(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Error loading Pest Detection model: {e}")
        return None

@st.cache_resource
def load_gemini_model():
    try:
        # Ensure API key is configured
        api_key = userdata.get('crop_key')
        if not api_key:
            st.error("Google API Key not found. Please set 'crop_key' in Colab secrets.")
            return None
        # Pass API key directly to the GenerativeModel constructor
        model = genai.GenerativeModel('gemini-pro', api_key=api_key)
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        return None

# --- Helper Functions (moved from main.py) ---
def predict_yield_helper(input_data_df: pd.DataFrame, model_pipeline_local) -> list[float]:
    if model_pipeline_local is None:
        st.error("Crop Yield Model is not loaded.")
        return []

    # Ensure column order matches training data
    expected_columns = ['Area', 'Item', 'Year', 'rainfall', 'pesticides', 'temperature']
    df_processed = input_data_df[expected_columns]

    log_predictions = model_pipeline_local.predict(df_processed)
    original_scale_predictions = np.expm1(log_predictions)
    return original_scale_predictions.tolist()

def detect_pest_helper(image_bytes: bytes, pest_model_local) -> list[dict]:
    if pest_model_local is None:
        st.error("Pest Detection Model is not loaded.")
        return []
    try:
        img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = pest_model_local.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]

        results = [{"label": label, "description": description, "probability": float(prob)}
                   for (_, label, prob) in decoded_preds]
        return results
    except Exception as e:
        st.error(f"Image processing or pest detection failed: {e}")
        return []

async def get_gemini_response(message: str, gemini_model_local) -> str:
    if gemini_model_local is None:
        st.error("Gemini model is not initialized.")
        return "Error: Gemini model not available."
    try:
        response = await gemini_model_local.generate_content_async(message)
        return response.text
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        return "Error: Could not get a response from AI."

# --- Load all models at startup ---
crop_yield_pipeline = load_crop_yield_model()
pest_detection_model = load_pest_detection_model()
gemini_llm = load_gemini_model()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Crop Yield Predictor & Advisory", layout="centered")

st.title("Crop Yield Prediction & Farmer Advisory")
st.markdown("Use the sections below to predict crop yields, detect pests, and get AI-powered advice.")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Crop Yield Prediction", "Pest Detection", "Farmer Advisory"])

with tab1:
    st.subheader("Input Parameters for Yield Prediction")

    AREAS = ['Zambia', 'Zimbabwe']
    ITEMS = ['Wheat', 'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Cassava', 'Sweet potatoes']
    MIN_YEAR = 2020
    MAX_YEAR = 2030
    MIN_RAINFALL = 0.0
    MAX_RAINFALL = 3500.0
    MIN_PESTICIDES = 0.0
    MAX_PESTICIDES = 150000.0
    MIN_TEMP = -20.0
    MAX_TEMP = 40.0

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            area = st.selectbox("Area", options=AREAS, index=0)
            item = st.selectbox("Item (Crop Type)", options=ITEMS, index=0)
            year = st.number_input("Year", min_value=MIN_YEAR, max_value=MAX_YEAR, value=2022, step=1)
        with col2:
            rainfall = st.number_input("Average Annual Rainfall (mm)", min_value=MIN_RAINFALL, max_value=MAX_RAINFALL, value=1200.0, step=10.0, format="%.1f")
            pesticides = st.number_input("Pesticides (tonnes)", min_value=MIN_PESTICIDES, max_value=MAX_PESTICIDES, value=60000.0, step=100.0, format="%.1f")
            temperature = st.number_input("Average Temperature (°C)", min_value=MIN_TEMP, max_value=MAX_TEMP, value=25.5, step=0.1, format="%.1f")

        submitted = st.form_submit_button("Get Prediction")

        if submitted:
            input_data = {
                "Area": area,
                "Item": item,
                "Year": year,
                "rainfall": rainfall,
                "pesticides": pesticides,
                "temperature": temperature
            }
            input_df = pd.DataFrame([input_data])

            if crop_yield_pipeline:
                predicted_yields = predict_yield_helper(input_df, crop_yield_pipeline)
                if predicted_yields:
                    predicted_yield = predicted_yields[0]
                    st.success(f"Predicted Crop Yield: **{predicted_yield:,.0f} hg/ha**")
                    st.balloons()
                else:
                    st.error("Prediction could not be made.")
            else:
                st.warning("Crop Yield Model not available. Please check the logs.")

with tab2:
    st.subheader("Pest Detection")
    st.write("Upload an image of a crop or pest to get a detection from our AI model.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Detecting pest...")

        image_bytes = uploaded_file.getvalue()

        if pest_detection_model:
            detections = detect_pest_helper(image_bytes, pest_detection_model)
            if detections:
                st.success("Detection Results:")
                for detection in detections:
                    st.write(f"- **{detection['description']}** (Confidence: {detection['probability']:.2f})")
            else:
                st.info("No significant detections found.")
        else:
            st.warning("Pest Detection Model not available. Please check the logs.")

with tab3:
    st.subheader("Farmer Advisory Chat")
    st.markdown("--- Nosindiso AI Demo --- ")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you, farmer?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Getting advice from the AI..."):
            if gemini_llm:
                response_text = st.session_state.get('gemini_response_cache', {})
                if prompt not in response_text:
                    response_text[prompt] = asyncio.run(get_gemini_response(prompt, gemini_llm))
                    st.session_state['gemini_response_cache'] = response_text

                ai_response = response_text.get(prompt, "Error: Could not get a response from AI.")
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
            else:
                error_msg = "AI chat model not available. Please check API key configuration."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
