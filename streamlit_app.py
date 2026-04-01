import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from PIL import Image
import asyncio
import google.colab

# Imports for Gemini API
import google.generativeai as genai
from google.colab import userdata

# Imports for Pest Detection
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# --- Configuration ---
MODEL_PATH = 'model_artifacts/xgboost_pipeline.joblib'
MARKET_PATH = 'market_trends.csv'

@st.cache_resource
def load_models():
    yield_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    pest_model = MobileNetV2(weights='imagenet')
    api_key = userdata.get('yield_key')
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel('gemini-pro')
    return yield_model, pest_model, gemini

yield_pipeline, pest_model, gemini_model = load_models()

st.set_page_config(page_title='Farmer Advisor & Market', layout='wide')
st.title('🌾 Crop Advisor & Marketplace')

tabs = st.tabs(['📈 Yield Prediction', '🪲 Pest Detection', '🤖 AI Advisor', '🛒 Market & Trends'])

with tabs[0]:
    st.header('Yield Prediction')
    with st.form('yield_form'):
        area = st.selectbox('Area', ['Zambia', 'Zimbabwe'])
        item = st.selectbox('Crop', ['Wheat', 'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans'])
        year = st.number_input('Year', 2024, 2030, 2025)
        rain = st.slider('Rainfall (mm)', 0, 3500, 1000)
        pest = st.slider('Pesticides (tonnes)', 0, 150000, 5000)
        temp = st.slider('Temp (°C)', 10, 45, 25)
        if st.form_submit_button('Predict'):
            input_df = pd.DataFrame([[area, item, year, rain, pest, temp]], columns=['Area', 'Item', 'Year', 'rainfall', 'pesticides', 'temperature'])
            pred = np.expm1(yield_pipeline.predict(input_df))[0]
            st.metric('Estimated Yield', f'{pred:,.2f} hg/ha')

with tabs[1]:
    st.header('Pest Identification')
    file = st.file_uploader('Upload leaf or pest image', type=['jpg', 'png'])
    if file:
        img = Image.open(file).resize((224, 224))
        st.image(img)
        x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
        preds = decode_predictions(pest_model.predict(x), top=3)[0]
        for _, label, prob in preds:
            st.write(f'**{label}**: {prob:.2%}')

with tabs[2]:
    st.header('AI Chatbot')
    if 'messages' not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: st.chat_message(m['role']).write(m['content'])
    if prompt := st.chat_input('How can I improve my soil?'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').write(prompt)
        response = gemini_model.generate_content(prompt)
        st.session_state.messages.append({'role': 'assistant', 'content': response.text})
        st.chat_message('assistant').write(response.text)

with tabs[3]:
    st.header('🛒 Farmer Marketplace & Trending Crops')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Trending Crops This Season')
        if os.path.exists(MARKET_PATH):
            df_trends = pd.read_csv(MARKET_PATH)
            st.dataframe(df_trends, use_container_width=True)
        else: st.info('No trend data available.')
    with col2:
        st.subheader('List Your Crop for Sale')
        with st.form('market_form'):
            seller_name = st.text_input('Name')
            crop_type = st.selectbox('Crop', ['Wheat', 'Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans'])
            quantity = st.number_input('Quantity (kg)', min_value=1)
            price = st.number_input('Asking Price ($)', min_value=1)
            if st.form_submit_button('Post Listing'):
                st.success(f'Listing created for {seller_name}! Others can now see your {crop_type}.')
                # In a real app, this would append to a database/CSV
