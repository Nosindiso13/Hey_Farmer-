import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from PIL import Image
import asyncio
from datetime import datetime, timedelta

# Authentication imports
from passlib.context import CryptContext

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Import for Gemini API
import google.genai as genai
from google.colab import userdata

# Imports for Pest Detection
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# --- Configuration & Global Variables ---
MODEL_PATH = 'model_artifacts/xgboost_pipeline.joblib'

# Security settings (local to Streamlit now)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database Setup (local to Streamlit now)
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- SQLAlchemy Models ---
class DBUser(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

# --- User Management Functions ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Caching Models for Efficiency ---
@st.cache_resource
def load_crop_yield_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Crop Yield Model file not found at {MODEL_PATH}.")
        return None
    try:
        # Ensure consistent use of joblib or pickle. Since the notebook created it with joblib,
        # we'll stick to joblib here.
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Error loading Crop Yield model: {e}")
        return None

@st.cache_resource
def load_pest_detection_model():
    try:
        # Explicitly configure TensorFlow to use CPU only
        tf.config.set_visible_devices([], 'GPU')
        model = MobileNetV2(weights='imagenet')
        return model
    except Exception as e:
        st.error(f"Error loading Pest Detection model: {e}")
        return None

@st.cache_resource
def load_gemini_model():
    try:
        api_key = userdata.get('yield_key')
        if not api_key:
            st.error("Google API Key not found. Please set 'yield_key' in Colab secrets.")
            return None
        model = genai.GenerativeModel('gemini-pro', api_key=api_key)
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        return None

# --- Helper Functions ---
def predict_yield_helper(input_data_df: pd.DataFrame, model_pipeline_local) -> list[float]:
    if model_pipeline_local is None:
        st.error("Crop Yield Model is not loaded.")
        return []
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

# --- Main App Logic ---
@st.cache_resource
def initialize_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    TEST_USERNAME = "testuser"
    TEST_PASSWORD = "testpassword123"
    db_user = db.query(DBUser).filter(DBUser.username == TEST_USERNAME).first()
    if not db_user:
        hashed_password = get_password_hash(TEST_PASSWORD)
        new_user = DBUser(username=TEST_USERNAME, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        st.success(f"Default test user '{TEST_USERNAME}' registered during startup.")
    db.close()

initialize_database()

crop_yield_pipeline = load_crop_yield_model()
pest_detection_model = load_pest_detection_model()
gemini_llm = load_gemini_model()

st.set_page_config(page_title='Farmer Advisor & Marketplace', layout='wide')
st.title('🌾 Crop Advisor & Marketplace')

# --- Authentication Logic (Direct DB Interaction) ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

def login_form():
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            db = next(get_db())
            user_data = db.query(DBUser).filter(DBUser.username == username).first()
            db.close()
            if user_data and verify_password(password, user_data.hashed_password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password.")

def register_form():
    st.subheader("Register New User")
    with st.form("register_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                db = next(get_db())
                db_user = db.query(DBUser).filter(DBUser.username == new_username).first()
                if db_user:
                    st.error("Username already registered")
                    db.close()
                else:
                    hashed_password = get_password_hash(new_password)
                    new_db_user = DBUser(username=new_username, hashed_password=hashed_password)
                    db.add(new_db_user)
                    db.commit()
                    db.refresh(new_db_user)
                    db.close()
                    st.success("Registration successful! Please login.")
                    st.session_state.logged_in = False
                    st.experimental_rerun()

if not st.session_state.logged_in:
    st.sidebar.title("Authentication")
    auth_option = st.sidebar.radio("", ["Login", "Register"])
    if auth_option == "Login":
        login_form()
    else:
        register_form()
    st.stop() # Stop execution if not logged in

# --- Main App Content (only shown if logged in) ---
st.sidebar.write(f"Logged in as: {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.experimental_rerun()

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
            input_data = {
                "Area": area,
                "Item": item,
                "Year": year,
                "rainfall": float(rain),
                "pesticides": float(pest),
                "temperature": float(temp)
            }
            input_df = pd.DataFrame([input_data])
            if crop_yield_pipeline:
                predicted_yields = predict_yield_helper(input_df, crop_yield_pipeline)
                if predicted_yields:
                    predicted_yield = predicted_yields[0]
                    st.success(f'Estimated Yield: {predicted_yield:,.2f} hg/ha')
                else:
                    st.error("Prediction could not be made.")
            else:
                st.warning("Crop Yield Model not available. Please check the logs.")

with tabs[1]:
    st.header('Pest Identification')
    file = st.file_uploader('Upload leaf or pest image', type=['jpg', 'png'])
    if file:
        st.image(file, caption="Uploaded Image", use_column_width=True)
        st.write('Detecting pest...')
        image_bytes = file.getvalue()
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

with tabs[2]:
    st.header('AI Chatbot')
    if 'messages' not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: st.chat_message(m['role']).write(m['content'])
    if prompt := st.chat_input('How can I improve my soil?'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').write(prompt)
        with st.spinner("Getting advice from the AI..."):
            if gemini_llm:
                response_text = asyncio.run(get_gemini_response(prompt, gemini_llm))
                st.session_state.messages.append({'role': 'assistant', 'content': response_text})
                st.chat_message('assistant').write(response_text)
            else:
                error_msg = "AI chat model not available. Please check API key configuration."
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
                st.chat_message('assistant').write(error_msg)

with tabs[3]:
    st.header('🛒 Farmer Marketplace & Trending Crops')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Trending Crops This Season')
        MARKET_PATH = 'market_trends.csv'
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
