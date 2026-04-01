from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
import joblib
import os
import io
from PIL import Image

# Import for Gemini API
import google.generativeai as genai

# New Tensorflow imports for pest detection
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# --- 0. Configure Gemini ---
# In a separate process, we use environment variables instead of userdata.get()
GOOGLE_API_KEY = os.getenv('YIELD_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Pydantic Models ---
class CropYieldRequest(BaseModel):
    Area: str
    Item: str
    Year: int = Field(..., gt=0)
    rainfall: float = Field(..., ge=0.0)
    pesticides: float = Field(..., ge=0.0)
    temperature: float = Field(..., ge=-50.0, le=70.0)

class ChatRequest(BaseModel):
    message: str

# --- 2. 
app = (title="Crop Yield & Farmer Advisory API")

# --- 3. Global Models ---
model_pipeline = None
gemini_model = None
pest_model = None
MODEL_PATH = 'model_artifacts/xgboost_pipeline.joblib'

@app.on_event('startup')
async def load_models():
    global model_pipeline, gemini_model, pest_model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}.")
    model_pipeline = joblib.load(MODEL_PATH)
    gemini_model = genai.GenerativeModel('gemini-pro')
    pest_model = MobileNetV2(weights='imagenet')
    print("All models loaded successfully.")

# --- 4. API Endpoints ---
@app.get('/')
async def read_root():
    return {'status': 'active'}

@app.post('/predict')
async def predict(request_data: List[CropYieldRequest]):
    if model_pipeline is None: raise HTTPException(status_code=500, detail="Model not loaded")
    df = pd.DataFrame([item.model_dump() for item in request_data])
    log_predictions = model_pipeline.predict(df)
    return {'predictions': np.expm1(log_predictions).tolist()}

@app.post('/chat')
async def chat(request: ChatRequest):
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini API not configured")
    response = await gemini_model.generate_content_async(request.message)
    return {'response': response.text}

@app.post('/detect_pest')
async def detect_pest_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    preds = decode_predictions(pest_model.predict(x), top=3)[0]
    return {"detections": [{"label": l, "description": d, "probability": float(p)} for (_, l, p) in preds]}
