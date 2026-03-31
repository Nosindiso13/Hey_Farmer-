
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
import joblib
import os
import io # New import for image handling
from PIL import Image # New import for image handling

# Import for Gemini API
import google.generativeai as genai
from google.colab import userdata

# New Tensorflow imports for pest detection
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image # New import

# --- 0. Configure Gemini (ensure API key is set)
GOOGLE_API_KEY = userdata.get('crop_key') if 'COLAB_JUPYTER_IP' in os.environ else os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Pydantic Model for Request Body (Crop Yield) ---
class CropYieldRequest(BaseModel):
    Area: str
    Item: str
    Year: int = Field(..., gt=0, description="Year of cultivation (positive integer)")
    rainfall: float = Field(..., ge=0.0, description="Average annual rainfall in mm (non-negative)")
    pesticides: float = Field(..., ge=0.0, description="Pesticides applied in tonnes (non-negative)")
    temperature: float = Field(..., ge=-50.0, le=70.0, description="Average temperature in Celsius (-50 to 70)")

# --- 1b. Pydantic Model for Request Body (Chat) ---
class ChatRequest(BaseModel):
    message: str

# --- 2. FastAPI Application Initialization ---
app = FastAPI(
    title="Crop Yield & Farmer Advisory API",
    description="API for predicting crop yields and providing farmer advice using a pre-trained XGBoost model and Gemini LLM.",
    version="1.0.0"
)

# --- 3. Global Model Variables ---
model_pipeline = None
gemini_model = None
pest_model = None # New global variable for pest detection model
MODEL_PATH = 'model_artifacts/xgboost_pipeline.joblib'

# --- 4. Load Models on Startup ---
@app.on_event('startup')
async def load_models():
    global model_pipeline, gemini_model, pest_model # Include pest_model

    # Load XGBoost model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please ensure it is saved.")
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"XGBoost Model pipeline loaded successfully from {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Error loading XGBoost model from {MODEL_PATH}: {e}")

    # Initialize Gemini Model
    try:
        if genai.get_default_options().get('api_key') is None:
             api_key = userdata.get('crop_key') if 'COLAB_JUPYTER_IP' in os.environ else os.getenv('GOOGLE_API_KEY')
             if not api_key: raise ValueError("Google API Key not found for Gemini.")
             genai.configure(api_key=api_key)

        gemini_model = genai.GenerativeModel('gemini-pro')
        print("Gemini model initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Gemini model: {e}")

    # Load MobileNetV2 model for pest detection
    try:
        pest_model = MobileNetV2(weights='imagenet')
        print("Pest detection model (MobileNetV2) loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading MobileNetV2 model: {e}")

# --- 5. Helper Prediction Function (Crop Yield) ---
def predict_yield_helper(input_data: List[CropYieldRequest]) -> List[float]:
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Crop Yield Model not loaded. Please restart the service or check model path.")

    # Convert Pydantic models to DataFrame
    data_dicts = [item.model_dump() for item in input_data]
    df = pd.DataFrame(data_dicts)

    # Ensure column order matches training data (important for ColumnTransformer)
    expected_columns = ['Area', 'Item', 'Year', 'rainfall', 'pesticides', 'temperature']
    if not all(col in df.columns for col in expected_columns):
        raise HTTPException(status_code=400, detail="Missing one or more required input columns for crop yield prediction.")
    df = df[expected_columns]

    # Make predictions (pipeline handles preprocessing)
    log_predictions = model_pipeline.predict(df)

    # Inverse transform to get predictions in original scale
    original_scale_predictions = np.expm1(log_predictions)

    return original_scale_predictions.tolist()

# --- 5b. Helper Chat Function (LLM Advisory) ---
async def get_gemini_response(message: str) -> str:
    if gemini_model is None:
        raise HTTPException(status_code=500, detail="Gemini model not initialized. Please check API key and restart service.")
    try:
        response = await gemini_model.generate_content_async(message)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API call failed: {e}")

# --- 5c. Helper Pest Detection Function ---
def detect_pest_helper(image_bytes: bytes) -> List[Dict]:
    if pest_model is None:
        raise HTTPException(status_code=500, detail="Pest Detection Model not loaded.")

    try:
        img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = pest_model.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]

        # Format predictions for JSON response
        results = [{"label": label, "description": description, "probability": float(prob)}
                   for (_, label, prob) in decoded_preds]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing or prediction failed: {e}")

# --- 6. API Endpoints ---
@app.get('/')
async def read_root():
    return {'message': 'Crop Yield Prediction & Farmer Advisory API is running. Visit /docs for API documentation.'}

@app.post('/predict', response_model=Dict[str, List[float]])
async def predict(request_data: List[CropYieldRequest]):
    """
    Predicts crop yields for a list of input crop parameters.

    Args:
        request_data (List[CropYieldRequest]): A list of crop parameters for prediction.

    Returns:
        Dict[str, List[float]]: A dictionary containing a list of predicted yields (hg/ha).
    """

    try:
        predictions = predict_yield_helper(request_data)
        return {'predictions': predictions}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop yield prediction failed: {e}")

@app.post('/chat', response_model=Dict[str, str])
async def chat(request: ChatRequest):
    """
    Provides farmer advisory services using a Large Language Model (LLM).

    Args:
        request (ChatRequest): A Pydantic model containing the farmer's message.

    Returns:
        Dict[str, str]: A dictionary containing the LLM's response.
    """
    try:
        response_text = await get_gemini_response(request.message)
        return {'response': response_text}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM advisory failed: {e}")

@app.post('/detect_pest', response_model=Dict[str, List[Dict]])
async def detect_pest_endpoint(file: UploadFile = File(...)): # New endpoint for pest detection
    """
    Detects pests/objects in an uploaded image using a pre-trained MobileNetV2 model.

    Args:
        file (UploadFile): The image file to analyze.

    Returns:
        Dict[str, List[Dict]]: A dictionary containing a list of detected objects
                                with their labels, descriptions, and probabilities.
    """
    if not file.content_type.startswith('image/') and not file.content_type.startswith('application/octet-stream'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    image_bytes = await file.read()
    try:
        results = detect_pest_helper(image_bytes)
        return {"detections": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pest detection failed: {e}")
