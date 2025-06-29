from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import joblib
import pandas as pd
import uvicorn

app = FastAPI()

# Optional CORS
# Uncomment if needed for your frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ==========================
# Load TensorFlow model
# ==========================

# Load Keras model (adjust your path)
try:
    keras_model = tf.keras.models.load_model("models/liver_model.h5")
    print("TensorFlow model loaded successfully.")
    print("Model input shape:", keras_model.input_shape)
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")
    keras_model = None

def preprocess_image(image_bytes):
    """Preprocess image for prediction"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # adjust for your model
    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if keras_model is None:
        raise HTTPException(status_code=500, detail="TensorFlow model not loaded.")
    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse({"error": "Only image files are supported."}, status_code=400)

        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        prediction = keras_model.predict(image)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return JSONResponse({
            "class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==========================
# Load sklearn/joblib model
# ==========================

try:
    skl_model = joblib.load("liver_disease_model.pkl")
    print(f"Loaded sklearn model: {type(skl_model)}")
except Exception as e:
    print(f"Error loading sklearn model: {e}")
    skl_model = None

class PatientData(BaseModel):
    age: int
    gender: str
    Tot_Bil: float
    Dir_Bil: float
    Alk_Phos: float
    ALT: float
    AST: float
    Tot_pro: float
    Alb: float
    A_G_Ratio: float

@app.post("/predict/tabular")
async def predict_tabular(data: PatientData):
    if skl_model is None:
        raise HTTPException(status_code=500, detail="Sklearn model not loaded.")

    try:
        input_data = pd.DataFrame([{
            "Age of the patient": data.age,
            "Gender": 1 if data.gender.lower() == "male" else 0,
            "Total Bilirubin": data.Tot_Bil,
            "Direct Bilirubin": data.Dir_Bil,
            "Alkphos Alkaline Phosphotase": data.Alk_Phos,
            "Sgpt Alamine Aminotransferase": data.ALT,
            "Sgot Aspartate Aminotransferase": data.AST,
            "Total Protiens": data.Tot_pro,
            "ALB Albumin": data.Alb,
            "AG Ratio Albumin and Globulin Ratio": data.A_G_Ratio
        }])

        prediction = skl_model.predict(input_data)
        result = "Liver Disease Present" if prediction[0] == 1 else "No Liver Disease"

        return {"prediction": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================
# Run with uvicorn
# ==========================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
