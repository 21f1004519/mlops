from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
import traceback

MODEL_PATH = "artifacts/model.joblib"

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchPredictRequest(BaseModel):
    records: list[PredictRequest]

app = FastAPI(title="Iris FastAPI Model")

model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("Model loaded from", MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        x = np.array([[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]])
        preds = model.predict(x)
        return {"prediction": preds[0]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    try:
        X = [[r.sepal_length, r.sepal_width, r.petal_length, r.petal_width] for r in req.records]
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
