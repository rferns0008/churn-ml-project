import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# Define the expected input schema based on your dataset features
class ChurnPredictionRequest(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: int
    internet_service: int

app = FastAPI(title="Churn Prediction API")

# Instrument the app to expose /metrics for Prometheus monitoring
Instrumentator().instrument(app).expose(app)

# Load the trained Scikit-learn model packaged from your CI pipeline
try:
    model = joblib.load("model.pkl")
except Exception as e:
    model = None
    print(f"Warning: Model not found. {e}")

@app.post("/predict")
def predict_churn(request: ChurnPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([request.model_dump()])
    
    # Generate prediction and probability
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability)
    }

# Kubernetes liveness/readiness probe endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}