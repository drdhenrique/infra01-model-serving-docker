from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

class PredictionRequest(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

model = joblib.load('models/model.pkl')

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = pd.DataFrame([request.model_dump()])
    prediction = model.predict(input_data)[0]
    return {"predicted_median_house_value": prediction}