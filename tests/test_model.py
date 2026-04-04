import joblib
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.app.main import app

@pytest.fixture(scope="module")
def model():
    return joblib.load("models/model.pkl")

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_model_loads(model):
    assert model is not None

SAMPLE_INPUT = {
      "longitude": -124.23,
      "latitude": 40.54,
      "housing_median_age": 52,
      "total_rooms": 2694,
      "total_bedrooms": 453.0,
      "population": 1152,
      "households": 435,
      "median_income": 3.0806,
      "ocean_proximity": "NEAR OCEAN"
    }

def test_prediction_shape(model):
    df = pd.DataFrame([SAMPLE_INPUT])
    prediction = model.predict(df)
    assert prediction.shape == (1,)



def test_prediction_is_finite(model):
    df = pd.DataFrame([SAMPLE_INPUT])
    prediction = model.predict(df)
    assert np.isfinite(prediction).all()

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_predict(client):
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200

def test_predict_estrutura(client):
    response = client.post("/predict", json=SAMPLE_INPUT)
    assert "predicted_median_house_value" in response.json()
    assert isinstance(response.json()["predicted_median_house_value"], float)

def test_predict_input_invalido(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_validation_gate():
    df = pd.read_csv("data/california-housing.csv")
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=31415)
    model = joblib.load("models/model.pkl")
    y_pred = model.predict(X_test)
    R2 = r2_score(y_test, y_pred)
    assert R2 >= 0.5, f"Model performance is not satisfactory. R2: {R2}"