from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from src.pipelines.prediction_pipeline import (
    PredictionPipeline,
)

app = FastAPI()


class ObesityData(BaseModel):

    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str


@app.get("/")
def home():

    return {
        "message": "HealthSense MLOps API Running"
    }


@app.post("/predict")
def predict(data: ObesityData):

    input_data = pd.DataFrame(
        {
            "Gender": [data.Gender],
            "Age": [data.Age],
            "Height": [data.Height],
            "Weight": [data.Weight],
            "family_history_with_overweight": [
                data.family_history_with_overweight
            ],
            "FAVC": [data.FAVC],
            "FCVC": [data.FCVC],
            "NCP": [data.NCP],
            "CAEC": [data.CAEC],
            "SMOKE": [data.SMOKE],
            "CH2O": [data.CH2O],
            "SCC": [data.SCC],
            "FAF": [data.FAF],
            "TUE": [data.TUE],
            "CALC": [data.CALC],
            "MTRANS": [data.MTRANS],
        }
    )

    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
        input_data
    )

    return {
        "prediction": prediction[0]
    }