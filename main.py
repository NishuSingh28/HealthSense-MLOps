import pandas as pd

from src.pipelines.prediction_pipeline import (
    PredictionPipeline,
)


data = pd.DataFrame(
    {
        "Gender": ["Male"],
        "Age": [25],
        "Height": [1.75],
        "Weight": [80],
        "family_history_with_overweight": [
            "yes"
        ],
        "FAVC": ["yes"],
        "FCVC": [2],
        "NCP": [3],
        "CAEC": ["Sometimes"],
        "SMOKE": ["no"],
        "CH2O": [2],
        "SCC": ["no"],
        "FAF": [1],
        "TUE": [1],
        "CALC": ["Sometimes"],
        "MTRANS": ["Public_Transportation"],
    }
)

pipeline = PredictionPipeline()

prediction = pipeline.predict(data)

print(prediction)