from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

reg_model = joblib.load("models/regression_model.joblib")
clf_model = joblib.load("models/classification_model.joblib")


class RegressionInput(BaseModel):
    age: int
    bmi: float
    children: int
    smoker: int


class ClassificationInput(BaseModel):
    age: int
    bmi: float
    children: int
    charges: float


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    row = pd.DataFrame(
        [[data.age, data.bmi, data.children, data.smoker]],
        columns=["age", "bmi", "children", "smoker"],
    )
    prediction = reg_model.predict(row)[0]
    return {"predicted_charges": float(prediction)}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    row = pd.DataFrame(
        [[data.age, data.bmi, data.children, data.charges]],
        columns=["age", "bmi", "children", "charges"],
    )
    prediction = clf_model.predict(row)[0]
    probability = clf_model.predict_proba(row)[0][1]
    return {
        "predicted_smoker": int(prediction),
        "smoker_probability": float(probability),
    }