import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# project folder = one level above the backend folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Insurance ML API")

# Allow the HTML page (opened from a file) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the regression model and the best classifier (default)
reg_model = joblib.load(os.path.join(ROOT, "models", "regression_model.joblib"))
clf_model = joblib.load(os.path.join(ROOT, "models", "classification_model.joblib"))

# Load every classifier too, for the dropdown on the website
with open(os.path.join(ROOT, "models", "classifier_info.json")) as f:
    clf_info = json.load(f)
classifiers = {}
for slug in clf_info["models"]:
    path = os.path.join(ROOT, "models", "classifiers", slug + ".joblib")
    classifiers[slug] = joblib.load(path)

# Must match train.py exactly (same names, same order)
REG_FEATURES = ["age", "bmi", "children", "smoker"]
CLF_FEATURES = ["age", "bmi", "children", "charges"]


class RegressionInput(BaseModel):
    age: int = Field(ge=18, le=100)
    bmi: float = Field(ge=10, le=60)
    children: int = Field(ge=0, le=10)
    smoker: int = Field(ge=0, le=1)       # 0 = no, 1 = yes


class ClassificationInput(BaseModel):
    age: int = Field(ge=18, le=100)
    bmi: float = Field(ge=10, le=60)
    children: int = Field(ge=0, le=10)
    charges: float = Field(ge=0)
    model: str = "best"                   # "best" or a model name like "svm"


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    row = pd.DataFrame(
        [[data.age, data.bmi, data.children, data.smoker]],
        columns=REG_FEATURES,
    )
    prediction = reg_model.predict(row)[0]
    return {"predicted_charges": round(float(prediction), 2)}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    # Pick the model the user chose in the dropdown
    if data.model == "best":
        slug = clf_info["best"]
        model = clf_model
    elif data.model in classifiers:
        slug = data.model
        model = classifiers[data.model]
    else:
        raise HTTPException(
            status_code=400,
            detail="Unknown model. Choose from: best, " + ", ".join(classifiers),
        )

    row = pd.DataFrame(
        [[data.age, data.bmi, data.children, data.charges]],
        columns=CLF_FEATURES,
    )
    prediction = int(model.predict(row)[0])
    prob_smoker = float(model.predict_proba(row)[0][1])
    return {
        "prediction": prediction,
        "label": "smoker" if prediction == 1 else "non-smoker",
        "probability_smoker": round(prob_smoker, 3),
        "model_used": clf_info["models"][slug]["name"],
        "model_accuracy": clf_info["models"][slug]["accuracy"],
    }
