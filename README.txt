# Insurance ML App

A small full-stack machine learning web app. It uses scikit-learn models, a FastAPI backend, and a plain HTML frontend.

- **Regression:** predicts a person's insurance charges from age, BMI, number of children, and smoker status.
- **Classification:** predicts whether a person is a smoker from age, BMI, number of children, and charges.

## Results

| Model | Metric | Score |
|---|---|---|
| Linear Regression | R² | 0.781 |
| Logistic Regression | Accuracy | 0.966 |

The classifier's baseline (always predicting "non-smoker") is 0.795, so the model performs clearly better than guessing the majority class.

## Dataset

The [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (`insurance.csv`, 1,338 rows) from Kaggle.

## Project Structure

```
├── server.py            # FastAPI backend
├── index.html           # Frontend page with two forms
├── PROJ3.ipynb          # Model training notebook
├── insurance.csv        # Dataset
├── requirements.txt     # Python dependencies
└── models/
    ├── regression_model.joblib
    └── classification_model.joblib
```

## How to Run

1. Install Python 3 and clone this repository:
   ```
   git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
   cd YOUR-REPO
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the server:
   ```
   uvicorn server:app --reload
   ```
   (On Windows, if `python` isn't recognized, use `py -m pip` and `py -m uvicorn` instead.)
4. Open **http://127.0.0.1:8000/** in your browser.

Interactive API docs are available at **http://127.0.0.1:8000/docs**.

## API Endpoints

### `POST /predict/regression`

Request:
```json
{ "age": 35, "bmi": 27.5, "children": 2, "smoker": 0 }
```
Response:
```json
{ "predicted_charges": 6741.69 }
```

### `POST /predict/classification`

Request:
```json
{ "age": 35, "bmi": 27.5, "children": 2, "charges": 30000 }
```
Response:
```json
{ "predicted_smoker": 1, "smoker_probability": 0.97 }
```

## Retraining the Models

Open `PROJ3.ipynb` and run all cells. This trains both models, prints R² and accuracy, and saves them into the `models/` folder with joblib. Use the same scikit-learn version when loading them.

## Limitations

- The models don't validate realistic ranges (for example, a negative age still returns a prediction).
- The classifier uses `charges` as an input feature. This works well on this dataset because smokers pay much higher charges, but in real use you would not know the charges in advance.
- The dataset is small, and the results shouldn't be used for real insurance decisions.

## Technologies

Python, pandas, scikit-learn, joblib, FastAPI, Uvicorn, HTML and JavaScript.