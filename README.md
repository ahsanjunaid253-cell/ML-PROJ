# Insurance ML App

A small full-stack machine learning app: scikit-learn models, a FastAPI backend, and an HTML frontend.

## Project structure
- `data/` raw data (`insurance.csv`) and cleaned data (`02-preprocessed/`)
- `training/` preprocessing and training scripts
- `models/` saved models (joblib)
- `backend/` FastAPI app
- `frontend/` HTML page
- `reports/` charts

## Dataset
Kaggle "Medical Cost Personal Datasets" (insurance.csv), 1338 rows.

## Models
- Regression: linear regression predicts `charges` from age, bmi, children and smoker (R2 about 0.78).
- Classification: predicts `smoker` from age, bmi, children and charges. Logistic regression, SVM, naive Bayes, decision tree and random forest were compared; the best (random forest, accuracy about 0.98, baseline 0.795) is saved.
- The website has a dropdown to choose which classifier makes the prediction (best model, logistic regression, SVM, naive Bayes, decision tree or random forest). All five are saved in `models/classifiers/`, and the best one is also saved as `classification_model.joblib`.
- `sex` and `region` were tested and excluded because they did not improve performance.

## How to run
1. `python -m pip install -r requirements.txt`
2. `python run_all.py` (cleans the data, trains and saves the models, starts the API)
   or run the steps separately:
   `python training/pre_processed.py`, `python training/train.py`, `uvicorn backend.main:app --reload`
3. Open `frontend/index.html` in your browser. API docs: http://127.0.0.1:8000/docs

## Endpoints
- POST `/predict/regression` with age, bmi, children, smoker (0 or 1)
- POST `/predict/classification` with age, bmi, children, charges and an optional `model` (`best`, `logistic_regression`, `svm`, `naive_bayes`, `decision_tree` or `random_forest`; default `best`)
