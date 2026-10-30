import os
import json
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")            # save charts as PNG files, no pop-up windows
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay)

# ---------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project folder
input_path = os.path.join(ROOT, "data", "02-preprocessed", "train_cleaned.csv")
models_dir = os.path.join(ROOT, "models")
reports_dir = os.path.join(ROOT, "reports")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load the cleaned data (made by pre_processed.py)
# ---------------------------------------------------------------
df = pd.read_csv(input_path)
print("Shape:", df.shape)
print(df["smoker"].value_counts())   # expect 1064 zeros, 274 ones
if df["smoker"].nunique() != 2:
    raise ValueError("smoker column should have both 0 and 1. Re-run pre_processed.py")

# ---------------------------------------------------------------
# 2. Feature lists (same names and order in main.py and index.html)
# ---------------------------------------------------------------
reg_features = ["age", "bmi", "children", "smoker"]
clf_features = ["age", "bmi", "children", "charges"]

# ---------------------------------------------------------------
# 3. Regression: linear regression (predict charges)
# ---------------------------------------------------------------
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    df[reg_features], df["charges"], test_size=0.2, random_state=42
)
reg_model = LinearRegression()
reg_model.fit(Xr_train, yr_train)
r2 = r2_score(yr_test, reg_model.predict(Xr_test))
print("\nRegression R2:", round(r2, 3))

# ---------------------------------------------------------------
# 4. Classification: compare 5 models (predict smoker)
# ---------------------------------------------------------------
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    df[clf_features], df["smoker"], test_size=0.2, random_state=42,
    stratify=df["smoker"]
)

clf_models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

baseline = 1 - df["smoker"].mean()
print("\n--- Classification results (accuracy on test set) ---")
print("Baseline (always non-smoker):", round(baseline, 3))
accuracies = {}
for name, model in clf_models.items():
    model.fit(Xc_train, yc_train)
    accuracies[name] = accuracy_score(yc_test, model.predict(Xc_test))
    print(f"{name:20s} accuracy = {accuracies[name]:.3f}")

best_name = max(accuracies, key=accuracies.get)
clf_model = clf_models[best_name]
print("\nBest classifier:", best_name)
print(confusion_matrix(yc_test, clf_model.predict(Xc_test)))



# ---------------------------------------------------------------
# 6. Save, reload, and test with one hand-typed row
# ---------------------------------------------------------------
reg_path = os.path.join(models_dir, "regression_model.joblib")
clf_path = os.path.join(models_dir, "classification_model.joblib")
joblib.dump(reg_model, reg_path)
joblib.dump(clf_model, clf_path)          # the best classifier (default)

# Also save EVERY classifier, so the website can offer a model dropdown
clf_dir = os.path.join(models_dir, "classifiers")
os.makedirs(clf_dir, exist_ok=True)
info = {"best": best_name.lower().replace(" ", "_"), "models": {}}
for name, model in clf_models.items():
    slug = name.lower().replace(" ", "_")          # "Random Forest" -> "random_forest"
    joblib.dump(model, os.path.join(clf_dir, slug + ".joblib"))
    info["models"][slug] = {"name": name, "accuracy": round(accuracies[name], 3)}
with open(os.path.join(models_dir, "classifier_info.json"), "w") as f:
    json.dump(info, f, indent=2)

reg_loaded = joblib.load(reg_path)
clf_loaded = joblib.load(clf_path)

sample_reg = pd.DataFrame([[35, 27.5, 2, 0]], columns=reg_features)
sample_clf = pd.DataFrame([[45, 32.0, 1, 40000]], columns=clf_features)
print("\nPredicted charges:", round(reg_loaded.predict(sample_reg)[0], 2))
print("Predicted smoker:", clf_loaded.predict(sample_clf)[0])
print("Smoker probability:", round(clf_loaded.predict_proba(sample_clf)[0][1], 3))
print("\nDone. Models saved in models/, charts saved in reports/.")
