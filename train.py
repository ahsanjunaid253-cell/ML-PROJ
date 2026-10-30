# train.py
# This file trains the machine-learning models and saves them in the models folder.

import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score


# 1. Find the project folders
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(ROOT, "data", "02-preprocessed", "train_cleaned.csv")
models_folder = os.path.join(ROOT, "models")
classifiers_folder = os.path.join(models_folder, "classifiers")

os.makedirs(models_folder, exist_ok=True)
os.makedirs(classifiers_folder, exist_ok=True)


# 2. Read the cleaned dataset
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("Smoker values:")
print(df["smoker"].value_counts())

if df["smoker"].nunique() != 2:
    raise ValueError("The smoker column must contain both 0 and 1.")


# 3. Choose the input columns for each task
regression_inputs = ["age", "bmi", "children", "smoker"]
classification_inputs = ["age", "bmi", "children", "charges"]


# 4. Train the regression model to predict insurance charges
X = df[regression_inputs]
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

predictions = regression_model.predict(X_test)
r2 = r2_score(y_test, predictions)

print("\nRegression R2 score:", round(r2, 3))


# 5. Prepare data to predict smoker or non-smoker
X = df[classification_inputs]
y = df["smoker"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 6. Create the five classification models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "SVM": make_pipeline(
        StandardScaler(), SVC(probability=True, random_state=42)
    ),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}


# 7. Train each model and calculate its accuracy
accuracies = {}

print("\nClassification accuracy:")

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    accuracies[name] = accuracy
    print(name, ":", round(accuracy, 3))


# 8. Find the model with the highest accuracy
best_name = max(accuracies, key=accuracies.get)
best_model = models[best_name]

print("\nBest model:", best_name)


# 9. Save the regression model
joblib.dump(
    regression_model,
    os.path.join(models_folder, "regression_model.joblib")
)


# 10. Save the best classification model
joblib.dump(
    best_model,
    os.path.join(models_folder, "classification_model.joblib")
)


# 11. Save all five classifiers and their accuracy information
model_info = {
    "best": best_name.lower().replace(" ", "_"),
    "models": {}
}

for name, model in models.items():

    # Convert the model name into a file-friendly name
    model_slug = name.lower().replace(" ", "_")

    # Save this model
    model_path = os.path.join(
        classifiers_folder, model_slug + ".joblib"
    )
    joblib.dump(model, model_path)

    # Store its proper name and accuracy
    model_info["models"][model_slug] = {
        "name": name,
        "accuracy": round(accuracies[name], 3)
    }


# Save model names and accuracies in a JSON file
info_path = os.path.join(models_folder, "classifier_info.json")

with open(info_path, "w") as file:
    json.dump(model_info, file, indent=2)


# 12. Test the saved models with sample information
saved_regression_model = joblib.load(
    os.path.join(models_folder, "regression_model.joblib")
)

saved_classification_model = joblib.load(
    os.path.join(models_folder, "classification_model.joblib")
)

sample_regression = pd.DataFrame(
    [[35, 27.5, 2, 0]],
    columns=regression_inputs
)

sample_classification = pd.DataFrame(
    [[45, 32.0, 1, 40000]],
    columns=classification_inputs
)

print("\nSample predicted charges:",
      round(saved_regression_model.predict(sample_regression)[0], 2))

print("Sample smoker prediction:",
      saved_classification_model.predict(sample_classification)[0])

print("Sample smoker probability:",
      round(saved_classification_model.predict_proba(sample_classification)[0][1], 3))

print("\nTraining finished. Models have been saved.")
