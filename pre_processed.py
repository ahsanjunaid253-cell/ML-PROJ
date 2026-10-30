import os
import pandas as pd

# project folder = one level above the training folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_path = os.path.join(ROOT, "data", "insurance.csv")
output_path = os.path.join(ROOT, "data", "02-preprocessed", "train_cleaned.csv")

# 1. Load and inspect
df = pd.read_csv(raw_path)
print("Raw shape:", df.shape)
print(df.isna().sum())

# 2. Clean: drop rows with missing values, encode smoker as 1/0
df = df.dropna(subset=["age", "bmi", "children", "smoker", "charges"])
df["smoker"] = (df["smoker"] == "yes").astype(int)
print("Clean shape:", df.shape)
print(df["smoker"].value_counts())              # expect 1064 zeros, 274 ones

# 3. Save the cleaned data for train.py
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print("Saved to", output_path)
