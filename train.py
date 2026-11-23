"""
train.py

Trains a classifier on data/data.csv (produced by collect_data.py) and
saves it to models/asl_letter_model.pkl.

HOW TO USE:
  Run once you've collected data for all the letters you want:
    python train.py

  It will print a precision/recall/F1 report per letter so you can see
  which letters are being confused (common ones: M/N/S, or K/V) and
  decide if you need to collect more/cleaner data for those.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

DATA_FILE = "data/data_normalized.csv"
MODEL_OUT = "models/asl_letter_model.pkl"

os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_FILE, header=None)

# Only static ASL letters (A-I and K-Y, excluding J)
VALID_LETTERS = list("ABCDEFGHI KLMNOPQRSTUVWXY".replace(" ", ""))

# Remove J and Z from the dataset
df = df[df.iloc[:, -1].isin(VALID_LETTERS)]

print("Letters included in training:", sorted(df.iloc[:, -1].unique()))
print("Total training samples:", len(df))

# Separate features and labels
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# Ensure every expected letter has training samples
missing = set(VALID_LETTERS) - set(y.unique())

if missing:
    raise ValueError(
        f"Missing training samples for these letters: {sorted(missing)}. "
        "Record these letters using collect_data.py first."
    )
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")
