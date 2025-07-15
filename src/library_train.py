import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_data

# === Config ===
FILE_PATH = "/mnt/c/Users/kashy/Documents/Code/Projects/stock-price-classifier/data/processed/HDFCBANK.NS.csv"

# === Load Data ===
x_train, x_test, y_train, y_test = load_data(FILE_PATH)

# === Initialize and Train Random Forest ===
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced',  # Handles class imbalance
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# === Evaluation ===
acc = accuracy_score(y_test, y_pred)
print(f"🌲 Random Forest Accuracy: {acc:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
