import numpy as np
from logistic_model import LogisticRegressionScratch
from data_loader import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

FILE_PATH = "stock-price-classifier/data/processed/ITC.NS.csv"
MODEL_PATH = "stock-price-classifier/models/theta.npy"

_, x_test, _, y_test = load_data(FILE_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f" Model weights not found at {MODEL_PATH}. Run train.py first.")

theta = np.load(MODEL_PATH)

model = LogisticRegressionScratch(learning_rate=0.01, epochs=0)  # epochs unused here
model.theta = theta

# === Predict on Test Set ===
y_pred = model.predict(x_test)

# === Evaluate Performance ===
acc = accuracy_score(y_test, y_pred)
print(f" Test Accuracy: {acc:.4f}\n")

print(" Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
