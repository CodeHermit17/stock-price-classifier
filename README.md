# 📈 Stock Price Classifier

This project aims to **predict whether a stock’s closing price will increase or decrease** on the next day using **logistic regression implemented from scratch in Python**. It is part of a broader roadmap focused on machine learning applications in finance and internship preparation for programs like **GSoC** and **quantitative research roles**.

---

## 🚀 Project Goals

- Predict binary stock movement (up/down) using past price data and technical indicators
- Implement logistic regression **from scratch** (no ML libraries)
- Visualize learning progress (loss, accuracy)
- Build reproducible, modular codebase for extension (e.g., regression models, sentiment analysis)
- Strengthen experience in ML for finance domain

---

## 🛠️ Tech Stack

- **Python** (core logic)
- **NumPy** (matrix ops)
- **Pandas** (data preprocessing)
- **Matplotlib** (visualization)
- **scikit-learn** (only for evaluation metrics)

---

## 📊 Results Summary

- ✅ **Training Accuracy**: ~55%
- ✅ **Test Accuracy**: ~51–52%
- 📉 Custom learning curves (loss/accuracy) show convergence
- ⚠️ Class imbalance and market noise present natural challenges

---

## 📁 Project Structure

stock-price-classifier/
├── data/ # Raw & processed stock data (CSV)
├── src/ # Custom ML modules & training pipeline
├── models/ # Saved weights (theta.npy)
├── results/ # Accuracy & loss plots
├── README.md
└── requirements.txt

---

## 🧠 Core Implementation

- ✅ `logistic_model.py`: Logistic Regression from scratch (no sklearn!)
- ✅ `train.py`: Trains over multiple learning rates, logs accuracy/loss
- ✅ `evaluate.py`: Loads saved weights and reports classification metrics
- ✅ `data_loader.py`: Loads and processes CSV stock data

---

## 📈 Sample Outputs

Plots saved in `/results/` folder:

- `accuracy_vs_epochs_lr_0p001.png`
- `loss_vs_epochs_lr_0p001.png`
- `lr_vs_accuracy.png`

*(Example plot: Accuracy vs Epochs at different learning rates)*

---

## ✅ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/CodeHermit17/stock-price-classifier.git
cd stock-price-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python src/train.py

# 4. Evaluate best saved model
python src/evaluate.py
