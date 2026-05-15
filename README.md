# ERP Anomaly Detector

Detects fraudulent transactions in ERP financial data using ML.

## Results
| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Isolation Forest | 25% | 33% | 0.28 | 0.19 |
| Random Forest | 96% | 76% | 0.85 | 0.88 |

## Tech Stack
Python, Scikit-learn, SHAP, Pandas, FastAPI

## Dataset
[Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## How to Run
pip install -r requirements.txt
python pipeline.py
