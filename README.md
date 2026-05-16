# 🔍 ERP Anomaly Detector

An AI-powered fraud detection system for ERP financial transactions using Machine Learning, SHAP explainability, FastAPI, and Streamlit.

---

## 📊 Results

| Model | Precision | Recall | F1-Score | PR-AUC |
|---|---|---|---|---|
| Isolation Forest | 25% | 33% | 0.28 | 0.19 |
| **Random Forest** | **96%** | **76%** | **0.85** | **0.88** |

Trained and evaluated on **284,807 real transactions** with a fraud rate of only **0.17%** (highly imbalanced).

---

## 🏗️ Architecture

```
creditcard.csv (Kaggle)
       │
       ▼
┌─────────────────────┐
│  Feature Engineering │  log_amount, hour_of_day, is_night, amount_zscore
└─────────────────────┘
       │
       ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Isolation Forest    │     │   Random Forest      │
│  (Unsupervised)      │     │  (Supervised)        │
│  Precision: 25%      │     │  Precision: 96%  ✅  │
└─────────────────────┘     └─────────────────────┘
                                      │
                                      ▼
                             ┌─────────────────────┐
                             │   SHAP Explainability│
                             │   Top risk factors   │
                             └─────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                   ┌─────────────┐       ┌─────────────────┐
                   │  FastAPI    │       │    Streamlit     │
                   │  REST API   │       │    Dashboard     │
                   │  /predict   │       │  3 pages + charts│
                   │  /batch     │       │  CSV upload      │
                   └─────────────┘       └─────────────────┘
```

---

## 🧠 Models Used

**Isolation Forest** — unsupervised anomaly detection. No labels needed. Isolates outliers in feature space. Used as baseline.

**Random Forest** — supervised classifier with `class_weight='balanced'` to handle the 0.17% fraud rate. Significantly outperforms Isolation Forest on this dataset.

**SHAP** — explains *why* each transaction was flagged by showing the top contributing features per prediction.

---

## ⚙️ Feature Engineering

| Raw Field | Engineered Feature | Description |
|---|---|---|
| `Amount` | `log_amount` | Log-transformed to reduce skew |
| `Amount` | `amount_zscore` | Z-score relative to dataset mean |
| `Time` | `hour_of_day` | Hour extracted from seconds |
| `Time` | `is_night` | 1 if transaction between 10pm–6am |

---

## 🗂️ Project Structure

```
ERP_ANOMALY_DETECTOR/
│
├── pipeline.py          # Full ML pipeline: EDA → models → SHAP → flagged transactions
├── save_model.py        # Trains model and saves .pkl artifacts
├── api.py               # FastAPI backend with /predict and /batch endpoints
├── dashboard.py         # Streamlit dashboard (3 pages)
│
├── model.pkl            # Trained Random Forest model
├── scaler.pkl           # StandardScaler fitted on training data
├── feature_cols.pkl     # Feature column names
│
├── requirements.txt     # Python dependencies
└── render.yaml          # Render deployment config
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/iamviplavkr/ERP_ANOMALY_DETECTOR.git
cd ERP_ANOMALY_DETECTOR
```

### 2. Set up virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 5. Train and save the model
```bash
python save_model.py
```

### 6. Run the ML pipeline
```bash
python pipeline.py
```

### 7. Start the FastAPI backend
```bash
python -m uvicorn api:app --reload
```
Open `http://127.0.0.1:8000/docs` for Swagger UI.

### 8. Launch the Streamlit dashboard
```bash
python -m streamlit run dashboard.py
```
Open `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check API status |
| `GET` | `/stats` | Model info and feature list |
| `POST` | `/predict` | Analyze a single transaction |
| `POST` | `/predict/batch` | Analyze multiple transactions |

### Sample Request — `/predict`
```json
{
  "vendor_id": "V00123",
  "department": "Finance",
  "approved_by": "mgr_01",
  "posting_time": 3600,
  "transaction_amount": 9999.99,
  "V1": -2.3, "V2": 1.9, "V3": -2.1, "V4": 3.2,
  "V5": -1.1, "V6": 0.5, "V7": -0.8, "V8": 0.3,
  "V9": -0.6, "V10": -2.4, "V11": 1.8, "V12": -3.1,
  "V13": 0.2, "V14": -2.5, "V15": 0.4, "V16": -1.2,
  "V17": -2.8, "V18": -0.3, "V19": 0.1, "V20": 0.2,
  "V21": 0.5, "V22": -0.1, "V23": 0.0, "V24": 0.3,
  "V25": 0.1, "V26": -0.2, "V27": 0.1, "V28": 0.0
}
```

### Sample Response
```json
{
  "vendor_id": "V00123",
  "department": "Finance",
  "anomaly_score": 0.87,
  "is_fraud": true,
  "risk_level": "HIGH",
  "alert_message": "⚠️ High-risk transaction flagged. Immediate review required.",
  "top_risk_factors": [
    { "feature": "V14", "importance": 0.1602, "value": -2.5 },
    { "feature": "V12", "importance": 0.1108, "value": -3.1 }
  ]
}
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| ML Models | Scikit-learn (Isolation Forest, Random Forest) |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Data | Pandas, NumPy |
| Version Control | Git + GitHub |

---

## 📁 Dataset

[Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions
- 492 fraud cases (0.17%)
- 30 features (V1–V28 PCA-transformed + Time + Amount)
- No missing values

> The dataset is not included in this repo due to its size (143MB). Download directly from Kaggle.

---

## 👤 Author

**Viplav Kumar**
B.Tech Computer Science — Manipal University Jaipur
GitHub: [@iamviplavkr](https://github.com/iamviplavkr)
