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

## 🏗️ Production-Ready Architecture

The codebase has been refactored into a scalable, production-ready, modular architecture:

```
ERP_ANOMALY_DETECTOR/
│
├── backend/                  # FastAPI REST API Backend
│   ├── api/                  # Versioned routers (/v1/predict, /v1/stats)
│   ├── auth/                 # API Key authentication middleware
│   ├── core/                 # Centralized Config (.env), Exceptions, Logging
│   ├── middleware/           # HTTP Request Logger, Global Error Handlers
│   ├── models/               # Domain model representations
│   ├── repositories/         # Thread-safe model artifact caching
│   ├── schemas/              # Pydantic Request/Response models
│   ├── utils/                # Risk-level evaluation and risk factor formats
│   └── main.py               # Uvicorn API entry point
│
├── frontend/                 # Streamlit UI Dashboard
│   └── dashboard.py          # Streamlit implementation (multi-page)
│
├── ml/                       # Machine Learning Pipeline Modules
│   ├── training/             # Model training pipelines (train.py)
│   ├── evaluation/           # Performance evaluators (evaluator.py)
│   ├── features/             # Shared Feature Engineering logic
│   └── explainability/       # SHAP interpretation wrapper (shap_explainer.py)
│
├── artifacts/                # Serialized model binaries (.pkl)
├── data/                     # Raw datasets (.csv)
├── tests/                    # API, service, feature and integration tests
├── scripts/                  # DevOps setup and run automation tasks
│
├── Makefile                  # Build, run and test command shortcuts
├── Dockerfile                # Production Docker build container
├── docker-compose.yml        # Docker compose stack file
├── requirements.txt          # Python dependencies
└── render.yaml               # Cloud deployment descriptor
```

---

## 🚀 How to Run

### 1. Clone and Navigate
```bash
git clone https://github.com/iamviplavkr/ERP_ANOMALY_DETECTOR.git
cd ERP_ANOMALY_DETECTOR
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 3. Initialize Directories and Model Artifacts
Run the setup script or make target to copy/create directories and setup templates:
```bash
make setup
# OR: python scripts/setup_artifacts.py
```
> Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` directory.

### 4. Train Model
```bash
make train
# OR: python ml/training/train.py
```

### 5. Launch FastAPI Backend
```bash
make run-api
# OR: python backend/main.py
```
Visit Swagger API Docs at `http://127.0.0.1:8000/docs`.

### 6. Launch Streamlit Dashboard
```bash
make run-dashboard
# OR: streamlit run frontend/dashboard.py
```

---

## 🧪 Running Tests

A complete suite of unit and integration tests is located in the `tests/` directory:
```bash
make test
# OR: pytest tests/ -v
```

---

## ⚙️ Configuration Management

The application loads environment variables using python-dotenv. Create a `.env` file in the root directory (based on `.env.example`) to configure parameters:

```env
APP_NAME="ERP Anomaly Detector"
DEBUG=true
PORT=8000
FRAUD_THRESHOLD=0.5
HIGH_RISK_THRESHOLD=0.8
REQUIRE_API_KEY=false
API_KEY="your-api-key"
```

---

## 👤 Author

**Viplav Kumar**
B.Tech Computer Science — Manipal University Jaipur
GitHub: [@iamviplavkr](https://github.com/iamviplavkr)
