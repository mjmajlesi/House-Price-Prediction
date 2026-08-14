<div align="center">

# 🏠 House Price Prediction — Austin, TX

**Predicting housing prices in Austin, Texas using modern machine learning and data engineering**

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)
[![Tests: pytest](https://img.shields.io/badge/Tests-pytest-green.svg)](https://docs.pytest.org/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange.svg)](https://mlflow.org/)
[![Model: LightGBM](https://img.shields.io/badge/Best%20Model-LightGBM-purple.svg)](https://lightgbm.readthedocs.io/)

---

</div>

---

## 📖 About the Project

This project is an **end-to-end pipeline — from data collection to model deployment** — for predicting listing prices (`list_price`) of properties in **Austin, Texas**. Data is automatically gathered from trusted real estate websites such as **Zillow, Redfin, and Realtor.com** using the `homeharvest` library.

The main goal is to build an accurate, reproducible regression model with proper **experiment tracking** that can reliably run in development and production environments.

> 🎯 **Current Best Model:** **LightGBM (Tuned)** with **$R^2 = 0.845$** and **MAE ≈ \$160,000** on test data.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🕷️ **Smart Scraping** | Automated collection of `for_sale` and `sold` listings for Austin ZIP codes with error handling and incremental storage. |
| 🗄️ **SQLite Database** | Clean storage with a standardized schema and `UPSERT` support (insert or update keyed on `property_id`). |
| ⚙️ **Advanced Feature Engineering** | New engineered features: `house_age`, `baths_per_bed`, `total_rooms`, `sqft_per_bed`, plus `log-transform` on the target variable. |
| 🧪 **Feature Selection** | Combination of **Feature Importance (Random Forest)** and **RFE (Ridge)** for robust feature subsetting. |
| 🤖 **Multi-Model Benchmarking** | Trains and evaluates: Linear, Ridge, Random Forest, XGBoost, and LightGBM. |
| 📊 **MLflow Experiment Tracking** | Logs parameters, metrics ($R^2, MAE, RMSE, MAPE$), and model artifacts in a visual dashboard. |
| ✅ **Unit Test Coverage** | 18 `pytest` tests covering database, scraper, pipeline, and training layers. |
| 📦 **Deployable Artifact** | Complete model bundle saved as `joblib` (preprocessor + model + metadata) ready for API/Streamlit use. |

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart LR
    A[🌐 Web Sources<br/>Zillow, Redfin, Realtor] -->|homeharvest| B[🕷️ Scraper<br/>src/scraper/]
    B --> C[(🗄️ SQLite DB<br/>data/database.sqlite)]
    B --> D[📄 Raw CSV<br/>data/raw/]
    C --> E[⚙️ Preprocessing<br/>Notebooks / Scripts]
    E --> F[🧹 Processed Data<br/>data/processed/]
    F --> G[🤖 Model Training<br/>src/train.py]
    G --> H[📊 MLflow Tracking<br/>mlruns/]
    G --> I[💾 Model Bundle<br/>notebooks/saved_models/*.joblib]
    I --> J[🚀 Serving / API<br/>(Future: FastAPI / Streamlit)]
```

---

## 📁 Directory Structure

```text
House-Price-Prediction/
├── .github/                      # GitHub Actions (CI/CD) — planned
├── data/
│   ├── raw/                      # Raw scraped CSV data
│   ├── processed/                # Cleaned and processed data
│   └── database.sqlite           # SQLite database
├── notebooks/
│   ├── data_scraping_test.ipynb  # Scraping test notebook
│   ├── exploratory_data_analysis.ipynb
│   ├── feature_selection_test.ipynb
│   ├── model_training.ipynb      # Main training notebook (legacy)
│   ├── data-preprocessing.ipynb  # Data preprocessing
│   └── saved_models/
│       └── house_price_model.joblib  # Final model bundle
├── src/
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── homeharvest_scraper.py # Core scraping module
│   │   └── run_pipeline.py        # Scraping CLI entry point
│   ├── utils/
│   │   ├── __init__.py
│   │   └── db_manager.py          # Database management (CRUD, UPSERT)
│   └── train.py                   # 🎯 Main training script + MLflow
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures
│   ├── test_db_manager.py         # Database layer tests
│   ├── test_scraper.py            # Scraper tests
│   ├── test_pipeline.py           # Full pipeline tests
│   └── test_train.py              # Training script tests
├── requirements.txt               # Project dependencies
├── .gitignore
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.11+** (tested on 3.11 and 3.13)
* `git` for cloning the repository

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Note:** If `homeharvest` fails to install on Python 3.13, you can temporarily remove it from `requirements.txt` (it is only needed for scraping and not for model training).

---

## ▶️ Usage Guide

### 1. Run the Scraper & Collect Data
Collects listings for Austin (or any custom city/ZIP code) and stores them in SQLite and CSV.

```bash
# Default: Austin, TX — 500 records
python src/scraper/run_pipeline.py

# Customized:
python src/scraper/run_pipeline.py --location "Dallas, TX" --limit 300 --listing-type for_sale --past-days 180
```

**Main Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--location` | "Austin, TX" | City, ZIP code, or "City, State" |
| `--limit` | 500 | Max number of listings per type |
| `--listing-type` | for_sale | `for_sale`, `sold`, `for_rent`, `pending` |
| `--past-days` | 365 | Look-back window for sold listings |
| `--mls-only` | True | Only MLS listings (more reliable) |

---

### 2. Train the Model + Log to MLflow

The `src/train.py` script reads the processed data, trains 5 models, logs all results to **MLflow**, and saves the best model to `notebooks/saved_models/house_price_model.joblib`.

```bash
python src/train.py
```

**Expected Output:**
```text
2026-08-12 10:00:00 - INFO - MLflow Experiment set to: 'House_Price_Prediction'
2026-08-12 10:00:00 - INFO - Loaded dataset ... shape: (2736, 28)
...
2026-08-12 10:02:30 - INFO - [LightGBM_Tuned] R2: 0.8452 | MAE: $160,782 | MAPE: 18.38%
2026-08-12 10:02:30 - INFO - Best Model: 'LightGBM_Tuned' (R2: 0.8452) saved to ...

MLflow Experiment Run Complete!
Best Model: LightGBM_Tuned with R2 = 0.8452
Run 'mlflow ui' in your terminal to inspect all runs!
```

---

### 3. View Results in the MLflow Dashboard

After training, visually compare the models:

```bash
mlflow ui
```
Then open **http://127.0.0.1:5000** in your browser. You can:
* See a comparison table of all runs.
* Plot and compare metrics ($R^2, MAE, RMSE, MAPE$).
* Download model artifacts (full pipeline).

---

### 4. Run Unit Tests

Verify the correctness of every code layer:

```bash
# Run all tests with details
pytest tests/ -v

# Run with code coverage report (optional)
pytest tests/ --cov=src --cov-report=term-missing
```

**Available Tests (18 total):**
* `test_db_manager.py` (7 tests): CRUD, UPSERT, in-memory connections
* `test_scraper.py` (6 tests): Network mocking, feature cleaning, CSV saving
* `test_pipeline.py` (2 tests): Full pipeline simulation
* `test_train.py` (3 tests): Feature/target separation, preprocessor, training flow

---

## 📈 Model Performance Comparison

Results from the latest run on **2,736 samples** (Austin, TX) with a stratified 80/20 split:

| Model | Test $R^2$ ↑ | CV $R^2$ Mean ↑ | MAE ($) ↓ | RMSE ($) ↓ | MAPE (%) ↓ |
|:---|---:|---:|---:|---:|---:|
| **LightGBM (Tuned)** | **0.8452** | **0.7613** | **$160,782** | $378,573 | **18.38%** |
| LightGBM (Base) | 0.8423 | 0.7587 | $159,226 | $382,066 | 18.41% |
| Random Forest | 0.8332 | 0.7610 | $159,831 | $392,897 | 18.94% |
| XGBoost | 0.7809 | 0.7140 | $159,630 | $417,162 | 17.84% |
| Linear Regression | 0.6384 | - | $204,909 | $578,562 | 31.40% |
| Ridge Regression | 0.6372 | - | $207,931 | $579,557 | 32.12% |
| Baseline (Median) | -0.0862 | 0.0000 | $489,863 | $1,002,780 | 63.84% |

> 💡 **Note:** Metrics are computed on the original dollar scale (not log-transformed) using `TransformedTargetRegressor` with `log1p` / `expm1` functions.

---

## 🔮 Roadmap

- [ ] **FastAPI Serving:** Build a `/predict` endpoint for model serving.
- [ ] **Streamlit Dashboard:** Interactive dashboard for end-user price prediction.
- [ ] **Dockerfile:** Full containerization for easy deployment.
- [ ] **GitHub Actions CI/CD:** Automated tests, linting, and training on every push.
- [ ] **Data Drift Monitoring:** Monitor input data drift in production.
- [ ] **Feature Store:** Centralize feature definitions with Feast or Parquet files.
- [ ] **Hyperparameter Optimization:** Integrate Optuna or Hyperopt for automated search.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place! To get started:

1. **Fork** the repository.
2. Create a new branch: `git checkout -b feature/amazing-feature`
3. Commit your changes with **meaningful messages**: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a **Pull Request**.

**Code Standards:**
* Code formatted with **Black** (`black src/ tests/`)
* Imports sorted with **isort** (`isort src/ tests/`)
* Linted with **flake8**
* Tests must pass (`pytest tests/ -v`)

---

## 📄 License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

```
MIT License
Copyright (c) 2026 Mohammad Javad Majlesi
```

---

## 🙏 Acknowledgments

* [homeharvest](https://github.com/alexperrine/homeharvest) — powerful real estate scraping library.
* [MLflow](https://mlflow.org/) — complete machine learning lifecycle platform.
* [LightGBM](https://lightgbm.readthedocs.io/) — fast and accurate gradient boosting.
* [scikit-learn](https://scikit-learn.org/) — core machine learning tooling.
* The Python and Data Science community 🇮🇷

---

<div align="center">

**If you find this project useful, please consider giving it a ⭐ on GitHub!**

Made with ❤️ and Python in Iran 🇮🇷

</div>
