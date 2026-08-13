"""
Training Script with MLflow Experiment Tracking for US House Price Prediction.

This script executes the model training pipeline:
1. Loads preprocessed housing dataset.
2. Sets up feature transformation pipelines (Imputer + Scaler + OneHotEncoder).
3. Evaluates multiple algorithms (Linear, Ridge, RandomForest, XGBoost, LightGBM).
4. Logs parameters, metrics (R2, MAE, RMSE, MAPE), and models into MLflow.
5. Saves the best model bundle to joblib format for deployment.

Usage:
    python src/train.py
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants & Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_housing_data.csv"
MODEL_SAVE_DIR = PROJECT_ROOT / "notebooks" / "saved_models"
EXPERIMENT_NAME = "House_Price_Prediction"


def load_data(filepath: Path) -> pd.DataFrame:
    """Load preprocessed housing dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Processed dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset from {filepath} with shape: {df.shape}")
    return df


def prepare_features_target(df: pd.DataFrame):
    """Separate features and target, clean leakage columns."""
    target_col = "list_price"
    leakage_cols = [
        "list_price", "log_list_price", "price_per_sqft",
        "days_on_market", "listing_status", "city", "listing_type"
    ]

    features_df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    X = features_df.copy()
    y = df[target_col]

    if "zip_code" in X.columns:
        X["zip_code"] = X["zip_code"].astype(str)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build column transformer for numeric and categorical columns."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )
    return preprocessor, num_cols, cat_cols


def run_training_experiment():
    """Main function to run training and MLflow tracking."""
    # Set MLflow experiment name
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow Experiment set to: '{EXPERIMENT_NAME}'")

    # 1. Load Data
    df = load_data(DATA_PATH)
    X, y = prepare_features_target(df)

    # 2. Train-Test Split (Stratified by price quantiles)
    # For small datasets, reduce the number of bins so each stratum has >= 2 samples.
    n_samples = len(y)
    n_bins = 10 if n_samples >= 100 else max(2, n_samples // 10)
    price_bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=price_bins
    )
    logger.info(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Build Preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # 4. Define Candidate Models
    models = {
        "Linear_Regression": LinearRegression(),
        "Ridge_Regression": Ridge(alpha=10.0),
        "Random_Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1
        ),
        "LightGBM_Tuned": LGBMRegressor(
            n_estimators=350, learning_rate=0.03, max_depth=6, subsample=0.8, random_state=42, n_jobs=-1, verbose=-1
        ),
    }

    best_r2 = -float("inf")
    best_pipeline_bundle = None
    best_model_name = ""

    # 5. Train & Evaluate Models with MLflow Logging
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model_obj in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"--- Training Model: {model_name} ---")

            # Wrap inside TransformedTargetRegressor for log1p transform
            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", TransformedTargetRegressor(
                    regressor=model_obj,
                    func=np.log1p,
                    inverse_func=np.expm1
                ))
            ])

            # Fit model
            full_pipeline.fit(X_train, y_train)

            # Predict on test set
            y_pred = full_pipeline.predict(X_test)

            # Compute Evaluation Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            # Cross-validation
            cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
            cv_r2_mean = cv_scores.mean()

            # --- Log to MLflow ---
            # 1. Parameters
            if hasattr(model_obj, "get_params"):
                params = {f"model_param_{k}": v for k, v in model_obj.get_params().items() if isinstance(v, (int, float, str, bool))}
                mlflow.log_params(params)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("log_transformed_target", True)

            # 2. Metrics
            mlflow.log_metrics({
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "CV_R2_Mean": cv_r2_mean
            })

            # 3. Log Model Artifact
            mlflow.sklearn.log_model(full_pipeline, artifact_path="model", serialization_format="pickle")

            logger.info(f"[{model_name}] R2: {r2:.4f} | MAE: ${mae:,.0f} | MAPE: {mape:.2f}%")

            # Track Best Model
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name
                best_pipeline_bundle = {
                    "model": full_pipeline.named_steps["regressor"].regressor_,
                    "feature_pipeline": full_pipeline.named_steps["preprocessor"],
                    "full_pipeline": full_pipeline,
                    "features": list(X.columns),
                    "num_cols": num_cols,
                    "cat_cols": cat_cols,
                    "is_log_target": True
                }

    # 6. Save Best Model to Joblib Bundle
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_SAVE_DIR / "house_price_model.joblib"
    joblib.dump(best_pipeline_bundle, best_model_path)
    logger.info(f"Best Model: '{best_model_name}' (R2: {best_r2:.4f}) saved to {best_model_path}")

    print("\n" + "="*60)
    print(f"MLflow Experiment Run Complete!")
    print(f"Best Model: {best_model_name} with R2 = {best_r2:.4f}")
    print(f"Run 'mlflow ui' in your terminal to inspect all runs!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_training_experiment()
