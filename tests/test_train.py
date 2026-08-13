import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

import src.train as trainer


@pytest.fixture
def mock_processed_df():
    """Returns a larger mock DataFrame matching processed housing dataset schema to allow quantile splitting."""
    # We will generate 30 records (3 per bin if q=10)
    data = {
        "list_price": [300000, 300000, 300000, 450000, 450000, 450000, 600000, 600000, 600000, 750000,
                       750000, 750000, 900000, 900000, 900000, 1200000, 1200000, 1200000, 1500000, 1500000,
                       1500000, 2000000, 2000000, 2000000, 500000, 500000, 500000, 650000, 650000, 650000],
        "beds": [2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 3, 3, 3, 4, 4, 4],
        "baths": [1.0] * 30,
        "sqft_living": [1000 + i * 100 for i in range(30)],
        "lot_sqft": [3000] * 30,
        "year_built": [2000] * 30,
        "zip_code": ["78701"] * 30,
        "latitude": [30.26] * 30,
        "longitude": [-97.74] * 30,
        "hoa_fee": [100] * 30,
        "parking_garage": [2] * 30,
        "stories": [2] * 30,
        "house_age": [26] * 30,
        "baths_per_bed": [0.5] * 30,
        "total_rooms": [5] * 30,
        "sqft_per_bed": [500] * 30,
        "log_sqft_living": [7.3] * 30,
        "property_type_LAND": [0] * 30,
        "property_type_MOBILE": [0] * 30,
        "property_type_MULTI_FAMILY": [0] * 30,
        "property_type_SINGLE_FAMILY": [1] * 30,
        "property_type_TOWNHOMES": [0] * 30
    }
    return pd.DataFrame(data)


def test_prepare_features_target(mock_processed_df):
    """Test feature and target separation."""
    X, y = trainer.prepare_features_target(mock_processed_df)

    assert "list_price" not in X.columns
    assert len(y) == len(mock_processed_df)
    assert X["zip_code"].dtype in ["object", "string"]


def test_build_preprocessor(mock_processed_df):
    """Test building column transformer."""
    X, _ = trainer.prepare_features_target(mock_processed_df)
    preprocessor, num_cols, cat_cols = trainer.build_preprocessor(X)

    assert "zip_code" in cat_cols
    assert "sqft_living" in num_cols


@patch("src.train.load_data")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metrics")
@patch("mlflow.sklearn.log_model")
@patch("joblib.dump")
def test_run_training_experiment(
    mock_joblib_dump,
    mock_log_model,
    mock_log_metrics,
    mock_log_params,
    mock_start_run,
    mock_set_experiment,
    mock_load_data,
    mock_processed_df
):
    """Test full training experiment workflow with mocked MLflow and file saving."""
    mock_load_data.return_value = mock_processed_df

    trainer.run_training_experiment()

    mock_set_experiment.assert_called_once_with("House_Price_Prediction")
    assert mock_start_run.call_count >= 5  # Evaluates 5 candidate models
    mock_log_metrics.assert_called()
    mock_log_params.assert_called()
    mock_joblib_dump.assert_called_once()
