import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

import src.scraper.homeharvest_scraper as scraper


@patch("src.scraper.homeharvest_scraper.scrape_property")
def test_fetch_listings_success(mock_scrape, sample_property_df):
    """Test fetching listings with mock success."""
    mock_scrape.return_value = sample_property_df

    # Fetch sold listings
    result = scraper.fetch_listings(
        location="Austin, TX",
        listing_types=["sold"],
        limit=5,
        past_days=30
    )

    assert not result.empty
    assert len(result) == 3
    # Listing type column should have been injected
    assert "listing_type" in result.columns
    assert "scraped_at" in result.columns
    assert (result["listing_type"] == "sold").all()
    mock_scrape.assert_called_once_with(
        location="Austin, TX",
        listing_type="sold",
        limit=5,
        past_days=30,
        mls_only=False
    )


@patch("src.scraper.homeharvest_scraper.scrape_property")
def test_fetch_listings_empty_or_error(mock_scrape):
    """Test fetch listings when scraper returns empty or throws an exception."""
    # Scenario A: empty DF
    mock_scrape.return_value = pd.DataFrame()
    result = scraper.fetch_listings()
    assert result.empty

    # Scenario B: throws exception
    mock_scrape.side_effect = Exception("API connection timed out")
    result_err = scraper.fetch_listings()
    assert result_err.empty


def test_clean_and_select_features(sample_property_df):
    """Test mapping, renaming, and type casting features."""
    cleaned = scraper.clean_and_select_features(sample_property_df)

    assert not cleaned.empty
    assert cleaned.shape[0] == 3

    # Check renamed columns
    assert "listing_status" in cleaned.columns  # mapped from 'status'
    assert "baths" in cleaned.columns           # mapped from 'full_baths'
    assert "sqft_living" in cleaned.columns     # mapped from 'sqft'
    assert "days_on_market" in cleaned.columns  # mapped from 'days_on_mls'
    assert "property_type" in cleaned.columns   # mapped from 'style'
    assert "address" in cleaned.columns         # mapped from 'formatted_address'

    # Check data type parsing (numeric conversion)
    assert cleaned["list_price"].dtype in ["float64", "int64"]
    assert cleaned["beds"].dtype in ["float64", "int64"]


def test_clean_and_select_features_empty(empty_df):
    """Test clean features with empty DataFrame."""
    cleaned = scraper.clean_and_select_features(empty_df)
    assert cleaned.empty


def test_save_raw_data(tmp_path, sample_property_df):
    """Test saving raw data to CSV."""
    with patch("src.scraper.homeharvest_scraper.RAW_DATA_DIR", tmp_path):
        filename = "test_raw.csv"
        filepath = scraper.save_raw_data(sample_property_df, filename)

        assert filepath.exists()
        df_read = pd.read_csv(filepath)
        assert len(df_read) == len(sample_property_df)
        assert list(df_read.columns) == list(sample_property_df.columns)


@patch("src.scraper.homeharvest_scraper.fetch_listings")
@patch("src.scraper.homeharvest_scraper.clean_and_select_features")
@patch("src.scraper.homeharvest_scraper.save_raw_data")
def test_run_scraper(mock_save, mock_clean, mock_fetch, sample_property_df):
    """Test run_scraper orchestration."""
    mock_fetch.return_value = sample_property_df
    mock_clean.return_value = sample_property_df
    mock_save.return_value = Path("test_file.csv")

    res = scraper.run_scraper(
        location="Austin, TX",
        listing_types=["for_sale"],
        limit=10,
        past_days=100,
        mls_only=True,
        save_csv=True
    )

    mock_fetch.assert_called_once_with("Austin, TX", ["for_sale"], 10, 100, True)
    mock_clean.assert_called_once_with(sample_property_df)
    mock_save.assert_called_once()
    assert res is not None
    assert len(res) == 3
