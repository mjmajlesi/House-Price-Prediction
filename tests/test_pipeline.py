import pytest
import sys
import pandas as pd
from unittest.mock import patch, MagicMock

import src.scraper.run_pipeline as pipeline


@patch("src.scraper.run_pipeline.run_scraper")
@patch("src.scraper.run_pipeline.upsert_properties")
@patch("src.scraper.run_pipeline.count_properties")
@patch("sys.argv", ["run_pipeline.py", "--location", "Dallas, TX", "--limit", "10", "--listing-type", "for_sale"])
def test_pipeline_main(mock_count, mock_upsert, mock_run_scraper, sample_property_df):
    """Test the complete end-to-end scraper pipeline execution logic."""
    mock_run_scraper.return_value = sample_property_df
    mock_upsert.return_value = 3
    mock_count.return_value = 100

    # Execute main
    pipeline.main()

    # Verify calls
    mock_run_scraper.assert_called_once_with(
        location="Dallas, TX",
        listing_types=["for_sale"],
        limit=10,
        past_days=365,
        mls_only=True,
        save_csv=True
    )
    mock_upsert.assert_called_once_with(sample_property_df)
    mock_count.assert_called_once()


@patch("src.scraper.run_pipeline.run_scraper")
@patch("sys.argv", ["run_pipeline.py", "--location", "Nowhere"])
def test_pipeline_main_empty_result(mock_run_scraper):
    """Test the pipeline exits correctly when no data is fetched."""
    mock_run_scraper.return_value = pd.DataFrame()

    with pytest.raises(SystemExit) as e:
        pipeline.main()

    assert e.value.code == 1
    mock_run_scraper.assert_called_once()
