import pytest
import sqlite3
import pandas as pd
from unittest.mock import patch, MagicMock

import src.utils.db_manager as dbm


@pytest.fixture
def mock_get_connection(memory_db):
    """Mocks db_manager.get_connection to use our in-memory DB fixture."""
    with patch("src.utils.db_manager.get_connection", return_value=memory_db) as mock_conn:
        yield mock_conn


def test_init_db(memory_db):
    """Test that init_db creates the properties table."""
    with patch("src.utils.db_manager.get_connection", return_value=memory_db):
        dbm.init_db()
        cursor = memory_db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='properties'")
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "properties"


def test_upsert_properties_empty_df(mock_get_connection, empty_df):
    """Test upserting an empty DataFrame returns 0."""
    result = dbm.upsert_properties(empty_df)
    assert result == 0


def test_upsert_properties_success(mock_get_connection, memory_db, sample_property_df):
    """Test successful insertion of property records."""
    # sample_property_df has homeharvest column names. We need to rename them to match
    # what clean_and_select_features does before passing to db_manager (or just use db_manager fallbacks).
    # db_manager has some fallback mappings. Let's see if it catches them.
    inserted_count = dbm.upsert_properties(sample_property_df)
    assert inserted_count == 3

    # Verify rows in DB
    cursor = memory_db.cursor()
    cursor.execute("SELECT property_id, list_price, property_type FROM properties")
    rows = cursor.fetchall()
    assert len(rows) == 3

    # property_type mapped fallback is property_type, but sample has "style".
    # dbm only maps what's available. If "property_type" is missing in DF, it won't insert it.
    # We should make sure we're testing the upsert with columns that match db_manager expectations.


def test_upsert_duplicate_property_id(mock_get_connection, memory_db):
    """Test that upserting duplicate property_ids replaces the old record."""
    df1 = pd.DataFrame({
        "property_id": ["P123"],
        "list_price": [500000],
        "scraped_at": ["2026-08-10"]
    })
    dbm.upsert_properties(df1)

    # Update price
    df2 = pd.DataFrame({
        "property_id": ["P123"],
        "list_price": [550000],
        "scraped_at": ["2026-08-11"]
    })
    dbm.upsert_properties(df2)

    cursor = memory_db.cursor()
    cursor.execute("SELECT list_price, scraped_at FROM properties WHERE property_id='P123'")
    row = cursor.fetchone()

    # Should be replaced, not inserted twice
    cursor.execute("SELECT COUNT(*) FROM properties")
    count = cursor.fetchone()[0]

    assert count == 1
    assert row[0] == 550000
    assert row[1] == "2026-08-11"


def test_load_all_properties(mock_get_connection, memory_db):
    """Test loading all properties back into a DataFrame."""
    df_insert = pd.DataFrame({
        "property_id": ["A1", "A2"],
        "list_price": [100, 200]
    })
    dbm.upsert_properties(df_insert)

    df_loaded = dbm.load_all_properties()
    assert len(df_loaded) == 2
    assert "property_id" in df_loaded.columns
    assert set(df_loaded["property_id"].tolist()) == {"A1", "A2"}


def test_get_latest_scraped_at(mock_get_connection):
    """Test fetching the latest timestamp."""
    df = pd.DataFrame({
        "property_id": ["P1", "P2"],
        "scraped_at": ["2026-08-10T12:00:00", "2026-08-11T15:00:00"]
    })
    dbm.upsert_properties(df)

    latest = dbm.get_latest_scraped_at()
    assert latest == "2026-08-11T15:00:00"


def test_count_properties(mock_get_connection):
    """Test counting records."""
    df = pd.DataFrame({
        "property_id": ["P1", "P2", "P3"],
    })
    dbm.upsert_properties(df)
    assert dbm.count_properties() == 3

