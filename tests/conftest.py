import pytest
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path


@pytest.fixture
def sample_property_df():
    """Returns a sample DataFrame mimicking raw scraped homeharvest data."""
    return pd.DataFrame({
        "property_id": ["P001", "P002", "P003"],
        "listing_id": ["L001", "L002", "L003"],
        "status": ["FOR_SALE", "SOLD", "FOR_SALE"],
        "list_price": [550000, 420000, 890000],
        "sold_price": [None, 415000, None],
        "beds": [3, 2, 4],
        "full_baths": [2.0, 1.5, 3.5],
        "sqft": [1800, 1200, 2800],
        "lot_sqft": [5000, 3000, 8000],
        "year_built": [2010, 1995, 2020],
        "style": ["SINGLE_FAMILY", "CONDO", "SINGLE_FAMILY"],
        "formatted_address": ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
        "city": ["Austin", "Austin", "Austin"],
        "state": ["TX", "TX", "TX"],
        "zip_code": ["78701", "78702", "78703"],
        "latitude": [30.2672, 30.2700, 30.2800],
        "longitude": [-97.7431, -97.7400, -97.7500],
        "hoa_fee": [150, 250, 0],
        "days_on_mls": [15, 45, 5],
        "price_per_sqft": [305.5, 350.0, 317.8],
        "listing_type": ["for_sale", "sold", "for_sale"],
        "scraped_at": ["2026-08-10T10:00:00", "2026-08-10T10:00:00", "2026-08-10T10:00:00"],
        "parking_garage": [1, 0, 2],
        "parking_spaces": [2, 1, 3],
        "heating": ["Central", "Electric", "Central"],
        "cooling": ["Central", "Central", "Central"],
        "basement": ["None", "None", "Finished"],
        "stories": [2, 1, 2],
    })


@pytest.fixture
def empty_df():
    """Returns an empty DataFrame."""
    return pd.DataFrame()


class _NoCloseConnection:
    """Wrapper around sqlite3.Connection that disables close() so db_manager code
    can call conn.close() without killing the shared in-memory database."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    # --- forward everything except close ---
    def close(self):
        pass  # intentionally no-op during tests

    def cursor(self):
        return self._conn.cursor()

    def commit(self):
        return self._conn.commit()

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.fixture
def memory_db():
    """
    Creates an in-memory SQLite connection initialized with the properties schema.
    Returns a wrapper that keeps the connection alive even if db_manager calls close().
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id TEXT UNIQUE,
            listing_id TEXT,
            listing_status TEXT,
            list_price REAL,
            sold_price REAL,
            beds INTEGER,
            baths REAL,
            sqft_living REAL,
            lot_sqft REAL,
            year_built INTEGER,
            property_type TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            zip_code TEXT,
            latitude REAL,
            longitude REAL,
            hoa_fee REAL,
            days_on_market INTEGER,
            price_per_sqft REAL,
            parking_garage INTEGER,
            parking_spaces INTEGER,
            heating_type TEXT,
            cooling_type TEXT,
            basement_type TEXT,
            stories REAL,
            listing_type TEXT,
            scraped_at TEXT
        )
    """)
    conn.commit()
    safe_conn = _NoCloseConnection(conn)
    yield safe_conn
    conn.close()  # actually close the real connection at teardown
