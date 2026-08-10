"""
Management of the SQLite database for storing scraped US housing listings.
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "database.sqlite"


def get_connection():
    """Create a connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    """Create the listings table if it doesn't exist."""
    conn = get_connection()
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
    conn.close()
    logger.info("SQLite database initialized.")


def upsert_properties(df: pd.DataFrame) -> int:
    """
    Insert or update property listings into the database.

    Records are keyed on property_id (UNIQUE). If a record already exists,
    it is replaced with the new data (e.g., updated price), otherwise inserted.
    """
    if df is None or df.empty:
        logger.warning("Empty dataframe. Nothing to save.")
        return 0

    init_db()
    conn = get_connection()

    # Standardize column names to match the DB schema (lowercase, snake_case)
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Cross-walk from DataFrame columns to DB column names
    column_map = {
        "property_id": "property_id",
        "listing_id": "listing_id",
        "listing_status": "listing_status",
        "list_price": "list_price",
        "price": "list_price",          # fallback: homeharvest 'price' column
        "sold_price": "sold_price",
        "beds": "beds",
        "baths": "baths",
        "sqft": "sqft_living",          # fallback
        "sqft_living": "sqft_living",
        "sqft_lot": "lot_sqft",         # fallback
        "lot_sqft": "lot_sqft",
        "year_built": "year_built",
        "property_type": "property_type",
        "address": "address",
        "city": "city",
        "state": "state",
        "zip_code": "zip_code",
        "latitude": "latitude",
        "longitude": "longitude",
        "hoa_fee": "hoa_fee",
        "hoa": "hoa_fee",               # fallback
        "days_on_market": "days_on_market",
        "price_per_sqft": "price_per_sqft",
        "parking_garage": "parking_garage",
        "parking_spaces": "parking_spaces",
        "heating_type": "heating_type",
        "heating": "heating_type",      # fallback
        "cooling_type": "cooling_type",
        "cooling": "cooling_type",      # fallback
        "basement_type": "basement_type",
        "basement": "basement_type",    # fallback
        "stories": "stories",
        "listing_type": "listing_type",
        "scraped_at": "scraped_at",
    }

    # Map available columns
    available_cols = {k: v for k, v in column_map.items() if k in df.columns}
    df_mapped = df[list(available_cols.keys())].rename(columns=available_cols)

    # Remove duplicate columns (e.g., if both 'price' and 'list_price' exist)
    df_mapped = df_mapped.loc[:, ~df_mapped.columns.duplicated()]

    # Ensure text fields are stringified and numerics are numeric
    text_cols = [
        "property_id", "listing_id", "listing_status", "property_type",
        "address", "city", "state", "zip_code", "heating_type",
        "cooling_type", "basement_type", "listing_type", "scraped_at",
    ]
    for col in text_cols:
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].astype(str)

    numeric_cols = [
        "list_price", "sold_price", "beds", "baths", "sqft_living", "lot_sqft",
        "year_built", "latitude", "longitude", "hoa_fee", "days_on_market",
        "price_per_sqft", "parking_garage", "parking_spaces", "stories",
    ]
    for col in numeric_cols:
        if col in df_mapped.columns:
            df_mapped[col] = pd.to_numeric(df_mapped[col], errors="coerce")

    # Save to DB
    db_cols = list(df_mapped.columns)
    placeholders = ", ".join(["?"] * len(db_cols))
    col_names_str = ", ".join(db_cols)

    insert_sql = f"""
        INSERT OR REPLACE INTO properties ({col_names_str})
        VALUES ({placeholders})
    """

    # Convert DataFrame to list of tuples, filling NaN as None
    records = df_mapped.where(pd.notnull(df_mapped), None).values.tolist()

    cursor = conn.cursor()
    cursor.executemany(insert_sql, records)
    conn.commit()
    inserted = cursor.rowcount
    conn.close()

    logger.info(f"{len(records)} records upserted into SQLite (keyed on property_id).")
    return len(records)


def load_all_properties() -> pd.DataFrame:
    """Load all properties from the database into a DataFrame."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM properties", conn)
    conn.close()
    logger.info(f"Loaded {len(df)} records from database.")
    return df


def get_latest_scraped_at() -> str | None:
    """Get the timestamp of the most recent scrape."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(scraped_at) FROM properties")
    result = cursor.fetchone()
    conn.close()
    return result[0] if result and result[0] else None


def count_properties() -> int:
    """Count total records in the properties table."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM properties")
    count = cursor.fetchone()[0]
    conn.close()
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print(f"Total properties in DB: {count_properties()}")