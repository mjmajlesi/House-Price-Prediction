"""
US Real Estate Data Scraper using `homeharvest` library.

This module scrapes property listings from Zillow, Redfin, and Realtor.com
for US locations (cities, zip codes, or states) and saves them to CSV
and SQLite database.

Features extracted:
- Price, Beds, Baths, Sqft (Living Area), Lot Size
- Year Built, Property Type, HOA Fees
- Address, City, State, Zip Code, Latitude, Longitude
- Listing Status (For Sale / Sold / Pending)
- Days on Market, Price History
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

try:
    from homeharvest import scrape_property
except ImportError:
    raise ImportError("Please install homeharvest: pip install homeharvest")

logger = logging.getLogger(__name__)

# Default location: Austin, TX (change as needed)
DEFAULT_LOCATION = "Austin, TX"
DEFAULT_LISTING_TYPES = ["for_sale", "sold"]
RAW_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_listings(
    location: str = DEFAULT_LOCATION,
    listing_types: List[str] = None,
    limit: int = 500,
    past_days: int = 365,
    mls_only: bool = False,
) -> pd.DataFrame:
    """
    Fetch property listings using homeharvest.

    Args:
        location: City name, zip code, or "City, State" (e.g., "Austin, TX" or "78701")
        listing_types: List of listing types to fetch. Options: "for_sale", "sold", "for_rent", "pending"
        limit: Maximum number of listings to fetch per listing_type
        past_days: For 'sold' listings, how many days back to search
        mls_only: If True, only return MLS listings (more reliable data)

    Returns:
        DataFrame with all fetched listings combined.
    """
    if listing_types is None:
        listing_types = DEFAULT_LISTING_TYPES

    all_dfs = []

    for listing_type in listing_types:
        logger.info(f"Fetching {listing_type} listings for {location} (limit={limit})...")

        try:
            df = scrape_property(
                location=location,
                listing_type=listing_type,
                limit=limit,
                past_days=past_days,
                mls_only=mls_only,
            )

            if df is not None and not df.empty:
                df["listing_type"] = listing_type
                df["scraped_at"] = datetime.now().isoformat()
                all_dfs.append(df)
                logger.info(f"  -> Fetched {len(df)} {listing_type} listings")
            else:
                logger.warning(f"  -> No {listing_type} listings found for {location}")

        except Exception as e:
            logger.error(f"Error fetching {listing_type} for {location}: {e}")

    if not all_dfs:
        logger.warning("No listings fetched for any listing type.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total listings fetched: {len(combined)}")
    return combined


def clean_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and rename relevant columns for our ML pipeline.
    homeharvest returns many columns; we keep the most predictive ones.
    """
    if df.empty:
        return df

    # Mapping from homeharvest column names to our standardized names
    column_mapping = {
        "property_id": "property_id",
        "listing_id": "listing_id",
        "status": "listing_status",  # FOR_SALE, SOLD, PENDING
        "list_price": "list_price",
        "sold_price": "sold_price",
        "beds": "beds",
        "full_baths": "baths",
        "sqft": "sqft_living",
        "lot_sqft": "lot_sqft",
        "year_built": "year_built",
        "style": "property_type",  # SINGLE_FAMILY, CONDO, TOWNHOUSE, etc.
        "formatted_address": "address",
        "city": "city",
        "state": "state",
        "zip_code": "zip_code",
        "latitude": "latitude",
        "longitude": "longitude",
        "hoa_fee": "hoa_fee",
        "days_on_mls": "days_on_market",
        "price_per_sqft": "price_per_sqft",
        "listing_type": "listing_type",
        "scraped_at": "scraped_at",
        # Optional rich features if available
        "parking_garage": "parking_garage",
        "parking_spaces": "parking_spaces",
        "heating": "heating_type",
        "cooling": "cooling_type",
        "basement": "basement_type",
        "stories": "stories",
    }

    # Select only columns that exist in the dataframe
    available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df_clean = df[list(available_cols.keys())].copy()
    df_clean = df_clean.rename(columns=available_cols)

    # Ensure numeric types
    numeric_cols = [
        "list_price", "beds", "baths", "sqft_living", "lot_sqft",
        "year_built", "hoa_fee", "days_on_market", "price_per_sqft",
        "parking_spaces", "stories", "latitude", "longitude"
    ]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Create target variable: for 'sold' listings, use actual sold price; for 'for_sale', use list price
    # We'll handle this in modeling phase. For now, keep both.
    logger.info(f"Cleaned dataframe shape: {df_clean.shape}")
    return df_clean


def save_raw_data(df: pd.DataFrame, filename: str = "us_properties_raw.csv") -> Path:
    """Save raw scraped data to CSV."""
    filepath = RAW_DATA_DIR / filename
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Raw data saved to {filepath}")
    return filepath


def run_scraper(
    location: str = DEFAULT_LOCATION,
    listing_types: List[str] = None,
    limit: int = 500,
    past_days: int = 365,
    mls_only: bool = False,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Main entry point to run the scraper pipeline.
    """
    logger.info(f"Starting scraper for location: {location}")

    # 1. Fetch data
    df_raw = fetch_listings(location, listing_types, limit, past_days, mls_only)

    if df_raw.empty:
        logger.error("No data fetched. Exiting.")
        return pd.DataFrame()

    # 2. Clean and select features
    df_clean = clean_and_select_features(df_raw)

    # 3. Save raw CSV
    if save_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"us_properties_{location.replace(', ', '_').replace(' ', '_')}_{timestamp}.csv"
        save_raw_data(df_clean, filename)

    logger.info("Scraper pipeline completed successfully.")
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Example: Scrape Austin, TX
    df = run_scraper(
        location="Austin, TX",
        listing_types=["for_sale", "sold"],
        limit=200,
        past_days=180,
        mls_only=True,
    )
    print(df.head())
    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {list(df.columns)}")