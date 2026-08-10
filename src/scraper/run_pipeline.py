"""
Complete data scraping pipeline for US real estate.

Steps:
  1. Fetch listings using homeharvest (Zillow / Redfin / Realtor.com)
  2. Clean & select relevant features
  3. Save raw data to CSV (data/raw/)
  4. Store into SQLite database (data/database.sqlite)

Usage:
    python src/scraper/run_pipeline.py --location "Austin, TX"
    python src/scraper/run_pipeline.py --location "78701" --limit 300 --listing-type for_sale
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is importable even if run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.scraper.homeharvest_scraper import run_scraper
from src.utils.db_manager import upsert_properties, count_properties


def main():
    parser = argparse.ArgumentParser(description="Scrape US real estate data and store in SQLite.")
    parser.add_argument(
        "--location",
        type=str,
        default="Austin, TX",
        help="City name, zip code, or 'City, State' (default: Austin, TX)",
    )
    parser.add_argument(
        "--listing-type",
        type=str,
        default="for_sale",
        choices=["for_sale", "sold", "for_rent", "pending"],
        help="Type of listing to scrape (default: for_sale)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of listings to fetch (default: 500)",
    )
    parser.add_argument(
        "--past-days",
        type=int,
        default=365,
        help="Days back to search for sold listings (default: 365)",
    )
    parser.add_argument(
        "--no-mls-only",
        action="store_true",
        help="Include non-MLS listings (by default only MLS listings are fetched)",
    )
    parser.add_argument(
        "--listing-types",
        nargs="+",
        default=None,
        help="Multiple listing types: for_sale sold (default: for_sale only)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    listing_types = args.listing_types or [args.listing_type]

    logging.info("=" * 60)
    logging.info("US Real Estate Data Scraping Pipeline")
    logging.info("=" * 60)

    # Step 1: Fetch & clean
    df = run_scraper(
        location=args.location,
        listing_types=listing_types,
        limit=args.limit,
        past_days=args.past_days,
        mls_only=not args.no_mls_only,
        save_csv=True,
    )

    if df.empty:
        logging.error("No data fetched. Exiting.")
        sys.exit(1)

    # Step 2: Store in SQLite
    saved = upsert_properties(df)
    logging.info(f"Upserted {saved} records into SQLite.")

    total = count_properties()
    logging.info(f"Total records now in database: {total}")

    logging.info("=" * 60)
    logging.info("Pipeline completed successfully!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()