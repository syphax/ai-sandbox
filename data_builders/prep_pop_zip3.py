#!/usr/bin/env python3
"""
Process NHGIS ZCTA data to create ZIP3-level population dataset.

Reads ZIP code (ZCTA) level data and aggregates to ZIP3 level with:
- Total population (sum)
- Population-weighted latitude and longitude
- Count of unique ZIP5 codes in each ZIP3
"""

import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_lat_lon(value):
    """
    Convert integer lat/lon format to decimal degrees.

    Format: ±DDDDDDDDD where first 2 (lat) or 3 (lon) digits are integer degrees,
    rest are decimal fraction.

    Examples:
        +42457201 -> 42.457201
        -122345678 -> -122.345678
    """
    try:
        value_str = str(value)

        # Handle sign
        if value_str.startswith('+') or value_str.startswith('-'):
            sign = 1 if value_str[0] == '+' else -1
            digits = value_str[1:]
        else:
            sign = 1
            digits = value_str

        # For latitude: 2 digits integer, rest decimal
        # For longitude: 3 digits integer, rest decimal
        # We'll detect based on length
        if len(digits) == 8:  # Latitude format
            integer_part = int(digits[:2])
            decimal_part = int(digits[2:])
            result = sign * (integer_part + decimal_part / 1000000)
        elif len(digits) == 9:  # Longitude format
            integer_part = int(digits[:3])
            decimal_part = int(digits[3:])
            result = sign * (integer_part + decimal_part / 1000000)
        else:
            logger.warning(f"Unexpected lat/lon format: {value}")
            return None

        return result
    except Exception as e:
        logger.warning(f"Error converting lat/lon value {value}: {e}")
        return None


def main():
    """Main processing function."""

    # Define paths
    input_file = Path(__file__).parent.parent / 'data' / 'nhgis0002_ds146_2000_zcta.csv'
    output_file = Path(__file__).parent.parent / 'data' / 'zip3_pop_2000.csv'

    logger.info(f"Reading input file: {input_file}")

    # Read the data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df):,} records")

    # Convert lat/lon to decimal format
    logger.info("Converting latitude and longitude to decimal degrees...")
    df['lat_decimal'] = df['INTPTLAT'].apply(convert_lat_lon)
    df['lon_decimal'] = df['INTPTLON'].apply(convert_lat_lon)

    # Remove rows with invalid lat/lon
    initial_count = len(df)
    df = df.dropna(subset=['lat_decimal', 'lon_decimal'])
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df)} rows with invalid lat/lon")

    # Calculate weighted lat/lon (weight by population)
    df['weighted_lat'] = df['lat_decimal'] * df['FL5001']
    df['weighted_lon'] = df['lon_decimal'] * df['FL5001']

    logger.info("Aggregating to ZIP3 level...")

    # Group by ZIP3 and aggregate
    zip3_data = df.groupby('ZIP3A').agg({
        'FL5001': 'sum',  # Total population
        'weighted_lat': 'sum',
        'weighted_lon': 'sum',
        'ZCTA5A': 'nunique'  # Count unique ZIP5 codes (assuming ZCTA5A is the ZIP5 column)
    }).reset_index()

    # Calculate population-weighted averages
    zip3_data['latitude'] = zip3_data['weighted_lat'] / zip3_data['FL5001']
    zip3_data['longitude'] = zip3_data['weighted_lon'] / zip3_data['FL5001']

    # Rename and select columns
    zip3_data = zip3_data.rename(columns={
        'ZIP3A': 'zip3',
        'FL5001': 'population',
        'ZCTA5A': 'cnt_zip5'
    })

    # Final output columns
    zip3_data = zip3_data[['zip3', 'population', 'latitude', 'longitude', 'cnt_zip5']]

    # Sort by ZIP3
    zip3_data = zip3_data.sort_values('zip3')

    logger.info(f"Created {len(zip3_data):,} ZIP3 records")
    logger.info(f"Total population: {zip3_data['population'].sum():,.0f}")

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    zip3_data.to_csv(output_file, index=False)
    logger.info(f"Saved output to: {output_file}")

    # Display sample
    print("\nSample of output data:")
    print(zip3_data.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
