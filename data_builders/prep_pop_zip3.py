#!/usr/bin/env python3
"""
Process NHGIS ZCTA data to create ZIP3-level population dataset.

Reads ZIP code (ZCTA) level data and aggregates to ZIP3 level with:
- Total population (sum)
- Population-weighted latitude and longitude
- Count of unique ZIP5 codes in each ZIP3

Supports multiple NHGIS dataset formats:
- 2000 Census (ds146): Uses ZIP3A, FL5001, integer lat/lon format
- 2020 Census (ds258): Uses ZCTAA (truncated), U7H001, decimal lat/lon format
"""

import pandas as pd
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Field mapping for different dataset formats
DATASET_CONFIGS = {
    '2000': {
        'zip3_field': 'ZIP3A',
        'zip5_field': 'ZCTAA',
        'population_field': 'FL5001',
        'lat_field': 'INTPTLAT',
        'lon_field': 'INTPLON',  # Note: 2000 uses INTPLON (no 'T')
        'lat_lon_format': 'integer',  # Integer format needs conversion
    },
    '2020': {
        'zip3_field': None,  # Need to create from ZCTAA
        'zip5_field': 'ZCTAA',
        'population_field': 'U7H001',
        'lat_field': 'INTPTLAT',
        'lon_field': 'INTPTLON',  # Note: 2020 uses INTPTLON (with 'T')
        'lat_lon_format': 'decimal',  # Already in decimal format
    }
}


def detect_dataset_year(df):
    """
    Auto-detect which dataset format we're working with based on column names.

    Args:
        df: Input DataFrame

    Returns:
        str: Year identifier ('2000' or '2020')
    """
    if 'FL5001' in df.columns:
        return '2000'
    elif 'U7H001' in df.columns:
        return '2020'
    else:
        raise ValueError("Unable to detect dataset format. Missing expected population fields.")


def convert_lat_lon_integer(value):
    """
    Convert integer lat/lon format to decimal degrees (2000 format).

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


def convert_lat_lon_decimal(value):
    """
    Convert decimal lat/lon format (2020 format).

    Format: Already in decimal degrees like "+18.1805555"

    Examples:
        +18.1805555 -> 18.1805555
        -066.7499615 -> -66.7499615
    """
    try:
        return float(value)
    except Exception as e:
        logger.warning(f"Error converting lat/lon value {value}: {e}")
        return None


def process_data(input_file, output_file, year=None):
    """
    Process ZCTA data and aggregate to ZIP3 level.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        year: Optional year identifier ('2000' or '2020'). If None, auto-detect.
    """
    logger.info(f"Reading input file: {input_file}")

    # Read the data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df):,} records")

    # Auto-detect year if not provided
    if year is None:
        year = detect_dataset_year(df)
        logger.info(f"Auto-detected dataset format: {year}")
    else:
        logger.info(f"Using specified dataset format: {year}")

    # Get configuration for this dataset
    config = DATASET_CONFIGS[year]

    # Create ZIP3 field if it doesn't exist
    if config['zip3_field'] is None:
        logger.info(f"Creating ZIP3 from {config['zip5_field']}")
        df['zip3'] = df[config['zip5_field']].astype(str).str[:3]
        zip3_field = 'zip3'
    else:
        zip3_field = config['zip3_field']

    # Convert lat/lon to decimal format
    logger.info("Converting latitude and longitude to decimal degrees...")
    if config['lat_lon_format'] == 'integer':
        df['lat_decimal'] = df[config['lat_field']].apply(convert_lat_lon_integer)
        df['lon_decimal'] = df[config['lon_field']].apply(convert_lat_lon_integer)
    else:  # decimal format
        df['lat_decimal'] = df[config['lat_field']].apply(convert_lat_lon_decimal)
        df['lon_decimal'] = df[config['lon_field']].apply(convert_lat_lon_decimal)

    # Remove rows with invalid lat/lon
    initial_count = len(df)
    df = df.dropna(subset=['lat_decimal', 'lon_decimal'])
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df)} rows with invalid lat/lon")

    # Calculate weighted lat/lon (weight by population)
    pop_field = config['population_field']
    df['weighted_lat'] = df['lat_decimal'] * df[pop_field]
    df['weighted_lon'] = df['lon_decimal'] * df[pop_field]

    logger.info("Aggregating to ZIP3 level...")

    # Group by ZIP3 and aggregate
    zip3_data = df.groupby(zip3_field).agg({
        pop_field: 'sum',  # Total population
        'weighted_lat': 'sum',
        'weighted_lon': 'sum',
        config['zip5_field']: 'nunique'  # Count unique ZIP5 codes
    }).reset_index()

    # Calculate population-weighted averages
    zip3_data['latitude'] = zip3_data['weighted_lat'] / zip3_data[pop_field]
    zip3_data['longitude'] = zip3_data['weighted_lon'] / zip3_data[pop_field]

    # Rename and select columns
    zip3_data = zip3_data.rename(columns={
        zip3_field: 'zip3',
        pop_field: 'population',
        config['zip5_field']: 'cnt_zip5'
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

    return zip3_data


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Process NHGIS ZCTA data to create ZIP3-level population dataset.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        help='Input CSV file path (relative to data/ directory or absolute path)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output CSV file path (relative to data/ directory or absolute path)'
    )
    parser.add_argument(
        '-y', '--year',
        type=str,
        choices=['2000', '2020'],
        help='Dataset year (auto-detected if not specified)'
    )

    args = parser.parse_args()

    # Determine input file
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.is_absolute():
            input_path = Path(__file__).parent.parent / 'data' / args.input_file
    else:
        # Default to 2000 data
        input_path = Path(__file__).parent.parent / 'data' / 'nhgis0002_ds146_2000_zcta.csv'

    # Determine output file
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path(__file__).parent.parent / 'data' / args.output
    else:
        # Auto-generate output filename based on input
        if '2000' in input_path.name:
            output_filename = 'zip3_pop_2000.csv'
        elif '2020' in input_path.name:
            output_filename = 'zip3_pop_2020.csv'
        else:
            output_filename = 'zip3_pop.csv'
        output_path = Path(__file__).parent.parent / 'data' / output_filename

    # Process the data
    process_data(input_path, output_path, args.year)


if __name__ == '__main__':
    main()
