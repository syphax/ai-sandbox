"""
Schema Migration Script: Add detailed address fields to warehouse database

This script adds the following new fields to the warehouse database:
- street_number (string): Street number (can be alphanumeric)
- street (string): Street name
- city (string): City or town name
- state (string): State/province
- postal_code (string): Primary postal code (ZIP5 in US)
- postal_code_ext (string): Extended postal code (ZIP+4 in US)
- flag_incomplete_address (boolean): True if address is incomplete

This script should be run rarely, typically only once or when schema changes are needed.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def migrate_schema(db_path: str, backup: bool = True) -> None:
    """
    Add new address fields to the warehouse database

    Args:
        db_path: Path to the warehouse parquet database
        backup: If True, create a backup before modifying
    """
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Loading database from {db_path}")
    df = pd.read_parquet(db_path)
    print(f"Loaded {len(df)} warehouses")

    # Create backup if requested
    if backup:
        backup_path = db_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet')
        df.to_parquet(backup_path, index=False)
        print(f"Created backup at {backup_path}")

    # Check if new fields already exist
    new_fields = ['street_number', 'street', 'city', 'state', 'postal_code',
                  'postal_code_ext', 'flag_incomplete_address']

    existing_new_fields = [f for f in new_fields if f in df.columns]
    if existing_new_fields:
        print(f"Warning: Some new fields already exist: {existing_new_fields}")
        print("This script will overwrite them with empty values.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            return

    # Add new fields with empty/default values
    print("Adding new address fields...")
    df['street_number'] = ''
    df['street'] = ''
    df['city'] = ''
    df['state'] = ''
    df['postal_code'] = ''
    df['postal_code_ext'] = ''
    df['flag_incomplete_address'] = True  # Default to True, parser will set to False for complete addresses

    # Update schema_version field (add if doesn't exist)
    df['schema_version'] = '2.0'  # Version with detailed address fields
    df['schema_updated'] = datetime.utcnow().isoformat()

    # Save updated database
    print(f"Saving updated database to {db_path}")
    df.to_parquet(db_path, index=False)

    print("Schema migration complete!")
    print(f"\nNew columns added:")
    for field in new_fields:
        print(f"  - {field}")

    print(f"\nNext step: Run parse_addresses.py to populate the new address fields")


def main():
    """Run schema migration"""
    # Default path to warehouse database
    data_dir = Path(__file__).parent.parent / "data"
    db_path = data_dir / "warehouses.parquet"

    print("=" * 60)
    print("Warehouse Address Schema Migration")
    print("=" * 60)
    print()
    print("This script will add new detailed address fields to your")
    print("warehouse database. A backup will be created automatically.")
    print()

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Please run warehouse_finder.py first to create the database")
        return

    # Run migration
    migrate_schema(str(db_path), backup=True)


if __name__ == "__main__":
    main()
