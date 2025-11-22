"""
Address Parser Script: Parse warehouse addresses into structured fields

This script parses the 'address' field in the warehouse database and populates
the following structured fields:
- street_number: Street number (string, can be alphanumeric)
- street: Street name
- city: City or town name
- state: State/province
- postal_code: Primary postal code (ZIP5 in US)
- postal_code_ext: Extended postal code (ZIP+4 in US)
- flag_incomplete_address: True if any required fields are missing

The script supports two modes:
1. Incremental mode (default): Only parse addresses where structured fields are blank
2. Force mode: Re-parse all addresses, overwriting existing structured data
"""

import pandas as pd
import re
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, Optional


class AddressParser:
    """Parse addresses into structured components"""

    # US state abbreviations for validation
    US_STATES = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'DC', 'PR', 'VI', 'GU', 'AS', 'MP'
    }

    def parse(self, address: str) -> Dict[str, str]:
        """
        Parse an address string into structured components

        Args:
            address: Address string (comma-separated components)

        Returns:
            Dictionary with parsed address fields
        """
        result = {
            'street_number': '',
            'street': '',
            'city': '',
            'state': '',
            'postal_code': '',
            'postal_code_ext': '',
            'flag_incomplete_address': True
        }

        if not address or not isinstance(address, str):
            return result

        # Split address by commas and strip whitespace
        parts = [p.strip() for p in address.split(',')]

        if len(parts) == 0:
            return result

        # Pattern 1: Number, Street, City, State, ZIP
        # Example: "90, Huntoon Memorial Highway, Leicester, MA, 01524"
        if len(parts) >= 3:
            result.update(self._parse_full_address(parts))
        # Pattern 2: Just street number and street
        # Example: "90, Huntoon Memorial Highway"
        elif len(parts) == 2:
            result.update(self._parse_partial_address(parts))
        # Pattern 3: Single component
        else:
            result.update(self._parse_single_component(parts[0]))

        # Check if address is complete
        required_fields = ['street_number', 'street', 'city', 'state', 'postal_code']
        result['flag_incomplete_address'] = not all(result.get(f) for f in required_fields)

        return result

    def _parse_full_address(self, parts: list) -> Dict[str, str]:
        """Parse a full address with multiple components"""
        result = {
            'street_number': '',
            'street': '',
            'city': '',
            'state': '',
            'postal_code': '',
            'postal_code_ext': ''
        }

        # Last part: Try to extract ZIP code
        if parts:
            last_part = parts[-1]
            postal_info = self._extract_postal_code(last_part)
            if postal_info['postal_code']:
                result.update(postal_info)
                parts = parts[:-1]  # Remove ZIP from parts

        # Second to last: Try to extract state
        if parts:
            state = self._extract_state(parts[-1])
            if state:
                result['state'] = state
                parts = parts[:-1]  # Remove state from parts

        # Third to last (or second to last if no state): City
        if parts:
            result['city'] = parts[-1]
            parts = parts[:-1]

        # Remaining parts: Street number and street
        if len(parts) >= 2:
            # First part is likely street number
            result['street_number'] = parts[0]
            # Rest is street name
            result['street'] = ', '.join(parts[1:])
        elif len(parts) == 1:
            # Try to split street number from street
            street_info = self._split_street_number(parts[0])
            result.update(street_info)

        return result

    def _parse_partial_address(self, parts: list) -> Dict[str, str]:
        """Parse a partial address (usually just number and street)"""
        result = {
            'street_number': '',
            'street': '',
            'city': '',
            'state': '',
            'postal_code': '',
            'postal_code_ext': ''
        }

        if len(parts) >= 1:
            # First part might be street number alone, or number + street
            if len(parts) == 2:
                result['street_number'] = parts[0]
                result['street'] = parts[1]
            else:
                # Try to split
                street_info = self._split_street_number(parts[0])
                result.update(street_info)

        return result

    def _parse_single_component(self, component: str) -> Dict[str, str]:
        """Parse a single address component"""
        result = {
            'street_number': '',
            'street': '',
            'city': '',
            'state': '',
            'postal_code': '',
            'postal_code_ext': ''
        }

        # Try to extract what we can
        street_info = self._split_street_number(component)
        result.update(street_info)

        return result

    def _split_street_number(self, text: str) -> Dict[str, str]:
        """
        Split street number from street name

        Args:
            text: Text that may contain street number and name

        Returns:
            Dict with street_number and street
        """
        result = {'street_number': '', 'street': ''}

        # Pattern: Starts with number (possibly alphanumeric like "123A")
        match = re.match(r'^(\d+[A-Za-z]?)\s+(.+)$', text.strip())
        if match:
            result['street_number'] = match.group(1)
            result['street'] = match.group(2)
        else:
            # No clear street number, put everything in street
            result['street'] = text.strip()

        return result

    def _extract_state(self, text: str) -> Optional[str]:
        """
        Extract state abbreviation from text

        Args:
            text: Text that may contain state abbreviation

        Returns:
            State abbreviation or None
        """
        # Look for 2-letter state code
        text = text.strip().upper()
        if text in self.US_STATES:
            return text

        # Try to find state code within text
        for state in self.US_STATES:
            if re.search(r'\b' + state + r'\b', text):
                return state

        return None

    def _extract_postal_code(self, text: str) -> Dict[str, str]:
        """
        Extract postal code from text

        Args:
            text: Text that may contain postal code

        Returns:
            Dict with postal_code and postal_code_ext
        """
        result = {'postal_code': '', 'postal_code_ext': ''}

        # Pattern: 5 digits, optionally followed by -4 digits
        match = re.search(r'\b(\d{5})(?:-(\d{4}))?\b', text)
        if match:
            result['postal_code'] = match.group(1)
            if match.group(2):
                result['postal_code_ext'] = match.group(2)

        return result


def parse_addresses(db_path: str, force: bool = False, backup: bool = True) -> None:
    """
    Parse addresses in the warehouse database

    Args:
        db_path: Path to the warehouse parquet database
        force: If True, re-parse all addresses; if False, only parse blank ones
        backup: If True, create a backup before modifying
    """
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Loading database from {db_path}")
    df = pd.read_parquet(db_path)
    print(f"Loaded {len(df)} warehouses")

    # Check if new address fields exist
    required_fields = ['street_number', 'street', 'city', 'state',
                      'postal_code', 'postal_code_ext', 'flag_incomplete_address']

    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        print(f"Error: Required address fields missing: {missing_fields}")
        print("Please run migrate_address_schema.py first")
        return

    # Create backup if requested
    if backup:
        backup_path = db_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet')
        df.to_parquet(backup_path, index=False)
        print(f"Created backup at {backup_path}")

    # Determine which addresses to parse
    if force:
        print("Force mode: Re-parsing all addresses")
        to_parse_mask = pd.Series([True] * len(df))
    else:
        print("Incremental mode: Parsing only addresses with blank structured fields")
        # Parse if any of the key fields are blank
        to_parse_mask = (
            (df['street_number'] == '') |
            (df['street'] == '') |
            (df['city'] == '') |
            (df['state'] == '') |
            (df['postal_code'] == '')
        )

    num_to_parse = to_parse_mask.sum()
    print(f"Addresses to parse: {num_to_parse} out of {len(df)}")

    if num_to_parse == 0:
        print("No addresses to parse. Use --force to re-parse all addresses.")
        return

    # Parse addresses
    parser = AddressParser()
    print("Parsing addresses...")

    parsed_count = 0
    complete_count = 0
    incomplete_count = 0

    for idx in df[to_parse_mask].index:
        address = df.at[idx, 'address']
        parsed = parser.parse(address)

        # Update structured fields
        df.at[idx, 'street_number'] = parsed['street_number']
        df.at[idx, 'street'] = parsed['street']
        df.at[idx, 'city'] = parsed['city']
        df.at[idx, 'state'] = parsed['state']
        df.at[idx, 'postal_code'] = parsed['postal_code']
        df.at[idx, 'postal_code_ext'] = parsed['postal_code_ext']
        df.at[idx, 'flag_incomplete_address'] = parsed['flag_incomplete_address']

        parsed_count += 1
        if parsed['flag_incomplete_address']:
            incomplete_count += 1
        else:
            complete_count += 1

        # Progress indicator
        if parsed_count % 100 == 0:
            print(f"  Parsed {parsed_count}/{num_to_parse} addresses...")

    print(f"Parsed {parsed_count} addresses")
    print(f"  Complete addresses: {complete_count}")
    print(f"  Incomplete addresses: {incomplete_count}")

    # Update parsing metadata
    df['addresses_parsed_at'] = datetime.utcnow().isoformat()

    # Save updated database
    print(f"Saving updated database to {db_path}")
    df.to_parquet(db_path, index=False)

    print("Address parsing complete!")

    # Show summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    total_incomplete = df['flag_incomplete_address'].sum()
    total_complete = len(df) - total_incomplete
    print(f"Total warehouses: {len(df)}")
    print(f"  Complete addresses: {total_complete} ({100*total_complete/len(df):.1f}%)")
    print(f"  Incomplete addresses: {total_incomplete} ({100*total_incomplete/len(df):.1f}%)")

    # Show examples of incomplete addresses
    if total_incomplete > 0:
        print("\nExample incomplete addresses:")
        incomplete_df = df[df['flag_incomplete_address'] == True].head(5)
        for idx, row in incomplete_df.iterrows():
            print(f"  Original: {row['address']}")
            print(f"    Parsed: {row['street_number']} {row['street']}, {row['city']}, {row['state']} {row['postal_code']}")
            print()


def main():
    """Run address parser"""
    parser = argparse.ArgumentParser(
        description='Parse warehouse addresses into structured fields',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse only addresses with blank structured fields (default)
  python parse_addresses.py

  # Force re-parse all addresses
  python parse_addresses.py --force

  # Parse without creating backup
  python parse_addresses.py --no-backup
        """
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-parsing of all addresses (default: only parse blank ones)'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup before modifying database'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        help='Path to warehouse database (default: ../data/warehouses.parquet)'
    )

    args = parser.parse_args()

    # Default path to warehouse database
    if args.db_path:
        db_path = Path(args.db_path)
    else:
        data_dir = Path(__file__).parent.parent / "data"
        db_path = data_dir / "warehouses.parquet"

    print("=" * 60)
    print("Warehouse Address Parser")
    print("=" * 60)
    print()

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Please run warehouse_finder.py first to create the database")
        return

    # Run parser
    parse_addresses(
        str(db_path),
        force=args.force,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
