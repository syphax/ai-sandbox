# Address Parsing Documentation

## Overview

The warehouse database has been enhanced with structured address fields to better handle address data. This document explains the new schema and how to use the migration and parsing scripts.

## New Address Schema

The warehouse database now includes the following address fields:

| Field | Type | Description |
|-------|------|-------------|
| `address` | string | Original full address from OSM (preserved) |
| `street_number` | string | Street number (can be alphanumeric like "123A") |
| `street` | string | Street name |
| `city` | string | City or town name |
| `state` | string | State/province (2-letter code for US) |
| `postal_code` | string | Primary postal code (ZIP5 in US) |
| `postal_code_ext` | string | Extended postal code (ZIP+4 in US) |
| `flag_incomplete_address` | boolean | True if any required fields are missing |

## Scripts

### 1. migrate_address_schema.py

**Purpose**: Add new address fields to the database schema (run rarely, typically once)

**Usage**:
```bash
cd data_builders
python3 migrate_address_schema.py
```

**What it does**:
- Adds all new address fields to the database
- Initializes fields with empty values
- Sets `flag_incomplete_address` to True by default
- Creates a timestamped backup automatically
- Adds schema version tracking

**When to run**: Only when setting up the new schema for the first time, or when schema changes are needed.

### 2. parse_addresses.py

**Purpose**: Parse addresses from the `address` field into structured components

**Usage**:

```bash
cd data_builders

# Parse only addresses with blank structured fields (default)
python3 parse_addresses.py

# Force re-parse ALL addresses (overwrites existing parsed data)
python3 parse_addresses.py --force

# Parse without creating backup
python3 parse_addresses.py --no-backup

# Specify custom database path
python3 parse_addresses.py --db-path /path/to/warehouses.parquet
```

**What it does**:
- Parses the `address` field into structured components
- Supports incremental mode (default): only parse blank addresses
- Supports force mode: re-parse all addresses
- Sets `flag_incomplete_address` based on completeness
- Creates timestamped backups (unless --no-backup specified)
- Shows summary statistics

**When to run**:
- After running warehouse_finder.py to add new warehouses
- When you want to refresh address parsing logic
- Use incremental mode (default) for regular updates
- Use force mode when you've improved the parser

## Parsing Logic

The address parser handles several formats:

### Complete Address
```
Input:  "742, Main Street, North Oxford, MA, 01537"
Output:
  street_number: "742"
  street: "Main Street"
  city: "North Oxford"
  state: "MA"
  postal_code: "01537"
  flag_incomplete_address: False
```

### Partial Address (No City/State/ZIP)
```
Input:  "90, Huntoon Memorial Highway"
Output:
  street_number: "90"
  street: "Huntoon Memorial Highway"
  city: ""
  state: ""
  postal_code: ""
  flag_incomplete_address: True
```

### ZIP+4 Format
```
Input:  "123, Main St, Boston, MA, 02101-1234"
Output:
  postal_code: "02101"
  postal_code_ext: "1234"
```

## Workflow

### Initial Setup
```bash
# 1. Migrate schema (one time)
python3 migrate_address_schema.py

# 2. Parse all addresses
python3 parse_addresses.py
```

### Regular Updates
```bash
# 1. Search for new warehouses
python3 warehouse_finder.py

# 2. Parse new addresses (incremental)
python3 parse_addresses.py
```

### After Parser Improvements
```bash
# Force re-parse all addresses with new logic
python3 parse_addresses.py --force
```

## Data Quality

After parsing, you can check data quality:

```python
import pandas as pd

df = pd.read_parquet('../data/warehouses.parquet')

# Check completeness
complete = len(df[df['flag_incomplete_address'] == False])
total = len(df)
print(f"Complete addresses: {complete}/{total} ({100*complete/total:.1f}%)")

# View incomplete addresses
incomplete = df[df['flag_incomplete_address'] == True]
print(incomplete[['address', 'street_number', 'street', 'city', 'state', 'postal_code']].head(10))
```

## Backups

Both scripts create automatic timestamped backups before modifying the database:
- Format: `warehouses.backup_YYYYMMDD_HHMMSS.parquet`
- Location: Same directory as the database file
- Can be disabled with `--no-backup` flag

To restore from backup:
```bash
cp data/warehouses.backup_20251120_123456.parquet data/warehouses.parquet
```
