"""
Simple script to view the warehouse database contents
"""

import pandas as pd
from pathlib import Path

# Load and display the warehouse database
data_dir = Path(__file__).parent.parent / "data"
db_path = data_dir / "warehouses.parquet"

if not db_path.exists():
    print(f"Database not found at {db_path}")
    print("Run warehouse_finder.py first to create the database")
    exit(1)

df = pd.read_parquet(db_path)

print(f"Total warehouses: {len(df)}")
print(f"\nFirst 10 warehouses:")
print(df.head(10).to_string())

print(f"\nColumn info:")
print(df.info())

print(f"\nSample of warehouses with names:")
named = df[df['name'] != '']
if len(named) > 0:
    print(named.head(5)[['name', 'address', 'latitude', 'longitude']])
else:
    print("No warehouses with names found")

print(f"\nSample of warehouses with owners:")
owned = df[df['owner'] != '']
if len(owned) > 0:
    print(owned.head(5)[['name', 'owner', 'address']])
else:
    print("No warehouses with owners found")
