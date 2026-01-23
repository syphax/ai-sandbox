"""
Simple script to view the warehouse database contents
"""

import pandas as pd
import numpy as np
from pathlib import Path

FLAG_SAVE_AS_CSV = True

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

list_blank_values = ['', 'None', np.nan]

print(f"\nSample of warehouses with names:")
named = df[~df['name'].isin(list_blank_values)]
if len(named) > 0:
    print(named.head(5)[['name', 'address', 'latitude', 'longitude']])
else:
    print("No warehouses with names found")

print(f"\nSample of warehouses with owners:")
owned = df[~df['owner'].isin(list_blank_values)]
if len(owned) > 0:
    print(owned.head(5)[['name', 'owner', 'address']])
else:
    print("No warehouses with owners found")

cnt_whs = df.shape[0]
cnt_whs_named = named.shape[0]
cnt_whs_owned = owned.shape[0]

print(f"\nSummary:")
print(f"Total warehouses: {cnt_whs}")
print(f"Warehouses with names: {cnt_whs_named} ({cnt_whs_named/cnt_whs:.1%})")
print(f"Warehouses with owners: {cnt_whs_owned} ({cnt_whs_owned/cnt_whs:.1%})")

if FLAG_SAVE_AS_CSV:
    df.to_csv(data_dir / "warehouses.csv", index=False)