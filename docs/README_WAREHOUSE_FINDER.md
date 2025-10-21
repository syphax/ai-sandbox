# Warehouse Finder

A Python tool to search OpenStreetMap for warehouses and build a searchable database.

## Features

- Search for warehouses in OpenStreetMap within a specified radius
- Store warehouse data in a Parquet database for efficient querying
- Update existing database without duplicates (based on OSM ID)
- Extract key warehouse information: location, name, address, owner

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Edit the `main()` function in [warehouse_finder.py](warehouse_finder.py) to specify your search location:

```python
# Set your search parameters
lat = 34.0522          # Latitude (e.g., Downtown LA)
lon = -118.2437        # Longitude
radius_meters = 10000  # 10km radius

# Initialize and run
finder = WarehouseFinder(db_path="warehouses.parquet")
warehouses = finder.search_warehouses(lat, lon, radius_meters)
finder.update_database(warehouses)
```

Then run:

```bash
python3 warehouse_finder.py
```

### Programmatic Usage

```python
from warehouse_finder import WarehouseFinder

# Initialize
finder = WarehouseFinder(db_path="my_warehouses.parquet")

# Search multiple locations
locations = [
    (34.0522, -118.2437, 10000),  # LA
    (40.7128, -74.0060, 15000),    # NYC
    (41.8781, -87.6298, 12000),    # Chicago
]

for lat, lon, radius in locations:
    warehouses = finder.search_warehouses(lat, lon, radius)
    finder.update_database(warehouses)

# Get statistics
stats = finder.get_statistics()
print(stats)
```

### Viewing the Database

Use the included viewer script:

```bash
python3 view_warehouses.py
```

Or load the parquet file directly:

```python
import pandas as pd
df = pd.read_parquet("warehouses.parquet")
```

## Database Schema

The warehouse database includes the following fields:

| Field | Type | Description |
|-------|------|-------------|
| osm_id | string | Unique identifier combining OSM type and element ID |
| osm_type | string | OSM element type (way or relation) |
| osm_element_id | int | OSM element ID number |
| latitude | float | Warehouse centroid latitude |
| longitude | float | Warehouse centroid longitude |
| name | string | Warehouse name (if available) |
| address | string | Formatted address from OSM tags |
| owner | string | Owner or operator name (if available) |
| building_type | string | Building type from OSM (usually "warehouse") |
| last_updated | datetime | Last time this record was updated |
| search_timestamp | datetime | When this warehouse was found in a search |

## Current Limitations

- Only searches for buildings with `building=warehouse` tag
- Does not yet calculate building area or filter by size
- Address data depends on OSM tagging quality
- API rate limits apply (respects Overpass API guidelines)

## Future Enhancements

Planned features:
1. Calculate building area from node coordinates
2. Filter by minimum building size (e.g., 200,000 sq ft)
3. Support additional data sources beyond OSM
4. Add reverse geocoding for better address data
5. Web interface for search and visualization
6. Export to additional formats (CSV, GeoJSON)

## Data Source

This tool uses the [Overpass API](https://overpass-api.de/) to query OpenStreetMap data. Please be respectful of API rate limits and consider running your own Overpass instance for heavy usage.

## Example Results

Running the default script (10km radius around Downtown LA) finds approximately 3,000+ warehouses with varying levels of detail:

- Some include full addresses, names, and owner information
- Others only have location coordinates
- Data quality depends on OSM contributor coverage in the area
