# Distribution Network Planning

A web application to help organizations design and optimize distribution networks.

## Project Status

This project is in early development. Currently building foundational datasets and utility scripts.

## Project Structure

```
.
├── data_builders/       # Scripts to build and maintain datasets
│   ├── warehouse_finder.py    # Search OSM for warehouses
│   ├── view_warehouses.py     # View warehouse database
│   └── map_warehouses.py      # Generate interactive map
├── data/                # Generated datasets (not in git)
│   └── warehouses.parquet     # Warehouse location database
├── docs/                # Documentation
│   └── README_WAREHOUSE_FINDER.md
├── output/              # Generated outputs and reports (not in git)
│   └── warehouse_map.html     # Interactive warehouse map
├── tests/               # Test files
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Current Capabilities

### Warehouse Database Builder

Search OpenStreetMap for warehouse locations and build a queryable database.

**Usage:**

```bash
# Install dependencies
pip install -r requirements.txt

# Run warehouse finder (searches Downtown LA by default)
cd data_builders
python3 warehouse_finder.py

# View results in terminal
python3 view_warehouses.py

# Generate interactive map (opens in browser)
python3 map_warehouses.py
```

The map generator creates an interactive HTML map with:
- Pins for all warehouses (blue = named, gray = unnamed)
- Clickable markers with warehouse details
- Marker clustering for better performance
- Multiple map style options
- Statistics overlay

See [docs/README_WAREHOUSE_FINDER.md](docs/README_WAREHOUSE_FINDER.md) for detailed documentation.

## Planned Features

### Phase 1: Data Collection (In Progress)
- [x] Warehouse location database from OpenStreetMap
- [ ] Calculate building areas from coordinates
- [ ] Filter warehouses by minimum size (e.g., 200,000+ sq ft)
- [ ] Additional data sources beyond OSM
- [ ] Demographics and market data
- [ ] Transportation network data

### Phase 2: Analysis Tools
- [ ] Network optimization algorithms
- [ ] Coverage analysis
- [ ] Cost modeling
- [ ] Demand forecasting

### Phase 3: Web Application
- [ ] Interactive map interface
- [ ] Search and filter capabilities
- [ ] Network visualization
- [ ] Optimization recommendations

## Development

This is an experimental sandbox repository for testing AI coding tools and building distribution network planning capabilities.

### Running Data Builders

All data builder scripts are in the `data_builders/` directory and output to `data/`:

```bash
cd data_builders
python3 warehouse_finder.py  # Builds warehouse database
```

### Data Storage

- All generated datasets are stored in `data/` (excluded from git)
- Primary format is Parquet for efficient storage and querying
- Can be easily converted to other formats (CSV, GeoJSON, etc.)

## Dependencies

- Python 3.9+
- pandas
- pyarrow (for parquet support)
- requests (for API calls)

See [requirements.txt](requirements.txt) for full list.

## License

TBD
