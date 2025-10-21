# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository is building a **Distribution Network Planning** web application to help organizations design and optimize distribution networks. The repository also serves as a testing ground for AI coding tools like Claude Code.

## Project Structure

```
.
├── data_builders/       # Python scripts to build and maintain datasets
├── data/                # Generated datasets (parquet files, not in git)
├── docs/                # Documentation for specific components
├── output/              # Generated outputs and reports
├── tests/               # Test files
└── requirements.txt     # Python dependencies
```

## Current State

**Phase 1: Data Collection** (In Progress)

The project is currently focused on building foundational datasets:
- Warehouse location database from OpenStreetMap
- Future: demographics, transportation networks, market data

The main web application does not exist yet.

## Development Approach

### Language and Stack
- Primary language: **Python 3.9+**
- Data format: **Parquet** files for efficient storage
- Future: Web framework TBD (likely Flask or FastAPI)

### Running Data Builders

All data builder scripts are in `data_builders/` and output to `data/`:

```bash
# Install dependencies first
pip install -r requirements.txt

# Run individual data builders
cd data_builders
python3 warehouse_finder.py    # Search OSM for warehouses
python3 view_warehouses.py     # View warehouse database
```

### Code Organization

- **data_builders/**: Standalone scripts that build/update datasets
  - Each script should be independently runnable
  - Output goes to `data/` directory
  - Should support both initial creation and incremental updates

- **data/**: Generated datasets (gitignored)
  - Use Parquet format for efficiency
  - Include metadata about when/how data was generated

- **tests/**: Unit and integration tests
  - TBD: Will add pytest once codebase grows

### Best Practices

- Data builders should be idempotent (safe to re-run)
- Support incremental updates without duplicates
- Include clear logging and progress indicators
- Document data sources and update frequency
- Store all generated data in `data/` directory
