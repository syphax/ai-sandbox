"""
Warehouse Finder - Search OpenStreetMap for warehouses and maintain a parquet database

This script searches for warehouses in a specified location and radius,
and stores/updates the results in a parquet file.
"""

import requests
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class WarehouseFinder:
    """Search for warehouses in OpenStreetMap and maintain a database"""

    def __init__(self, db_path: str = "warehouses.parquet"):
        """
        Initialize the warehouse finder

        Args:
            db_path: Path to the parquet database file
        """
        self.db_path = db_path
        self.overpass_url = "https://overpass-api.de/api/interpreter"

    def search_warehouses(self, lat: float, lon: float, radius_meters: int) -> List[Dict]:
        """
        Search for warehouses within a radius of a location

        Args:
            lat: Latitude of center point
            lon: Longitude of center point
            radius_meters: Search radius in meters

        Returns:
            List of warehouse dictionaries with extracted data
        """
        # Overpass QL query to find buildings with building=warehouse tag
        # Using around filter instead of center coordinates
        query = f"""
        [out:json][timeout:25];
        (
          way["building"="warehouse"](around:{radius_meters},{lat},{lon});
          relation["building"="warehouse"](around:{radius_meters},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """

        print(f"Searching for warehouses within {radius_meters}m of ({lat}, {lon})...")

        try:
            response = requests.post(self.overpass_url, data={"data": query}, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying Overpass API: {e}")
            return []

        warehouses = []
        elements = data.get("elements", [])

        # Filter for ways and relations (not nodes)
        buildings = [e for e in elements if e.get("type") in ["way", "relation"]]

        print(f"Found {len(buildings)} warehouse buildings")

        for element in buildings:
            warehouse = self._extract_warehouse_data(element, elements)
            if warehouse:
                warehouses.append(warehouse)

        return warehouses

    def _extract_warehouse_data(self, element: Dict, all_elements: List[Dict]) -> Optional[Dict]:
        """
        Extract relevant warehouse data from an OSM element

        Args:
            element: OSM element (way or relation)
            all_elements: All elements from the query (for node lookups)

        Returns:
            Dictionary with warehouse data or None
        """
        tags = element.get("tags", {})

        # Calculate centroid from nodes
        lat, lon = self._calculate_centroid(element, all_elements)

        if lat is None or lon is None:
            return None

        warehouse = {
            "osm_id": f"{element['type']}/{element['id']}",
            "osm_type": element["type"],
            "osm_element_id": element["id"],
            "latitude": lat,
            "longitude": lon,
            "name": tags.get("name", ""),
            "address": self._build_address(tags),
            "owner": tags.get("owner", tags.get("operator", "")),
            "building_type": tags.get("building", "warehouse"),
            "last_updated": datetime.utcnow().isoformat(),
            "search_timestamp": datetime.utcnow().isoformat(),
        }

        return warehouse

    def _calculate_centroid(self, element: Dict, all_elements: List[Dict]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate the centroid (center point) of a building

        Args:
            element: OSM element (way or relation)
            all_elements: All elements from the query (for node lookups)

        Returns:
            Tuple of (latitude, longitude) or (None, None)
        """
        # For ways, get nodes from the 'nodes' list
        if element["type"] == "way":
            node_ids = element.get("nodes", [])
        # For relations, this is more complex - skip for now
        else:
            # Use center coordinates if available
            if "center" in element:
                return element["center"]["lat"], element["center"]["lon"]
            return None, None

        # Build a map of node IDs to coordinates
        node_map = {e["id"]: e for e in all_elements if e["type"] == "node"}

        # Get coordinates for all nodes
        coords = []
        for node_id in node_ids:
            if node_id in node_map:
                node = node_map[node_id]
                coords.append((node["lat"], node["lon"]))

        if not coords:
            return None, None

        # Calculate centroid
        avg_lat = sum(c[0] for c in coords) / len(coords)
        avg_lon = sum(c[1] for c in coords) / len(coords)

        return avg_lat, avg_lon

    def _build_address(self, tags: Dict) -> str:
        """
        Build an address string from OSM tags

        Args:
            tags: OSM tags dictionary

        Returns:
            Formatted address string
        """
        addr_parts = []

        # House number and street
        if "addr:housenumber" in tags:
            addr_parts.append(tags["addr:housenumber"])
        if "addr:street" in tags:
            addr_parts.append(tags["addr:street"])

        # City, state, postal code
        if "addr:city" in tags:
            addr_parts.append(tags["addr:city"])
        if "addr:state" in tags:
            addr_parts.append(tags["addr:state"])
        if "addr:postcode" in tags:
            addr_parts.append(tags["addr:postcode"])

        return ", ".join(addr_parts) if addr_parts else ""

    def load_database(self) -> pd.DataFrame:
        """
        Load existing warehouse database

        Returns:
            DataFrame with existing warehouses, or empty DataFrame
        """
        if os.path.exists(self.db_path):
            print(f"Loading existing database from {self.db_path}")
            return pd.read_parquet(self.db_path)
        else:
            print("No existing database found, will create new one")
            return pd.DataFrame()

    def update_database(self, new_warehouses: List[Dict]) -> None:
        """
        Update the warehouse database with new data

        Args:
            new_warehouses: List of warehouse dictionaries to add/update
        """
        if not new_warehouses:
            print("No warehouses to add")
            return

        # Load existing data
        existing_df = self.load_database()

        # Convert new warehouses to DataFrame
        new_df = pd.DataFrame(new_warehouses)

        if existing_df.empty:
            # No existing data, just save new data
            combined_df = new_df
            print(f"Creating new database with {len(new_df)} warehouses")
        else:
            # Merge with existing data, updating duplicates based on osm_id
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Remove duplicates, keeping the most recent entry
            combined_df = combined_df.sort_values("last_updated", ascending=False)
            combined_df = combined_df.drop_duplicates(subset=["osm_id"], keep="first")

            print(f"Updated database: {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total (after deduplication)")

        # Save to parquet
        combined_df.to_parquet(self.db_path, index=False)
        print(f"Database saved to {self.db_path}")

    def get_statistics(self) -> Dict:
        """
        Get statistics about the warehouse database

        Returns:
            Dictionary with database statistics
        """
        df = self.load_database()

        if df.empty:
            return {"total_warehouses": 0}

        stats = {
            "total_warehouses": len(df),
            "warehouses_with_names": df["name"].notna().sum(),
            "warehouses_with_addresses": df["address"].notna().sum(),
            "warehouses_with_owners": df["owner"].notna().sum(),
            "unique_owners": df["owner"].nunique(),
        }

        return stats


def main():
    """Example usage of the WarehouseFinder"""

    # Example: Search for warehouses in Los Angeles area
    # Downtown LA coordinates
    lat = 34.0522
    lon = -118.2437
    radius_meters = 10000  # 10km radius

    # Initialize finder - stores in data/ directory
    db_path = DATA_DIR / "warehouses.parquet"
    finder = WarehouseFinder(db_path=str(db_path))

    # Search for warehouses
    warehouses = finder.search_warehouses(lat, lon, radius_meters)

    # Update database
    finder.update_database(warehouses)

    # Show statistics
    stats = finder.get_statistics()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
