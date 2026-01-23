"""
Warehouse Finder - Search OpenStreetMap for warehouses and maintain a parquet database

This script searches for warehouses in a specified location and radius,
and stores/updates the results in a parquet file.
"""

import requests
import pandas as pd
import os
import math
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime


# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Config file path
CONFIG_FILE = Path(__file__).parent.parent / "cfg" / "warehouse_finder.yaml"

# Conversion constants
METERS_PER_DEGREE_LAT = 111320  # Approximately constant
SQ_FEET_PER_SQ_METER = 10.7639


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

    def search_warehouses(
        self,
        lat: float,
        lon: float,
        radius_meters: int,
        include_building_types: Optional[List[str]] = None,
        exclude_building_types: Optional[List[str]] = None,
        min_area_sq_ft: Optional[float] = None,
        use_precise_area: bool = False
    ) -> List[Dict]:
        """
        Search for warehouses within a radius of a location

        Args:
            lat: Latitude of center point
            lon: Longitude of center point
            radius_meters: Search radius in meters
            include_building_types: List of building types to include (e.g., ['warehouse', 'industrial'])
                                   If None, defaults to ['warehouse']
            exclude_building_types: List of building types to exclude (e.g., ['office', 'retail'])
            min_area_sq_ft: Minimum building area in square feet (filters after retrieval)
            use_precise_area: If True, use Shoelace formula for precise area calculation.
                            If False, use faster bounding box estimation.

        Returns:
            List of warehouse dictionaries with extracted data
        """
        # Default to searching for warehouses if no types specified
        if include_building_types is None:
            include_building_types = ['warehouse']

        # Build Overpass QL query dynamically based on building types and exclusions
        query_parts = []

        if min_area_sq_ft is not None and not include_building_types:
            # Area filtering with no type preference: get ALL buildings, filter by size later
            # This is the slowest case - only use when you want large buildings of any type
            print(f"Area filtering: retrieving ALL buildings (exclusions applied after size calculation)")
            query_parts.append(f'way["building"](around:{radius_meters},{lat},{lon});')
            query_parts.append(f'relation["building"](around:{radius_meters},{lat},{lon});')
        elif min_area_sq_ft is not None and include_building_types:
            # Area filtering WITH type hints: get specific types, then filter by size
            # This is more efficient - we only get buildings we're interested in
            print(f"Area filtering: retrieving {include_building_types} buildings (then filtering by size)")
            for building_type in include_building_types:
                query_parts.append(f'way["building"="{building_type}"](around:{radius_meters},{lat},{lon});')
                query_parts.append(f'relation["building"="{building_type}"](around:{radius_meters},{lat},{lon});')
        else:
            # No area filtering, search only for specific building types
            for building_type in include_building_types:
                query_parts.append(f'way["building"="{building_type}"](around:{radius_meters},{lat},{lon});')
                query_parts.append(f'relation["building"="{building_type}"](around:{radius_meters},{lat},{lon});')

        query = f"""
        [out:json][timeout:25];
        (
          {' '.join(query_parts)}
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

        print(f"Found {len(buildings)} buildings from API")

        # Process each building
        for element in buildings:
            warehouse = self._extract_warehouse_data(
                element, elements,
                include_building_types, exclude_building_types,
                min_area_sq_ft, use_precise_area
            )
            if warehouse:
                warehouses.append(warehouse)

        print(f"After filtering: {len(warehouses)} buildings match criteria")

        return warehouses

    def _extract_warehouse_data(
        self,
        element: Dict,
        all_elements: List[Dict],
        include_building_types: Optional[List[str]] = None,
        exclude_building_types: Optional[List[str]] = None,
        min_area_sq_ft: Optional[float] = None,
        use_precise_area: bool = False
    ) -> Optional[Dict]:
        """
        Extract relevant warehouse data from an OSM element

        Args:
            element: OSM element (way or relation)
            all_elements: All elements from the query (for node lookups)
            include_building_types: Building types to include
            exclude_building_types: Building types to exclude
            min_area_sq_ft: Minimum area filter
            use_precise_area: Use precise area calculation

        Returns:
            Dictionary with warehouse data or None (if filtered out)
        """
        tags = element.get("tags", {})
        building_type = tags.get("building", "")

        # Filter by building type exclusions
        if exclude_building_types and building_type in exclude_building_types:
            return None

        # Get node coordinates for area calculation
        coords = self._get_building_coordinates(element, all_elements)
        if not coords:
            return None

        # Calculate area if needed
        area_sq_ft = None
        if min_area_sq_ft is not None or True:  # Always calculate for storage
            if use_precise_area:
                area_sq_ft = self._calculate_polygon_area(coords)
            else:
                area_sq_ft = self._calculate_bounding_box_area(coords)

        # Filter by building type inclusions (if min_area is set, we got all buildings)
        if min_area_sq_ft is not None:
            # When filtering by area, check both type and size
            if include_building_types and building_type not in include_building_types:
                # Not in include list, check if it meets size requirement
                if area_sq_ft is None or area_sq_ft < min_area_sq_ft:
                    return None
            elif area_sq_ft is None or area_sq_ft < min_area_sq_ft:
                # In include list or no include list, but doesn't meet size
                return None

        # Calculate centroid from coordinates
        lat = sum(c[0] for c in coords) / len(coords)
        lon = sum(c[1] for c in coords) / len(coords)

        warehouse = {
            "osm_id": f"{element['type']}/{element['id']}",
            "osm_type": element["type"],
            "osm_element_id": element["id"],
            "latitude": lat,
            "longitude": lon,
            "name": tags.get("name", ""),
            "address": self._build_address(tags),
            "owner": tags.get("owner", tags.get("operator", "")),
            "building_type": building_type,
            "area_sq_ft": area_sq_ft if area_sq_ft is not None else 0.0,
            "last_updated": datetime.utcnow().isoformat(),
            "search_timestamp": datetime.utcnow().isoformat(),
        }

        return warehouse

    def _get_building_coordinates(self, element: Dict, all_elements: List[Dict]) -> List[Tuple[float, float]]:
        """
        Get the coordinates of all nodes that define a building

        Args:
            element: OSM element (way or relation)
            all_elements: All elements from the query (for node lookups)

        Returns:
            List of (lat, lon) tuples
        """
        # For ways, get nodes from the 'nodes' list
        if element["type"] == "way":
            node_ids = element.get("nodes", [])
        else:
            # For relations, this is more complex - would need to parse members
            return []

        # Build a map of node IDs to coordinates
        node_map = {e["id"]: e for e in all_elements if e["type"] == "node"}

        # Get coordinates for all nodes
        coords = []
        for node_id in node_ids:
            if node_id in node_map:
                node = node_map[node_id]
                coords.append((node["lat"], node["lon"]))

        return coords

    def _calculate_bounding_box_area(self, coords: List[Tuple[float, float]]) -> float:
        """
        Calculate approximate area using bounding box (fast but less accurate)

        Args:
            coords: List of (lat, lon) tuples

        Returns:
            Area in square feet
        """
        if len(coords) < 3:
            return 0.0

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Convert lat/lon differences to meters
        # Latitude: approximately constant
        height_meters = (max_lat - min_lat) * METERS_PER_DEGREE_LAT

        # Longitude: varies by latitude, use average latitude
        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * abs(((avg_lat * 3.14159) / 180))
        width_meters = (max_lon - min_lon) * meters_per_degree_lon

        # Calculate area in square meters, then convert to square feet
        area_sq_meters = height_meters * width_meters
        area_sq_ft = area_sq_meters * SQ_FEET_PER_SQ_METER

        return area_sq_ft

    def _calculate_polygon_area(self, coords: List[Tuple[float, float]]) -> float:
        """
        Calculate precise area using Shoelace formula (slower but accurate)

        Args:
            coords: List of (lat, lon) tuples

        Returns:
            Area in square feet
        """
        if len(coords) < 3:
            return 0.0

        # Convert all coordinates to meters relative to first point
        first_lat, first_lon = coords[0]
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * abs(((first_lat * 3.14159) / 180))

        # Convert to (x, y) in meters
        points_meters = []
        for lat, lon in coords:
            x = (lon - first_lon) * meters_per_degree_lon
            y = (lat - first_lat) * METERS_PER_DEGREE_LAT
            points_meters.append((x, y))

        # Apply Shoelace formula
        area = 0.0
        n = len(points_meters)
        for i in range(n):
            j = (i + 1) % n
            area += points_meters[i][0] * points_meters[j][1]
            area -= points_meters[j][0] * points_meters[i][1]

        area = abs(area) / 2.0  # Area in square meters

        # Convert to square feet
        area_sq_ft = area * SQ_FEET_PER_SQ_METER

        return area_sq_ft

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
    """Search for warehouses using locations defined in config file"""

    # Load config file
    if not CONFIG_FILE.exists():
        print(f"Error: Config file not found at {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    locations = config.get("locations", [])
    if not locations:
        print("Error: No locations defined in config file")
        return

    # Initialize finder - stores in data/ directory
    db_path = DATA_DIR / "warehouses.parquet"
    finder = WarehouseFinder(db_path=str(db_path))

    # Process each location from config
    for location in locations:
        name = location.get("name", "Unnamed")
        lat = location.get("lat")
        lon = location.get("lon")
        radius_meters = location.get("radius_meters", 10000)

        if lat is None or lon is None:
            print(f"Skipping '{name}': missing lat or lon")
            continue

        print(f"\n{'='*60}")
        print(f"Searching: {name}")
        print(f"  Center: ({lat}, {lon}), Radius: {radius_meters}m")
        print(f"{'='*60}")

        warehouses = finder.search_warehouses(
            lat, lon, radius_meters,
            include_building_types=['warehouse', 'industrial', 'retail'],
            exclude_building_types=['office', 'residential'],
            min_area_sq_ft=100000,
            use_precise_area=True
        )

        print(f"\nFound buildings with area >= 100,000 sq ft:")
        for w in warehouses:
            print(f"  - {w['name'] or 'Unnamed'}: {w['area_sq_ft']:,.0f} sq ft ({w['building_type']})")

        # Update database with results from this location
        finder.update_database(warehouses)

    # Show final statistics
    stats = finder.get_statistics()
    print(f"\n{'='*60}")
    print("Final Database Statistics:")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
