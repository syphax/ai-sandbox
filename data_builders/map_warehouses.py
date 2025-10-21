"""
Map Warehouses - Generate an interactive map of warehouses from the database

This script creates an HTML map with pins for all warehouses in the database.
The map is interactive and can be opened in a web browser.
"""

import pandas as pd
from pathlib import Path
import folium
from folium.plugins import MarkerCluster
import webbrowser


# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "warehouses.parquet"
MAP_OUTPUT = OUTPUT_DIR / "warehouse_map.html"


def load_warehouses() -> pd.DataFrame:
    """Load warehouse database"""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Run warehouse_finder.py first to create the database")
        exit(1)

    print(f"Loading warehouse database from {DB_PATH}")
    return pd.read_parquet(DB_PATH)


def create_warehouse_map(df: pd.DataFrame, use_clustering: bool = True) -> folium.Map:
    """
    Create an interactive map with warehouse markers

    Args:
        df: DataFrame with warehouse data
        use_clustering: Whether to cluster nearby markers for better performance

    Returns:
        Folium map object
    """
    # Calculate center point (mean of all coordinates)
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    print(f"Creating map centered at ({center_lat:.4f}, {center_lon:.4f})")

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # Add layer control for different map styles
    folium.TileLayer('CartoDB positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter').add_to(m)

    # Create marker cluster if enabled
    if use_clustering:
        marker_cluster = MarkerCluster(name="Warehouses").add_to(m)
        marker_container = marker_cluster
    else:
        marker_container = m

    # Add markers for each warehouse
    print(f"Adding {len(df)} warehouse markers...")

    for idx, row in df.iterrows():
        # Build popup content
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 200px;">
            <h4 style="margin: 0 0 10px 0;">{row['name'] if row['name'] else 'Warehouse'}</h4>
            <table style="width: 100%;">
                <tr><td><b>OSM ID:</b></td><td>{row['osm_id']}</td></tr>
                <tr><td><b>Address:</b></td><td>{row['address'] if row['address'] else 'N/A'}</td></tr>
                <tr><td><b>Owner:</b></td><td>{row['owner'] if row['owner'] else 'N/A'}</td></tr>
                <tr><td><b>Coordinates:</b></td><td>{row['latitude']:.5f}, {row['longitude']:.5f}</td></tr>
            </table>
            <p style="margin: 10px 0 0 0; font-size: 10px; color: #666;">
                <a href="https://www.openstreetmap.org/{row['osm_type']}/{row['osm_element_id']}" target="_blank">View on OpenStreetMap</a>
            </p>
        </div>
        """

        # Choose icon color based on whether warehouse has a name
        icon_color = 'blue' if row['name'] else 'gray'

        # Create marker
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row['name'] if row['name'] else f"Warehouse {row['osm_element_id']}",
            icon=folium.Icon(color=icon_color, icon='warehouse', prefix='fa')
        ).add_to(marker_container)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add statistics box
    stats_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        right: 10px;
        width: 200px;
        background-color: white;
        border: 2px solid gray;
        z-index: 9999;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    ">
        <h4 style="margin: 0 0 10px 0;">Warehouse Statistics</h4>
        <p style="margin: 5px 0;"><b>Total:</b> {len(df)}</p>
        <p style="margin: 5px 0;"><b>Named:</b> {(df['name'] != '').sum()}</p>
        <p style="margin: 5px 0;"><b>With Addresses:</b> {(df['address'] != '').sum()}</p>
        <p style="margin: 5px 0;"><b>With Owners:</b> {(df['owner'] != '').sum()}</p>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 10px; color: #666;">
            <span style="color: blue;">●</span> Named warehouse<br>
            <span style="color: gray;">●</span> Unnamed warehouse
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    return m


def main():
    """Generate and display warehouse map"""
    # Load data
    df = load_warehouses()

    # Create map
    print("\nGenerating map...")
    warehouse_map = create_warehouse_map(df, use_clustering=True)

    # Save map
    warehouse_map.save(str(MAP_OUTPUT))
    print(f"\nMap saved to: {MAP_OUTPUT}")
    print(f"File size: {MAP_OUTPUT.stat().st_size / 1024:.1f} KB")

    # Open in browser
    print("\nOpening map in browser...")
    webbrowser.open(f"file://{MAP_OUTPUT.absolute()}")

    print("\nDone! The map should open in your default browser.")
    print("You can also open it manually by opening:")
    print(f"  {MAP_OUTPUT}")


if __name__ == "__main__":
    main()
