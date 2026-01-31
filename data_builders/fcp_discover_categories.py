#!/usr/bin/env python3
"""
FCP Euro Category Discovery Script

Discovers all product line categories from FCP Euro website and generates
a YAML configuration file for the main scraper.

Usage:
    python fcp_discover_categories.py
"""

from curl_cffi import requests
from bs4 import BeautifulSoup
import yaml
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Brand URL slugs mapping
BRANDS = {
    'Mercedes': 'Mercedes~Benz-parts',
    'BMW': 'BMW-parts',
    'Volvo': 'Volvo-parts',
    'VW': 'Volkswagen-parts',
    'Audi': 'Audi-parts',
    'Porsche': 'Porsche-parts',
}

BASE_URL = 'https://www.fcpeuro.com'
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 2  # seconds


def fetch_page(url: str, max_retries: int = 3) -> Optional[str]:
    """Fetch a page with retry logic using browser impersonation."""
    for attempt in range(max_retries):
        try:
            # Use chrome browser impersonation to bypass bot detection
            response = requests.get(url, impersonate="chrome", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(DELAY_BETWEEN_REQUESTS * (attempt + 1))  # Backoff

    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None


def extract_categories(html: str) -> Dict[str, List[str]]:
    """
    Extract product line categories from the page HTML.

    Returns a dict mapping primary categories to lists of subcategories.
    """
    soup = BeautifulSoup(html, 'html.parser')
    categories = {}

    # Find the taxons navigation div
    taxons_div = soup.find('div', class_='taxons taxons--browse')
    if not taxons_div:
        logger.warning("Could not find taxons navigation div")
        return categories

    # Find all category links in the taxon list
    taxon_list = taxons_div.find('ul', class_='taxonList')
    if not taxon_list:
        logger.warning("Could not find taxon list")
        return categories

    # Process each top-level category
    for item in taxon_list.find_all('li', class_='taxonList__item', recursive=False):
        # Get the primary category link
        primary_link = item.find('a', class_=['taxonList__link', 'taxonList__link--active'])
        if not primary_link:
            continue

        primary_name = primary_link.get_text(strip=True)
        subcategories = []

        # Look for subcategories
        sub_list = item.find('ul', class_='taxonList__sub')
        if sub_list:
            for sub_item in sub_list.find_all('li', class_='taxonList__item'):
                sub_link = sub_item.find('a')
                if sub_link:
                    sub_name = sub_link.get_text(strip=True)
                    subcategories.append(sub_name)

        categories[primary_name] = subcategories

    return categories


def discover_all_categories() -> Dict[str, Dict[str, List[str]]]:
    """
    Discover categories for all brands.

    Returns a dict mapping brand names to their category structures.
    """
    all_categories = {}

    for brand_name, brand_slug in BRANDS.items():
        logger.info(f"Discovering categories for {brand_name}...")

        # Fetch a category page to get the navigation
        # Using a generic path that should have the full category tree
        url = f"{BASE_URL}/{brand_slug}/"
        html = fetch_page(url)

        if html:
            categories = extract_categories(html)
            all_categories[brand_name] = categories
            logger.info(f"  Found {len(categories)} primary categories for {brand_name}")
        else:
            all_categories[brand_name] = {}
            logger.warning(f"  Could not fetch categories for {brand_name}")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    return all_categories


def is_product_category(name: str) -> bool:
    """
    Determine if a category name is a product category vs a vehicle model.
    Vehicle models typically end with 'parts' (e.g., '128i parts', 'A4 Quattro parts').
    Product categories don't (e.g., 'AC and Climate Control', 'Steering', 'Brake').
    """
    return not name.lower().endswith(' parts') and not name.lower().endswith(' parts')


def generate_config(all_categories: Dict[str, Dict[str, List[str]]]) -> dict:
    """Generate the YAML configuration structure."""

    # Collect all unique primary categories across all brands
    # Filter to only include product categories (not vehicle models)
    all_primary = set()
    for brand_categories in all_categories.values():
        for cat_name in brand_categories.keys():
            if is_product_category(cat_name):
                all_primary.add(cat_name)

    # Build product_lines config with all set to True
    product_lines = {name: True for name in sorted(all_primary)}

    config = {
        'brands': list(BRANDS.keys()),
        'brand_slugs': BRANDS,
        'settings': {
            'min_delay_seconds': 2,
            'max_delay_seconds': 5,
            'max_pages': None,  # None = no limit
            'max_retries': 3,
            'request_timeout': 30,
        },
        'product_lines': product_lines,
        'discovered_categories': {
            brand: {
                'primary': list(cats.keys()),
                'subcategories': cats
            }
            for brand, cats in all_categories.items()
        }
    }

    return config


def save_config(config: dict, output_path: Path) -> None:
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info(f"Configuration saved to {output_path}")


def main():
    """Main entry point."""
    logger.info("Starting FCP Euro category discovery...")

    # Discover categories
    all_categories = discover_all_categories()

    # Generate config
    config = generate_config(all_categories)

    # Save config
    output_path = Path(__file__).parent / 'fcp_config.yaml'
    save_config(config, output_path)

    # Print summary
    total_primary = len(config['product_lines'])
    logger.info(f"\nDiscovery complete!")
    logger.info(f"  Total unique primary categories: {total_primary}")
    logger.info(f"  Configuration file: {output_path}")

    print("\n--- Primary Categories Found ---")
    for cat in sorted(config['product_lines'].keys()):
        print(f"  - {cat}")


if __name__ == '__main__':
    main()
