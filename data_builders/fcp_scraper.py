#!/usr/bin/env python3
"""
FCP Euro Parts Scraper

Scrapes auto parts data from FCP Euro website based on configuration.
Extracts data from both JavaScript arrays and HTML elements for completeness.

Usage:
    python fcp_scraper.py [--config CONFIG_PATH] [--brand BRAND] [--category CATEGORY] [--cookies COOKIES_FILE]

Cookie Export Instructions:
    1. Open Chrome and navigate to fcpeuro.com
    2. Complete the captcha/verification if prompted
    3. Install "EditThisCookie" or "Cookie-Editor" browser extension
    4. Click the extension icon and export cookies as JSON
    5. Save to a file (e.g., fcp_cookies.json)
    6. Run: python fcp_scraper.py --cookies fcp_cookies.json
"""

import argparse
import csv
import json
import logging
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from curl_cffi import requests
import yaml
from bs4 import BeautifulSoup

# Global cookies dict (loaded from file if provided)
COOKIES: Optional[Dict[str, str]] = None

# Browser profiles for curl_cffi - randomly selected at startup
BROWSER_PROFILES = [
    "chrome",
    "chrome110",
    "chrome116",
    "chrome119",
    "chrome120",
    "safari",
    "safari15_3",
    "safari15_5",
]

# Selected browser profile for this run (set in main)
BROWSER_PROFILE = None

# Track consecutive 403 errors to detect blocking
CONSECUTIVE_403_COUNT = 0
MAX_CONSECUTIVE_403 = 3  # Bail after this many consecutive 403s

# Configure logging - will be fully set up in setup_logging()
logger = logging.getLogger(__name__)


def setup_logging() -> Path:
    """Set up logging to both stdout and a timestamped log file."""
    # Create log directory if needed
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'fcp_euro_scrape-{timestamp}.log'

    # Set up root logger
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler (append mode)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return log_file

BASE_URL = 'https://www.fcpeuro.com'
OUTPUT_DIR = Path(__file__).parent.parent / 'data'
LOG_DIR = Path(__file__).parent.parent / 'log'
OUTPUT_FILE = OUTPUT_DIR / 'fcp-euro-parts.csv'

# CSV columns
CSV_COLUMNS = [
    'part_id',
    'sku',
    'name',
    'brand',
    'price',
    'available',
    'in_stock',
    'part_type',
    'product_line',
    'source_brand',
    'page_url',
    # Redundant fields from both sources for comparison
    'js_id',
    'js_name',
    'js_brand',
    'js_price',
    'html_name',
    'html_brand',
    'html_price',
    'html_sku',
]


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_cookies(cookies_path: Path) -> Dict[str, str]:
    """
    Load cookies from a JSON file exported from browser.

    Supports two formats:
    1. Browser extension format (EditThisCookie, Cookie-Editor):
       Array of objects with 'name' and 'value' fields
    2. Simple dict format: {"cookie_name": "cookie_value", ...}

    Returns a simple dict for use with requests.
    """
    with open(cookies_path) as f:
        data = json.load(f)

    cookies = {}

    if isinstance(data, list):
        # Browser extension format - array of cookie objects
        for cookie in data:
            name = cookie.get('name')
            value = cookie.get('value')
            if name and value is not None:
                cookies[name] = str(value)
    elif isinstance(data, dict):
        # Simple dict format
        cookies = {k: str(v) for k, v in data.items()}

    return cookies


def get_random_delay(config: dict) -> float:
    """Get a randomized delay using uniform distribution between min and max settings."""
    settings = config['settings']
    min_delay = settings.get('min_delay_seconds', 2)
    max_delay = settings.get('max_delay_seconds', 5)
    return random.uniform(min_delay, max_delay)


class BotBlockedException(Exception):
    """Raised when repeated 403 errors indicate bot blocking."""
    pass


def fetch_page(url: str, config: dict) -> Optional[str]:
    """Fetch a page with retry logic using browser impersonation."""
    global CONSECUTIVE_403_COUNT

    max_retries = config['settings'].get('max_retries', 3)
    timeout = config['settings'].get('request_timeout', 30)

    for attempt in range(max_retries):
        try:
            # Use randomly selected browser profile for this run
            # Include cookies if loaded from file
            response = requests.get(
                url,
                impersonate=BROWSER_PROFILE,
                timeout=timeout,
                cookies=COOKIES
            )
            response.raise_for_status()
            # Success - reset 403 counter
            CONSECUTIVE_403_COUNT = 0
            return response.text
        except Exception as e:
            error_str = str(e)
            is_403 = '403' in error_str

            if is_403:
                CONSECUTIVE_403_COUNT += 1
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e} "
                             f"(consecutive 403s: {CONSECUTIVE_403_COUNT})")

                if CONSECUTIVE_403_COUNT >= MAX_CONSECUTIVE_403:
                    logger.error(f"Detected {CONSECUTIVE_403_COUNT} consecutive 403 errors - "
                               f"likely bot blocked. Stopping.")
                    raise BotBlockedException(
                        f"Bot blocked after {CONSECUTIVE_403_COUNT} consecutive 403 errors"
                    )
            else:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")

            if attempt < max_retries - 1:
                # Randomized backoff delay (longer for 403s)
                delay = get_random_delay(config) * (attempt + 1)
                if is_403:
                    delay *= 2  # Double delay for 403 errors
                time.sleep(delay)

    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None


def extract_js_products(html: str) -> List[dict]:
    """Extract products from the JavaScript 'products' array in the page."""
    products = []

    # Look for the products array in script tags
    pattern = r'var\s+products\s*=\s*(\[.*?\]);'
    match = re.search(pattern, html, re.DOTALL)

    if match:
        try:
            products_json = match.group(1)
            products = json.loads(products_json)
            logger.debug(f"Extracted {len(products)} products from JS array")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse products JSON: {e}")

    return products


def extract_html_products(html: str) -> List[dict]:
    """Extract products from HTML .hit elements."""
    soup = BeautifulSoup(html, 'html.parser')
    products = []

    for hit in soup.find_all('div', class_='hit'):
        product = {}

        # Get data attributes from the hit div
        product['html_name'] = hit.get('data-name', '')
        product['html_brand'] = hit.get('data-brand', '')
        product['html_price'] = hit.get('data-price', '')
        product['html_sku'] = hit.get('data-sku', '')

        # Get availability status
        avail_div = hit.find('div', class_='hit__fulfill')
        if avail_div:
            avail_text = avail_div.get_text(strip=True)
            product['available'] = 'Available' in avail_text

            # Get stock status from fulfillDesc
            stock_div = avail_div.find('div', class_='hit__fulfillDesc')
            if stock_div:
                product['in_stock'] = stock_div.get_text(strip=True)
            else:
                product['in_stock'] = ''
        else:
            product['available'] = False
            product['in_stock'] = ''

        # Get part type (OE, Genuine, OEM, etc.)
        flag_div = hit.find('div', class_='hit__flag')
        if flag_div:
            product['part_type'] = flag_div.get_text(strip=True)
        else:
            product['part_type'] = ''

        # Get cart data for variant ID
        cart_div = hit.find('div', attrs={'data-controller': 'cart'})
        if cart_div:
            product['variant_id'] = cart_div.get('data-cart-variant-id-value', '')

        products.append(product)

    logger.debug(f"Extracted {len(products)} products from HTML")
    return products


def merge_products(js_products: List[dict], html_products: List[dict]) -> List[dict]:
    """
    Merge products from JS and HTML sources.
    Uses position to match products (both are in the same order on page).

    Note: The JS products array may not be present on all pages (site structure changes).
    When JS data is unavailable, we use HTML data with variant_id or SKU as part_id.
    """
    merged = []

    # Match by position (index)
    max_len = max(len(js_products), len(html_products)) if js_products or html_products else 0

    for i in range(max_len):
        product = {}

        # JS data (may not be available on newer page versions)
        if i < len(js_products):
            js = js_products[i]
            product['js_id'] = js.get('id', '')
            product['js_name'] = js.get('name', '')
            product['js_brand'] = js.get('brand', '')
            product['js_price'] = js.get('price', '')

            # Use JS as primary source for these fields
            product['part_id'] = js.get('id', '')
            product['name'] = js.get('name', '')
            product['brand'] = js.get('brand', '')
            product['price'] = js.get('price', '')

        # HTML data
        if i < len(html_products):
            html = html_products[i]
            product['html_name'] = html.get('html_name', '')
            product['html_brand'] = html.get('html_brand', '')
            product['html_price'] = html.get('html_price', '')
            product['html_sku'] = html.get('html_sku', '')
            product['available'] = html.get('available', False)
            product['in_stock'] = html.get('in_stock', '')
            product['part_type'] = html.get('part_type', '')

            # Use HTML SKU
            product['sku'] = html.get('html_sku', '')

            # Use variant_id as part_id if JS data not available
            if not product.get('part_id'):
                product['part_id'] = html.get('variant_id', '') or html.get('html_sku', '')

            # If we didn't have JS data, use HTML
            if not product.get('name'):
                product['name'] = html.get('html_name', '')
            if not product.get('brand'):
                product['brand'] = html.get('html_brand', '')
            if not product.get('price'):
                product['price'] = html.get('html_price', '')

        merged.append(product)

    return merged


def get_max_page(html: str) -> int:
    """Extract the maximum page number from pagination."""
    soup = BeautifulSoup(html, 'html.parser')

    pages_nav = soup.find('nav', class_='pages')
    if not pages_nav:
        return 1

    # Find all page links and get the highest number
    max_page = 1
    for span in pages_nav.find_all('span', class_=['pages__span', 'pages__span--current']):
        link = span.find('a')
        if link:
            text = link.get_text(strip=True)
            # Skip navigation arrows
            if text.isdigit():
                max_page = max(max_page, int(text))

    return max_page


def category_to_url_slug(category: str) -> str:
    """Convert category name to URL slug format."""
    # Replace spaces with hyphens, handle special chars
    slug = category.replace(' ', '-')
    slug = slug.replace('/', '-')
    slug = slug.replace('&', 'and')
    return slug


def backup_csv() -> Optional[Path]:
    """
    Create a timestamped backup of the CSV file if it exists.
    Returns the backup path, or None if no backup was needed.
    """
    if not OUTPUT_FILE.exists():
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = OUTPUT_DIR / f'fcp-euro-parts.{timestamp}.bak'
    shutil.copy2(OUTPUT_FILE, backup_path)
    logger.info(f"Created backup: {backup_path}")

    return backup_path


def write_products_to_csv(products: List[dict], product_line: str, source_brand: str,
                          page_url: str) -> None:
    """Write products to CSV file. Always appends if file exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = OUTPUT_FILE.exists()
    mode = 'a' if file_exists else 'w'
    write_header = not file_exists

    if not file_exists:
        logger.info(f"Creating new output file: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')

        if write_header:
            writer.writeheader()

        for product in products:
            product['product_line'] = product_line
            product['source_brand'] = source_brand
            product['page_url'] = page_url
            writer.writerow(product)


def deduplicate_csv() -> int:
    """Remove duplicate entries from the CSV file based on part_id."""
    if not OUTPUT_FILE.exists():
        return 0

    # Read all rows
    with open(OUTPUT_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    original_count = len(rows)

    # Deduplicate by part_id, keeping first occurrence
    seen = set()
    unique_rows = []
    for row in rows:
        part_id = row.get('part_id', '')
        if part_id and part_id not in seen:
            seen.add(part_id)
            unique_rows.append(row)
        elif not part_id:
            # Keep rows without part_id (shouldn't happen but just in case)
            unique_rows.append(row)

    # Write back
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(unique_rows)

    removed = original_count - len(unique_rows)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate entries")

    return removed


def scrape_category(brand_name: str, brand_slug: str, category: str,
                    config: dict, sort_order: Optional[str] = None,
                    overall_count: Optional[List[int]] = None) -> int:
    """
    Scrape all products from a category.
    Returns the number of products scraped.

    Args:
        sort_order: Optional sort parameter (e.g., 'desc_by_popularity')
        overall_count: Single-element list tracking cumulative total across all categories
    """
    category_slug = category_to_url_slug(category)
    base_url = f"{BASE_URL}/{brand_slug}/{category_slug}"
    if sort_order:
        base_url = f"{base_url}?order={sort_order}"

    logger.info(f"  Scraping {category}...")

    # Fetch first page to get pagination
    html = fetch_page(base_url, config)
    if not html:
        logger.warning(f"  Could not fetch {category}")
        return 0

    max_page = get_max_page(html)
    page_limit = config['settings'].get('max_pages')
    if page_limit:
        max_page = min(max_page, page_limit)

    logger.info(f"    Found {max_page} page(s)")

    total_products = 0

    for page in range(1, max_page + 1):
        if page > 1:
            page_url = f"{base_url}&page={page}" if sort_order else f"{base_url}?page={page}"
            time.sleep(get_random_delay(config))
            html = fetch_page(page_url, config)
            if not html:
                logger.warning(f"    Could not fetch page {page}")
                continue
        else:
            page_url = base_url

        # Extract products from both sources
        js_products = extract_js_products(html)
        html_products = extract_html_products(html)

        # Merge the data
        products = merge_products(js_products, html_products)

        if products:
            write_products_to_csv(products, category, brand_name, page_url)
            total_products += len(products)
            if overall_count is not None:
                overall_count[0] += len(products)
            overall_total = overall_count[0] if overall_count is not None else total_products
            logger.info(f"    Page {page}: {len(products)} products; "
                        f"{total_products} total for {category}; "
                        f"{overall_total} total")

    return total_products


def scrape_brand(brand_name: str, config: dict,
                 overall_count: Optional[List[int]] = None) -> int:
    """
    Scrape all enabled categories for a brand.
    Returns the total number of products scraped.
    """
    brand_slug = config['brand_slugs'].get(brand_name)
    if not brand_slug:
        logger.warning(f"No URL slug configured for brand: {brand_name}")
        return 0

    logger.info(f"Scraping {brand_name}...")

    enabled_categories = [
        cat for cat, enabled in config['product_lines'].items()
        if enabled
    ]

    total = 0

    for category in enabled_categories:
        count = scrape_category(brand_name, brand_slug, category, config,
                                sort_order='desc_by_popularity',
                                overall_count=overall_count)
        total += count
        time.sleep(get_random_delay(config))

    # Deduplicate after completing brand
    deduplicate_csv()

    logger.info(f"  {brand_name} complete: {total} products")
    return total


def main():
    parser = argparse.ArgumentParser(description='FCP Euro Parts Scraper')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parent.parent / 'cfg' / 'fcp_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--brand', type=str,
                        help='Scrape only this brand')
    parser.add_argument('--category', type=str,
                        help='Scrape only this category (requires --brand)')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh (delete existing output file)')
    parser.add_argument('--cookies', type=Path,
                        help='Path to cookies JSON file exported from browser')
    args = parser.parse_args()

    # Set up logging to stdout and file
    log_file = setup_logging()
    logger.info(f"Log file: {log_file}")

    # Select random browser profile for this run
    global BROWSER_PROFILE
    BROWSER_PROFILE = random.choice(BROWSER_PROFILES)
    logger.info(f"Browser profile: {BROWSER_PROFILE}")

    # Load cookies if provided
    global COOKIES
    if args.cookies:
        if not args.cookies.exists():
            logger.error(f"Cookies file not found: {args.cookies}")
            return 1
        COOKIES = load_cookies(args.cookies)
        logger.info(f"Loaded {len(COOKIES)} cookies from {args.cookies}")
    else:
        logger.info("No cookies file provided - requests will be unauthenticated")

    # Load config
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Run fcp_discover_categories.py first to generate it.")
        return 1

    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Always create backup of existing data before any scraping
    if OUTPUT_FILE.exists():
        backup_path = backup_csv()
        if backup_path:
            logger.info(f"Existing data backed up to: {backup_path}")

    # Fresh start if requested
    if args.fresh and OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        logger.info("Deleted existing output file (backup was created first)")

    # Log append/fresh status
    if OUTPUT_FILE.exists():
        existing_rows = sum(1 for _ in open(OUTPUT_FILE)) - 1  # subtract header
        logger.info(f"APPENDING to existing file with {existing_rows} rows")
    else:
        logger.info("Starting with new output file")

    # Determine what to scrape
    if args.brand:
        brands = [args.brand]
    else:
        brands = config.get('brands', [])

    if args.category:
        if not args.brand:
            logger.error("--category requires --brand to be specified")
            return 1
        # Temporarily override config to only scrape this category
        config['product_lines'] = {args.category: True}

    # Scrape
    overall_count = [0]
    try:
        for brand in brands:
            scrape_brand(brand, config, overall_count=overall_count)
    except BotBlockedException as e:
        logger.error(f"Scraping aborted: {e}")
        logger.info(f"Partial results saved. Total products before abort: {overall_count[0]}")
        logger.info(f"Output file: {OUTPUT_FILE}")
        return 2  # Exit code 2 indicates bot blocking

    grand_total = overall_count[0]
    logger.info(f"\nScraping complete! Total products: {grand_total}")
    logger.info(f"Output file: {OUTPUT_FILE}")

    return 0


if __name__ == '__main__':
    exit(main())
