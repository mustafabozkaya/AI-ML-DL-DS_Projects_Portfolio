#!/usr/bin/env python3
"""
Web Scraper Demo — Production-grade web scraping toolkit.

A modular, well-documented web scraper with:
  - Requests + BeautifulSoup4 engine
  - Retry logic with exponential backoff
  - Stealth mode (rotating user agents)
  - CSV, JSON, and Excel export
  - CLI interface via argparse

Usage:
    python scraper.py --url https://books.toscrape.com --output data.csv
    python scraper.py --url https://example.com --stealth --delay 2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Optional imports — degrade gracefully if not installed
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from fake_useragent import UserAgent
    HAS_UA = True
except ImportError:
    HAS_UA = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ScraperConfig:
    """Configuration for a scraping job."""

    url: str
    item_selector: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    output: str = ""
    stealth: bool = False
    delay: float = 1.0
    max_retries: int = 3
    timeout: int = 30


# ---------------------------------------------------------------------------
# Scraper Engine
# ---------------------------------------------------------------------------

class WebScraper:
    """
    A reusable web scraper with retry logic, stealth mode, and multiple export formats.

    Usage:
        scraper = WebScraper(stealth=True, delay=1.5)
        data = scraper.scrape("https://example.com", config={...})
        scraper.export(data, "output.csv")
    """

    def __init__(
        self,
        stealth: bool = False,
        delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.stealth = stealth
        self.delay = delay
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(self._get_headers())

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def scrape(
        self,
        url: str,
        config: Optional[dict] = None,
    ) -> list[dict[str, str]]:
        """
        Scrape a URL and return structured data.

        Args:
            url: Target URL to scrape.
            config: Optional dict with:
                - item_selector: CSS selector for each item container
                - fields: dict mapping field names to CSS selectors + extraction

        Returns:
            List of dicts with extracted data.
        """
        config = config or {}
        item_selector = config.get("item_selector", "")
        fields = config.get("fields", {})

        html = self._fetch(url)
        soup = BeautifulSoup(html, "lxml")

        if not item_selector:
            # Return full page text as a single entry
            return [{"content": soup.get_text(strip=True)}]

        items = soup.select(item_selector)
        logger.info("Found %d items with selector '%s'", len(items), item_selector)

        results: list[dict[str, str]] = []
        for item in items:
            row: dict[str, str] = {}
            for field_name, selector_expr in fields.items():
                row[field_name] = self._extract_field(item, selector_expr)
            results.append(row)

        return results

    def export(self, data: list[dict[str, Any]], output_path: str) -> str:
        """
        Export scraped data to a file. Format is inferred from the extension.

        Supported formats: .csv, .json, .xlsx (if pandas is installed).

        Args:
            data: List of dicts to export.
            output_path: File path for the output.

        Returns:
            Absolute path to the created file.
        """
        path = Path(output_path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            self._export_csv(data, path)
        elif suffix == ".json":
            self._export_json(data, path)
        elif suffix == ".xlsx" and HAS_PANDAS:
            self._export_excel(data, path)
        else:
            msg = f"Unsupported format: {suffix}. Use .csv, .json, or .xlsx"
            raise ValueError(msg)

        logger.info("Exported %d records to %s", len(data), path.resolve())
        return str(path.resolve())

    # -----------------------------------------------------------------------
    # Internal: Network
    # -----------------------------------------------------------------------

    def _fetch(self, url: str) -> str:
        """Fetch URL with retry logic and exponential backoff."""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.stealth:
                    self._rotate_headers()

                logger.info("Fetching [%d/%d]: %s", attempt, self.max_retries, url)
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()

                # Respect robots.txt implicitly via delays
                if self.delay > 0:
                    time.sleep(self.delay)

                return resp.text

            except requests.RequestException as e:
                last_error = e
                logger.warning("Attempt %d failed: %s", attempt, e)
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.info("Retrying in %ds...", wait)
                    time.sleep(wait)

        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts") from last_error

    # -----------------------------------------------------------------------
    # Internal: Parsing
    # -----------------------------------------------------------------------

    def _extract_field(self, item: BeautifulSoup, selector: str) -> str:
        """
        Extract a single field using a selector expression.

        Supports:
          - "tag::text"  → get text content
          - "tag::attr(href)" → get attribute value
          - "tag"  → get text by default
        """
        if "::" in selector:
            css_sel, action = selector.split("::", 1)
        else:
            css_sel, action = selector, "text"

        elements = item.select(css_sel)
        if not elements:
            return ""

        if action.startswith("attr("):
            attr_name = action[5:-1]  # "attr(href)" → "href"
            return elements[0].get(attr_name, "").strip()
        else:
            return elements[0].get_text(strip=True)

    # -----------------------------------------------------------------------
    # Internal: Stealth
    # -----------------------------------------------------------------------

    def _get_headers(self) -> dict[str, str]:
        """Generate initial headers."""
        if self.stealth and HAS_UA:
            try:
                ua = UserAgent()
                return {"User-Agent": ua.random}
            except Exception:
                pass
        return {"User-Agent": random.choice(DEFAULT_USER_AGENTS)}

    def _rotate_headers(self) -> None:
        """Rotate User-Agent and add random cache-busting headers."""
        self._session.headers.update(self._get_headers())
        # Add a cache-busting Accept header
        self._session.headers["Accept"] = (
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        )

    # -----------------------------------------------------------------------
    # Internal: Export
    # -----------------------------------------------------------------------

    def _export_csv(self, data: list[dict], path: Path) -> None:
        """Export to CSV with UTF-8 BOM for Excel compatibility."""
        if not data:
            path.write_text("")
            return
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    def _export_json(self, data: list[dict], path: Path) -> None:
        """Export to JSON with indentation."""
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _export_excel(self, data: list[dict], path: Path) -> None:
        """Export to Excel using pandas."""
        df = pd.DataFrame(data)
        df.to_excel(path, index=False, engine="openpyxl")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Production-grade web scraper with stealth mode and multiple export formats.",
    )
    parser.add_argument("--url", required=True, help="Target URL to scrape")
    parser.add_argument("--item-selector", default="", help="CSS selector for each item (e.g., 'article.product')")
    parser.add_argument("--fields", nargs="*", default=[], help="Field selectors: 'name:.title::text' 'price:.price::text'")
    parser.add_argument("--output", "-o", default="output.csv", help="Output file path (.csv, .json, .xlsx)")
    parser.add_argument("--stealth", action="store_true", help="Enable stealth mode (rotate user agents)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    return parser


def parse_fields(field_args: list[str]) -> dict[str, str]:
    """Parse --fields arguments into a dict."""
    fields: dict[str, str] = {}
    for arg in field_args:
        if ":" in arg:
            key, value = arg.split(":", 1)
        else:
            key, value = arg, "::text"
        fields[key.strip()] = value.strip()
    return fields


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    scraper = WebScraper(
        stealth=args.stealth,
        delay=args.delay,
        max_retries=args.max_retries,
    )

    fields = parse_fields(args.fields)

    config = {
        "item_selector": args.item_selector,
        "fields": fields,
    }

    try:
        data = scraper.scrape(args.url, config=config)
        output_path = scraper.export(data, args.output)
        print(f"\n✅ Done! {len(data)} records saved to: {output_path}")
    except Exception as e:
        logger.error("Scraping failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
