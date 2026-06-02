#!/usr/bin/env python3
"""
Example: Scrape book data from Books to Scrape (a sandbox site).

This demonstrates the WebScraper class with a real-world scraping task:
extracting book titles, prices, availability, and links from a paginated
e-commerce catalogue.

Usage:
    python examples/scrape_books.py --pages 3 --output books.csv
"""

import argparse
import sys
from pathlib import Path

# Add parent dir to path so we can import scraper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper import WebScraper

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"


def scrape_books(max_pages: int = 3) -> list[dict]:
    """Scrape book data across multiple pages."""
    scraper = WebScraper(stealth=True, delay=1.0)

    all_books = []
    for page in range(1, max_pages + 1):
        url = BASE_URL.format(page)
        print(f"📄 Scraping page {page}...")

        data = scraper.scrape(url, config={
            "item_selector": "article.product_pod",
            "fields": {
                "title": "h3 a::attr(title)",
                "price": ".price_color::text",
                "availability": ".instock.availability::text",
                "rating": "p.star-rating::attr(class)",
                "link": "h3 a::attr(href)",
            },
        })

        all_books.extend(data)
        print(f"   → {len(data)} books found on page {page}")

    return all_books


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape books.toscrape.com")
    parser.add_argument("--pages", type=int, default=3, help="Number of pages to scrape")
    parser.add_argument("--output", "-o", default="books.csv", help="Output file (.csv, .json, .xlsx)")
    args = parser.parse_args()

    print(f"🚀 Starting scrape of {args.pages} pages...")
    books = scrape_books(max_pages=args.pages)

    if not books:
        print("❌ No books found!")
        return

    scraper = WebScraper()
    output_path = scraper.export(books, args.output)
    print(f"\n✅ Done! {len(books)} books saved to: {output_path}")

    # Show sample
    print("\n📚 First 3 books:")
    for b in books[:3]:
        print(f"   - {b['title']} | {b['price']} | ⭐ {b['rating']}")


if __name__ == "__main__":
    main()
