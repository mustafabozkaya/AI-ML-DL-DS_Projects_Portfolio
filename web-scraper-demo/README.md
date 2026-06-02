# 🌐 Web Scraper Demo

A production-grade web scraping toolkit built with Python, Requests, and BeautifulSoup4. 
Designed for data extraction, cleaning, and structured export.

> **Purpose:** Portfolio project for Upwork profile — demonstrates Python scripting, 
> web scraping, data extraction, and clean code practices.

---

## ✨ Features

- **Multi-site scraping** — Extract data from any public HTML website
- **Smart retry & error handling** — Automatic retry with exponential backoff
- **Proxy support** — Rotate user agents, handle rate limits
- **Structured output** — Export to CSV, JSON, or Excel
- **Stealth mode** — Bypass basic anti-scraping protections
- **CLI interface** — Easy to use from command line
- **Clean code** — Type hints, docstrings, modular design

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/MrBozkay/web-scraper-demo.git
cd web-scraper-demo

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# Scrape a simple page
python scraper.py --url https://books.toscrape.com --output books.csv

# Scrape with custom selectors
python scraper.py --url https://example.com/products \
  --item-selector ".product" \
  --fields "title:h2 a::text,price:.price::text,link:a::attr(href)" \
  --output products.json

# Stealth mode with rotating user agents
python scraper.py --url https://example.com --stealth --delay 2.0
```

---

## 🧪 Example: Scraping Books to Scrape

```python
from scraper import WebScraper
import json

scraper = WebScraper(stealth=True, delay=1.0)

# Scrape with custom configuration
data = scraper.scrape(
    url="https://books.toscrape.com/catalogue/page-1.html",
    config={
        "item_selector": "article.product_pod",
        "fields": {
            "title": "h3 a::attr(title)",
            "price": ".price_color::text",
            "availability": ".instock.availability::text",
            "link": "h3 a::attr(href)",
        }
    }
)

# Save results
with open("books.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Scraped {len(data)} books")
```

**Output sample:**
```json
[
  {
    "title": "A Light in the Attic",
    "price": "£51.77",
    "availability": "In stock",
    "link": "catalogue/a-light-in-the-attic_1000/index.html"
  }
]
```

---

## 🏗️ Project Structure

```
web-scraper-demo/
├── scraper.py          # Main scraper engine
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── examples/
    └── scrape_books.py  # Example usage script
```

## 🔧 Requirements

- Python 3.8+
- requests
- beautifulsoup4
- lxml
- pandas (optional, for Excel export)
- fake-useragent (optional, for stealth mode)

---

## 📋 Use Cases on Upwork

| Job Type | How This Helps |
|---|---|
| Web Scraping | Ready-to-use scraper for any site |
| Data Collection | Extract structured data from websites |
| Python Script | Demonstrates clean, modular Python |
| Data Extraction | CSV/JSON output for further processing |
| E-commerce Research | Scrape product prices, reviews, stock info |

---

## 📄 License

MIT — Free to use and modify for your projects.
