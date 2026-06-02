# 🧹 Data Cleaning Pipeline

A comprehensive data cleaning and preprocessing toolkit built with Python and Pandas.
Automatically detects and fixes common data quality issues — missing values, duplicates,
outliers, type mismatches, and formatting inconsistencies.

> **Purpose:** Portfolio project for Upwork — demonstrates data cleaning, ETL pipeline 
> development, data analysis, and Python scripting skills.

---

## ✨ Features

- **Automatic data profiling** — Detect data types, missing values, duplicates, outliers
- **Intelligent cleaning** — Auto-fix mode for common issues
- **Multiple input formats** — CSV, Excel, JSON, Parquet
- **Configurable cleaning rules** — YAML configuration files
- **Detailed cleaning report** — See exactly what was changed
- **CLI & Python API** — Use as a command-line tool or import as a library
- **Pipeline chaining** — Build multi-stage cleaning workflows

---

## 📦 Installation

```bash
git clone https://github.com/MrBozkay/data-cleaning-pipeline.git
cd data-cleaning-pipeline
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# Quick auto-clean
python clean.py --input data/messy_data.csv --output data/cleaned.csv

# Generate a profiling report
python clean.py --input data/messy_data.csv --profile-only

# Custom cleaning with config
python clean.py --input data/messy_data.csv --config cleaning_config.yaml

# Export cleaning report
python clean.py --input data/messy_data.csv --report cleaning_report.json
```

---

## 🐍 Python API

```python
from clean import CleaningPipeline

# Load and auto-clean
pipeline = CleaningPipeline("data/messy_data.csv")
df = pipeline.auto_clean()

# Custom cleaning pipeline
pipeline = CleaningPipeline("data/messy_data.csv")
df = (pipeline
    .drop_duplicates()
    .fill_missing(strategy="mean")  
    .fix_dtypes()
    .remove_outliers(method="iqr", threshold=1.5)
    .standardize_dates()
    .trim_whitespace()
    .execute())

# Get cleaning report
report = pipeline.get_report()
print(f"Fixed {report['total_fixes']} issues across {report['columns_fixed']} columns")

# Export
pipeline.export("data/cleaned_data.xlsx")
```

---

## 🏗️ Project Structure

```
data-cleaning-pipeline/
├── clean.py              # Main cleaning engine
├── cleaning_config.yaml  # Sample configuration file
├── requirements.txt      # Dependencies
├── README.md            # This file
└── data/
    └── messy_data.csv   # Sample messy dataset
```

## 🔧 Requirements

- Python 3.8+
- pandas
- numpy
- pyyaml
- openpyxl (for Excel export)
- tabulate (for console reports)

---

## 📋 Use Cases on Upwork

| Job Type | How This Helps |
|---|---|
| Data Cleaning | Directly solves the problem |
| Data Analysis | Provides clean data for analysis |
| Python Script | Demonstrates modular Python engineering |
| Data Processing | ETL pipeline foundation |
| Data Entry | Validate and clean imported data |
| Lead List Building | Clean/validate lead lists |

---

## 📄 License

MIT — Free to use and modify for your projects.
