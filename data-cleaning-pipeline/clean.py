#!/usr/bin/env python3
"""
Data Cleaning Pipeline — Automated data quality toolkit.

Detects and fixes common data issues:
  - Missing values, duplicates, outliers
  - Type mismatches, date format inconsistencies
  - Whitespace issues, categorical inconsistencies
  - Encoding problems

Usage:
    python clean.py --input data/messy_data.csv --output data/cleaned.csv
    python clean.py --input data/messy_data.csv --profile-only
    python clean.py --input data/messy_data.csv --config cleaning_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress chained assignment warnings from pandas
pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------
# Cleaning Report Data Model
# ---------------------------------------------------------------------------

@dataclass
class CleaningReport:
    """Detailed record of all cleaning operations performed."""

    total_rows_before: int = 0
    total_rows_after: int = 0
    duplicates_removed: int = 0
    missing_filled: dict[str, int] = field(default_factory=dict)
    outliers_removed: int = 0
    dtypes_fixed: list[str] = field(default_factory=list)
    whitespace_fixed: int = 0
    dates_standardized: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_fixes(self) -> int:
        return (
            self.duplicates_removed
            + sum(self.missing_filled.values())
            + self.outliers_removed
            + len(self.dtypes_fixed)
            + self.whitespace_fixed
            + self.dates_standardized
        )

    def to_dict(self) -> dict:
        return {
            "total_rows_before": self.total_rows_before,
            "total_rows_after": self.total_rows_after,
            "rows_removed": self.total_rows_before - self.total_rows_after,
            "duplicates_removed": self.duplicates_removed,
            "missing_filled": self.missing_filled,
            "outliers_removed": self.outliers_removed,
            "dtypes_fixed": self.dtypes_fixed,
            "whitespace_fixed": self.whitespace_fixed,
            "dates_standardized": self.dates_standardized,
            "total_fixes": self.total_fixes,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Cleaning Pipeline
# ---------------------------------------------------------------------------

class CleaningPipeline:
    """
    A chainable data cleaning pipeline with automatic profiling and fixes.

    Usage:
        pipeline = CleaningPipeline("data.csv")
        df = pipeline.drop_duplicates().fill_missing().remove_outliers().execute()
    """

    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.df: pd.DataFrame = self._load_data()
        self._operations: list[str] = []
        self.report = CleaningReport(
            total_rows_before=len(self.df),
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def profile(self) -> dict:
        """Generate a comprehensive data profile."""
        df = self.df
        profile = {
            "filename": self.input_path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "statistics": {},
        }

        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "null_pct": round(float(df[col].isna().mean() * 100), 2),
                "unique": int(df[col].nunique()),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                })
            elif pd.api.types.is_string_dtype(df[col]):
                col_info.update({
                    "min_length": int(df[col].str.len().min()) if not df[col].isna().all() else None,
                    "max_length": int(df[col].str.len().max()) if not df[col].isna().all() else None,
                })

            profile["statistics"][col] = col_info

        return profile

    def auto_clean(self) -> pd.DataFrame:
        """
        Run the full automatic cleaning pipeline:
        1. Drop fully empty rows/columns
        2. Trim whitespace
        3. Fix data types
        4. Drop duplicates
        5. Fill missing values
        6. Remove outliers
        7. Standardize dates
        """
        logger.info("Starting auto-clean pipeline...")
        return (
            self.drop_empty()
            .trim_whitespace()
            .fix_dtypes()
            .drop_duplicates()
            .fill_missing()
            .remove_outliers()
            .standardize_dates()
            .execute()
        )

    def drop_empty(self) -> "CleaningPipeline":
        """Drop rows and columns that are completely empty."""
        before = len(self.df)
        self.df = self.df.dropna(how="all")
        self.df = self.df.dropna(axis=1, how="all")
        removed = before - len(self.df)
        if removed:
            logger.info("Dropped %d fully empty rows/columns", removed)
            self._operations.append(f"drop_empty: {removed}")
        return self

    def drop_duplicates(self) -> "CleaningPipeline":
        """Remove duplicate rows."""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        self.report.duplicates_removed = before - len(self.df)
        if self.report.duplicates_removed:
            logger.info("Removed %d duplicate rows", self.report.duplicates_removed)
            self._operations.append(f"drop_duplicates: {self.report.duplicates_removed}")
        return self

    def trim_whitespace(self) -> "CleaningPipeline":
        """Trim leading/trailing whitespace from all string columns."""
        count = 0
        for col in self.df.select_dtypes(include=["object", "string"]).columns:
            trimmed = self.df[col].astype(str).str.strip()
            mask = self.df[col].astype(str) != trimmed
            if mask.any():
                self.df[col] = self.df[col].astype(str).str.strip()
                count += int(mask.sum())

        self.report.whitespace_fixed = count
        if count:
            logger.info("Trimmed whitespace in %d cells", count)
            self._operations.append(f"trim_whitespace: {count}")
        return self

    def fix_dtypes(self) -> "CleaningPipeline":
        """
        Automatically fix data type issues:
        - Convert numeric strings to numbers
        - Parse date strings
        - Convert booleans
        """
        fixed: list[str] = []
        for col in self.df.columns:
            if self.df[col].dtype != object:
                continue

            # Try numeric conversion
            numeric_sample = pd.to_numeric(self.df[col], errors="coerce")
            if numeric_sample.notna().sum() > len(self.df) * 0.8:
                self.df[col] = numeric_sample
                fixed.append(f"{col} → numeric")
                continue

            # Try datetime conversion
            try:
                date_sample = pd.to_datetime(self.df[col], errors="coerce", dayfirst=True)
                if date_sample.notna().sum() > len(self.df) * 0.8:
                    self.df[col] = date_sample
                    fixed.append(f"{col} → datetime")
                    continue
            except (ValueError, TypeError):
                pass

        self.report.dtypes_fixed = fixed
        if fixed:
            logger.info("Fixed dtypes: %s", ", ".join(fixed))
            self._operations.extend(fixed)
        return self

    def fill_missing(self, strategy: str = "auto") -> "CleaningPipeline":
        """
        Fill missing values intelligently.

        Strategies:
        - 'auto': numeric → median, categorical → mode
        - 'mean': numeric columns only
        - 'median': numeric columns only
        - 'mode': most frequent value
        - 'drop': remove rows with missing values
        - 'zero': fill with 0
        - str: fill with a specific value
        """
        filled: dict[str, int] = {}

        for col in self.df.columns:
            null_count = int(self.df[col].isna().sum())
            if null_count == 0:
                continue

            if strategy == "drop":
                self.df = self.df.dropna(subset=[col])
                filled[col] = null_count
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                if strategy in ("auto", "median"):
                    fill_val = self.df[col].median()
                elif strategy == "mean":
                    fill_val = self.df[col].mean()
                elif strategy == "zero":
                    fill_val = 0
                else:
                    fill_val = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 0
            else:
                if strategy in ("auto", "mode"):
                    fill_val = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "Unknown"
                elif strategy == "zero":
                    fill_val = "0"
                else:
                    fill_val = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "Unknown"

            self.df[col] = self.df[col].fillna(fill_val)
            filled[col] = null_count

        self.report.missing_filled = filled
        if filled:
            total = sum(filled.values())
            logger.info("Filled %d missing values across %d columns", total, len(filled))
            self._operations.append(f"fill_missing: {total}")
        return self

    def remove_outliers(self, method: str = "iqr", threshold: float = 1.5) -> "CleaningPipeline":
        """
        Remove outliers from numeric columns.

        Methods:
        - 'iqr': Interquartile range (default, threshold=1.5)
        - 'zscore': Z-score (threshold=3)
        """
        before = len(self.df)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if method == "iqr":
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]

        elif method == "zscore":
            for col in numeric_cols:
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]

        self.report.outliers_removed = before - len(self.df)
        if self.report.outliers_removed:
            logger.info("Removed %d outlier rows", self.report.outliers_removed)
            self._operations.append(f"remove_outliers: {self.report.outliers_removed}")
        return self

    def standardize_dates(self) -> "CleaningPipeline":
        """Standardize all datetime columns to ISO format."""
        count = 0
        for col in self.df.select_dtypes(include=["datetime64"]).columns:
            before = self.df[col].isna().sum()
            self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
            # Format as ISO string for consistent export
            self.df[col] = self.df[col].dt.strftime("%Y-%m-%d")
            count += int(self.df[col].notna().sum())

        self.report.dates_standardized = count
        if count:
            logger.info("Standardized %d date values", count)
            self._operations.append(f"standardize_dates: {count}")
        return self

    def execute(self) -> pd.DataFrame:
        """Finalize and return the cleaned DataFrame."""
        self.report.total_rows_after = len(self.df)
        logger.info(
            "Cleaning complete: %d → %d rows (%d removed), %d total fixes",
            self.report.total_rows_before,
            self.report.total_rows_after,
            self.report.total_rows_before - self.report.total_rows_after,
            self.report.total_fixes,
        )
        return self.df

    def get_report(self) -> CleaningReport:
        """Get the cleaning report."""
        return self.report

    def export(self, output_path: str) -> str:
        """
        Export cleaned data to a file. Format inferred from extension.
        Supported: .csv, .json, .xlsx, .parquet
        """
        path = Path(output_path)
        suffix = path.suffix.lower()

        export_map = {
            ".csv": lambda: self.df.to_csv(path, index=False, encoding="utf-8-sig"),
            ".json": lambda: self.df.to_json(path, orient="records", indent=2, date_format="iso"),
            ".xlsx": lambda: self.df.to_excel(path, index=False, engine="openpyxl"),
            ".parquet": lambda: self.df.to_parquet(path, index=False),
        }

        exporter = export_map.get(suffix)
        if exporter is None:
            raise ValueError(f"Unsupported format: {suffix}. Use .csv, .json, .xlsx, .parquet")

        exporter()
        logger.info("Exported cleaned data to %s", path.resolve())
        return str(path.resolve())

    # -----------------------------------------------------------------------
    # Internal: Data Loading
    # -----------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """Load data from file, inferring format from extension."""
        path = self.input_path
        suffix = path.suffix.lower()

        loaders = {
            ".csv": lambda: pd.read_csv(path, encoding_errors="replace"),
            ".tsv": lambda: pd.read_csv(path, sep="\t", encoding_errors="replace"),
            ".xlsx": lambda: pd.read_excel(path, engine="openpyxl"),
            ".xls": lambda: pd.read_excel(path, engine="xlrd"),
            ".json": lambda: pd.read_json(path),
            ".parquet": lambda: pd.read_parquet(path),
        }

        loader = loaders.get(suffix)
        if loader is None:
            raise ValueError(f"Unsupported input format: {suffix}")

        df = loader()
        logger.info("Loaded %s: %d rows × %d columns", path.name, len(df), len(df.columns))
        return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Data Cleaning Pipeline — automated data quality toolkit",
    )
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", default="", help="Output file path")
    parser.add_argument("--config", "-c", default="", help="YAML configuration file")
    parser.add_argument("--profile-only", action="store_true", help="Only show data profile, don't clean")
    parser.add_argument("--report", "-r", default="", help="Save cleaning report to JSON")
    parser.add_argument("--fill-strategy", default="auto", help="Missing value strategy")
    parser.add_argument("--outlier-method", default="iqr", choices=["iqr", "zscore"], help="Outlier detection method")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load data
    pipeline = CleaningPipeline(args.input)

    # Profile only mode
    if args.profile_only:
        profile = pipeline.profile()
        print(json.dumps(profile, indent=2, default=str))
        return

    # Load config if provided
    fill_strategy = args.fill_strategy
    outlier_method = args.outlier_method

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        fill_strategy = config.get("fill_strategy", fill_strategy)
        outlier_method = config.get("outlier_method", outlier_method)

    # Run cleaning
    pipeline.fill_missing(strategy=fill_strategy)
    pipeline.remove_outliers(method=outlier_method)
    df = pipeline.execute()

    # Output
    if args.output:
        pipeline.export(args.output)
    else:
        print(df.head(10).to_string())

    # Report
    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(pipeline.get_report().to_dict(), indent=2))
        logger.info("Report saved to %s", report_path.resolve())

    print(f"\n✅ Cleaning complete. {len(df)} rows, {len(df.columns)} columns.")


if __name__ == "__main__":
    main()
