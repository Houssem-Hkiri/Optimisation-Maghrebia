"""Clean official corporate curve data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .common import configure_logging, ensure_directories, normalize_date, parse_number, read_csv, resolve_paths, write_csv

LOGGER = logging.getLogger(__name__)
SECTOR_MAP = {"BANKING": "BANCAIRE", "BANCAIRE": "BANCAIRE", "LEASING": "LEASING", "MICROFINANCE": "MICROFINANCE"}

FINAL_COLUMNS = [
    "date",
    "sector",
    "maturity_years",
    "corporate_rate",
    "corporate_rate_percent",
    "ytm",
    "ytm_percent",
    "dirty_price",
    "clean_price",
    "source",
    "extraction_method",
    "data_quality_flag",
    "extraction_status",
]


def _sector(value: object) -> str:
    """Normalize corporate sector labels."""

    return SECTOR_MAP.get(str(value or "").strip().upper(), str(value or "").strip().upper())


def _flag(row: pd.Series) -> str:
    """Return quality flag for one corporate observation."""

    if pd.isna(row["date"]):
        return "INVALID_DATE"
    if row["sector"] not in {"BANCAIRE", "LEASING", "MICROFINANCE"}:
        return "UNRECOGNIZED_SECTOR"
    if pd.isna(row["maturity_years"]):
        return "UNRECOGNIZED_MATURITY"
    if pd.isna(row["corporate_rate"]) and pd.isna(row["ytm"]):
        return "MISSING_RATE"
    for field in ["corporate_rate", "ytm"]:
        value = row.get(field)
        if pd.notna(value) and value < 0:
            return "NEGATIVE_RATE"
        if pd.notna(value) and value > 0.50:
            return "ABNORMAL_RATE_GT_50PCT"
    return "OK"


def clean_corporate_curves(project_root: Path | None = None) -> pd.DataFrame:
    """Clean corporate raw curve observations without interpolation or sector mixing."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    raw = read_csv(paths.corporate_raw_csv)
    if raw.empty:
        clean = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        clean = pd.DataFrame(
            {
                "date": raw["curve_date"].map(normalize_date),
                "sector": raw["sector"].map(_sector),
                "maturity_years": raw["maturity_years"].map(parse_number),
                "corporate_rate": raw["corporate_rate_decimal"].map(parse_number),
                "corporate_rate_percent": raw["corporate_rate_percent"].map(parse_number),
                "ytm": raw["ytm_decimal"].map(parse_number),
                "ytm_percent": raw["ytm_percent"].map(parse_number),
                "dirty_price": raw["dirty_price"].map(parse_number),
                "clean_price": raw["clean_price"].map(parse_number),
                "source": raw.get("source", ""),
                "extraction_method": raw.get("extraction_method", ""),
                "extraction_status": raw.get("extraction_status", ""),
            }
        )
        clean["data_quality_flag"] = clean.apply(_flag, axis=1)
        clean = clean[FINAL_COLUMNS].drop_duplicates()
        clean = clean.sort_values(["date", "sector", "maturity_years"], na_position="last").reset_index(drop=True)

    write_csv(clean, paths.corporate_clean_csv)
    LOGGER.info("Saved clean corporate curves: %s rows -> %s", len(clean), paths.corporate_clean_csv)
    return clean


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    clean_corporate_curves()


if __name__ == "__main__":
    main()
