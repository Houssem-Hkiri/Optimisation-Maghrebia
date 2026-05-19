"""Clean official sovereign zero-coupon curves."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .common import configure_logging, ensure_directories, normalize_date, parse_number, read_csv, resolve_paths, write_csv

LOGGER = logging.getLogger(__name__)

FINAL_COLUMNS = [
    "date",
    "maturity_years",
    "zero_coupon_rate",
    "zero_coupon_rate_percent",
    "source",
    "extraction_method",
    "data_quality_flag",
    "extraction_status",
]


def _flag(row: pd.Series) -> str:
    """Return quality flag for one sovereign curve point."""

    if pd.isna(row["date"]):
        return "INVALID_DATE"
    if pd.isna(row["maturity_years"]):
        return "UNRECOGNIZED_MATURITY"
    if pd.isna(row["zero_coupon_rate"]):
        return "MISSING_RATE"
    if row["zero_coupon_rate"] < 0:
        return "NEGATIVE_RATE"
    if row["zero_coupon_rate"] > 0.30:
        return "ABNORMAL_RATE_GT_30PCT"
    return "OK"


def clean_sovereign_curves(project_root: Path | None = None) -> pd.DataFrame:
    """Clean sovereign raw curve points without interpolation or filling."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    raw = read_csv(paths.sovereign_raw_csv)
    if raw.empty:
        clean = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        clean = pd.DataFrame(
            {
                "date": raw["curve_date"].map(normalize_date),
                "maturity_years": raw["maturity_years"].map(parse_number),
                "zero_coupon_rate": raw["zero_coupon_rate_decimal"].map(parse_number),
                "zero_coupon_rate_percent": raw["zero_coupon_rate_percent"].map(parse_number),
                "source": raw.get("source", ""),
                "extraction_method": raw.get("extraction_method", ""),
                "extraction_status": raw.get("extraction_status", ""),
            }
        )
        clean["data_quality_flag"] = clean.apply(_flag, axis=1)
        clean = clean[FINAL_COLUMNS].drop_duplicates()
        clean = clean.sort_values(["date", "maturity_years"], na_position="last").reset_index(drop=True)

    write_csv(clean, paths.sovereign_clean_csv)
    LOGGER.info("Saved clean sovereign curves: %s rows -> %s", len(clean), paths.sovereign_clean_csv)
    return clean


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    clean_sovereign_curves()


if __name__ == "__main__":
    main()
