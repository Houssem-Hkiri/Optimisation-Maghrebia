"""Build quality controls for Tunisia Yield Curve 2025 datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .common import all_2025_dates, configure_logging, ensure_directories, read_csv, resolve_paths, write_csv

LOGGER = logging.getLogger(__name__)


def _missing_dates(sovereign: pd.DataFrame, corporate: pd.DataFrame) -> pd.DataFrame:
    """Build the missing-date control table."""

    dates = pd.DataFrame({"date": [d.date().isoformat() for d in all_2025_dates()]})
    sov_dates = set(sovereign.get("date", pd.Series(dtype=str)).dropna().astype(str))
    corp_dates = set(corporate.get("date", pd.Series(dtype=str)).dropna().astype(str))
    dates["sovereign_available"] = dates["date"].isin(sov_dates)
    dates["corporate_available"] = dates["date"].isin(corp_dates)
    dates["missing_sovereign"] = ~dates["sovereign_available"]
    dates["missing_corporate"] = ~dates["corporate_available"]
    return dates


def _coverage_by_month(missing: pd.DataFrame) -> pd.DataFrame:
    """Summarize availability by month."""

    frame = missing.copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M").astype(str)
    return (
        frame.groupby("month")
        .agg(
            dates_requested=("date", "count"),
            sovereign_dates=("sovereign_available", "sum"),
            corporate_dates=("corporate_available", "sum"),
            sovereign_missing=("missing_sovereign", "sum"),
            corporate_missing=("missing_corporate", "sum"),
        )
        .reset_index()
    )


def _sector_coverage(corporate: pd.DataFrame) -> pd.DataFrame:
    """Summarize corporate availability by sector and month."""

    if corporate.empty:
        return pd.DataFrame(columns=["month", "sector", "dates_available", "observations"])
    frame = corporate.copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M").astype(str)
    return (
        frame.groupby(["month", "sector"], dropna=False)
        .agg(dates_available=("date", "nunique"), observations=("date", "size"))
        .reset_index()
    )


def _anomalies(sovereign: pd.DataFrame, corporate: pd.DataFrame) -> pd.DataFrame:
    """Collect duplicate, missing maturity, and abnormal-rate controls."""

    rows: list[dict[str, object]] = []
    if not sovereign.empty:
        dup = sovereign.duplicated(subset=["date", "maturity_years"], keep=False)
        rows.extend({"dataset": "sovereign", "issue": "DUPLICATE_DATE_MATURITY", **r} for r in sovereign[dup].to_dict("records"))
        bad = sovereign[sovereign["data_quality_flag"].ne("OK")]
        rows.extend({"dataset": "sovereign", "issue": r.get("data_quality_flag"), **r} for r in bad.to_dict("records"))
    if not corporate.empty:
        dup = corporate.duplicated(subset=["date", "sector", "maturity_years", "ytm", "clean_price", "dirty_price"], keep=False)
        rows.extend({"dataset": "corporate", "issue": "DUPLICATE_OBSERVATION", **r} for r in corporate[dup].to_dict("records"))
        bad = corporate[corporate["data_quality_flag"].ne("OK")]
        rows.extend({"dataset": "corporate", "issue": r.get("data_quality_flag"), **r} for r in bad.to_dict("records"))
    return pd.DataFrame(rows)


def build_quality_report(project_root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Write the 2025 yield-curve quality report workbook."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    sovereign = read_csv(paths.sovereign_clean_csv)
    corporate = read_csv(paths.corporate_clean_csv)
    log = read_csv(paths.scraping_log_csv)
    missing = _missing_dates(sovereign, corporate)
    coverage = _coverage_by_month(missing)
    sector_coverage = _sector_coverage(corporate)
    anomalies = _anomalies(sovereign, corporate)
    write_csv(missing, paths.missing_dates_csv)

    sheets = {
        "Sovereign_Curves": sovereign,
        "Corporate_Curves": corporate,
        "Missing_Dates": missing,
        "Extraction_Log": log,
        "Coverage_By_Month": coverage,
        "Corporate_Sectors_Coverage": sector_coverage,
        "Anomalies": anomalies,
    }
    paths.quality_report_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(paths.quality_report_xlsx, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    LOGGER.info("Saved quality report -> %s", paths.quality_report_xlsx)
    return sheets


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    build_quality_report()


if __name__ == "__main__":
    main()
