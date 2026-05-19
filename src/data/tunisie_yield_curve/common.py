"""Shared utilities for Tunisia Yield Curve extraction."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

YEAR = 2025
START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2025-12-31")
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
REQUEST_TIMEOUT = 75

SOVEREIGN_PUBLIC_URL = "https://www.tunisiayieldcurve.tn/public/"
SOVEREIGN_API_BASE = "https://www.tunisiayieldcurve.tn/service/api/Yield"
SOVEREIGN_REPORT_BASE = "https://www.tunisiayieldcurve.tn/yield/upload"

CORPORATE_PUBLIC_URL = "https://www.tunisiayieldcurve.tn/PublicCorporateSector/"
CORPORATE_API_BASE = "https://www.tunisiayieldcurve.tn/apiCorporateSector/api/Yield"
CORPORATE_REPORT_BASE = "https://www.tunisiayieldcurve.tn/CorporateCurveSector/upload"

CORPORATE_SECTORS = {
    "BANKING": "BANCAIRE",
    "LEASING": "LEASING",
    "MICROFINANCE": "MICROFINANCE",
}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class YieldCurvePaths:
    """Resolved paths for the Tunisia Yield Curve pipeline."""

    project_root: Path
    raw_dir: Path
    raw_sovereign_dir: Path
    raw_corporate_dir: Path
    interim_dir: Path
    processed_dir: Path
    endpoints_json: Path
    sovereign_raw_csv: Path
    corporate_raw_csv: Path
    scraping_log_csv: Path
    missing_dates_csv: Path
    sovereign_clean_csv: Path
    corporate_clean_csv: Path
    quality_report_xlsx: Path


def project_root_from_module() -> Path:
    """Infer repository root from this module path."""

    return Path(__file__).resolve().parents[3]


def resolve_paths(project_root: Path | None = None) -> YieldCurvePaths:
    """Resolve all pipeline paths relative to the project root."""

    root = Path(project_root).resolve() if project_root else project_root_from_module()
    raw = root / "data" / "raw" / "tunisie_yield_curve"
    interim = root / "data" / "interim" / "tunisie_yield_curve"
    processed = root / "data" / "processed" / "tunisie_yield_curve"
    return YieldCurvePaths(
        project_root=root,
        raw_dir=raw,
        raw_sovereign_dir=raw / "sovereign_daily_raw",
        raw_corporate_dir=raw / "corporate_daily_raw",
        interim_dir=interim,
        processed_dir=processed,
        endpoints_json=interim / "endpoints_discovered_2025.json",
        sovereign_raw_csv=interim / "sovereign_curves_daily_raw_2025.csv",
        corporate_raw_csv=interim / "corporate_curves_daily_raw_2025.csv",
        scraping_log_csv=interim / "scraping_log_2025.csv",
        missing_dates_csv=interim / "missing_dates_2025.csv",
        sovereign_clean_csv=processed / "sovereign_zero_coupon_curves_daily_2025.csv",
        corporate_clean_csv=processed / "corporate_curves_daily_2025.csv",
        quality_report_xlsx=processed / "curves_quality_report_2025.xlsx",
    )


def ensure_directories(paths: YieldCurvePaths) -> None:
    """Create all output directories used by the pipeline."""

    for path in [paths.raw_sovereign_dir, paths.raw_corporate_dir, paths.interim_dir, paths.processed_dir]:
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console logging for command-line execution."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def request_session() -> requests.Session:
    """Return a requests session with browser-like headers."""

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"})
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.75,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def all_2025_dates() -> pd.DatetimeIndex:
    """Return every calendar date in 2025."""

    return pd.date_range(START_DATE, END_DATE, freq="D")


def normalize_date(value: Any) -> str | None:
    """Normalize a date-like value to ``YYYY-MM-DD``."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?", text):
        parsed = pd.to_datetime(text[:10], format="%Y-%m-%d", errors="coerce")
        return None if pd.isna(parsed) else parsed.date().isoformat()
    parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def date_for_api(value: Any) -> str:
    """Format a date for JSON endpoints."""

    normalized = normalize_date(value)
    if normalized is None:
        raise ValueError(f"Invalid date: {value!r}")
    return normalized


def date_for_report(value: Any) -> str:
    """Format a date for the corporate Excel report endpoint."""

    normalized = normalize_date(value)
    if normalized is None:
        raise ValueError(f"Invalid date: {value!r}")
    return pd.Timestamp(normalized).strftime("%d/%m/%Y")


def parse_number(value: Any) -> float | None:
    """Parse French/English numeric text to float."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\xa0", "").replace(" ", "")
    if not text:
        return None
    text = text.replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def decimal_to_percent(value: Any) -> float | None:
    """Convert decimal rate to percent units."""

    number = parse_number(value)
    if number is None:
        return None
    return number * 100.0


def percent_to_decimal(value: Any) -> float | None:
    """Convert percent units to decimal rate."""

    number = parse_number(value)
    if number is None:
        return None
    return number / 100.0


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV with UTF-8 BOM for Excel compatibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a CSV if present, otherwise return an empty DataFrame."""

    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=list(columns or []))


def write_json(data: Any, path: Path) -> None:
    """Write indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_log(paths: YieldCurvePaths, rows: list[dict[str, Any]]) -> None:
    """Append extraction log rows to ``scraping_log_2025.csv``."""

    if not rows:
        return
    log = pd.DataFrame(rows)
    log["timestamp"] = log.get("timestamp", pd.Series([datetime.now().isoformat(timespec="seconds")] * len(log)))
    existing = read_csv(paths.scraping_log_csv)
    combined = pd.concat([existing, log], ignore_index=True) if not existing.empty else log
    dedupe_cols = ["date_requested", "curve_type", "sector", "status", "message", "extraction_method"]
    available = [col for col in dedupe_cols if col in combined.columns]
    if available:
        combined = combined.drop_duplicates(subset=available, keep="last")
    write_csv(combined, paths.scraping_log_csv)


def log_row(date_requested: str, curve_type: str, sector: str, status: str, message: str, method: str) -> dict[str, Any]:
    """Build a standardized scraping log row."""

    return {
        "date_requested": date_requested,
        "curve_type": curve_type,
        "sector": sector,
        "status": status,
        "message": message,
        "extraction_method": method,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
