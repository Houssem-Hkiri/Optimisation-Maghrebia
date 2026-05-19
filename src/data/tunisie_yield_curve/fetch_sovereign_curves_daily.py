"""Fetch daily official sovereign zero-coupon curves for 2025."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    SOVEREIGN_API_BASE,
    all_2025_dates,
    append_log,
    configure_logging,
    date_for_api,
    decimal_to_percent,
    ensure_directories,
    log_row,
    read_csv,
    request_session,
    resolve_paths,
    write_csv,
)

LOGGER = logging.getLogger(__name__)

RAW_COLUMNS = [
    "curve_date",
    "maturity",
    "maturity_unit",
    "maturity_years",
    "zero_coupon_rate_percent",
    "zero_coupon_rate_decimal",
    "source",
    "source_url",
    "extraction_method",
    "extraction_status",
    "raw_value",
]


def _published_curve_index(session) -> dict[str, int]:
    """Return a mapping ``YYYY-MM-DD -> ID_CURVE`` for published sovereign curves."""

    url = f"{SOVEREIGN_API_BASE}/GetPublishedCurve"
    response = session.get(url, timeout=90)
    response.raise_for_status()
    records = response.json()
    index: dict[str, int] = {}
    for row in records:
        date = date_for_api(row.get("DATE_CURVE"))
        if not date or not date.startswith("2025-"):
            continue
        if int(row.get("PUBLISHED", 0)) != 1:
            continue
        index[date] = int(row["ID_CURVE"])
    return index


def _rows_from_curve(curve_date: str, curve_id: int, payload: list[dict[str, Any]], source_url: str) -> list[dict[str, Any]]:
    """Normalize API payload rows into raw sovereign rows."""

    rows: list[dict[str, Any]] = []
    for item in payload:
        rate_decimal = item.get("TAUX_ZC_ACTUARIEL_365")
        rows.append(
            {
                "curve_date": curve_date,
                "maturity": item.get("MATURITE"),
                "maturity_unit": "YEARS",
                "maturity_years": item.get("MATURITE"),
                "zero_coupon_rate_percent": decimal_to_percent(rate_decimal),
                "zero_coupon_rate_decimal": rate_decimal,
                "source": "Tunisia Yield Curve sovereign API - YieldCurve",
                "source_url": source_url,
                "extraction_method": "requests_json",
                "extraction_status": "EXTRACTED",
                "raw_value": json.dumps(item, ensure_ascii=False),
            }
        )
    return rows


def fetch_sovereign_curves_daily(project_root: Path | None = None) -> pd.DataFrame:
    """Fetch official sovereign zero-coupon curves for all available 2025 dates."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    session = request_session()
    existing = read_csv(paths.sovereign_raw_csv, RAW_COLUMNS)
    existing_dates = set(existing["curve_date"].astype(str)) if not existing.empty else set()
    published = _published_curve_index(session)
    rows = existing.to_dict("records") if not existing.empty else []
    logs: list[dict[str, Any]] = []

    for date in all_2025_dates():
        date_str = date.date().isoformat()
        if date_str in existing_dates:
            continue
        curve_id = published.get(date_str)
        if curve_id is None:
            logs.append(log_row(date_str, "SOVEREIGN", "", "MISSING", "No published sovereign curve for this date.", "requests_json"))
            continue
        raw_path = paths.raw_sovereign_dir / f"{date_str}.json"
        source_url = f"{SOVEREIGN_API_BASE}/YieldCurve?ID_CURVE={curve_id}"
        try:
            if raw_path.exists():
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                response = session.get(source_url, timeout=90)
                response.raise_for_status()
                payload = response.json()
                raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if not payload:
                logs.append(log_row(date_str, "SOVEREIGN", "", "MISSING", f"Empty curve payload for ID_CURVE={curve_id}.", "requests_json"))
                continue
            rows.extend(_rows_from_curve(date_str, curve_id, payload, source_url))
            logs.append(log_row(date_str, "SOVEREIGN", "", "EXTRACTED", f"ID_CURVE={curve_id}; points={len(payload)}", "requests_json"))
        except Exception as exc:
            logs.append(log_row(date_str, "SOVEREIGN", "", "FAILED", f"{type(exc).__name__}: {exc}", "requests_json"))

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS).drop_duplicates()
    write_csv(frame, paths.sovereign_raw_csv)
    append_log(paths, logs)
    LOGGER.info("Saved sovereign raw curves: %s rows -> %s", len(frame), paths.sovereign_raw_csv)
    return frame


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    fetch_sovereign_curves_daily()


if __name__ == "__main__":
    main()
