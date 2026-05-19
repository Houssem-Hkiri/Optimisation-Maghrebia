"""Fetch daily official corporate curve data for 2025."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    CORPORATE_API_BASE,
    CORPORATE_REPORT_BASE,
    CORPORATE_SECTORS,
    all_2025_dates,
    append_log,
    configure_logging,
    date_for_report,
    decimal_to_percent,
    ensure_directories,
    log_row,
    parse_number,
    read_csv,
    request_session,
    resolve_paths,
    write_csv,
)

LOGGER = logging.getLogger(__name__)

RAW_COLUMNS = [
    "curve_date",
    "sector",
    "maturity",
    "maturity_unit",
    "maturity_years",
    "corporate_rate_percent",
    "corporate_rate_decimal",
    "ytm_percent",
    "ytm_decimal",
    "dirty_price",
    "clean_price",
    "source",
    "source_url",
    "extraction_method",
    "extraction_status",
    "raw_value",
]


def _sector_state(session, date_str: str, sector_api: str, paths) -> dict[str, Any] | None:
    """Fetch and cache corporate sector state for one date."""

    raw_json = paths.raw_corporate_dir / f"{date_str}_{sector_api}_state.json"
    if raw_json.exists():
        payload = json.loads(raw_json.read_text(encoding="utf-8"))
        return payload or None
    url = f"{CORPORATE_API_BASE}/GetCorporateJournee?DATE_CORPORATE={date_str}&TYPE_SECTEUR={sector_api}"
    response = session.get(url, timeout=75)
    response.raise_for_status()
    payload = response.json()
    raw_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload or None


def _download_actuarial_report(session, date_str: str, paths) -> Path | None:
    """Generate and download the official corporate actuarial Excel report for one date."""

    raw_xlsx = paths.raw_corporate_dir / f"{date_str}_actuarial.xlsx"
    if raw_xlsx.exists() and raw_xlsx.stat().st_size > 0:
        return raw_xlsx

    report_date = date_for_report(date_str)
    url = f"{CORPORATE_API_BASE}/ReportActuariel?DATES={report_date}&TYPE_SECTEUR=LEASING"
    response = session.get(url, timeout=120)
    if response.status_code >= 400:
        return None
    payload = response.json()
    if str(payload.get("success")) != "True" or not payload.get("fileName"):
        return None
    download_url = f"{CORPORATE_REPORT_BASE}/{payload['fileName']}"
    download = session.get(download_url, timeout=120)
    download.raise_for_status()
    if not download.content.startswith(b"PK"):
        return None
    raw_xlsx.write_bytes(download.content)
    meta = {"request_url": url, "download_url": download_url, "payload": payload}
    raw_xlsx.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw_xlsx


def _cell(row: pd.Series, candidates: list[str]) -> Any:
    """Return the first non-empty value found in a row for candidate columns."""

    for candidate in candidates:
        if candidate in row and pd.notna(row.get(candidate)):
            return row.get(candidate)
    return None


def _parse_actuarial_report(path: Path, date_str: str) -> list[dict[str, Any]]:
    """Parse official corporate actuarial Excel report into raw rows."""

    rows: list[dict[str, Any]] = []
    source_url = str(path)
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        try:
            source_url = json.loads(meta_path.read_text(encoding="utf-8")).get("download_url", source_url)
        except json.JSONDecodeError:
            source_url = str(path)

    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        sector_api = None
        for candidate in CORPORATE_SECTORS:
            if candidate in sheet.upper():
                sector_api = candidate
                break
        if not sector_api:
            continue
        sector = CORPORATE_SECTORS[sector_api]
        frame = pd.read_excel(path, sheet_name=sheet)
        frame.columns = [str(col).strip() for col in frame.columns]
        for _, item in frame.iterrows():
            maturity = parse_number(
                _cell(item, ["Maturit\u00e9 R\u00e9siduelle", "Maturite Residuelle", "MaturitÃ© RÃ©siduelle", "maturity_years"])
            )
            ytm = parse_number(_cell(item, ["YTM", "ytm", "ytm_decimal"]))
            if maturity is None and ytm is None:
                continue
            rows.append(
                {
                    "curve_date": date_str,
                    "sector": sector,
                    "maturity": maturity,
                    "maturity_unit": "YEARS",
                    "maturity_years": maturity,
                    "corporate_rate_percent": decimal_to_percent(ytm),
                    "corporate_rate_decimal": ytm,
                    "ytm_percent": decimal_to_percent(ytm),
                    "ytm_decimal": ytm,
                    "dirty_price": parse_number(_cell(item, ["Prix Plein Coupon", "dirty_price"])),
                    "clean_price": parse_number(_cell(item, ["Prix Pied Coupon", "clean_price"])),
                    "source": "Tunisia Yield Curve corporate API - ReportActuariel Excel",
                    "source_url": source_url,
                    "extraction_method": "requests_excel_report",
                    "extraction_status": "EXTRACTED",
                    "raw_value": item.dropna().to_json(force_ascii=False),
                }
            )
    return rows


def fetch_corporate_curves_daily(project_root: Path | None = None) -> pd.DataFrame:
    """Fetch official corporate curve data for dates where at least one sector is published."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    session = request_session()
    existing = read_csv(paths.corporate_raw_csv, RAW_COLUMNS)
    existing_dates = set(existing["curve_date"].astype(str)) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    logs: list[dict[str, Any]] = []

    for date in all_2025_dates():
        date_str = date.date().isoformat()
        if date_str in existing_dates:
            continue
        available_sectors: list[str] = []
        for sector_api, sector_label in CORPORATE_SECTORS.items():
            try:
                state = _sector_state(session, date_str, sector_api, paths)
                if state and int(state.get("PUBLISHED", 0)) == 1:
                    available_sectors.append(sector_label)
                    logs.append(log_row(date_str, "CORPORATE", sector_label, "AVAILABLE", "Sector state is published.", "requests_json"))
                else:
                    logs.append(log_row(date_str, "CORPORATE", sector_label, "MISSING", "No published sector state.", "requests_json"))
            except Exception as exc:
                logs.append(log_row(date_str, "CORPORATE", sector_label, "FAILED", f"{type(exc).__name__}: {exc}", "requests_json"))

        if not available_sectors:
            continue
        report = _download_actuarial_report(session, date_str, paths)
        if report is None:
            logs.append(log_row(date_str, "CORPORATE", "ALL", "FAILED", "Official actuarial Excel report unavailable.", "requests_excel_report"))
            continue
        parsed = _parse_actuarial_report(report, date_str)
        rows.extend(parsed)
        logs.append(log_row(date_str, "CORPORATE", "ALL", "EXTRACTED", f"Rows parsed from Excel report: {len(parsed)}", "requests_excel_report"))

        if len(rows) % 5000 < len(parsed):
            write_csv(pd.DataFrame(rows, columns=RAW_COLUMNS).drop_duplicates(), paths.corporate_raw_csv)
            append_log(paths, logs)
            logs = []

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS).drop_duplicates()
    write_csv(frame, paths.corporate_raw_csv)
    append_log(paths, logs)
    LOGGER.info("Saved corporate raw curves: %s rows -> %s", len(frame), paths.corporate_raw_csv)
    return frame


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    fetch_corporate_curves_daily()


if __name__ == "__main__":
    main()
