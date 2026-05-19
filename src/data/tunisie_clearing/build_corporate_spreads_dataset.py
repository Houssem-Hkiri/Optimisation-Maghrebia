"""Build clean corporate spread datasets and Excel quality controls."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .common import configure_logging, ensure_directories, percent_to_decimal, read_csv_if_exists, resolve_paths, write_csv
from .download_bulletins import download_bulletins
from .extract_corporate_spreads import extract_corporate_spreads
from .extract_pdf_text import extract_pdf_text
from .scrape_bulletins import scrape_bulletins

LOGGER = logging.getLogger(__name__)

FINAL_COLUMNS = [
    "date",
    "year",
    "month",
    "sector",
    "spread_type",
    "spread_percent",
    "spread_decimal",
    "previous_spread_percent",
    "final_spread_percent",
    "source_pdf",
    "source_url",
    "source_text_snippet",
    "extraction_method",
    "extraction_confidence",
    "data_quality_flag",
    "extraction_status",
]


def _clean_raw_spreads(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw spread rows into the final schema."""

    if raw.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    clean = raw.copy()
    clean = clean.rename(columns={"bulletin_date": "date"})
    for col in FINAL_COLUMNS:
        if col not in clean.columns:
            clean[col] = None
    for col in ["spread_percent", "previous_spread_percent", "final_spread_percent"]:
        clean[col] = clean[col].map(lambda value: None if pd.isna(value) or value == "" else float(str(value).replace(",", ".")))
    clean["spread_decimal"] = clean["spread_percent"].map(percent_to_decimal)
    clean["sector"] = clean["sector"].fillna("").astype(str).str.upper()
    clean["data_quality_flag"] = clean["data_quality_flag"].fillna("OK")
    clean["extraction_status"] = clean["extraction_status"].fillna("EXTRACTED")
    clean = clean[FINAL_COLUMNS].drop_duplicates()
    return clean.sort_values(["year", "month", "sector", "spread_type"], na_position="last").reset_index(drop=True)


def _build_quality_tables(
    bulletins: pd.DataFrame,
    text_index: pd.DataFrame,
    raw: pd.DataFrame,
    clean: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create quality-control sheets for the Excel report."""

    if bulletins.empty:
        bulletins = pd.DataFrame(columns=["year", "download_status"])
    if text_index.empty:
        text_index = pd.DataFrame(columns=["year", "contains_corporate_section", "detected_keywords", "pdf_filename"])
    if raw.empty:
        raw = pd.DataFrame(columns=["year", "sector", "extraction_status", "data_quality_flag"])
    if clean.empty:
        clean = pd.DataFrame(columns=FINAL_COLUMNS)

    extraction_status = (
        text_index.groupby(["year", "extraction_status"], dropna=False)
        .size()
        .reset_index(name="pdf_count")
        if {"year", "extraction_status"}.issubset(text_index.columns)
        else pd.DataFrame(columns=["year", "extraction_status", "pdf_count"])
    )
    detected_sections = text_index[text_index.get("contains_corporate_section", pd.Series(False, index=text_index.index)).astype(bool)].copy()
    missing_or_ambiguous = clean[
        clean["extraction_status"].astype(str).str.contains("AMBIG|NO_OFFICIAL|MISSING|FAILED", case=False, na=False)
        | clean["data_quality_flag"].astype(str).str.contains("AMBIG|NO_OFFICIAL|LIMITED|MISSING|RECONSTRUCT", case=False, na=False)
    ].copy()

    bulletins_by_year = bulletins.groupby("year", dropna=False).size().reset_index(name="bulletins_found")
    downloaded_by_year = (
        bulletins.assign(is_downloaded=bulletins["download_status"].astype(str).str.contains("DOWNLOADED|ALREADY_EXISTS", na=False))
        .groupby("year", dropna=False)["is_downloaded"]
        .sum()
        .reset_index(name="pdf_downloaded")
    )
    corporate_by_year = (
        text_index.assign(has_corporate=text_index["contains_corporate_section"].astype(bool))
        .groupby("year", dropna=False)["has_corporate"]
        .sum()
        .reset_index(name="pdf_with_corporate_section")
    )
    spreads_by_year_sector = (
        clean[clean["extraction_status"].eq("EXTRACTED")]
        .groupby(["year", "sector"], dropna=False)
        .size()
        .reset_index(name="spreads_extracted")
    )
    yearly_summary = bulletins_by_year.merge(downloaded_by_year, on="year", how="outer").merge(corporate_by_year, on="year", how="outer")
    if not spreads_by_year_sector.empty:
        pivot = spreads_by_year_sector.pivot_table(index="year", columns="sector", values="spreads_extracted", fill_value=0).reset_index()
        yearly_summary = yearly_summary.merge(pivot, on="year", how="outer")
    yearly_summary["coverage_comment"] = yearly_summary["year"].map(
        {
            2023: "Couverture probablement absente avant fin decembre 2023; aucune valeur n'est inventee.",
            2024: "Couverture attendue surtout autour du leasing selon les bulletins disponibles.",
            2025: "Couverture sectorielle attendue plus large: bancaire, leasing, microfinance.",
        }
    )

    corporate_but_no_spread = detected_sections[
        ~detected_sections["pdf_filename"].isin(raw.get("source_pdf", pd.Series(dtype=str)).astype(str).map(lambda value: Path(value).name))
    ].copy()
    prime_ambiguous = clean[clean["sector"].eq("AMBIGU")].copy()
    if not corporate_but_no_spread.empty or not prime_ambiguous.empty:
        missing_or_ambiguous = pd.concat([missing_or_ambiguous, corporate_but_no_spread, prime_ambiguous], ignore_index=True, sort=False)

    return {
        "Bulletins_Index": bulletins,
        "Extraction_Status": extraction_status,
        "Detected_Corporate_Sections": detected_sections,
        "Corporate_Spreads_Raw": raw,
        "Corporate_Spreads_Clean": clean,
        "Missing_Or_Ambiguous": missing_or_ambiguous,
        "Yearly_Summary": yearly_summary,
    }


def _write_quality_report(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Write the quality report workbook with one sheet per control table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            safe_name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False)


def build_corporate_spreads_dataset(project_root: Path | None = None) -> pd.DataFrame:
    """Build final CSV/XLSX datasets and the Excel quality report from intermediates."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    bulletins = read_csv_if_exists(paths.bulletins_index_csv)
    text_index = read_csv_if_exists(paths.extracted_text_index_csv)
    raw = read_csv_if_exists(paths.corporate_spreads_raw_csv)

    clean = _clean_raw_spreads(raw)
    write_csv(clean, paths.corporate_spreads_clean_csv)
    with pd.ExcelWriter(paths.corporate_spreads_clean_xlsx, engine="openpyxl") as writer:
        clean.to_excel(writer, sheet_name="Corporate_Spreads_Clean", index=False)

    quality_tables = _build_quality_tables(bulletins, text_index, raw, clean)
    _write_quality_report(paths.quality_report_xlsx, quality_tables)
    LOGGER.info("Saved clean CSV -> %s", paths.corporate_spreads_clean_csv)
    LOGGER.info("Saved clean XLSX -> %s", paths.corporate_spreads_clean_xlsx)
    LOGGER.info("Saved quality report -> %s", paths.quality_report_xlsx)
    return clean


def run_full_pipeline(
    project_root: Path | None = None,
    download: bool = True,
    extract_text: bool = True,
    extract_spreads: bool = True,
) -> pd.DataFrame:
    """Run the full Tunisie Clearing pipeline end to end."""

    scrape_bulletins(project_root=project_root)
    if download:
        download_bulletins(project_root=project_root)
    if extract_text:
        extract_pdf_text(project_root=project_root)
    if extract_spreads:
        extract_corporate_spreads(project_root=project_root)
    return build_corporate_spreads_dataset(project_root=project_root)


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    run_full_pipeline()


if __name__ == "__main__":
    main()
