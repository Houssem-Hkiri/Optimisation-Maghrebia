"""Shared utilities for the Tunisie Clearing corporate spreads pipeline."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd


SOURCE_PAGE_URL = "https://www.tunisieclearing.com/tc/fr/statistiques/historiquebulletin"
START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2025-12-31")
REQUEST_TIMEOUT_SECONDS = 45
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

KEYWORDS = [
    "CORPORATE",
    "courbe des taux CORPORATE",
    "Analyse de la courbe des taux CORPORATE",
    "Prime Finale",
    "Prime finale",
    "Bancaire",
    "Banques",
    "Leasing",
    "Microfinance",
    "emprunts obligataires ordinaires",
    "emprunts obligataires subordonnés",
    "prime",
]

SECTOR_ALIASES = {
    "BANCAIRE": [r"\bbancaire\b", r"\bbanques?\b", r"\bsecteur\s+bancaire\b"],
    "LEASING": [r"\bleasing\b", r"\bsecteur\s+leasing\b"],
    "MICROFINANCE": [r"\bmicro[\s-]?finance\b", r"\bmicrofinance\b"],
}


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved project paths used by the pipeline."""

    project_root: Path
    raw_bulletins_dir: Path
    interim_dir: Path
    processed_dir: Path
    text_dir: Path
    bulletins_index_csv: Path
    extracted_text_index_csv: Path
    corporate_spreads_raw_csv: Path
    corporate_spreads_clean_csv: Path
    corporate_spreads_clean_xlsx: Path
    quality_report_xlsx: Path


def project_root_from_module() -> Path:
    """Return the repository root from this module location."""

    return Path(__file__).resolve().parents[3]


def resolve_paths(project_root: Path | None = None) -> PipelinePaths:
    """Resolve all output paths relative to the repository root."""

    root = Path(project_root).resolve() if project_root else project_root_from_module()
    interim = root / "data" / "interim" / "tunisie_clearing"
    processed = root / "data" / "processed" / "tunisie_clearing"
    raw = root / "data" / "raw" / "tunisie_clearing_bulletins"
    return PipelinePaths(
        project_root=root,
        raw_bulletins_dir=raw,
        interim_dir=interim,
        processed_dir=processed,
        text_dir=interim / "text",
        bulletins_index_csv=interim / "bulletins_index.csv",
        extracted_text_index_csv=interim / "extracted_text_index.csv",
        corporate_spreads_raw_csv=interim / "corporate_spreads_raw.csv",
        corporate_spreads_clean_csv=processed / "corporate_spreads_2023_2025_clean.csv",
        corporate_spreads_clean_xlsx=processed / "corporate_spreads_2023_2025_clean.xlsx",
        quality_report_xlsx=processed / "corporate_spreads_quality_report.xlsx",
    )


def ensure_directories(paths: PipelinePaths) -> None:
    """Create required output folders without touching raw input files."""

    paths.raw_bulletins_dir.mkdir(parents=True, exist_ok=True)
    for year in (2023, 2024, 2025):
        (paths.raw_bulletins_dir / str(year)).mkdir(parents=True, exist_ok=True)
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    paths.text_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a concise root logger for command-line pipeline execution."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("tunisie_clearing")


def normalize_spaces(value: str) -> str:
    """Collapse whitespace in extracted PDF or HTML text."""

    return re.sub(r"\s+", " ", value or "").strip()


def strip_accents(value: str) -> str:
    """Return a lowercase accent-free string for robust French matching."""

    normalized = unicodedata.normalize("NFD", value or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").lower()


def parse_french_percent(value: str | float | int | None) -> float | None:
    """Parse a French percentage number such as ``0,955`` into percent units."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def percent_to_decimal(value: str | float | int | None) -> float | None:
    """Convert a percentage value to decimal units; ``0,90 %`` becomes ``0.009``."""

    percent = parse_french_percent(value)
    if percent is None:
        return None
    return percent / 100.0


def detect_keywords(text: str) -> list[str]:
    """Return configured keywords detected in text, preserving canonical labels."""

    normalized = strip_accents(text)
    found: list[str] = []
    for keyword in KEYWORDS:
        if strip_accents(keyword) in normalized:
            found.append(keyword)
    return found


def is_pdf_file(path: Path) -> bool:
    """Validate that a local file starts with the PDF signature."""

    try:
        with Path(path).open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def safe_filename_from_url(pdf_url: str) -> str:
    """Return a conservative local filename derived from a PDF URL."""

    raw_name = Path(pdf_url.split("?", 1)[0]).name or "bulletin.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)
    return name if name.lower().endswith(".pdf") else f"{name}.pdf"


def absolute_url(href: str, base_url: str = SOURCE_PAGE_URL) -> str:
    """Resolve a possibly relative URL against Tunisie Clearing's bulletin page."""

    return urljoin(base_url, href)


def local_pdf_path(paths: PipelinePaths, year: int, pdf_filename: str) -> Path:
    """Return the expected local PDF path for a bulletin."""

    return paths.raw_bulletins_dir / str(year) / pdf_filename


def find_date_from_text(value: str) -> pd.Timestamp | None:
    """Extract a date from nearby link text or URL path when possible."""

    if not value:
        return None
    text = value.replace("\\", "/")
    for pattern in (
        r"(?P<day>\d{1,2})[/-](?P<month>\d{1,2})[/-](?P<year>20\d{2})",
        r"(?P<year>20\d{2})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})",
    ):
        match = re.search(pattern, text)
        if match:
            try:
                return pd.Timestamp(
                    date(
                        int(match.group("year")),
                        int(match.group("month")),
                        int(match.group("day")),
                    )
                )
            except ValueError:
                return None
    match = re.search(r"(?<!\d)(?P<year>20\d{2})(?P<month>\d{2})(?!\d)", text)
    if match:
        year, month = int(match.group("year")), int(match.group("month"))
        if 1 <= month <= 12:
            return pd.Timestamp(date(year, month, 1))
    return None


def year_month_from_record(bulletin_date: pd.Timestamp | None, pdf_url: str) -> tuple[int | None, int | None]:
    """Infer year and month from a date first, then from a URL segment."""

    if bulletin_date is not None and not pd.isna(bulletin_date):
        return int(bulletin_date.year), int(bulletin_date.month)
    match = re.search(r"/upload/(?P<year>20\d{2})(?P<month>\d{2})/", pdf_url)
    if match:
        return int(match.group("year")), int(match.group("month"))
    return None, None


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV with stable UTF-8-SIG encoding for Excel compatibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_if_exists(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise return an empty frame with optional columns."""

    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=list(columns or []))
