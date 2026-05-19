"""Extract published corporate spreads from extracted bulletin text."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .common import (
    SECTOR_ALIASES,
    configure_logging,
    ensure_directories,
    normalize_spaces,
    parse_french_percent,
    percent_to_decimal,
    read_csv_if_exists,
    resolve_paths,
    strip_accents,
    write_csv,
)

LOGGER = logging.getLogger(__name__)
PERCENT_RE = r"(?P<value>\d{1,2}(?:[,.]\d{1,4})?)\s*%"
TRANSITION_RE = re.compile(
    r"(?:pass(?:e|ee|ée|es)?|varie|evolu\w*|prime[^.]{0,80}?)\s+de\s+"
    r"(?P<previous>\d{1,2}(?:[,.]\d{1,4})?)\s*%\s+(?:a|à)\s+"
    r"(?P<final>\d{1,2}(?:[,.]\d{1,4})?)\s*%",
    re.I,
)
PRIME_FINALE_RE = re.compile(
    r"prime\s+finale[^0-9%]{0,80}(?P<value>\d{1,2}(?:[,.]\d{1,4})?)\s*%",
    re.I,
)


def _sector_patterns() -> dict[str, re.Pattern[str]]:
    """Compile sector alias patterns."""

    return {sector: re.compile("|".join(aliases), re.I) for sector, aliases in SECTOR_ALIASES.items()}


def _section_windows(text: str, window: int = 2200) -> list[str]:
    """Return text windows around corporate/prime keywords."""

    normalized = strip_accents(text)
    keywords = [
        "corporate",
        "prime finale",
        "prime",
        "emprunts obligataires ordinaires",
        "emprunts obligataires subordonnes",
    ]
    windows: list[str] = []
    seen: set[tuple[int, int]] = set()
    for keyword in keywords:
        for match in re.finditer(re.escape(strip_accents(keyword)), normalized):
            start = max(0, match.start() - window // 3)
            end = min(len(text), match.end() + window)
            key = (start, end)
            if key not in seen:
                seen.add(key)
                windows.append(text[start:end])
    return windows


def _snippet(text: str, start: int, end: int, radius: int = 420) -> str:
    """Return an audit snippet around a matched expression."""

    return normalize_spaces(text[max(0, start - radius) : min(len(text), end + radius)])


def _sector_for_context(context: str) -> list[str]:
    """Return normalized sectors detected in a context window."""

    normalized = strip_accents(context)
    sectors = []
    for sector, pattern in _sector_patterns().items():
        if pattern.search(normalized):
            sectors.append(sector)
    return sectors


def _rows_from_context(
    context: str,
    bulletin: pd.Series,
    source_url: str,
    extraction_method: str,
    source_pdf: str,
) -> list[dict[str, object]]:
    """Extract spread rows from a single corporate context window."""

    rows: list[dict[str, object]] = []
    normalized_context = strip_accents(context)
    if not any(
        marker in normalized_context
        for marker in ("corporate", "prime", "emprunts obligataires ordinaires", "emprunts obligataires subordonnes")
    ):
        return rows
    sectors = _sector_for_context(context)
    base = {
        "bulletin_date": bulletin.get("bulletin_date"),
        "year": int(bulletin.get("year")),
        "month": int(bulletin.get("month")),
        "source_pdf": source_pdf,
        "source_url": source_url,
        "extraction_method": extraction_method,
    }

    for match in TRANSITION_RE.finditer(normalized_context):
        previous = parse_french_percent(match.group("previous"))
        final = parse_french_percent(match.group("final"))
        sector_list = sectors or ["AMBIGU"]
        for sector in sector_list:
            rows.append(
                {
                    **base,
                    "sector": sector,
                    "spread_type": "PRIME_FINALE_TRANSITION",
                    "spread_percent": final,
                    "spread_decimal": percent_to_decimal(final),
                    "previous_spread_percent": previous,
                    "final_spread_percent": final,
                    "source_text_snippet": _snippet(context, match.start(), match.end()),
                    "extraction_confidence": "HIGH" if sector != "AMBIGU" else "LOW",
                    "data_quality_flag": "OK" if sector != "AMBIGU" else "AMBIGUOUS_SECTOR",
                    "extraction_status": "EXTRACTED" if sector != "AMBIGU" else "AMBIGUOUS_SECTOR",
                }
            )

    sector_patterns = _sector_patterns()
    for sector, sector_pattern in sector_patterns.items():
        for sector_match in sector_pattern.finditer(normalized_context):
            local_start = max(0, sector_match.start() - 260)
            local_end = min(len(context), sector_match.end() + 520)
            local = context[local_start:local_end]
            local_norm = strip_accents(local)

            value: float | None = None
            local_sector = sector_pattern.search(local_norm)
            after_sector = local_norm[local_sector.end() : local_sector.end() + 180] if local_sector else local_norm
            before_sector = local_norm[max(0, (local_sector.start() if local_sector else 0) - 180) : (local_sector.start() if local_sector else 0)]
            direct = re.search(PERCENT_RE, after_sector, re.I)
            reverse_matches = list(re.finditer(PERCENT_RE, before_sector, re.I))
            reverse = reverse_matches[-1] if reverse_matches else None
            prime = PRIME_FINALE_RE.search(local_norm)
            chosen = direct or reverse or prime
            if chosen:
                value = parse_french_percent(chosen.group("value"))
            if value is None:
                continue

            contains_prime = "prime" in local_norm or "corporate" in local_norm
            rows.append(
                {
                    **base,
                    "sector": sector,
                    "spread_type": "PRIME_FINALE" if contains_prime else "SPREAD_PUBLIE",
                    "spread_percent": value,
                    "spread_decimal": percent_to_decimal(value),
                    "previous_spread_percent": None,
                    "final_spread_percent": value,
                    "source_text_snippet": normalize_spaces(local),
                    "extraction_confidence": "HIGH" if contains_prime else "MEDIUM",
                    "data_quality_flag": "OK",
                    "extraction_status": "EXTRACTED",
                }
            )

    if not rows and "prime" in normalized_context:
        for match in re.finditer(PERCENT_RE, normalized_context):
            rows.append(
                {
                    **base,
                    "sector": "AMBIGU",
                    "spread_type": "PRIME_AMBIGUE",
                    "spread_percent": parse_french_percent(match.group("value")),
                    "spread_decimal": percent_to_decimal(match.group("value")),
                    "previous_spread_percent": None,
                    "final_spread_percent": parse_french_percent(match.group("value")),
                    "source_text_snippet": _snippet(context, match.start(), match.end()),
                    "extraction_confidence": "LOW",
                    "data_quality_flag": "AMBIGUOUS_SECTOR",
                    "extraction_status": "AMBIGUOUS_SECTOR",
                }
            )
    return rows


def extract_spreads_from_text(
    text: str,
    bulletin: pd.Series,
    source_url: str = "",
    extraction_method: str = "",
    source_pdf: str = "",
) -> list[dict[str, object]]:
    """Extract corporate spread observations from one bulletin text."""

    contexts = _section_windows(text)
    rows: list[dict[str, object]] = []
    for context in contexts:
        rows.extend(_rows_from_context(context, bulletin, source_url, extraction_method, source_pdf))

    if rows:
        frame = pd.DataFrame(rows)
        frame = frame.drop_duplicates(
            subset=["bulletin_date", "sector", "spread_type", "spread_percent", "previous_spread_percent", "source_text_snippet"]
        )
        return frame.to_dict("records")
    return []


def extract_corporate_spreads(project_root: Path | None = None) -> pd.DataFrame:
    """Extract corporate spread rows from all previously extracted texts."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    text_index = read_csv_if_exists(paths.extracted_text_index_csv)
    bulletins_index = read_csv_if_exists(paths.bulletins_index_csv)
    if text_index.empty:
        raise FileNotFoundError(f"No extracted text index found at {paths.extracted_text_index_csv}.")

    source_url_by_pdf = dict(zip(bulletins_index.get("pdf_filename", []), bulletins_index.get("pdf_url", [])))
    rows: list[dict[str, object]] = []
    for _, row in text_index.iterrows():
        text_path = str(row.get("extracted_text_path", "") or "")
        if not text_path:
            continue
        full_text_path = paths.project_root / text_path
        if not full_text_path.exists():
            continue
        text = full_text_path.read_text(encoding="utf-8", errors="replace")
        source_pdf = str(row.get("local_pdf_path", ""))
        extracted = extract_spreads_from_text(
            text=text,
            bulletin=row,
            source_url=str(source_url_by_pdf.get(row.get("pdf_filename"), "")),
            extraction_method=str(row.get("extraction_method", "")),
            source_pdf=source_pdf,
        )
        rows.extend(extracted)

    raw = pd.DataFrame(rows)
    if raw.empty:
        raw = pd.DataFrame(
            columns=[
                "bulletin_date",
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
        )

    detected_years = set(raw.loc[raw["extraction_status"].eq("EXTRACTED"), "year"].astype(int).tolist()) if not raw.empty else set()
    missing_rows: list[dict[str, object]] = []
    for year in sorted(text_index.get("year", pd.Series(dtype=int)).dropna().astype(int).unique()):
        if year in detected_years:
            continue
        if year == 2023:
            status = "NO_OFFICIAL_CORPORATE_SPREAD_DETECTED"
            flag = "NO_OFFICIAL_CORPORATE_CURVE_BEFORE_END_2023"
        else:
            status = "NO_OFFICIAL_CORPORATE_SPREAD_DETECTED_IN_YEAR"
            flag = "NO_CORPORATE_SPREAD_SECTION_DETECTED"
        missing_rows.append(
            {
                "bulletin_date": "",
                "year": year,
                "month": "",
                "sector": "",
                "spread_type": "",
                "spread_percent": None,
                "spread_decimal": None,
                "previous_spread_percent": None,
                "final_spread_percent": None,
                "source_pdf": "",
                "source_url": "",
                "source_text_snippet": "",
                "extraction_method": "",
                "extraction_confidence": "",
                "data_quality_flag": flag,
                "extraction_status": status,
            }
        )
    if missing_rows:
        raw = pd.concat([raw, pd.DataFrame(missing_rows)], ignore_index=True)

    has_2023_detection = 2023 in detected_years
    if has_2023_detection:
        raw.loc[raw["year"] == 2023, "data_quality_flag"] = "LIMITED_2023_COVERAGE"

    write_csv(raw, paths.corporate_spreads_raw_csv)
    LOGGER.info("Saved raw corporate spread extraction: %s rows -> %s", len(raw), paths.corporate_spreads_raw_csv)
    return raw


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    extract_corporate_spreads()


if __name__ == "__main__":
    main()
