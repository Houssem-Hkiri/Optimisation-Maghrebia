"""Scrape the historical bulletin index from Tunisie Clearing."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .common import (
    END_DATE,
    REQUEST_TIMEOUT_SECONDS,
    SOURCE_PAGE_URL,
    START_DATE,
    USER_AGENT,
    absolute_url,
    configure_logging,
    ensure_directories,
    find_date_from_text,
    local_pdf_path,
    normalize_spaces,
    resolve_paths,
    safe_filename_from_url,
    write_csv,
    year_month_from_record,
)

LOGGER = logging.getLogger(__name__)
PUBLIC_API_BASE = "https://www.tunisieclearing.com/service/api"
PUBLIC_UPLOAD_BASE = "https://www.tunisieclearing.com/upload"
PUBLIC_BULLETIN_TYPES = (
    "bulletin-annuel",
    "bulletin-trimestriel",
    "bulletin-quotidien-des-activit\u00e9s",
)


def _extract_pdf_links_from_html(html: str, source_url: str) -> list[dict[str, str]]:
    """Extract PDF links from static HTML and embedded scripts."""

    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        if ".pdf" not in href.lower():
            continue
        pdf_url = absolute_url(href, source_url)
        if pdf_url in seen:
            continue
        seen.add(pdf_url)
        records.append(
            {
                "pdf_url": pdf_url,
                "link_text": normalize_spaces(link.get_text(" ", strip=True)),
                "context_text": normalize_spaces(link.parent.get_text(" ", strip=True) if link.parent else ""),
            }
        )

    pdf_regex = re.compile(r"""(?P<url>(?:https?://[^"' <>)]+|/[^"' <>)]+)[.]pdf(?:\?[^"' <>)]+)?)""", re.I)
    for match in pdf_regex.finditer(html):
        pdf_url = absolute_url(match.group("url"), source_url)
        if pdf_url in seen:
            continue
        seen.add(pdf_url)
        start, end = max(0, match.start() - 160), min(len(html), match.end() + 160)
        records.append(
            {
                "pdf_url": pdf_url,
                "link_text": "",
                "context_text": normalize_spaces(BeautifulSoup(html[start:end], "html.parser").get_text(" ")),
            }
        )

    return records


def _extract_pdf_links_with_requests(source_url: str) -> list[dict[str, str]]:
    """Fetch the bulletin page with requests and parse visible PDF links."""

    session = requests.Session()
    response = session.get(
        source_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return _extract_pdf_links_from_html(response.text, source_url)


def _extract_pdf_links_from_public_api() -> list[dict[str, str]]:
    """Use Tunisie Clearing's public Angular API when the static page has no PDF links."""

    session = requests.Session()
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for bulletin_type in PUBLIC_BULLETIN_TYPES:
        encoded_type = quote(bulletin_type, safe="")
        for year in range(2023, 2026):
            url = f"{PUBLIC_API_BASE}/Public/Bulletin/{encoded_type}/fr/{year}"
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            response.raise_for_status()
            items = response.json()
            if not isinstance(items, list):
                continue
            for item in items:
                media = item.get("T_MEDIA") or {}
                if media.get("typeMedia") and "pdf" not in str(media.get("typeMedia")).lower():
                    continue
                folder = str(media.get("folderMedia") or "").strip("/")
                media_name = str(media.get("urlMedia") or "").strip("/")
                if not folder or not media_name:
                    continue
                pdf_url = f"{PUBLIC_UPLOAD_BASE}/{folder}/{media_name}"
                if pdf_url in seen:
                    continue
                seen.add(pdf_url)
                records.append(
                    {
                        "pdf_url": pdf_url,
                        "link_text": normalize_spaces(str(item.get("libBulletin") or media.get("libMedia") or "")),
                        "context_text": " ".join(
                            [
                                str(item.get("dateBulletin") or ""),
                                str(item.get("catBulletin") or ""),
                                str(item.get("yearBulletin") or ""),
                                str(item.get("monthBulletin") or ""),
                                str(item.get("idBulletin") or ""),
                                bulletin_type,
                            ]
                        ),
                        "api_date": item.get("dateBulletin"),
                        "api_year": item.get("yearBulletin"),
                        "api_month": item.get("monthBulletin"),
                        "api_id": item.get("idBulletin"),
                        "api_type": bulletin_type,
                        "api_media_filename": media.get("libMedia") or "",
                    }
                )
    return records


def _extract_pdf_links_with_playwright(source_url: str) -> list[dict[str, str]]:
    """Use Playwright when links are generated dynamically client-side."""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed; install it only if static scraping fails.") from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(source_url, wait_until="networkidle", timeout=60_000)
        html = page.content()
        browser.close()
    return _extract_pdf_links_from_html(html, source_url)


def scrape_bulletins(
    project_root: Path | None = None,
    source_url: str = SOURCE_PAGE_URL,
    use_playwright_fallback: bool = True,
) -> pd.DataFrame:
    """Scrape and save the bulletin index for 2023-2025.

    Parameters
    ----------
    project_root:
        Repository root. When omitted, it is inferred from this file location.
    source_url:
        Historical bulletin page URL.
    use_playwright_fallback:
        If True, retry with headless Playwright when static HTML exposes no PDF.

    Returns
    -------
    pandas.DataFrame
        Bulletin index saved to ``data/interim/tunisie_clearing/bulletins_index.csv``.
    """

    paths = resolve_paths(project_root)
    ensure_directories(paths)

    try:
        raw_links = _extract_pdf_links_with_requests(source_url)
        LOGGER.info("Static scraping found %s PDF links.", len(raw_links))
    except Exception as exc:
        LOGGER.warning("Static scraping failed: %s", exc)
        raw_links = []

    if not raw_links:
        try:
            raw_links = _extract_pdf_links_from_public_api()
            LOGGER.info("Public API scraping found %s PDF links.", len(raw_links))
        except Exception as exc:
            LOGGER.warning("Public API scraping failed: %s", exc)

    if not raw_links and use_playwright_fallback:
        raw_links = _extract_pdf_links_with_playwright(source_url)
        LOGGER.info("Playwright scraping found %s PDF links.", len(raw_links))

    rows: list[dict[str, object]] = []
    for item in raw_links:
        pdf_url = item["pdf_url"]
        filename = safe_filename_from_url(pdf_url)
        text_for_date = " ".join([item.get("link_text", ""), item.get("context_text", ""), pdf_url, filename])
        bulletin_date = pd.to_datetime(item.get("api_date"), errors="coerce") if item.get("api_date") else find_date_from_text(text_for_date)
        if pd.isna(bulletin_date):
            bulletin_date = find_date_from_text(text_for_date)
        year = int(item.get("api_year")) if item.get("api_year") else None
        month = int(item.get("api_month")) if item.get("api_month") else None
        if year is None or month is None:
            year, month = year_month_from_record(bulletin_date, pdf_url)
        if year not in {2023, 2024, 2025} or month is None:
            continue

        effective_date = bulletin_date if bulletin_date is not None else pd.Timestamp(year=year, month=month, day=1)
        if effective_date < START_DATE or effective_date > END_DATE:
            continue

        path = local_pdf_path(paths, int(year), filename)
        bulletin_id_match = re.search(r"([A-Za-z0-9_-]+)[.]pdf(?:\?|$)", filename)
        rows.append(
            {
                "bulletin_date": effective_date.date().isoformat(),
                "year": int(year),
                "month": int(month),
                "pdf_url": pdf_url,
                "pdf_filename": filename,
                "local_pdf_path": str(path.relative_to(paths.project_root)),
                "source_page_url": source_url,
                "download_status": "ALREADY_EXISTS" if path.exists() else "PENDING",
                "bulletin_id": item.get("api_id") or (bulletin_id_match.group(1) if bulletin_id_match else ""),
                "link_text": item.get("link_text", ""),
                "bulletin_type": item.get("api_type", ""),
                "source_media_filename": item.get("api_media_filename", ""),
            }
        )

    index = pd.DataFrame(rows)
    if not index.empty:
        index = index.drop_duplicates(subset=["pdf_url"]).sort_values(["year", "month", "pdf_filename"])
    expected_cols = [
        "bulletin_date",
        "year",
        "month",
        "pdf_url",
        "pdf_filename",
        "local_pdf_path",
        "source_page_url",
        "download_status",
        "bulletin_id",
        "link_text",
        "bulletin_type",
        "source_media_filename",
    ]
    index = index.reindex(columns=expected_cols)
    write_csv(index, paths.bulletins_index_csv)
    LOGGER.info("Saved bulletin index: %s rows -> %s", len(index), paths.bulletins_index_csv)
    return index


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    scrape_bulletins()


if __name__ == "__main__":
    main()
