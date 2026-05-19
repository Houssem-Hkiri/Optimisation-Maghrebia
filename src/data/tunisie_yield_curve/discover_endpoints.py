"""Discover public endpoints used by Tunisia Yield Curve pages."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

from .common import (
    CORPORATE_API_BASE,
    CORPORATE_PUBLIC_URL,
    CORPORATE_REPORT_BASE,
    SOVEREIGN_API_BASE,
    SOVEREIGN_PUBLIC_URL,
    SOVEREIGN_REPORT_BASE,
    configure_logging,
    ensure_directories,
    request_session,
    resolve_paths,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def _script_urls(html: str, base_url: str) -> list[str]:
    """Extract script URLs from a page."""

    urls = []
    for match in re.finditer(r"<script[^>]+src=[\"']([^\"']+)", html, re.I):
        urls.append(urljoin(base_url, match.group(1)))
    return urls


def _extract_endpoint_paths(script_text: str) -> list[str]:
    """Extract Angular endpoint paths from a JavaScript service file."""

    paths = sorted(set(re.findall(r'"/api/Yield/[^" ?]+', script_text)))
    return [path.strip('"') for path in paths]


def _network_urls_with_playwright(page_url: str) -> dict[str, object]:
    """Capture network URLs with Playwright when static JavaScript inspection is insufficient."""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return {"status": "SKIPPED", "reason": f"Playwright is not installed: {exc}"}

    urls: set[str] = set()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("request", lambda request: urls.add(request.url))
            page.goto(page_url, wait_until="networkidle", timeout=60000)
            browser.close()
    except Exception as exc:
        return {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}", "urls": sorted(urls)}

    relevant = [url for url in sorted(urls) if "api" in url.lower() or "yield" in url.lower()]
    return {"status": "OK", "urls": relevant}


def discover_endpoints(project_root: Path | None = None) -> dict[str, object]:
    """Inspect pages and JavaScript files, then save endpoint documentation."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    session = request_session()
    discoveries: dict[str, object] = {
        "sovereign": {
            "page_url": SOVEREIGN_PUBLIC_URL,
            "api_base": SOVEREIGN_API_BASE,
            "report_base": SOVEREIGN_REPORT_BASE,
            "scripts": [],
            "endpoint_paths": [],
            "playwright_network_capture": {},
            "confirmed_endpoints": {
                "published_curves": f"{SOVEREIGN_API_BASE}/GetPublishedCurve",
                "curve_index_by_date": f"{SOVEREIGN_API_BASE}/CurveGet?DATE_CURVE={{YYYY-MM-DD}}&ID_USER=0&IS_TEST=0",
                "zero_coupon_curve_by_id": f"{SOVEREIGN_API_BASE}/YieldCurve?ID_CURVE={{ID_CURVE}}",
                "zero_coupon_report": f"{SOVEREIGN_API_BASE}/ReportZero?DATES={{ID_CURVE}}",
            },
        },
        "corporate": {
            "page_url": CORPORATE_PUBLIC_URL,
            "api_base": CORPORATE_API_BASE,
            "report_base": CORPORATE_REPORT_BASE,
            "scripts": [],
            "endpoint_paths": [],
            "playwright_network_capture": {},
            "confirmed_endpoints": {
                "sector_state_by_date": f"{CORPORATE_API_BASE}/GetCorporateJournee?DATE_CORPORATE={{YYYY-MM-DD}}&TYPE_SECTEUR={{BANKING|LEASING|MICROFINANCE}}",
                "sector_parameters_by_date": f"{CORPORATE_API_BASE}/ParamCurve?DATE_CURVE={{YYYY-MM-DD}}&TYPE_SECTEUR={{BANKING|LEASING|MICROFINANCE}}",
                "actuarial_excel_report": f"{CORPORATE_API_BASE}/ReportActuariel?DATES={{DD/MM/YYYY}}&TYPE_SECTEUR=LEASING",
                "corporate_curve_excel_report": f"{CORPORATE_API_BASE}/ExportCurveCorporate?DATE_CORPORATE={{YYYY-MM-DD}}&TYPE_SECTEUR={{BANKING|LEASING|MICROFINANCE}}",
            },
        },
        "discovery_errors": [],
        "notes": [
            "The sovereign page loads data through JSON endpoints; no browser automation is required when these endpoints respond.",
            "The corporate page exposes sector state/parameter JSON endpoints and official generated Excel reports.",
            "Playwright is a fallback only if the public endpoints stop responding.",
        ],
    }

    for key, page_url in {"sovereign": SOVEREIGN_PUBLIC_URL, "corporate": CORPORATE_PUBLIC_URL}.items():
        try:
            response = session.get(page_url, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            discoveries["discovery_errors"].append({"page": page_url, "error": f"{type(exc).__name__}: {exc}"})
            continue
        scripts = _script_urls(response.text, page_url)
        discoveries[key]["scripts"] = scripts
        endpoint_paths: set[str] = set()
        for script_url in scripts:
            if "/angular/" not in script_url:
                continue
            try:
                script_response = session.get(script_url, timeout=60)
                if script_response.ok:
                    endpoint_paths.update(_extract_endpoint_paths(script_response.text))
            except requests.RequestException as exc:
                discoveries["discovery_errors"].append({"script": script_url, "error": f"{type(exc).__name__}: {exc}"})
        discoveries[key]["endpoint_paths"] = sorted(endpoint_paths)
        if not endpoint_paths:
            discoveries[key]["playwright_network_capture"] = _network_urls_with_playwright(page_url)

    write_json(discoveries, paths.endpoints_json)
    LOGGER.info("Saved endpoint discovery -> %s", paths.endpoints_json)
    return discoveries


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    discover_endpoints()


if __name__ == "__main__":
    main()
