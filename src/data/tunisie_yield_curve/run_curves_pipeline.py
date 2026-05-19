"""Run the Tunisia Yield Curve 2025 pipeline end to end."""

from __future__ import annotations

import logging
from pathlib import Path

from .build_quality_report import build_quality_report
from .clean_corporate_curves import clean_corporate_curves
from .clean_sovereign_curves import clean_sovereign_curves
from .common import configure_logging
from .discover_endpoints import discover_endpoints
from .fetch_corporate_curves_daily import fetch_corporate_curves_daily
from .fetch_sovereign_curves_daily import fetch_sovereign_curves_daily

LOGGER = logging.getLogger(__name__)


def run_curves_pipeline(project_root: Path | None = None, fetch: bool = True) -> dict[str, object]:
    """Run endpoint discovery, raw extraction, cleaning and quality controls."""

    endpoints = discover_endpoints(project_root)
    if fetch:
        sovereign_raw = fetch_sovereign_curves_daily(project_root)
        corporate_raw = fetch_corporate_curves_daily(project_root)
    else:
        sovereign_raw = None
        corporate_raw = None
    sovereign = clean_sovereign_curves(project_root)
    corporate = clean_corporate_curves(project_root)
    quality = build_quality_report(project_root)
    return {
        "endpoints": endpoints,
        "sovereign_raw": sovereign_raw,
        "corporate_raw": corporate_raw,
        "sovereign": sovereign,
        "corporate": corporate,
        "quality": quality,
    }


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    run_curves_pipeline()


if __name__ == "__main__":
    main()
