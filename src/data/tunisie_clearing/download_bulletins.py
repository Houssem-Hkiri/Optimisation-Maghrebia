"""Download and validate Tunisie Clearing bulletin PDFs."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable, **_: object):
        return iterable

from .common import (
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
    configure_logging,
    ensure_directories,
    is_pdf_file,
    local_pdf_path,
    read_csv_if_exists,
    resolve_paths,
    safe_filename_from_url,
    write_csv,
)

LOGGER = logging.getLogger(__name__)


def _safe_unlink(path: Path) -> None:
    """Best-effort deletion for temporary files that may be locked by Windows/OneDrive."""

    try:
        path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("Could not remove temporary file because it is locked: %s", path)


def _replace_with_retries(tmp_path: Path, destination: Path, retries: int = 5) -> None:
    """Replace a destination file, retrying briefly for transient Windows locks."""

    last_error: OSError | None = None
    for attempt in range(retries):
        try:
            tmp_path.replace(destination)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    if last_error is not None:
        raise last_error


def _download_pdf(pdf_url: str, destination: Path) -> str:
    """Download a single PDF and return a download status."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and is_pdf_file(destination):
        return "ALREADY_EXISTS"

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(
            pdf_url,
            stream=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        handle.write(chunk)
        if not is_pdf_file(tmp_path):
            _safe_unlink(tmp_path)
            return f"FAILED_NOT_PDF_CONTENT_TYPE={content_type}"
        _replace_with_retries(tmp_path, destination)
        return "DOWNLOADED"
    except Exception as exc:
        _safe_unlink(tmp_path)
        return f"FAILED_{type(exc).__name__}: {exc}"


def download_bulletins(project_root: Path | None = None, checkpoint_every: int = 25) -> pd.DataFrame:
    """Download all PDFs referenced by ``bulletins_index.csv`` and update statuses."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    index = read_csv_if_exists(paths.bulletins_index_csv)
    if index.empty:
        raise FileNotFoundError(f"No bulletin index found at {paths.bulletins_index_csv}. Run scrape_bulletins first.")

    if "download_status" not in index.columns:
        index["download_status"] = "PENDING"
    iterator = tqdm(index.itertuples(), total=len(index), desc="Downloading bulletins")
    for position, row in enumerate(iterator, start=1):
        year = int(getattr(row, "year"))
        pdf_url = str(getattr(row, "pdf_url"))
        filename = str(getattr(row, "pdf_filename") or safe_filename_from_url(pdf_url))
        destination = local_pdf_path(paths, year, filename)
        status = _download_pdf(pdf_url, destination)
        index.at[row.Index, "download_status"] = status
        index.at[row.Index, "local_pdf_path"] = str(destination.relative_to(paths.project_root))
        if checkpoint_every > 0 and position % checkpoint_every == 0:
            write_csv(index, paths.bulletins_index_csv)
            LOGGER.info("Checkpointed download index after %s/%s files.", position, len(index))

    write_csv(index, paths.bulletins_index_csv)
    LOGGER.info("Updated download statuses -> %s", paths.bulletins_index_csv)
    return index


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    download_bulletins()


if __name__ == "__main__":
    main()
