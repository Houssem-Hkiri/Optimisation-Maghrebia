"""Extract full text from Tunisie Clearing bulletin PDFs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    def tqdm(iterable, **_: object):
        return iterable

from .common import configure_logging, detect_keywords, ensure_directories, is_pdf_file, read_csv_if_exists, resolve_paths, write_csv

LOGGER = logging.getLogger(__name__)


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    """Extract PDF text with pdfplumber."""

    import pdfplumber

    texts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
    return "\n".join(texts).strip()


def _extract_with_pypdf(pdf_path: Path) -> str:
    """Extract PDF text with pypdf as a fallback."""

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    texts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(texts).strip()


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, str, str]:
    """Extract text from one PDF and return ``(text, method, status)``."""

    if not pdf_path.exists():
        return "", "", "FAILED_FILE_NOT_FOUND"
    if not is_pdf_file(pdf_path):
        return "", "", "FAILED_NOT_A_VALID_PDF"

    errors: list[str] = []
    for method, extractor in (("pdfplumber", _extract_with_pdfplumber), ("pypdf", _extract_with_pypdf)):
        try:
            text = extractor(pdf_path)
            if text:
                return text, method, "EXTRACTED"
            errors.append(f"{method}: empty text")
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")
    return "", "", "FAILED_" + " | ".join(errors)[:450]


def extract_pdf_text(project_root: Path | None = None, checkpoint_every: int = 25) -> pd.DataFrame:
    """Extract text for all downloaded bulletins and save an extraction index."""

    paths = resolve_paths(project_root)
    ensure_directories(paths)
    bulletins = read_csv_if_exists(paths.bulletins_index_csv)
    if bulletins.empty:
        raise FileNotFoundError(f"No bulletin index found at {paths.bulletins_index_csv}.")

    previous_index = read_csv_if_exists(paths.extracted_text_index_csv)
    previous_by_pdf = {
        str(row.get("pdf_filename")): row
        for _, row in previous_index.iterrows()
    } if not previous_index.empty else {}
    rows: list[dict[str, object]] = []
    iterator = tqdm(bulletins.itertuples(index=False), total=len(bulletins), desc="Extracting PDF text")
    for position, row in enumerate(iterator, start=1):
        local_pdf_rel = Path(str(getattr(row, "local_pdf_path")))
        local_pdf = paths.project_root / local_pdf_rel
        year = int(getattr(row, "year"))
        text_output = paths.text_dir / str(year) / f"{Path(str(getattr(row, 'pdf_filename'))).stem}.txt"
        text_output.parent.mkdir(parents=True, exist_ok=True)

        previous = previous_by_pdf.get(str(getattr(row, "pdf_filename")))
        if text_output.exists() and text_output.stat().st_size > 0:
            text = text_output.read_text(encoding="utf-8", errors="replace")
            method = str(previous.get("extraction_method")) if previous is not None and previous.get("extraction_method") else "cached_text"
            status = "EXTRACTED"
        else:
            text, method, status = extract_text_from_pdf(local_pdf)
        if text:
            text_output.write_text(text, encoding="utf-8")
        keywords = detect_keywords(text)
        rows.append(
            {
                "bulletin_date": getattr(row, "bulletin_date"),
                "year": year,
                "month": int(getattr(row, "month")),
                "pdf_filename": getattr(row, "pdf_filename"),
                "local_pdf_path": str(local_pdf_rel),
                "extracted_text_path": str(text_output.relative_to(paths.project_root)) if text else "",
                "extraction_method": method,
                "extraction_status": status,
                "text_length": len(text),
                "contains_corporate_section": any("corporate" in kw.lower() for kw in keywords),
                "detected_keywords": "; ".join(keywords),
            }
        )
        if checkpoint_every > 0 and position % checkpoint_every == 0:
            write_csv(pd.DataFrame(rows), paths.extracted_text_index_csv)
            LOGGER.info("Checkpointed text extraction index after %s/%s files.", position, len(bulletins))

    index = pd.DataFrame(rows)
    write_csv(index, paths.extracted_text_index_csv)
    LOGGER.info("Saved extracted text index: %s rows -> %s", len(index), paths.extracted_text_index_csv)
    return index


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    extract_pdf_text()


if __name__ == "__main__":
    main()
