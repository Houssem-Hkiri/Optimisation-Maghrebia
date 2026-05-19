from pathlib import Path

import pandas as pd

from data.tunisie_clearing.build_corporate_spreads_dataset import build_corporate_spreads_dataset
from data.tunisie_clearing.common import resolve_paths
from data.tunisie_clearing.extract_corporate_spreads import extract_spreads_from_text


def test_extract_transition_leasing_spread():
    text = (
        "Analyse de la courbe des taux CORPORATE. "
        "Pour le secteur leasing, la prime des emprunts obligataires ordinaires "
        "est passee de 0,979 % a 0,955 %."
    )
    bulletin = pd.Series({"bulletin_date": "2024-04-30", "year": 2024, "month": 4})

    rows = extract_spreads_from_text(text, bulletin, source_pdf="bulletin.pdf")

    assert rows
    leasing = [row for row in rows if row["sector"] == "LEASING"][0]
    assert leasing["previous_spread_percent"] == 0.979
    assert leasing["final_spread_percent"] == 0.955
    assert leasing["spread_decimal"] == 0.00955
    assert leasing["extraction_status"] == "EXTRACTED"


def test_extract_direct_2025_sector_spreads():
    text = (
        "Courbe des taux CORPORATE - Prime Finale. "
        "Bancaire : 0,90 %. Leasing : 0,95 %. Microfinance : 1,20 %."
    )
    bulletin = pd.Series({"bulletin_date": "2025-01-31", "year": 2025, "month": 1})

    rows = extract_spreads_from_text(text, bulletin, source_pdf="bulletin.pdf")
    sectors = {row["sector"]: row["spread_percent"] for row in rows}

    assert sectors["BANCAIRE"] == 0.90
    assert sectors["LEASING"] == 0.95
    assert sectors["MICROFINANCE"] == 1.20


def test_build_dataset_creates_clean_outputs(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "bulletin_date": "2025-01-31",
                "year": 2025,
                "month": 1,
                "pdf_filename": "b.pdf",
                "pdf_url": "https://example.test/b.pdf",
                "local_pdf_path": "data/raw/tunisie_clearing_bulletins/2025/b.pdf",
                "download_status": "DOWNLOADED",
            }
        ]
    ).to_csv(paths.bulletins_index_csv, index=False)
    pd.DataFrame(
        [
            {
                "bulletin_date": "2025-01-31",
                "year": 2025,
                "month": 1,
                "pdf_filename": "b.pdf",
                "contains_corporate_section": True,
                "extraction_status": "EXTRACTED",
            }
        ]
    ).to_csv(paths.extracted_text_index_csv, index=False)
    pd.DataFrame(
        [
            {
                "bulletin_date": "2025-01-31",
                "year": 2025,
                "month": 1,
                "sector": "BANCAIRE",
                "spread_type": "PRIME_FINALE",
                "spread_percent": 0.9,
                "spread_decimal": 0.009,
                "previous_spread_percent": None,
                "final_spread_percent": 0.9,
                "source_pdf": "data/raw/tunisie_clearing_bulletins/2025/b.pdf",
                "source_url": "https://example.test/b.pdf",
                "source_text_snippet": "Bancaire : 0,90 %",
                "extraction_method": "pdfplumber",
                "extraction_confidence": "HIGH",
                "data_quality_flag": "OK",
                "extraction_status": "EXTRACTED",
            }
        ]
    ).to_csv(paths.corporate_spreads_raw_csv, index=False)

    clean = build_corporate_spreads_dataset(tmp_path)

    assert not clean.empty
    assert paths.corporate_spreads_clean_csv.exists()
    assert paths.corporate_spreads_clean_xlsx.exists()
    assert paths.quality_report_xlsx.exists()
