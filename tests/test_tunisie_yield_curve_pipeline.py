from pathlib import Path

import pandas as pd

from data.tunisie_yield_curve.clean_corporate_curves import clean_corporate_curves
from data.tunisie_yield_curve.clean_sovereign_curves import clean_sovereign_curves
from data.tunisie_yield_curve.common import parse_number, resolve_paths
from data.tunisie_yield_curve.fetch_corporate_curves_daily import _parse_actuarial_report


def test_parse_number_accepts_french_decimal_text():
    assert parse_number("7,50 %") == 7.50
    assert parse_number(" 0,0831 ") == 0.0831


def test_clean_sovereign_keeps_official_points_only(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "curve_date": "2025-01-02",
                "maturity": 5,
                "maturity_unit": "YEARS",
                "maturity_years": 5,
                "zero_coupon_rate_percent": 7.5,
                "zero_coupon_rate_decimal": 0.075,
                "source": "official-api",
                "source_url": "https://example.test",
                "extraction_method": "requests_json",
                "extraction_status": "EXTRACTED",
                "raw_value": "{}",
            }
        ]
    ).to_csv(paths.sovereign_raw_csv, index=False)

    clean = clean_sovereign_curves(tmp_path)

    assert list(clean["date"]) == ["2025-01-02"]
    assert clean.loc[0, "zero_coupon_rate"] == 0.075
    assert clean.loc[0, "zero_coupon_rate_percent"] == 7.5
    assert clean.loc[0, "data_quality_flag"] == "OK"


def test_parse_and_clean_corporate_actuarial_excel(tmp_path: Path):
    paths = resolve_paths(tmp_path)
    paths.raw_corporate_dir.mkdir(parents=True, exist_ok=True)
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    report = paths.raw_corporate_dir / "2025-12-31_actuarial.xlsx"
    with pd.ExcelWriter(report, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {
                    "Maturité Résiduelle": 2.5,
                    "YTM": 0.091,
                    "Prix Pied Coupon": 99.4,
                    "Prix Plein Coupon": 101.2,
                }
            ]
        ).to_excel(writer, sheet_name="2025-12-31LEASING", index=False)

    rows = _parse_actuarial_report(report, "2025-12-31")
    pd.DataFrame(rows).to_csv(paths.corporate_raw_csv, index=False)
    clean = clean_corporate_curves(tmp_path)

    assert rows[0]["sector"] == "LEASING"
    assert rows[0]["maturity_years"] == 2.5
    assert rows[0]["ytm_percent"] == 9.1
    assert clean.loc[0, "corporate_rate"] == 0.091
    assert clean.loc[0, "data_quality_flag"] == "OK"
