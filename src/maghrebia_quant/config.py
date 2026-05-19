"""Configuration centrale du projet pfe-maghrebia-quant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"
FIGURES_DIR = EXPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

ANALYSIS_START_DATE = "2025-01-01"
ANALYSIS_END_DATE = "2025-12-31"
PERIODS_PER_YEAR = 52
PERIODS_PER_YEAR_DAILY = 252
ANNUAL_RISK_FREE_RATE = 0.08
RF_ANNUAL_FALLBACK = 0.08
MIN_OBSERVATIONS_DIAGNOSTIC = 26
MIN_OBSERVATIONS_OPTIMIZATION = 60
BTA_NOMINAL = 1000
EN_NOMINAL = 100

TUNISIA_MARKET_HOLIDAYS_2025 = [
    "2025-01-01",
    "2025-03-20",
    "2025-03-31",
    "2025-04-01",
    "2025-04-09",
    "2025-05-01",
    "2025-06-06",
    "2025-06-07",
    "2025-06-26",
    "2025-07-25",
    "2025-08-13",
    "2025-09-04",
    "2025-10-15",
    "2025-12-17",
]

PORTFOLIO_FILE = "Maghrebia Portfolio.xlsx"
BVMT_PRICE_FILES = ["histo_cotation_2025.csv"]
SOVEREIGN_CURVES_FILE = "data/raw/sovereign_curves_daily_raw_2025.csv"
CORPORATE_CURVES_FILE = "data/raw/corporate_curves_daily_raw_2025.csv"

QUALITY_FLAGS = [
    "RETURN_SERIES_OK",
    "SHORT_SERIES_WARNING",
    "DATA_MISSING",
    "DIVIDENDS_NOT_INCLUDED",
    "DIVIDEND_PRICE_RETURN_ONLY",
    "CORPORATE_RETURN_OUTLIER_EXCLUDED",
    "CORPORATE_ACTION_ADJUSTED",
    "ZC_MATURITY_OUT_OF_RANGE",
    "SPREAD_SECTOR_MISSING",
    "SPREAD_LAST_AVAILABLE_USED",
    "MODEL_BASED_VALUATION",
    "MODEL_BASED_PROXY_CORPORATE_VALUATION",
    "PRO_FORMA_BACKFILLED_FIXED_INCOME_SERIES",
    "LOW_VOLATILITY_WARNING",
    "ONE_WAY_RETURN_SERIES_WARNING",
    "SMOOTHED_MODEL_SERIES_WARNING",
    "RECONCILIATION_GAP_HIGH",
    "RISK_FREE_RATE_FALLBACK_USED",
    "COVARIANCE_NOT_PSD",
    "NEGATIVE_RISK_CONTRIBUTION",
    "LOOK_AHEAD_CURRENT_HOLDINGS_WARNING",
    "CORPORATE_ACTION_REQUIRES_MANUAL_VALIDATION",
    "MISSING_BOND_TERMS",
    "MODEL_BASED_CORPORATE_VALUATION",
    "MISSING_SECURITY_IDENTIFIER",
    "MODEL_NOT_RELIABLE_WITHOUT_TERMS",
    "RISK_FREE_RATE_MISSING",
    "CURRENT_PORTFOLIO_PARTIAL_COVERAGE",
]


@dataclass(frozen=True)
class BondDefaults:
    """Paramètres documentés, non utilisés pour inventer des termes manquants."""

    bta_nominal: float = 1000.0
    national_bond_nominal: float = 100.0
    corporate_nominal: float = 100.0


BOND_DEFAULTS = BondDefaults()


def analysis_start() -> "pd.Timestamp":
    """Date de début de fenêtre sous forme Timestamp."""

    import pandas as pd

    return pd.Timestamp(ANALYSIS_START_DATE)


def analysis_end() -> "pd.Timestamp":
    """Date de fin de fenêtre sous forme Timestamp."""

    import pandas as pd

    return pd.Timestamp(ANALYSIS_END_DATE)


def ensure_directories() -> None:
    """Crée les dossiers validés du projet."""

    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXPORTS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def data_path(filename: str) -> Path:
    """Retourne un input en cherchant dans data/raw puis data."""

    candidate = Path(filename)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    project_relative = PROJECT_ROOT / candidate
    if project_relative.exists():
        return project_relative
    raw = RAW_DIR / filename
    if raw.exists():
        return raw
    direct = DATA_DIR / filename
    if direct.exists():
        return direct
    return raw
