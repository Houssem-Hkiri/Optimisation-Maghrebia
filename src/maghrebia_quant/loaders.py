"""Chargeurs et conversions robustes pour les fichiers du diagnostic."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_text(value: object) -> str:
    """Normalise un texte pour les jointures et comparaisons."""

    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).upper()


def slugify(value: object) -> str:
    """Crée un identifiant ASCII stable."""

    text = re.sub(r"[^A-Z0-9]+", "_", normalize_text(value)).strip("_")
    return text or "UNKNOWN"


def standardize_column_name(column: object) -> str:
    """Convertit un nom de colonne en snake_case ASCII."""

    return re.sub(r"[^a-z0-9]+", "_", normalize_text(column).lower()).strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise les colonnes d'une table."""

    out = df.copy()
    out.columns = [standardize_column_name(c) for c in out.columns]
    return out


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convertit des nombres français ou anglais en float."""

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def coerce_rate_to_decimal(series: pd.Series) -> pd.Series:
    """Convertit un taux en décimal si la source est en pourcentage."""

    out = coerce_numeric(series)
    if out.abs().median(skipna=True) > 1:
        out = out / 100.0
    return out


def validate_required_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    """Vérifie les colonnes obligatoires."""

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{context}: colonnes manquantes {missing}")


def read_csv_flexible(path: Path, sep: str = ";") -> pd.DataFrame:
    """Lit un CSV avec les encodages usuels."""

    if not path.exists():
        raise FileNotFoundError(path)
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return standardize_columns(pd.read_csv(path, sep=sep, encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Impossible de lire {path} avec les encodages usuels.")


def filter_date_window(
    df: pd.DataFrame,
    date_col: str = "date",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filtre une table sur une fenêtre inclusive."""

    from .config import ANALYSIS_END_DATE, ANALYSIS_START_DATE

    start_ts = pd.Timestamp(start or ANALYSIS_START_DATE)
    end_ts = pd.Timestamp(end or ANALYSIS_END_DATE)
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out.loc[out[date_col].between(start_ts, end_ts)].copy()


def quality_flag_table(records: list[dict[str, object]]) -> pd.DataFrame:
    """Retourne une table qualité normalisée."""

    columns = ["date", "asset_id", "flag", "message", "severity"]
    if not records:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(records)
    for col in columns:
        if col not in df:
            df[col] = np.nan
    return df[columns].sort_values(["severity", "flag", "asset_id"], ascending=[False, True, True])


def print_input_summary(df: pd.DataFrame, name: str, date_col: str | None = None) -> pd.DataFrame:
    """Retourne un résumé concis d'un input pour audit du notebook."""

    summary = {
        "input": name,
        "rows": len(df),
        "columns": ", ".join(map(str, df.columns)),
        "date_min": pd.NaT,
        "date_max": pd.NaT,
        "top_missing": "",
    }
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        summary["date_min"] = dates.min()
        summary["date_max"] = dates.max()
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(5)
    summary["top_missing"] = "; ".join(f"{idx}: {int(val)}" for idx, val in missing.items())
    return pd.DataFrame([summary])
