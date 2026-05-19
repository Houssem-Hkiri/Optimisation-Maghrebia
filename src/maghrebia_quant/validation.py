"""Contrôles qualité et diagnostic de lissage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PERIODS_PER_YEAR


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown sur une série de rendements."""

    r = returns.dropna()
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def detect_smoothed_series(returns: pd.Series) -> dict[str, object]:
    """Détecte les séries obligataires potentiellement trop lissées."""

    r = returns.dropna()
    if r.empty:
        return {
            "annualized_volatility": np.nan,
            "number_negative_weeks": 0,
            "share_positive_weeks": np.nan,
            "autocorrelation_lag1": np.nan,
            "max_drawdown": np.nan,
            "min_return": np.nan,
            "max_return": np.nan,
            "flags": "DATA_MISSING",
        }
    annualized_volatility = float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(r) > 1 else np.nan
    number_negative = int((r < 0).sum())
    share_positive = float((r > 0).mean())
    autocorr = float(r.autocorr(lag=1)) if len(r) > 2 else np.nan
    dd = max_drawdown(r)
    flags: list[str] = []
    if pd.notna(annualized_volatility) and annualized_volatility < 0.005:
        flags.append("LOW_VOLATILITY_WARNING")
    if pd.notna(share_positive) and share_positive > 0.95:
        flags.append("ONE_WAY_RETURN_SERIES_WARNING")
    if pd.notna(autocorr) and autocorr > 0.70:
        flags.append("SMOOTHED_MODEL_SERIES_WARNING")
    if pd.notna(dd) and abs(dd) < 0.001:
        flags.append("LOW_DRAWDOWN_WARNING")
    return {
        "annualized_volatility": annualized_volatility,
        "number_negative_weeks": number_negative,
        "share_positive_weeks": share_positive,
        "autocorrelation_lag1": autocorr,
        "max_drawdown": dd,
        "min_return": float(r.min()),
        "max_return": float(r.max()),
        "flags": ";".join(flags) if flags else "OK",
    }
