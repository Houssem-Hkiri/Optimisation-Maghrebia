"""Shared portfolio metrics for notebook 02."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def _array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _weights(weights: Iterable[float], n: int | None = None) -> np.ndarray:
    w = _array(weights, "weights")
    if n is not None and len(w) != n:
        raise ValueError(f"weights length {len(w)} does not match expected length {n}.")
    total = float(w.sum())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1. Current sum is {total:.10f}.")
    if (w < -1e-8).any():
        raise ValueError("weights must be long-only.")
    return np.maximum(w, 0.0)


def portfolio_return(weights: Iterable[float], mu: pd.Series | Iterable[float]) -> float:
    mu_arr = _array(np.asarray(mu, dtype=float), "mu")
    w = _weights(weights, len(mu_arr))
    return float(w @ mu_arr)


def portfolio_variance(weights: Iterable[float], sigma: pd.DataFrame | np.ndarray) -> float:
    sig = np.asarray(sigma, dtype=float)
    if sig.ndim != 2 or sig.shape[0] != sig.shape[1]:
        raise ValueError("sigma must be a square matrix.")
    w = _weights(weights, sig.shape[0])
    return float(max(w @ sig @ w, 0.0))


def portfolio_volatility(weights: Iterable[float], sigma: pd.DataFrame | np.ndarray) -> float:
    return math.sqrt(portfolio_variance(weights, sigma))


def portfolio_returns_series(weights: Iterable[float], returns: pd.DataFrame) -> pd.Series:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    values = returns.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("returns contains NaN values.")
    w = _weights(weights, values.shape[1])
    return pd.Series(values.to_numpy(float) @ w, index=returns.index, name="portfolio_return")


def portfolio_var(portfolio_returns: pd.Series | Iterable[float], alpha: float) -> float:
    r = _array(np.asarray(portfolio_returns, dtype=float), "portfolio_returns")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    return float(max(0.0, np.quantile(-r, alpha)))


def portfolio_cvar(portfolio_returns: pd.Series | Iterable[float], alpha: float) -> float:
    r = _array(np.asarray(portfolio_returns, dtype=float), "portfolio_returns")
    var = portfolio_var(r, alpha)
    losses = -r
    tail = losses[losses >= var - EPS]
    return float(max(0.0, tail.mean() if len(tail) else var))


def portfolio_drawdown(portfolio_returns: pd.Series | Iterable[float]) -> float:
    r = _array(np.asarray(portfolio_returns, dtype=float), "portfolio_returns")
    wealth = np.cumprod(1.0 + r)
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    return float(dd.min())


def portfolio_sharpe(portfolio_returns: pd.Series | Iterable[float], rf_annual: float, periods_per_year: int = 52) -> float:
    r = _array(np.asarray(portfolio_returns, dtype=float), "portfolio_returns")
    mean_ann = float(np.mean(r) * periods_per_year)
    vol_ann = float(np.std(r, ddof=1) * math.sqrt(periods_per_year))
    return float((mean_ann - rf_annual) / vol_ann) if vol_ann > EPS else np.nan


def portfolio_sortino(portfolio_returns: pd.Series | Iterable[float], rf_annual: float, periods_per_year: int = 52) -> float:
    r = _array(np.asarray(portfolio_returns, dtype=float), "portfolio_returns")
    target_daily = rf_annual / periods_per_year
    downside = np.minimum(r - target_daily, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)) * math.sqrt(periods_per_year))
    mean_ann = float(np.mean(r) * periods_per_year)
    return float((mean_ann - rf_annual) / downside_dev) if downside_dev > EPS else np.nan


def portfolio_hhi(weights: Iterable[float]) -> float:
    w = _weights(weights)
    return float(np.sum(w**2))


def max_weight(weights: Iterable[float]) -> float:
    return float(np.max(_weights(weights)))


def distance_l1(weights: Iterable[float], current_weights: Iterable[float]) -> float:
    w = _weights(weights)
    c = _weights(current_weights, len(w))
    return float(np.abs(w - c).sum())


def turnover_proxy(weights: Iterable[float], current_weights: Iterable[float]) -> float:
    return 0.5 * distance_l1(weights, current_weights)


def risk_contribution(weights: Iterable[float], sigma: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    sig = np.asarray(sigma, dtype=float)
    w = _weights(weights, sig.shape[0])
    variance = float(w @ sig @ w)
    if variance <= EPS:
        raise ValueError("portfolio variance is zero; risk contribution is undefined.")
    vol = math.sqrt(variance)
    marginal = sig @ w / vol
    contribution = w * marginal
    return pd.DataFrame(
        {
            "Weight": w,
            "Marginal_Risk": marginal,
            "Risk_Contribution": contribution,
            "Risk_Contribution_Percent": contribution / vol,
        }
    )


def evaluate_portfolio(
    model: str,
    scenario: str,
    weights: Iterable[float],
    mu: pd.Series,
    sigma: pd.DataFrame,
    returns: pd.DataFrame,
    current_weights: Iterable[float],
    rf_annual: float,
    regulatory_status: str,
    capital_social_status: str,
    solver_status: str,
    constraint_status: str,
    target_roe: float | None = None,
    periods_per_year: int = 52,
) -> dict[str, object]:
    """Evaluate one portfolio with the common notebook 02 metric set."""

    w = _weights(weights, len(mu))
    r = portfolio_returns_series(w, returns)
    expected = portfolio_return(w, mu)
    variance = portfolio_variance(w, sigma)
    target_gap = np.nan if target_roe is None or not np.isfinite(target_roe) else float(target_roe - expected)
    target_shortfall = np.nan if not np.isfinite(target_gap) else max(0.0, target_gap)
    target_excess = np.nan if not np.isfinite(target_gap) else max(0.0, -target_gap)
    row = {
        "Scenario_Methodological_Name": scenario,
        "Model": model,
        "Expected_Return": expected,
        "Expected_Return_Annualized": expected,
        "Portfolio_Return": expected,
        "Volatility": math.sqrt(max(variance, 0.0)),
        "Volatility_Annualized": math.sqrt(max(variance, 0.0)),
        "Volatility_Status": "ALREADY_ANNUALIZED",
        "Variance": variance,
        "VaR_95": portfolio_var(r, 0.95),
        "CVaR_95": portfolio_cvar(r, 0.95),
        "VaR_98_5": portfolio_var(r, 0.985),
        "CVaR_98_5": portfolio_cvar(r, 0.985),
        "VaR_99_5": portfolio_var(r, 0.995),
        "CVaR_99_5": portfolio_cvar(r, 0.995),
        "Max_Drawdown": portfolio_drawdown(r),
        "Sharpe": portfolio_sharpe(r, rf_annual, periods_per_year),
        "Sortino": portfolio_sortino(r, rf_annual, periods_per_year),
        "HHI": portfolio_hhi(w),
        "Max_Weight": max_weight(w),
        "Distance_L1_Current": distance_l1(w, current_weights),
        "Turnover_Proxy": turnover_proxy(w, current_weights),
        "Target_ROE_Gap": target_gap,
        "Target_Return": target_roe,
        "Target_ROE_Shortfall": target_shortfall,
        "Target_ROE_Excess": target_excess,
        "Target_Status": (
            "DATA_MISSING"
            if target_roe is None or not np.isfinite(target_roe)
            else ("TARGET_REACHED_OR_EXCEEDED" if target_shortfall <= EPS else "TARGET_NOT_REACHED")
        ),
        "Regulatory_Status": regulatory_status,
        "Capital_Social_Status": capital_social_status,
        "Solver_Status": solver_status,
        "Constraint_Status": constraint_status,
        "Decision_Eligibility": "ELIGIBLE" if "FAILED" not in str(constraint_status) and "BREACH" not in str(regulatory_status) else "REJECTED_CONSTRAINT",
    }
    for level_label, level in [("95", 0.95), ("98_5", 0.985), ("99_5", 0.995)]:
        var = row[f"VaR_{level_label}"]
        cvar = row[f"CVaR_{level_label}"]
        row[f"VaR_{level_label}_Periodic"] = var
        row[f"CVaR_{level_label}_Periodic"] = cvar
        row[f"VaR_{level_label}_Annualized"] = var * periods_per_year
        row[f"CVaR_{level_label}_Annualized"] = cvar * periods_per_year
    for lam in (2, 5, 10, 20):
        row[f"Utility_Lambda_{lam}"] = expected - (lam / 2.0) * variance
    return row
