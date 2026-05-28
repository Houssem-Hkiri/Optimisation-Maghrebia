"""Robustness helpers for notebook 02.

These functions add institutional validation blocks around the internal
optimisation engine. They do not replace the main recommendation logic.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from .optimization_apt import (
    APTOptimizationConfig,
    cvxpy_constraints,
    optimized_weights_table,
    portfolio_metrics,
)


def _as_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w[np.abs(w) < 1e-10] = 0.0
    total = float(w.sum())
    if total <= 1e-12:
        raise ValueError("weights sum is zero")
    return w / total


def solve_mean_cvar_turnover_constrained(
    mu: pd.Series,
    sigma: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    regulatory_map: pd.DataFrame,
    rf_annual: float,
    current_weights: np.ndarray,
    config: APTOptimizationConfig,
    turnover_limits: tuple[float, ...] = (0.20, 0.30, 0.40),
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Solve CVaR portfolios with explicit L1 turnover limits."""

    ret = returns.to_numpy(float)
    mu_v = mu.to_numpy(float)
    current = _as_weights(current_weights)
    n_obs, n_assets = ret.shape
    rows: list[dict[str, object]] = []
    compliances: list[pd.DataFrame] = []
    portfolios: dict[str, np.ndarray] = {}
    current_return = float(mu_v @ current)

    for limit in turnover_limits:
        label = "Mean_CVaR_Turnover_Constrained" if abs(limit - 0.30) < 1e-12 else f"Mean_CVaR_Turnover_{int(round(limit * 100))}"
        local_config = replace(config, turnover_thresholds=(float(limit),))
        w = cp.Variable(n_assets)
        alpha = cp.Variable()
        u = cp.Variable(n_obs, nonneg=True)
        loss = -ret @ w
        cvar = alpha + (1.0 / ((1.0 - config.cvar_beta) * n_obs)) * cp.sum(u)
        constraints = cvxpy_constraints(w, universe, current, context, local_config, float(limit))
        constraints += [u >= loss - alpha, mu_v @ w >= current_return - 1e-10]
        problem = cp.Problem(cp.Minimize(cvar - 0.05 * (mu_v @ w)), constraints)
        try:
            problem.solve(solver="CLARABEL", verbose=False)
        except Exception:
            problem.solve(verbose=False)

        if w.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
            weights = current.copy()
            status = f"FAILED_{problem.status}"
            success = False
        else:
            weights = _as_weights(np.maximum(np.asarray(w.value, dtype=float), 0.0))
            status = str(problem.status)
            success = True

        portfolios[label] = weights
        row, compliance = portfolio_metrics(
            label,
            weights,
            mu,
            sigma,
            returns,
            rf_annual,
            current,
            universe,
            context,
            regulatory_map,
            status,
        )
        row["turnover_limit"] = float(limit)
        row["success"] = bool(success)
        row["Comment"] = (
            "CVaR avec contrainte explicite de turnover ; variante de robustesse institutionnelle."
            if success
            else "Non retenu : solveur non convergent sous la limite de turnover."
        )
        rows.append(row)
        compliances.append(compliance)

    weights_df = optimized_weights_table(portfolios, universe, context)
    metrics_df = pd.DataFrame(rows)
    compliance_df = pd.concat(compliances, ignore_index=True) if compliances else pd.DataFrame()
    return portfolios, metrics_df, weights_df


def build_robust_cvar_wasserstein(
    skfolio_metrics: pd.DataFrame,
    skfolio_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the distributionally robust CVaR benchmark from skfolio output."""

    model = "Skfolio_DistributionallyRobustCVaR"
    rows = skfolio_metrics.loc[skfolio_metrics["Model"].eq(model)].copy()
    if rows.empty:
        return (
            pd.DataFrame(
                [
                    {
                        "Model": "Robust_CVaR_Wasserstein",
                        "Status": "NOT_EXECUTED",
                        "Reason": "skfolio DistributionallyRobustCVaR non disponible.",
                    }
                ]
            ),
            pd.DataFrame(columns=["Model", "Asset", "Weight"]),
        )
    rows["Model"] = "Robust_CVaR_Wasserstein"
    rows["Interpretation"] = (
        "Benchmark robuste de CVaR sous incertitude de distribution ; non utilisé comme recommandation principale."
    )
    weights = skfolio_weights.loc[skfolio_weights["Model"].eq(model)].copy()
    if not weights.empty:
        weights["Model"] = "Robust_CVaR_Wasserstein"
    return rows, weights


def build_rorac_proxy(candidates: pd.DataFrame) -> pd.DataFrame:
    """Build a simplified return-to-risk proxy, not a regulatory capital model."""

    df = candidates.copy()
    if "Model" not in df.columns and "portfolio_name" in df.columns:
        df = df.rename(columns={"portfolio_name": "Model"})
    rename = {
        "expected_return_APT": "Expected_Return_APT",
        "cvar_95_historical": "CVaR_95",
        "herfindahl_index": "HHI",
        "regulatory_status": "Regulatory_Status",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ["Expected_Return_APT", "CVaR_95", "HHI"]:
        if col not in df.columns:
            df[col] = np.nan
    if "Regulatory_Status" not in df.columns:
        df["Regulatory_Status"] = "NOT_AVAILABLE"

    cvar = df["CVaR_95"].abs().replace(0, np.nan)
    max_weight = df["Max_Weight"] if "Max_Weight" in df.columns else np.sqrt(df["HHI"].clip(lower=0))
    concentration_penalty = 0.02 * df["HHI"].fillna(0.0) + 0.01 * max_weight.fillna(0.0)
    regulatory_penalty = np.select(
        [
            df["Regulatory_Status"].astype(str).str.contains("FAILED|BREACH", case=False, na=False),
            df["Regulatory_Status"].astype(str).str.contains("NON_TESTABLE|UNTESTED", case=False, na=False),
        ],
        [0.02, 0.005],
        default=0.0,
    )
    denominator = cvar.fillna(df["CVaR_95"].abs().median() if df["CVaR_95"].notna().any() else 0.01)
    df["Return_to_CVaR"] = df["Expected_Return_APT"] / denominator.replace(0, np.nan)
    df["Return_to_CVaR_Adjusted"] = df["Expected_Return_APT"] / (denominator + concentration_penalty + regulatory_penalty)
    df["Interpretation"] = np.where(
        df["Regulatory_Status"].astype(str).str.contains("FAILED|BREACH", case=False, na=False),
        "Ratio indicatif pénalisé par une limite réglementaire ou interne.",
        "Proxy indicatif de rendement par unité de risque consommé.",
    )
    cols = [
        "Model",
        "Expected_Return_APT",
        "CVaR_95",
        "HHI",
        "Regulatory_Status",
        "Return_to_CVaR",
        "Return_to_CVaR_Adjusted",
        "Interpretation",
    ]
    return df[[c for c in cols if c in df.columns]].sort_values("Return_to_CVaR_Adjusted", ascending=False)


def build_factor_risk_sensitivity(
    metrics: pd.DataFrame,
    stress_tests: pd.DataFrame,
    key_models: list[str],
) -> pd.DataFrame:
    """Build a factor-sensitivity fallback when APT loadings are unavailable."""

    rows: list[dict[str, object]] = []
    metrics_idx = metrics.set_index("portfolio_name") if "portfolio_name" in metrics.columns else pd.DataFrame()
    for model in key_models:
        stress = stress_tests.loc[stress_tests["portfolio_name"].eq(model)].copy()
        market = stress.loc[stress["scenario_name"].str.contains("actions", case=False, na=False), "estimated_portfolio_impact"].abs().max()
        rates = stress.loc[stress["scenario_name"].str.contains("taux", case=False, na=False), "estimated_portfolio_impact"].abs().max()
        credit = stress.loc[stress["scenario_name"].str.contains("spread", case=False, na=False), "estimated_portfolio_impact"].abs().max()
        cvar = float(metrics_idx.loc[model, "cvar_95_historical"]) if not metrics_idx.empty and model in metrics_idx.index else np.nan
        explained = np.nansum([market, rates, credit])
        rows.append(
            {
                "Portfolio": model,
                "Market_Factor_Contribution": market if pd.notna(market) else np.nan,
                "Rate_Factor_Contribution": rates if pd.notna(rates) else np.nan,
                "Credit_Spread_Contribution": credit if pd.notna(credit) else np.nan,
                "Inflation_Contribution": np.nan,
                "Specific_Risk_Contribution": max(0.0, cvar - explained) if pd.notna(cvar) else np.nan,
                "Comment": "Sensibilité par chocs utilisée faute de matrice de loadings APT complète.",
            }
        )
    return pd.DataFrame(rows)


def build_pareto_filter(candidates: pd.DataFrame) -> pd.DataFrame:
    """Identify portfolios not dominated on return, volatility, CVaR, HHI and turnover."""

    df = candidates.copy().reset_index(drop=True)
    if "Model" not in df.columns and "portfolio_name" in df.columns:
        df = df.rename(columns={"portfolio_name": "Model"})
    rename = {
        "expected_return_APT": "Expected_Return_APT",
        "volatility_APT": "Volatility_APT",
        "cvar_95_historical": "CVaR_95",
        "herfindahl_index": "HHI",
        "turnover": "Turnover_vs_Current",
        "regulatory_status": "Regulatory_Status",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = ["Expected_Return_APT", "Volatility_APT", "CVaR_95", "HHI", "Turnover_vs_Current"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    if "Regulatory_Status" not in df.columns:
        df["Regulatory_Status"] = "NOT_AVAILABLE"
    work = df.dropna(subset=required).copy()
    work = work.loc[~work["Regulatory_Status"].astype(str).str.contains("FAILED|BREACH", case=False, na=False)].copy()
    dominated_by: dict[int, str] = {}
    values = work[required].to_numpy(float)
    indices = work.index.to_list()
    for pos_i, idx_i in enumerate(indices):
        ret_i, vol_i, cvar_i, hhi_i, turn_i = values[pos_i]
        for pos_j, idx_j in enumerate(indices):
            if idx_i == idx_j:
                continue
            ret_j, vol_j, cvar_j, hhi_j, turn_j = values[pos_j]
            no_worse = (
                ret_j >= ret_i - 1e-12
                and vol_j <= vol_i + 1e-12
                and cvar_j <= cvar_i + 1e-12
                and hhi_j <= hhi_i + 1e-12
                and turn_j <= turn_i + 1e-12
            )
            strictly_better = (
                ret_j > ret_i + 1e-12
                or vol_j < vol_i - 1e-12
                or cvar_j < cvar_i - 1e-12
                or hhi_j < hhi_i - 1e-12
                or turn_j < turn_i - 1e-12
            )
            if no_worse and strictly_better:
                dominated_by[idx_i] = str(work.loc[idx_j, "Model"])
                break
    df["Is_Pareto_Efficient"] = ~df.index.isin(dominated_by)
    df["Dominated_By"] = df.index.map(dominated_by).fillna("")
    df["Pareto_Comment"] = np.where(
        df["Is_Pareto_Efficient"],
        "Non dominé sur les critères rendement-risque-concentration-turnover.",
        "Dominé par une alternative plus équilibrée sur les critères retenus.",
    )
    cols = ["Model", "Source", *required, "Regulatory_Status", "Is_Pareto_Efficient", "Dominated_By", "Pareto_Comment"]
    return df[[c for c in cols if c in df.columns]]


def bootstrap_stability_check(
    returns: pd.DataFrame,
    mu: pd.Series,
    weights: np.ndarray,
    universe: pd.DataFrame,
    n_bootstrap: int = 300,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate recommended portfolio stability under return resampling."""

    rng = np.random.default_rng(random_seed)
    ret = returns.to_numpy(float)
    w = _as_weights(weights)
    n_obs = ret.shape[0]
    sampled_returns: list[float] = []
    sampled_cvar: list[float] = []
    sampled_volatility: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n_obs, size=n_obs)
        series = ret[idx] @ w
        losses = -series
        var_95 = float(np.quantile(losses, 0.95))
        tail = losses[losses >= var_95]
        sampled_returns.append(float(mu.to_numpy(float) @ w))
        sampled_cvar.append(float(tail.mean()) if len(tail) else var_95)
        sampled_volatility.append(float(np.std(series, ddof=1) * math.sqrt(252.0)))

    asset = universe[["asset_id", "asset_name", "asset_class"]].copy()
    asset["Mean_Weight"] = w
    asset["Std_Weight"] = 0.0
    asset["Selection_Frequency"] = (w > 0.0001).astype(float)
    asset["Stability_Status"] = np.where(w > 0.0001, "POSITION_STABLE_EVALUATED", "NOT_SELECTED")
    asset["Comment"] = "Poids fixes ; stabilité évaluée sur les métriques par bootstrap."

    portfolio = pd.DataFrame(
        [
            {
                "Portfolio": "Portefeuille recommandé",
                "Mean_Return": float(np.mean(sampled_returns)),
                "Std_Return": float(np.std(sampled_returns, ddof=1)),
                "Mean_CVaR": float(np.mean(sampled_cvar)),
                "Std_CVaR": float(np.std(sampled_cvar, ddof=1)),
                "Mean_Volatility": float(np.mean(sampled_volatility)),
                "Std_Volatility": float(np.std(sampled_volatility, ddof=1)),
                "N_Bootstrap": int(n_bootstrap),
                "Bootstrap_Stability_Status": "PERFORMANCE_STABILITY_EVALUATED",
                "Comment": "Test de stabilité des métriques ; pas un modèle d'allocation supplémentaire.",
            }
        ]
    )
    return asset, portfolio
