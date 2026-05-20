"""External skfolio benchmarks for notebook 02.

The custom optimisation engine remains the methodological reference. This
module only provides optional external benchmarks and recalculates all metrics
with the project's APT expected returns, APT covariance and custom regulatory
checks.
"""

from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .optimization_apt import evaluate_constraints


@dataclass(frozen=True)
class SkfolioStatus:
    available: bool
    message: str


def _import_skfolio() -> tuple[SkfolioStatus, dict[str, Any]]:
    """Import skfolio components defensively."""

    try:
        import skfolio  # noqa: F401
    except ImportError:
        return SkfolioStatus(False, "skfolio non installé ; benchmark externe non exécuté. Installer avec : pip install skfolio"), {}

    objects: dict[str, Any] = {}
    try:
        from skfolio.optimization import (
            DistributionallyRobustCVaR,
            HierarchicalRiskParity,
            InverseVolatility,
            MaximumDiversification,
            MeanRisk,
            ObjectiveFunction,
            RiskBudgeting,
        )
        from skfolio.measures import RiskMeasure
    except Exception as exc:  # pragma: no cover - depends on local package version
        return SkfolioStatus(False, f"skfolio installé mais API incompatible : {str(exc).splitlines()[0][:160]}"), {}

    objects.update(
        DistributionallyRobustCVaR=DistributionallyRobustCVaR,
        HierarchicalRiskParity=HierarchicalRiskParity,
        InverseVolatility=InverseVolatility,
        MaximumDiversification=MaximumDiversification,
        MeanRisk=MeanRisk,
        ObjectiveFunction=ObjectiveFunction,
        RiskBudgeting=RiskBudgeting,
        RiskMeasure=RiskMeasure,
    )
    return SkfolioStatus(True, "skfolio disponible ; benchmark externe exécuté."), objects


def _short_error(exc: Exception) -> str:
    return str(exc).splitlines()[0][:180]


def _short_warnings(caught: list[warnings.WarningMessage]) -> str:
    messages = []
    for warning in caught[:3]:
        message = str(warning.message).splitlines()[0][:140]
        if message:
            messages.append(message)
    return " | ".join(messages)


def _safe_risk_measure(risk_measure_cls: Any, *names: str) -> Any | None:
    for name in names:
        if hasattr(risk_measure_cls, name):
            return getattr(risk_measure_cls, name)
    return None


def _build_estimator(cls: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(cls)
    accepted = {k: v for k, v in kwargs.items() if k in signature.parameters and v is not None}
    return cls(**accepted)


def _clean_weights(raw_weights: Any, n_assets: int, max_weight: float) -> np.ndarray:
    weights = np.asarray(raw_weights, dtype=float).reshape(-1)
    if len(weights) != n_assets or not np.isfinite(weights).all():
        raise ValueError("Poids skfolio invalides ou dimension incohérente.")
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 1e-12:
        raise ValueError("Poids skfolio nuls après nettoyage.")
    weights = weights / total
    if float(weights.max()) > max_weight + 1e-6:
        raise ValueError(f"Poids maximum superieur a la borne interne ({weights.max():.2%}).")
    return weights


def _portfolio_var_cvar(returns: np.ndarray) -> tuple[float, float]:
    losses = -np.asarray(returns, dtype=float)
    var_95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var_95]
    cvar_95 = float(tail.mean()) if len(tail) else var_95
    return max(0.0, var_95), max(0.0, cvar_95)


def _metrics(
    model_name: str,
    weights: np.ndarray,
    returns_df: pd.DataFrame,
    mu_apt: pd.Series,
    cov_apt: pd.DataFrame,
    current_weights: np.ndarray,
    universe: pd.DataFrame,
    context: dict[str, object],
    regulatory_map: pd.DataFrame,
    rf_annual: float,
) -> dict[str, object]:
    values = returns_df.to_numpy(float) @ weights
    expected = float(mu_apt.to_numpy(float) @ weights)
    variance = float(weights.T @ cov_apt.to_numpy(float) @ weights)
    volatility = math.sqrt(max(variance, 0.0))
    var_95, cvar_95 = _portfolio_var_cvar(values)
    compliance = evaluate_constraints(weights, universe, context, regulatory_map, model_name)
    breaches = int(compliance["compliance_status"].eq("BREACH").sum())
    missing = int(compliance["compliance_status"].eq("CHECK_NOT_ENFORCED_MISSING_DATA").sum())
    regulatory_status = "FAILED" if breaches else ("NON_TESTABLE_DATA_MISSING" if missing else "PASSED")
    return {
        "Model": model_name,
        "Source": "skfolio",
        "Expected_Return_APT": expected,
        "Volatility_APT": volatility,
        "Sharpe_Indicative": (expected - rf_annual) / volatility if volatility > 1e-12 else np.nan,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Max_Weight": float(weights.max()),
        "HHI": float(np.sum(weights**2)),
        "Active_Positions": int(np.sum(weights > 0.0001)),
        "Turnover_vs_Current": float(np.sum(np.abs(weights - current_weights))),
        "Regulatory_Status": regulatory_status,
        "Status": "PASSED" if regulatory_status != "FAILED" else "FAILED",
        "Comment": "Benchmark externe skfolio ; métriques recalculées avec les inputs APT du projet.",
    }


def _failed_row(model_name: str, reason: str) -> dict[str, object]:
    return {
        "Model": model_name,
        "Source": "skfolio",
        "Expected_Return_APT": np.nan,
        "Volatility_APT": np.nan,
        "Sharpe_Indicative": np.nan,
        "VaR_95": np.nan,
        "CVaR_95": np.nan,
        "Max_Weight": np.nan,
        "HHI": np.nan,
        "Active_Positions": np.nan,
        "Turnover_vs_Current": np.nan,
        "Regulatory_Status": "NOT_AVAILABLE",
        "Status": "FAILED",
        "Reason": reason,
        "Comment": "Échec isolé du benchmark ; le moteur interne reste inchangé.",
    }


def _estimator_specs(objects: dict[str, Any], max_weight: float) -> list[tuple[str, Any] | tuple[str, Exception]]:
    RiskMeasure = objects["RiskMeasure"]
    ObjectiveFunction = objects["ObjectiveFunction"]
    cvar = _safe_risk_measure(RiskMeasure, "CVAR", "CVaR", "CONDITIONAL_VALUE_AT_RISK")
    semivar = _safe_risk_measure(RiskMeasure, "SEMI_VARIANCE", "SEMI_VARIANCE_ANNUALIZED")
    variance = _safe_risk_measure(RiskMeasure, "VARIANCE")

    specs: list[tuple[str, Any] | tuple[str, Exception]] = []

    def add(name: str, cls: Any, **kwargs: Any) -> None:
        try:
            specs.append((name, _build_estimator(cls, min_weights=0.0, max_weights=max_weight, **kwargs)))
        except Exception as exc:  # pragma: no cover - package-version dependent
            specs.append((name, exc))

    add("Skfolio_InverseVolatility", objects["InverseVolatility"])
    add("Skfolio_MaximumDiversification", objects["MaximumDiversification"])
    if cvar is not None:
        add("Skfolio_Min_CVaR", objects["MeanRisk"], risk_measure=cvar, objective_function=ObjectiveFunction.MINIMIZE_RISK, cvar_beta=0.95)
        add("Skfolio_CVaR_RiskBudgeting", objects["RiskBudgeting"], risk_measure=cvar, cvar_beta=0.95, solver="SCS")
        add("Skfolio_HRP_CVaR", objects["HierarchicalRiskParity"], risk_measure=cvar)
    else:
        specs.append(("Skfolio_Min_CVaR", ValueError("RiskMeasure CVaR indisponible.")))
        specs.append(("Skfolio_CVaR_RiskBudgeting", ValueError("RiskMeasure CVaR indisponible.")))
        if variance is not None:
            add("Skfolio_HRP_Variance", objects["HierarchicalRiskParity"], risk_measure=variance)
    if variance is not None:
        add("Skfolio_HRP_Variance", objects["HierarchicalRiskParity"], risk_measure=variance)
    if semivar is not None:
        add("Skfolio_Min_SemiVariance", objects["MeanRisk"], risk_measure=semivar, objective_function=ObjectiveFunction.MINIMIZE_RISK)
    else:
        specs.append(("Skfolio_Min_SemiVariance", ValueError("RiskMeasure Semi-Variance indisponible.")))
    add("Skfolio_DistributionallyRobustCVaR", objects["DistributionallyRobustCVaR"], cvar_beta=0.95, wasserstein_ball_radius=0.02)
    return specs


def run_skfolio_benchmarks(
    returns_df: pd.DataFrame,
    mu_apt: pd.Series,
    cov_apt: pd.DataFrame,
    asset_names: list[str],
    current_weights: np.ndarray,
    constraints_config: Any,
    universe: pd.DataFrame,
    context: dict[str, object],
    regulatory_map: pd.DataFrame,
    rf_annual: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run optional skfolio benchmarks and recalculate metrics internally.

    Returns
    -------
    weights_df, metrics_df, status_df
    """

    status, objects = _import_skfolio()
    status_df = pd.DataFrame([{"SKFOLIO_AVAILABLE": status.available, "Message": status.message}])
    if not status.available:
        metrics = pd.DataFrame([_failed_row("Skfolio_Status", status.message)])
        metrics.loc[0, "Status"] = "SKIPPED"
        metrics.loc[0, "Regulatory_Status"] = "NOT_EXECUTED"
        metrics.loc[0, "Comment"] = "Benchmark externe non exécuté ; le moteur interne reste la référence."
        return pd.DataFrame(columns=["Model", "Asset", "Weight"]), metrics, status_df

    x = returns_df.loc[:, asset_names].copy()
    max_weight = float(constraints_config.max_weight_per_asset)
    n_assets = len(asset_names)
    metrics_rows: list[dict[str, object]] = []
    weights_rows: list[dict[str, object]] = []

    for model_name, estimator_or_exc in _estimator_specs(objects, max_weight):
        if isinstance(estimator_or_exc, Exception):
            metrics_rows.append(_failed_row(model_name, _short_error(estimator_or_exc)))
            continue
        warning_message = ""
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator_or_exc.fit(x)
            warning_message = _short_warnings(caught)
            raw_weights = getattr(estimator_or_exc, "weights_", None)
            if raw_weights is None:
                raise ValueError("Attribut weights_ absent après fit(X).")
            weights = _clean_weights(raw_weights, n_assets, max_weight)
            metric = _metrics(
                model_name,
                weights,
                x,
                mu_apt,
                cov_apt,
                np.asarray(current_weights, dtype=float),
                universe,
                context,
                regulatory_map,
                rf_annual,
            )
            if warning_message:
                metric["Comment"] = f"{metric['Comment']} Avertissement skfolio documenté : {warning_message}"
            metrics_rows.append(metric)
            weights_rows.extend({"Model": model_name, "Asset": asset, "Weight": float(weight)} for asset, weight in zip(asset_names, weights))
        except Exception as exc:  # pragma: no cover - package-version dependent
            reason = _short_error(exc)
            if warning_message:
                reason = f"{reason} | avertissement : {warning_message}"[:180]
            metrics_rows.append(_failed_row(model_name, reason))

    return pd.DataFrame(weights_rows), pd.DataFrame(metrics_rows), status_df
