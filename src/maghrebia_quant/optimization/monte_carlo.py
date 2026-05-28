"""Monte Carlo exploration for notebook 02."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from maghrebia_quant.optimization_apt import APTOptimizationConfig, class_masks, upper_bounds_vector
from .metrics import distance_l1, portfolio_cvar, portfolio_var
from .optimization_core import issuer_groups, state_min_weight_on_optimisable
from .stress_tests import worst_stress_summary_for_weights


N_MONTE_CARLO = 30_000


def run_monte_carlo(
    mu: pd.Series,
    sigma: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    regulatory_map: pd.DataFrame,
    rf_annual: float,
    scenario: str,
    n_portfolios: int = N_MONTE_CARLO,
    optimized_anchors: list[np.ndarray] | None = None,
    periods_per_year: int = 52,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Generate admissible long-only portfolios and standardize metrics."""

    config = APTOptimizationConfig(monte_carlo_required=int(n_portfolios), monte_carlo_max_attempts=max(600_000, int(n_portfolios) * 40))
    rng = np.random.default_rng(config.random_seed)
    n_assets = len(mu)
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()
    anchors = [current]
    if optimized_anchors:
        for anchor in optimized_anchors:
            a = np.maximum(np.asarray(anchor, dtype=float), 0.0)
            if a.sum() > 0:
                anchors.append(a / a.sum())
    ret = returns.to_numpy(float)
    sig = sigma.to_numpy(float)
    mu_v = mu.to_numpy(float)
    upper = upper_bounds_vector(universe, context, config)
    masks = class_masks(universe)
    state_min, state_status, _ = state_min_weight_on_optimisable(context)
    groups, issuer_status = issuer_groups(universe)
    turnover_limit = config.primary_turnover_limit

    def feasible(candidates: np.ndarray) -> np.ndarray:
        turnover = np.abs(candidates - current).sum(axis=1)
        ok = (
            np.isclose(candidates.sum(axis=1), 1.0, atol=1e-8)
            & (candidates >= -1e-12).all(axis=1)
            & (candidates <= upper + 1e-12).all(axis=1)
            & (candidates[:, masks["listed_equity"]].sum(axis=1) <= config.max_equity_weight + 1e-12)
            & (candidates[:, masks["corporate_bond"]].sum(axis=1) <= config.max_corporate_weight + 1e-12)
            & (turnover <= turnover_limit + 1e-12)
        )
        if state_status == "DATA_AVAILABLE":
            ok &= candidates[:, masks["government_bond"]].sum(axis=1) + 1e-12 >= state_min
        if issuer_status == "PASSED":
            for idx in groups.values():
                ok &= candidates[:, idx].sum(axis=1) <= config.max_weight_per_issuer + 1e-12
        return ok

    def max_violation(weights: np.ndarray) -> tuple[float, int, str]:
        violations = [
            abs(float(weights.sum()) - 1.0),
            max(0.0, -float(weights.min())),
            max(0.0, float(np.max(weights - upper))),
            max(0.0, float(weights[masks["listed_equity"]].sum() - config.max_equity_weight)),
            max(0.0, float(weights[masks["corporate_bond"]].sum() - config.max_corporate_weight)),
            max(0.0, float(np.abs(weights - current).sum() - turnover_limit)),
        ]
        if state_status == "DATA_AVAILABLE":
            violations.append(max(0.0, float(state_min - weights[masks["government_bond"]].sum())))
        else:
            return np.nan, 0, "NOT_TESTED_DATA_MISSING"
        if issuer_status == "PASSED":
            violations.extend(max(0.0, float(weights[idx].sum() - config.max_weight_per_issuer)) for idx in groups.values())
        else:
            return np.nan, 0, "NOT_TESTED_DATA_MISSING"
        max_v = max(violations)
        return float(max_v), int(sum(v > 1e-8 for v in violations)), "PASSED" if max_v <= 1e-8 else "INFEASIBLE_OR_CONSTRAINT_VIOLATION"

    rows: list[dict[str, object]] = []
    portfolio_value = float(context["optimisable_value"])
    standardized_weights: dict[str, np.ndarray] = {}
    attempts = 0
    batch = 20_000
    current_alpha = np.maximum(current * 800.0, 1.0)
    uniform_alpha = np.ones(n_assets)
    feasible_anchors = [a for a in anchors if feasible(np.asarray([a], dtype=float))[0]]
    if not feasible_anchors:
        feasible_anchors = anchors

    while len(rows) < n_portfolios and attempts < config.monte_carlo_max_attempts:
        n_current = int(batch * 0.25)
        n_uniform = int(batch * 0.15)
        n_sparse = int(batch * 0.10)
        n_anchor = batch - n_current - n_uniform - n_sparse
        parts = [rng.dirichlet(current_alpha, size=n_current), rng.dirichlet(uniform_alpha, size=n_uniform)]
        sparse = np.zeros((n_sparse, n_assets))
        k = max(3, min(n_assets, int(np.sqrt(n_assets)) + 2))
        for i in range(n_sparse):
            idx = rng.choice(n_assets, size=k, replace=False)
            sparse[i, idx] = rng.dirichlet(np.ones(k))
        parts.append(sparse)
        anchor_candidates = []
        anchor_matrix = np.vstack(feasible_anchors)
        for _ in range(n_anchor):
            coeff = rng.dirichlet(np.ones(len(feasible_anchors)))
            anchor_candidates.append(coeff @ anchor_matrix)
        parts.append(np.vstack(anchor_candidates) if anchor_candidates else np.empty((0, n_assets)))
        candidates = np.vstack(parts)
        attempts += len(candidates)
        candidates = candidates[feasible(candidates)]
        for weights in candidates:
            if len(rows) >= n_portfolios:
                break
            port_returns = ret @ weights
            cvar_95 = portfolio_cvar(port_returns, 0.95)
            cvar_985 = portfolio_cvar(port_returns, 0.985)
            cvar_995 = portfolio_cvar(port_returns, 0.995)
            var_95 = portfolio_var(port_returns, 0.95)
            var_985 = portfolio_var(port_returns, 0.985)
            var_995 = portfolio_var(port_returns, 0.995)
            stress_summary = worst_stress_summary_for_weights(weights, universe, portfolio_value)
            max_v, nb_v, constraint_status = max_violation(weights)
            vol = float(np.sqrt(max(weights @ sig @ weights, 0.0)))
            expected = float(weights @ mu_v)
            mc_id = f"MC_{len(rows)+1:05d}"
            standardized_weights[mc_id] = weights
            rows.append(
                {
                    "Scenario_Methodological_Name": scenario,
                    "Portfolio_ID": mc_id,
                    "Expected_Return": expected,
                    "Expected_Return_Annualized": expected,
                    "Volatility": vol,
                    "Volatility_Annualized": vol,
                    "Volatility_Status": "ALREADY_ANNUALIZED",
                    "Sharpe": float((expected - rf_annual) / vol) if vol > 1e-12 else np.nan,
                    "VaR_95": var_95,
                    "VaR_98_5": var_985,
                    "VaR_99_5": var_995,
                    "CVaR_95": cvar_95,
                    "CVaR_98_5": cvar_985,
                    "CVaR_99_5": cvar_995,
                    "CVaR_95_Periodic": cvar_95,
                    "CVaR_98_5_Periodic": cvar_985,
                    "CVaR_99_5_Periodic": cvar_995,
                    "CVaR_95_Annualized": cvar_95 * periods_per_year,
                    "CVaR_98_5_Annualized": cvar_985 * periods_per_year,
                    "CVaR_99_5_Annualized": cvar_995 * periods_per_year,
                    "CVaR_Level": 0.995,
                    "HHI": float(np.sum(weights**2)),
                    "Distance_L1_Current": distance_l1(weights, current),
                    "Worst_Stress_Loss_TND": stress_summary["Worst_Stress_Loss_TND"],
                    "Worst_Stress_Loss_Percent": stress_summary["Worst_Stress_Loss_Percent"],
                    "Stress_Test_Status": stress_summary["Stress_Test_Status"],
                    "Worst_Stress_Status": stress_summary["Worst_Stress_Status"],
                    "Nb_Stress_Tests_Missing": stress_summary["Nb_Stress_Tests_Missing"],
                    "Robustness_Score_Adjusted": stress_summary["Robustness_Score_Adjusted"],
                    "Feasible": constraint_status == "PASSED",
                    "Nb_Constraint_Violations": nb_v,
                    "Max_Constraint_Violation": max_v,
                    "Constraint_Violation_Status": constraint_status,
                    "Decision_Eligibility": "MONTE_CARLO_EXPLORATORY_ONLY",
                    "Feasibility_Status": "ADMISSIBLE",
                    "Weights_JSON": json.dumps({asset: float(w) for asset, w in zip(mu.index, weights)}, ensure_ascii=False),
                }
            )
    if len(rows) != n_portfolios:
        raise RuntimeError(f"Monte Carlo generated {len(rows)} admissible portfolios, expected {n_portfolios}.")
    return pd.DataFrame(rows), standardized_weights
