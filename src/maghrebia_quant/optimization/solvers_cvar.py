"""Rockafellar-Uryasev CVaR solvers for notebook 02."""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
import pandas as pd

from maghrebia_quant.optimization_apt import APTOptimizationConfig
from .optimization_core import aggregate_constraint_status, compute_constraint_violations, cvxpy_constraints_core


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("CVaR solver returned zero weights.")
    return w / total


def _audit(
    model: str,
    scenario: str,
    status: str,
    success: bool,
    value: float,
    elapsed: float,
    objective: str,
    message: str = "",
    constraint_status: str | None = None,
    max_constraint_violation: float | None = None,
) -> dict[str, object]:
    return {
        "Model": model,
        "Scenario": scenario,
        "Objective_Function": objective,
        "Solver_Name": "CLARABEL",
        "Solver_Status": status,
        "Success": success,
        "Objective_Value": value,
        "Constraint_Status": constraint_status or ("PASSED" if success else "INFEASIBLE_OR_CONSTRAINT_VIOLATION"),
        "Max_Constraint_Violation": np.nan if max_constraint_violation is None else max_constraint_violation,
        "Runtime_Seconds": elapsed,
        "Message": message,
    }


def _solve_cvar_problem(
    model: str,
    mu: pd.Series,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    scenario: str,
    config: APTOptimizationConfig,
    theta: float = 0.0,
    target_return: float | None = None,
    cvar_limit: float | None = None,
    maximize_return: bool = False,
    periods_per_year: int = 52,
    beta_override: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()
    ret = returns.to_numpy(float)
    mu_v = mu.to_numpy(float)
    n_obs, n_assets = ret.shape
    beta = float(config.cvar_beta if beta_override is None else beta_override)

    for threshold in config.turnover_thresholds:
        w = cp.Variable(n_assets)
        alpha = cp.Variable()
        u = cp.Variable(n_obs, nonneg=True)
        loss = -ret @ w
        cvar_periodic = alpha + (1.0 / ((1.0 - beta) * n_obs)) * cp.sum(u)
        cvar_annualized = cvar_periodic * float(periods_per_year)
        constraints = cvxpy_constraints_core(w, universe, current, context, config, threshold)
        constraints += [u >= loss - alpha, u >= 0]
        if target_return is not None and np.isfinite(target_return):
            constraints.append(mu_v @ w >= float(target_return))
        if cvar_limit is not None and np.isfinite(cvar_limit):
            constraints.append(cvar_annualized <= float(cvar_limit))
        if maximize_return:
            objective = cp.Maximize(mu_v @ w)
        else:
            objective = cp.Minimize(cvar_annualized - float(theta) * (mu_v @ w))
        problem = cp.Problem(objective, constraints)
        start = time.perf_counter()
        try:
            problem.solve(solver="CLARABEL", verbose=False)
        except Exception:
            problem.solve(verbose=False)
        elapsed = time.perf_counter() - start
        if w.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
            weights = _normalize_weights(w.value)
            violations = compute_constraint_violations(weights, universe, current, context, config, threshold)
            constraint_status, max_violation, warning = aggregate_constraint_status(violations)
            msg = f"turnover_limit={threshold}; beta={beta}; theta={theta}; periods_per_year={periods_per_year}; {warning}"
            if target_return is not None and np.isfinite(target_return):
                msg += f"; target_return={target_return:.8f}"
            return weights, _audit(
                model,
                scenario,
                str(problem.status),
                True,
                float(problem.value) if problem.value is not None else np.nan,
                elapsed,
                "Rockafellar-Uryasev: min CVaR_beta_annualized(w) - theta * expected_return_annual" if not maximize_return else "max expected_return_annual sous contrainte CVaR annualisee",
                msg,
                constraint_status,
                max_violation,
            )
    return current.copy(), _audit(
        model,
        scenario,
        str(problem.status) if "problem" in locals() else "NOT_SOLVED",
        False,
        np.nan,
        elapsed if "elapsed" in locals() else np.nan,
        "Rockafellar-Uryasev CVaR",
        "No feasible CVaR solution.",
    )


def choose_mean_cvar_target(mu: pd.Series, current_weights: np.ndarray, target_roe: float | None) -> tuple[float, str]:
    current_ret = float(mu.to_numpy(float) @ current_weights)
    median_ret = float(mu.median())
    candidates = [current_ret, median_ret]
    if target_roe is not None and np.isfinite(target_roe):
        candidates.append(float(target_roe))
    return max(candidates), "CURRENT_MEDIAN_TARGET_ROE_MAX"


def solve_cvar_models(
    mu: pd.Series,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    scenario: str,
    target_roe: float | None,
    config: APTOptimizationConfig | None = None,
    periods_per_year: int = 52,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Solve Min-CVaR, Mean-CVaR, Max-Return-CVaR and Robust-CVaR models."""

    config = config or APTOptimizationConfig()
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()
    portfolios: dict[str, np.ndarray] = {}
    audit_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []

    weights, audit = _solve_cvar_problem("Min_CVaR", mu, returns, universe, context, scenario, config, theta=0.0, periods_per_year=periods_per_year)
    portfolios["Min_CVaR"] = weights
    audit_rows.append(audit)

    initial_target, rule = choose_mean_cvar_target(mu, current, target_roe)
    target_candidates = [
        initial_target,
        max(float(mu.to_numpy(float) @ current), float(mu.median())),
        float(mu.to_numpy(float) @ current),
        float(mu.quantile(0.75)),
        float(mu.median()),
        np.nan,
    ]
    theta = 1.0
    for model_name, beta_level in [("Mean_CVaR_95", 0.95), ("Mean_CVaR_98_5", 0.985), ("Mean_CVaR_99_5", 0.995)]:
        target_status = "TARGET_RETURN_USED"
        mean_weights = current.copy()
        mean_audit = None
        used_target = np.nan
        for i, target in enumerate(target_candidates):
            target_arg = None if not np.isfinite(target) else target
            weights, audit = _solve_cvar_problem(
                model_name,
                mu,
                returns,
                universe,
                context,
                scenario,
                config,
                theta=theta,
                target_return=target_arg,
                periods_per_year=periods_per_year,
                beta_override=beta_level,
            )
            if audit["Success"]:
                mean_weights = weights
                mean_audit = audit
                used_target = target_arg if target_arg is not None else np.nan
                if i > 0:
                    target_status = "TARGET_RETURN_INFEASIBLE_RELAXED"
                break
        if mean_audit is None:
            mean_weights, mean_audit = _solve_cvar_problem(
                model_name,
                mu,
                returns,
                universe,
                context,
                scenario,
                config,
                theta=theta,
                target_return=None,
                periods_per_year=periods_per_year,
                beta_override=beta_level,
            )
            target_status = "TARGET_RETURN_INFEASIBLE_RELAXED"
        portfolios[model_name] = mean_weights
        mean_audit["Message"] = f"{mean_audit.get('Message','')}; target_rule={rule}; target_status={target_status}"
        audit_rows.append(mean_audit)
        diag_rows.append(
            {
                "Scenario": scenario,
                "Model": model_name,
                "Initial_Target_Return": initial_target,
                "Target_Return_Used": used_target,
                "Target_Return_Status": target_status,
                "Mean_CVaR_Theta": theta,
                "Objective": "minimize(CVaR_beta_annualized(w) - theta * expected_return_annual)",
                "Beta_Config": beta_level,
                "CVaR_Level": beta_level,
                "Periods_Per_Year": periods_per_year,
            }
        )

    # Use Min-CVaR CVaR as a conservative cap relaxed by 10% for return maximisation.
    losses = -(returns.to_numpy(float) @ portfolios["Min_CVaR"])
    min_cvar = float(losses[losses >= np.quantile(losses, float(config.cvar_beta))].mean()) * float(periods_per_year)
    weights, audit = _solve_cvar_problem(
        "Max_Return_CVaR_Constrained",
        mu,
        returns,
        universe,
        context,
        scenario,
        config,
        cvar_limit=1.10 * min_cvar,
        maximize_return=True,
        periods_per_year=periods_per_year,
    )
    portfolios["Max_Return_CVaR_Constrained"] = weights
    audit_rows.append(audit)

    robust_mu = mu.copy() - 0.25 * mu.std()
    weights, audit = _solve_cvar_problem("Robust_CVaR_Conservative_Proxy", robust_mu, returns, universe, context, scenario, config, theta=0.25, periods_per_year=periods_per_year)
    portfolios["Robust_CVaR_Conservative_Proxy"] = weights
    audit_rows.append(audit)

    return portfolios, pd.DataFrame(audit_rows), pd.DataFrame(diag_rows)
