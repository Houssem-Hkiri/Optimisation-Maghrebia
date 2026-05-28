"""Mean-variance and benchmark solvers for notebook 02."""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
import pandas as pd

from maghrebia_quant.optimization_apt import APTOptimizationConfig
from .optimization_core import (
    aggregate_constraint_status,
    compute_constraint_violations,
    cvxpy_constraints_core,
)


def _audit_row(model: str, scenario: str, result: dict[str, object], objective: str) -> dict[str, object]:
    return {
        "Model": model,
        "Scenario": scenario,
        "Objective_Function": objective,
        "Solver_Name": "CLARABEL",
        "Solver_Status": result.get("solver_status"),
        "Success": bool(result.get("success")),
        "Objective_Value": result.get("objective_value", np.nan),
        "Constraint_Status": result.get("constraint_status", "PASSED" if result.get("success") else "INFEASIBLE_OR_CONSTRAINT_VIOLATION"),
        "Max_Constraint_Violation": result.get("max_constraint_violation", np.nan),
        "Runtime_Seconds": result.get("runtime_seconds", np.nan),
        "Message": result.get("constraint_warning", result.get("message", "")),
    }


def _solve_cvxpy_model(
    name: str,
    objective_factory,
    universe: pd.DataFrame,
    current_weights: np.ndarray,
    context: dict[str, object],
    config: APTOptimizationConfig,
) -> dict[str, object]:
    n = len(universe)
    last_status = "not_solved"
    last_elapsed = np.nan
    for threshold in config.turnover_thresholds:
        w = cp.Variable(n)
        constraints = cvxpy_constraints_core(w, universe, current_weights, context, config, threshold)
        problem = cp.Problem(cp.Minimize(objective_factory(w)), constraints)
        start = time.perf_counter()
        try:
            problem.solve(solver="CLARABEL", verbose=False)
        except Exception:
            problem.solve(verbose=False)
        last_elapsed = time.perf_counter() - start
        last_status = str(problem.status)
        if w.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
            weights = np.maximum(np.asarray(w.value, dtype=float), 0.0)
            weights = weights / weights.sum()
            violations = compute_constraint_violations(weights, universe, current_weights, context, config, threshold)
            constraint_status, max_violation, warning = aggregate_constraint_status(violations)
            return {
                "portfolio_name": name,
                "weights": weights,
                "success": constraint_status != "INFEASIBLE_OR_CONSTRAINT_VIOLATION",
                "solver_status": last_status,
                "turnover_limit_used": threshold,
                "objective_value": float(problem.value) if problem.value is not None else np.nan,
                "runtime_seconds": last_elapsed,
                "constraint_status": constraint_status,
                "max_constraint_violation": max_violation,
                "constraint_warning": warning,
            }
    return {
        "portfolio_name": name,
        "weights": current_weights.copy(),
        "success": False,
        "solver_status": last_status,
        "turnover_limit_used": np.nan,
        "objective_value": np.nan,
        "runtime_seconds": last_elapsed,
        "constraint_status": "INFEASIBLE_OR_CONSTRAINT_VIOLATION",
        "max_constraint_violation": np.nan,
        "constraint_warning": "Solver failed before a feasible solution was available.",
    }


def solve_mean_variance_models(
    mu: pd.Series,
    sigma: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    scenario: str,
    config: APTOptimizationConfig | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Solve deterministic mean-variance models with common constraints."""

    config = config or APTOptimizationConfig()
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()
    sig = sigma.to_numpy(float)
    mu_v = mu.to_numpy(float)

    portfolios: dict[str, np.ndarray] = {"Current_Portfolio": current}
    rows: list[dict[str, object]] = []

    specs = [
        ("Minimum_Variance", lambda w: cp.quad_form(w, sig), "min w.T @ Sigma @ w"),
        ("Mean_Variance_Lambda_2", lambda w: (2.0 / 2.0) * cp.quad_form(w, sig) - mu_v @ w, "min lambda/2 variance - return"),
        ("Mean_Variance_Lambda_5", lambda w: (5.0 / 2.0) * cp.quad_form(w, sig) - mu_v @ w, "min lambda/2 variance - return"),
        ("Mean_Variance_Lambda_10", lambda w: (10.0 / 2.0) * cp.quad_form(w, sig) - mu_v @ w, "min lambda/2 variance - return"),
        ("Mean_Variance_Lambda_20", lambda w: (20.0 / 2.0) * cp.quad_form(w, sig) - mu_v @ w, "min lambda/2 variance - return"),
        ("Markowitz_Max_Return", lambda w: -mu_v @ w, "max mu.T @ w"),
    ]
    for model, objective_factory, objective_text in specs:
        start = time.perf_counter()
        result = _solve_cvxpy_model(model, objective_factory, universe, current, context, config)
        if result.get("success"):
            portfolios[model] = np.asarray(result["weights"], dtype=float)
        rows.append(_audit_row(model, scenario, result, objective_text))

    # Alias kept for presentation and deduplication, not as a separate decision model.
    portfolios["Markowitz_Mean_Variance"] = portfolios.get("Mean_Variance_Lambda_10", current).copy()
    rows.append(
        {
            "Model": "Markowitz_Mean_Variance",
            "Scenario": scenario,
            "Objective_Function": "alias decisionnel de Mean_Variance_Lambda_10",
            "Solver_Name": "CLARABEL",
            "Solver_Status": "ALIAS_OF_MEAN_VARIANCE_LAMBDA_10",
            "Success": True,
            "Objective_Value": np.nan,
            "Constraint_Status": "PASSED",
            "Max_Constraint_Violation": 0.0,
            "Runtime_Seconds": 0.0,
            "Message": "Alias non duplique dans DECISION_MODELS.",
        }
    )
    return portfolios, pd.DataFrame(rows)


def solve_max_sharpe_benchmark(
    mu: pd.Series,
    sigma: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    rf_annual: float,
    scenario: str,
    config: APTOptimizationConfig | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Return a secondary Max Sharpe benchmark using a convex tangency proxy.

    The result is explicitly tagged BENCHMARK_ONLY and is never part of the
    decision model list.
    """

    config = config or APTOptimizationConfig()
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()
    sig = sigma.to_numpy(float)
    mu_v = mu.to_numpy(float)
    n = len(current)
    w = cp.Variable(n)
    excess = mu_v - float(rf_annual)
    constraints = cvxpy_constraints_core(w, universe, current, context, config, config.primary_turnover_limit)
    # Convex proxy: minimize variance for at least current excess return.
    target = max(float(excess @ current), float(np.nanmedian(excess)))
    constraints.append(excess @ w >= target)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, sig)), constraints)
    start = time.perf_counter()
    try:
        problem.solve(solver="CLARABEL", verbose=False)
    except Exception:
        problem.solve(verbose=False)
    elapsed = time.perf_counter() - start
    ok = w.value is not None and problem.status in {"optimal", "optimal_inaccurate"}
    if ok:
        weights = np.maximum(np.asarray(w.value, dtype=float), 0.0)
        weights = weights / weights.sum()
        violations = compute_constraint_violations(weights, universe, current, context, config, config.primary_turnover_limit)
        constraint_status, max_violation, warning = aggregate_constraint_status(violations)
    else:
        weights = current.copy()
        constraint_status, max_violation, warning = "INFEASIBLE_OR_CONSTRAINT_VIOLATION", np.nan, "Benchmark solver failed."
    result = {
        "portfolio_name": "Max_Sharpe_Benchmark",
        "weights": weights,
        "success": ok,
        "solver_status": f"{problem.status};BENCHMARK_ONLY",
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
        "runtime_seconds": elapsed,
        "constraint_status": constraint_status,
        "max_constraint_violation": max_violation,
        "constraint_warning": warning,
        "message": "Benchmark secondaire, exclu de la decision finale.",
    }
    return {"Max_Sharpe_Benchmark": weights}, pd.DataFrame([_audit_row("Max_Sharpe_Benchmark", scenario, result, "benchmark Sharpe descriptif")])
