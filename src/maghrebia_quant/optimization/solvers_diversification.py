"""Diversification solvers for notebook 02."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from maghrebia_quant.optimization_apt import (
    APTOptimizationConfig,
    feasible_random_starts,
    risk_parity_objective,
    solve_slsqp_model,
)
from .optimization_core import aggregate_constraint_status, compute_constraint_violations
from .metrics import risk_contribution


def solve_diversification_models(
    sigma: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    scenario: str,
    config: APTOptimizationConfig | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or APTOptimizationConfig()
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()
    sig = sigma.to_numpy(float)
    starts = feasible_random_starts(universe, current, context, config, n_starts=config.slsqp_random_starts)
    start = time.perf_counter()
    result = solve_slsqp_model("Risk_Parity", lambda w: risk_parity_objective(w, sig), universe, current, context, config, starts)
    result["runtime_seconds"] = time.perf_counter() - start
    if result.get("success"):
        violations = compute_constraint_violations(result["weights"], universe, current, context, config, result.get("turnover_limit_used"))
        constraint_status, max_violation, warning = aggregate_constraint_status(violations)
    else:
        constraint_status, max_violation, warning = "INFEASIBLE_OR_CONSTRAINT_VIOLATION", np.nan, "SLSQP failed."
    portfolios: dict[str, np.ndarray] = {}
    if result.get("success"):
        portfolios["Risk_Parity"] = np.asarray(result["weights"], dtype=float)
    audit = pd.DataFrame(
        [
            {
                "Model": "Risk_Parity",
                "Scenario": scenario,
                "Objective_Function": "min sum((RC_i - mean(RC))^2)",
                "Solver_Name": "SLSQP",
                "Solver_Status": result.get("solver_status"),
                "Success": bool(result.get("success")),
                "Objective_Value": result.get("objective_value", np.nan),
                "Constraint_Status": constraint_status,
                "Max_Constraint_Violation": max_violation,
                "Runtime_Seconds": result.get("runtime_seconds"),
                "Message": warning,
            }
        ]
    )
    if result.get("success"):
        rc = risk_contribution(result["weights"], sigma)
        rc.insert(0, "Asset", list(sigma.index))
        rc.insert(0, "Scenario", scenario)
        rc.insert(0, "Model", "Risk_Parity")
    else:
        rc = pd.DataFrame()
    not_impl = pd.DataFrame(
        [
            {
                "Model": "Maximum_Diversification",
                "Scenario": scenario,
                "Status": "NOT_COMPUTED",
                "Solver_Status": "NOT_COMPUTED",
                "Quality_Flag": "MODEL_NOT_AVAILABLE",
                "Decision_Eligibility": "EXCLUDED_MODEL_FAILED",
                "Comment": "Non retenu sans formulation stable dans le perimetre actuel.",
            }
        ]
    )
    return portfolios, audit, rc, not_impl
