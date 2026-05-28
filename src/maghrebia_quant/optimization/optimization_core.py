"""Core optimisation helpers for notebook 02.

The functions in this module expose the clean ExAnte/Ledoit-Wolf vocabulary
used by notebook 02. Legacy loaders remain available through the historical
module for compatibility with existing Notebook 01 exports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp

from maghrebia_quant.optimization_apt import (
    APTOptimizationConfig,
    build_context,
    build_regulatory_constraints_map,
    build_universe,
    class_masks,
    load_apt_mu_scenarios as load_exante_return_scenarios,
    load_apt_optimization_inputs as load_notebook01_optimization_inputs,
    solve_efficient_frontier,
    upper_bounds_vector,
)


def infer_or_validate_frequency(returns_index: pd.Index, fallback: str = "weekly") -> tuple[int, str, str]:
    """Infer return frequency and annualisation factor from a date index."""

    dates = pd.to_datetime(pd.Index(returns_index), errors="coerce").dropna().sort_values()
    if len(dates) < 3:
        if fallback == "daily":
            return 252, "FREQUENCY_FALLBACK_DAILY", "Frequency not detectable; daily fallback used."
        return 52, "FREQUENCY_FALLBACK_WEEKLY", "Frequency not detectable; weekly fallback used."
    median_gap = float(pd.Series(dates).diff().dt.days.dropna().median())
    if 0.5 <= median_gap <= 3.5:
        return 252, "DAILY_FREQUENCY_DETECTED", f"Median date gap is {median_gap:.1f} day(s)."
    if 4.0 <= median_gap <= 10.0:
        return 52, "WEEKLY_FREQUENCY_DETECTED", f"Median date gap is {median_gap:.1f} day(s)."
    if fallback == "daily":
        return 252, "FREQUENCY_NOT_DETECTED_DAILY_FALLBACK", f"Median date gap is {median_gap:.1f}; daily fallback used."
    return 52, "FREQUENCY_NOT_DETECTED_WEEKLY_FALLBACK", f"Median date gap is {median_gap:.1f}; weekly fallback used by project convention."


def fixed_state_value(context: dict[str, object]) -> tuple[float, str]:
    """Return fixed-pocket state exposure.

    Notebook 02 treats all state securities as part of the optimisable pocket.
    A non-zero fixed state pocket would make the 20% CGA minimum ambiguous, so
    it is deliberately not netted from the optimisable constraint.
    """

    return 0.0, "DATA_AVAILABLE"


def state_min_weight_on_optimisable(context: dict[str, object]) -> tuple[float, str, str]:
    pt = float(context.get("technical_provisions", np.nan))
    opt_value = float(context.get("optimisable_value", np.nan))
    fixed_state, data_status = fixed_state_value(context)
    if not np.isfinite(pt) or not np.isfinite(opt_value) or opt_value <= 0 or data_status != "DATA_AVAILABLE":
        return 0.0, "NOT_TESTABLE_DATA_MISSING", "Fixed state exposure or denominator missing; state minimum is not forced."
    required_state_value = max(0.0, 0.20 * pt - fixed_state)
    return float(required_state_value / opt_value), "DATA_AVAILABLE", "20% state rule applied to the optimisable pocket; state securities are treated as optimisable."


def issuer_groups(universe: pd.DataFrame) -> tuple[dict[str, np.ndarray], str]:
    if "issuer" not in universe.columns:
        return {}, "NOT_TESTABLE_DATA_MISSING"
    issuers = universe["issuer"].astype(str).replace({"nan": ""})
    if issuers.str.strip().eq("").all():
        return {}, "NOT_TESTABLE_DATA_MISSING"
    return {issuer: np.flatnonzero(issuers.eq(issuer).to_numpy()) for issuer in sorted(issuers.unique())}, "PASSED"


def cvxpy_constraints_core(
    w: cp.Variable,
    universe: pd.DataFrame,
    current_weights: np.ndarray,
    context: dict[str, object],
    config: APTOptimizationConfig,
    turnover_limit: float | None,
) -> list:
    """Common convex constraints with issuer and fixed-pocket CGA adjustment."""

    masks = class_masks(universe)
    upper = upper_bounds_vector(universe, context, config)
    state_min, state_status, _ = state_min_weight_on_optimisable(context)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= upper,
        cp.sum(cp.multiply(masks["listed_equity"].astype(float), w)) <= config.max_equity_weight,
        cp.sum(cp.multiply(masks["corporate_bond"].astype(float), w)) <= config.max_corporate_weight,
    ]
    if state_status == "DATA_AVAILABLE":
        constraints.append(cp.sum(cp.multiply(masks["government_bond"].astype(float), w)) >= state_min)
    groups, issuer_status = issuer_groups(universe)
    if issuer_status == "PASSED":
        for idx in groups.values():
            constraints.append(cp.sum(w[idx]) <= config.max_weight_per_issuer)
    if turnover_limit is not None:
        constraints.append(cp.norm1(w - current_weights) <= turnover_limit)
    return constraints


def compute_constraint_violations(
    weights: np.ndarray,
    universe: pd.DataFrame,
    current_weights: np.ndarray,
    context: dict[str, object],
    config: APTOptimizationConfig,
    turnover_limit: float | None,
) -> pd.DataFrame:
    """Recompute all material optimisation constraint violations."""

    w = np.asarray(weights, dtype=float)
    current = np.asarray(current_weights, dtype=float)
    upper = upper_bounds_vector(universe, context, config)
    masks = class_masks(universe)
    rows: list[dict[str, object]] = []

    def add(name: str, value: float, limit: float, violation: float, status: str, comment: str) -> None:
        rows.append(
            {
                "Constraint_Name": name,
                "Current_Value": value,
                "Limit": limit,
                "Violation": max(float(violation), 0.0) if np.isfinite(violation) else np.nan,
                "Status": status,
                "Comment": comment,
            }
        )

    add("Budget_Full_Investment", float(w.sum()), 1.0, abs(float(w.sum()) - 1.0), "TESTABLE", "sum(w)=1")
    add("No_Short_Selling", float(w.min()), 0.0, max(0.0, -float(w.min())), "TESTABLE", "w_i >= 0")
    add("Asset_Upper_Bound", float(np.max(w - upper)), 0.0, float(np.max(w - upper)), "TESTABLE", "w_i <= upper_bound_i")
    state_min, state_status, state_comment = state_min_weight_on_optimisable(context)
    state_weight = float(np.sum(w[masks["government_bond"]]))
    add("State_Min_20pct_Adjusted", state_weight, state_min, state_min - state_weight, state_status, state_comment)
    equity_weight = float(np.sum(w[masks["listed_equity"]]))
    add("Equity_Class_Max", equity_weight, config.max_equity_weight, equity_weight - config.max_equity_weight, "TESTABLE", "Equity class upper bound.")
    corp_weight = float(np.sum(w[masks["corporate_bond"]]))
    add("Corporate_Class_Max", corp_weight, config.max_corporate_weight, corp_weight - config.max_corporate_weight, "TESTABLE", "Corporate bond class upper bound.")
    groups, issuer_status = issuer_groups(universe)
    if issuer_status == "PASSED":
        max_issuer = max(float(np.sum(w[idx])) for idx in groups.values()) if groups else 0.0
        add("Issuer_Max_Weight", max_issuer, config.max_weight_per_issuer, max_issuer - config.max_weight_per_issuer, issuer_status, "Issuer concentration upper bound.")
    else:
        add("Issuer_Max_Weight", np.nan, config.max_weight_per_issuer, np.nan, issuer_status, "Issuer column missing; not assumed.")
    if turnover_limit is not None:
        turnover = float(np.abs(w - current).sum())
        add("Turnover_Limit", turnover, turnover_limit, turnover - turnover_limit, "TESTABLE", "L1 turnover governance limit.")
    out = pd.DataFrame(rows)
    return out


def aggregate_constraint_status(violations: pd.DataFrame, tolerance: float = 1e-6) -> tuple[str, float, str]:
    numeric_violations = pd.to_numeric(violations["Violation"], errors="coerce") if not violations.empty else pd.Series(dtype=float)
    testable = ~violations["Status"].astype(str).str.contains("NOT_TESTABLE_DATA_MISSING", na=False) if not violations.empty else pd.Series(dtype=bool)
    max_violation = float(numeric_violations[testable].max()) if not violations.empty and numeric_violations[testable].notna().any() else np.nan
    if max_violation > tolerance:
        return "INFEASIBLE_OR_CONSTRAINT_VIOLATION", max_violation, "Constraint violation above tolerance."
    if violations["Status"].astype(str).str.contains("NOT_TESTABLE_DATA_MISSING").any():
        return "NOT_TESTED_DATA_MISSING", max_violation, "Some constraints are not testable with available data."
    return "PASSED", max_violation, "All testable constraints passed."
