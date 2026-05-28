"""Stress tests for notebook 02."""

from __future__ import annotations

import numpy as np
import pandas as pd


STRESS_DEFINITIONS = [
    ("Actions_-10pct", -0.10, 0.0, 0.0),
    ("Actions_-20pct", -0.20, 0.0, 0.0),
    ("Actions_-30pct", -0.30, 0.0, 0.0),
    ("Taux_+100bps", 0.0, 0.0100, 0.0),
    ("Taux_+200bps", 0.0, 0.0200, 0.0),
    ("Spread_Corporate_+100bps", 0.0, 0.0, 0.0100),
    ("Spread_Corporate_+200bps", 0.0, 0.0, 0.0200),
    ("Combine_Actions_-20pct_Taux_+100bps_Spread_+150bps", -0.20, 0.0100, 0.0150),
    ("Concentration_Shock_Top3_-15pct", 0.0, 0.0, 0.0),
]


def _duration_vector(universe: pd.DataFrame) -> tuple[np.ndarray | None, str, str]:
    for col in ["modified_duration", "Modified_Duration", "duration", "Duration"]:
        if col in universe.columns:
            values = pd.to_numeric(universe[col], errors="coerce").to_numpy(float)
            if np.isfinite(values).any():
                return values, "MODIFIED_DURATION_AVAILABLE", "Duration modifiee utilisee."
    return None, "DURATION_DATA_MISSING", "Duration indisponible : impact taux/spread non calculable sans inventer la donnee."


def stress_loss_for_weights(weights: np.ndarray, universe: pd.DataFrame, portfolio_value: float, stress_name: str) -> tuple[float, float, str, str]:
    w = np.asarray(weights, dtype=float)
    types = universe["asset_type"].astype(str).str.lower()
    equity = types.str.contains("equity|action").to_numpy()
    gov = types.str.contains("government|etat|bta|emprunt").to_numpy()
    corp = types.str.contains("corporate|obligation").to_numpy()
    durations, duration_status, duration_comment = _duration_vector(universe)
    shock = next((s for s in STRESS_DEFINITIONS if s[0] == stress_name), None)
    if shock is None:
        raise ValueError(f"Unknown stress scenario: {stress_name}")
    _, equity_shock, rate_shock, spread_shock = shock
    if stress_name == "Concentration_Shock_Top3_-15pct":
        top_idx = np.argsort(w)[-3:]
        impact = float(np.sum(w[top_idx] * -0.15))
        loss_pct = max(0.0, -impact)
        return loss_pct * portfolio_value, loss_pct, "PASSED", "Choc -15 % applique aux trois plus grandes lignes."
    equity_impact = float(np.sum(w[equity]) * equity_shock)
    if (rate_shock or spread_shock) and durations is None:
        return np.nan, np.nan, "DATA_MISSING", duration_comment
    bond_impact = 0.0
    if durations is not None:
        dur = np.nan_to_num(durations, nan=0.0)
        bond_impact += float(np.sum(w[gov] * (-dur[gov] * rate_shock)))
        bond_impact += float(np.sum(w[corp] * (-dur[corp] * rate_shock)))
        bond_impact += float(np.sum(w[corp] * (-dur[corp] * spread_shock)))
    impact = equity_impact + bond_impact
    loss_pct = max(0.0, -impact)
    return loss_pct * portfolio_value, loss_pct, duration_status, duration_comment


def run_stress_tests(
    portfolios: dict[tuple[str, str], np.ndarray],
    universe: pd.DataFrame,
    portfolio_value: float,
    technical_provisions: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scenario, model), weights in portfolios.items():
        for stress_name, *_ in STRESS_DEFINITIONS:
            loss_tnd, loss_pct, status, comment = stress_loss_for_weights(weights, universe, portfolio_value, stress_name)
            rows.append(
                {
                    "Scenario_Methodological_Name": scenario,
                    "Model": model,
                    "Stress_Name": stress_name,
                    "Loss_TND": loss_tnd,
                    "Loss_Percent": loss_pct,
                    "Loss_vs_Technical_Provisions": loss_tnd / technical_provisions if technical_provisions and np.isfinite(loss_tnd) else np.nan,
                    "Status": status,
                    "Comment": comment,
                }
            )
    return pd.DataFrame(rows)


def stress_data_availability_check(stress_tests: pd.DataFrame) -> pd.DataFrame:
    if stress_tests.empty:
        return pd.DataFrame([{"Stress_Data_Item": "Stress_Tests", "Status": "DATA_MISSING", "Comment": "No stress test output."}])
    rows = []
    for stress_name, part in stress_tests.groupby("Stress_Name"):
        statuses = set(part["Status"].astype(str))
        missing_count = int(part["Status"].astype(str).eq("DATA_MISSING").sum())
        if "DATA_MISSING" in statuses:
            status = "PASSED_WITH_WARNINGS"
            comment = "At least one required market input is missing; stress not converted to zero loss."
        else:
            status = "PASSED"
            comment = "Stress scenario computed with available data."
        rows.append({"Stress_Data_Item": stress_name, "Status": status, "Nb_Missing": missing_count, "Comment": comment})
    return pd.DataFrame(rows)


def worst_stress_summary_for_weights(weights: np.ndarray, universe: pd.DataFrame, portfolio_value: float) -> dict[str, object]:
    values = []
    missing = 0
    for stress_name, *_ in STRESS_DEFINITIONS:
        loss_tnd, loss_pct, status, _ = stress_loss_for_weights(weights, universe, portfolio_value, stress_name)
        if status == "DATA_MISSING":
            missing += 1
        if np.isfinite(loss_tnd):
            values.append((loss_tnd, loss_pct))
    if not values:
        return {
            "Worst_Stress_Loss_TND": np.nan,
            "Worst_Stress_Loss_Percent": np.nan,
            "Worst_Stress_Status": "DATA_MISSING_CRITICAL",
            "Nb_Stress_Tests_Missing": missing,
            "Stress_Test_Status": "DATA_MISSING_CRITICAL",
            "Robustness_Score_Adjusted": 0.0,
        }
    worst_tnd, worst_pct = max(values, key=lambda x: x[0])
    status = "DATA_MISSING_CRITICAL" if missing else "PASSED"
    return {
        "Worst_Stress_Loss_TND": float(worst_tnd),
        "Worst_Stress_Loss_Percent": float(worst_pct),
        "Worst_Stress_Status": status,
        "Nb_Stress_Tests_Missing": missing,
        "Stress_Test_Status": status,
        "Robustness_Score_Adjusted": 0.50 if missing else 1.0,
    }


def worst_stress_loss(weights: np.ndarray, universe: pd.DataFrame, portfolio_value: float) -> tuple[float, float]:
    summary = worst_stress_summary_for_weights(weights, universe, portfolio_value)
    return summary["Worst_Stress_Loss_TND"], summary["Worst_Stress_Loss_Percent"]
