"""Pareto filtering and multicriteria scoring for notebook 02."""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPLORATORY_MODELS = [
    "Current_Portfolio",
    "Equal_Weighted",
    "Minimum_Variance",
    "Mean_Variance_Lambda_2",
    "Mean_Variance_Lambda_5",
    "Mean_Variance_Lambda_10",
    "Mean_Variance_Lambda_20",
    "Markowitz_Mean_Variance",
    "Markowitz_Max_Return",
    "Max_Sharpe_Benchmark",
    "Min_CVaR",
    "Mean_CVaR_95",
    "Mean_CVaR_98_5",
    "Mean_CVaR_99_5",
    "Max_Return_CVaR_Constrained",
    "Robust_CVaR_Conservative_Proxy",
    "Risk_Parity",
]

DECISION_MODELS = [
    "Current_Portfolio",
    "Minimum_Variance",
    "Mean_Variance_Lambda_10",
    "Min_CVaR",
    "Mean_CVaR_95",
    "Mean_CVaR_98_5",
    "Mean_CVaR_99_5",
    "Robust_CVaR_Conservative_Proxy",
    "Risk_Parity",
]


def pareto_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    data = df.copy().reset_index(drop=True)
    criteria = [
        ("Expected_Return_Annualized", 1),
        ("Volatility_Annualized", -1),
        ("CVaR_99_5_Annualized", -1),
        ("Worst_Stress_Loss_TND", -1),
        ("HHI", -1),
        ("Distance_L1_Current", -1),
    ]
    critical = [c for c, _ in criteria] + ["Feasible", "Max_Constraint_Violation", "Target_ROE_Shortfall"]
    required = [*critical, "Regulatory_Status"]
    missing_cols = [col for col in required if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing Pareto columns: {missing_cols}")
    metric_missing = data[[c for c in critical if c != "Feasible"]].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)
    feasible_missing = data["Feasible"].isna()
    missing_critical = metric_missing | feasible_missing | data["Regulatory_Status"].isna()
    data["Pareto_Eligibility"] = np.where(~missing_critical, True, False)
    data["Quality_Flag"] = np.where(missing_critical, "CRITICAL_METRIC_MISSING", data.get("Quality_Flag", "OK"))
    data["Pareto_Status"] = "NOT_ELIGIBLE_MISSING_CRITICAL_DATA"
    eligible_idx = data.index[data["Pareto_Eligibility"]].to_numpy()
    if len(eligible_idx) == 0:
        return data
    eligible = data.loc[eligible_idx].copy()
    adjusted = []
    for col, direction in criteria:
        s = pd.to_numeric(eligible[col], errors="coerce")
        fill_value = -np.inf if direction == 1 else np.inf
        adjusted.append(s.fillna(fill_value).to_numpy(float))
    values = np.column_stack(adjusted)
    dirs = np.array([d for _, d in criteria], dtype=float)
    score_values = values * dirs
    efficient = np.ones(len(eligible), dtype=bool)
    for i in range(len(eligible)):
        if not efficient[i]:
            continue
        better_or_equal = np.all(score_values >= score_values[i] - 1e-12, axis=1)
        strictly_better = np.any(score_values > score_values[i] + 1e-12, axis=1)
        if np.any(better_or_equal & strictly_better):
            efficient[i] = False
    data.loc[eligible.index, "Pareto_Status"] = np.where(efficient, "PARETO_EFFICIENT", "PARETO_DOMINATED")
    return data


def _norm_high(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    den = s.max() - s.min()
    if not np.isfinite(den) or abs(den) <= 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / den


def _norm_low(series: pd.Series) -> pd.Series:
    return 1.0 - _norm_high(series)


def multicriteria_scoring(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    if data.empty:
        return data, pd.DataFrame()
    data = data.loc[data["Model"].isin(DECISION_MODELS)].copy()
    ranking_universe = data.copy()
    data = data.loc[
        data["Pareto_Eligibility"].eq(True)
        & data["Pareto_Status"].eq("PARETO_EFFICIENT")
        & data["Feasible"].astype(bool)
    ].copy()
    if data.empty:
        return data, pd.DataFrame()
    data["Regulatory_Bonus"] = np.where(data["Regulatory_Status"].astype(str).str.contains("FAILED|BREACH", regex=True), 0.0, 1.0)
    stress_status = data["Worst_Stress_Status"] if "Worst_Stress_Status" in data.columns else pd.Series("PASSED", index=data.index)
    quality_flag = data["Quality_Flag"] if "Quality_Flag" in data.columns else pd.Series("OK", index=data.index)
    data["Stress_Bonus"] = np.where(stress_status.astype(str).str.contains("DATA_MISSING_CRITICAL", regex=True), 0.0, 1.0)
    data["Data_Quality_Bonus"] = np.where(quality_flag.astype(str).str.contains("CRITICAL|DATA_MISSING", regex=True), 0.0, 1.0)
    eps = 1e-9
    data["Return_to_CVaR"] = data["Expected_Return_Annualized"] / np.maximum(np.abs(data["CVaR_99_5_Annualized"]), eps)
    data["CVaR_Efficiency"] = data["Expected_Return_Annualized"] - 2.0 * data["CVaR_99_5_Annualized"]

    normalized = pd.DataFrame(index=data.index)
    normalized["Expected_Return"] = _norm_high(data["Expected_Return_Annualized"])
    normalized["Volatility"] = _norm_low(data["Volatility_Annualized"])
    normalized["CVaR_99_5"] = _norm_low(data["CVaR_99_5_Annualized"])
    normalized["Worst_Stress_Loss_TND"] = _norm_low(data["Worst_Stress_Loss_TND"])
    normalized["HHI"] = _norm_low(data["HHI"])
    normalized["Distance_L1_Current"] = _norm_low(data["Distance_L1_Current"])
    normalized["Turnover_Proxy"] = _norm_low(data["Turnover_Proxy"])
    if "Target_ROE_Shortfall" not in data.columns:
        data["Target_ROE_Shortfall"] = np.maximum(0.0, pd.to_numeric(data["Target_ROE_Gap"], errors="coerce"))
    normalized["Target_ROE_Shortfall"] = _norm_low(data["Target_ROE_Shortfall"])
    normalized["Regulatory_Status"] = data["Regulatory_Bonus"]
    normalized["Return_to_CVaR"] = _norm_high(data["Return_to_CVaR"])
    normalized["Stress_Test_Status"] = data["Stress_Bonus"]
    normalized["Data_Quality"] = data["Data_Quality_Bonus"]

    weights = {
        "Score_Prudent": {
            "Expected_Return": 0.10,
            "Volatility": 0.15,
            "CVaR_99_5": 0.23,
            "Worst_Stress_Loss_TND": 0.16,
            "HHI": 0.12,
            "Distance_L1_Current": 0.08,
            "Turnover_Proxy": 0.05,
            "Target_ROE_Shortfall": 0.02,
            "Regulatory_Status": 0.04,
            "Stress_Test_Status": 0.03,
            "Data_Quality": 0.02,
        },
        "Score_Central": {
            "Expected_Return": 0.18,
            "Volatility": 0.14,
            "CVaR_99_5": 0.18,
            "Worst_Stress_Loss_TND": 0.12,
            "HHI": 0.10,
            "Distance_L1_Current": 0.10,
            "Turnover_Proxy": 0.06,
            "Target_ROE_Shortfall": 0.05,
            "Regulatory_Status": 0.04,
            "Stress_Test_Status": 0.02,
            "Data_Quality": 0.01,
        },
        "Score_Return_Oriented": {
            "Expected_Return": 0.32,
            "Return_to_CVaR": 0.18,
            "Volatility": 0.08,
            "CVaR_99_5": 0.10,
            "Worst_Stress_Loss_TND": 0.08,
            "HHI": 0.06,
            "Distance_L1_Current": 0.06,
            "Turnover_Proxy": 0.04,
            "Target_ROE_Shortfall": 0.04,
            "Regulatory_Status": 0.03,
            "Stress_Test_Status": 0.02,
            "Data_Quality": 0.03,
        },
    }
    for score_name, score_weights in weights.items():
        data[score_name] = 0.0
        for criterion, weight in score_weights.items():
            component_col = f"Component_{score_name}_{criterion}"
            data[component_col] = weight * normalized[criterion]
            data[score_name] += data[component_col]

    data["Scoring_Criteria_Documentation"] = (
        "Expected_Return and Return_to_CVaR higher_is_better; Volatility, CVaR_99_5, "
        "Worst_Stress_Loss_TND, HHI, Distance_L1_Current, Turnover_Proxy and Target_ROE_Shortfall lower_is_better."
    )
    stability = ranking_universe[
        [
            "Scenario_Methodological_Name",
            "Model",
            "Expected_Return_Annualized",
            "CVaR_99_5_Annualized",
            "Target_ROE_Shortfall",
            "Feasible",
        ]
    ].copy()
    score_lookup = data[
        [
            "Scenario_Methodological_Name",
            "Model",
            "Score_Prudent",
            "Score_Central",
            "Score_Return_Oriented",
        ]
    ].copy()
    stability = stability.merge(score_lookup, on=["Scenario_Methodological_Name", "Model"], how="left")
    score_rank_source = stability["Score_Central"].fillna(-np.inf)
    stability["Rank_By_Score"] = score_rank_source.groupby(stability["Scenario_Methodological_Name"]).rank(ascending=False, method="min").astype(int)
    stability["Rank_By_Return"] = stability.groupby("Scenario_Methodological_Name")["Expected_Return_Annualized"].rank(ascending=False, method="min").astype(int)
    stability["Rank_By_CVaR"] = stability.groupby("Scenario_Methodological_Name")["CVaR_99_5_Annualized"].rank(ascending=True, method="min").astype(int)
    stability["Rank_By_Target_Shortfall"] = stability.groupby("Scenario_Methodological_Name")["Target_ROE_Shortfall"].rank(ascending=True, method="min").astype(int)
    stability["Rank_By_Feasibility"] = stability.groupby("Scenario_Methodological_Name")["Feasible"].rank(ascending=False, method="min").astype(int)
    stability["Recommendation_Stability"] = "STABLE_IF_SELECTED_ACROSS_SCORES"
    return data, stability


def build_score_components(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    component_cols = [col for col in scored.columns if col.startswith("Component_")]
    score_names = ["Score_Prudent", "Score_Central", "Score_Return_Oriented"]
    for _, row in scored.iterrows():
        for col in component_cols:
            suffix = col.removeprefix("Component_")
            score_name = next((name for name in score_names if suffix.startswith(name + "_")), "")
            criterion = suffix.removeprefix(score_name + "_") if score_name else suffix
            rows.append(
                {
                    "Scenario_Methodological_Name": row.get("Scenario_Methodological_Name"),
                    "Model": row.get("Model"),
                    "Score_Name": score_name,
                    "Criterion": criterion,
                    "Contribution": row.get(col),
                    "Higher_Is_Better": criterion in {"Expected_Return", "Return_to_CVaR", "Regulatory_Status", "Stress_Test_Status", "Data_Quality"},
                }
            )
    return pd.DataFrame(rows)


def assign_decision_roles(scored: pd.DataFrame, reference_scenario: str = "ExAnte_Central") -> pd.DataFrame:
    data = scored.copy()
    data["Decision_Role"] = "Not_Selected"
    data.loc[data["Scenario_Methodological_Name"].eq("Historical_Raw_Comparative"), "Decision_Role"] = "Comparative_Only"
    eligible = data.loc[
        data["Scenario_Methodological_Name"].eq(reference_scenario)
        & ~data["Regulatory_Status"].astype(str).str.contains("FAILED|BREACH", regex=True)
        & data["Pareto_Eligibility"].eq(True)
        & data["Pareto_Status"].eq("PARETO_EFFICIENT")
        & data["Feasible"].astype(bool)
        & data["Model"].isin([m for m in DECISION_MODELS if m not in {"Current_Portfolio", "Markowitz_Max_Return"}])
    ].copy()
    if not eligible.empty:
        central_idx = eligible["Score_Central"].idxmax()
        data.loc[central_idx, "Decision_Role"] = "Recommended_Central"
        prudent = eligible.loc[eligible["Model"].isin(["Mean_CVaR_99_5", "Mean_CVaR_98_5", "Mean_CVaR_95"])].sort_values("CVaR_99_5")
        if not prudent.empty and central_idx not in prudent.index:
            data.loc[prudent.index[0], "Decision_Role"] = "Prudent_Alternative"
        robust = eligible.loc[eligible["Model"].eq("Robust_CVaR_Conservative_Proxy")]
        if not robust.empty and data.loc[robust.index[0], "Decision_Role"] == "Not_Selected":
            data.loc[robust.index[0], "Decision_Role"] = "Conservative_Alternative"
        rp = eligible.loc[eligible["Model"].eq("Risk_Parity")]
        if not rp.empty and data.loc[rp.index[0], "Decision_Role"] == "Not_Selected":
            data.loc[rp.index[0], "Decision_Role"] = "Diversification_Alternative"
    data.loc[data["Regulatory_Status"].astype(str).str.contains("FAILED|BREACH", regex=True), "Decision_Role"] = "Rejected_Constraint"
    return data


def build_model_deduplication_check() -> pd.DataFrame:
    rows = []
    for model in EXPLORATORY_MODELS:
        canonical = model
        kept = model in DECISION_MODELS
        reason = "Kept in decision set."
        if model == "Markowitz_Mean_Variance":
            canonical, kept, reason = "Mean_Variance_Lambda_10", False, "Alias of Lambda 10, removed from decision list to avoid duplicate."
        elif model in {"Mean_Variance_Lambda_2", "Mean_Variance_Lambda_5", "Mean_Variance_Lambda_20"}:
            canonical, kept, reason = "Mean_Variance_Lambda_10", False, "Exploratory risk aversion variant only."
        elif model == "Markowitz_Max_Return":
            canonical, kept, reason = "Extreme_Return_Comparative_Case", False, "Extreme comparative case, removed from decision list."
        elif model == "Max_Sharpe_Benchmark":
            canonical, kept, reason = "Sharpe_Descriptive_Benchmark", False, "Benchmark only, not a final decision model."
        rows.append(
            {
                "Original_Model": model,
                "Canonical_Model": canonical,
                "Kept_In_Decision": kept,
                "Reason": reason,
                "Correlation_Or_Distance_To_Canonical": "NOT_COMPUTED_REGISTER_LEVEL",
                "Status": "PASSED",
            }
        )
    return pd.DataFrame(rows)


def build_model_formulation_register() -> pd.DataFrame:
    specs = {
        "Current_Portfolio": ("Reference", "No optimization", "w = current weights"),
        "Minimum_Variance": ("Mean-Variance", "min variance", "min w.T @ Sigma @ w"),
        "Mean_Variance_Lambda_10": ("Markowitz", "mean-variance trade-off", "min lambda/2 * w.T @ Sigma @ w - mu.T @ w"),
        "Min_CVaR": ("CVaR", "min tail risk", "min CVaR_beta(w)"),
        "Mean_CVaR_95": ("Mean-CVaR 95%", "return-tail risk trade-off", "min CVaR_95_annualized(w) - theta * expected_return_annual ; theta=1.0"),
        "Mean_CVaR_98_5": ("Mean-CVaR 98.5%", "return-tail risk trade-off", "min CVaR_98_5_annualized(w) - theta * expected_return_annual ; theta=1.0"),
        "Mean_CVaR_99_5": ("Mean-CVaR 99.5%", "return-tail risk trade-off", "min CVaR_99_5_annualized(w) - theta * expected_return_annual ; theta=1.0"),
        "Robust_CVaR_Conservative_Proxy": ("Robust CVaR", "conservative proxy", "min CVaR_beta_annualized(w) - theta * expected_return_conservative ; theta=0.25"),
        "Risk_Parity": ("Diversification", "equal risk contribution", "min sum((RC_i - mean(RC))^2)"),
    }
    return pd.DataFrame(
        [
            {
                "Model_Family": family,
                "Model_Name": model,
                "Objective_Function": obj,
                "Mathematical_Formulation": form,
                "Constraints": "sum(w)=1; w>=0; bornes internes; contraintes CGA testables",
                "Solver": "CLARABEL" if model != "Risk_Parity" else "SLSQP",
                "Decision_Role": "Candidate" if model != "Current_Portfolio" else "Reference",
                "Limitations": "Results depend on ExAnte inputs and historical sample quality.",
            }
            for model, (family, obj, form) in specs.items()
        ]
    )
