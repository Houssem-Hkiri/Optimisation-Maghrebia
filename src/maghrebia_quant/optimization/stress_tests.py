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

NARRATIVE_STRESS_DEFINITIONS = [
    {
        "Stress_Name": "COVID_19",
        "Stress_Label": "COVID-19",
        "Description": (
            "Choc systémique combinant baisse des actions, tension de liquidité "
            "et élargissement des spreads de crédit."
        ),
        "Equity_Shock": -0.22,
        "Rate_Shock": 0.0050,
        "Corporate_Spread_Shock": 0.0150,
        "Liquidity_Haircut": -0.02,
    },
    {
        "Stress_Name": "Guerre_Geopolitique_Globale",
        "Stress_Label": "Guerre géopolitique globale",
        "Description": (
            "Choc géopolitique sévère transmis par l'aversion au risque, "
            "la baisse des marchés actions, la hausse des taux et l'élargissement des spreads."
        ),
        "Equity_Shock": -0.18,
        "Rate_Shock": 0.0100,
        "Corporate_Spread_Shock": 0.0200,
        "Liquidity_Haircut": -0.02,
    },
    {
        "Stress_Name": "Crise_Inflationniste_Choc_Taux",
        "Stress_Label": "Crise inflationniste / choc de taux",
        "Description": (
            "Choc de marché caractérisé par une forte hausse des taux, "
            "une pression sur les obligations et une correction des actifs risqués."
        ),
        "Equity_Shock": -0.10,
        "Rate_Shock": 0.0200,
        "Corporate_Spread_Shock": 0.0150,
        "Liquidity_Haircut": 0.0,
    },
    {
        "Stress_Name": "Crise_Credit",
        "Stress_Label": "Crise de crédit",
        "Description": "Stress centré sur le risque de crédit et la dégradation des spreads corporate.",
        "Equity_Shock": -0.08,
        "Rate_Shock": 0.0075,
        "Corporate_Spread_Shock": 0.0250,
        "Liquidity_Haircut": 0.0,
    },
]

NARRATIVE_STRESS_METHODOLOGY_NOTE = (
    "Les scénarios COVID-19, guerre géopolitique globale, crise inflationniste "
    "et crise de crédit sont des stress tests narratifs. Ils ne constituent pas "
    "des prévisions, mais des hypothèses de choc destinées à mesurer la robustesse "
    "relative des portefeuilles. Le backtesting des 10 pires séances 2025 mesure "
    "le comportement des allocations candidates sur les séances les plus défavorables "
    "observées dans l'historique disponible. Ces résultats doivent être interprétés "
    "comme des indicateurs de robustesse, et non comme une garantie de performance future."
)


def _duration_vector(universe: pd.DataFrame) -> tuple[np.ndarray | None, str, str]:
    for col in ["modified_duration", "Modified_Duration", "duration", "Duration"]:
        if col in universe.columns:
            values = pd.to_numeric(universe[col], errors="coerce").to_numpy(float)
            if np.isfinite(values).any():
                return values, "MODIFIED_DURATION_AVAILABLE", "Duration modifiee utilisee."
    return None, "DURATION_DATA_MISSING", "Duration indisponible : impact taux/spread non calculable sans inventer la donnee."


def _asset_masks(universe: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    types = universe["asset_type"].astype(str).str.lower()
    equity = types.str.contains("equity|action")
    gov = types.str.contains("government|etat|bta|emprunt")
    corp = types.str.contains("corporate|obligation")
    return equity, gov, corp


def _liquidity_mask(universe: pd.DataFrame) -> tuple[pd.Series | None, str]:
    candidate_cols = [
        "liquidity_status",
        "Liquidity_Status",
        "liquidity_bucket",
        "Liquidity_Bucket",
        "liquidity",
        "Liquidity",
        "liquidite",
        "Liquidite",
        "is_illiquid",
        "Illiquid",
    ]
    for col in candidate_cols:
        if col not in universe.columns:
            continue
        values = universe[col]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).astype(bool), f"Haircut applique selon {col}."
        text = values.astype(str).str.lower()
        mask = text.str.contains("peu|faible|low|illiquid|non liquide|limited|reduite", regex=True, na=False)
        if mask.any():
            return mask, f"Haircut applique aux actifs peu liquides identifies par {col}."
    return None, "Information de liquidite absente : aucun haircut de liquidite n'est invente."


def _status_from_parts(calculated_parts: list[str], missing_parts: list[str]) -> str:
    if not calculated_parts and missing_parts:
        return "DATA_MISSING_CRITICAL"
    if missing_parts:
        return "PARTIAL_DATA"
    return "CALCULATED"


def _bond_shock_impact(
    weights: np.ndarray,
    universe: pd.DataFrame,
    rate_shock: float,
    spread_shock: float,
) -> tuple[float, str, str]:
    if not rate_shock and not spread_shock:
        return 0.0, "CALCULATED", "Duration non requise pour ce scenario."

    equity, gov, corp = _asset_masks(universe)
    del equity
    gov_mask = gov.to_numpy()
    corp_mask = corp.to_numpy()
    required_mask = ((gov_mask | corp_mask) & bool(rate_shock)) | (corp_mask & bool(spread_shock))
    exposed_mask = required_mask & (np.asarray(weights, dtype=float) > 1e-12)
    if not exposed_mask.any():
        return 0.0, "CALCULATED", "Aucune exposition obligataire sensible au choc taux/spread."

    durations, _duration_status, duration_comment = _duration_vector(universe)
    if durations is None:
        return np.nan, "DATA_MISSING_CRITICAL", duration_comment

    finite_duration = np.isfinite(durations)
    available_mask = exposed_mask & finite_duration
    missing_mask = exposed_mask & ~finite_duration
    if not available_mask.any():
        return np.nan, "DATA_MISSING_CRITICAL", duration_comment

    bond_impact = 0.0
    if rate_shock:
        rate_mask = available_mask & (gov_mask | corp_mask)
        bond_impact += float(np.sum(weights[rate_mask] * (-durations[rate_mask] * rate_shock)))
    if spread_shock:
        spread_mask = available_mask & corp_mask
        bond_impact += float(np.sum(weights[spread_mask] * (-durations[spread_mask] * spread_shock)))

    if missing_mask.any():
        return (
            bond_impact,
            "PARTIAL_DATA",
            "Duration manquante sur une partie des actifs sensibles ; calcul partiel sans remplacement par zero.",
        )
    return bond_impact, "CALCULATED", "Duration disponible pour les actifs sensibles au choc taux/spread."


def narrative_stress_scenarios_table() -> pd.DataFrame:
    rows = []
    for definition in NARRATIVE_STRESS_DEFINITIONS:
        rows.append(
            {
                "Stress_Name": definition["Stress_Name"],
                "Stress_Label": definition["Stress_Label"],
                "Description": definition["Description"],
                "Equity_Shock": definition["Equity_Shock"],
                "Rate_Shock_Bps": definition["Rate_Shock"] * 10_000,
                "Corporate_Spread_Shock_Bps": definition["Corporate_Spread_Shock"] * 10_000,
                "Liquidity_Haircut": definition["Liquidity_Haircut"],
                "Methodological_Comment": NARRATIVE_STRESS_METHODOLOGY_NOTE,
            }
        )
    return pd.DataFrame(rows)


def narrative_stress_loss_for_weights(
    weights: np.ndarray,
    universe: pd.DataFrame,
    portfolio_value: float,
    definition: dict[str, object],
) -> dict[str, object]:
    w = np.asarray(weights, dtype=float)
    equity, gov, corp = _asset_masks(universe)
    durations, _duration_status, duration_comment = _duration_vector(universe)
    liquidity, liquidity_comment = _liquidity_mask(universe)
    equity_shock = float(definition["Equity_Shock"])
    rate_shock = float(definition["Rate_Shock"])
    spread_shock = float(definition["Corporate_Spread_Shock"])
    liquidity_haircut = float(definition["Liquidity_Haircut"])

    calculated_parts: list[str] = []
    missing_parts: list[str] = []
    equity_impact = float(np.sum(w[equity.to_numpy()] * equity_shock))
    calculated_parts.append("listed_equity")

    gov_rate_impact = np.nan
    corp_rate_impact = np.nan
    corp_spread_impact = np.nan
    if rate_shock or spread_shock:
        gov_mask = gov.to_numpy()
        corp_mask = corp.to_numpy()
        required_mask = ((gov_mask | corp_mask) & bool(rate_shock)) | (corp_mask & bool(spread_shock))
        exposed_mask = required_mask & (w > 1e-12)
        if not exposed_mask.any():
            calculated_parts.append("rate_spread_no_exposure")
        elif durations is None:
            missing_parts.append("duration")
        else:
            finite_duration = np.isfinite(durations)
            available_mask = exposed_mask & finite_duration
            missing_duration_mask = exposed_mask & ~finite_duration
            if not available_mask.any():
                missing_parts.append("duration")
            else:
                if rate_shock:
                    gov_rate_mask = available_mask & gov_mask
                    corp_rate_mask = available_mask & corp_mask
                    gov_rate_impact = float(np.sum(w[gov_rate_mask] * (-durations[gov_rate_mask] * rate_shock)))
                    corp_rate_impact = float(np.sum(w[corp_rate_mask] * (-durations[corp_rate_mask] * rate_shock)))
                if spread_shock:
                    corp_spread_mask = available_mask & corp_mask
                    corp_spread_impact = float(np.sum(w[corp_spread_mask] * (-durations[corp_spread_mask] * spread_shock)))
                if missing_duration_mask.any():
                    missing_parts.append("duration")
            calculated_parts.extend(["government_rate", "corporate_rate_spread"])

    liquidity_impact = np.nan
    if liquidity_haircut:
        if liquidity is None:
            missing_parts.append("liquidity")
        else:
            liquidity_impact = float(np.sum(w[liquidity.to_numpy()] * liquidity_haircut))
            calculated_parts.append("liquidity")

    impacts = [equity_impact, gov_rate_impact, corp_rate_impact, corp_spread_impact, liquidity_impact]
    finite_impacts = [x for x in impacts if np.isfinite(x)]
    total_impact = float(np.sum(finite_impacts)) if finite_impacts else np.nan
    loss_pct = max(0.0, -total_impact) if np.isfinite(total_impact) else np.nan
    status = _status_from_parts(calculated_parts, missing_parts)
    missing_comment = []
    if "duration" in missing_parts:
        missing_comment.append(duration_comment)
    if "liquidity" in missing_parts:
        missing_comment.append(liquidity_comment)
    if not missing_comment:
        missing_comment.append("Tous les chocs requis sont calcules avec les donnees disponibles.")
    elif np.isfinite(loss_pct):
        missing_comment.append("Calcul partiel realise sans remplacer les donnees manquantes par zero.")
    return {
        "Stress_Name": definition["Stress_Name"],
        "Stress_Label": definition["Stress_Label"],
        "Stress_Category": "NARRATIVE_GLOBAL_CRISIS",
        "Scenario_Description": definition["Description"],
        "Equity_Shock": equity_shock,
        "Rate_Shock_Bps": rate_shock * 10_000,
        "Corporate_Spread_Shock_Bps": spread_shock * 10_000,
        "Liquidity_Haircut": liquidity_haircut,
        "Loss_TND": loss_pct * portfolio_value if np.isfinite(loss_pct) else np.nan,
        "Loss_Percent": loss_pct,
        "Equity_Contribution_Percent": max(0.0, -equity_impact) if np.isfinite(equity_impact) else np.nan,
        "Government_Rate_Contribution_Percent": max(0.0, -gov_rate_impact) if np.isfinite(gov_rate_impact) else np.nan,
        "Corporate_Rate_Contribution_Percent": max(0.0, -corp_rate_impact) if np.isfinite(corp_rate_impact) else np.nan,
        "Corporate_Spread_Contribution_Percent": max(0.0, -corp_spread_impact) if np.isfinite(corp_spread_impact) else np.nan,
        "Liquidity_Contribution_Percent": max(0.0, -liquidity_impact) if np.isfinite(liquidity_impact) else np.nan,
        "Calculation_Status": status,
        "Status": status,
        "Comment": " ".join(missing_comment),
        "Methodological_Comment": NARRATIVE_STRESS_METHODOLOGY_NOTE,
    }


def stress_loss_for_weights(weights: np.ndarray, universe: pd.DataFrame, portfolio_value: float, stress_name: str) -> tuple[float, float, str, str]:
    w = np.asarray(weights, dtype=float)
    types = universe["asset_type"].astype(str).str.lower()
    equity = types.str.contains("equity|action").to_numpy()
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
    bond_impact, bond_status, bond_comment = _bond_shock_impact(w, universe, rate_shock, spread_shock)
    if not equity_shock and (rate_shock or spread_shock) and not np.isfinite(bond_impact):
        return np.nan, np.nan, bond_status, bond_comment
    if not np.isfinite(bond_impact):
        bond_impact = 0.0
    impact = equity_impact + bond_impact
    loss_pct = max(0.0, -impact)
    if rate_shock or spread_shock:
        status = bond_status
        comment = bond_comment
    else:
        status = "CALCULATED"
        comment = "Choc actions calcule ; duration et spread non requis."
    return loss_pct * portfolio_value, loss_pct, status, comment


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
        for definition in NARRATIVE_STRESS_DEFINITIONS:
            row = narrative_stress_loss_for_weights(weights, universe, portfolio_value, definition)
            row.update(
                {
                    "Scenario_Methodological_Name": scenario,
                    "Model": model,
                    "Loss_vs_Technical_Provisions": (
                        row["Loss_TND"] / technical_provisions
                        if technical_provisions and np.isfinite(row["Loss_TND"])
                        else np.nan
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def stress_data_availability_check(stress_tests: pd.DataFrame) -> pd.DataFrame:
    if stress_tests.empty:
        return pd.DataFrame([{"Stress_Data_Item": "Stress_Tests", "Status": "DATA_MISSING", "Comment": "No stress test output."}])
    rows = []
    standard_shocks = {name: (equity, rate, spread, 0.0) for name, equity, rate, spread in STRESS_DEFINITIONS}
    for stress_name, part in stress_tests.groupby("Stress_Name"):
        equity_shock, rate_shock, spread_shock, liquidity_haircut = standard_shocks.get(stress_name, (0.0, 0.0, 0.0, 0.0))
        if "Equity_Shock" in part.columns and part["Equity_Shock"].notna().any():
            equity_shock = float(pd.to_numeric(part["Equity_Shock"], errors="coerce").dropna().iloc[0])
        if "Rate_Shock_Bps" in part.columns and part["Rate_Shock_Bps"].notna().any():
            rate_shock = float(pd.to_numeric(part["Rate_Shock_Bps"], errors="coerce").dropna().iloc[0]) / 10_000
        if "Corporate_Spread_Shock_Bps" in part.columns and part["Corporate_Spread_Shock_Bps"].notna().any():
            spread_shock = float(pd.to_numeric(part["Corporate_Spread_Shock_Bps"], errors="coerce").dropna().iloc[0]) / 10_000
        if "Liquidity_Haircut" in part.columns and part["Liquidity_Haircut"].notna().any():
            liquidity_haircut = float(pd.to_numeric(part["Liquidity_Haircut"], errors="coerce").dropna().iloc[0])
        required_components = []
        if equity_shock:
            required_components.append("actions")
        if rate_shock:
            required_components.append("duration taux")
        if spread_shock:
            required_components.append("duration spread corporate")
        if liquidity_haircut:
            required_components.append("liquidite")
        statuses = set(part["Status"].astype(str))
        status_text = part["Status"].astype(str)
        pure_equity = bool(equity_shock) and not rate_shock and not spread_shock and not liquidity_haircut
        if pure_equity:
            missing_count = 0
            partial_count = 0
        else:
            missing_count = int(status_text.isin(["DATA_MISSING", "DATA_MISSING_CRITICAL"]).sum())
            partial_count = int(status_text.eq("PARTIAL_DATA").sum())
        if missing_count:
            status = "PASSED_WITH_WARNINGS"
            comment = "Donnee requise manquante pour ce scenario ; aucune substitution par zero."
        elif partial_count:
            status = "PASSED_WITH_WARNINGS"
            comment = "Stress calcule sur les composantes disponibles ; donnees manquantes signalees."
        else:
            status = "PASSED"
            comment = "Stress scenario computed with available data."
        if required_components:
            comment = f"Donnees requises: {', '.join(required_components)}. {comment}"
        rows.append({"Stress_Data_Item": stress_name, "Status": status, "Nb_Missing": missing_count, "Nb_Partial": partial_count, "Comment": comment})
    return pd.DataFrame(rows)


def _portfolio_return_on_available_assets(weights: np.ndarray, returns_row: pd.Series, critical_threshold: float = 0.80) -> tuple[float, str, float]:
    w = pd.Series(np.asarray(weights, dtype=float), index=returns_row.index)
    valid = returns_row.notna()
    available_weight = float(w.loc[valid].sum())
    if available_weight <= 1e-12:
        return np.nan, "DATA_MISSING_CRITICAL", available_weight
    value = float((w.loc[valid] * returns_row.loc[valid].astype(float)).sum() / available_weight)
    if available_weight < critical_threshold:
        return value, "DATA_MISSING_CRITICAL", available_weight
    if available_weight < 1.0 - 1e-8:
        return value, "PARTIAL_DATA", available_weight
    return value, "CALCULATED", available_weight


def build_worst_10_sessions_2025_backtest(
    portfolios: dict[tuple[str, str], np.ndarray],
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    recommended_model: str,
    scenario: str = "ExAnte_Central",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_ids = universe["asset_id"].astype(str).tolist()
    returns_2025 = returns.copy()
    returns_2025.index = pd.to_datetime(returns_2025.index, errors="coerce")
    returns_2025 = returns_2025.loc[returns_2025.index.year == 2025, [c for c in asset_ids if c in returns_2025.columns]]
    if returns_2025.empty:
        empty_backtest = pd.DataFrame(
            [
                {
                    "Date": pd.NaT,
                    "Stress_Session_Rank": np.nan,
                    "Current_Portfolio_Return": np.nan,
                    "Recommended_Portfolio_Return": np.nan,
                    "Avoided_or_Additional_Loss_vs_Current": np.nan,
                    "Best_Portfolio_On_Session": "DATA_MISSING_CRITICAL",
                    "Worst_Portfolio_On_Session": "DATA_MISSING_CRITICAL",
                    "Data_Status": "DATA_MISSING_CRITICAL",
                    "Comment": "Matrice de rendements 2025 indisponible.",
                }
            ]
        )
        return empty_backtest, pd.DataFrame()

    selected_models = [
        "Current_Portfolio",
        recommended_model,
        "Minimum_Variance",
        "Markowitz_Mean_Variance",
        "Mean_Variance_Lambda_10",
        "Mean_CVaR_95",
        "Mean_CVaR_98_5",
        "Mean_CVaR_99_5",
        "Risk_Parity",
        "MonteCarlo_Best",
    ]
    selected_models = list(dict.fromkeys([m for m in selected_models if m and (scenario, m) in portfolios]))
    aligned_returns = returns_2025.reindex(columns=asset_ids)
    model_returns: dict[str, pd.Series] = {}
    model_statuses: dict[str, pd.Series] = {}
    for model in selected_models:
        weights = portfolios[(scenario, model)]
        values = []
        statuses = []
        for _, row in aligned_returns.iterrows():
            value, status, _available_weight = _portfolio_return_on_available_assets(weights, row)
            values.append(value)
            statuses.append(status)
        model_returns[model] = pd.Series(values, index=aligned_returns.index, dtype=float)
        model_statuses[model] = pd.Series(statuses, index=aligned_returns.index, dtype=object)

    current = model_returns.get("Current_Portfolio", pd.Series(dtype=float)).dropna()
    if current.empty:
        return pd.DataFrame(
            [
                {
                    "Date": pd.NaT,
                    "Stress_Session_Rank": np.nan,
                    "Current_Portfolio_Return": np.nan,
                    "Recommended_Portfolio_Return": np.nan,
                    "Avoided_or_Additional_Loss_vs_Current": np.nan,
                    "Best_Portfolio_On_Session": "DATA_MISSING_CRITICAL",
                    "Worst_Portfolio_On_Session": "DATA_MISSING_CRITICAL",
                    "Data_Status": "DATA_MISSING_CRITICAL",
                    "Comment": "Rendement du portefeuille actuel indisponible pour identifier les 10 pires seances.",
                }
            ]
        ), pd.DataFrame()

    worst_dates = current.sort_values().head(10).index
    rows: list[dict[str, object]] = []
    for rank, date in enumerate(worst_dates, start=1):
        per_model = {model: model_returns[model].loc[date] for model in selected_models}
        valid_model_returns = {model: value for model, value in per_model.items() if np.isfinite(value)}
        statuses = [model_statuses[model].loc[date] for model in selected_models]
        if any(status == "DATA_MISSING_CRITICAL" for status in statuses):
            data_status = "DATA_MISSING_CRITICAL"
        elif any(status == "PARTIAL_DATA" for status in statuses):
            data_status = "PARTIAL_DATA"
        else:
            data_status = "CALCULATED"
        current_return = per_model.get("Current_Portfolio", np.nan)
        recommended_return = per_model.get(recommended_model, np.nan)
        row = {
            "Date": date.date(),
            "Stress_Session_Rank": rank,
            "Current_Portfolio_Return": current_return,
            "Recommended_Portfolio_Return": recommended_return,
            "Avoided_or_Additional_Loss_vs_Current": recommended_return - current_return if np.isfinite(recommended_return) and np.isfinite(current_return) else np.nan,
            "Best_Portfolio_On_Session": max(valid_model_returns, key=valid_model_returns.get) if valid_model_returns else "DATA_MISSING_CRITICAL",
            "Worst_Portfolio_On_Session": min(valid_model_returns, key=valid_model_returns.get) if valid_model_returns else "DATA_MISSING_CRITICAL",
            "Data_Status": data_status,
            "Comment": (
                "Rendements calcules sur les actifs disponibles et renormalises lorsque des observations manquent ; "
                "aucun rendement manquant n'est remplace par zero."
            ),
        }
        for model, value in per_model.items():
            row[f"Return_{model}"] = value
        rows.append(row)
    backtest = pd.DataFrame(rows)

    summary_rows = []
    for model in selected_models:
        values = pd.Series([row[f"Return_{model}"] for row in rows], dtype=float)
        current_values = pd.Series([row["Current_Portfolio_Return"] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.empty:
            avg_loss = worst_loss = cumulative_loss = np.nan
            outperform = underperform = 0
        else:
            avg_loss = float((-finite).mean())
            worst_loss = float(-finite.min())
            cumulative_loss = float((-finite).sum())
            outperform = int((values > current_values).sum())
            underperform = int((values < current_values).sum())
        summary_rows.append(
            {
                "Model": model,
                "Average_Loss_Worst_10_Sessions": avg_loss,
                "Worst_Observed_Loss": worst_loss,
                "Nb_Sessions_Outperform_Current": outperform,
                "Nb_Sessions_Underperform_Current": underperform,
                "Cumulative_Loss_Worst_10_Sessions": cumulative_loss,
                "Data_Status": "DATA_MISSING_CRITICAL" if finite.empty else ("PARTIAL_DATA" if backtest["Data_Status"].astype(str).eq("PARTIAL_DATA").any() else "CALCULATED"),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["Average_Loss_Worst_10_Sessions", "Worst_Observed_Loss"],
        ascending=[True, True],
        na_position="last",
    )
    summary["Robustness_Rank"] = range(1, len(summary) + 1)
    return backtest, summary


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
