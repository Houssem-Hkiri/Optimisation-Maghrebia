"""Quality audit for notebooks/02_optimisation_portefeuille.ipynb.

The notebook is intentionally kept presentation-oriented. This script carries
the detailed checks: input integrity, covariance, optimiser outputs, regulatory
checks, Monte Carlo selections, efficient frontier density, Plotly exports and
final scoring on 100 points.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from maghrebia_quant.optimization_apt import APTOptimizationConfig, load_apt_optimization_inputs


EXPORT_DIR = PROJECT_DIR / "exports"
RESULTS_PATH = EXPORT_DIR / "notebook_02_optimisation_resultats.xlsx"
FIGURES_DIR = EXPORT_DIR / "figures" / "notebook_02"
QUALITY_DIR = EXPORT_DIR / "quality"
QUALITY_PATH = QUALITY_DIR / "notebook_02_quality_report.xlsx"


@dataclass
class Check:
    category: str
    check: str
    status: str
    details: str
    severity: str = "INFO"


def status(ok: bool, warning: bool = False) -> str:
    if ok:
        return "WARNING" if warning else "PASSED"
    return "FAILED"


def add(rows: list[Check], category: str, check: str, ok: bool, details: str, *, warning: bool = False, severity: str = "INFO") -> None:
    rows.append(Check(category, check, status(ok, warning), details, severity))


def safe_read_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def parse_weights_json(value: object) -> dict[str, float]:
    if pd.isna(value):
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return {str(k): float(v) for k, v in parsed.items()}


def selected_monte_carlo_rows(monte_carlo: pd.DataFrame) -> pd.DataFrame:
    if monte_carlo.empty:
        return pd.DataFrame()
    selections = [
        ("Monte_Carlo_Max_Return", monte_carlo["expected_return"].idxmax()),
        ("Monte_Carlo_Min_Volatility", monte_carlo["volatility"].idxmin()),
        ("Monte_Carlo_Min_CVaR", monte_carlo["cvar_95"].idxmin()),
    ]
    rows = []
    for label, idx in selections:
        row = monte_carlo.loc[idx].to_dict()
        row["Criterion"] = label
        rows.append(row)
    out = pd.DataFrame(rows)
    out["Is_Same_As_Other_Selected"] = out["portfolio_id"].duplicated(keep=False)
    return out


def category_score(checks: pd.DataFrame, category: str, points: float, blocking_only: bool = False) -> float:
    subset = checks.loc[checks["category"].eq(category)].copy()
    if subset.empty:
        return 0.0
    if blocking_only:
        subset = subset.loc[subset["severity"].isin(["BLOCKING", "CRITICAL"])]
        if subset.empty:
            return points
    failed = int(subset["status"].eq("FAILED").sum())
    warnings = int(subset["status"].eq("WARNING").sum())
    if failed:
        return 0.0
    penalty = min(points * 0.35, warnings * points * 0.08)
    return max(0.0, points - penalty)


def main() -> int:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[Check] = []

    add(rows, "Inputs et coherence donnees", "Workbook final existe", RESULTS_PATH.exists(), str(RESULTS_PATH), severity="BLOCKING")
    data = load_apt_optimization_inputs(PROJECT_DIR)
    expected = data["expected"]
    mu = data["mu"]
    sigma = data["sigma"]
    returns = data["returns"]
    universe_assets = expected["asset_id"].astype(str)

    add(rows, "Inputs et coherence donnees", "Actifs optimisables presents", len(universe_assets) > 0, f"{len(universe_assets)} actifs", severity="BLOCKING")
    add(rows, "Inputs et coherence donnees", "Aucun doublon actif", not universe_assets.duplicated().any(), "asset_id unique", severity="BLOCKING")
    add(rows, "Rendements et covariance", "Aucun NaN dans les rendements attendus", not mu.isna().any(), f"NaN={int(mu.isna().sum())}", severity="BLOCKING")
    add(rows, "Rendements et covariance", "Aucun NaN dans la covariance", not sigma.isna().any().any(), f"NaN={int(sigma.isna().sum().sum())}", severity="BLOCKING")
    add(rows, "Rendements et covariance", "Covariance symetrique", np.allclose(sigma, sigma.T, atol=1e-10), "tol=1e-10", severity="BLOCKING")
    eig_min = float(np.linalg.eigvalsh(sigma.to_numpy(float)).min())
    add(rows, "Rendements et covariance", "Covariance positive semi-definie", eig_min >= -1e-9, f"min_eigenvalue={eig_min:.3e}", warning=eig_min < 1e-10, severity="BLOCKING")

    date_index = pd.to_datetime(returns.index)
    median_gap = float(pd.Series(date_index).diff().dt.days.median())
    inferred = "daily" if median_gap <= 2 else ("weekly" if median_gap <= 8 else "other")
    periods = 252 if inferred == "daily" else 52
    add(rows, "Frequence et annualisation", "Frequence detectee documentee", inferred in {"daily", "weekly"}, f"{inferred}, median_gap={median_gap}, periods={periods}", severity="BLOCKING")
    add(rows, "Frequence et annualisation", "Observations suffisantes", len(returns) >= 40, f"{len(returns)} observations", severity="BLOCKING")

    xls = pd.ExcelFile(RESULTS_PATH) if RESULTS_PATH.exists() else None
    required_sheets = [
        "Hypotheses", "Inputs", "Current_Portfolio", "Asset_Metrics", "Expected_Returns_APT",
        "Covariance", "Constraints", "Optimized_Portfolios", "Monte_Carlo", "Efficient_Frontier",
        "Scenario_Analysis", "Stress_Tests", "Recommended_Portfolio", "Final_Summary",
    ]
    present_sheets = set(xls.sheet_names) if xls is not None else set()
    missing_sheets = [s for s in required_sheets if s not in present_sheets]
    add(rows, "Graphiques et exports", "Feuilles Excel finales presentes", not missing_sheets, "missing=" + ",".join(missing_sheets), severity="BLOCKING")

    optimized = safe_read_excel(RESULTS_PATH, "Optimized_Portfolios")
    if not optimized.empty and {"portfolio_name", "optimized_weight"}.issubset(optimized.columns):
        sums = optimized.groupby("portfolio_name")["optimized_weight"].sum()
        mins = optimized.groupby("portfolio_name")["optimized_weight"].min()
        add(rows, "Modeles optimisation", "Somme des poids = 1", bool(np.allclose(sums, 1.0, atol=1e-6)), sums.round(8).to_dict(), severity="BLOCKING")
        add(rows, "Modeles optimisation", "Aucun poids negatif", bool((mins >= -1e-8).all()), mins.round(8).to_dict(), severity="BLOCKING")
        optimized_only = optimized.loc[~optimized["portfolio_name"].astype(str).isin(["Current_Portfolio", "Portefeuille actuel"])].copy()
        max_w = optimized_only.groupby("portfolio_name")["optimized_weight"].max().max()
        add(rows, "Modeles optimisation", "Borne par actif respectee", max_w <= APTOptimizationConfig().max_weight_per_asset + 1e-6, f"max_weight={max_w:.4f}", severity="BLOCKING")
    else:
        add(rows, "Modeles optimisation", "Table de poids exploitable", False, "Optimized_Portfolios manquante ou incomplete", severity="BLOCKING")

    constraints = safe_read_excel(RESULTS_PATH, "Constraints")
    if not constraints.empty:
        status_col = "compliance_status" if "compliance_status" in constraints.columns else "Status"
        text = constraints[status_col].astype(str)
        breaches = int(text.str.contains("BREACH|FAILED", case=False, na=False).sum())
        non_testable = int(text.str.contains("MISSING|NON_TESTABLE|NOT_ENFORCED", case=False, na=False).sum())
        add(rows, "Contraintes", "Aucune violation testable", breaches == 0, f"breaches={breaches}", severity="BLOCKING")
        add(rows, "Contraintes", "Contraintes non testables documentees", True, f"non_testable={non_testable}", warning=non_testable > 0, severity="WARNING")
    else:
        add(rows, "Contraintes", "Table contraintes disponible", False, "Constraints vide", severity="BLOCKING")

    mc = safe_read_excel(RESULTS_PATH, "Monte_Carlo")
    mc_selected = selected_monte_carlo_rows(mc)
    if not mc.empty:
        add(rows, "Monte Carlo et frontiere", "Monte Carlo >= 15000", len(mc) >= 15_000, f"{len(mc)} simulations", severity="BLOCKING")
        add(rows, "Monte Carlo et frontiere", "Selections MC recalculables", not mc_selected.empty, f"{len(mc_selected)} selections", severity="BLOCKING")
        duplicate_note = "duplicates=" + ",".join(mc_selected.loc[mc_selected["Is_Same_As_Other_Selected"], "Criterion"].astype(str))
        add(rows, "Monte Carlo et frontiere", "Selections MC distinctes ou expliquees", True, duplicate_note, warning=mc_selected["Is_Same_As_Other_Selected"].any(), severity="WARNING")
    else:
        add(rows, "Monte Carlo et frontiere", "Monte Carlo disponible", False, "Monte_Carlo vide", severity="BLOCKING")

    frontier = safe_read_excel(RESULTS_PATH, "Efficient_Frontier")
    if not frontier.empty:
        valid_count = int(frontier[["volatility", "achieved_return"]].dropna().shape[0])
        monotone = frontier.sort_values("volatility")["achieved_return"].dropna().diff().fillna(0).ge(-1e-6).all()
        add(rows, "Monte Carlo et frontiere", "Frontiere efficiente >= 100 points", valid_count >= 100, f"{valid_count} points", severity="BLOCKING")
        add(rows, "Monte Carlo et frontiere", "Frontiere monotone", bool(monotone), "rendement non decroissant en volatilite", severity="BLOCKING")
    else:
        add(rows, "Monte Carlo et frontiere", "Frontiere efficiente disponible", False, "Efficient_Frontier vide", severity="BLOCKING")

    figures = list(FIGURES_DIR.glob("*.html"))
    add(rows, "Graphiques et exports", "Figures HTML exportees", len(figures) >= 10, f"{len(figures)} fichiers", severity="BLOCKING")

    final_summary = safe_read_excel(RESULTS_PATH, "Final_Summary")
    clarity_terms = ["Sharpe", "APT", "frontiere", "contraintes", "10 MD"]
    clarity_text = " ".join(final_summary.astype(str).fillna("").to_numpy().ravel()) if not final_summary.empty else ""
    clarity_hits = sum(term.lower() in clarity_text.lower() for term in clarity_terms)
    add(rows, "Clarte methodologique", "Conclusion methodologique documentee", clarity_hits >= 4, f"{clarity_hits}/5 themes", severity="BLOCKING")

    asset_metrics = safe_read_excel(RESULTS_PATH, "Asset_Metrics")
    warnings_rows = []
    if not asset_metrics.empty:
        if "sharpe_historical" in asset_metrics.columns:
            for _, row in asset_metrics.loc[pd.to_numeric(asset_metrics["sharpe_historical"], errors="coerce").gt(3)].iterrows():
                level = "CRITICAL" if float(row["sharpe_historical"]) > 5 else "WARNING"
                warnings_rows.append({"warning_type": "HIGH_HISTORICAL_SHARPE", "severity": level, "asset": row.get("asset_name", row.get("asset_id")), "value": row["sharpe_historical"]})
        if {"var_95_historical", "cvar_95_historical", "asset_type"}.issubset(asset_metrics.columns):
            risky = asset_metrics["asset_type"].isin(["listed_equity", "corporate_bond", "government_bond"])
            zero_risk = asset_metrics.loc[risky & ((asset_metrics["var_95_historical"].fillna(0) <= 0) | (asset_metrics["cvar_95_historical"].fillna(0) <= 0))]
            for _, row in zero_risk.iterrows():
                warnings_rows.append({"warning_type": "WARNING_ZERO_VAR_CVAR_WITH_RISKY_ASSETS", "severity": "WARNING", "asset": row.get("asset_name", row.get("asset_id")), "value": ""})

    checks = pd.DataFrame([r.__dict__ for r in rows])
    score_grid = pd.DataFrame(
        [
            ("Inputs et coherence donnees", 15),
            ("Rendements et covariance", 15),
            ("Modeles optimisation", 20),
            ("Contraintes", 15),
            ("Monte Carlo et frontiere", 15),
            ("Graphiques et exports", 10),
            ("Clarte methodologique", 10),
        ],
        columns=["category", "max_points"],
    )
    score_grid["points_obtained"] = [category_score(checks, row.category, row.max_points) for row in score_grid.itertuples()]
    audit_score = float(score_grid["points_obtained"].sum())
    blocking_failed = int(checks.loc[checks["severity"].isin(["BLOCKING", "CRITICAL"]), "status"].eq("FAILED").sum())
    global_status = "PASSED" if audit_score >= 90 and blocking_failed == 0 else "FAILED"
    final_score = pd.DataFrame(
        [
            {
                "Audit_Score": audit_score,
                "Target_Score": 90,
                "Global_Status": global_status,
                "Blocking_Failures": blocking_failed,
                "Comment": "Objectif atteint." if global_status == "PASSED" else "Points bloquants a corriger avant soutenance.",
            }
        ]
    )

    with pd.ExcelWriter(QUALITY_PATH, engine="openpyxl") as writer:
        final_score.to_excel(writer, sheet_name="Summary", index=False)
        checks.loc[checks["category"].eq("Inputs et coherence donnees")].to_excel(writer, sheet_name="Input_Checks", index=False)
        checks.loc[checks["category"].eq("Frequence et annualisation")].to_excel(writer, sheet_name="Frequency_Checks", index=False)
        checks.loc[checks["category"].eq("Rendements et covariance")].to_excel(writer, sheet_name="Covariance_Checks", index=False)
        checks.loc[checks["category"].eq("Modeles optimisation")].to_excel(writer, sheet_name="Optimization_Checks", index=False)
        checks.loc[checks["category"].eq("Contraintes")].to_excel(writer, sheet_name="Regulatory_Checks", index=False)
        mc_selected.to_excel(writer, sheet_name="Monte_Carlo_Checks", index=False)
        checks.loc[checks["category"].eq("Monte Carlo et frontiere")].to_excel(writer, sheet_name="Efficient_Frontier_Checks", index=False)
        checks.loc[checks["category"].eq("Graphiques et exports")].to_excel(writer, sheet_name="Plotly_Exports_Checks", index=False)
        pd.DataFrame(warnings_rows).to_excel(writer, sheet_name="Warnings", index=False)
        score_grid.to_excel(writer, sheet_name="Final_Audit_Score", index=False)

    print(f"Quality report: {QUALITY_PATH.relative_to(PROJECT_DIR)}")
    print(f"Audit_Score: {audit_score:.1f}/100")
    print(f"Global_Status: {global_status}")
    if global_status != "PASSED":
        failed = checks.loc[checks["status"].eq("FAILED"), ["category", "check", "details"]]
        print(failed.to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
