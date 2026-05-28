"""Compatibility wrapper for legacy Notebook 02 optimisation helpers.

New Notebook 02 code should import from ``maghrebia_quant.optimization``.
The names kept here are technical aliases required by older exports from
Notebook 01; they are not methodological labels for the final report.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew


@dataclass(frozen=True)
class APTOptimizationConfig:
    """Documented optimisation parameters."""

    max_weight_per_asset: float = 0.30
    max_weight_per_issuer: float = 0.35
    max_equity_weight: float = 0.30
    max_corporate_weight: float = 0.65
    turnover_thresholds: tuple[float, ...] = (0.30, 0.40, 0.50)
    cvar_beta: float = 0.95
    monte_carlo_required: int = 15_000
    monte_carlo_max_attempts: int = 300_000
    frontier_points: int = 500
    random_seed: int = 20250517
    rf_annual_fallback: float = 0.076
    slsqp_random_starts: int = 60

    @property
    def primary_turnover_limit(self) -> float:
        """Turnover limit used consistently by deterministic models and Monte Carlo."""

        return float(self.turnover_thresholds[0])


REFERENCE_ROWS = [
    ("Markowitz 1952", "Portfolio Selection. The Journal of Finance, 7(1), 77-91.", "Minimum variance, moyenne-variance et frontière efficiente."),
    ("Markowitz 1959", "Portfolio Selection: Efficient Diversification of Investments. Yale University Press.", "Diversification efficiente."),
    ("Sharpe 1966", "Mutual Fund Performance. The Journal of Business, 39(1), 119-138.", "Ratio de Sharpe."),
    ("Rockafellar & Uryasev 2000", "Optimization of Conditional Value-at-Risk. Journal of Risk, 2(3), 21-41.", "Formulation Mean-CVaR."),
    ("Rockafellar & Uryasev 2002", "Conditional Value-at-Risk for General Loss Distributions. Journal of Banking & Finance, 26(7), 1443-1471.", "Fondement CVaR."),
    ("Maillard, Roncalli & Teiletche 2010", "The Properties of Equally Weighted Risk Contribution Portfolios. Journal of Portfolio Management, 36(4), 60-70.", "Risk Parity / Equal Risk Contribution."),
    ("Lobo, Fazel & Boyd 2007", "Portfolio Optimization with Linear and Fixed Transaction Costs. Annals of Operations Research, 152, 341-365.", "Pénalité de turnover."),
    ("Boyd & Vandenberghe 2004", "Convex Optimization. Cambridge University Press.", "Optimisation convexe et contraintes."),
    ("Glasserman 2004", "Monte Carlo Methods in Financial Engineering. Springer.", "Simulation Monte Carlo."),
    ("CFA Institute", "Quantitative Investment Analysis, 4th Edition.", "Mesures de risque et analyse quantitative."),
    ("Dachert et al. 2022", "Multicriteria asset allocation in practice. OR Spectrum, 44, 349-373.", "Allocation multi-objectif."),
    ("Braun, Schmeiser & Schreiber 2013", "Portfolio Optimization Under Solvency II: Implicit Constraints Imposed by the Market Risk Standard Formula.", "Assurance, solvabilité et contraintes."),
]


def read_matrix_csv(path: str | Path) -> pd.DataFrame:
    """Read a square matrix exported with an ``asset_id`` first column."""

    df = pd.read_csv(path)
    if "asset_id" not in df.columns:
        raise ValueError(f"Matrice invalide sans colonne asset_id : {path}")
    mat = df.set_index("asset_id")
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    mat = mat.apply(pd.to_numeric, errors="coerce")
    return mat


def nearest_psd(matrix: pd.DataFrame, eps: float = 1e-10) -> tuple[pd.DataFrame, bool, float]:
    """Return a symmetric PSD matrix and whether eigenvalue clipping was used."""

    sym = (matrix + matrix.T) / 2
    values = sym.to_numpy(float)
    eigvals, eigvecs = np.linalg.eigh(values)
    min_eig = float(eigvals.min())
    repaired = min_eig < -eps
    eigvals = np.maximum(eigvals, eps)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T
    psd = pd.DataFrame((out + out.T) / 2, index=matrix.index, columns=matrix.columns)
    return psd, repaired, min_eig


def load_apt_optimization_inputs(project_dir: str | Path) -> dict[str, pd.DataFrame | pd.Series | float | bool]:
    """Load APT exports, workbook outputs and validate primary inputs."""

    project = Path(project_dir)
    opt_dir = project / "data" / "exports" / "optimization_inputs"
    workbook_path = project / "data" / "exports" / "diagnostic_pre_optimisation_2025.xlsx"
    required = {
        "expected": opt_dir / "apt_expected_returns_2025.csv",
        "covariance": opt_dir / "apt_covariance_matrix_2025.csv",
        "betas": opt_dir / "apt_betas_2025.csv",
        "factors": opt_dir / "apt_factors_weekly_2025.csv",
        "diagnostics": opt_dir / "apt_diagnostics_2025.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Exports APT manquants : {missing}")
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook notebook 01 introuvable : {workbook_path}")

    expected = pd.read_csv(required["expected"])
    sigma = read_matrix_csv(required["covariance"])
    betas = pd.read_csv(required["betas"])
    factors = pd.read_csv(required["factors"])
    diagnostics = pd.read_csv(required["diagnostics"])
    xl = pd.ExcelFile(workbook_path)
    sheets = {
        name: pd.read_excel(workbook_path, sheet_name=name)
        for name in [
            "Optimisable_Pocket",
            "Non_Optimisable",
            "Portfolio_Summary",
            "Returns_Model",
            "Risk_Free_Daily_2025",
            "Asset_Metrics",
            "Final_Control",
            "Stress_Test_Summary",
            "Portfolio_Metrics",
        ]
        if name in xl.sheet_names
    }

    expected["asset_id"] = expected["asset_id"].astype(str)
    assets = expected["asset_id"].tolist()
    sigma = sigma.reindex(index=assets, columns=assets)
    sigma, sigma_repaired, min_eig_before = nearest_psd(sigma)
    mu = expected.set_index("asset_id")["expected_return_annualized_final"].astype(float).reindex(assets)
    if mu.isna().any() or sigma.isna().any().any():
        raise ValueError("mu_APT ou Sigma_APT contient des valeurs manquantes après alignement.")

    returns = sheets["Returns_Model"].copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.set_index("date").reindex(columns=assets).apply(pd.to_numeric, errors="coerce")
    if returns.isna().any().any():
        missing_count = int(returns.isna().sum().sum())
        raise ValueError(f"Rendements historiques incomplets pour métriques ex-post : {missing_count} NaN.")

    rf = sheets["Risk_Free_Daily_2025"].copy()
    rf_annual = float(pd.to_numeric(rf["rf_annual_decimal"], errors="coerce").dropna().mean())
    if not np.isfinite(rf_annual):
        rf_annual = APTOptimizationConfig().rf_annual_fallback

    return {
        "project_dir": project,
        "workbook_path": workbook_path,
        "expected": expected,
        "mu": mu,
        "sigma": sigma,
        "sigma_repaired": sigma_repaired,
        "sigma_min_eig_before": min_eig_before,
        "returns": returns,
        "betas": betas,
        "factors": factors,
        "diagnostics": diagnostics,
        "rf_annual": rf_annual,
        **sheets,
    }


def load_apt_mu_scenarios(project_dir: str | Path, assets: list[str]) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Load APT expected-return scenarios plus the historical raw comparison when available."""

    project = Path(project_dir)
    scenario_path = project / "data" / "processed" / "apt_expected_returns_scenarios.csv"
    legacy_scenario_path = project / "data" / "processed" / "expected_returns_apt_scenarios.csv"
    fallback_path = project / "data" / "exports" / "optimization_inputs" / "apt_expected_returns_2025.csv"
    if scenario_path.exists():
        df = pd.read_csv(scenario_path)
    elif legacy_scenario_path.exists():
        df = pd.read_csv(legacy_scenario_path)
    elif fallback_path.exists():
        base = pd.read_csv(fallback_path)
        scenario_cols = ["asset_id", "mu_apt_prudent", "mu_apt_central", "mu_apt_optimistic"]
        if "mu_historical_raw" in base.columns:
            scenario_cols.append("mu_historical_raw")
        df = base[scenario_cols].copy()
    else:
        raise FileNotFoundError("Aucun export de scénarios APT n'est disponible.")
    asset_col = "asset_id" if "asset_id" in df.columns else "Asset"
    df[asset_col] = df[asset_col].astype(str)
    df = df.drop_duplicates(asset_col, keep="last").set_index(asset_col)
    required = ["mu_apt_prudent", "mu_apt_central", "mu_apt_optimistic"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes de scénarios APT manquantes : {missing}")
    df = df.reindex(assets)
    if df[required].isna().any().any():
        raise ValueError("Scénarios APT incomplets après alignement avec les actifs optimisables.")
    scenarios = {
        "APT_Prudent": df["mu_apt_prudent"].astype(float),
        "APT_Central": df["mu_apt_central"].astype(float),
        "APT_Optimistic": df["mu_apt_optimistic"].astype(float),
    }
    if "mu_historical_raw" in df.columns and df["mu_historical_raw"].notna().all():
        scenarios["Historical_Raw"] = df["mu_historical_raw"].astype(float)
    audit_cols = [*required, *(["mu_historical_raw"] if "mu_historical_raw" in df.columns else [])]
    audit = df[audit_cols].reset_index().rename(columns={asset_col: "asset_id"})
    return scenarios, audit


def build_universe(data: dict[str, object]) -> pd.DataFrame:
    """Build the aligned optimisation universe from APT expected returns."""

    expected = data["expected"].copy()
    opt = data["Optimisable_Pocket"].copy()
    opt["asset_id"] = opt["asset_id"].astype(str)
    cols = [
        "asset_id", "asset_name", "asset_type", "asset_class_standardized", "sector",
        "market_value", "optimisable_weight", "portfolio_weight", "isin",
        "maturity_date", "maturity_date_schedule",
    ]
    opt = opt[[c for c in cols if c in opt.columns]].copy()
    universe = expected.merge(opt, on="asset_id", how="left", suffixes=("", "_portfolio"))
    universe["asset_name"] = universe["asset_name"].fillna(universe.get("asset_name_portfolio"))
    universe["asset_class"] = universe["asset_class"].fillna(universe.get("asset_class_standardized"))
    universe["asset_type"] = universe["asset_type"].fillna(universe.get("asset_type_portfolio"))
    universe["sector"] = universe["sector"].fillna(universe.get("sector_portfolio")).fillna("OTHER")
    universe["current_value"] = pd.to_numeric(universe["current_value"], errors="coerce").fillna(pd.to_numeric(universe.get("market_value"), errors="coerce"))
    universe["current_weight_optimisable"] = pd.to_numeric(universe["current_weight_optimisable"], errors="coerce")
    universe["issuer"] = universe["asset_id"]
    maturity_primary = pd.to_datetime(universe.get("maturity_date"), errors="coerce")
    maturity_secondary = pd.to_datetime(universe.get("maturity_date_schedule"), errors="coerce")
    universe["maturity_date"] = maturity_primary.fillna(maturity_secondary)
    universe["residual_maturity"] = ((universe["maturity_date"] - pd.Timestamp("2025-12-31")).dt.days / 365.0).clip(lower=0)
    total_weight = universe["current_weight_optimisable"].sum()
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        universe["current_weight_optimisable"] = universe["current_weight_optimisable"] / total_weight
    universe["expected_return_annualized"] = universe["expected_return_annualized_final"].astype(float)
    universe["optimisable_flag"] = universe["asset_type"].isin(["listed_equity", "government_bond", "corporate_bond"])
    return universe.loc[universe["optimisable_flag"]].reset_index(drop=True)


def build_asset_alignment(data: dict[str, object], universe: pd.DataFrame) -> pd.DataFrame:
    """Audit asset presence in all optimisation inputs."""

    portfolio_assets = set(data["Optimisable_Pocket"]["asset_id"].astype(str))
    mu_assets = set(data["mu"].index.astype(str))
    sigma_assets = set(data["sigma"].index.astype(str))
    return_assets = set(data["returns"].columns.astype(str))
    rows = []
    for asset in sorted(portfolio_assets | mu_assets | sigma_assets | return_assets):
        row = {
            "asset_id": asset,
            "asset_name": universe.set_index("asset_id")["asset_name"].to_dict().get(asset, asset),
            "present_in_portfolio": asset in portfolio_assets,
            "present_in_mu_APT": asset in mu_assets,
            "present_in_Sigma_APT": asset in sigma_assets,
            "present_in_returns": asset in return_assets,
            "present_in_constraints": asset in portfolio_assets,
        }
        row["final_status"] = "OK" if all(row[k] for k in row if k.startswith("present_")) else "MISSING_INPUT"
        rows.append(row)
    return pd.DataFrame(rows)


def technical_provisions_from_portfolio(project_dir: str | Path) -> float:
    """Extract technical provisions from the portfolio workbook."""

    path = Path(project_dir) / "data" / "Maghrebia Portfolio.xlsx"
    df = pd.read_excel(path, sheet_name="Principal")
    mask = df["Désignation des actifs"].astype(str).str.contains("Montant des Provisions Techniques", case=False, na=False)
    if not mask.any():
        raise ValueError("Provisions techniques introuvables dans le portefeuille.")
    return float(pd.to_numeric(df.loc[mask, "Coût d'entrée au bilan"], errors="coerce").dropna().iloc[0])


def build_context(data: dict[str, object], universe: pd.DataFrame, project_dir: str | Path) -> dict[str, object]:
    """Build context used for total-portfolio regulatory checks."""

    summary = data["Portfolio_Summary"].iloc[0]
    fixed = data["Non_Optimisable"].copy()
    factors = data.get("factors", pd.DataFrame())
    if isinstance(factors, pd.DataFrame) and "week_date" in factors.columns:
        factor_dates = pd.Series(pd.to_datetime(factors["week_date"], errors="coerce")).dropna()
    else:
        factor_dates = pd.Series(dtype="datetime64[ns]")
    median_factor_gap = float(factor_dates.sort_values().diff().dt.days.median()) if len(factor_dates) > 1 else np.nan
    sigma = data["sigma"]
    sigma_diag = np.diag(sigma.to_numpy(float))
    annualization_ok = bool(
        np.isfinite(median_factor_gap)
        and 5 <= median_factor_gap <= 9
        and np.isfinite(sigma_diag).all()
        and float(np.nanmedian(sigma_diag)) > 0
    )
    annualization_checks = pd.DataFrame([{
        "Check": "ANNUALIZATION_CONSISTENCY",
        "Status": "PASSED" if annualization_ok else "FAILED",
        "Return_Frequency": "daily",
        "Return_Periods_Per_Year": 252,
        "Sigma_APT_Annualization": "weekly_factor_covariance_x52_from_notebook_01",
        "Median_Factor_Gap_Days": median_factor_gap,
        "Comment": "Les rendements historiques restent quotidiens; Sigma_APT est l'export annuel du modèle APT hebdomadaire du notebook 01.",
    }])
    return {
        "technical_provisions": technical_provisions_from_portfolio(project_dir),
        "total_value": float(summary["total_portfolio_value"]),
        "optimisable_value": float(summary["optimisable_value"]),
        "fixed": fixed,
        "universe": universe.copy(),
        "annualization_checks": annualization_checks,
    }


def build_regulatory_constraints_map(referential_found: bool = False) -> pd.DataFrame:
    """Document testable, legal and non-testable regulatory constraints."""

    source = "Referentiel reglementaire.docx" if referential_found else "Contraintes matérialisées dans le pipeline notebook 01 et cahier des charges utilisateur"
    rows = [
        ("COVERAGE_GLOBAL", "Couverture globale des provisions techniques", "min", 1.00, "technical_provisions", "all_assets", True, "ENFORCED", "LEGAL_REGULATORY", source),
        ("STATE_MIN_20_PT", "Titres émis ou garantis par l'État >= 20% PT", "min", 0.20, "technical_provisions", "government_bond", True, "ENFORCED", "LEGAL_REGULATORY", source),
        ("LISTED_EQUITY_PER_COMPANY_MAX_10_PT", "Action cotée BVMT par société <= 10% PT", "max", 0.10, "technical_provisions", "listed_equity_per_company", True, "ENFORCED", "LEGAL_REGULATORY", source),
        ("REAL_ESTATE_TOTAL_MAX_20_PT", "Immobilier total <= 20% PT", "max", 0.20, "technical_provisions", "real_estate", True, "POST_CHECK", "LEGAL_REGULATORY", source),
        ("SICAR_TOTAL_MAX_10_PT", "SICAR/SICAF total <= 10% PT", "max", 0.10, "technical_provisions", "sicar_total", True, "POST_CHECK", "LEGAL_REGULATORY", source),
        ("CAPITAL_SOCIAL_LIMITS", "Limites en pourcentage du capital social", "max", np.nan, "capital_social", "issuer", True, "CHECK_NOT_ENFORCED_MISSING_DATA", "UNTESTED_MISSING_DATA", source),
        ("OPCVM_ENTITY_LIMITS", "OPCVM par entité", "max", np.nan, "technical_provisions", "fund_per_entity", True, "CHECK_NOT_ENFORCED_MISSING_DATA", "UNTESTED_MISSING_DATA", source),
        ("SICAR_ENTITY_LIMITS", "SICAR/SICAF par société", "max", np.nan, "technical_provisions", "sicar_per_company", True, "CHECK_NOT_ENFORCED_MISSING_DATA", "UNTESTED_MISSING_DATA", source),
        ("OTHER_SECURITIES_LIMITS", "Autres valeurs mobilières", "max", np.nan, "technical_provisions", "other_securities", True, "CHECK_NOT_ENFORCED_MISSING_DATA", "UNTESTED_MISSING_DATA", source),
    ]
    return pd.DataFrame(rows, columns=[
        "constraint_id", "constraint_name", "threshold_type", "threshold_value", "denominator",
        "applies_to", "hard_constraint", "implementation_status", "constraint_origin", "source_reference",
    ])

def class_masks(universe: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks by asset class/type."""

    return {
        "government_bond": universe["asset_type"].eq("government_bond").to_numpy(),
        "listed_equity": universe["asset_type"].eq("listed_equity").to_numpy(),
        "corporate_bond": universe["asset_type"].eq("corporate_bond").to_numpy(),
    }


def upper_bounds_vector(universe: pd.DataFrame, context: dict[str, object], config: APTOptimizationConfig) -> np.ndarray:
    """Per-asset upper bounds including testable regulatory equity limits."""

    upper = np.full(len(universe), config.max_weight_per_asset, dtype=float)
    v_opt = float(context["optimisable_value"])
    pt = float(context["technical_provisions"])
    equity_limit = 0.10 * pt / v_opt
    equity_mask = universe["asset_type"].eq("listed_equity").to_numpy()
    upper[equity_mask] = np.minimum(upper[equity_mask], equity_limit)
    return upper


def build_internal_constraints_map(config: APTOptimizationConfig) -> pd.DataFrame:
    """Document internal governance constraints separately from legal rules."""

    return pd.DataFrame([
        ("BUDGET", "Somme des poids optimisables = 1", 1.0, True, "INTERNAL_GOVERNANCE"),
        ("NO_SHORT_SELLING", "Poids négatifs interdits", 0.0, True, "INTERNAL_GOVERNANCE"),
        ("MAX_WEIGHT_PER_ASSET", "Poids maximal par actif", config.max_weight_per_asset, True, "INTERNAL_GOVERNANCE"),
        ("MAX_WEIGHT_PER_ISSUER", "Poids maximal par émetteur", config.max_weight_per_issuer, True, "INTERNAL_GOVERNANCE"),
        ("MAX_EQUITY_WEIGHT", "Poids actions maximal dans la poche optimisable", config.max_equity_weight, True, "INTERNAL_GOVERNANCE"),
        ("MAX_CORPORATE_WEIGHT", "Poids corporate maximal dans la poche optimisable", config.max_corporate_weight, True, "INTERNAL_GOVERNANCE"),
        ("TURNOVER_LIMIT", "Turnover maximal de gouvernance", config.turnover_thresholds[0], True, "INTERNAL_GOVERNANCE"),
    ], columns=["constraint_id", "description", "value", "hard_constraint", "constraint_origin"])

def cvxpy_constraints(
    w: cp.Variable,
    universe: pd.DataFrame,
    current_weights: np.ndarray,
    context: dict[str, object],
    config: APTOptimizationConfig,
    turnover_limit: float | None,
) -> list:
    """Common convex constraints for long-only optimisation."""

    masks = class_masks(universe)
    upper = upper_bounds_vector(universe, context, config)
    v_opt = float(context["optimisable_value"])
    pt = float(context["technical_provisions"])
    state_min_weight = max(0.0, 0.20 * pt / v_opt)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= upper,
        cp.sum(cp.multiply(masks["government_bond"].astype(float), w)) >= state_min_weight,
        cp.sum(cp.multiply(masks["listed_equity"].astype(float), w)) <= config.max_equity_weight,
        cp.sum(cp.multiply(masks["corporate_bond"].astype(float), w)) <= config.max_corporate_weight,
    ]
    if turnover_limit is not None:
        constraints.append(cp.norm1(w - current_weights) <= turnover_limit)
    return constraints


def evaluate_constraints(weights: np.ndarray, universe: pd.DataFrame, context: dict[str, object], regulatory_map: pd.DataFrame, portfolio_name: str) -> pd.DataFrame:
    """Check regulatory constraints on total portfolio after optimisation."""

    w = np.asarray(weights, dtype=float)
    pt = float(context["technical_provisions"])
    v_opt = float(context["optimisable_value"])
    total_value = float(context["total_value"])
    fixed = context["fixed"].copy()
    fixed_value_by_type = fixed.groupby("asset_type")["market_value"].sum().to_dict() if not fixed.empty else {}
    opt_values = universe.assign(optimized_value=w * v_opt)
    rows = []

    def row(cid: str, exposure: float | None, limit_pct: float | None, threshold_type: str, status: str | None = None) -> None:
        if exposure is None or limit_pct is None or not np.isfinite(limit_pct):
            compliance = status or "CHECK_NOT_ENFORCED_MISSING_DATA"
            threshold_value = np.nan
            breach = np.nan
        else:
            threshold_value = limit_pct * pt
            if threshold_type == "min":
                breach = max(0.0, threshold_value - exposure)
            else:
                breach = max(0.0, exposure - threshold_value)
            compliance = "COMPLIANT" if breach <= 1e-6 else "BREACH"
        rows.append({
            "portfolio_name": portfolio_name,
            "constraint_id": cid,
            "exposure_value": np.nan if exposure is None else float(exposure),
            "threshold_value": threshold_value,
            "denominator_value": pt,
            "exposure_pct_denominator": np.nan if exposure is None else float(exposure) / pt,
            "limit_pct": limit_pct,
            "compliance_status": compliance,
            "breach_amount": breach,
            "data_quality_flag": compliance if compliance == "CHECK_NOT_ENFORCED_MISSING_DATA" else "OK",
        })

    row("COVERAGE_GLOBAL", total_value, 1.00, "min")
    state_exp = float(opt_values.loc[opt_values["asset_type"].eq("government_bond"), "optimized_value"].sum() + fixed_value_by_type.get("government_bond", 0.0))
    row("STATE_MIN_20_PT", state_exp, 0.20, "min")
    real_estate = float(fixed_value_by_type.get("real_estate", 0.0))
    row("REAL_ESTATE_TOTAL_MAX_20_PT", real_estate, 0.20, "max")
    sicar = float(fixed_value_by_type.get("sicar", 0.0))
    row("SICAR_TOTAL_MAX_10_PT", sicar, 0.10, "max")
    for _, eq in opt_values.loc[opt_values["asset_type"].eq("listed_equity")].iterrows():
        row(f"LISTED_EQUITY_PER_COMPANY_MAX_10_PT::{eq['asset_id']}", float(eq["optimized_value"]), 0.10, "max")
    for cid in ["CAPITAL_SOCIAL_LIMITS", "OPCVM_ENTITY_LIMITS", "SICAR_ENTITY_LIMITS", "OTHER_SECURITIES_LIMITS"]:
        row(cid, None, None, "max", "CHECK_NOT_ENFORCED_MISSING_DATA")
    out = pd.DataFrame(rows)
    out = out.merge(regulatory_map[["constraint_id", "constraint_name", "implementation_status", "constraint_origin"]], on="constraint_id", how="left")
    out["constraint_name"] = out["constraint_name"].fillna(out["constraint_id"])
    out["constraint_origin"] = out["constraint_origin"].fillna("LEGAL_REGULATORY")
    return out


def solve_cvxpy_model(
    name: str,
    objective_factory: Callable[[cp.Variable], cp.Expression],
    universe: pd.DataFrame,
    current_weights: np.ndarray,
    context: dict[str, object],
    config: APTOptimizationConfig,
    turnover_required: bool = True,
) -> dict[str, object]:
    """Solve a convex model with turnover threshold relaxation."""

    n = len(universe)
    thresholds = config.turnover_thresholds if turnover_required else (None,)
    last_status = "not_solved"
    for threshold in thresholds:
        w = cp.Variable(n)
        constraints = cvxpy_constraints(w, universe, current_weights, context, config, threshold)
        problem = cp.Problem(cp.Minimize(objective_factory(w)), constraints)
        try:
            problem.solve(solver="CLARABEL", verbose=False)
        except Exception:
            problem.solve(verbose=False)
        last_status = str(problem.status)
        if w.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
            weights = np.asarray(w.value, dtype=float)
            weights = np.maximum(weights, 0.0)
            weights = weights / weights.sum()
            return {
                "portfolio_name": name,
                "weights": weights,
                "success": True,
                "solver_status": last_status,
                "turnover_limit_used": threshold,
                "objective_value": float(problem.value) if problem.value is not None else np.nan,
            }
    return {
        "portfolio_name": name,
        "weights": current_weights.copy(),
        "success": False,
        "solver_status": last_status,
        "turnover_limit_used": np.nan,
        "objective_value": np.nan,
    }


def scipy_common_constraints(universe: pd.DataFrame, current_weights: np.ndarray, context: dict[str, object], config: APTOptimizationConfig, turnover_limit: float | None) -> list[dict[str, object]]:
    """Common SLSQP constraints."""

    masks = class_masks(universe)
    v_opt = float(context["optimisable_value"])
    pt = float(context["technical_provisions"])
    state_min_weight = max(0.0, 0.20 * pt / v_opt)
    constraints = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        {"type": "ineq", "fun": lambda w: float(np.sum(w[masks["government_bond"]]) - state_min_weight)},
        {"type": "ineq", "fun": lambda w: float(config.max_equity_weight - np.sum(w[masks["listed_equity"]]))},
        {"type": "ineq", "fun": lambda w: float(config.max_corporate_weight - np.sum(w[masks["corporate_bond"]]))},
    ]
    if turnover_limit is not None:
        constraints.append({"type": "ineq", "fun": lambda w: float(turnover_limit - np.sum(np.abs(w - current_weights)))})
    return constraints


def solve_slsqp_model(
    name: str,
    objective: Callable[[np.ndarray], float],
    universe: pd.DataFrame,
    current_weights: np.ndarray,
    context: dict[str, object],
    config: APTOptimizationConfig,
    starts: list[np.ndarray],
) -> dict[str, object]:
    """Solve non-convex models with multiple starts and turnover relaxation."""

    upper = upper_bounds_vector(universe, context, config)
    bounds = [(0.0, float(ub)) for ub in upper]
    best = None
    best_threshold = None
    for threshold in config.turnover_thresholds:
        for x0 in starts:
            res = minimize(
                objective,
                x0=np.asarray(x0, dtype=float),
                method="SLSQP",
                bounds=bounds,
                constraints=scipy_common_constraints(universe, current_weights, context, config, threshold),
                options={"maxiter": 3000, "ftol": 1e-11, "disp": False},
            )
            if res.success:
                if best is None or float(res.fun) < float(best.fun):
                    best = res
                    best_threshold = threshold
        if best is not None:
            break
    if best is None:
        return {
            "portfolio_name": name,
            "weights": current_weights.copy(),
            "success": False,
            "solver_status": "SLSQP_FAILED",
            "turnover_limit_used": np.nan,
            "objective_value": np.nan,
        }
    weights = np.maximum(np.asarray(best.x, dtype=float), 0.0)
    weights = weights / weights.sum()
    return {
        "portfolio_name": name,
        "weights": weights,
        "success": True,
        "solver_status": str(best.message),
        "turnover_limit_used": best_threshold,
        "objective_value": float(best.fun),
    }


def feasible_random_starts(universe: pd.DataFrame, current_weights: np.ndarray, context: dict[str, object], config: APTOptimizationConfig, n_starts: int = 60) -> list[np.ndarray]:
    """Generate starts around current and equal-weight portfolios."""

    rng = np.random.default_rng(config.random_seed)
    n = len(current_weights)
    starts = [current_weights.copy()]
    eq = np.ones(n) / n
    starts.append(eq)
    for _ in range(n_starts):
        noise = rng.dirichlet(np.ones(n))
        starts.append(0.70 * current_weights + 0.30 * noise)
    return [s / s.sum() for s in starts]


def solve_all_models(
    mu: pd.Series,
    sigma: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    config: APTOptimizationConfig | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Solve all requested deterministic optimisation models."""

    config = config or APTOptimizationConfig()
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()
    sig = sigma.to_numpy(float)
    mu_v = mu.to_numpy(float)
    turnover_limit = config.primary_turnover_limit
    ret = returns.to_numpy(float)
    rows = []
    portfolios: dict[str, np.ndarray] = {"Current_Portfolio": current}

    def add(result: dict[str, object]) -> None:
        portfolios[str(result["portfolio_name"])] = np.asarray(result["weights"], dtype=float)
        rows.append({k: v for k, v in result.items() if k != "weights"})

    add(solve_cvxpy_model("Minimum_Variance", lambda w: cp.quad_form(w, sig), universe, current, context, config))
    for lam in [2, 5, 10, 20]:
        add(solve_cvxpy_model(f"Mean_Variance_lambda_{lam}", lambda w, lam=lam: (lam / 2.0) * cp.quad_form(w, sig) - mu_v @ w, universe, current, context, config))
    add(solve_cvxpy_model("Max_Return", lambda w: -mu_v @ w, universe, current, context, config))
    add(solve_cvxpy_model("MeanVariance_TurnoverPenalty", lambda w: (5.0 / 2.0) * cp.quad_form(w, sig) - mu_v @ w + 0.05 * cp.norm1(w - current), universe, current, context, config))

    n_obs, n_assets = ret.shape
    def cvar_objective(w: cp.Variable) -> cp.Expression:
        alpha = cp.Variable()
        u = cp.Variable(n_obs, nonneg=True)
        loss = -ret @ w
        return alpha + (1.0 / ((1.0 - config.cvar_beta) * n_obs)) * cp.sum(u) + 0.0 * cp.sum(w)

    # CVaR needs its auxiliary variables inside the problem, so solve explicitly.
    for threshold in config.turnover_thresholds:
        w = cp.Variable(n_assets)
        alpha = cp.Variable()
        u = cp.Variable(n_obs, nonneg=True)
        loss = -ret @ w
        cvar = alpha + (1.0 / ((1.0 - config.cvar_beta) * n_obs)) * cp.sum(u)
        constraints = cvxpy_constraints(w, universe, current, context, config, threshold) + [u >= loss - alpha]
        problem = cp.Problem(cp.Minimize(cvar - 0.10 * (mu_v @ w)), constraints)
        try:
            problem.solve(solver="CLARABEL", verbose=False)
        except Exception:
            problem.solve(verbose=False)
        if w.value is not None and problem.status in {"optimal", "optimal_inaccurate"}:
            weights = np.maximum(np.asarray(w.value, dtype=float), 0.0)
            weights = weights / weights.sum()
            add({"portfolio_name": "Mean_CVaR_95", "weights": weights, "success": True, "solver_status": str(problem.status), "turnover_limit_used": threshold, "objective_value": float(problem.value)})
            break
    else:
        add({"portfolio_name": "Mean_CVaR_95", "weights": current.copy(), "success": False, "solver_status": "CVXPY_FAILED", "turnover_limit_used": np.nan, "objective_value": np.nan})

    starts = feasible_random_starts(universe, current, context, config, n_starts=config.slsqp_random_starts)
    add(solve_slsqp_model("Risk_Parity", lambda w: risk_parity_objective(w, sig), universe, current, context, config, starts))
    return portfolios, pd.DataFrame(rows)


def risk_parity_objective(weights: np.ndarray, sigma_values: np.ndarray) -> float:
    """Equal-risk-contribution objective."""

    w = np.asarray(weights, dtype=float)
    var = float(w.T @ sigma_values @ w)
    if var <= 1e-14:
        return 1e6
    sigma_p = math.sqrt(var)
    rc = w * (sigma_values @ w) / sigma_p
    return float(np.sum((rc - rc.mean()) ** 2))


def portfolio_metrics(
    name: str,
    weights: np.ndarray,
    mu: pd.Series,
    sigma: pd.DataFrame,
    returns: pd.DataFrame,
    rf_annual: float,
    current_weights: np.ndarray,
    universe: pd.DataFrame,
    context: dict[str, object],
    regulatory_map: pd.DataFrame,
    optimization_status: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute APT performance metrics and historical risk diagnostics."""

    w = np.asarray(weights, dtype=float)
    scenario = returns.to_numpy(float) @ w
    expected = float(mu.to_numpy(float) @ w)
    variance = float(w.T @ sigma.to_numpy(float) @ w)
    vol = math.sqrt(max(variance, 0.0))
    var_loss = float(np.quantile(-scenario, 0.95))
    cvar_loss = float((-scenario)[(-scenario) >= var_loss].mean()) if np.any((-scenario) >= var_loss) else var_loss
    wealth = np.cumprod(1.0 + scenario)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    downside = np.minimum(scenario, 0.0)
    compliance = evaluate_constraints(w, universe, context, regulatory_map, name)
    breaches = compliance["compliance_status"].eq("BREACH").sum()
    not_tested = compliance["compliance_status"].eq("CHECK_NOT_ENFORCED_MISSING_DATA").sum()
    masks = class_masks(universe)
    row = {
        "portfolio_name": name,
        "expected_return_APT": expected,
        "volatility_APT": vol,
        "variance_APT": variance,
        "sharpe_ratio": (expected - rf_annual) / vol if vol > 1e-12 else np.nan,
        "var_95_historical": max(0.0, var_loss),
        "cvar_95_historical": max(0.0, cvar_loss),
        "downside_deviation": float(np.sqrt(np.mean(downside ** 2)) * math.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "skewness": float(skew(scenario, bias=False)),
        "kurtosis": float(kurtosis(scenario, bias=False)),
        "turnover": float(np.sum(np.abs(w - current_weights))),
        "herfindahl_index": float(np.sum(w ** 2)),
        "equity_weight": float(np.sum(w[masks["listed_equity"]])),
        "state_weight": float(np.sum(w[masks["government_bond"]])),
        "corporate_weight": float(np.sum(w[masks["corporate_bond"]])),
        "active_constraints_count": int(np.sum(np.isclose(w, APTOptimizationConfig().max_weight_per_asset, atol=1e-4)) + np.sum(np.isclose(w, 0, atol=1e-7))),
        "regulatory_status": "BREACH_DETECTED" if breaches else ("COMPLIANT_WITH_UNTESTED_CONSTRAINTS" if not_tested else "COMPLIANT_TESTABLE_CONSTRAINTS"),
        "number_of_regulatory_breaches": int(breaches),
        "optimization_status": optimization_status,
    }
    if row["cvar_95_historical"] + 1e-12 < row["var_95_historical"]:
        row["optimization_status"] = f"{optimization_status};CVAR_LT_VAR_ERROR"
    if row["cvar_95_historical"] == 0 or row["var_95_historical"] == 0:
        row["optimization_status"] = f"{optimization_status};SUSPICIOUS_ZERO_VAR_CVAR"
    if name == "Max_Return" and (w.max() > 0.25 or row["equity_weight"] > 0.25):
        row["optimization_status"] = f"{row['optimization_status']};AGGRESSIVE_PORTFOLIO"
    return row, compliance


def risk_contributions(name: str, weights: np.ndarray, sigma: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Compute risk contributions that sum to portfolio volatility."""

    w = np.asarray(weights, dtype=float)
    sig = sigma.to_numpy(float)
    variance = float(w.T @ sig @ w)
    if variance <= 1e-14:
        raise ValueError("Variance nulle : contributions au risque non définies.")
    sigma_p = math.sqrt(variance)
    mrc = sig @ w / sigma_p
    rc = w * mrc
    out = universe[["asset_id", "asset_name", "asset_class", "asset_type"]].copy()
    out["portfolio_name"] = name
    out["weight"] = w
    out["marginal_risk_contribution"] = mrc
    out["risk_contribution"] = rc
    out["risk_contribution_pct"] = rc / sigma_p
    return out[["portfolio_name", "asset_id", "asset_name", "asset_class", "asset_type", "weight", "marginal_risk_contribution", "risk_contribution", "risk_contribution_pct"]]



def generate_monte_carlo(
    mu: pd.Series,
    sigma: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    regulatory_map: pd.DataFrame,
    rf_annual: float,
    config: APTOptimizationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Generate random portfolios and return all simulated plus feasible subsets.

    The feasible subset applies the same primary turnover limit as the
    deterministic models. Calculations are vectorized to keep notebook
    execution tractable.
    """

    config = config or APTOptimizationConfig()
    rng = np.random.default_rng(config.random_seed)
    n = len(mu)
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()
    all_rows: list[dict[str, object]] = []
    feasible_rows: list[dict[str, object]] = []
    weights_map: dict[str, np.ndarray] = {}
    attempts = 0
    batch = 25_000
    turnover_limit = config.primary_turnover_limit
    sig = sigma.to_numpy(float)
    mu_v = mu.to_numpy(float)
    ret = returns.to_numpy(float)
    upper = upper_bounds_vector(universe, context, config)
    masks = class_masks(universe)
    state_min = 0.20 * float(context["technical_provisions"]) / float(context["optimisable_value"])
    anchor_var = cp.Variable(n)
    anchor_problem = cp.Problem(
        cp.Minimize(cp.norm1(anchor_var - current)),
        cvxpy_constraints(anchor_var, universe, current, context, config, turnover_limit),
    )
    try:
        anchor_problem.solve(solver="CLARABEL", verbose=False)
    except Exception:
        anchor_problem.solve(verbose=False)
    if anchor_var.value is not None and anchor_problem.status in {"optimal", "optimal_inaccurate"}:
        anchor = np.maximum(np.asarray(anchor_var.value, dtype=float), 0.0)
        anchor = anchor / anchor.sum()
    else:
        anchor = np.minimum(current, upper)
        anchor = anchor / anchor.sum()
    dirichlet_alpha = np.maximum(anchor * 800.0, 1.0)

    while len(feasible_rows) < config.monte_carlo_required and attempts < config.monte_carlo_max_attempts:
        candidates = rng.dirichlet(dirichlet_alpha, size=batch)
        attempts += batch
        turnovers = np.abs(candidates - current).sum(axis=1)
        feasible_mask = (
            np.isclose(candidates.sum(axis=1), 1.0, atol=1e-8)
            & (candidates >= -1e-12).all(axis=1)
            & (candidates <= upper + 1e-12).all(axis=1)
            & (candidates[:, masks["government_bond"]].sum(axis=1) + 1e-12 >= state_min)
            & (candidates[:, masks["listed_equity"]].sum(axis=1) <= config.max_equity_weight + 1e-12)
            & (candidates[:, masks["corporate_bond"]].sum(axis=1) <= config.max_corporate_weight + 1e-12)
            & (turnovers <= turnover_limit + 1e-12)
        )
        scenario = ret @ candidates.T
        expected = candidates @ mu_v
        variances = np.einsum("ij,jk,ik->i", candidates, sig, candidates)
        volatility = np.sqrt(np.maximum(variances, 0.0))
        sharpe = np.where(volatility > 1e-12, (expected - rf_annual) / volatility, np.nan)
        losses = -scenario
        var_95 = np.quantile(losses, 0.95, axis=0)
        cvar_95 = np.array([
            losses[:, i][losses[:, i] >= var_95[i]].mean() if np.any(losses[:, i] >= var_95[i]) else var_95[i]
            for i in range(candidates.shape[0])
        ])
        wealth = np.cumprod(1.0 + scenario, axis=0)
        drawdowns = wealth / np.maximum.accumulate(wealth, axis=0) - 1.0
        max_dd = drawdowns.min(axis=0)

        for i, w in enumerate(candidates):
            pid = len(all_rows)
            feasible = bool(feasible_mask[i])
            row = {
                "portfolio_id": pid,
                "expected_return": float(expected[i]),
                "volatility": float(volatility[i]),
                "variance": float(variances[i]),
                "sharpe": float(sharpe[i]) if np.isfinite(sharpe[i]) else np.nan,
                "var_95": float(max(0.0, var_95[i])),
                "cvar_95": float(max(0.0, cvar_95[i])),
                "max_drawdown": float(max_dd[i]),
                "turnover": float(turnovers[i]),
                "turnover_limit_used": turnover_limit,
                "compliance_status": "COMPLIANT_WITH_UNTESTED_CONSTRAINTS" if feasible else "INFEASIBLE",
                "feasibility_status": "FEASIBLE" if feasible else "INFEASIBLE",
                "feasibility_reason": "OK" if feasible else "INTERNAL_OR_TURNOVER_CONSTRAINT_FAILED",
                "weights_json": json.dumps({asset: float(x) for asset, x in zip(mu.index, w)}, ensure_ascii=False),
            }
            all_rows.append(row)
            if feasible:
                feasible_rows.append(row)
                weights_map[str(pid)] = w.copy()
                if len(feasible_rows) >= config.monte_carlo_required:
                    break

    all_out = pd.DataFrame(all_rows)
    feasible_out = pd.DataFrame(feasible_rows)
    all_out.attrs["attempts"] = attempts
    feasible_out.attrs["attempts"] = attempts
    return all_out, feasible_out, weights_map


def basic_feasible(weights: np.ndarray, universe: pd.DataFrame, current_weights: np.ndarray, context: dict[str, object], config: APTOptimizationConfig, turnover_limit: float | None) -> bool:
    """Fast internal feasibility check."""

    w = np.asarray(weights, dtype=float)
    upper = upper_bounds_vector(universe, context, config)
    if abs(w.sum() - 1) > 1e-6 or (w < -1e-10).any() or (w > upper + 1e-10).any():
        return False
    masks = class_masks(universe)
    v_opt = float(context["optimisable_value"])
    pt = float(context["technical_provisions"])
    if w[masks["government_bond"]].sum() + 1e-10 < 0.20 * pt / v_opt:
        return False
    if w[masks["listed_equity"]].sum() > config.max_equity_weight + 1e-10:
        return False
    if w[masks["corporate_bond"]].sum() > config.max_corporate_weight + 1e-10:
        return False
    if turnover_limit is not None and np.sum(np.abs(w - current_weights)) > turnover_limit + 1e-10:
        return False
    return True


def solve_efficient_frontier(
    mu: pd.Series,
    sigma: pd.DataFrame,
    universe: pd.DataFrame,
    context: dict[str, object],
    rf_annual: float,
    config: APTOptimizationConfig | None = None,
    target_returns: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a constrained efficient frontier with target-return validation."""

    config = config or APTOptimizationConfig()
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()
    sig = sigma.to_numpy(float)
    mu_v = mu.to_numpy(float)
    turnover_limit = config.primary_turnover_limit
    # Build the target grid on the feasible constrained range, not on raw
    # single-asset returns. This keeps the plotted frontier consistent with
    # the same bounds, turnover and regulatory constraints used elsewhere.
    n = len(mu_v)
    w_min = cp.Variable(n)
    min_problem = cp.Problem(
        cp.Minimize(cp.quad_form(w_min, sig)),
        cvxpy_constraints(w_min, universe, current, context, config, turnover_limit),
    )
    try:
        min_problem.solve(solver="CLARABEL", verbose=False)
    except Exception:
        min_problem.solve(verbose=False)
    if w_min.value is not None and min_problem.status in {"optimal", "optimal_inaccurate"}:
        min_weights = np.maximum(np.asarray(w_min.value, dtype=float), 0.0)
        min_weights = min_weights / min_weights.sum()
        min_target = float(mu_v @ min_weights)
    else:
        min_target = float(mu_v @ current)

    w_max = cp.Variable(n)
    max_problem = cp.Problem(
        cp.Maximize(mu_v @ w_max),
        cvxpy_constraints(w_max, universe, current, context, config, turnover_limit),
    )
    try:
        max_problem.solve(solver="CLARABEL", verbose=False)
    except Exception:
        max_problem.solve(verbose=False)
    if w_max.value is not None and max_problem.status in {"optimal", "optimal_inaccurate"}:
        max_weights = np.maximum(np.asarray(w_max.value, dtype=float), 0.0)
        max_weights = max_weights / max_weights.sum()
        max_target = float(mu_v @ max_weights)
    else:
        max_target = float(mu.max())
    if max_target < min_target:
        min_target, max_target = max_target, min_target
    # Cosine spacing gives more points near the low-risk and high-return ends,
    # where constrained frontiers usually bend and where visual interpretation
    # is most sensitive.
    grid = np.linspace(0.0, math.pi, config.frontier_points)
    targets = min_target + (max_target - min_target) * (0.5 - 0.5 * np.cos(grid))
    if target_returns is not None:
        extra_targets = np.asarray(target_returns, dtype=float)
        extra_targets = extra_targets[np.isfinite(extra_targets)]
        extra_targets = extra_targets[(extra_targets >= min_target - 1e-10) & (extra_targets <= max_target + 1e-10)]
        targets = np.unique(np.concatenate([targets, extra_targets]))
    rows = []
    for target in targets:
        w_var = cp.Variable(n)
        constraints = cvxpy_constraints(w_var, universe, current, context, config, turnover_limit)
        constraints.append(mu_v @ w_var >= target)
        problem = cp.Problem(cp.Minimize(cp.quad_form(w_var, sig)), constraints)
        try:
            problem.solve(solver="CLARABEL", verbose=False)
        except Exception:
            problem.solve(verbose=False)
        if w_var.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
            rows.append({
                "target_return": target,
                "achieved_return": np.nan,
                "volatility": np.nan,
                "variance": np.nan,
                "sharpe": np.nan,
                "return_gap": np.nan,
                "turnover_limit_used": turnover_limit,
                "optimization_status": str(problem.status),
                "feasibility_status": "INFEASIBLE",
                "weights_json": "",
            })
            continue
        w = np.maximum(np.asarray(w_var.value, dtype=float), 0.0)
        w = w / w.sum()
        achieved = float(mu_v @ w)
        variance = float(w.T @ sig @ w)
        vol = math.sqrt(max(variance, 0.0))
        gap = max(0.0, target - achieved)
        internal_ok = basic_feasible(w, universe, current, context, config, turnover_limit)
        status = "success" if problem.status in {"optimal", "optimal_inaccurate"} else str(problem.status)
        feasibility = "VALID" if status == "success" and gap <= 1e-4 and internal_ok else "FRONTIER_TARGET_NOT_REACHED"
        rows.append({
            "target_return": target,
            "achieved_return": achieved,
            "volatility": vol,
            "variance": variance,
            "sharpe": (achieved - rf_annual) / vol if vol > 1e-12 else np.nan,
            "return_gap": gap,
            "turnover_limit_used": turnover_limit,
            "optimization_status": status,
            "feasibility_status": feasibility,
            "weights_json": json.dumps({asset: float(x) for asset, x in zip(mu.index, w)}, ensure_ascii=False),
        })
    frontier = pd.DataFrame(rows)
    valid = frontier.loc[frontier["feasibility_status"].eq("VALID")].copy()
    if valid.empty:
        frontier["frontier_point_id"] = np.arange(1, len(frontier) + 1)
        return frontier
    valid["volatility_round"] = valid["volatility"].round(8)
    valid["return_round"] = valid["achieved_return"].round(8)
    valid = valid.sort_values(["volatility", "achieved_return"]).drop_duplicates(["volatility_round", "return_round"], keep="last")
    valid = valid.sort_values("volatility").reset_index(drop=True)
    valid["cummax_return"] = valid["achieved_return"].cummax()
    valid = valid.loc[valid["achieved_return"] >= valid["cummax_return"] - 1e-7].copy()
    valid = valid.drop(columns=["volatility_round", "return_round", "cummax_return"])
    valid["frontier_point_id"] = np.arange(1, len(valid) + 1)
    return valid.reset_index(drop=True)


def optimized_weights_table(portfolios: dict[str, np.ndarray], universe: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    """Long-form optimised weights."""

    rows = []
    v_opt = float(context["optimisable_value"])
    for name, w in portfolios.items():
        part = universe[["asset_id", "asset_name", "asset_class", "asset_type", "issuer", "current_weight_optimisable"]].copy()
        part["portfolio_name"] = name
        part["current_weight"] = part["current_weight_optimisable"]
        part["optimized_weight"] = w
        part["weight_change"] = part["optimized_weight"] - part["current_weight"]
        part["optimized_value"] = part["optimized_weight"] * v_opt
        part["quality_flag"] = np.where(part["optimized_weight"] >= -1e-8, "OK", "NEGATIVE_WEIGHT")
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def build_multiobjective_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    """Score feasible candidates using a transparent normalized multi-criteria rule."""

    df = candidates.copy()
    df = df.loc[~df["regulatory_status"].eq("BREACH_DETECTED")].copy()
    if df.empty:
        return pd.DataFrame()

    def norm_high(s: pd.Series) -> pd.Series:
        den = s.max() - s.min()
        return pd.Series(0.5, index=s.index) if den <= 1e-12 else (s - s.min()) / den

    def norm_low(s: pd.Series) -> pd.Series:
        return 1.0 - norm_high(s)

    df["normalized_return"] = norm_high(df["expected_return_APT"])
    df["normalized_volatility"] = norm_low(df["volatility_APT"])
    df["normalized_cvar"] = norm_low(df["cvar_95_historical"])
    df["normalized_CVaR"] = df["normalized_cvar"]
    df["normalized_turnover"] = norm_low(df["turnover"])
    df["normalized_concentration"] = norm_low(df["herfindahl_index"])
    df["score_return"] = 0.30 * df["normalized_return"]
    df["score_risk"] = 0.25 * df["normalized_volatility"]
    df["score_cvar"] = 0.20 * df["normalized_cvar"]
    df["score_turnover"] = 0.15 * df["normalized_turnover"]
    df["score_concentration"] = 0.10 * df["normalized_concentration"]
    df["score_compliance"] = np.where(df["regulatory_status"].eq("BREACH_DETECTED"), 0.0, 1.0)
    df["final_score"] = df["score_return"] + df["score_risk"] + df["score_cvar"] + df["score_turnover"] + df["score_concentration"]
    df["score_total"] = df["final_score"]
    df["score_method"] = "SCORING_MULTICRITERE_NORMALISE_AIDE_DECISION"
    df = df.sort_values("final_score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    df["model_selected_as_recommendation"] = df["rank"].eq(1)
    return df

def stress_tests_by_portfolio(portfolios: dict[str, np.ndarray], universe: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    """Compute first-order stress diagnostics using duration when available.

    If modified duration is absent but residual maturity exists, a conservative
    proxy ``min(residual_maturity, 7)`` is used and explicitly flagged.
    """

    masks = class_masks(universe)
    duration_candidates = [
        "modified_duration", "duration_modifiee", "duration_modified",
        "duration_years", "duration",
    ]
    maturity_candidates = ["residual_maturity", "maturite_residuelle", "residual_maturity_years"]
    duration_col = next((c for c in duration_candidates if c in universe.columns), None)
    maturity_col = next((c for c in maturity_candidates if c in universe.columns), None)
    rows = []
    v_opt = float(context["optimisable_value"])
    shocks = [
        ("Choc actions -10%", "listed_equity", -0.10, "pct"),
        ("Choc actions -20%", "listed_equity", -0.20, "pct"),
        ("Choc taux souverain +100 bps", "government_bond", 0.01, "bps"),
        ("Choc taux souverain +200 bps", "government_bond", 0.02, "bps"),
        ("Choc spread corporate +100 bps", "corporate_bond", 0.01, "bps"),
        ("Choc spread corporate +200 bps", "corporate_bond", 0.02, "bps"),
    ]

    for pname, w in portfolios.items():
        for scenario_name, atype, shock, shock_kind in shocks:
            mask = masks[atype]
            exposure_weight = float(w[mask].sum())
            exposure_value = exposure_weight * v_opt
            if atype == "listed_equity":
                impact_pct = shock * exposure_weight
                duration_source = "NOT_APPLICABLE_EQUITY_SHOCK"
                flag = "OK"
            else:
                local_weights = np.maximum(w[mask], 0.0)
                if local_weights.sum() <= 1e-12:
                    avg_duration = 0.0
                    duration_source = "NO_EXPOSURE"
                    flag = "OK"
                elif duration_col is not None:
                    durations = pd.to_numeric(universe.loc[mask, duration_col], errors="coerce")
                    avg_duration = float(np.average(durations.fillna(durations.median()).fillna(0.0), weights=np.maximum(local_weights, 1e-12)))
                    duration_source = duration_col
                    flag = "OK"
                elif maturity_col is not None:
                    maturities = pd.to_numeric(universe.loc[mask, maturity_col], errors="coerce").clip(lower=0, upper=7)
                    avg_duration = float(np.average(maturities.fillna(maturities.median()).fillna(0.0), weights=np.maximum(local_weights, 1e-12)))
                    duration_source = f"PROXY_MIN_{maturity_col}_7Y"
                    flag = "DURATION_PROXY_USED"
                else:
                    avg_duration = np.nan
                    duration_source = "MISSING"
                    flag = "DURATION_MISSING_FOR_STRESS_TEST"
                impact_pct = -avg_duration * shock * exposure_weight if np.isfinite(avg_duration) else np.nan
            rows.append({
                "portfolio_name": pname,
                "scenario_name": scenario_name,
                "affected_asset_class": atype,
                "shock_bps_or_pct": shock,
                "exposure_weight": exposure_weight,
                "exposure_value": exposure_value,
                "estimated_portfolio_impact": impact_pct,
                "estimated_impact_pct": impact_pct,
                "estimated_impact_value": impact_pct * v_opt if np.isfinite(impact_pct) else np.nan,
                "duration_source": duration_source,
                "data_quality_flag": flag,
                "quality_flag": flag,
            })
    return pd.DataFrame(rows)

def final_recommendation(multi_scores: pd.DataFrame) -> pd.DataFrame:
    """Return the best multi-objective candidate."""

    if multi_scores.empty:
        return pd.DataFrame([{"portfolio_name": None, "recommendation_status": "NO_FEASIBLE_CANDIDATE"}])
    row = multi_scores.iloc[0].copy()
    row["recommendation_status"] = "SELECTED_BY_MULTICRITERIA_SCORING_NOT_PARETO_SOLVER"
    return pd.DataFrame([row])


def _weights_json(asset_ids: pd.Series, weights: np.ndarray) -> str:
    return json.dumps({str(asset): float(w) for asset, w in zip(asset_ids, weights)}, ensure_ascii=False)


def _concentration(weights: np.ndarray, n: int) -> float:
    w = np.sort(np.asarray(weights, dtype=float))[::-1]
    return float(w[:n].sum())


def _scenario_result_aliases(df: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    """Add the cross-scenario column names requested by the notebook."""

    out = df.copy()
    out["Scenario"] = scenario_id
    out["Model"] = out["portfolio_name"]
    out["Expected_Return"] = out["expected_return_APT"]
    out["Volatility"] = out["volatility_APT"]
    out["Variance"] = out["variance_APT"]
    out["Sharpe"] = out["sharpe_ratio"]
    out["VaR_95"] = out["var_95_historical"]
    out["CVaR_95"] = out["cvar_95_historical"]
    out["Max_Drawdown"] = out["max_drawdown"]
    out["Turnover"] = out["turnover"]
    out["Top_3_Concentration"] = out["weights_array"].map(lambda w: _concentration(w, 3))
    out["Top_5_Concentration"] = out["weights_array"].map(lambda w: _concentration(w, 5))
    out["Regulatory_Status"] = out["regulatory_status"]
    out["Solver_Status"] = out["optimization_status"]
    out["Objective_Status"] = np.where(out["optimization_status"].astype(str).str.contains("FAILED", na=False), "NOT_AVAILABLE", "AVAILABLE")
    out["Weights_JSON"] = out["weights_array"].map(lambda w: _weights_json(out.attrs["asset_ids"], w))
    out.attrs = {}
    ordered = [
        "Scenario", "Model", "Expected_Return", "Volatility", "Variance", "Sharpe",
        "VaR_95", "CVaR_95", "Max_Drawdown", "Turnover", "Top_3_Concentration",
        "Top_5_Concentration", "Regulatory_Status", "Solver_Status", "Objective_Status",
        "Weights_JSON", "portfolio_name", "expected_return_APT", "volatility_APT",
        "variance_APT", "sharpe_ratio", "var_95_historical", "cvar_95_historical",
        "max_drawdown", "turnover", "herfindahl_index", "equity_weight",
        "state_weight", "corporate_weight", "number_of_regulatory_breaches",
        "optimization_status",
    ]
    return out[[c for c in ordered if c in out.columns]]


def _selected_monte_carlo_portfolios(mc: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Select Monte Carlo portfolios by several criteria and document duplicates."""

    rows = []
    selections = [
        ("Monte_Carlo_Max_Return", mc.sort_values("expected_return", ascending=False).iloc[0]),
        ("Monte_Carlo_Min_Volatility", mc.sort_values("volatility", ascending=True).iloc[0]),
        ("Monte_Carlo_Min_CVaR", mc.sort_values("cvar_95", ascending=True).iloc[0]),
    ]
    if not scores.empty and "portfolio_id" in scores.columns:
        selections.append(("Monte_Carlo_Best_Scoring", scores.iloc[0]))
    seen: dict[int, str] = {}
    for label, row in selections:
        item = row.to_dict()
        pid = int(item["portfolio_id"])
        item["selection_label"] = label
        item["selection_note"] = f"Même portefeuille que {seen[pid]}" if pid in seen else "Sélection distincte"
        seen.setdefault(pid, label)
        rows.append(item)
    return pd.DataFrame(rows)


def run_single_apt_scenario(
    scenario_id: str,
    mu: pd.Series,
    data: dict[str, object],
    config: APTOptimizationConfig | None = None,
) -> dict[str, object]:
    """Run deterministic, Monte Carlo, frontier and scoring logic for one APT scenario."""

    config = config or APTOptimizationConfig()
    scenario_data = dict(data)
    scenario_data["mu"] = mu.copy()
    scenario_expected = scenario_data["expected"].copy()
    scenario_expected["expected_return_annualized_final"] = scenario_expected["asset_id"].map(mu)
    scenario_expected["expected_return_annualized"] = scenario_expected["expected_return_annualized_final"]
    scenario_data["expected"] = scenario_expected
    scenario_data["sigma"] = data["sigma"]
    scenario_data["returns"] = data["returns"]

    universe = build_universe(scenario_data)
    context = build_context(scenario_data, universe, data["project_dir"])
    context["rf_annual"] = float(data["rf_annual"])
    regulatory_map = build_regulatory_constraints_map()
    current = universe["current_weight_optimisable"].to_numpy(float)
    current = current / current.sum()

    portfolios, solver_diagnostics = solve_all_models(mu, data["sigma"], data["returns"], universe, context, config)
    metrics_rows = []
    reg_rows = []
    risk_rows = []
    for name, weights in portfolios.items():
        status = "BENCHMARK" if name == "Current_Portfolio" else solver_diagnostics.set_index("portfolio_name")["solver_status"].to_dict().get(name, "MODEL_NOT_SOLVED")
        row, compliance = portfolio_metrics(name, weights, mu, data["sigma"], data["returns"], float(data["rf_annual"]), current, universe, context, regulatory_map, status)
        row["weights_array"] = np.asarray(weights, dtype=float)
        metrics_rows.append(row)
        compliance["Scenario"] = scenario_id
        reg_rows.append(compliance)
        try:
            rc = risk_contributions(name, weights, data["sigma"], universe)
            rc["Scenario"] = scenario_id
            risk_rows.append(rc)
        except Exception:
            pass

    mc_all, mc_feasible, mc_weights_map = generate_monte_carlo(mu, data["sigma"], data["returns"], universe, context, regulatory_map, float(data["rf_annual"]), config)
    mc_for_score = mc_feasible.rename(columns={
        "expected_return": "expected_return_APT",
        "volatility": "volatility_APT",
        "variance": "variance_APT",
        "sharpe": "sharpe_ratio",
        "var_95": "var_95_historical",
        "cvar_95": "cvar_95_historical",
        "compliance_status": "regulatory_status",
    }).copy()
    mc_for_score["portfolio_name"] = "Monte_Carlo_" + mc_for_score["portfolio_id"].astype(str)
    mc_for_score["herfindahl_index"] = mc_for_score["portfolio_id"].astype(str).map(
        lambda pid: float(np.sum(mc_weights_map[pid] ** 2)) if pid in mc_weights_map else np.nan
    )
    mc_scores = build_multiobjective_scores(mc_for_score)
    mc_selected = _selected_monte_carlo_portfolios(mc_feasible, mc_scores)
    for selected in mc_selected.itertuples():
        pid = str(int(selected.portfolio_id))
        if pid not in mc_weights_map:
            continue
        name = str(selected.selection_label)
        weights = mc_weights_map[pid]
        row, compliance = portfolio_metrics(name, weights, mu, data["sigma"], data["returns"], float(data["rf_annual"]), current, universe, context, regulatory_map, "MONTE_CARLO_SELECTED")
        row["weights_array"] = np.asarray(weights, dtype=float)
        row["portfolio_id"] = int(selected.portfolio_id)
        row["selection_note"] = str(selected.selection_note)
        metrics_rows.append(row)
        compliance["Scenario"] = scenario_id
        reg_rows.append(compliance)

    metrics = pd.DataFrame(metrics_rows)
    metrics.attrs["asset_ids"] = universe["asset_id"]
    scores = build_multiobjective_scores(metrics)
    scores.attrs = {}
    if scores.empty:
        recommended_name = "NO_FEASIBLE_CANDIDATE"
    else:
        recommended_name = str(scores.iloc[0]["portfolio_name"])
    metrics["model_selected_as_recommendation"] = metrics["portfolio_name"].eq(recommended_name)
    results = _scenario_result_aliases(metrics, scenario_id)
    weights = optimized_weights_table({row["portfolio_name"]: row["weights_array"] for row in metrics_rows}, universe, context)
    weights["Scenario"] = scenario_id
    frontier_targets = [float(mu.to_numpy(float) @ np.asarray(w, dtype=float)) for w in portfolios.values()]
    frontier = solve_efficient_frontier(mu, data["sigma"], universe, context, float(data["rf_annual"]), config, target_returns=frontier_targets)
    frontier["Scenario"] = scenario_id
    mc_feasible = mc_feasible.copy()
    mc_feasible["Scenario"] = scenario_id
    reg = pd.concat(reg_rows, ignore_index=True) if reg_rows else pd.DataFrame()
    risk = pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame()
    scores = scores.copy()
    if not scores.empty:
        scores["Scenario"] = scenario_id
    return {
        "Scenario": scenario_id,
        "mu": mu,
        "universe": universe,
        "context": context,
        "portfolios": portfolios,
        "solver_diagnostics": solver_diagnostics.assign(Scenario=scenario_id),
        "results": results,
        "weights": weights,
        "monte_carlo": mc_feasible,
        "monte_carlo_selected": mc_selected.assign(Scenario=scenario_id),
        "frontier": frontier,
        "regulatory": reg,
        "risk_contributions": risk,
        "scores": scores,
        "recommended_model": recommended_name,
    }


def _wide_recommended_weights(scenario_runs: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for scenario_id, run in scenario_runs.items():
        weights = run["weights"]
        rec = run["recommended_model"]
        part = weights.loc[weights["portfolio_name"].eq(rec), ["asset_id", "asset_name", "asset_class", "optimized_weight"]].copy()
        part["Scenario"] = scenario_id
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_scenario_weight_stability(scenario_runs: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Compare recommended weights to the central scenario recommendation."""

    long = _wide_recommended_weights(scenario_runs)
    if long.empty or "APT_Central" not in scenario_runs:
        return pd.DataFrame()
    wide = long.pivot_table(index=["asset_id", "asset_name", "asset_class"], columns="Scenario", values="optimized_weight", fill_value=0.0).reset_index()
    central = wide.get("APT_Central", pd.Series(0.0, index=wide.index))
    rows = []
    for scenario_id in sorted(scenario_runs):
        if scenario_id == "APT_Central" or scenario_id not in wide.columns:
            continue
        diff = (wide[scenario_id] - central).abs()
        rows.append(
            {
                "Scenario": scenario_id,
                "distance_L1_to_central": float(diff.sum()),
                "turnover_between_scenarios": float(0.5 * diff.sum()),
                "weight_stability_score": float(max(0.0, 1.0 - 0.5 * diff.sum())),
            }
        )
    asset_sensitivity = wide.copy()
    if {"APT_Prudent", "APT_Optimistic"}.issubset(asset_sensitivity.columns):
        asset_sensitivity["prudent_to_optimistic_change"] = asset_sensitivity["APT_Optimistic"] - asset_sensitivity["APT_Prudent"]
        asset_sensitivity["abs_prudent_to_optimistic_change"] = asset_sensitivity["prudent_to_optimistic_change"].abs()
        asset_sensitivity = asset_sensitivity.sort_values("abs_prudent_to_optimistic_change", ascending=False)
    return pd.DataFrame(rows), asset_sensitivity


def run_apt_scenario_analysis(project_dir: str | Path, config: APTOptimizationConfig | None = None) -> dict[str, object]:
    """Run legacy technical-alias scenario sensitivity analysis for notebook 02."""

    config = config or APTOptimizationConfig()
    data = load_apt_optimization_inputs(project_dir)
    assets = data["mu"].index.astype(str).tolist()
    mu_scenarios, scenario_input = load_apt_mu_scenarios(project_dir, assets)
    sigma = data["sigma"]
    for scenario_id, mu in mu_scenarios.items():
        if list(mu.index) != list(sigma.index) or list(sigma.columns) != list(mu.index):
            raise ValueError(f"Alignement mu/Sigma invalide pour {scenario_id}.")
        if mu.isna().any() or mu.index.duplicated().any():
            raise ValueError(f"Rendements attendus invalides pour {scenario_id}.")

    runs = {
        scenario_id: run_single_apt_scenario(scenario_id, mu, data, config)
        for scenario_id, mu in mu_scenarios.items()
    }
    comparison = pd.concat([run["results"] for run in runs.values()], ignore_index=True)
    recommended = pd.DataFrame(
        [
            {
                "Scenario": scenario_id,
                "Recommended_Model": run["recommended_model"],
                "Comment": "Sélection par scoring multicritère propre au scénario.",
            }
            for scenario_id, run in runs.items()
        ]
    )
    stability, asset_sensitivity = build_scenario_weight_stability(runs)
    recommended_weights = _wide_recommended_weights(runs)
    class_weights = recommended_weights.groupby(["Scenario", "asset_class"], as_index=False)["optimized_weight"].sum() if not recommended_weights.empty else pd.DataFrame()
    scenario_scores = pd.concat([run["scores"] for run in runs.values() if not run["scores"].empty], ignore_index=True)
    recommended_performance_rows = []
    for scenario_id, run in runs.items():
        rec_name = run["recommended_model"]
        rec_row = run["results"].loc[run["results"]["Model"].eq(rec_name)].copy()
        if not rec_row.empty:
            recommended_performance_rows.append(rec_row.iloc[0])
    recommended_performance = pd.DataFrame(recommended_performance_rows)
    if not recommended_weights.empty:
        recommended_weights = recommended_weights.sort_values(["Scenario", "optimized_weight"], ascending=[True, False]).reset_index(drop=True)
    central_rec = recommended.loc[recommended["Scenario"].eq("APT_Central"), "Recommended_Model"]
    if not central_rec.empty:
        global_model = str(central_rec.iloc[0])
        global_reason = "Le scénario central est retenu comme référence institutionnelle, avec lecture de robustesse prudent/optimiste."
    else:
        global_model = str(recommended.iloc[0]["Recommended_Model"])
        global_reason = "Fallback : premier scénario disponible faute de recommandation centrale."
    final_conclusion = pd.DataFrame(
        [
            {
                "Question": "Conclusion de robustesse",
                "Réponse": (
                    "L'utilisation de trois scénarios APT permet de ne pas dépendre d'une estimation unique des rendements attendus. "
                    "Le scénario prudent limite le risque de surestimation, le scénario central constitue la référence, tandis que le scénario optimiste teste une hypothèse de marché favorable. "
                    "La décision finale doit privilégier une allocation robuste, conforme et institutionnellement défendable, plutôt qu'une allocation maximisant uniquement le rendement dans le scénario le plus favorable."
                ),
            },
            {"Question": "Recommended_Model_Global", "Réponse": global_model},
            {"Question": "Reason", "Réponse": global_reason},
        ]
    )
    return {
        "APT_Scenarios_Input": scenario_input.reset_index(drop=True),
        "scenario_runs": runs,
        "Results_APT_Prudent": runs["APT_Prudent"]["results"],
        "Results_APT_Central": runs["APT_Central"]["results"],
        "Results_APT_Optimistic": runs["APT_Optimistic"]["results"],
        "Weights_APT_Prudent": runs["APT_Prudent"]["weights"],
        "Weights_APT_Central": runs["APT_Central"]["weights"],
        "Weights_APT_Optimistic": runs["APT_Optimistic"]["weights"],
        "Monte_Carlo_APT_Prudent": runs["APT_Prudent"]["monte_carlo"],
        "Monte_Carlo_APT_Central": runs["APT_Central"]["monte_carlo"],
        "Monte_Carlo_APT_Optimistic": runs["APT_Optimistic"]["monte_carlo"],
        "Scenario_Comparison": comparison,
        "Scenario_Recommended_Models": recommended.assign(Recommended_Model_Global=global_model),
        "Scenario_Recommended_Weights": recommended_weights,
        "Scenario_Recommended_Performance": recommended_performance,
        "Scenario_Weight_Stability": stability,
        "Scenario_Asset_Sensitivity": asset_sensitivity,
        "Scenario_Class_Weights": class_weights,
        "Scenario_Regulatory_Checks": pd.concat([run["regulatory"] for run in runs.values()], ignore_index=True),
        "Scenario_Scores": scenario_scores,
        "Scenario_Efficient_Frontiers": pd.concat([run["frontier"] for run in runs.values()], ignore_index=True),
        "Scenario_Final_Conclusion": final_conclusion,
    }


def export_apt_scenario_sheets(result: dict[str, object], output_path: str | Path) -> Path:
    """Append APT scenario sheets to the notebook 02 Excel workbook."""

    path = Path(output_path)
    mode = "a" if path.exists() else "w"
    sheets = [
        "APT_Scenarios_Input",
        "Results_APT_Prudent",
        "Results_APT_Central",
        "Results_APT_Optimistic",
        "Weights_APT_Prudent",
        "Weights_APT_Central",
        "Weights_APT_Optimistic",
        "Monte_Carlo_APT_Prudent",
        "Monte_Carlo_APT_Central",
        "Monte_Carlo_APT_Optimistic",
        "Scenario_Comparison",
        "Scenario_Recommended_Models",
        "Scenario_Recommended_Weights",
        "Scenario_Recommended_Performance",
        "Scenario_Weight_Stability",
        "Scenario_Asset_Sensitivity",
        "Scenario_Class_Weights",
        "Scenario_Regulatory_Checks",
        "Scenario_Final_Conclusion",
    ]
    writer_kwargs = {"engine": "openpyxl", "mode": mode}
    if mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"
    with pd.ExcelWriter(path, **writer_kwargs) as writer:
        for sheet in sheets:
            df = result[sheet].copy()
            if sheet.startswith("Monte_Carlo") and len(df) > 20_000:
                df = df.head(20_000)
            if "weights_array" in df.columns:
                df = df.drop(columns=["weights_array"])
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return path


def references_table() -> pd.DataFrame:
    """Return optimisation references table."""

    return pd.DataFrame(REFERENCE_ROWS, columns=["reference", "bibliographic_note", "role"])
