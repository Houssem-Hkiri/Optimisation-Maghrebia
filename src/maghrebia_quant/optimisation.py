"""Portfolio optimisation utilities for notebook 02.

The functions in this module consume the audited outputs from notebook 01 and
build optimisation results under no-short-selling, internal limits, and
regulatory checks on the total portfolio.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew


REQUIRED_NOTEBOOK01_SHEETS = [
    "Expected_Returns_Final",
    "Covariance_LW_Annual",
    "Correlation",
    "Asset_Metrics",
    "Portfolio_Metrics",
    "Portfolio_Recon",
    "Optimisable_Pocket",
    "Non_Optimisable",
    "Portfolio_Summary",
    "Returns_Model",
    "Risk_Free_Daily_2025",
    "Final_Control",
]


@dataclass(frozen=True)
class OptimisationParameters:
    """Documented parameters used by the optimisation notebook."""

    max_weight_per_asset: float = 0.30
    lambda_risk: float = 5.0
    lambda_cvar: float = 3.0
    cvar_alpha: float = 0.95
    monte_carlo_requested: int = 15_000
    monte_carlo_max_attempts: int = 200_000
    frontier_points: int = 50
    random_seed: int = 202505


def _bound_and_normalize_weights(weights: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    """Apply no-short-selling and upper bounds."""

    bounded = np.minimum(np.maximum(np.asarray(weights, dtype=float), 0.0), np.asarray(upper_bounds, dtype=float))
    total = float(bounded.sum())
    if total <= 1e-12:
        raise RuntimeError("Somme des poids nulle apres bornage.")
    return bounded / total


def load_notebook01_outputs(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the required workbook exported by notebook 01.

    Parameters
    ----------
    path:
        Path to ``diagnostic_pre_optimisation_2025.xlsx``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        DataFrames keyed by sheet name.

    Raises
    ------
    FileNotFoundError
        If the workbook does not exist.
    ValueError
        If a required sheet is missing.
    """

    workbook = Path(path)
    if not workbook.exists():
        raise FileNotFoundError(f"Export notebook 01 introuvable : {workbook}")
    xl = pd.ExcelFile(workbook)
    missing = [sheet for sheet in REQUIRED_NOTEBOOK01_SHEETS if sheet not in xl.sheet_names]
    if missing:
        raise ValueError(f"Feuilles notebook 01 manquantes : {missing}")
    return {sheet: pd.read_excel(workbook, sheet_name=sheet) for sheet in REQUIRED_NOTEBOOK01_SHEETS}


def matrix_from_excel_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Convert an Excel-exported matrix with an ``asset_id`` column to a square DataFrame."""

    if "asset_id" not in df.columns:
        raise ValueError("La matrice exportée doit contenir une colonne asset_id.")
    out = df.set_index("asset_id")
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out.apply(pd.to_numeric, errors="coerce")


def extract_technical_provisions(portfolio_path: str | Path) -> float:
    """Extract technical provisions from the source portfolio workbook."""

    path = Path(portfolio_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier portefeuille introuvable : {path}")
    df = pd.read_excel(path, sheet_name="Principal")
    label_col = "Désignation des actifs"
    value_col = "Coût d'entrée au bilan"
    if label_col not in df.columns or value_col not in df.columns:
        raise ValueError("Colonnes portefeuille attendues absentes pour les provisions techniques.")
    mask = df[label_col].astype(str).str.contains("Montant des Provisions Techniques", case=False, na=False)
    if not mask.any():
        raise ValueError("Ligne 'Montant des Provisions Techniques' introuvable.")
    value = pd.to_numeric(df.loc[mask, value_col], errors="coerce").dropna()
    if value.empty or float(value.iloc[0]) <= 0:
        raise ValueError("Montant de provisions techniques invalide.")
    return float(value.iloc[0])


def load_regulatory_constraints_from_docx(path: str | Path) -> dict[str, object]:
    """Read a DOCX regulatory reference without requiring python-docx."""

    docx_path = Path(path)
    if not docx_path.exists():
        raise FileNotFoundError(f"Référentiel réglementaire introuvable : {docx_path}")
    with zipfile.ZipFile(docx_path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    texts = [node.text for node in root.findall(".//w:t", ns) if node.text]
    content = "\n".join(texts)
    return {
        "path": str(docx_path),
        "loaded": True,
        "characters": len(content),
        "text_excerpt": content[:2000],
    }


def build_regulatory_constraints_map(docx_info: dict[str, object]) -> pd.DataFrame:
    """Build the regulatory constraints table documented from the loaded DOCX."""

    source = f"Referentiel reglementaire.docx ({docx_info.get('path', '')})"
    rows = [
        ("COVERAGE_GLOBAL", "Couverture globale des provisions techniques", "portfolio_total", "min", 1.00, "technical_provisions", "all_assets", True, "technical_provisions,total_assets", "ENFORCED", source),
        ("STATE_MIN_20_PT", "Titres émis ou garantis par l'État >= 20% PT", "portfolio_total", "min", 0.20, "technical_provisions", "government_bond", True, "asset_class_mapping,technical_provisions", "ENFORCED", source),
        ("REAL_ESTATE_TOTAL_MAX_20_PT", "Placements immobiliers <= 20% PT", "portfolio_total", "max", 0.20, "technical_provisions", "real_estate", True, "asset_class_mapping,technical_provisions", "POST_CHECK", source),
        ("REAL_ESTATE_SINGLE_MAX_10_PT", "Immeuble déterminé <= 10% PT hors siège", "portfolio_total", "max", 0.10, "technical_provisions", "real_estate_single", True, "building_level_detail,technical_provisions", "POST_CHECK", source),
        ("LISTED_EQUITY_PER_COMPANY_MAX_10_PT", "Action cotée BVMT par société <= 10% PT", "portfolio_total", "max", 0.10, "technical_provisions", "listed_equity_per_company", True, "issuer_mapping,technical_provisions", "ENFORCED", source),
        ("LISTED_EQUITY_CAPITAL_SOCIAL_MAX_30", "Action cotée <= 30% du capital social", "portfolio_total", "max", 0.30, "capital_social", "listed_equity_per_company", True, "capital_social", "CHECK_NOT_ENFORCED_MISSING_DATA", source),
        ("OPCVM_PER_ENTITY_MAX_10_PT", "OPCVM par entité <= 10% PT", "portfolio_total", "max", 0.10, "technical_provisions", "fund_per_entity", True, "fund_entity_detail,technical_provisions", "CHECK_NOT_ENFORCED_MISSING_DATA", source),
        ("OPCVM_CAPITAL_SOCIAL_MAX_30", "OPCVM <= 30% du capital social si disponible", "portfolio_total", "max", 0.30, "capital_social", "fund_per_entity", True, "capital_social", "CHECK_NOT_ENFORCED_MISSING_DATA", source),
        ("SICAR_PER_COMPANY_MAX_5_PT", "SICAR/SICAF par société <= 5% PT", "portfolio_total", "max", 0.05, "technical_provisions", "sicar_per_company", True, "sicar_entity_detail,technical_provisions", "CHECK_NOT_ENFORCED_MISSING_DATA", source),
        ("SICAR_TOTAL_MAX_10_PT", "SICAR/SICAF total <= 10% PT", "portfolio_total", "max", 0.10, "technical_provisions", "sicar_total", True, "asset_class_mapping,technical_provisions", "POST_CHECK", source),
        ("OTHER_SECURITIES_PER_ORG_MAX_5_PT", "Autres actions et valeurs mobilières par organisme <= 5% PT", "portfolio_total", "max", 0.05, "technical_provisions", "other_securities_per_org", True, "organism_mapping", "CHECK_NOT_ENFORCED_MISSING_DATA", source),
        ("OTHER_SECURITIES_TOTAL_MAX_20_PT", "Autres actions et valeurs mobilières total <= 20% PT", "portfolio_total", "max", 0.20, "technical_provisions", "other_securities_total", True, "category_mapping", "CHECK_NOT_ENFORCED_MISSING_DATA", source),
        ("ARTICLE31_CATEGORY_2_4_5_8_9_MAX_50_PT", "Catégories 2,4,5,8,9 article 31 <= 50% PT", "portfolio_total", "max", 0.50, "technical_provisions", "article31_categories", True, "article31_category_mapping", "CHECK_NOT_ENFORCED_MISSING_DATA", source),
        ("DAC_NON_LIFE_MAX_22_UPR", "Frais d'acquisition reportés non-vie <= 22% PPNA", "portfolio_total", "max", 0.22, "provision_primes_non_acquises", "deferred_acquisition_costs", True, "non_life_upr", "CHECK_NOT_ENFORCED_MISSING_DATA", source),
        ("PREMIUM_RECEIVABLES_MAX_10_NET_PREMIUMS", "Quittances non encaissées <= 10% primes nettes et <= 3 mois", "portfolio_total", "max", 0.10, "net_premiums", "premium_receivables", True, "net_premiums,receivable_age", "CHECK_NOT_ENFORCED_MISSING_DATA", source),
    ]
    return pd.DataFrame(rows, columns=[
        "constraint_id",
        "constraint_name",
        "scope",
        "threshold_type",
        "threshold_value",
        "denominator",
        "applies_to",
        "hard_constraint",
        "data_required",
        "implementation_status",
        "source_reference",
    ])


def build_optimization_universe(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the optimisable asset universe from notebook 01 outputs."""

    expected = outputs["Expected_Returns_Final"].copy()
    opt = outputs["Optimisable_Pocket"].copy()
    metrics = outputs["Asset_Metrics"].copy()
    keep_cols = [
        "asset_id",
        "asset_name",
        "asset_class_standardized",
        "asset_type",
        "market_value",
        "portfolio_weight",
        "optimisable_weight",
        "sector",
        "isin",
    ]
    opt = opt[[c for c in keep_cols if c in opt.columns]].copy()
    out = opt.merge(
        expected[[
            "asset_id",
            "expected_return_annualized_final",
            "current_weight_optimisable",
            "quality_flag",
        ]],
        on="asset_id",
        how="inner",
        suffixes=("", "_expected"),
    )
    vol = metrics[["asset_id", "annualized_volatility"]].copy() if "annualized_volatility" in metrics.columns else pd.DataFrame(columns=["asset_id", "annualized_volatility"])
    out = out.merge(vol, on="asset_id", how="left")
    out["issuer"] = out.get("issuer", out["asset_id"])
    out["asset_class"] = out["asset_class_standardized"]
    out = out.rename(
        columns={
            "market_value": "current_value",
            "portfolio_weight": "current_weight_total",
            "expected_return_annualized_final": "expected_return_annualized",
            "annualized_volatility": "volatility_annualized",
        }
    )
    out["optimisable_flag"] = out["asset_type"].isin(["listed_equity", "government_bond", "corporate_bond"])
    out = out.loc[out["optimisable_flag"]].copy()
    return out[[
        "asset_id",
        "asset_name",
        "asset_class",
        "asset_type",
        "issuer",
        "sector",
        "current_value",
        "current_weight_optimisable",
        "current_weight_total",
        "expected_return_annualized",
        "volatility_annualized",
        "optimisable_flag",
        "quality_flag",
    ]]


def prepare_optimization_inputs(outputs: dict[str, pd.DataFrame], universe: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align expected returns, covariance and scenario returns."""

    asset_ids = universe["asset_id"].astype(str).tolist()
    mu = universe.set_index("asset_id")["expected_return_annualized"].astype(float).reindex(asset_ids)
    sigma = matrix_from_excel_sheet(outputs["Covariance_LW_Annual"]).reindex(index=asset_ids, columns=asset_ids)
    returns = outputs["Returns_Model"].copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.set_index("date").reindex(columns=asset_ids).apply(pd.to_numeric, errors="coerce")
    check = pd.DataFrame([
        {
            "check": "mu_available",
            "status": bool(mu.notna().all()),
            "detail": f"{int(mu.notna().sum())}/{len(mu)} rendements espérés disponibles",
        },
        {
            "check": "sigma_available",
            "status": bool(not sigma.isna().any().any()),
            "detail": f"Matrice {sigma.shape}",
        },
        {
            "check": "returns_available",
            "status": bool(not returns.isna().any().any()),
            "detail": f"{returns.shape[0]} scénarios x {returns.shape[1]} actifs",
        },
    ])
    if mu.isna().any() or sigma.isna().any().any() or returns.isna().any().any():
        raise ValueError("Inputs financiers incomplets pour l'optimisation.")
    sigma = (sigma + sigma.T) / 2
    eig = np.linalg.eigvalsh(sigma.to_numpy(float))
    if eig.min() < -1e-10:
        sigma = nearest_psd(sigma)
        check.loc[len(check)] = ["sigma_psd_repair", True, "SIGMA_REPAIRED_TO_PSD"]
    else:
        check.loc[len(check)] = ["sigma_psd", True, f"min eigenvalue={eig.min():.3e}"]
    return mu, sigma, returns, check


def nearest_psd(sigma: pd.DataFrame, eps: float = 1e-10) -> pd.DataFrame:
    """Project a symmetric matrix to PSD by clipping eigenvalues."""

    values = ((sigma + sigma.T) / 2).to_numpy(float)
    eigvals, eigvecs = np.linalg.eigh(values)
    eigvals = np.maximum(eigvals, eps)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return pd.DataFrame((out + out.T) / 2, index=sigma.index, columns=sigma.columns)


def build_internal_constraints_map(params: OptimisationParameters) -> pd.DataFrame:
    """Return documented internal constraints."""

    return pd.DataFrame([
        {"constraint_id": "BUDGET", "description": "Somme des poids de la poche optimisable = 1", "value": 1.0, "hard_constraint": True},
        {"constraint_id": "NO_SHORT_SELLING", "description": "Aucune vente à découvert", "value": 0.0, "hard_constraint": True},
        {"constraint_id": "MAX_WEIGHT_PER_ASSET", "description": "Poids maximal par actif dans la poche optimisable", "value": params.max_weight_per_asset, "hard_constraint": True},
    ])


def build_context(
    outputs: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    technical_provisions: float,
) -> dict[str, object]:
    """Build portfolio context used by regulatory checks and optimisation constraints."""

    summary = outputs["Portfolio_Summary"].iloc[0].to_dict()
    fixed = outputs["Non_Optimisable"].copy()
    for col in ["market_value", "asset_type", "asset_class_standardized", "asset_id", "asset_name"]:
        if col not in fixed.columns:
            fixed[col] = np.nan
    return {
        "technical_provisions": float(technical_provisions),
        "total_value": float(summary["total_portfolio_value"]),
        "optimisable_value": float(summary["optimisable_value"]),
        "fixed_value": float(summary["non_optimisable_value"]),
        "universe": universe.copy(),
        "fixed": fixed,
    }


def regulatory_linear_limits(context: dict[str, object], constraints_map: pd.DataFrame, params: OptimisationParameters) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Build per-asset upper bounds and group constraints from testable rules."""

    universe = context["universe"].copy()
    v_opt = float(context["optimisable_value"])
    pt = float(context["technical_provisions"])
    upper = np.full(len(universe), params.max_weight_per_asset, dtype=float)
    equity_limit = 0.10 * pt / v_opt
    for i, row in universe.reset_index(drop=True).iterrows():
        if row["asset_type"] == "listed_equity":
            upper[i] = min(upper[i], equity_limit)
    state_min = 0.20 * pt / v_opt
    group_constraints = [
        {
            "constraint_id": "STATE_MIN_20_PT",
            "asset_type": "government_bond",
            "sense": ">=",
            "rhs": state_min,
            "description": "Poids minimal de titres d'État dans la poche optimisable pour respecter 20% des PT sur portefeuille total.",
        }
    ]
    return upper, group_constraints


def _total_exposures(weights: np.ndarray, context: dict[str, object]) -> pd.DataFrame:
    universe = context["universe"].copy().reset_index(drop=True)
    fixed = context["fixed"].copy()
    v_opt = float(context["optimisable_value"])
    universe["optimized_value"] = np.asarray(weights, dtype=float) * v_opt
    opt_exp = universe[["asset_id", "asset_name", "asset_type", "asset_class", "optimized_value"]].rename(columns={"optimized_value": "exposure_value"})
    fixed_exp = fixed[["asset_id", "asset_name", "asset_type", "asset_class_standardized", "market_value"]].rename(columns={"asset_class_standardized": "asset_class", "market_value": "exposure_value"})
    fixed_exp["asset_id"] = fixed_exp["asset_id"].astype(str)
    return pd.concat([opt_exp, fixed_exp], ignore_index=True)


def check_regulatory_compliance(
    weights: np.ndarray,
    context: dict[str, object],
    regulatory_map: pd.DataFrame,
    portfolio_name: str,
) -> pd.DataFrame:
    """Check regulatory constraints on the total portfolio."""

    exposures = _total_exposures(weights, context)
    pt = float(context["technical_provisions"])
    total_value = float(context["total_value"])
    rows: list[dict[str, object]] = []

    def add_row(cid: str, exposure: float | None, denom: float | None, status_override: str | None = None, quality: str = "OK") -> None:
        meta = regulatory_map.loc[regulatory_map["constraint_id"].eq(cid)].iloc[0]
        threshold = float(meta["threshold_value"]) * denom if denom is not None and pd.notna(denom) else np.nan
        if status_override:
            status = status_override
            breach = np.nan
        else:
            if exposure is None or denom is None or pd.isna(exposure) or pd.isna(threshold):
                status = "CHECK_NOT_ENFORCED_MISSING_DATA"
                breach = np.nan
            elif meta["threshold_type"] == "min":
                breach = max(0.0, threshold - float(exposure))
                status = "COMPLIANT" if breach <= 1e-6 else "BREACH"
            else:
                breach = max(0.0, float(exposure) - threshold)
                status = "COMPLIANT" if breach <= 1e-6 else "BREACH"
        rows.append({
            "portfolio_name": portfolio_name,
            "constraint_id": cid,
            "constraint_name": meta["constraint_name"],
            "exposure_value": np.nan if exposure is None else float(exposure),
            "threshold_value": threshold,
            "denominator_value": denom,
            "exposure_pct_denominator": (float(exposure) / denom) if exposure is not None and denom not in (None, 0) and pd.notna(denom) else np.nan,
            "limit_pct": float(meta["threshold_value"]),
            "compliance_status": status,
            "breach_amount": breach,
            "breach_pct": (breach / denom) if denom not in (None, 0) and pd.notna(breach) else np.nan,
            "data_quality_flag": quality if status != "CHECK_NOT_ENFORCED_MISSING_DATA" else "CHECK_NOT_ENFORCED_MISSING_DATA",
        })

    state = exposures.loc[exposures["asset_type"].eq("government_bond"), "exposure_value"].sum()
    real_estate = exposures.loc[exposures["asset_type"].eq("real_estate"), "exposure_value"].sum()
    sicar = exposures.loc[exposures["asset_type"].eq("sicar"), "exposure_value"].sum()
    add_row("COVERAGE_GLOBAL", total_value, pt)
    add_row("STATE_MIN_20_PT", state, pt)
    add_row("REAL_ESTATE_TOTAL_MAX_20_PT", real_estate, pt)
    max_real_estate_single = exposures.loc[exposures["asset_type"].eq("real_estate"), "exposure_value"].max()
    add_row("REAL_ESTATE_SINGLE_MAX_10_PT", max_real_estate_single if pd.notna(max_real_estate_single) else 0.0, pt)
    for _, equity in exposures.loc[exposures["asset_type"].eq("listed_equity")].iterrows():
        add_row("LISTED_EQUITY_PER_COMPANY_MAX_10_PT", float(equity["exposure_value"]), pt)
        rows[-1]["constraint_id"] = f"LISTED_EQUITY_PER_COMPANY_MAX_10_PT::{equity['asset_id']}"
        rows[-1]["constraint_name"] = f"Action cotée BVMT <= 10% PT - {equity['asset_name']}"
    add_row("SICAR_TOTAL_MAX_10_PT", sicar, pt)
    for cid in regulatory_map.loc[regulatory_map["implementation_status"].eq("CHECK_NOT_ENFORCED_MISSING_DATA"), "constraint_id"]:
        add_row(cid, None, None, "CHECK_NOT_ENFORCED_MISSING_DATA")
    return pd.DataFrame(rows)


def scipy_constraints(upper_bounds: np.ndarray, universe: pd.DataFrame, group_constraints: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build SLSQP constraints."""

    constraints: list[dict[str, object]] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for gc in group_constraints:
        mask = universe["asset_type"].eq(gc["asset_type"]).to_numpy(float)
        rhs = float(gc["rhs"])
        constraints.append({"type": "ineq", "fun": lambda w, mask=mask, rhs=rhs: float(mask @ w - rhs)})
    return constraints


def _bounds(upper_bounds: np.ndarray) -> list[tuple[float, float]]:
    return [(0.0, float(ub)) for ub in upper_bounds]


def _feasible_start(upper_bounds: np.ndarray, universe: pd.DataFrame, group_constraints: list[dict[str, object]], current_weights: np.ndarray | None = None) -> np.ndarray:
    n = len(upper_bounds)
    if current_weights is not None:
        x = np.minimum(np.maximum(current_weights.astype(float), 0.0), upper_bounds)
        if x.sum() > 0:
            x = x / x.sum()
            if np.all(x <= upper_bounds + 1e-10) and _group_feasible(x, universe, group_constraints):
                return x
    x = np.minimum(np.ones(n) / n, upper_bounds)
    state_idx = np.where(universe["asset_type"].eq("government_bond").to_numpy())[0]
    for gc in group_constraints:
        if gc["asset_type"] == "government_bond" and state_idx.size:
            need = max(0.0, float(gc["rhs"]) - x[state_idx].sum())
            room = upper_bounds[state_idx] - x[state_idx]
            if need > 0 and room.sum() > need:
                x[state_idx] += need * room / room.sum()
                non_state = np.setdiff1d(np.arange(n), state_idx)
                x[non_state] *= (1 - x[state_idx].sum()) / max(x[non_state].sum(), 1e-12)
    x = np.minimum(x, upper_bounds)
    return x / x.sum()


def _group_feasible(w: np.ndarray, universe: pd.DataFrame, group_constraints: list[dict[str, object]]) -> bool:
    for gc in group_constraints:
        mask = universe["asset_type"].eq(gc["asset_type"]).to_numpy(float)
        if mask @ w < float(gc["rhs"]) - 1e-8:
            return False
    return True


def solve_slsqp_model(
    name: str,
    objective,
    mu: pd.Series,
    sigma: pd.DataFrame,
    universe: pd.DataFrame,
    upper_bounds: np.ndarray,
    group_constraints: list[dict[str, object]],
    current_weights: np.ndarray,
) -> dict[str, object]:
    """Solve a constrained SLSQP optimisation model."""

    x0 = _feasible_start(upper_bounds, universe, group_constraints, current_weights)
    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=_bounds(upper_bounds),
        constraints=scipy_constraints(upper_bounds, universe, group_constraints),
        options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
    )
    w = _bound_and_normalize_weights(result.x if result.success else x0, upper_bounds)
    return {
        "portfolio_name": name,
        "weights": w,
        "success": bool(result.success),
        "solver_status": str(result.message),
        "objective_value": float(result.fun) if np.isfinite(result.fun) else np.nan,
    }


def solve_mean_cvar(
    mu: pd.Series,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    upper_bounds: np.ndarray,
    group_constraints: list[dict[str, object]],
    params: OptimisationParameters,
) -> dict[str, object]:
    """Solve Mean-CVaR with the Rockafellar-Uryasev formulation."""

    r = returns.to_numpy(float)
    n_obs, n_assets = r.shape
    w = cp.Variable(n_assets)
    eta = cp.Variable()
    u = cp.Variable(n_obs, nonneg=True)
    losses = -r @ w
    cvar_daily = eta + (1.0 / ((1.0 - params.cvar_alpha) * n_obs)) * cp.sum(u)
    constraints = [cp.sum(w) == 1, w >= 0, w <= upper_bounds, u >= losses - eta]
    for gc in group_constraints:
        mask = universe["asset_type"].eq(gc["asset_type"]).to_numpy(float)
        constraints.append(mask @ w >= float(gc["rhs"]))
    objective = cp.Minimize(params.lambda_cvar * cvar_daily * math.sqrt(252) - mu.to_numpy(float) @ w)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL", verbose=False)
    except Exception:
        problem.solve(verbose=False)
    if w.value is None:
        raise RuntimeError("Mean-CVaR n'a pas convergé.")
    weights = _bound_and_normalize_weights(w.value, upper_bounds)
    return {
        "portfolio_name": "Mean_CVaR_95",
        "weights": weights,
        "success": problem.status in {"optimal", "optimal_inaccurate"},
        "solver_status": str(problem.status),
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
    }


def compute_portfolio_metrics_from_weights(
    name: str,
    weights: np.ndarray,
    mu: pd.Series,
    sigma: pd.DataFrame,
    returns: pd.DataFrame,
    rf_annual: float,
    current_weights: np.ndarray,
    context: dict[str, object],
    regulatory_map: pd.DataFrame,
    optimization_status: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute performance and regulatory metrics for a portfolio."""

    w = np.asarray(weights, dtype=float)
    scenario = returns.to_numpy(float) @ w
    expected_return = float(mu.to_numpy(float) @ w)
    variance = float(w.T @ sigma.to_numpy(float) @ w)
    volatility = math.sqrt(max(variance, 0.0))
    sharpe = (expected_return - rf_annual) / volatility if volatility > 1e-12 else np.nan
    var_ret = float(np.quantile(scenario, 0.05))
    cvar_ret = float(scenario[scenario <= var_ret].mean()) if np.any(scenario <= var_ret) else var_ret
    wealth = np.cumprod(1.0 + scenario)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    downside = np.minimum(scenario, 0.0)
    compliance = check_regulatory_compliance(w, context, regulatory_map, name)
    breaches = compliance.loc[compliance["compliance_status"].eq("BREACH")]
    not_tested = compliance.loc[compliance["compliance_status"].eq("CHECK_NOT_ENFORCED_MISSING_DATA")]
    status = "BREACH" if not breaches.empty else ("COMPLIANT_WITH_NOT_TESTED" if not not_tested.empty else "COMPLIANT")
    universe = context["universe"].reset_index(drop=True)
    v_opt = float(context["optimisable_value"])
    pt = float(context["technical_provisions"])
    row = {
        "portfolio_name": name,
        "expected_return_annualized": expected_return,
        "historical_return_annualized": float((np.prod(1 + scenario) ** (252 / max(1, len(scenario)))) - 1),
        "volatility_annualized": volatility,
        "variance_annualized": variance,
        "sharpe": sharpe,
        "var_95": max(0.0, -var_ret),
        "cvar_95": max(0.0, -cvar_ret),
        "downside_deviation": float(np.sqrt(np.mean(downside**2)) * math.sqrt(252)),
        "max_drawdown": float(np.min(drawdown)),
        "skewness": float(skew(scenario, bias=False)),
        "kurtosis": float(kurtosis(scenario, bias=False)),
        "turnover_vs_current": float(np.abs(w - current_weights).sum() / 2),
        "state_exposure_pct_PT": float((w[universe["asset_type"].eq("government_bond").to_numpy()] @ np.full(np.sum(universe["asset_type"].eq("government_bond")), v_opt)) / pt),
        "equity_exposure_pct_PT": float((w[universe["asset_type"].eq("listed_equity").to_numpy()] @ np.full(np.sum(universe["asset_type"].eq("listed_equity")), v_opt)) / pt),
        "corporate_exposure_pct_PT": float((w[universe["asset_type"].eq("corporate_bond").to_numpy()] @ np.full(np.sum(universe["asset_type"].eq("corporate_bond")), v_opt)) / pt),
        "compliance_status": status,
        "number_of_breaches": int(len(breaches)),
        "optimization_status": optimization_status,
    }
    return row, compliance


def risk_contributions_for_portfolio(name: str, weights: np.ndarray, sigma: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage contribution to portfolio variance."""

    w = np.asarray(weights, dtype=float)
    sig = sigma.to_numpy(float)
    variance = float(w.T @ sig @ w)
    if variance <= 1e-14:
        raise ValueError("Variance de portefeuille nulle : contribution au risque impossible.")
    marginal = sig @ w
    contrib = w * marginal / variance
    out = universe[["asset_id", "asset_name", "asset_class"]].copy()
    out["portfolio_name"] = name
    out["weight"] = w
    out["marginal_risk_contribution"] = marginal
    out["risk_contribution"] = contrib
    out["risk_contribution_pct"] = contrib
    return out[["portfolio_name", "asset_id", "asset_name", "asset_class", "weight", "marginal_risk_contribution", "risk_contribution", "risk_contribution_pct"]]


def generate_monte_carlo(
    mu: pd.Series,
    sigma: pd.DataFrame,
    returns: pd.DataFrame,
    rf_annual: float,
    universe: pd.DataFrame,
    upper_bounds: np.ndarray,
    group_constraints: list[dict[str, object]],
    context: dict[str, object],
    regulatory_map: pd.DataFrame,
    params: OptimisationParameters,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Generate feasible random portfolios using Dirichlet sampling."""

    rng = np.random.default_rng(params.random_seed)
    n = len(mu)
    feasible_rows: list[dict[str, object]] = []
    feasible_weights: list[np.ndarray] = []
    attempts = 0
    batch_size = 5000
    sig = sigma.to_numpy(float)
    ret = returns.to_numpy(float)
    while len(feasible_rows) < params.monte_carlo_requested and attempts < params.monte_carlo_max_attempts:
        batch = min(batch_size, params.monte_carlo_max_attempts - attempts)
        weights = rng.dirichlet(np.ones(n), size=batch)
        attempts += batch
        ok = (weights <= upper_bounds + 1e-12).all(axis=1)
        for gc in group_constraints:
            mask = universe["asset_type"].eq(gc["asset_type"]).to_numpy(float)
            ok &= (weights @ mask >= float(gc["rhs"]) - 1e-12)
        for w in weights[ok]:
            compliance = check_regulatory_compliance(w, context, regulatory_map, "Monte_Carlo")
            if compliance["compliance_status"].eq("BREACH").any():
                continue
            pr = ret @ w
            er = float(mu.to_numpy(float) @ w)
            var = float(w.T @ sig @ w)
            vol = math.sqrt(max(var, 0.0))
            var_ret = float(np.quantile(pr, 0.05))
            cvar_ret = float(pr[pr <= var_ret].mean()) if np.any(pr <= var_ret) else var_ret
            wealth = np.cumprod(1.0 + pr)
            dd = wealth / np.maximum.accumulate(wealth) - 1.0
            feasible_weights.append(w.copy())
            feasible_rows.append({
                "portfolio_id": len(feasible_rows),
                "expected_return": er,
                "volatility": vol,
                "variance": var,
                "sharpe": (er - rf_annual) / vol if vol > 1e-12 else np.nan,
                "var_95": max(0.0, -var_ret),
                "cvar_95": max(0.0, -cvar_ret),
                "max_drawdown": float(np.min(dd)),
                "compliance_status": "COMPLIANT",
                "weights_json": json.dumps({asset: float(x) for asset, x in zip(mu.index, w)}, ensure_ascii=False),
            })
            if len(feasible_rows) >= params.monte_carlo_requested:
                break
    out = pd.DataFrame(feasible_rows)
    out.attrs["attempts"] = attempts
    return out, feasible_weights


def solve_efficient_frontier(
    mu: pd.Series,
    sigma: pd.DataFrame,
    rf_annual: float,
    universe: pd.DataFrame,
    upper_bounds: np.ndarray,
    group_constraints: list[dict[str, object]],
    n_points: int = 50,
) -> pd.DataFrame:
    """Solve a constrained efficient frontier."""

    max_ret = float(mu.max())
    min_ret = float(mu.min())
    targets = np.linspace(min_ret, max_ret, n_points)
    rows: list[dict[str, object]] = []
    n = len(mu)
    sig = sigma.to_numpy(float)
    mu_values = mu.to_numpy(float)
    for target in targets:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, w >= 0, w <= upper_bounds, mu_values @ w >= target]
        for gc in group_constraints:
            mask = universe["asset_type"].eq(gc["asset_type"]).to_numpy(float)
            constraints.append(mask @ w >= float(gc["rhs"]))
        problem = cp.Problem(cp.Minimize(cp.quad_form(w, sig)), constraints)
        try:
            problem.solve(solver="CLARABEL", verbose=False)
        except Exception:
            problem.solve(verbose=False)
        if w.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
            rows.append({"target_return": target, "status": str(problem.status), "weights_json": ""})
            continue
        weights = _bound_and_normalize_weights(w.value, upper_bounds)
        achieved = float(mu_values @ weights)
        var = float(weights.T @ sig @ weights)
        vol = math.sqrt(max(var, 0.0))
        rows.append({
            "target_return": target,
            "achieved_return": achieved,
            "volatility": vol,
            "variance": var,
            "sharpe": (achieved - rf_annual) / vol if vol > 1e-12 else np.nan,
            "status": str(problem.status),
            "weights_json": json.dumps({asset: float(x) for asset, x in zip(mu.index, weights)}, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def optimized_weights_table(portfolios: dict[str, np.ndarray], universe: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    """Long-form weights table for all optimized portfolios."""

    rows: list[pd.DataFrame] = []
    v_opt = float(context["optimisable_value"])
    for name, w in portfolios.items():
        part = universe[["asset_id", "asset_name", "asset_class", "issuer", "current_weight_optimisable"]].copy()
        part["portfolio_name"] = name
        part["current_weight"] = part["current_weight_optimisable"]
        part["optimized_weight"] = w
        part["weight_change"] = part["optimized_weight"] - part["current_weight"]
        part["optimized_value"] = part["optimized_weight"] * v_opt
        part["quality_flag"] = np.where(part["optimized_weight"] >= -1e-10, "OK", "NEGATIVE_WEIGHT_ERROR")
        rows.append(part[["portfolio_name", "asset_id", "asset_name", "asset_class", "issuer", "current_weight", "optimized_weight", "weight_change", "optimized_value", "quality_flag"]])
    return pd.concat(rows, ignore_index=True)
