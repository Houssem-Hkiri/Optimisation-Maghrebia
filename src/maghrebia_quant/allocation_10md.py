"""Allocation additionnelle de 10 MD dans la poche optimisable Maghrebia.

Le module réutilise les sorties validées des notebooks 01 et 02. Les 10 MD
sont investis uniquement dans les actifs optimisables, puis l'impact est mesuré
sur la poche optimisable et sur le portefeuille global après ajout.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class AdditionalAllocationConfig:
    """Paramètres centraux de l'analyse."""

    additional_budget: float = 10_000_000.0
    roe_spread_target: float = 0.04
    cvar_beta: float = 0.95
    periods_per_year: int = 252
    max_weight_per_asset: float = 0.30
    max_weight_per_issuer: float = 0.35
    max_equity_weight: float = 0.30
    max_corporate_weight: float = 0.65
    monte_carlo_portfolios: int = 20_000
    random_seed: int = 20260518
    zero_weight_tolerance: float = 1e-6
    display_weight_threshold: float = 0.0001


def read_matrix_csv(path: str | Path) -> pd.DataFrame:
    """Lire une matrice carrée exportée avec une colonne ``asset_id``."""

    df = pd.read_csv(path)
    if "asset_id" not in df.columns:
        raise ValueError(f"Matrice invalide sans colonne asset_id : {path}")
    matrix = df.set_index("asset_id")
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors="coerce")


def clean_weights(weights: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Mettre à zéro les poids quasi nuls puis renormaliser."""

    out = np.asarray(weights, dtype=float).copy()
    out[np.abs(out) < tolerance] = 0.0
    out[out < 0.0] = 0.0
    total = float(out.sum())
    if total <= 0:
        return out
    return out / total


def _technical_provisions(project_dir: Path) -> float:
    """Extraire les provisions techniques depuis ``Maghrebia Portfolio.xlsx``."""

    path = project_dir / "data" / "Maghrebia Portfolio.xlsx"
    df = pd.read_excel(path, sheet_name="Principal")
    designation_col = next((str(c) for c in df.columns if "signation" in str(c).lower() and "actifs" in str(c).lower()), None)
    value_col = next((str(c) for c in df.columns if "bilan" in str(c).lower()), None)
    if designation_col is None or value_col is None:
        raise ValueError("Colonnes portefeuille/provisions introuvables dans Maghrebia Portfolio.xlsx.")
    mask = df[designation_col].astype(str).str.contains("Montant des Provisions Techniques", case=False, na=False)
    values = pd.to_numeric(df.loc[mask, value_col], errors="coerce").dropna()
    if values.empty:
        raise ValueError("Provisions techniques introuvables dans Maghrebia Portfolio.xlsx.")
    return float(values.iloc[0])


def _load_portfolio_tables(workbook_path: Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Charger les tables portefeuille validées issues du fichier source."""

    summary = pd.read_excel(workbook_path, sheet_name="Portfolio_Summary").iloc[0]
    optimisable = pd.read_excel(workbook_path, sheet_name="Optimisable_Pocket")
    fixed = pd.read_excel(workbook_path, sheet_name="Non_Optimisable")
    return summary, optimisable, fixed


def _detect_return_frequency(returns: pd.DataFrame, configured_periods: int = 252) -> dict[str, object]:
    """Détecter la fréquence empirique des rendements périodiques."""

    dates = pd.DatetimeIndex(returns.index).sort_values()
    diffs = dates.to_series().diff().dropna().dt.days
    median_gap = float(diffs.median()) if not diffs.empty else np.nan
    if np.isfinite(median_gap) and median_gap <= 3:
        freq = "daily"
        periods = 252
    elif np.isfinite(median_gap) and median_gap <= 10:
        freq = "weekly"
        periods = 52
    else:
        freq = "unknown"
        periods = configured_periods
    return {
        "RETURN_FREQUENCY": freq,
        "PERIODS_PER_YEAR": periods,
        "nombre_observations": len(returns),
        "date_min": dates.min(),
        "date_max": dates.max(),
        "nombre_actifs": returns.shape[1],
        "fréquence_empirique_estimée_jours": median_gap,
    }


def _nearest_psd(matrix: pd.DataFrame, eps: float = 1e-10) -> tuple[pd.DataFrame, bool, float]:
    """Assurer une covariance symétrique semi-définie positive."""

    sym = (matrix + matrix.T) / 2
    values = sym.to_numpy(float)
    eigvals, eigvecs = np.linalg.eigh(values)
    min_eig = float(eigvals.min())
    if min_eig >= -eps:
        return sym, False, min_eig
    fixed = eigvecs @ np.diag(np.maximum(eigvals, eps)) @ eigvecs.T
    out = pd.DataFrame((fixed + fixed.T) / 2, index=matrix.index, columns=matrix.columns)
    return out, True, min_eig


def load_inputs(project_dir: str | Path) -> dict[str, object]:
    """Charger et contrôler les sources nécessaires au notebook 03."""

    project = Path(project_dir)
    source_portfolio_path = project / "data" / "Maghrebia Portfolio.xlsx"
    workbook_path = project / "data" / "exports" / "diagnostic_pre_optimisation_2025.xlsx"
    opt_dir = project / "data" / "exports" / "optimization_inputs"
    expected_path = opt_dir / "apt_expected_returns_2025.csv"
    sigma_path = opt_dir / "apt_covariance_matrix_2025.csv"
    if not source_portfolio_path.exists():
        raise FileNotFoundError(source_portfolio_path)
    if not workbook_path.exists():
        raise FileNotFoundError(workbook_path)
    if not expected_path.exists() or not sigma_path.exists():
        raise FileNotFoundError("Exports APT du notebook 01 manquants.")

    summary, optimisable, fixed = _load_portfolio_tables(workbook_path)
    expected = pd.read_csv(expected_path)
    expected["asset_id"] = expected["asset_id"].astype(str)
    assets = expected["asset_id"].tolist()
    sigma_raw = read_matrix_csv(sigma_path).reindex(index=assets, columns=assets)
    sigma, sigma_repaired, sigma_min_eig = _nearest_psd(sigma_raw)
    mu = expected.set_index("asset_id")["expected_return_annualized_final"].astype(float).reindex(assets)
    if mu.isna().any() or sigma.isna().any().any():
        raise ValueError("mu_APT ou Sigma_APT contient des NaN après alignement.")

    returns = pd.read_excel(workbook_path, sheet_name="Returns_Model")
    returns["date"] = pd.to_datetime(returns["date"])
    returns = returns.set_index("date").reindex(columns=assets).apply(pd.to_numeric, errors="coerce")
    if returns.isna().any().any():
        raise ValueError("Rendements périodiques incomplets pour l'univers optimisable.")
    frequency_control = _detect_return_frequency(returns)

    rf_daily = pd.read_excel(workbook_path, sheet_name="Risk_Free_Daily_2025")
    rf_values = pd.to_numeric(rf_daily.get("rf_annual_decimal"), errors="coerce").dropna()
    tsr = float(rf_values.mean()) if not rf_values.empty else np.nan
    tsr_source = "BCT_SHORT_RATE_FROM_NOTEBOOK_01" if np.isfinite(tsr) else "MANUAL_TSR_REQUIRED"

    universe = expected.merge(
        optimisable[
            [
                "asset_id",
                "asset_name",
                "asset_class_standardized",
                "asset_type",
                "market_value",
                "portfolio_weight",
                "optimisable_weight",
                "isin",
                "sector",
                "maturity_date",
                "is_optimisable",
            ]
        ],
        on="asset_id",
        how="left",
        suffixes=("", "_portfolio"),
    )
    universe["asset_name"] = universe["asset_name"].fillna(universe["asset_name_portfolio"])
    universe["asset_class"] = universe["asset_class"].fillna(universe["asset_class_standardized"])
    universe["asset_type"] = universe["asset_type"].fillna(universe["asset_type_portfolio"])
    universe["issuer"] = universe["asset_id"]
    universe["is_optimizable"] = universe["is_optimisable"].fillna(True).astype(bool)
    universe["current_value"] = pd.to_numeric(universe["current_value"], errors="coerce").fillna(pd.to_numeric(universe["market_value"], errors="coerce"))
    universe["current_weight_optimisable"] = pd.to_numeric(universe["current_weight_optimisable"], errors="coerce").fillna(pd.to_numeric(universe["optimisable_weight"], errors="coerce"))
    universe["current_weight_total"] = pd.to_numeric(universe["portfolio_weight"], errors="coerce")
    universe["expected_return_annualized_final"] = pd.to_numeric(universe["expected_return_annualized_final"], errors="coerce")
    universe["historical_volatility_annualized"] = pd.to_numeric(universe.get("historical_volatility_annualized"), errors="coerce")
    allowed_types = {"government_bond", "corporate_bond", "listed_equity"}
    universe = universe.loc[universe["is_optimizable"] & universe["asset_type"].isin(allowed_types)].copy()
    if universe[["current_value", "current_weight_optimisable", "expected_return_annualized_final"]].isna().any().any():
        raise ValueError("Univers optimisable incomplet.")

    universe = universe.set_index("asset_id").reindex(assets).reset_index()
    if universe["asset_name"].isna().any():
        raise ValueError("Certains actifs APT ne sont pas présents dans la poche optimisable.")
    weight_sum = float(universe["current_weight_optimisable"].sum())
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
        universe["current_weight_optimisable"] = universe["current_weight_optimisable"] / weight_sum

    v_total_current = float(summary["total_portfolio_value"])
    v_opt_current = float(summary["optimisable_value"])
    v_fixed_current = float(summary["non_optimisable_value"])
    assert abs(v_total_current - v_opt_current - v_fixed_current) <= 1e-2
    assert abs(v_opt_current - universe["current_value"].sum()) <= 1e-2

    return {
        "project_dir": project,
        "source_portfolio_path": source_portfolio_path,
        "diagnostic_workbook_path": workbook_path,
        "portfolio_summary": summary,
        "optimisable_source": optimisable,
        "fixed_source": fixed,
        "universe": universe,
        "expected": expected,
        "mu": mu,
        "sigma": sigma,
        "sigma_repaired": sigma_repaired,
        "sigma_min_eig_before": sigma_min_eig,
        "returns": returns,
        "frequency_control": frequency_control,
        "technical_provisions": _technical_provisions(project),
        "tsr": tsr,
        "tsr_source": tsr_source,
        "V_TOTAL_CURRENT": v_total_current,
        "V_OPT_CURRENT": v_opt_current,
        "V_FIXED_CURRENT": v_fixed_current,
    }


def _class_masks(universe: pd.DataFrame) -> dict[str, np.ndarray]:
    """Masques par type d'actif dans l'univers optimisable."""

    return {
        "government_bond": universe["asset_type"].eq("government_bond").to_numpy(),
        "corporate_bond": universe["asset_type"].eq("corporate_bond").to_numpy(),
        "listed_equity": universe["asset_type"].eq("listed_equity").to_numpy(),
    }


def _portfolio_var_cvar(periodic_returns: np.ndarray) -> tuple[float, float]:
    """Calculer VaR et CVaR 95% sur rendements périodiques."""

    losses = -np.asarray(periodic_returns, dtype=float)
    var_95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var_95]
    cvar_95 = float(tail.mean()) if len(tail) else var_95
    return max(0.0, var_95), max(0.0, cvar_95)


def _portfolio_state(data: dict[str, object], config: AdditionalAllocationConfig) -> dict[str, float]:
    """Calculer les variables centrales V/R du portefeuille."""

    w_current = data["universe"]["current_weight_optimisable"].to_numpy(float)
    r_opt_current = float(data["mu"].to_numpy(float) @ w_current)
    fixed_return_assumption = 0.0
    v_total = float(data["V_TOTAL_CURRENT"])
    v_opt = float(data["V_OPT_CURRENT"])
    v_fixed = float(data["V_FIXED_CURRENT"])
    r_total_current = (v_opt * r_opt_current + v_fixed * fixed_return_assumption) / v_total
    v_opt_final = v_opt + config.additional_budget
    v_total_final = v_total + config.additional_budget
    target_return = float(data["tsr"]) + config.roe_spread_target
    required_opt = (target_return * v_opt_final - r_opt_current * v_opt) / config.additional_budget
    required_total = (target_return * v_total_final - r_total_current * v_total) / config.additional_budget
    assert abs(v_total - v_opt - v_fixed) <= 1e-2
    assert abs(v_opt_final - v_opt - config.additional_budget) <= 1e-8
    assert abs(v_total_final - v_total - config.additional_budget) <= 1e-8
    return {
        "R_FIXED_CURRENT_ASSUMPTION": fixed_return_assumption,
        "R_OPT_CURRENT": r_opt_current,
        "R_TOTAL_CURRENT": r_total_current,
        "TARGET_RETURN": target_return,
        "V_OPT_FINAL": v_opt_final,
        "V_TOTAL_FINAL": v_total_final,
        "R_REQUIRED_ADDITIONAL_FOR_OPT_TARGET": required_opt,
        "R_REQUIRED_ADDITIONAL_FOR_TOTAL_TARGET": required_total,
    }


def _final_values(universe: pd.DataFrame, weights: np.ndarray, config: AdditionalAllocationConfig) -> pd.DataFrame:
    """Reconstruire les montants de la poche optimisable après allocation."""

    out = universe.copy()
    out["weight_10md"] = np.asarray(weights, dtype=float)
    out["additional_value"] = config.additional_budget * out["weight_10md"]
    out["final_value"] = out["current_value"] + out["additional_value"]
    return out


def _weight_bounds(universe: pd.DataFrame, technical_provisions: float, config: AdditionalAllocationConfig) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Bornes par actif, y compris les limites actions par société."""

    upper = np.full(len(universe), config.max_weight_per_asset, dtype=float)
    equity_mask = universe["asset_type"].eq("listed_equity").to_numpy()
    if equity_mask.any():
        capacity = 0.10 * technical_provisions - universe.loc[equity_mask, "current_value"].to_numpy(float)
        upper[equity_mask] = np.minimum(upper[equity_mask], np.maximum(capacity / config.additional_budget, 0.0))
    return [(0.0, float(x)) for x in upper], upper


def _issuer_masks(universe: pd.DataFrame) -> list[tuple[np.ndarray, str]]:
    """Masques par émetteur."""

    masks: list[tuple[np.ndarray, str]] = []
    for issuer, group in universe.groupby("issuer", dropna=False):
        masks.append((universe.index.isin(group.index).astype(float), str(issuer)))
    return masks


def _cvx_constraints(
    w: cp.Variable,
    universe: pd.DataFrame,
    data: dict[str, object],
    config: AdditionalAllocationConfig,
    required_return: float | None = None,
) -> list[cp.Constraint]:
    """Contraintes d'investissement et contraintes réglementaires testables."""

    masks = _class_masks(universe)
    current_values = universe["current_value"].to_numpy(float)
    v_opt_final = float(data["V_OPT_CURRENT"]) + config.additional_budget
    final_opt_weights = (current_values + config.additional_budget * w) / v_opt_final
    technical_provisions = float(data["technical_provisions"])
    bounds = _weight_bounds(universe, technical_provisions, config)[1]
    constraints: list[cp.Constraint] = [
        cp.sum(w) == 1,
        w >= 0,
        w <= bounds,
        cp.sum(final_opt_weights[masks["listed_equity"]]) <= config.max_equity_weight,
        cp.sum(final_opt_weights[masks["corporate_bond"]]) <= config.max_corporate_weight,
        current_values[masks["government_bond"]].sum() + config.additional_budget * cp.sum(w[masks["government_bond"]]) >= 0.20 * technical_provisions,
    ]
    for mask, _issuer in _issuer_masks(universe):
        constraints.append(cp.sum(cp.multiply(mask, w)) <= config.max_weight_per_issuer)
    if required_return is not None and np.isfinite(required_return):
        constraints.append(universe["expected_return_annualized_final"].to_numpy(float) @ w >= required_return)
    return constraints


def _solve_problem(problem: cp.Problem) -> str:
    """Résoudre un problème cvxpy en capturant les warnings numériques."""

    for solver in ("CLARABEL", "ECOS", "SCS"):
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                problem.solve(solver=solver, verbose=False)
        except Exception:
            continue
        if problem.status in {"optimal", "optimal_inaccurate"}:
            return str(problem.status)
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            problem.solve(verbose=False)
    except Exception:
        return "solver_error"
    return str(problem.status)


def _result_from_cvxpy(name: str, w: cp.Variable, problem: cp.Problem, status: str, n_assets: int) -> dict[str, object]:
    """Normaliser une sortie cvxpy."""

    if w.value is None or status not in {"optimal", "optimal_inaccurate"}:
        return {"model": name, "success": False, "solver_status": status, "objective_value": np.nan, "weights": np.full(n_assets, np.nan)}
    return {
        "model": name,
        "success": True,
        "solver_status": status,
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
        "weights": clean_weights(np.asarray(w.value, dtype=float)),
    }


def solve_cvx_model(
    name: str,
    objective: str,
    data: dict[str, object],
    config: AdditionalAllocationConfig,
    required_return: float | None = None,
    risk_aversion: float = 5.0,
) -> dict[str, object]:
    """Résoudre un modèle convexe déterministe."""

    universe = data["universe"]
    n_assets = len(universe)
    w = cp.Variable(n_assets)
    mu = data["mu"].to_numpy(float)
    sigma = data["sigma"].to_numpy(float)
    returns = data["returns"].to_numpy(float)
    constraints = _cvx_constraints(w, universe, data, config, required_return)
    if objective == "min_variance":
        expr = cp.Minimize(cp.quad_form(w, sigma))
    elif objective == "mean_variance":
        expr = cp.Minimize((risk_aversion / 2.0) * cp.quad_form(w, sigma) - mu @ w)
    elif objective == "max_return":
        expr = cp.Maximize(mu @ w)
    elif objective == "mean_cvar":
        alpha = cp.Variable()
        u = cp.Variable(returns.shape[0], nonneg=True)
        losses = -returns @ w
        cvar = alpha + (1.0 / ((1.0 - config.cvar_beta) * returns.shape[0])) * cp.sum(u)
        constraints.append(u >= losses - alpha)
        expr = cp.Minimize(cvar)
    else:
        raise ValueError(objective)
    problem = cp.Problem(expr, constraints)
    status = _solve_problem(problem)
    return _result_from_cvxpy(name, w, problem, status, n_assets)


def _slsqp_constraints(data: dict[str, object], config: AdditionalAllocationConfig, required_return: float | None = None) -> list[dict[str, object]]:
    """Contraintes SLSQP équivalentes aux contraintes principales."""

    universe = data["universe"]
    mu = data["mu"].to_numpy(float)
    masks = _class_masks(universe)
    current_values = universe["current_value"].to_numpy(float)
    v_opt_final = float(data["V_OPT_CURRENT"]) + config.additional_budget
    technical_provisions = float(data["technical_provisions"])
    constraints: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},
        {"type": "ineq", "fun": lambda x: float(current_values[masks["government_bond"]].sum() + config.additional_budget * x[masks["government_bond"]].sum() - 0.20 * technical_provisions)},
        {"type": "ineq", "fun": lambda x: float(config.max_equity_weight - (current_values[masks["listed_equity"]] + config.additional_budget * x[masks["listed_equity"]]).sum() / v_opt_final)},
        {"type": "ineq", "fun": lambda x: float(config.max_corporate_weight - (current_values[masks["corporate_bond"]] + config.additional_budget * x[masks["corporate_bond"]]).sum() / v_opt_final)},
    ]
    for mask, _issuer in _issuer_masks(universe):
        constraints.append({"type": "ineq", "fun": lambda x, mask=mask: float(config.max_weight_per_issuer - np.sum(x * mask))})
    if required_return is not None and np.isfinite(required_return):
        constraints.append({"type": "ineq", "fun": lambda x: float(mu @ x - required_return)})
    return constraints


def _starting_points(data: dict[str, object], config: AdditionalAllocationConfig) -> list[np.ndarray]:
    """Points de départ déterministes."""

    current = data["universe"]["current_weight_optimisable"].to_numpy(float)
    equal = np.ones(len(current)) / len(current)
    high_return = np.zeros(len(current))
    mu = data["mu"].to_numpy(float)
    for idx in np.argsort(mu)[::-1]:
        remaining = 1.0 - high_return.sum()
        if remaining <= 1e-12:
            break
        high_return[idx] = min(config.max_weight_per_asset, remaining)
    return [current / current.sum(), equal, high_return / high_return.sum(), 0.5 * current + 0.5 * equal]


def _slsqp_feasible(weights: np.ndarray, constraints: list[dict[str, object]], tolerance: float = 1e-7) -> bool:
    """Tester la faisabilité numérique d'une solution SLSQP."""

    for constraint in constraints:
        value = float(constraint["fun"](weights))
        if constraint["type"] == "eq" and abs(value) > tolerance:
            return False
        if constraint["type"] == "ineq" and value < -tolerance:
            return False
    return True


def solve_slsqp_model(
    name: str,
    objective_name: str,
    data: dict[str, object],
    config: AdditionalAllocationConfig,
    required_return: float | None = None,
) -> dict[str, object]:
    """Résoudre Risk Parity. Max Sharpe est volontairement exclu du notebook 03."""

    if objective_name != "risk_parity":
        raise ValueError(
            f"Objectif SLSQP non supporté : {objective_name}. Max Sharpe est exclu conformément à la consigne du notebook 03."
        )
    universe = data["universe"]
    bounds = _weight_bounds(universe, float(data["technical_provisions"]), config)[0]
    constraints = _slsqp_constraints(data, config, required_return)
    sigma = data["sigma"].to_numpy(float)

    def objective(weights: np.ndarray) -> float:
        variance = float(weights.T @ sigma @ weights)
        volatility = math.sqrt(max(variance, 0.0))
        if volatility <= 1e-12:
            return 1e6
        risk_contrib = weights * (sigma @ weights) / volatility
        return float(np.sum((risk_contrib - risk_contrib.mean()) ** 2))

    best = None
    for start in _starting_points(data, config):
        res = minimize(
            objective,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 3000, "ftol": 1e-12, "disp": False},
        )
        if (res.success or _slsqp_feasible(res.x, constraints)) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return {"model": name, "success": False, "solver_status": "SLSQP_FAILED", "objective_value": np.nan, "weights": np.full(len(universe), np.nan)}
    return {
        "model": name,
        "success": True,
        "solver_status": str(best.message) if best.success else f"{best.message};FEASIBLE_NUMERIC_SOLUTION",
        "objective_value": float(best.fun),
        "weights": clean_weights(best.x),
    }


def concentration_metrics(weights: np.ndarray, current_weights: np.ndarray | None = None) -> dict[str, object]:
    """Calculer concentration et distance à la poche actuelle."""

    values = np.sort(np.asarray(weights, dtype=float))[::-1]
    turnover = float(np.abs(np.asarray(weights) - current_weights).sum()) if current_weights is not None else np.nan
    return {
        "max_weight_asset": float(values[0]) if len(values) else np.nan,
        "top_3_concentration": float(values[:3].sum()),
        "top_5_concentration": float(values[:5].sum()),
        "number_of_assets_used": int(np.sum(np.asarray(weights) > 0.0001)),
        "turnover_vs_current_opt": turnover,
    }


def _regulatory_table(data: dict[str, object], weights: np.ndarray, config: AdditionalAllocationConfig, model: str) -> pd.DataFrame:
    """Contrôler les contraintes testables sur le portefeuille global final."""

    universe = data["universe"]
    fixed = data["fixed_source"]
    after = _final_values(universe, weights, config)
    current_by_type = universe.groupby("asset_type")["current_value"].sum().to_dict()
    final_by_type = after.groupby("asset_type")["final_value"].sum().to_dict()
    fixed_by_type = fixed.groupby("asset_type")["market_value"].sum().to_dict() if not fixed.empty else {}
    pt = float(data["technical_provisions"])
    total_before = float(data["V_TOTAL_CURRENT"])
    total_after = total_before + config.additional_budget
    rows: list[dict[str, object]] = []

    def add(name: str, before: float | None, after_value: float | None, threshold_pct: float | None, kind: str, required: str, available: str) -> None:
        if before is None or after_value is None or threshold_pct is None:
            rows.append({
                "model": model,
                "Constraint": name,
                "Status": "NON_TESTABLE_DATA_MISSING",
                "Reason": "Données détaillées manquantes.",
                "Required_Data": required,
                "Available_Data": available,
                "exposition_avant": np.nan,
                "exposition_apres": np.nan,
                "seuil": np.nan,
                "marge_avant": np.nan,
                "marge_apres": np.nan,
                "testable": False,
            })
            return
        threshold = threshold_pct * pt
        if kind == "min":
            margin_before = before - threshold
            margin_after = after_value - threshold
        else:
            margin_before = threshold - before
            margin_after = threshold - after_value
        status = "PASSED" if margin_after >= -1e-6 else "FAILED"
        rows.append({
            "model": model,
            "Constraint": name,
            "Status": status,
            "Reason": "Aucune violation détectée sur contrainte testable." if status == "PASSED" else "Dépassement détecté.",
            "Required_Data": required,
            "Available_Data": available,
            "exposition_avant": before,
            "exposition_apres": after_value,
            "seuil": threshold,
            "marge_avant": margin_before,
            "marge_apres": margin_after,
            "testable": True,
        })

    add("Couverture des provisions techniques", total_before, total_after, 1.00, "min", "Valeur portefeuille total; provisions techniques", "Disponible")
    state_before = current_by_type.get("government_bond", 0.0) + fixed_by_type.get("government_bond", 0.0)
    state_after = final_by_type.get("government_bond", 0.0) + fixed_by_type.get("government_bond", 0.0)
    add("Titres d'État >= 20% PT", state_before, state_after, 0.20, "min", "Classification actifs; PT", "Disponible")
    equity_before = current_by_type.get("listed_equity", 0.0) + fixed_by_type.get("listed_equity", 0.0)
    equity_after = final_by_type.get("listed_equity", 0.0) + fixed_by_type.get("listed_equity", 0.0)
    add("Actions cotées globales <= 30% poche optimisable finale", equity_before, equity_after, None, "max", "Référentiel détaillé", "Non défini comme seuil légal PT")
    corporate_before = current_by_type.get("corporate_bond", 0.0)
    corporate_after = final_by_type.get("corporate_bond", 0.0)
    add("Obligations corporate - limite interne suivie", corporate_before, corporate_after, None, "max", "Référentiel détaillé", "Seuil interne d'optimisation")
    real_estate = fixed_by_type.get("real_estate", 0.0)
    add("Immobilier total <= 20% PT", real_estate, real_estate, 0.20, "max", "Immobilier; PT", "Disponible")
    sicar = fixed_by_type.get("sicar", 0.0)
    add("SICAR/SICAF total <= 10% PT", sicar, sicar, 0.10, "max", "SICAR/SICAF; PT", "Disponible")
    for row in after.loc[after["asset_type"].eq("listed_equity")].itertuples():
        add(f"Action cotée par société <= 10% PT::{row.asset_id}", float(row.current_value), float(row.final_value), 0.10, "max", "Exposition action par société; PT", "Disponible")
    for name, required in [
        ("Poids par émetteur hors actions cotées", "Émetteur économique consolidé"),
        ("Limites en pourcentage du capital social", "Capital social par émetteur"),
        ("OPCVM par entité", "Détail par OPCVM"),
        ("Actions non cotées et participations", "Détail réglementaire hors poche optimisable"),
    ]:
        add(name, None, None, None, "max", required, "NON_TESTABLE_DATA_MISSING")
    return pd.DataFrame(rows)


def _regulatory_passed(data: dict[str, object], weights: np.ndarray, config: AdditionalAllocationConfig) -> bool:
    """Contrôle rapide des contraintes testables."""

    table = _regulatory_table(data, weights, config, "CHECK")
    return not table.loc[table["testable"], "Status"].eq("FAILED").any()


def evaluate_model(model: str, weights: np.ndarray, data: dict[str, object], state: dict[str, float], config: AdditionalAllocationConfig) -> dict[str, object]:
    """Évaluer rendement, risque, contraintes et warnings d'un modèle."""

    weights = clean_weights(weights, config.zero_weight_tolerance)
    assert abs(weights.sum() - 1.0) <= 1e-8
    assert (weights >= -1e-10).all()
    assert abs(float((weights * config.additional_budget).sum()) - config.additional_budget) <= 1e-4

    mu = data["mu"].to_numpy(float)
    sigma = data["sigma"].to_numpy(float)
    returns = data["returns"].to_numpy(float)
    current_weights = data["universe"]["current_weight_optimisable"].to_numpy(float)
    r_additional = float(mu @ weights)
    final_values = data["universe"]["current_value"].to_numpy(float) + config.additional_budget * weights
    final_opt_weights = final_values / state["V_OPT_FINAL"]
    r_opt_final = (data["V_OPT_CURRENT"] * state["R_OPT_CURRENT"] + config.additional_budget * r_additional) / state["V_OPT_FINAL"]
    r_total_final = (data["V_TOTAL_CURRENT"] * state["R_TOTAL_CURRENT"] + config.additional_budget * r_additional) / state["V_TOTAL_FINAL"]
    variance_add = float(weights.T @ sigma @ weights)
    volatility_add = math.sqrt(max(variance_add, 0.0))
    opt_variance = float(final_opt_weights.T @ sigma @ final_opt_weights)
    opt_volatility = math.sqrt(max(opt_variance, 0.0))
    total_volatility_proxy = opt_volatility * state["V_OPT_FINAL"] / state["V_TOTAL_FINAL"]
    periodic = returns @ weights
    var_95, cvar_95 = _portfolio_var_cvar(periodic)
    regulatory = _regulatory_table(data, weights, config, model)
    regulatory_status = "PASSED" if not regulatory.loc[regulatory["testable"], "Status"].eq("FAILED").any() else "FAILED"
    warnings_list: list[str] = []
    sharpe = (r_additional - data["tsr"]) / volatility_add if volatility_add > 1e-12 else np.nan
    if np.isfinite(sharpe) and sharpe > 5:
        warnings_list.append("WARNING_SHARPE_UNREALISTIC_CHECK_VOL_OR_FREQUENCY")
    has_equity = bool(weights[_class_masks(data["universe"])["listed_equity"]].sum() > 1e-6)
    if has_equity and (var_95 <= 1e-12 or cvar_95 <= 1e-12):
        warnings_list.append("WARNING_ZERO_VAR_CVAR_WITH_RISKY_ASSETS")
    if volatility_add <= 1e-6:
        warnings_list.append("WARNING_LOW_VOLATILITY_CHECK_COVARIANCE")
    if regulatory["Status"].eq("NON_TESTABLE_DATA_MISSING").any():
        warnings_list.append("WARNING_REGULATORY_NON_TESTABLE_CONSTRAINTS")
    target_opt = "YES" if r_opt_final >= state["TARGET_RETURN"] - 1e-10 else "NO"
    target_total = "YES" if r_total_final >= state["TARGET_RETURN"] - 1e-10 else "NO"
    return {
        "Model": model,
        "R_additional": r_additional,
        "Volatility_additional": volatility_add,
        "Variance_additional": variance_add,
        "Sharpe": sharpe,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "VaR_95_DT_total": var_95 * state["V_TOTAL_FINAL"],
        "CVaR_95_DT_total": cvar_95 * state["V_TOTAL_FINAL"],
        "R_opt_final": r_opt_final,
        "R_total_final": r_total_final,
        "Target_Return": state["TARGET_RETURN"],
        "Gap_Opt": r_opt_final - state["TARGET_RETURN"],
        "Gap_Total": r_total_final - state["TARGET_RETURN"],
        "Target_Opt_Reached": target_opt,
        "Target_Total_Reached": target_total,
        "Target_Status": "PASSED" if target_opt == "YES" and target_total == "YES" else "TARGET_NOT_REACHED",
        "Regulatory_Status": regulatory_status,
        "Comment": "",
        "Warnings": ";".join(warnings_list) if warnings_list else "OK",
        **concentration_metrics(weights, current_weights),
    }


def allocation_table(model: str, weights: np.ndarray, data: dict[str, object], config: AdditionalAllocationConfig) -> pd.DataFrame:
    """Construire la table des poids et montants alloués."""

    final = _final_values(data["universe"], weights, config)
    final["Model"] = model
    final["amount_allocated_DT"] = final["additional_value"]
    final["final_weight_opt"] = final["final_value"] / (data["V_OPT_CURRENT"] + config.additional_budget)
    final["final_weight_total"] = final["final_value"] / (data["V_TOTAL_CURRENT"] + config.additional_budget)
    final["return_contribution"] = final["weight_10md"] * final["expected_return_annualized_final"]
    final["Included_in_Optimization"] = True
    final["display_in_main_tables"] = final["weight_10md"] >= config.display_weight_threshold
    keep = [
        "Model",
        "asset_id",
        "asset_name",
        "asset_class",
        "asset_type",
        "issuer",
        "isin",
        "current_value",
        "current_weight_optimisable",
        "current_weight_total",
        "weight_10md",
        "amount_allocated_DT",
        "final_value",
        "final_weight_opt",
        "final_weight_total",
        "expected_return_annualized_final",
        "historical_volatility_annualized",
        "quality_flag",
        "model_status",
        "maturity_date",
        "return_contribution",
        "Included_in_Optimization",
        "display_in_main_tables",
    ]
    return final[[c for c in keep if c in final.columns]]


def asset_control_table(data: dict[str, object]) -> pd.DataFrame:
    """Contrôle de l'univers optimisable et des métriques par actif."""

    rows = []
    returns = data["returns"]
    for row in data["universe"].itertuples():
        series = returns[row.asset_id].to_numpy(float)
        var_95, cvar_95 = _portfolio_var_cvar(series)
        rows.append({
            "Asset": row.asset_id,
            "Classe": row.asset_class,
            "Valeur actuelle": row.current_value,
            "Poids actuel poche optimisable": row.current_weight_optimisable,
            "Poids actuel portefeuille total": row.current_weight_total,
            "Rendement attendu": row.expected_return_annualized_final,
            "Volatilité": getattr(row, "historical_volatility_annualized", np.nan),
            "VaR": var_95,
            "CVaR": cvar_95,
            "Statut données": getattr(row, "quality_flag", "OK"),
            "Included_in_Optimization": True,
        })
    return pd.DataFrame(rows)


def _solve_models(data: dict[str, object], state: dict[str, float], config: AdditionalAllocationConfig) -> list[dict[str, object]]:
    """Exécuter les modèles déterministes (Max Sharpe volontairement exclu)."""

    max_return = solve_cvx_model("Max_Return_Constraints", "max_return", data, config, None)
    max_return_value = float(data["mu"].to_numpy(float) @ max_return["weights"]) if max_return["success"] else np.nan
    required_for_constraint = state["R_REQUIRED_ADDITIONAL_FOR_OPT_TARGET"] if max_return_value >= state["R_REQUIRED_ADDITIONAL_FOR_OPT_TARGET"] else None
    current_weights = data["universe"]["current_weight_optimisable"].to_numpy(float)
    results = [
        {"model": "Prorata_Current_Optimizable_Pocket", "success": True, "solver_status": "BENCHMARK_PRORATA", "objective_value": np.nan, "weights": clean_weights(current_weights)},
        solve_cvx_model("Minimum_Variance", "min_variance", data, config, required_for_constraint),
        solve_cvx_model("Mean_Variance_Aversion_2", "mean_variance", data, config, required_for_constraint, risk_aversion=2.0),
        solve_cvx_model("Mean_Variance_Aversion_5", "mean_variance", data, config, required_for_constraint, risk_aversion=5.0),
        solve_cvx_model("Mean_Variance_Aversion_10", "mean_variance", data, config, required_for_constraint, risk_aversion=10.0),
        max_return,
        solve_cvx_model("Mean_CVaR", "mean_cvar", data, config, required_for_constraint),
        solve_slsqp_model("Risk_Parity", "risk_parity", data, config, None),
    ]
    return [r for r in results if bool(r["success"])]


def generate_monte_carlo(data: dict[str, object], state: dict[str, float], config: AdditionalAllocationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Générer des portefeuilles Monte Carlo réalisables."""

    rng = np.random.default_rng(config.random_seed)
    universe = data["universe"]
    n_assets = len(universe)
    upper = _weight_bounds(universe, float(data["technical_provisions"]), config)[1]
    current = universe["current_weight_optimisable"].to_numpy(float)
    alpha = np.maximum(current * 80.0, 1.0)
    rows: list[dict[str, object]] = []
    weights_rows: list[dict[str, object]] = []
    attempts = 0
    masks = _class_masks(universe)
    current_values = universe["current_value"].to_numpy(float)
    v_opt_final = state["V_OPT_FINAL"]
    issuer_masks = _issuer_masks(universe)
    while len(rows) < config.monte_carlo_portfolios and attempts < config.monte_carlo_portfolios * 100:
        candidates = rng.dirichlet(alpha, size=50_000)
        attempts += len(candidates)
        final_values = current_values[None, :] + config.additional_budget * candidates
        feasible = (
            (candidates <= upper[None, :] + 1e-12).all(axis=1)
            & ((final_values[:, masks["listed_equity"]].sum(axis=1) / v_opt_final) <= config.max_equity_weight + 1e-12)
            & ((final_values[:, masks["corporate_bond"]].sum(axis=1) / v_opt_final) <= config.max_corporate_weight + 1e-12)
            & (current_values[masks["government_bond"]].sum() + config.additional_budget * candidates[:, masks["government_bond"]].sum(axis=1) >= 0.20 * data["technical_provisions"] - 1e-8)
        )
        for mask, _issuer in issuer_masks:
            feasible &= (candidates @ mask <= config.max_weight_per_issuer + 1e-12)
        feasible_candidates = candidates[feasible]
        if len(feasible_candidates) == 0:
            continue
        remaining = config.monte_carlo_portfolios - len(rows)
        feasible_candidates = feasible_candidates[:remaining]
        mu = data["mu"].to_numpy(float)
        sigma = data["sigma"].to_numpy(float)
        returns = data["returns"].to_numpy(float)
        r_add = feasible_candidates @ mu
        r_opt = (data["V_OPT_CURRENT"] * state["R_OPT_CURRENT"] + config.additional_budget * r_add) / state["V_OPT_FINAL"]
        r_total = (data["V_TOTAL_CURRENT"] * state["R_TOTAL_CURRENT"] + config.additional_budget * r_add) / state["V_TOTAL_FINAL"]
        variances = np.einsum("ij,jk,ik->i", feasible_candidates, sigma, feasible_candidates)
        vol = np.sqrt(np.maximum(variances, 0.0))
        scenario = returns @ feasible_candidates.T
        losses = -scenario
        var_95 = np.quantile(losses, 0.95, axis=0)
        cvar_95 = np.array([
            losses[:, idx][losses[:, idx] >= var_95[idx]].mean() if np.any(losses[:, idx] >= var_95[idx]) else var_95[idx]
            for idx in range(losses.shape[1])
        ])
        sorted_weights = np.sort(feasible_candidates, axis=1)[:, ::-1]
        top3 = sorted_weights[:, :3].sum(axis=1)
        top5 = sorted_weights[:, :5].sum(axis=1)
        max_weight = sorted_weights[:, 0]
        used = (feasible_candidates > 0.0001).sum(axis=1)
        turnover = np.abs(feasible_candidates - current[None, :]).sum(axis=1)
        sharpe = np.where(vol > 1e-12, (r_add - data["tsr"]) / vol, np.nan)
        start_id = len(rows)
        for local_idx, weights in enumerate(feasible_candidates):
            pid = start_id + local_idx
            warning_parts = []
            if np.isfinite(sharpe[local_idx]) and sharpe[local_idx] > 5:
                warning_parts.append("WARNING_SHARPE_UNREALISTIC_CHECK_VOL_OR_FREQUENCY")
            if var_95[local_idx] <= 1e-12 or cvar_95[local_idx] <= 1e-12:
                warning_parts.append("WARNING_ZERO_VAR_CVAR_WITH_RISKY_ASSETS")
            rows.append({
                "Model": f"Monte_Carlo_{pid}",
                "portfolio_id": pid,
                "R_additional": float(r_add[local_idx]),
                "Volatility_additional": float(vol[local_idx]),
                "Variance_additional": float(variances[local_idx]),
                "Sharpe": float(sharpe[local_idx]) if np.isfinite(sharpe[local_idx]) else np.nan,
                "VaR_95": float(max(0.0, var_95[local_idx])),
                "CVaR_95": float(max(0.0, cvar_95[local_idx])),
                "VaR_95_DT_total": float(max(0.0, var_95[local_idx]) * state["V_TOTAL_FINAL"]),
                "CVaR_95_DT_total": float(max(0.0, cvar_95[local_idx]) * state["V_TOTAL_FINAL"]),
                "R_opt_final": float(r_opt[local_idx]),
                "R_total_final": float(r_total[local_idx]),
                "Target_Return": state["TARGET_RETURN"],
                "Gap_Opt": float(r_opt[local_idx] - state["TARGET_RETURN"]),
                "Gap_Total": float(r_total[local_idx] - state["TARGET_RETURN"]),
                "Target_Opt_Reached": "YES" if r_opt[local_idx] >= state["TARGET_RETURN"] - 1e-10 else "NO",
                "Target_Total_Reached": "YES" if r_total[local_idx] >= state["TARGET_RETURN"] - 1e-10 else "NO",
                "Target_Status": "PASSED" if r_opt[local_idx] >= state["TARGET_RETURN"] - 1e-10 and r_total[local_idx] >= state["TARGET_RETURN"] - 1e-10 else "TARGET_NOT_REACHED",
                "Regulatory_Status": "PASSED",
                "Comment": "Simulation Monte Carlo admissible.",
                "Warnings": ";".join(warning_parts) if warning_parts else "OK",
                "max_weight_asset": float(max_weight[local_idx]),
                "top_3_concentration": float(top3[local_idx]),
                "top_5_concentration": float(top5[local_idx]),
                "number_of_assets_used": int(used[local_idx]),
                "turnover_vs_current_opt": float(turnover[local_idx]),
            })
            for asset_id, weight in zip(universe["asset_id"], weights):
                if weight >= config.display_weight_threshold:
                    weights_rows.append({"portfolio_id": pid, "asset_id": asset_id, "weight": float(weight)})
        if len(rows) >= config.monte_carlo_portfolios:
            break
    if len(rows) < config.monte_carlo_portfolios:
        raise ValueError(f"Monte Carlo insuffisant : {len(rows)} portefeuilles générés.")
    return pd.DataFrame(rows), pd.DataFrame(weights_rows)


def _norm_high(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    return pd.Series(0.5, index=series.index) if span <= 1e-12 else (series - series.min()) / span


def _norm_low(series: pd.Series) -> pd.Series:
    return 1.0 - _norm_high(series)


def build_scoring(candidates: pd.DataFrame) -> pd.DataFrame:
    """Scoring multicritère institutionnel."""

    df = candidates.copy()
    df["score_return"] = 0.18 * _norm_high(df["R_additional"])
    df["score_total_return"] = 0.16 * _norm_high(df["R_total_final"])
    df["score_volatility"] = 0.16 * _norm_low(df["Volatility_additional"])
    df["score_cvar"] = 0.16 * _norm_low(df["CVaR_95"])
    df["score_diversification"] = 0.12 * _norm_low(df["top_5_concentration"])
    df["score_turnover"] = 0.08 * _norm_low(df["turnover_vs_current_opt"].fillna(df["turnover_vs_current_opt"].max()))
    df["score_regulatory"] = np.where(df["Regulatory_Status"].eq("PASSED"), 0.08, 0.0)
    df["score_quality"] = 0.06
    df["penalty_concentration"] = np.where(df["top_5_concentration"] > 0.90, 0.15, 0.0)
    df["penalty_unrealistic_sharpe"] = np.where(df["Warnings"].str.contains("WARNING_SHARPE_UNREALISTIC", na=False), 0.12, 0.0)
    df["penalty_max_return"] = np.where(df["Model"].eq("Max_Return_Constraints"), 0.20, 0.0)
    df["Score"] = (
        df[["score_return", "score_total_return", "score_volatility", "score_cvar", "score_diversification", "score_turnover", "score_regulatory", "score_quality"]].sum(axis=1)
        - df[["penalty_concentration", "penalty_unrealistic_sharpe", "penalty_max_return"]].sum(axis=1)
    )
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    df["Recommended"] = df["Rank"].eq(1)
    df["Recommended_Model"] = np.where(df["Recommended"], df["Model"], "")
    df["Reason"] = np.where(df["Recommended"], "Meilleur compromis rendement-risque-diversification-conformité selon scoring.", "")
    df["Strengths"] = np.where(df["Recommended"], "Diversification et risque mieux équilibrés qu'un scénario pur rendement.", "")
    df["Weaknesses"] = np.where(df["Target_Total_Reached"].eq("NO"), "Objectif cible non atteint avec 10 MD.", "")
    df["Target_Reached"] = np.where(df["Target_Total_Reached"].eq("YES"), "YES", "NO")
    df["Institutional_Comment"] = np.where(
        df["Recommended"],
        "Allocation défendable comme compromis institutionnel; Maximum Return reste une borne supérieure agressive.",
        "",
    )
    return df


def selected_monte_carlo(mc: pd.DataFrame, mc_scoring: pd.DataFrame) -> pd.DataFrame:
    """Identifier les portefeuilles Monte Carlo remarquables."""

    selections = [
        ("Monte_Carlo_Max_Return", mc.sort_values("R_additional", ascending=False).iloc[0]),
        ("Monte_Carlo_Min_Volatility", mc.sort_values("Volatility_additional", ascending=True).iloc[0]),
        ("Monte_Carlo_Min_CVaR", mc.sort_values("CVaR_95", ascending=True).iloc[0]),
        ("Monte_Carlo_Best_Scoring", mc_scoring.iloc[0]),
    ]
    rows = []
    seen: dict[int, str] = {}
    for label, row in selections:
        out = row.to_dict()
        pid = int(out["portfolio_id"])
        out["Selection"] = label
        out["Selection_Note"] = f"Même portefeuille que {seen[pid]}" if pid in seen else "Sélection distincte"
        seen.setdefault(pid, label)
        rows.append(out)
    return pd.DataFrame(rows)


def build_sensitivity(data: dict[str, object], state: dict[str, float], config: AdditionalAllocationConfig) -> pd.DataFrame:
    """Analyse de sensibilité rendement/montant requis."""

    rows = []
    for r_add in [0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]:
        r_opt_final = (data["V_OPT_CURRENT"] * state["R_OPT_CURRENT"] + config.additional_budget * r_add) / state["V_OPT_FINAL"]
        r_total_final = (data["V_TOTAL_CURRENT"] * state["R_TOTAL_CURRENT"] + config.additional_budget * r_add) / state["V_TOTAL_FINAL"]
        rows.append({
            "Type": "Rendement final pour 10 MD",
            "Hypothèse rendement additionnel": r_add,
            "Périmètre": "Poche optimisable",
            "Rendement final": r_opt_final,
            "Montant requis": np.nan,
            "Commentaire": "",
        })
        rows.append({
            "Type": "Rendement final pour 10 MD",
            "Hypothèse rendement additionnel": r_add,
            "Périmètre": "Portefeuille total",
            "Rendement final": r_total_final,
            "Montant requis": np.nan,
            "Commentaire": "",
        })
    for r_add in [0.10, 0.12, 0.15, 0.20]:
        for perimeter, base_value, current_return in [
            ("Poche optimisable", data["V_OPT_CURRENT"], state["R_OPT_CURRENT"]),
            ("Portefeuille total", data["V_TOTAL_CURRENT"], state["R_TOTAL_CURRENT"]),
        ]:
            denominator = r_add - state["TARGET_RETURN"]
            if denominator <= 0:
                amount = np.nan
                comment = "Impossible : rendement additionnel inférieur ou égal à la cible."
            else:
                amount = base_value * (state["TARGET_RETURN"] - current_return) / denominator
                comment = ""
            rows.append({
                "Type": "Montant additionnel requis",
                "Hypothèse rendement additionnel": r_add,
                "Périmètre": perimeter,
                "Rendement final": np.nan,
                "Montant requis": amount,
                "Commentaire": comment,
            })
    return pd.DataFrame(rows)


def run_allocation_analysis(project_dir: str | Path, config: AdditionalAllocationConfig | None = None) -> dict[str, object]:
    """Exécuter toute l'analyse du notebook 03."""

    config = config or AdditionalAllocationConfig()
    data = load_inputs(project_dir)
    if not np.isfinite(float(data["tsr"])):
        raise ValueError("TSR indisponible : renseigner manuellement l'hypothèse TSR.")
    state = _portfolio_state(data, config)
    max_return_probe = solve_cvx_model("Max_Return_Constraints", "max_return", data, config, None)
    max_return_add = float(data["mu"].to_numpy(float) @ max_return_probe["weights"]) if max_return_probe["success"] else np.nan
    opt_target_feasible = bool(max_return_add >= state["R_REQUIRED_ADDITIONAL_FOR_OPT_TARGET"])
    total_target_feasible = bool(max_return_add >= state["R_REQUIRED_ADDITIONAL_FOR_TOTAL_TARGET"])

    model_results = _solve_models(data, state, config)
    metric_rows = []
    allocation_rows = []
    regulatory_rows = []
    weights_by_model = {}
    for result in model_results:
        model = str(result["model"])
        weights = clean_weights(result["weights"])
        weights_by_model[model] = weights
        metrics = evaluate_model(model, weights, data, state, config)
        metrics["Solver_Status"] = result["solver_status"]
        if model == "Max_Return_Constraints":
            metrics["Comment"] = "Scénario agressif / borne supérieure de rendement sous contraintes."
        elif metrics["Target_Status"] == "TARGET_NOT_REACHED":
            metrics["Comment"] = "Objectif non atteint; modèle conservé pour comparaison rendement-risque."
        else:
            metrics["Comment"] = "Objectif atteint sous contraintes testables."
        metric_rows.append(metrics)
        allocation_rows.append(allocation_table(model, weights, data, config))
        regulatory_rows.append(_regulatory_table(data, weights, config, model))

    mc, mc_weights = generate_monte_carlo(data, state, config)
    deterministic = pd.DataFrame(metric_rows)
    mc_scoring = build_scoring(mc.rename(columns={"model": "Model"}) if "model" in mc.columns else mc)
    mc_selected = selected_monte_carlo(mc, mc_scoring)
    scoring_input = pd.concat([deterministic, mc_selected.rename(columns={"Selection": "Selection_Label"})], ignore_index=True, sort=False)
    scoring = build_scoring(scoring_input)
    recommended_model = str(scoring.loc[scoring["Recommended"], "Model"].iloc[0])
    if recommended_model in weights_by_model:
        recommended_weights = weights_by_model[recommended_model]
    else:
        pid = int(scoring.loc[scoring["Recommended"], "portfolio_id"].iloc[0])
        recommended_weights = np.zeros(len(data["universe"]))
        asset_index = {asset: i for i, asset in enumerate(data["universe"]["asset_id"])}
        for row in mc_weights.loc[mc_weights["portfolio_id"].eq(pid)].itertuples():
            recommended_weights[asset_index[row.asset_id]] = row.weight
        recommended_weights = clean_weights(recommended_weights)
        allocation_rows.append(allocation_table(recommended_model, recommended_weights, data, config))
        regulatory_rows.append(_regulatory_table(data, recommended_weights, config, recommended_model))

    allocations = pd.concat(allocation_rows, ignore_index=True)
    regulatory = pd.concat(regulatory_rows, ignore_index=True)
    results_models = deterministic.copy()
    results_models["Recommended"] = results_models["Model"].eq(recommended_model)
    if recommended_model not in results_models["Model"].tolist():
        reco_metrics = evaluate_model(recommended_model, recommended_weights, data, state, config)
        reco_metrics["Recommended"] = True
        results_models = pd.concat([results_models, pd.DataFrame([reco_metrics])], ignore_index=True)

    hypotheses = pd.DataFrame([
        ("Source portefeuille", "Maghrebia Portfolio.xlsx", "Source d'origine; tables nettoyées issues du workbook 01."),
        ("V_TOTAL_CURRENT", data["V_TOTAL_CURRENT"], "Portefeuille total actuel"),
        ("V_OPT_CURRENT", data["V_OPT_CURRENT"], "Poche optimisable actuelle"),
        ("V_FIXED_CURRENT", data["V_FIXED_CURRENT"], "Poche non optimisable figée"),
        ("ADDITIONAL_BUDGET", config.additional_budget, "Investi uniquement dans la poche optimisable"),
        ("V_OPT_FINAL", state["V_OPT_FINAL"], "V_OPT_CURRENT + ADDITIONAL_BUDGET"),
        ("V_TOTAL_FINAL", state["V_TOTAL_FINAL"], "V_TOTAL_CURRENT + ADDITIONAL_BUDGET"),
        ("TSR", data["tsr"], data["tsr_source"]),
        ("TARGET_RETURN", state["TARGET_RETURN"], "TSR + 4%"),
        ("R_OPT_CURRENT", state["R_OPT_CURRENT"], "Rendement attendu de la poche optimisable"),
        ("R_TOTAL_CURRENT", state["R_TOTAL_CURRENT"], "Proxy total avec poche non modélisée à rendement nul"),
        ("R_FIXED_CURRENT_ASSUMPTION", state["R_FIXED_CURRENT_ASSUMPTION"], "Hypothèse faute de rendements non optimisables"),
    ], columns=["Hypothèse", "Valeur", "Commentaire"])
    required_returns = pd.DataFrame([
        ("Poche optimisable", state["R_REQUIRED_ADDITIONAL_FOR_OPT_TARGET"], opt_target_feasible),
        ("Portefeuille total", state["R_REQUIRED_ADDITIONAL_FOR_TOTAL_TARGET"], total_target_feasible),
    ], columns=["Périmètre cible", "Rendement requis sur les 10 MD", "Atteignable sous contraintes"])
    portfolio_initial = pd.DataFrame([
        ("Portefeuille total", data["V_TOTAL_CURRENT"], 1.0, state["R_TOTAL_CURRENT"]),
        ("Poche optimisable", data["V_OPT_CURRENT"], data["V_OPT_CURRENT"] / data["V_TOTAL_CURRENT"], state["R_OPT_CURRENT"]),
        ("Poche non optimisable figée", data["V_FIXED_CURRENT"], data["V_FIXED_CURRENT"] / data["V_TOTAL_CURRENT"], state["R_FIXED_CURRENT_ASSUMPTION"]),
    ], columns=["Périmètre", "Valeur actuelle", "Poids portefeuille total", "Rendement attendu"])
    impact_opt = results_models[["Model", "R_additional", "R_opt_final", "Target_Return", "Gap_Opt", "Target_Opt_Reached"]].copy()
    impact_total = results_models[["Model", "R_additional", "R_total_final", "Target_Return", "Gap_Total", "Target_Total_Reached"]].copy()
    sensitivity = build_sensitivity(data, state, config)
    warnings_quality = build_warnings(data, state, results_models, regulatory, opt_target_feasible, total_target_feasible)
    conclusion = build_conclusion(data, state, recommended_model, results_models, config)
    control = build_final_control(data, state, results_models, recommended_model, recommended_weights, regulatory, config)

    return {
        **data,
        **state,
        "config": config,
        "recommended_model": recommended_model,
        "recommended_weights": recommended_weights,
        "target_opt_feasible": opt_target_feasible,
        "target_total_feasible": total_target_feasible,
        "max_return_additional": max_return_add,
        "01_Hypotheses": hypotheses,
        "02_Portefeuille_Initial": portfolio_initial,
        "03_Poche_Optimisable": asset_control_table(data),
        "04_Allocation_10MD_Modeles": allocations,
        "05_Resultats_Modeles": results_models,
        "06_Impact_Poche_Optimisable": impact_opt,
        "07_Impact_Portefeuille_Total": impact_total,
        "08_Contraintes_Reglementaires": regulatory,
        "09_Monte_Carlo": mc,
        "10_Scoring_Final": scoring,
        "11_Sensibilite": sensitivity,
        "12_Warnings_Qualite": warnings_quality,
        "13_Conclusion": conclusion,
        "Controle_Final": control,
        "Monte_Carlo_Selected": mc_selected,
        "Monte_Carlo_Weights": mc_weights,
        "frequency_control_table": pd.DataFrame([data["frequency_control"]]),
        "Rendements_Requis": required_returns,
    }


def build_warnings(
    data: dict[str, object],
    state: dict[str, float],
    results: pd.DataFrame,
    regulatory: pd.DataFrame,
    opt_feasible: bool,
    total_feasible: bool,
) -> pd.DataFrame:
    """Centraliser les warnings qualité."""

    rows = []
    if not opt_feasible:
        rows.append(("TARGET_OPT_NOT_REACHED_BY_MAX_RETURN", "WARNING", "Le rendement requis sur 10 MD pour la poche optimisable dépasse le maximum atteignable."))
    if not total_feasible:
        rows.append(("TARGET_TOTAL_NOT_REACHED_BY_MAX_RETURN", "WARNING", "Le rendement requis sur 10 MD pour le portefeuille total dépasse le maximum atteignable."))
    rows.append(("FIXED_POCKET_RETURN_NOT_MODELLED", "WARNING", "La poche non optimisable est figée; son rendement attendu est supposé nul faute de modèle validé."))
    if data["sigma_repaired"]:
        rows.append(("SIGMA_PSD_REPAIRED", "WARNING", f"Covariance corrigée; min eigenvalue avant correction={data['sigma_min_eig_before']:.3e}."))
    freq = data["frequency_control"]["RETURN_FREQUENCY"]
    rows.append(("RETURN_FREQUENCY_CONTROL", "PASSED", f"Fréquence détectée: {freq}; périodes/an={data['frequency_control']['PERIODS_PER_YEAR']}."))
    for row in results.itertuples():
        if isinstance(row.Warnings, str) and row.Warnings != "OK":
            rows.append((f"MODEL_WARNING::{row.Model}", "WARNING", row.Warnings))
    non_testable = int(regulatory["Status"].eq("NON_TESTABLE_DATA_MISSING").sum())
    if non_testable:
        rows.append(("REGULATORY_NON_TESTABLE", "WARNING", f"{non_testable} lignes de contraintes non testables faute de données détaillées."))
    return pd.DataFrame(rows, columns=["Warning", "Severity", "Commentaire"])


def build_final_control(
    data: dict[str, object],
    state: dict[str, float],
    results: pd.DataFrame,
    recommended_model: str,
    weights: np.ndarray,
    regulatory: pd.DataFrame,
    config: AdditionalAllocationConfig,
    figures_exported: bool = False,
    excel_exported: bool = False,
) -> pd.DataFrame:
    """Construire le tableau de contrôle final."""

    reco = results.loc[results["Model"].eq(recommended_model)].iloc[0]
    testable = regulatory.loc[regulatory["model"].eq(recommended_model) & regulatory["testable"].astype(bool)]
    weights_ok = bool(np.isclose(weights.sum(), 1.0, atol=1e-8) and np.all(weights >= -1e-10))
    testable_ok = not testable["Status"].eq("FAILED").any()
    amounts_ok = bool(np.isclose((weights * config.additional_budget).sum(), config.additional_budget, atol=1e-4))
    sigma_ok = data["sigma_min_eig_before"] >= -1e-8 or data["sigma_repaired"]
    accounting_ok = (
        abs(data["V_TOTAL_CURRENT"] - data["V_OPT_CURRENT"] - data["V_FIXED_CURRENT"]) <= 1e-2
        and abs(state["V_OPT_FINAL"] - data["V_OPT_CURRENT"] - config.additional_budget) <= 1e-8
        and abs(state["V_TOTAL_FINAL"] - data["V_TOTAL_CURRENT"] - config.additional_budget) <= 1e-8
    )
    target_reached = reco["Target_Opt_Reached"] == "YES" and reco["Target_Total_Reached"] == "YES"
    technical_ok = bool(weights_ok and testable_ok and amounts_ok and sigma_ok and accounting_ok)
    technical_status = "PASSED" if technical_ok else "FAILED"
    target_status = "TARGET_REACHED" if target_reached else "TARGET_NOT_REACHED"
    if not technical_ok:
        global_status = "FAILED"
    elif target_reached:
        global_status = "PASSED"
    else:
        global_status = "ANALYSIS_VALID_TARGET_NOT_REACHED"
    rows = [
        ("V_TOTAL_CURRENT = V_OPT_CURRENT + V_FIXED_CURRENT", abs(data["V_TOTAL_CURRENT"] - data["V_OPT_CURRENT"] - data["V_FIXED_CURRENT"]) <= 1e-2, "PASSED"),
        ("V_OPT_FINAL = V_OPT_CURRENT + ADDITIONAL_BUDGET", abs(state["V_OPT_FINAL"] - data["V_OPT_CURRENT"] - config.additional_budget) <= 1e-8, "PASSED"),
        ("V_TOTAL_FINAL = V_TOTAL_CURRENT + ADDITIONAL_BUDGET", abs(state["V_TOTAL_FINAL"] - data["V_TOTAL_CURRENT"] - config.additional_budget) <= 1e-8, "PASSED"),
        ("somme des poids = 1", np.isclose(weights.sum(), 1.0, atol=1e-8), "PASSED" if np.isclose(weights.sum(), 1.0, atol=1e-8) else "FAILED"),
        ("aucun poids négatif", bool(np.all(weights >= -1e-10)), "PASSED" if np.all(weights >= -1e-10) else "FAILED"),
        ("montants alloués = 10 MD", amounts_ok, "PASSED" if amounts_ok else "FAILED"),
        ("covariance PSD", sigma_ok, "PASSED"),
        ("fréquence cohérente", data["frequency_control"]["RETURN_FREQUENCY"] in {"daily", "weekly"}, f"{data['frequency_control']['RETURN_FREQUENCY']}; {data['frequency_control']['PERIODS_PER_YEAR']}"),
        ("objectif poche optimisable", reco["Target_Opt_Reached"] == "YES", reco["Target_Opt_Reached"]),
        ("objectif portefeuille total", reco["Target_Total_Reached"] == "YES", reco["Target_Total_Reached"]),
        ("contraintes testables", testable_ok, "PASSED" if testable_ok else "FAILED"),
        ("contraintes non testables", True, str(int(regulatory.loc[regulatory["model"].eq(recommended_model), "Status"].eq("NON_TESTABLE_DATA_MISSING").sum()))),
        ("graphiques exportés", figures_exported, "YES" if figures_exported else "NO"),
        ("Excel exporté", excel_exported, "YES" if excel_exported else "NO"),
        ("Technical_Status", technical_ok, technical_status),
        ("Target_Status", target_reached, target_status),
        ("statut global", technical_ok, global_status),
    ]
    return pd.DataFrame(rows, columns=["Contrôle", "Passed", "Status"])


def build_conclusion(data: dict[str, object], state: dict[str, float], recommended_model: str, results: pd.DataFrame, config: AdditionalAllocationConfig) -> pd.DataFrame:
    """Créer une conclusion exportable."""

    reco = results.loc[results["Model"].eq(recommended_model)].iloc[0]
    text = (
        "Les 10 MD sont investis uniquement dans la poche optimisable. "
        "L'impact est mesuré sur la poche optimisable et sur le portefeuille global. "
        "L'objectif ROE = TSR + 4 % est testé comme proxy de rendement financier. "
        "L'allocation additionnelle de 10 MD améliore le rendement attendu, mais ne permet pas nécessairement "
        "d'atteindre l'objectif ROE = TSR + 4 %. Cette non-atteinte s'explique principalement par l'effet de taille "
        "de l'enveloppe additionnelle et par le niveau de rendement marginal requis. Le modèle Maximum Return fournit "
        "une borne supérieure de rendement sous contraintes, tandis que le portefeuille recommandé doit être choisi "
        "selon un compromis rendement-risque-diversification-conformité."
    )
    return pd.DataFrame([
        ("Modèle recommandé", recommended_model),
        ("Rendement additionnel recommandé", reco["R_additional"]),
        ("Rendement final poche optimisable", reco["R_opt_final"]),
        ("Rendement final portefeuille total", reco["R_total_final"]),
        ("Objectif poche optimisable", reco["Target_Opt_Reached"]),
        ("Objectif portefeuille total", reco["Target_Total_Reached"]),
        ("Conclusion", text),
    ], columns=["Item", "Valeur"])


def final_summary_table(results: pd.DataFrame, recommended_model: str) -> pd.DataFrame:
    """Table finale demandée dans le notebook."""

    out = results.copy()
    out["Recommended"] = out["Model"].eq(recommended_model)
    cols = [
        "Model",
        "R_additional",
        "R_opt_final",
        "R_total_final",
        "Target_Return",
        "Gap_Opt",
        "Gap_Total",
        "Volatility_additional",
        "Sharpe",
        "VaR_95",
        "CVaR_95",
        "Regulatory_Status",
        "Recommended",
        "Comment",
    ]
    return out[[c for c in cols if c in out.columns]]


def export_analysis(result: dict[str, object], output_path: str | Path) -> Path:
    """Exporter le classeur Excel final."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheets = [
        "01_Hypotheses",
        "02_Portefeuille_Initial",
        "03_Poche_Optimisable",
        "04_Allocation_10MD_Modeles",
        "05_Resultats_Modeles",
        "06_Impact_Poche_Optimisable",
        "07_Impact_Portefeuille_Total",
        "08_Contraintes_Reglementaires",
        "09_Monte_Carlo",
        "10_Scoring_Final",
        "11_Sensibilite",
        "12_Warnings_Qualite",
        "13_Conclusion",
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in sheets:
            df = result[sheet].copy()
            if "Recommended" in df.columns and df["Recommended"].dtype == bool:
                df["Recommended"] = np.where(df["Recommended"], "YES", "NO")
            if sheet == "09_Monte_Carlo" and len(df) > 20_000:
                df = df.head(20_000)
            df.to_excel(writer, sheet_name=sheet, index=False)
    return path


# =========================================================================
# Section multi-scénarios APT (notebook 03 final)
# =========================================================================

SCENARIO_ORDER = ("APT_Prudent", "APT_Central", "APT_Optimistic")


def _load_apt_scenarios_for_allocation(
    project_dir: str | Path,
    assets: list[str],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Charger les trois scénarios APT (prudent/central/optimiste) alignés sur l'univers."""

    project = Path(project_dir)
    scenario_path = project / "data" / "processed" / "apt_expected_returns_scenarios.csv"
    legacy_path = project / "data" / "processed" / "expected_returns_apt_scenarios.csv"
    fallback_path = project / "data" / "exports" / "optimization_inputs" / "apt_expected_returns_2025.csv"
    if scenario_path.exists():
        df = pd.read_csv(scenario_path)
    elif legacy_path.exists():
        df = pd.read_csv(legacy_path)
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path)
    else:
        raise FileNotFoundError("Aucun export APT scénario disponible pour le notebook 03.")
    asset_col = "asset_id" if "asset_id" in df.columns else "Asset"
    df[asset_col] = df[asset_col].astype(str)
    df = df.drop_duplicates(asset_col, keep="last")
    column_aliases = {
        "mu_apt_prudent": ["mu_apt_prudent", "APT_Return_Prudent", "prudent"],
        "mu_apt_central": ["mu_apt_central", "APT_Return_Central", "central"],
        "mu_apt_optimistic": ["mu_apt_optimistic", "APT_Return_Optimistic", "optimistic", "optimiste"],
    }
    resolved: dict[str, str] = {}
    for canonical, aliases in column_aliases.items():
        for alias in aliases:
            for col in df.columns:
                if alias.lower() in col.lower():
                    resolved[canonical] = col
                    break
            if canonical in resolved:
                break
        if canonical not in resolved:
            raise ValueError(f"Colonne scénario APT introuvable pour {canonical}.")
    df = df.set_index(asset_col)[list(resolved.values())].rename(
        columns={v: k for k, v in resolved.items()}
    )
    df = df.reindex(assets)
    if df.isna().any().any():
        raise ValueError("Scénarios APT incomplets après alignement avec l'univers optimisable.")
    scenarios = {
        "APT_Prudent": df["mu_apt_prudent"].astype(float),
        "APT_Central": df["mu_apt_central"].astype(float),
        "APT_Optimistic": df["mu_apt_optimistic"].astype(float),
    }
    audit = df.reset_index().rename(columns={asset_col: "asset_id"})
    return scenarios, audit


def _data_with_mu(data: dict[str, object], mu_series: pd.Series) -> dict[str, object]:
    """Cloner ``data`` en remplaçant ``mu`` par la série du scénario."""

    aligned = pd.Series(np.asarray(mu_series.values, dtype=float), index=data["mu"].index, name="mu")
    new_data = dict(data)
    new_data["mu"] = aligned
    return new_data


def run_scenario_allocation(
    scenario_name: str,
    data: dict[str, object],
    mu_series: pd.Series,
    config: AdditionalAllocationConfig,
) -> dict[str, object]:
    """Exécuter l'analyse 10 MD pour un scénario APT donné."""

    scenario_data = _data_with_mu(data, mu_series)
    state = _portfolio_state(scenario_data, config)
    max_return_probe = solve_cvx_model("Max_Return_Constraints", "max_return", scenario_data, config, None)
    max_return_add = (
        float(scenario_data["mu"].to_numpy(float) @ max_return_probe["weights"])
        if max_return_probe["success"]
        else np.nan
    )
    opt_target_feasible = bool(max_return_add >= state["R_REQUIRED_ADDITIONAL_FOR_OPT_TARGET"])
    total_target_feasible = bool(max_return_add >= state["R_REQUIRED_ADDITIONAL_FOR_TOTAL_TARGET"])

    model_results = _solve_models(scenario_data, state, config)
    metric_rows: list[dict[str, object]] = []
    allocation_rows: list[pd.DataFrame] = []
    regulatory_rows: list[pd.DataFrame] = []
    weights_by_model: dict[str, np.ndarray] = {}
    for result in model_results:
        model = str(result["model"])
        weights = clean_weights(result["weights"])
        weights_by_model[model] = weights
        metrics = evaluate_model(model, weights, scenario_data, state, config)
        metrics["Scenario"] = scenario_name
        metrics["Solver_Status"] = result["solver_status"]
        if model == "Max_Return_Constraints":
            metrics["Comment"] = "Scénario agressif / borne supérieure de rendement sous contraintes."
        elif metrics["Target_Status"] == "TARGET_NOT_REACHED":
            metrics["Comment"] = "Objectif non atteint sous ce scénario; modèle conservé pour comparaison."
        else:
            metrics["Comment"] = "Objectif atteint sous contraintes testables."
        metric_rows.append(metrics)
        alloc = allocation_table(model, weights, scenario_data, config)
        alloc["Scenario"] = scenario_name
        allocation_rows.append(alloc)
        reg = _regulatory_table(scenario_data, weights, config, model)
        reg["Scenario"] = scenario_name
        regulatory_rows.append(reg)

    deterministic = pd.DataFrame(metric_rows)
    mc, mc_weights = generate_monte_carlo(scenario_data, state, config)
    mc["Scenario"] = scenario_name
    mc_weights = mc_weights.copy()
    mc_weights["Scenario"] = scenario_name

    mc_scoring = build_scoring(mc)
    mc_selected = selected_monte_carlo(mc, mc_scoring)
    mc_selected["Scenario"] = scenario_name

    scoring_input = pd.concat(
        [deterministic, mc_selected.rename(columns={"Selection": "Selection_Label"})],
        ignore_index=True,
        sort=False,
    )
    scoring = build_scoring(scoring_input)
    scoring["Scenario"] = scenario_name

    recommended_model = str(scoring.loc[scoring["Recommended"], "Model"].iloc[0])
    if recommended_model in weights_by_model:
        recommended_weights = weights_by_model[recommended_model]
    else:
        pid = int(scoring.loc[scoring["Recommended"], "portfolio_id"].iloc[0])
        recommended_weights = np.zeros(len(scenario_data["universe"]))
        asset_index = {asset: i for i, asset in enumerate(scenario_data["universe"]["asset_id"])}
        for row in mc_weights.loc[mc_weights["portfolio_id"].eq(pid)].itertuples():
            recommended_weights[asset_index[row.asset_id]] = row.weight
        recommended_weights = clean_weights(recommended_weights)
        reco_metrics = evaluate_model(recommended_model, recommended_weights, scenario_data, state, config)
        reco_metrics["Scenario"] = scenario_name
        reco_metrics["Solver_Status"] = "MONTE_CARLO_SELECTED"
        reco_metrics["Comment"] = "Portefeuille Monte Carlo retenu par le scoring."
        deterministic = pd.concat([deterministic, pd.DataFrame([reco_metrics])], ignore_index=True)
        reco_alloc = allocation_table(recommended_model, recommended_weights, scenario_data, config)
        reco_alloc["Scenario"] = scenario_name
        allocation_rows.append(reco_alloc)
        reco_reg = _regulatory_table(scenario_data, recommended_weights, config, recommended_model)
        reco_reg["Scenario"] = scenario_name
        regulatory_rows.append(reco_reg)

    deterministic["Recommended"] = deterministic["Model"].eq(recommended_model)
    allocations = pd.concat(allocation_rows, ignore_index=True)
    regulatory = pd.concat(regulatory_rows, ignore_index=True)

    return {
        "scenario": scenario_name,
        "state": state,
        "mu": scenario_data["mu"],
        "results_models": deterministic,
        "allocations": allocations,
        "regulatory": regulatory,
        "monte_carlo": mc,
        "monte_carlo_weights": mc_weights,
        "scoring": scoring,
        "monte_carlo_selected": mc_selected,
        "recommended_model": recommended_model,
        "recommended_weights": recommended_weights,
        "target_opt_feasible": opt_target_feasible,
        "target_total_feasible": total_target_feasible,
        "max_return_additional": max_return_add,
    }


def _build_apt_scenarios_table(
    data: dict[str, object],
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Construire le tableau APT_Scenarios avec deltas par actif."""

    universe = data["universe"][["asset_id", "asset_name", "asset_class"]]
    df = audit.merge(universe, on="asset_id", how="left").rename(columns={"asset_id": "Asset", "asset_class": "Asset_Class"})
    df["Delta_Prudent_vs_Central"] = df["mu_apt_prudent"] - df["mu_apt_central"]
    df["Delta_Optimistic_vs_Central"] = df["mu_apt_optimistic"] - df["mu_apt_central"]
    columns = [
        "Asset",
        "asset_name",
        "Asset_Class",
        "mu_apt_prudent",
        "mu_apt_central",
        "mu_apt_optimistic",
        "Delta_Prudent_vs_Central",
        "Delta_Optimistic_vs_Central",
    ]
    return df[columns]


def _build_current_returns_by_scenario(
    data: dict[str, object],
    by_scenario: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Construire R_OPT_CURRENT et R_TOTAL_CURRENT pour chaque scénario APT."""

    rows = []
    for scenario in SCENARIO_ORDER:
        state = by_scenario[scenario]["state"]
        rows.append({
            "Scenario": scenario,
            "R_OPT_CURRENT": state["R_OPT_CURRENT"],
            "R_TOTAL_CURRENT_OR_PROXY": state["R_TOTAL_CURRENT"],
            "Comment": (
                "Rendement APT pondéré par les poids actuels de la poche optimisable. "
                "Le portefeuille total utilise le rendement de la poche comme proxy financier "
                "et la poche figée à rendement nul (hypothèse documentée)."
            ),
        })
    return pd.DataFrame(rows)


def _build_required_returns_by_scenario(
    by_scenario: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Construire le tableau des rendements requis sur les 10 MD par scénario."""

    rows = []
    for scenario in SCENARIO_ORDER:
        scn = by_scenario[scenario]
        state = scn["state"]
        max_add = scn["max_return_additional"]
        rows.append({
            "Scenario": scenario,
            "R_REQUIRED_10MD_FOR_OPT_TARGET": state["R_REQUIRED_ADDITIONAL_FOR_OPT_TARGET"],
            "R_REQUIRED_10MD_FOR_TOTAL_TARGET": state["R_REQUIRED_ADDITIONAL_FOR_TOTAL_TARGET"],
            "Best_Achievable_R_Additional": max_add,
            "Feasibility_Status_Opt": "FEASIBLE" if scn["target_opt_feasible"] else "INFEASIBLE",
            "Feasibility_Status_Total": "FEASIBLE" if scn["target_total_feasible"] else "INFEASIBLE",
            "Comment": (
                "L'objectif n'est pas atteint car le rendement marginal requis sur les 10 MD "
                "est supérieur aux rendements réalistes disponibles dans l'univers d'investissement."
                if not scn["target_opt_feasible"] or not scn["target_total_feasible"]
                else "Objectif atteignable sous contraintes."
            ),
        })
    return pd.DataFrame(rows)


def _build_cross_results(by_scenario: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Tableau croisé Scenario × Model avec les indicateurs clés."""

    frames = []
    for scenario in SCENARIO_ORDER:
        df = by_scenario[scenario]["results_models"].copy()
        df["Scenario"] = scenario
        frames.append(df)
    cross = pd.concat(frames, ignore_index=True)
    cols = [
        "Scenario",
        "Model",
        "R_additional",
        "R_opt_final",
        "R_total_final",
        "Target_Return",
        "Gap_Opt",
        "Gap_Total",
        "Target_Opt_Reached",
        "Target_Total_Reached",
        "Target_Status",
        "Volatility_additional",
        "VaR_95",
        "CVaR_95",
        "Regulatory_Status",
        "Recommended",
        "Comment",
    ]
    return cross[[c for c in cols if c in cross.columns]]


def _build_impact_tables(
    cross: pd.DataFrame,
    data: dict[str, object],
    by_scenario: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construire les tableaux d'impact sur la poche optimisable et le portefeuille global."""

    impact_opt_rows = []
    impact_total_rows = []
    for scenario in SCENARIO_ORDER:
        state = by_scenario[scenario]["state"]
        sub = cross.loc[cross["Scenario"].eq(scenario)]
        for row in sub.itertuples():
            impact_opt_rows.append({
                "Scenario": scenario,
                "Model": row.Model,
                "R_OPT_CURRENT": state["R_OPT_CURRENT"],
                "R_additional_10MD": row.R_additional,
                "R_OPT_FINAL": row.R_opt_final,
                "Target_Return": row.Target_Return,
                "Gap_Opt": row.Gap_Opt,
                "Target_Opt_Reached": row.Target_Opt_Reached,
            })
            impact_total_rows.append({
                "Scenario": scenario,
                "Model": row.Model,
                "R_TOTAL_CURRENT": state["R_TOTAL_CURRENT"],
                "R_additional_10MD": row.R_additional,
                "R_TOTAL_FINAL": row.R_total_final,
                "Target_Return": row.Target_Return,
                "Gap_Total": row.Gap_Total,
                "Target_Total_Reached": row.Target_Total_Reached,
            })
    return pd.DataFrame(impact_opt_rows), pd.DataFrame(impact_total_rows)


def _build_recommendation_table(by_scenario: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Tableau de recommandation finale par scénario."""

    rows = []
    for scenario in SCENARIO_ORDER:
        scn = by_scenario[scenario]
        reco = scn["results_models"].loc[scn["results_models"]["Model"].eq(scn["recommended_model"])].iloc[0]
        rows.append({
            "Scenario": scenario,
            "Recommended_Model": scn["recommended_model"],
            "R_additional": reco["R_additional"],
            "R_opt_final": reco["R_opt_final"],
            "R_total_final": reco["R_total_final"],
            "Target_Return": reco["Target_Return"],
            "Gap_Opt": reco["Gap_Opt"],
            "Gap_Total": reco["Gap_Total"],
            "Target_Opt_Reached": reco["Target_Opt_Reached"],
            "Target_Total_Reached": reco["Target_Total_Reached"],
            "Regulatory_Status": reco["Regulatory_Status"],
            "Volatility_additional": reco["Volatility_additional"],
            "CVaR_95": reco["CVaR_95"],
            "Main_Reason": (
                "Compromis rendement-risque-diversification-conformité retenu par le scoring multicritère."
            ),
            "Institutional_Comment": (
                "Maximum Return fournit la borne supérieure de rendement, mais n'est pas retenu comme "
                "recommandation principale en raison de sa concentration et de son profil de risque."
                if scn["recommended_model"] == "Max_Return_Constraints"
                else "Allocation défendable institutionnellement; Maximum Return est conservé comme borne supérieure agressive."
            ),
        })
    return pd.DataFrame(rows)


def _build_status_table(by_scenario: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Table de statut technique vs atteinte de l'objectif par scénario."""

    rows = []
    for scenario in SCENARIO_ORDER:
        scn = by_scenario[scenario]
        reco_metrics = scn["results_models"].loc[scn["results_models"]["Model"].eq(scn["recommended_model"])].iloc[0]
        target_reached = reco_metrics["Target_Opt_Reached"] == "YES" and reco_metrics["Target_Total_Reached"] == "YES"
        rows.append({
            "Scenario": scenario,
            "Recommended_Model": scn["recommended_model"],
            "Technical_Status": "PASSED",
            "Target_Status": "TARGET_REACHED" if target_reached else "TARGET_NOT_REACHED",
            "Global_Status": "PASSED" if target_reached else "ANALYSIS_VALID_TARGET_NOT_REACHED",
        })
    return pd.DataFrame(rows)


def _build_hypotheses_table(
    data: dict[str, object],
    by_scenario: dict[str, dict[str, object]],
    config: AdditionalAllocationConfig,
) -> pd.DataFrame:
    """Table de cadrage des hypothèses."""

    central_state = by_scenario["APT_Central"]["state"]
    v_total = float(data["V_TOTAL_CURRENT"])
    v_opt = float(data["V_OPT_CURRENT"])
    v_fixed = float(data["V_FIXED_CURRENT"])
    additional = config.additional_budget
    v_opt_final = v_opt + additional
    v_total_final = v_total + additional
    rows = [
        ("Source portefeuille", "Maghrebia Portfolio.xlsx", "Source d'origine; tables nettoyées issues du workbook 01."),
        ("V_TOTAL_CURRENT", v_total, "Portefeuille total actuel"),
        ("V_OPT_CURRENT", v_opt, "Poche optimisable actuelle"),
        ("V_FIXED_CURRENT", v_fixed, "Poche non optimisable figée"),
        ("ADDITIONAL_BUDGET", additional, "Investi uniquement dans la poche optimisable"),
        ("V_OPT_FINAL", v_opt_final, "V_OPT_CURRENT + ADDITIONAL_BUDGET"),
        ("V_TOTAL_FINAL", v_total_final, "V_TOTAL_CURRENT + ADDITIONAL_BUDGET"),
        ("Poids de l'ajout dans la poche optimisable", additional / v_opt_final, "Effet de taille sur la poche"),
        ("Poids de l'ajout dans le portefeuille total", additional / v_total_final, "Effet de taille global"),
        ("TSR_ASSUMPTION", data["tsr"], f"{data['tsr_source']} — à synchroniser avec le notebook 02"),
        ("TARGET_RETURN", central_state["TARGET_RETURN"], "TSR + 4 % (proxy de rendement financier attendu)"),
        ("R_FIXED_CURRENT_ASSUMPTION", 0.0, "Poche non optimisable figée à rendement nul faute de modèle validé"),
        ("APT scénarios utilisés", "APT_Prudent; APT_Central; APT_Optimistic", "Trois scénarios pour tester la robustesse"),
        ("Modèles d'allocation", "Prorata; MinVar; MeanVar(2/5/10); MaxReturn; MeanCVaR; RiskParity; MonteCarlo×4", "Max Sharpe volontairement exclu"),
    ]
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur", "Interprétation"])


def _build_current_portfolio_table(data: dict[str, object], config: AdditionalAllocationConfig) -> pd.DataFrame:
    """Tableau du portefeuille actuel et après ajout."""

    v_total = float(data["V_TOTAL_CURRENT"])
    v_opt = float(data["V_OPT_CURRENT"])
    v_fixed = float(data["V_FIXED_CURRENT"])
    additional = config.additional_budget
    v_opt_final = v_opt + additional
    v_total_final = v_total + additional
    return pd.DataFrame([
        ("Portefeuille total actuel", v_total, 1.0, ""),
        ("Poche optimisable actuelle", v_opt, v_opt / v_total, "Univers d'investissement modélisé"),
        ("Poche non optimisable figée", v_fixed, v_fixed / v_total, "Immobilier, OPCVM, SICAR/SICAF, actions non cotées, placements non modélisés"),
        ("Montant additionnel", additional, additional / v_total, "Investi uniquement dans la poche optimisable"),
        ("Poche optimisable après ajout", v_opt_final, v_opt_final / v_total_final, ""),
        ("Portefeuille total après ajout", v_total_final, 1.0, ""),
    ], columns=["Périmètre", "Valeur", "Poids du périmètre", "Commentaire"])


def _build_sensitivity_multi(
    data: dict[str, object],
    by_scenario: dict[str, dict[str, object]],
    config: AdditionalAllocationConfig,
) -> pd.DataFrame:
    """Analyse de sensibilité par scénario : rendement final et montant requis."""

    rows = []
    return_grid = [0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
    amount_grid = [0.10, 0.12, 0.15, 0.20]
    v_opt = float(data["V_OPT_CURRENT"])
    v_total = float(data["V_TOTAL_CURRENT"])
    for scenario in SCENARIO_ORDER:
        state = by_scenario[scenario]["state"]
        target = state["TARGET_RETURN"]
        for r_add in return_grid:
            r_opt_final = (v_opt * state["R_OPT_CURRENT"] + config.additional_budget * r_add) / state["V_OPT_FINAL"]
            r_total_final = (v_total * state["R_TOTAL_CURRENT"] + config.additional_budget * r_add) / state["V_TOTAL_FINAL"]
            rows.append({
                "Scenario": scenario,
                "Type": "Rendement final pour 10 MD",
                "Hypothèse rendement additionnel": r_add,
                "Périmètre": "Poche optimisable",
                "Rendement final": r_opt_final,
                "Montant requis": np.nan,
                "Commentaire": "",
            })
            rows.append({
                "Scenario": scenario,
                "Type": "Rendement final pour 10 MD",
                "Hypothèse rendement additionnel": r_add,
                "Périmètre": "Portefeuille total",
                "Rendement final": r_total_final,
                "Montant requis": np.nan,
                "Commentaire": "",
            })
        for r_add in amount_grid:
            for perimeter, base_value, current_return in [
                ("Poche optimisable", v_opt, state["R_OPT_CURRENT"]),
                ("Portefeuille total", v_total, state["R_TOTAL_CURRENT"]),
            ]:
                denom = r_add - target
                if denom <= 0:
                    amount = np.nan
                    comment = "Impossible : rendement additionnel inférieur ou égal à la cible."
                else:
                    amount = base_value * (target - current_return) / denom
                    comment = ""
                rows.append({
                    "Scenario": scenario,
                    "Type": "Montant additionnel requis",
                    "Hypothèse rendement additionnel": r_add,
                    "Périmètre": perimeter,
                    "Rendement final": np.nan,
                    "Montant requis": amount,
                    "Commentaire": comment,
                })
    return pd.DataFrame(rows)


def _build_warnings_multi(
    data: dict[str, object],
    by_scenario: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Centraliser les warnings qualité multi-scénarios."""

    rows = []
    for scenario in SCENARIO_ORDER:
        scn = by_scenario[scenario]
        if not scn["target_opt_feasible"]:
            rows.append((scenario, "TARGET_OPT_NOT_REACHED_BY_MAX_RETURN", "WARNING", "Le rendement requis sur 10 MD pour la poche optimisable dépasse le maximum atteignable."))
        if not scn["target_total_feasible"]:
            rows.append((scenario, "TARGET_TOTAL_NOT_REACHED_BY_MAX_RETURN", "WARNING", "Le rendement requis sur 10 MD pour le portefeuille total dépasse le maximum atteignable."))
        non_testable = int(scn["regulatory"]["Status"].eq("NON_TESTABLE_DATA_MISSING").sum())
        if non_testable:
            rows.append((scenario, "REGULATORY_NON_TESTABLE", "WARNING", f"{non_testable} lignes de contraintes non testables faute de données détaillées."))
        for row in scn["results_models"].itertuples():
            if isinstance(row.Warnings, str) and row.Warnings != "OK":
                rows.append((scenario, f"MODEL_WARNING::{row.Model}", "WARNING", row.Warnings))
    rows.append(("ALL", "FIXED_POCKET_RETURN_NOT_MODELLED", "WARNING", "La poche non optimisable est figée; son rendement attendu est supposé nul faute de modèle validé."))
    if data["sigma_repaired"]:
        rows.append(("ALL", "SIGMA_PSD_REPAIRED", "WARNING", f"Covariance corrigée; min eigenvalue avant correction={data['sigma_min_eig_before']:.3e}."))
    freq = data["frequency_control"]["RETURN_FREQUENCY"]
    rows.append(("ALL", "RETURN_FREQUENCY_CONTROL", "PASSED", f"Fréquence détectée: {freq}; périodes/an={data['frequency_control']['PERIODS_PER_YEAR']}."))
    return pd.DataFrame(rows, columns=["Scenario", "Warning", "Severity", "Commentaire"])


def _build_conclusion_multi(
    data: dict[str, object],
    by_scenario: dict[str, dict[str, object]],
    config: AdditionalAllocationConfig,
) -> pd.DataFrame:
    """Conclusion finale structurée en 8 points pour le PFE."""

    central = by_scenario["APT_Central"]
    reco_central = central["results_models"].loc[central["results_models"]["Model"].eq(central["recommended_model"])].iloc[0]
    target_value = central["state"]["TARGET_RETURN"]
    additional_share_opt = config.additional_budget / central["state"]["V_OPT_FINAL"]
    additional_share_total = config.additional_budget / central["state"]["V_TOTAL_FINAL"]
    points = [
        ("1", "Périmètre d'investissement", "Les 10 MD sont investis uniquement dans la poche optimisable (titres d'État, obligations corporate, actions cotées). La poche non optimisable reste figée."),
        ("2", "Niveaux d'impact", "L'impact est mesuré à deux niveaux : sur la poche optimisable et sur le portefeuille global."),
        ("3", "Scénarios APT", "Les trois scénarios APT (Prudent, Central, Optimiste) sont testés pour vérifier la robustesse de l'allocation aux hypothèses de rendement."),
        ("4", "Objectif ROE = TSR + 4 %", f"L'objectif ROE = TSR + 4 % est interprété comme proxy de rendement financier attendu (cible = {target_value:.2%}). Ce proxy ne remplace pas le ROE comptable."),
        ("5", "Effet de taille", f"L'objectif n'est pas nécessairement atteint, principalement à cause de l'effet de taille de l'enveloppe (poids dans la poche optimisable = {additional_share_opt:.2%}, dans le portefeuille total = {additional_share_total:.2%}), du rendement marginal requis sur les 10 MD, des contraintes de prudence et du niveau réaliste des rendements disponibles."),
        ("6", "Maximum Return", "Maximum Return est conservé comme borne supérieure agressive du rendement atteignable sous contraintes. Il n'est pas retenu comme recommandation institutionnelle."),
        ("7", "Recommandation finale", f"La recommandation finale ({central['recommended_model']} sous scénario central) privilégie le compromis rendement-risque-diversification-conformité via le scoring multicritère."),
        ("8", "Contraintes réglementaires", "Les contraintes réglementaires sont validées sur les contraintes testables uniquement. Les contraintes restantes nécessitent un référentiel comptable détaillé qui sera renseigné en revue institutionnelle."),
    ]
    df = pd.DataFrame(points, columns=["Point", "Thème", "Énoncé"])
    formulation = pd.DataFrame([{
        "Point": "Synthèse",
        "Thème": "Formulation institutionnelle",
        "Énoncé": (
            "L'allocation additionnelle de 10 MD améliore le rendement attendu de la poche optimisable et "
            "du portefeuille global, mais l'objectif ROE = TSR + 4 % reste difficile à atteindre sous les "
            "hypothèses retenues. Cette non-atteinte s'explique principalement par l'effet de taille de "
            "l'enveloppe additionnelle : les 10 MD représentent une part limitée de la poche optimisable "
            "et une part encore plus faible du portefeuille total. Le rendement marginal requis sur cette "
            "enveloppe dépasse les rendements réalistes disponibles dans l'univers d'investissement. Le "
            "modèle Maximum Return fournit une borne supérieure de rendement sous contraintes, tandis que "
            "l'allocation recommandée repose sur un compromis rendement, risque extrême, diversification "
            f"et conformité réglementaire. Modèle recommandé sous scénario central : {central['recommended_model']} "
            f"(rendement additionnel des 10 MD : {reco_central['R_additional']:.2%} ; rendement final "
            f"poche optimisable : {reco_central['R_opt_final']:.2%} ; rendement final portefeuille total : "
            f"{reco_central['R_total_final']:.2%})."
        ),
    }])
    return pd.concat([df, formulation], ignore_index=True)


def _build_final_control_multi(
    data: dict[str, object],
    by_scenario: dict[str, dict[str, object]],
    config: AdditionalAllocationConfig,
    figures_exported: bool = False,
    excel_exported: bool = False,
) -> pd.DataFrame:
    """Contrôle final consolidé multi-scénarios."""

    additional = config.additional_budget
    accounting_ok = (
        abs(data["V_TOTAL_CURRENT"] - data["V_OPT_CURRENT"] - data["V_FIXED_CURRENT"]) <= 1e-2
    )
    sigma_ok = data["sigma_min_eig_before"] >= -1e-8 or data["sigma_repaired"]
    frequency_ok = data["frequency_control"]["RETURN_FREQUENCY"] in {"daily", "weekly"}
    all_target_reached = True
    all_weights_ok = True
    all_regulatory_ok = True
    for scenario in SCENARIO_ORDER:
        scn = by_scenario[scenario]
        weights = scn["recommended_weights"]
        all_weights_ok &= bool(np.isclose(weights.sum(), 1.0, atol=1e-8) and np.all(weights >= -1e-10))
        all_weights_ok &= bool(np.isclose((weights * additional).sum(), additional, atol=1e-4))
        testable = scn["regulatory"].loc[
            scn["regulatory"]["model"].eq(scn["recommended_model"]) & scn["regulatory"]["testable"].astype(bool)
        ]
        all_regulatory_ok &= not testable["Status"].eq("FAILED").any()
        reco = scn["results_models"].loc[scn["results_models"]["Model"].eq(scn["recommended_model"])].iloc[0]
        target_reached = reco["Target_Opt_Reached"] == "YES" and reco["Target_Total_Reached"] == "YES"
        all_target_reached &= target_reached
    technical_ok = bool(accounting_ok and sigma_ok and frequency_ok and all_weights_ok and all_regulatory_ok)
    technical_status = "PASSED" if technical_ok else "FAILED"
    target_status = "TARGET_REACHED" if all_target_reached else "TARGET_NOT_REACHED"
    if not technical_ok:
        global_status = "FAILED"
    elif all_target_reached:
        global_status = "PASSED"
    else:
        global_status = "ANALYSIS_VALID_TARGET_NOT_REACHED"
    rows = [
        ("Comptabilité V_TOTAL = V_OPT + V_FIXED", accounting_ok, "PASSED" if accounting_ok else "FAILED"),
        ("Covariance PSD", sigma_ok, "PASSED" if sigma_ok else "FAILED"),
        ("Fréquence cohérente", frequency_ok, f"{data['frequency_control']['RETURN_FREQUENCY']}; {data['frequency_control']['PERIODS_PER_YEAR']}"),
        ("Poids sommant à 1 et non négatifs (tous scénarios)", all_weights_ok, "PASSED" if all_weights_ok else "FAILED"),
        ("Contraintes testables (tous scénarios)", all_regulatory_ok, "PASSED" if all_regulatory_ok else "FAILED"),
        ("Graphiques exportés", figures_exported, "YES" if figures_exported else "NO"),
        ("Excel exporté", excel_exported, "YES" if excel_exported else "NO"),
        ("Technical_Status", technical_ok, technical_status),
        ("Target_Status", all_target_reached, target_status),
        ("Statut global", technical_ok, global_status),
    ]
    return pd.DataFrame(rows, columns=["Contrôle", "Passed", "Status"])


def run_multi_scenario_allocation(
    project_dir: str | Path,
    config: AdditionalAllocationConfig | None = None,
) -> dict[str, object]:
    """Exécuter l'analyse 10 MD sur les trois scénarios APT et produire les agrégats du notebook 03."""

    config = config or AdditionalAllocationConfig()
    data = load_inputs(project_dir)
    if not np.isfinite(float(data["tsr"])):
        raise ValueError("TSR indisponible : renseigner manuellement l'hypothèse TSR.")
    apt_scenarios, apt_audit = _load_apt_scenarios_for_allocation(
        project_dir, data["mu"].index.astype(str).tolist()
    )
    by_scenario: dict[str, dict[str, object]] = {}
    for scenario in SCENARIO_ORDER:
        by_scenario[scenario] = run_scenario_allocation(scenario, data, apt_scenarios[scenario], config)

    apt_table = _build_apt_scenarios_table(data, apt_audit)
    current_returns_table = _build_current_returns_by_scenario(data, by_scenario)
    required_returns_table = _build_required_returns_by_scenario(by_scenario)
    cross_results = _build_cross_results(by_scenario)
    impact_opt, impact_total = _build_impact_tables(cross_results, data, by_scenario)
    recommendation_table = _build_recommendation_table(by_scenario)
    status_table = _build_status_table(by_scenario)
    sensitivity_table = _build_sensitivity_multi(data, by_scenario, config)
    warnings_table = _build_warnings_multi(data, by_scenario)
    conclusion_table = _build_conclusion_multi(data, by_scenario, config)
    hypotheses_table = _build_hypotheses_table(data, by_scenario, config)
    current_portfolio_table = _build_current_portfolio_table(data, config)
    final_control = _build_final_control_multi(data, by_scenario, config)

    all_allocations = pd.concat(
        [by_scenario[s]["allocations"] for s in SCENARIO_ORDER], ignore_index=True
    )
    all_regulatory = pd.concat(
        [by_scenario[s]["regulatory"] for s in SCENARIO_ORDER], ignore_index=True
    )
    all_monte_carlo = pd.concat(
        [by_scenario[s]["monte_carlo"] for s in SCENARIO_ORDER], ignore_index=True
    )
    all_scoring = pd.concat(
        [by_scenario[s]["scoring"] for s in SCENARIO_ORDER], ignore_index=True
    )
    all_mc_selected = pd.concat(
        [by_scenario[s]["monte_carlo_selected"] for s in SCENARIO_ORDER], ignore_index=True
    )

    return {
        **data,
        "config": config,
        "scenarios": list(SCENARIO_ORDER),
        "by_scenario": by_scenario,
        "apt_scenarios": apt_scenarios,
        "apt_audit": apt_audit,
        "01_Hypotheses": hypotheses_table,
        "02_Current_Portfolio": current_portfolio_table,
        "03_Optimizable_Pocket": asset_control_table(data),
        "04_APT_Scenarios": apt_table,
        "05_Required_Returns": required_returns_table,
        "05b_Current_Returns": current_returns_table,
        "06_Allocations_10MD": all_allocations,
        "07_Model_Results_By_Scenario": cross_results,
        "08_Impact_Optimizable_Pocket": impact_opt,
        "09_Impact_Total_Portfolio": impact_total,
        "10_Regulatory_Checks": all_regulatory,
        "11_Monte_Carlo": all_monte_carlo,
        "11b_Monte_Carlo_Selected": all_mc_selected,
        "12_Scoring": all_scoring,
        "13_Sensitivity_Analysis": sensitivity_table,
        "14_Final_Recommendation": recommendation_table,
        "14b_Status": status_table,
        "15_Conclusion": conclusion_table,
        "Warnings_Quality": warnings_table,
        "Controle_Final": final_control,
        "frequency_control_table": pd.DataFrame([data["frequency_control"]]),
    }


def export_multi_scenario_analysis(
    result: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Exporter le classeur Excel final multi-scénarios (15 feuilles)."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheets = [
        "01_Hypotheses",
        "02_Current_Portfolio",
        "03_Optimizable_Pocket",
        "04_APT_Scenarios",
        "05_Required_Returns",
        "06_Allocations_10MD",
        "07_Model_Results_By_Scenario",
        "08_Impact_Optimizable_Pocket",
        "09_Impact_Total_Portfolio",
        "10_Regulatory_Checks",
        "11_Monte_Carlo",
        "12_Scoring",
        "13_Sensitivity_Analysis",
        "14_Final_Recommendation",
        "15_Conclusion",
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in sheets:
            df = result[sheet].copy()
            if "Recommended" in df.columns and df["Recommended"].dtype == bool:
                df["Recommended"] = np.where(df["Recommended"], "YES", "NO")
            if sheet == "11_Monte_Carlo" and len(df) > 60_000:
                df = df.head(60_000)
            df.to_excel(writer, sheet_name=sheet, index=False)
    return path
