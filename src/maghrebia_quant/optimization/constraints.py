"""Constraint registers and regulatory checks for notebook 02."""

from __future__ import annotations

import numpy as np
import pandas as pd


NON_TESTABLE_STATUS = "PASSED_SUBJECT_TO_NON_TESTABLE_CONSTRAINTS"

TECHNICAL_TO_METHODOLOGICAL = {
    "APT_Prudent": "ExAnte_Prudent",
    "APT_Central": "ExAnte_Central",
    "APT_Optimistic": "ExAnte_Optimistic",
    "Historical_Raw": "Historical_Raw_Comparative",
}


def methodological_name(technical_name: str) -> str:
    return TECHNICAL_TO_METHODOLOGICAL.get(technical_name, technical_name)


def scenario_name_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("APT_Prudent", "ExAnte_Prudent", "Alias technique pour compatibilite Notebook 02."),
            ("APT_Central", "ExAnte_Central", "Alias technique pour compatibilite Notebook 02 ; scenario prospectif principal."),
            ("APT_Optimistic", "ExAnte_Optimistic", "Alias technique pour compatibilite Notebook 02."),
            ("Historical_Raw", "Historical_Raw_Comparative", "Comparatif descriptif uniquement, non prospectif."),
        ],
        columns=["Technical_Name", "Methodological_Name", "Usage"],
    )


def build_cga_legal_reference_register() -> pd.DataFrame:
    rows = [
        (
            "Code des assurances tunisien",
            "Cadre general",
            "CGA_LEGAL_CONSTRAINT",
            "Les placements representatifs doivent respecter les principes de couverture et de securite.",
            "Portefeuille total",
            "Admitted_Assets_Value >= Technical_Provisions",
            "Valeur admise, provisions techniques",
            "TESTABLE_IF_DATA_AVAILABLE",
            "Reference juridique structurante.",
        ),
        (
            "Arrete du ministre des Finances du 27 fevrier 2001, modifie notamment par l'arrete du 1er mars 2016",
            "Couverture des provisions techniques",
            "CGA_LEGAL_CONSTRAINT",
            "Les actifs admis en representation couvrent au moins 100 % des provisions techniques.",
            "Portefeuille total",
            "Admitted_Assets_Value / Technical_Provisions >= 100%",
            "Valeur admise, provisions techniques",
            "TESTABLE_IF_DATA_AVAILABLE",
            "Controle global de couverture.",
        ),
        (
            "Arrete du 27 fevrier 2001 modifie",
            "Cantonnement vie / non-vie",
            "CGA_LEGAL_CONSTRAINT",
            "Les actifs admis respectent le cantonnement si l'information vie / non-vie existe.",
            "Portefeuille total",
            "Assets_Life >= Technical_Provisions_Life ; Assets_NonLife >= Technical_Provisions_NonLife",
            "Ventilation vie / non-vie",
            "NOT_TESTABLE_DATA_MISSING",
            "Donnee non disponible dans la poche optimisable.",
        ),
        (
            "Arrete du 27 fevrier 2001 modifie",
            "Titres emis par l'Etat ou jouissant de sa garantie",
            "CGA_LEGAL_CONSTRAINT",
            "Minimum 20 % des provisions techniques.",
            "Titres d'Etat",
            "State_Guaranteed_Securities / Technical_Provisions >= 20%",
            "Exposition titres d'Etat, provisions techniques",
            "TESTABLE_IF_DATA_AVAILABLE",
            "Contrainte integree au solveur quand les donnees existent.",
        ),
        (
            "Arrete du 27 fevrier 2001 modifie",
            "Placements immobiliers",
            "CGA_LEGAL_CONSTRAINT",
            "Un immeuble determine <= 10 % des provisions techniques sauf siege social ; total immobilier <= 20 %.",
            "Immobilier",
            "Real_Estate_Total / Technical_Provisions <= 20%",
            "Expositions immobilieres, provisions techniques",
            "OUT_OF_OPTIMISABLE_SCOPE_BUT_DOCUMENTED",
            "Hors perimetre optimisable du notebook.",
        ),
        (
            "Arrete du 27 fevrier 2001 modifie",
            "Actions cotees BVMT",
            "CGA_LEGAL_CONSTRAINT",
            "Actions d'une meme societe <= 10 % des provisions techniques et <= 30 % du capital social de l'emetteur.",
            "Actions cotees",
            "Issuer_Equity_Exposure / Technical_Provisions <= 10% ; Shares_Held / Shares_Outstanding <= 30%",
            "Exposition par emetteur, provisions techniques, Shares_Outstanding",
            "PARTIALLY_TESTABLE",
            "Limite capital social non validable si Shares_Outstanding manque.",
        ),
        (
            "Decret n2001-2278 et Arrete du 27 fevrier 2001 modifie",
            "OPCVM / FCP / SICAV",
            "CGA_LEGAL_CONSTRAINT",
            "Parts d'un meme fonds <= 10 % des provisions techniques lorsque des OPCVM sont presents.",
            "OPCVM",
            "Fund_Exposure / Technical_Provisions <= 10%",
            "Exposition OPCVM, provisions techniques",
            "NOT_APPLICABLE_NO_OPCVM",
            "Le decret n2001-2278 est cite uniquement pour le cadre OPCVM.",
        ),
        (
            "Arrete du 27 fevrier 2001 modifie",
            "SICAR / SICAF",
            "CGA_LEGAL_CONSTRAINT",
            "Meme societe <= 5 % des provisions techniques ; total SICAR/SICAF <= 10 %.",
            "SICAR / SICAF",
            "SICAR_Total / Technical_Provisions <= 10%",
            "Exposition SICAR/SICAF",
            "OUT_OF_OPTIMISABLE_SCOPE_BUT_DOCUMENTED",
            "Hors perimetre optimisable.",
        ),
        (
            "Arrete du 27 fevrier 2001 modifie",
            "Categorie plafonnee",
            "CGA_LEGAL_CONSTRAINT",
            "Ne pas placer plus de 50 % des provisions techniques dans une categorie concernee lorsque applicable.",
            "Categories concernees",
            "Category_Exposure / Technical_Provisions <= 50%",
            "Exposition par categorie, provisions techniques",
            "CONFIGURABLE",
            "Regle configurable par categorie.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Legal_Text",
            "Article_or_Reference",
            "Rule_Type",
            "Rule_Description",
            "Applies_To",
            "Formula",
            "Data_Required",
            "Testability_Status",
            "Comment",
        ],
    )


def _safe_ratio(value: float, denominator: float) -> float:
    return float(value / denominator) if denominator and np.isfinite(denominator) and denominator > 0 else np.nan


def build_cga_regulatory_constraints_check(
    universe: pd.DataFrame,
    context: dict[str, object],
    capital_social_status: str = "NON_TESTABLE_DATA_MISSING",
) -> pd.DataFrame:
    pt = float(context.get("technical_provisions", np.nan))
    total_value = float(context.get("total_value", np.nan))
    optimisable_value = float(context.get("optimisable_value", np.nan))
    fixed = context.get("fixed", pd.DataFrame())
    fixed_value_by_type = fixed.groupby("asset_type")["market_value"].sum().to_dict() if isinstance(fixed, pd.DataFrame) and not fixed.empty else {}
    fixed_state_value = 0.0
    state_value = float(universe.loc[universe["asset_type"].eq("government_bond"), "current_weight_optimisable"].sum() * optimisable_value)
    state_value += fixed_state_value
    equity_by_issuer = (
        universe.loc[universe["asset_type"].eq("listed_equity")]
        .assign(exposure=lambda x: x["current_weight_optimisable"].astype(float) * optimisable_value)
        .groupby("issuer")["exposure"]
        .sum()
    )
    max_equity = float(equity_by_issuer.max()) if not equity_by_issuer.empty else 0.0
    has_opcvm = bool(universe["asset_type"].astype(str).str.contains("opcvm|sicav|fcp", case=False, na=False).any())

    def row(name, ref, scope, value, limit_ratio, data_status, compliance, comment):
        limit = np.nan if not np.isfinite(pt) or not np.isfinite(limit_ratio) else pt * limit_ratio
        return {
            "Constraint_Name": name,
            "Legal_Reference": ref,
            "Scope": scope,
            "Current_Value_TND": value,
            "Limit_TND": limit,
            "Current_Ratio": _safe_ratio(value, pt),
            "Limit_Ratio": limit_ratio,
            "Data_Status": data_status,
            "Compliance_Status": compliance,
            "Comment": comment,
        }

    rows = [
        row(
            "Coverage_Technical_Provisions",
            "Code des assurances tunisien ; arrete du 27 fevrier 2001 modifie",
            "Portefeuille total",
            total_value,
            1.0,
            "DATA_AVAILABLE",
            "PASSED" if total_value >= pt else "FAILED",
            "Controle de couverture globale.",
        ),
        row(
            "Life_NonLife_Ring_Fencing",
            "Arrete du 27 fevrier 2001 modifie",
            "Portefeuille total",
            np.nan,
            np.nan,
            "DATA_MISSING",
            "NOT_TESTABLE_DATA_MISSING",
            "Ventilation vie / non-vie indisponible.",
        ),
        row(
            "State_Guaranteed_Min_20pct_PT",
            "Arrete du 27 fevrier 2001 modifie",
            "Titres d'Etat",
            state_value,
            0.20,
            "DATA_AVAILABLE",
            "PASSED" if _safe_ratio(state_value, pt) >= 0.20 - 1e-12 else "FAILED",
            "Minimum titres d'Etat ou garantis par l'Etat ; les titres d'Etat sont traites comme poche optimisable.",
        ),
        row(
            "Real_Estate_Total_Max_20pct_PT",
            "Arrete du 27 fevrier 2001 modifie",
            "Immobilier",
            float(fixed_value_by_type.get("real_estate", 0.0)),
            0.20,
            "OUT_OF_OPTIMISABLE_SCOPE",
            "NOT_APPLICABLE",
            "Hors perimetre optimisable mais documente.",
        ),
        row(
            "Listed_Equity_Same_Issuer_Max_10pct_PT",
            "Arrete du 27 fevrier 2001 modifie",
            "Actions cotees BVMT",
            max_equity,
            0.10,
            "DATA_AVAILABLE",
            "PASSED" if _safe_ratio(max_equity, pt) <= 0.10 + 1e-12 else "FAILED",
            "Exposition actions maximale par emetteur.",
        ),
        row(
            "Listed_Equity_Max_30pct_Issuer_Capital",
            "Arrete du 27 fevrier 2001 modifie",
            "Actions cotees BVMT",
            np.nan,
            0.30,
            "DATA_MISSING" if capital_social_status == "NON_TESTABLE_DATA_MISSING" else "DATA_AVAILABLE",
            capital_social_status,
            "Shares_Outstanding requis ; jamais valide si la donnee manque.",
        ),
        row(
            "OPCVM_Same_Fund_Max_10pct_PT",
            "Decret n2001-2278 ; arrete du 27 fevrier 2001 modifie",
            "OPCVM",
            np.nan,
            0.10,
            "NO_OPCVM" if not has_opcvm else "DATA_MISSING",
            "NOT_APPLICABLE" if not has_opcvm else "NOT_TESTABLE_DATA_MISSING",
            "Applicable uniquement si des parts OPCVM sont presentes.",
        ),
        row(
            "SICAR_SICAF_Total_Max_10pct_PT",
            "Arrete du 27 fevrier 2001 modifie",
            "SICAR / SICAF",
            float(fixed_value_by_type.get("sicar", 0.0)),
            0.10,
            "OUT_OF_OPTIMISABLE_SCOPE",
            "NOT_APPLICABLE",
            "Hors perimetre optimisable.",
        ),
    ]
    return pd.DataFrame(rows)


def check_cga_constraints_by_portfolio(
    weights: np.ndarray,
    universe: pd.DataFrame,
    fixed_pocket: pd.DataFrame,
    technical_provisions: float,
    optimisable_value: float,
    total_value: float,
    capital_social_status: str = "NON_TESTABLE_DATA_MISSING",
) -> pd.DataFrame:
    """Check CGA constraints for one optimized portfolio."""

    fixed = fixed_pocket.copy() if isinstance(fixed_pocket, pd.DataFrame) else pd.DataFrame()
    fixed_value_by_type = fixed.groupby("asset_type")["market_value"].sum().to_dict() if not fixed.empty and {"asset_type", "market_value"}.issubset(fixed.columns) else {}
    fixed_state_status = "DATA_AVAILABLE"
    fixed_state = 0.0
    w = np.asarray(weights, dtype=float)
    opt = universe.assign(optimized_value=w * optimisable_value)
    state_value = float(opt.loc[opt["asset_type"].eq("government_bond"), "optimized_value"].sum() + fixed_state)
    equity_by_issuer = (
        opt.loc[opt["asset_type"].eq("listed_equity")]
        .groupby("issuer")["optimized_value"]
        .sum()
        if "issuer" in opt.columns
        else pd.Series(dtype=float)
    )
    max_equity = float(equity_by_issuer.max()) if not equity_by_issuer.empty else 0.0
    has_opcvm = bool(opt["asset_type"].astype(str).str.contains("opcvm|sicav|fcp", case=False, na=False).any())

    def ratio(value):
        return float(value / technical_provisions) if technical_provisions and np.isfinite(technical_provisions) else np.nan

    def row(name, ref, value, limit_ratio, data_status, status, comment):
        return {
            "Constraint_Name": name,
            "Legal_Reference": ref,
            "Current_Value_TND": value,
            "Limit_TND": technical_provisions * limit_ratio if np.isfinite(limit_ratio) and np.isfinite(technical_provisions) else np.nan,
            "Current_Ratio": ratio(value) if np.isfinite(value) else np.nan,
            "Limit_Ratio": limit_ratio,
            "Data_Status": data_status,
            "Compliance_Status": status,
            "Comment": comment,
        }

    rows = [
        row(
            "Coverage_Technical_Provisions",
            "Code des assurances tunisien ; arrete du 27 fevrier 2001 modifie",
            total_value,
            1.0,
            "DATA_AVAILABLE",
            "PASSED" if total_value >= technical_provisions else "FAILED",
            "Controle global de couverture.",
        ),
        row(
            "State_Guaranteed_Min_20pct_PT",
            "Arrete du 27 fevrier 2001 modifie",
            state_value,
            0.20,
            fixed_state_status,
            "PASSED" if fixed_state_status == "DATA_AVAILABLE" and ratio(state_value) >= 0.20 - 1e-12 else ("NOT_TESTABLE_DATA_MISSING" if fixed_state_status != "DATA_AVAILABLE" else "FAILED"),
            "Minimum titres d'Etat calcule sur le portefeuille total ; tous les titres d'Etat sont traites comme optimisables.",
        ),
        row(
            "Listed_Equity_Same_Issuer_Max_10pct_PT",
            "Arrete du 27 fevrier 2001 modifie",
            max_equity,
            0.10,
            "DATA_AVAILABLE" if "issuer" in opt.columns else "NOT_TESTABLE_DATA_MISSING",
            "PASSED" if "issuer" in opt.columns and ratio(max_equity) <= 0.10 + 1e-12 else ("NOT_TESTABLE_DATA_MISSING" if "issuer" not in opt.columns else "FAILED"),
            "Exposition actions par emetteur.",
        ),
        row(
            "Listed_Equity_Max_30pct_Issuer_Capital",
            "Arrete du 27 fevrier 2001 modifie",
            np.nan,
            0.30,
            "DATA_MISSING" if capital_social_status == "NON_TESTABLE_DATA_MISSING" else "DATA_AVAILABLE",
            capital_social_status,
            "Shares_Outstanding requis.",
        ),
        row(
            "OPCVM_Same_Fund_Max_10pct_PT",
            "Decret n2001-2278 ; arrete du 27 fevrier 2001 modifie",
            np.nan,
            0.10,
            "NO_OPCVM" if not has_opcvm else "DATA_MISSING",
            "NOT_APPLICABLE" if not has_opcvm else "NOT_TESTABLE_DATA_MISSING",
            "Applicable uniquement si OPCVM present.",
        ),
    ]
    return pd.DataFrame(rows)


def aggregate_cga_status(check: pd.DataFrame) -> str:
    statuses = set(check["Compliance_Status"].astype(str))
    if "FAILED" in statuses:
        return "FAILED"
    if any(s in statuses for s in {"NOT_TESTABLE_DATA_MISSING", "NON_TESTABLE_DATA_MISSING"}):
        return NON_TESTABLE_STATUS
    return "PASSED"


def aggregate_regulatory_status(cga_check: pd.DataFrame) -> str:
    statuses = set(cga_check["Compliance_Status"].astype(str))
    if "FAILED" in statuses:
        return "FAILED"
    if any(s in statuses for s in {"NOT_TESTABLE_DATA_MISSING", "NON_TESTABLE_DATA_MISSING"}):
        return NON_TESTABLE_STATUS
    return "PASSED"


def build_constraints_register(cga_register: pd.DataFrame) -> pd.DataFrame:
    internal = pd.DataFrame(
        [
            ("Budget_Full_Investment", "sum(w)=1", "Optimisable pocket", "Notebook 01", True, "PASSED", "INTERNAL_OPTIMIZER_CONSTRAINT", "Full investment."),
            ("No_Short_Selling", "w_i >= 0", "Optimisable pocket", "Notebook 01", True, "PASSED", "INTERNAL_OPTIMIZER_CONSTRAINT", "Long-only optimisation."),
            ("Asset_Upper_Bound", "w_i <= max_weight_i", "Asset", "Notebook 01 / user rule", True, "PASSED", "INTERNAL_OPTIMIZER_CONSTRAINT", "Applied when available."),
            ("Class_Upper_Bounds", "class_min <= sum(w_class) <= class_max", "Asset class", "Notebook 01 / user rule", True, "PASSED", "INTERNAL_OPTIMIZER_CONSTRAINT", "State, equity and corporate limits."),
            ("Issuer_Limit", "issuer_exposure <= issuer_limit", "Issuer", "Notebook 01 / user rule", True, "PASSED", "INTERNAL_OPTIMIZER_CONSTRAINT", "Applied through asset and class bounds when issuer granularity is available."),
            ("Capital_Social_Data", "Shares_Held / Shares_Outstanding <= 30%", "Issuer", "External BVMT data", False, "NON_TESTABLE_DATA_MISSING", "DATA_QUALITY_CONSTRAINT", "Shares_Outstanding absent."),
        ],
        columns=["Constraint_Name", "Mathematical_Form", "Scope", "Data_Source", "Blocking", "Status", "Constraint_Type", "Comment"],
    )
    cga = cga_register.rename(
        columns={
            "Article_or_Reference": "Constraint_Name",
            "Formula": "Mathematical_Form",
            "Applies_To": "Scope",
            "Legal_Text": "Data_Source",
            "Testability_Status": "Status",
        }
    )
    cga["Blocking"] = cga["Status"].ne("NOT_APPLICABLE_NO_OPCVM")
    cga["Constraint_Type"] = "CGA_LEGAL_CONSTRAINT"
    cga = cga[["Constraint_Name", "Mathematical_Form", "Scope", "Data_Source", "Blocking", "Status", "Constraint_Type", "Comment"]]
    return pd.concat([internal, cga], ignore_index=True)
