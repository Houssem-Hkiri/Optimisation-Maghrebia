"""Lecture du portefeuille et identification de la poche optimisable."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import coerce_numeric, normalize_text, slugify, standardize_columns

logger = logging.getLogger(__name__)

EXPECTED_TOTAL_PORTFOLIO_VALUE = 509_000_000.0
RECONCILIATION_TOLERANCE = 0.01


def clean_asset_name(value: object) -> str:
    """Nettoie le libellé portefeuille."""

    return "" if pd.isna(value) else str(value).strip().lstrip("-* ").strip()


def infer_corporate_sector(asset_name: str) -> str:
    """Infère un secteur corporate simple depuis le libellé exact disponible."""

    norm = normalize_text(asset_name)
    if any(token in norm for token in ["ATL", "HL", "BH LEASING", "LEASING"]):
        return "LEASING"
    if any(token in norm for token in ["BNA", "AMEN", "AB SUB", "BANQUE", "BANK"]):
        return "BANKING"
    if "ADVANS" in norm or "MICRO" in norm:
        return "MICROFINANCE"
    return "OTHER"


def infer_bond_type(asset_name: str) -> str:
    """Infère le type de dette corporate."""

    norm = normalize_text(asset_name)
    return "SUBORDINATED" if "SUB" in norm else "ORDINARY"


def _infer_asset_type(name: str, parent_class: str) -> str:
    norm = normalize_text(name)
    parent = normalize_text(parent_class)
    if "BTA" in norm or "EMPRUNT NATIONAL" in norm:
        return "government_bond"
    if "E.O" in norm or norm.startswith("EO ") or "SUB" in norm or "OBLIGATAIRE" in parent:
        return "corporate_bond"
    if "ACTIONS COTEES" in parent or str(name).strip().startswith("*"):
        return "listed_equity"
    if "IMMOBILIER" in parent or "IMMOBILIER" in norm:
        return "real_estate"
    if "OPCVM" in parent or "OPCVM" in norm:
        return "fund"
    if "SICAR" in parent or "SICAR" in norm:
        return "sicar"
    if "NON COTES" in parent or "NON COTES" in norm:
        return "private_equity"
    if "DEPOT" in parent or "DEPOT" in norm:
        return "deposit"
    return "other"


def _standardize_asset_class(name: str, parent_class: str, asset_type: str) -> str:
    norm = normalize_text(name)
    parent = normalize_text(parent_class)
    if asset_type == "government_bond":
        return "Titres de l'État"
    if asset_type == "corporate_bond":
        return "Obligations corporate"
    if asset_type == "listed_equity":
        return "Actions cotées"
    if asset_type == "real_estate":
        return "Immobilier"
    if asset_type == "fund":
        return "OPCVM"
    if asset_type == "sicar":
        return "SICAR"
    if asset_type == "private_equity":
        return "Actions non cotées"
    if asset_type == "deposit":
        return "Dépôts"
    if "PLACEMENT MONETAIRE" in norm or "PLACEMENT MONETAIRE" in parent:
        return "Placements monétaires"
    if "FRAIS D ACQUISITION" in norm:
        return "Frais d'acquisition reportés"
    if "QUITTANCES" in norm:
        return "Quittances non encaissées"
    if "LETTRE DE GARANTIE" in norm:
        return "Lettres de garantie"
    if "CREANCES" in norm:
        return "Créances de réassurance"
    return "Autres actifs non couverts"


def _is_total(name: str) -> bool:
    norm = normalize_text(name)
    return "TOTAUX" in norm or "PROVISIONS TECHNIQUES" in norm


def _read_details(path: Path) -> pd.DataFrame:
    try:
        df = standardize_columns(pd.read_excel(path, sheet_name="Porefeuille des placements MAG"))
    except ValueError:
        return pd.DataFrame()
    rename = {
        "designation_des_actifs": "asset_name",
        "valeur_d_usage": "usage_value",
        "taux": "coupon_rate",
        "date_de_jouissance": "issue_date",
        "date_d_echeance": "maturity_date",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df})
    if "asset_name" not in df:
        return pd.DataFrame()
    df["asset_name_clean"] = df["asset_name"].map(clean_asset_name)
    for col in ["coupon_rate", "usage_value"]:
        if col in df:
            df[col] = coerce_numeric(df[col])
    if "coupon_rate" in df:
        df.loc[df["coupon_rate"].abs() > 1, "coupon_rate"] /= 100.0
    for col in ["issue_date", "maturity_date"]:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    keep = ["asset_name_clean", "usage_value", "coupon_rate", "issue_date", "maturity_date"]
    return df[[c for c in keep if c in df]].drop_duplicates("asset_name_clean")


def _extract_schedule_meta(path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    except ValueError:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    current: str | None = None
    for _, row in raw.iterrows():
        first = row.iloc[0]
        second = normalize_text(row.iloc[1]) if len(row) > 1 else ""
        if pd.notna(first) and "CAPITAL" in second:
            current = clean_asset_name(first)
            continue
        if current is None:
            continue
        date = pd.to_datetime(first, errors="coerce", dayfirst=True)
        if pd.isna(date):
            continue
        records.append(
            {
                "schedule_name": current,
                "schedule_key": slugify(current),
                "cashflow_date": date,
                "capital_begin": coerce_numeric(pd.Series([row.iloc[1]])).iloc[0],
                "interest": coerce_numeric(pd.Series([row.iloc[2]])).iloc[0],
                "principal": coerce_numeric(pd.Series([row.iloc[3]])).iloc[0],
            }
        )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.groupby(["schedule_name", "schedule_key"], as_index=False).agg(
        maturity_date_schedule=("cashflow_date", "max"),
        nominal_proxy=("capital_begin", "max"),
        coupon_amount_proxy=("interest", "max"),
    )


def _canonical_key(value: str) -> str:
    parts = slugify(value.replace("E.O", "EO")).split("_")
    return "_".join(str(int(part)) if part.isdigit() else part for part in parts)


def _match_schedule_key(asset_name: str, schedule_keys: pd.Series) -> str | None:
    asset = _canonical_key(asset_name)
    if len(asset) < 5:
        return None
    for key in schedule_keys.dropna().astype(str):
        key_norm = _canonical_key(key)
        if key_norm in asset or asset in key_norm:
            return key
    compact_asset = asset.replace("EMPRUNT_NATIONAL_", "").replace("EO_", "")
    for key in schedule_keys.dropna().astype(str):
        compact_key = _canonical_key(key).replace("EMPRUNT_NATIONAL_", "").replace("EO_", "")
        if compact_key and (compact_key in compact_asset or compact_asset in compact_key):
            return key
    return None


def load_portfolio(path: Path) -> pd.DataFrame:
    """Lit la feuille portefeuille sans supprimer les classes non optimisables."""

    if not path.exists():
        raise FileNotFoundError(path)
    return standardize_columns(pd.read_excel(path, sheet_name="Principal"))


def prepare_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les colonnes et convertit les montants."""

    out = df.copy()
    out = out.rename(
        columns={
            "designation_des_actifs": "asset_name",
            "cout_d_entree_au_bilan": "book_value",
            "valeur_au_31_12_2025": "market_value",
            "quantite_au_31_12_2025": "quantity",
            "poids": "weight",
            "optimisable": "optimisable_source",
            "isin": "isin",
        }
    )
    if "asset_name" not in out:
        raise ValueError("Colonne de désignation absente du portefeuille.")
    if "market_value" not in out:
        candidates = ["valeur_portefeuille", "valeur_actuelle", "valeur_d_usage", "book_value"]
        found = next((col for col in candidates if col in out), None)
        if found is None:
            raise ValueError("Aucune colonne de valeur portefeuille exploitable.")
        out = out.rename(columns={found: "market_value"})

    out = out.dropna(how="all").copy()
    out["asset_name_raw"] = out["asset_name"]
    out["asset_name"] = out["asset_name"].map(clean_asset_name)
    for col in ["book_value", "market_value", "quantity", "weight", "optimisable_source"]:
        out[col] = coerce_numeric(out[col]) if col in out else np.nan
    out["isin"] = out.get("isin", np.nan).replace("-", np.nan)
    return out


def classify_portfolio_assets(df: pd.DataFrame) -> pd.DataFrame:
    """Classe les lignes valorisées sans double compter les en-têtes."""

    out = df.copy()
    out["is_total_row"] = out["asset_name"].map(_is_total)
    explicit_child = out["asset_name_raw"].astype(str).str.strip().str.startswith(("-", "*"))
    implicit_child = out["weight"].isna() & out["optimisable_source"].isna() & ~out["is_total_row"]
    out["is_child_line"] = explicit_child | implicit_child

    parent = ""
    parents: list[str] = []
    parent_keys: list[str] = []
    for _, row in out.iterrows():
        is_child = bool(row["is_child_line"])
        if not is_child and not bool(row["is_total_row"]):
            parent = row["asset_name"]
        parents.append(parent)
        parent_keys.append(slugify(parent))
    out["asset_class"] = parents
    out["asset_class_original"] = out["asset_class"]
    out["_parent_key"] = parent_keys

    child_counts = out.loc[out["is_child_line"]].groupby("_parent_key").size()
    out["has_child_lines"] = out["_parent_key"].map(child_counts).fillna(0).astype(int).gt(0) & ~out["is_child_line"]
    out["is_position"] = (~out["is_total_row"]) & (out["is_child_line"] | ~out["has_child_lines"])
    out["asset_type"] = [_infer_asset_type(n, c) for n, c in zip(out["asset_name"], out["asset_class"])]
    out["asset_class_standardized"] = [
        _standardize_asset_class(n, c, t) for n, c, t in zip(out["asset_name"], out["asset_class"], out["asset_type"])
    ]
    out["asset_id"] = np.where(out["isin"].notna(), out["isin"].astype(str), out["asset_name"].map(slugify))
    out["is_optimisable"] = out["is_position"] & out["asset_type"].isin(["listed_equity", "government_bond", "corporate_bond"])

    total_value = out.loc[out["is_position"], "market_value"].sum()
    missing_weight = out["is_position"] & out["weight"].isna() & out["market_value"].notna()
    out.loc[missing_weight, "weight"] = out.loc[missing_weight, "market_value"] / total_value
    out["portfolio_weight"] = np.where(out["is_position"], out["market_value"] / total_value, np.nan)
    return out


def load_and_prepare_portfolio(path: Path) -> pd.DataFrame:
    """Lit le portefeuille Maghrebia et ajoute les champs de diagnostic."""

    df = classify_portfolio_assets(prepare_portfolio(load_portfolio(path)))

    details = _read_details(path)
    if not details.empty:
        df["asset_name_clean"] = df["asset_name"]
        df = df.merge(details, on="asset_name_clean", how="left")

    meta = pd.concat(
        [_extract_schedule_meta(path, "TITRES ETAT 31-12-2025"), _extract_schedule_meta(path, "CORPORATE 31-12-2025")],
        ignore_index=True,
    )
    if not meta.empty:
        df["schedule_key"] = df["asset_name"].map(lambda x: _match_schedule_key(x, meta["schedule_key"]))
        df.loc[~df["asset_type"].isin(["government_bond", "corporate_bond"]), "schedule_key"] = np.nan
        df = df.merge(meta, on="schedule_key", how="left")
        df["maturity_date"] = df.get("maturity_date", pd.NaT).fillna(df["maturity_date_schedule"])
        df["nominal"] = df["nominal_proxy"]
        df["coupon_rate"] = df.get("coupon_rate", np.nan).fillna(df["coupon_amount_proxy"] / df["nominal_proxy"])
    else:
        df["schedule_key"] = np.nan
        df["nominal"] = np.nan

    df["sector"] = df["asset_name"].map(infer_corporate_sector)
    df["bond_type"] = df["asset_name"].map(infer_bond_type)
    return df


def get_optimisable_portfolio(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les positions couvertes par le diagnostic pré-optimisation."""

    out = portfolio_df.loc[portfolio_df["is_optimisable"] & portfolio_df["is_position"]].copy()
    total = out["market_value"].sum()
    out["optimisable_weight"] = out["market_value"] / total
    if (out["optimisable_weight"] < -1e-12).any():
        raise ValueError("Poids négatif détecté dans la poche optimisable.")
    return out


def get_non_optimisable_portfolio(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les lignes valorisées non couvertes par le diagnostic de marché."""

    return portfolio_df.loc[portfolio_df["is_position"] & ~portfolio_df["is_optimisable"]].copy()


def build_portfolio_classification_check(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Synthétise la classification optimisable / non optimisable."""

    positions = portfolio_df.loc[portfolio_df["is_position"]].copy()
    total = positions["market_value"].sum()
    grouped = (
        positions.groupby(["asset_class_original", "asset_class_standardized", "is_optimisable"], dropna=False)
        .agg(number_lines=("asset_id", "count"), total_value=("market_value", "sum"))
        .reset_index()
    )
    grouped["total_weight"] = grouped["total_value"] / total if total else np.nan
    return grouped[
        ["asset_class_original", "asset_class_standardized", "number_lines", "total_value", "total_weight", "is_optimisable"]
    ].sort_values("total_value", ascending=False)


def portfolio_summary(portfolio_df: pd.DataFrame) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Calcule les agrégats portefeuille et le flag de rapprochement éventuel."""

    positions = portfolio_df.loc[portfolio_df["is_position"]].copy()
    total = float(positions["market_value"].sum())
    optimisable = float(positions.loc[positions["is_optimisable"], "market_value"].sum())
    non_optimisable = total - optimisable
    gap = (total - EXPECTED_TOTAL_PORTFOLIO_VALUE) / EXPECTED_TOTAL_PORTFOLIO_VALUE
    flag = "OK" if abs(gap) <= RECONCILIATION_TOLERANCE else "PORTFOLIO_TOTAL_RECONCILIATION_WARNING"
    summary = {
        "total_portfolio_value": total,
        "optimisable_value": optimisable,
        "optimisable_weight": optimisable / total if total else np.nan,
        "non_optimisable_value": non_optimisable,
        "non_optimisable_weight": non_optimisable / total if total else np.nan,
        "reconciliation_gap_vs_expected": gap,
        "quality_flag": flag,
    }
    flags = pd.DataFrame(
        [
            {
                "asset_id": "PORTFOLIO",
                "asset_name": "Portefeuille total",
                "asset_class": "Portefeuille",
                "flag": flag,
                "description": "Contrôle de cohérence du total portefeuille vs ordre de grandeur attendu.",
            }
        ]
        if flag != "OK"
        else [],
    )
    return summary, flags
