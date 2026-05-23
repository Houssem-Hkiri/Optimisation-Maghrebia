"""Weekly macro-financial APT utilities for notebook 01.

The module builds a defensible weekly APT dataset for the Tunisian market using
TUNINDEX20, the 5-year sovereign zero-coupon rate, the 5-year corporate spread,
inflation and a short-term TMM risk-free rate. It does not create synthetic
asset returns and does not impute missing returns with zero.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


APT_FACTORS_CORE = [
    "mkt_excess_tunindex20",
    "delta_zc_5y",
    "delta_credit_spread_5y",
    "delta_inflation_yoy",
]

BOUNDS_APT_PRUDENT = {
    "Actions cotées": (-0.05, 0.16),
    "Titres de l'État": (0.05, 0.105),
    "Emprunts obligataires": (0.06, 0.12),
}

BOUNDS_APT_CENTRAL = {
    "Actions cotées": (-0.05, 0.18),
    "Titres de l'État": (0.05, 0.12),
    "Emprunts obligataires": (0.06, 0.14),
}

BOUNDS_APT_OPTIMISTIC = {
    "Actions cotées": (-0.02, 0.25),
    "Titres de l'État": (0.06, 0.13),
    "Emprunts obligataires": (0.07, 0.16),
}

APT_SCENARIO_BOUNDS = {
    "prudent": BOUNDS_APT_PRUDENT,
    "central": BOUNDS_APT_CENTRAL,
    "optimistic": BOUNDS_APT_OPTIMISTIC,
}

APT_SCENARIO_FACTOR_ASSUMPTIONS = {
    "prudent": {
        "mkt_excess_tunindex20": 0.020,
        "delta_zc_5y": 0.005,
        "delta_credit_spread_5y": 0.006,
        "delta_inflation_yoy": 0.003,
    },
    "central": {
        "mkt_excess_tunindex20": 0.050,
        "delta_zc_5y": 0.000,
        "delta_credit_spread_5y": 0.000,
        "delta_inflation_yoy": 0.000,
    },
    "optimistic": {
        "mkt_excess_tunindex20": 0.070,
        "delta_zc_5y": 0.000,
        "delta_credit_spread_5y": 0.000,
        "delta_inflation_yoy": 0.000,
    },
}

APT_SCENARIO_CLASS_ADJUSTMENTS = {
    "prudent": {
        "listed_equity": -0.015,
        "government_bond": -0.003,
        "corporate_bond": -0.006,
    },
    "central": {
        "listed_equity": 0.000,
        "government_bond": 0.000,
        "corporate_bond": 0.000,
    },
    "optimistic": {
        "listed_equity": 0.020,
        "government_bond": 0.003,
        "corporate_bond": 0.008,
    },
}

APT_OPTIMISTIC_FACTOR_TREATMENT = pd.DataFrame(
    [
        {
            "Factor": "mkt_excess_tunindex20",
            "Central_Treatment": "Conservé avec prime de marché annuelle de référence",
            "Optimistic_Treatment": "Conserve avec prime de marche moderement relevee",
            "Kept_Or_Neutralized": "KEPT",
            "Statistical_Reason": "Facteur de marché disponible sur l'échantillon et économiquement discriminant pour les actions.",
            "Economic_Justification": "Un environnement favorable peut soutenir la prime actions sans extrapoler les performances extremes de 2025.",
            "Impact_Expected": "Hausse modérée des rendements attendus des actions cotées.",
        },
        {
            "Factor": "delta_zc_5y",
            "Central_Treatment": "Conservé dans la régression, vue directionnelle nulle",
            "Optimistic_Treatment": "Neutralise comme choc directionnel defavorable",
            "Kept_Or_Neutralized": "NEUTRALIZED_DIRECTIONAL_SHOCK",
            "Statistical_Reason": "Facteur de taux fragile sur un échantillon court et sensible aux observations de courbe.",
            "Economic_Justification": "Un scenario favorable n'ajoute pas de choc de hausse des taux aux actifs obligataires.",
            "Impact_Expected": "Réduction de la pénalisation des titres obligataires en scénario favorable.",
        },
        {
            "Factor": "delta_credit_spread_5y",
            "Central_Treatment": "Conservé dans la régression, vue directionnelle nulle",
            "Optimistic_Treatment": "Neutralise comme choc d'elargissement de spread",
            "Kept_Or_Neutralized": "NEUTRALIZED_DIRECTIONAL_SHOCK",
            "Statistical_Reason": "Spread corporate observé sur un marché peu liquide, donc potentiellement instable.",
            "Economic_Justification": "Un regime favorable suppose une absence d'elargissement du credit, sans supprimer le facteur du modele.",
            "Impact_Expected": "Hausse relative des rendements attendus corporate par détente du risque de crédit.",
        },
        {
            "Factor": "delta_inflation_yoy",
            "Central_Treatment": "Conservé dans la régression, vue directionnelle nulle",
            "Optimistic_Treatment": "Neutralise comme penalisation macro additionnelle",
            "Kept_Or_Neutralized": "NEUTRALIZED_IF_PENALIZING",
            "Statistical_Reason": "Pouvoir explicatif fragile sur une seule année d'observation.",
            "Economic_Justification": "La neutralisation evite de sur-penaliser les actifs risqués lorsque l'environnement macro-financier est suppose plus detendu.",
            "Impact_Expected": "Diminution d'une pénalité macro additionnelle sur les actifs risqués.",
        },
    ]
)

TMM_FALLBACK_2025 = pd.DataFrame(
    {
        "month": pd.period_range("2025-01", "2025-12", freq="M").astype(str),
        "tmm_annual_decimal": [
            0.0799,
            0.0799,
            0.0791,
            0.0750,
            0.0750,
            0.0750,
            0.0750,
            0.0750,
            0.0749,
            0.0749,
            0.0749,
            0.0749,
        ],
        "source": "TMM_RECONSTRUCTED_FROM_PUBLIC_BCT_REFERENCES",
        "source_note": (
            "Série mensuelle de repli fournie dans le cahier des charges. "
            "À remplacer par une extraction directe BCT si un fichier interne est disponible."
        ),
    }
)

FRENCH_MONTHS = {
    "jan": 1,
    "janv": 1,
    "fev": 2,
    "fév": 2,
    "mar": 3,
    "mars": 3,
    "avr": 4,
    "mai": 5,
    "jui": 6,
    "juin": 6,
    "jul": 7,
    "juil": 7,
    "aou": 8,
    "aoû": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    "déc": 12,
}


@dataclass(frozen=True)
class APTParameters:
    """Configuration used by the weekly APT model."""

    alpha_shrinkage: float = 0.25
    expected_mkt_premium_annual: float = 0.05
    min_obs_warning: int = 30
    min_obs_ok: int = 40
    periods_per_year: int = 52
    equity_bounds: tuple[float, float] = (-0.05, 0.18)
    government_bounds: tuple[float, float] = (0.05, 0.12)
    corporate_bounds: tuple[float, float] = (0.06, 0.14)


def _snake(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    return text.strip("_")


def _parse_month_label(value: object) -> pd.Period | pd.NaT:
    text = str(value).strip().lower()
    text = text.replace("é", "e").replace("è", "e").replace("ê", "e").replace("û", "u").replace("û", "u")
    match = re.search(r"([a-z]+)\s+(\d{4})", text)
    if not match:
        return pd.NaT
    month_key = match.group(1)[:4]
    month = FRENCH_MONTHS.get(month_key) or FRENCH_MONTHS.get(month_key[:3])
    if month is None:
        return pd.NaT
    return pd.Period(year=int(match.group(2)), month=int(month), freq="M")


def _to_decimal(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "").replace(",", ".")
        num = pd.to_numeric(cleaned, errors="coerce")
    else:
        num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return np.nan
    num = float(num)
    if abs(num) > 1.0:
        return num / 100.0
    return num


def short_name(value: object, n: int = 28) -> str:
    """Return a compact label for charts."""

    text = str(value).strip()
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def weekly_compounded_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound available daily returns into weekly returns without imputing NaNs.

    Weeks are labelled with the last real observation date in the week, not with
    the theoretical Friday label produced by ``resample('W-FRI')``. This avoids
    creating a 2026-01-02 label for the final shortened 2025 week.
    """

    data = daily_returns.copy()
    data.index = pd.to_datetime(data.index)
    data = data.loc[data.index <= pd.Timestamp("2025-12-31")]

    def _compound(series: pd.Series) -> float:
        clean = series.dropna()
        return np.nan if clean.empty else float((1.0 + clean).prod() - 1.0)

    week_key = data.index.to_period("W-FRI")
    weekly = data.groupby(week_key).apply(lambda frame: frame.apply(_compound))
    weekly.index = pd.Series(data.index, index=data.index).groupby(week_key).max().to_numpy()
    weekly.index = pd.to_datetime(weekly.index)
    weekly.index.name = "week_date"
    return weekly.dropna(how="all")


def load_tunindex20_weekly(path: str | Path) -> pd.DataFrame:
    """Load TUNINDEX20 and compute weekly price returns."""

    raw = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    lib = raw["LIB_INDICE"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    tun = raw.loc[lib.eq("TUNINDEX20")].copy()
    if tun.empty:
        raise ValueError("TUNINDEX20 absent de histo_indice_2025.csv.")
    level_col = next((c for c in tun.columns if c.strip().upper().startswith("INDICE_JOUR")), None)
    if level_col is None:
        raise ValueError("Colonne INDICE_JOUR absente pour TUNINDEX20.")
    tun["date"] = pd.to_datetime(tun["SEANCE"], dayfirst=True, errors="coerce")
    tun["tunindex20_level"] = pd.to_numeric(tun[level_col], errors="coerce")
    tun = tun.dropna(subset=["date", "tunindex20_level"]).sort_values("date")
    tun = tun.loc[tun["date"] <= pd.Timestamp("2025-12-31")]
    week_key = tun["date"].dt.to_period("W-FRI")
    weekly = tun.groupby(week_key).tail(1).set_index("date")[["tunindex20_level"]]
    weekly["tunindex20_return"] = weekly["tunindex20_level"].pct_change()
    weekly.index.name = "week_date"
    return weekly.reset_index()


def load_tmm_weekly(data_dir: str | Path, week_dates: Iterable[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an internal TMM file or use the documented 2025 fallback values."""

    data_dir = Path(data_dir)
    candidates = list(data_dir.glob("*tmm*.csv")) + list(data_dir.glob("*TMM*.csv")) + list(data_dir.glob("*tmm*.xlsx")) + list(data_dir.glob("*TMM*.xlsx"))
    source_flag = "TMM_RECONSTRUCTED_FROM_PUBLIC_BCT_REFERENCES"
    if candidates:
        path = candidates[0]
        if path.suffix.lower() == ".csv":
            monthly = pd.read_csv(path, sep=None, engine="python")
        else:
            monthly = pd.read_excel(path)
        monthly.columns = [_snake(c) for c in monthly.columns]
        if "month" not in monthly.columns or "tmm_annual_decimal" not in monthly.columns:
            raise ValueError(f"Fichier TMM trouvé mais colonnes attendues absentes : {path}")
        source_flag = "TMM_INTERNAL_FILE"
        monthly["source"] = monthly.get("source", source_flag)
        monthly["source_note"] = monthly.get("source_note", str(path))
    else:
        monthly = TMM_FALLBACK_2025.copy()
    monthly["month_period"] = pd.PeriodIndex(monthly["month"].astype(str), freq="M")
    monthly["tmm_annual_decimal"] = monthly["tmm_annual_decimal"].map(_to_decimal)
    weeks = pd.DataFrame({"week_date": pd.to_datetime(list(week_dates))})
    weeks["month_period"] = weeks["week_date"].dt.to_period("M")
    weekly = weeks.merge(monthly[["month_period", "tmm_annual_decimal", "source", "source_note"]], on="month_period", how="left")
    weekly["rf_weekly"] = (1.0 + weekly["tmm_annual_decimal"]) ** (1.0 / 52.0) - 1.0
    weekly["tmm_quality_flag"] = np.where(weekly["tmm_annual_decimal"].notna(), source_flag, "TMM_MISSING")
    assumptions = pd.DataFrame(
        [
            {
                "factor": "rf_annual",
                "expected_periodic_value": float(weekly["rf_weekly"].dropna().mean()),
                "expected_annualized_value": float(weekly["tmm_annual_decimal"].dropna().mean()),
                "source": str(weekly["source"].dropna().mode().iloc[0]) if weekly["source"].notna().any() else source_flag,
                "justification": str(weekly["source_note"].dropna().iloc[0]) if weekly["source_note"].notna().any() else "",
            }
        ]
    )
    return weekly.drop(columns=["month_period"]), assumptions


def load_inflation_monthly(path: str | Path) -> pd.DataFrame:
    """Extract global YoY family CPI inflation from the socio-economic workbook."""

    raw = pd.read_excel(path, sheet_name=0, header=None)
    row_mask = raw.iloc[:, 0].astype(str).str.contains("Glissement annuel de l'Indice de Prix", case=False, na=False)
    if not row_mask.any():
        raise ValueError("Ligne inflation IPC en glissement annuel introuvable.")
    header = raw.iloc[1]
    values = raw.loc[row_mask].iloc[0]
    rows: list[dict[str, object]] = []
    for col in raw.columns[1:]:
        period = _parse_month_label(header[col])
        if pd.isna(period):
            continue
        rows.append({"month_period": period, "inflation_yoy": _to_decimal(values[col])})
    out = pd.DataFrame(rows).dropna(subset=["month_period", "inflation_yoy"])
    return out.loc[out["month_period"].dt.year.eq(2025)].sort_values("month_period")


def inflation_to_weekly(monthly: pd.DataFrame, week_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Apply inflation of month M to weeks of month M+1 to limit look-ahead bias."""

    shifted = monthly.copy()
    shifted["application_month"] = shifted["month_period"] + 1
    weeks = pd.DataFrame({"week_date": pd.to_datetime(list(week_dates))})
    weeks["application_month"] = weeks["week_date"].dt.to_period("M")
    out = weeks.merge(shifted[["application_month", "inflation_yoy"]], on="application_month", how="left")
    out["delta_inflation_yoy"] = out["inflation_yoy"].diff()
    out["inflation_quality_flag"] = np.where(out["inflation_yoy"].notna(), "MONTHLY_INFLATION_LAGGED_ONE_MONTH", "INFLATION_MISSING")
    return out.drop(columns=["application_month"])


def _interpolate_by_date(df: pd.DataFrame, date_col: str, maturity_col: str, value_col: str, target: float = 5.0) -> pd.DataFrame:
    rows = []
    local = df.dropna(subset=[date_col, maturity_col, value_col]).copy()
    local[date_col] = pd.to_datetime(local[date_col])
    for date, group in local.groupby(date_col):
        curve = group.sort_values(maturity_col)
        x = curve[maturity_col].astype(float).to_numpy()
        y = curve[value_col].astype(float).to_numpy()
        if len(x) == 0 or target < np.nanmin(x) or target > np.nanmax(x):
            val = np.nan
        else:
            val = float(np.interp(target, x, y))
        rows.append({"week_date": pd.Timestamp(date), value_col: val})
    return pd.DataFrame(rows).sort_values("week_date")


def load_zc_5y_weekly(path: str | Path) -> pd.DataFrame:
    """Load weekly sovereign ZC and interpolate at 5 years."""

    zc = pd.read_excel(path, sheet_name="zc_weekly_standardized")
    zc["requested_date"] = pd.to_datetime(zc["requested_date"])
    out = _interpolate_by_date(zc, "requested_date", "maturity_years", "zc_actuarial_decimal", 5.0)
    out = out.rename(columns={"zc_actuarial_decimal": "zc_5y"})
    out["delta_zc_5y"] = out["zc_5y"].diff()
    return out


def load_corporate_5y_weekly(path: str | Path, corporate_sector_weights: pd.Series | None = None) -> pd.DataFrame:
    """Load weekly corporate curves and build an aggregate 5Y corporate yield/spread."""

    corp = pd.read_excel(path, sheet_name="corporate_sector_curves")
    corp["requested_date"] = pd.to_datetime(corp["requested_date"])
    corp["sector"] = corp["sector"].astype(str).str.upper().str.strip().replace({"BANKING": "BANCAIRE"})
    five = corp.loc[corp["maturity_years"].astype(float).between(4.99, 5.01)].copy()
    if five.empty:
        five = corp.groupby(["requested_date", "sector"], as_index=False).apply(
            lambda g: pd.Series(
                {
                    "corporate_yield_decimal": np.interp(5.0, g["maturity_years"].astype(float), g["corporate_yield_decimal"].astype(float))
                    if g["maturity_years"].min() <= 5.0 <= g["maturity_years"].max()
                    else np.nan,
                    "sovereign_zc_decimal": np.interp(5.0, g["maturity_years"].astype(float), g["sovereign_zc_decimal"].astype(float))
                    if g["maturity_years"].min() <= 5.0 <= g["maturity_years"].max()
                    else np.nan,
                }
            ),
            include_groups=False,
        ).reset_index()
    five["credit_spread_5y_sector"] = five["corporate_yield_decimal"] - five["sovereign_zc_decimal"]
    weights = None
    source = "SIMPLE_SECTOR_MEAN"
    if corporate_sector_weights is not None and not corporate_sector_weights.empty:
        weights = corporate_sector_weights.copy()
        weights.index = weights.index.astype(str).str.upper().str.strip().str.replace("BANKING", "BANCAIRE", regex=False)
        weights = weights / weights.sum()
        source = "PORTFOLIO_CORPORATE_SECTOR_WEIGHTED"
    rows: list[dict[str, object]] = []
    for date, group in five.groupby("requested_date"):
        group = group.dropna(subset=["corporate_yield_decimal", "sovereign_zc_decimal", "credit_spread_5y_sector"])
        if group.empty:
            rows.append({"week_date": date, "corporate_yield_5y": np.nan, "credit_spread_5y": np.nan, "credit_spread_source": "NO_CORPORATE_5Y"})
            continue
        if weights is not None:
            joined = group.set_index("sector").join(weights.rename("weight"), how="inner")
            if joined.empty or joined["weight"].sum() <= 0:
                y = float(group["corporate_yield_decimal"].mean())
                spread = float(group["credit_spread_5y_sector"].mean())
                used_source = "SIMPLE_SECTOR_MEAN_NO_MATCHING_WEIGHTS"
            else:
                w = joined["weight"] / joined["weight"].sum()
                y = float((joined["corporate_yield_decimal"] * w).sum())
                spread = float((joined["credit_spread_5y_sector"] * w).sum())
                used_source = source
        else:
            y = float(group["corporate_yield_decimal"].mean())
            spread = float(group["credit_spread_5y_sector"].mean())
            used_source = source
        rows.append({"week_date": pd.Timestamp(date), "corporate_yield_5y": y, "credit_spread_5y": spread, "credit_spread_source": used_source})
    out = pd.DataFrame(rows).sort_values("week_date")
    out["delta_credit_spread_5y"] = out["credit_spread_5y"].diff()
    return out


def build_apt_factors_weekly(
    data_dir: str | Path,
    daily_returns: pd.DataFrame,
    portfolio_optimisable: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the weekly factor table used by the APT model."""

    data_dir = Path(data_dir)
    weekly_returns = weekly_compounded_returns(daily_returns)
    week_dates = pd.Series(weekly_returns.index, name="week_date")
    tun = load_tunindex20_weekly(data_dir / "histo_indice_2025.csv")
    tmm, tmm_assumption = load_tmm_weekly(data_dir, week_dates)
    inflation_path = next(data_dir.glob("Socio*.xlsx"))
    inflation = inflation_to_weekly(load_inflation_monthly(inflation_path), week_dates)
    zc = load_zc_5y_weekly(data_dir / "zc_zero_coupon_weekly_2022_2025_standardized.xlsx")
    corp_weights = (
        portfolio_optimisable.loc[portfolio_optimisable["asset_type"].eq("corporate_bond")]
        .groupby("sector")["market_value"]
        .sum()
    )
    corp = load_corporate_5y_weekly(data_dir / "corporate_curves_weekly_2022_2025_standardized.xlsx", corp_weights)
    factors = pd.DataFrame({"week_date": week_dates})
    for part in [tun, tmm, zc, corp, inflation]:
        factors = factors.merge(part, on="week_date", how="left")
    factors = factors.loc[factors["week_date"] <= pd.Timestamp("2025-12-31")].copy()
    factors["mkt_excess_tunindex20"] = factors["tunindex20_return"] - factors["rf_weekly"]
    flag_cols = [c for c in ["tmm_quality_flag", "inflation_quality_flag", "credit_spread_source"] if c in factors.columns]
    factors["source_flags"] = factors[flag_cols].astype(str).agg(";".join, axis=1)
    factors = factors[
        [
            "week_date",
            "tunindex20_level",
            "tunindex20_return",
            "tmm_annual_decimal",
            "rf_weekly",
            "mkt_excess_tunindex20",
            "zc_5y",
            "delta_zc_5y",
            "corporate_yield_5y",
            "credit_spread_5y",
            "delta_credit_spread_5y",
            "inflation_yoy",
            "delta_inflation_yoy",
            "source_flags",
        ]
    ]
    return factors, tmm_assumption


def build_rate_factor_justification(portfolio_optimisable: pd.DataFrame) -> pd.DataFrame:
    """Document whether available durations support the 5Y rate pivot."""

    duration_cols = [
        c
        for c in portfolio_optimisable.columns
        if "duration" in str(c).lower() and pd.to_numeric(portfolio_optimisable[c], errors="coerce").notna().any()
    ]
    rows = []
    if duration_cols:
        col = duration_cols[0]
        for asset_type, label in [("government_bond", "Titres d'État"), ("corporate_bond", "Obligations corporate")]:
            part = portfolio_optimisable.loc[portfolio_optimisable["asset_type"].eq(asset_type)].copy()
            dur = pd.to_numeric(part[col], errors="coerce")
            weights = pd.to_numeric(part["market_value"], errors="coerce")
            ok = dur.notna() & weights.notna() & weights.gt(0)
            avg = float(np.average(dur[ok], weights=weights[ok])) if ok.any() else np.nan
            rows.append(
                {
                    "scope": label,
                    "weighted_average_duration": avg,
                    "duration_source_column": col,
                    "rate_factor_maturity_years": 5.0,
                    "quality_flag": "OK" if np.isfinite(avg) else "DURATION_DATA_NOT_AVAILABLE",
                }
            )
        all_bonds = portfolio_optimisable.loc[portfolio_optimisable["asset_type"].isin(["government_bond", "corporate_bond"])].copy()
        dur = pd.to_numeric(all_bonds[col], errors="coerce")
        weights = pd.to_numeric(all_bonds["market_value"], errors="coerce")
        ok = dur.notna() & weights.notna() & weights.gt(0)
        avg = float(np.average(dur[ok], weights=weights[ok])) if ok.any() else np.nan
        rows.append(
            {
                "scope": "Poche obligataire",
                "weighted_average_duration": avg,
                "duration_source_column": col,
                "rate_factor_maturity_years": 5.0,
                "quality_flag": "OK" if np.isfinite(avg) else "DURATION_DATA_NOT_AVAILABLE",
            }
        )
    else:
        for scope in ["Titres d'État", "Obligations corporate", "Poche obligataire"]:
            rows.append(
                {
                    "scope": scope,
                    "weighted_average_duration": np.nan,
                    "duration_source_column": "",
                    "rate_factor_maturity_years": 5.0,
                    "quality_flag": "DURATION_DATA_NOT_AVAILABLE",
                }
            )
    out = pd.DataFrame(rows)
    out["justification"] = np.where(
        out["quality_flag"].eq("OK"),
        "Le facteur ZC 5 ans est comparé à la duration moyenne pondérée disponible.",
        "Durations complètes absentes : le ZC 5 ans est conservé comme maturité pivot méthodologique, à renforcer lorsque les durations seront disponibles.",
    )
    return out


def compute_factor_variability(factors: pd.DataFrame) -> pd.DataFrame:
    """Compute variability diagnostics for the four core APT factors."""

    rows = []
    for factor in APT_FACTORS_CORE:
        series = pd.to_numeric(factors[factor], errors="coerce").dropna()
        if series.empty:
            rows.append(
                {
                    "factor": factor,
                    "mean": np.nan,
                    "weekly_std": np.nan,
                    "minimum": np.nan,
                    "maximum": np.nan,
                    "distinct_values": 0,
                    "zero_change_share": np.nan,
                    "quality_flag": "FACTOR_MISSING",
                }
            )
            continue
        std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        distinct = int(series.round(12).nunique())
        zero_share = float(series.abs().le(1e-12).mean())
        flags = []
        if std < 1e-6:
            flags.append("LOW_FACTOR_VARIABILITY")
        if distinct <= 3:
            flags.append("NEAR_CONSTANT_FACTOR")
        if factor == "delta_credit_spread_5y" and (std < 1e-5 or zero_share > 0.80):
            flags.append("CREDIT_SPREAD_LOW_VARIABILITY")
        rows.append(
            {
                "factor": factor,
                "mean": float(series.mean()),
                "weekly_std": std,
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "distinct_values": distinct,
                "zero_change_share": zero_share,
                "quality_flag": ";".join(flags) if flags else "OK",
            }
        )
    return pd.DataFrame(rows)


def build_stress_test_summary(
    portfolio_optimisable: pd.DataFrame,
    duration_justification: pd.DataFrame,
) -> pd.DataFrame:
    """Build minimal economic stress tests without inventing missing durations."""

    rows = []
    duration_available = duration_justification["quality_flag"].eq("OK").any()
    for asset_type, label, shocks, shock_type in [
        ("government_bond", "Stress taux souverain", [0.01, 0.02], "RATE_UP"),
        ("corporate_bond", "Stress spread corporate", [0.01, 0.02], "SPREAD_UP"),
        ("listed_equity", "Stress actions", [-0.10, -0.20], "EQUITY_DOWN"),
    ]:
        exposure = float(pd.to_numeric(portfolio_optimisable.loc[portfolio_optimisable["asset_type"].eq(asset_type), "market_value"], errors="coerce").sum())
        for shock in shocks:
            if asset_type in {"government_bond", "corporate_bond"}:
                scope = "Titres d'État" if asset_type == "government_bond" else "Obligations corporate"
                duration_row = duration_justification.loc[duration_justification["scope"].eq(scope)]
                duration = float(duration_row["weighted_average_duration"].iloc[0]) if not duration_row.empty and pd.notna(duration_row["weighted_average_duration"].iloc[0]) else np.nan
                if np.isfinite(duration):
                    impact_pct = -duration * shock
                    flag = "OK"
                else:
                    impact_pct = np.nan
                    flag = "DURATION_MISSING_FOR_STRESS_TEST"
            else:
                impact_pct = shock
                duration = np.nan
                flag = "OK"
            rows.append(
                {
                    "stress_name": label,
                    "asset_type": asset_type,
                    "shock_type": shock_type,
                    "shock_size": shock,
                    "weighted_duration_used": duration,
                    "exposure_value": exposure,
                    "estimated_impact_pct": impact_pct,
                    "estimated_impact_value": exposure * impact_pct if np.isfinite(impact_pct) else np.nan,
                    "quality_flag": flag,
                }
            )
    return pd.DataFrame(rows)


def compute_vif(factors: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF for the core factors."""

    x = factors[APT_FACTORS_CORE].dropna()
    rows = []
    if len(x) < 3:
        return pd.DataFrame({"factor": APT_FACTORS_CORE, "vif": np.nan, "quality_flag": "INSUFFICIENT_FACTOR_OBSERVATIONS"})
    values = sm.add_constant(x.astype(float), has_constant="add").to_numpy(float)
    columns = ["const"] + APT_FACTORS_CORE
    for i, factor in enumerate(columns):
        if factor == "const":
            continue
        vif = float(variance_inflation_factor(values, i))
        flag = "SEVERE_MULTICOLLINEARITY" if vif > 10 else ("WARNING_MULTICOLLINEARITY" if vif > 5 else "OK")
        rows.append({"factor": factor, "vif": vif, "quality_flag": flag})
    return pd.DataFrame(rows)


def detect_available_apt_factors(factors_df: pd.DataFrame) -> pd.DataFrame:
    """Detect available APT factors and assess whether they are usable.

    The function does not invent unavailable factors. It inspects the factor
    dataset built from project inputs and documents coverage, sample size and
    variance before the APT estimation is interpreted.
    """

    candidate_factors = [
        "mkt_excess_tunindex20",
        "delta_zc_5y",
        "delta_credit_spread_5y",
        "delta_inflation_yoy",
        "slope_zc_10y_2y",
        "delta_slope_zc_10y_2y",
        "liquidity_factor",
        "equity_market_factor",
    ]
    rows: list[dict[str, object]] = []
    n_total = max(1, len(factors_df))
    for factor in candidate_factors:
        available = factor in factors_df.columns
        values = pd.to_numeric(factors_df[factor], errors="coerce") if available else pd.Series(dtype=float)
        n_obs = int(values.notna().sum())
        coverage = float(n_obs / n_total)
        volatility = float(values.std(ddof=1)) if n_obs >= 2 else np.nan
        usable = available and coverage >= 0.80 and n_obs >= 30 and np.isfinite(volatility) and volatility > 1e-10
        if not available:
            status = "NON_AVAILABLE"
        elif coverage < 0.80:
            status = "FRAGILE_LOW_COVERAGE"
        elif n_obs < 30:
            status = "FRAGILE_SHORT_SAMPLE"
        elif not np.isfinite(volatility) or volatility <= 1e-10:
            status = "NON_INFORMATIVE_LOW_VARIANCE"
        else:
            status = "USABLE"
        rows.append(
            {
                "Factor": factor,
                "Available": bool(available),
                "Coverage_Ratio": coverage if available else 0.0,
                "N_Obs": n_obs,
                "Mean": float(values.mean()) if n_obs else np.nan,
                "Volatility": volatility,
                "Usable": bool(usable),
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


def build_apt_factor_diagnostics(factors_df: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Build a PFE-facing diagnostic table for available APT factors."""

    available = detect_available_apt_factors(factors_df)
    returns = asset_returns.copy()
    returns.index = pd.to_datetime(returns.index)
    factors = factors_df.copy()
    if "week_date" in factors.columns:
        factors["week_date"] = pd.to_datetime(factors["week_date"])
        factors = factors.set_index("week_date")
    factors.index = pd.to_datetime(factors.index)

    economic_relevance = {
        "mkt_excess_tunindex20": "Risque de marché actions tunisien",
        "delta_zc_5y": "Risque de taux souverain moyen terme",
        "delta_credit_spread_5y": "Risque de crédit corporate",
        "delta_inflation_yoy": "Risque macro-inflation",
        "slope_zc_10y_2y": "Pente de courbe",
        "delta_slope_zc_10y_2y": "Variation de pente de courbe",
        "liquidity_factor": "Risque de liquidité",
        "equity_market_factor": "Facteur actions alternatif",
    }
    central_factors = set(APT_FACTORS_CORE)
    rows: list[dict[str, object]] = []
    for _, row in available.iterrows():
        factor = str(row["Factor"])
        if bool(row["Available"]) and factor in factors.columns:
            factor_series = pd.to_numeric(factors[factor], errors="coerce")
            aligned = returns.join(factor_series.rename(factor), how="inner")
            corr_values = []
            for asset in returns.columns:
                pair = aligned[[asset, factor]].dropna()
                if len(pair) >= 10 and pair[asset].std(ddof=1) > 0 and pair[factor].std(ddof=1) > 0:
                    corr_values.append(abs(float(pair[asset].corr(pair[factor]))))
            avg_corr = float(np.nanmean(corr_values)) if corr_values else np.nan
        else:
            avg_corr = np.nan

        coverage = float(row["Coverage_Ratio"])
        n_obs = int(row["N_Obs"])
        vol = float(row["Volatility"]) if pd.notna(row["Volatility"]) else np.nan
        weak_corr = pd.isna(avg_corr) or avg_corr < 0.05
        low_coverage = coverage < 0.80
        short_sample = n_obs < 30
        low_variance = (not np.isfinite(vol)) or vol <= 1e-10
        significance_status = "FAIBLE_DISCRIMINATION" if weak_corr else "DISCRIMINANT"
        stability_status = "NON_INFORMATIF" if low_variance else ("FRAGILE" if low_coverage or short_sample else "EXPLOITABLE")
        available_and_in_core = bool(row["Available"]) and factor in central_factors and not low_variance
        use_central = available_and_in_core and coverage > 0 and n_obs > 0
        use_prudent = use_central
        use_optimistic = use_central and not (stability_status == "FRAGILE" and weak_corr)
        if not bool(row["Available"]):
            reason = "Série absente des données disponibles."
        elif low_variance:
            reason = "Variance quasi nulle : facteur non informatif."
        elif low_coverage:
            reason = "Couverture inférieure à 80 %, facteur fragile."
        elif short_sample:
            reason = "Moins de 30 observations, facteur fragile."
        elif weak_corr:
            reason = "Corrélation moyenne faible ; conservé si pertinent économiquement."
        else:
            reason = "Facteur exploitable et économiquement pertinent."
        rows.append(
            {
                "Factor": factor,
                "Available": bool(row["Available"]),
                "Coverage_Ratio": coverage,
                "N_Obs": n_obs,
                "Mean": row["Mean"],
                "Volatility": row["Volatility"],
                "Correlation_With_Asset_Returns_Avg": avg_corr,
                "Significance_Status": significance_status,
                "Stability_Status": stability_status,
                "Economic_Relevance": economic_relevance.get(factor, "Facteur détecté dans les données"),
                "Use_In_Central": bool(use_central),
                "Use_In_Prudent": bool(use_prudent),
                "Use_In_Optimistic": bool(use_optimistic),
                "Reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_apt_bounds_by_asset_class() -> pd.DataFrame:
    """Return documented annual expected-return bounds by scenario and asset class."""

    justifications = {
        "Actions cotées": "Bornes plus larges car le risque actions est plus élevé ; plafond encadré pour ne pas extrapoler 2025.",
        "Titres de l'État": "Bornes serrées autour du TSR et de la prime de maturité souveraine observée.",
        "Emprunts obligataires": "Bornes intermédiaires intégrant une prime de crédit sans créer de spread absent.",
    }
    rows: list[dict[str, object]] = []
    for scenario, bounds in APT_SCENARIO_BOUNDS.items():
        for asset_class, (lower, upper) in bounds.items():
            rows.append(
                {
                    "Scenario": scenario,
                    "Asset_Class": asset_class,
                    "Lower_Bound": float(lower),
                    "Upper_Bound": float(upper),
                    "Justification": justifications.get(asset_class, "Borne de calibration économique par classe d'actifs."),
                }
            )
    return pd.DataFrame(rows)


def build_scenario_differentiation_check(expected: pd.DataFrame, scenario_summary: pd.DataFrame) -> pd.DataFrame:
    """Check that prudent, central and optimistic APT scenarios are meaningfully distinct."""

    required = ["mu_apt_prudent", "mu_apt_central", "mu_apt_optimistic"]
    if any(col not in expected.columns for col in required):
        raise ValueError("Colonnes de scénarios APT manquantes pour le contrôle de différenciation.")
    mu = expected[required].astype(float)
    means = mu.mean()
    monotone_share = float(((mu["mu_apt_prudent"] <= mu["mu_apt_central"] + 1e-12) & (mu["mu_apt_central"] <= mu["mu_apt_optimistic"] + 1e-12)).mean())
    opt_gap = float((mu["mu_apt_optimistic"] - mu["mu_apt_central"]).mean())
    prudent_gap = float((mu["mu_apt_central"] - mu["mu_apt_prudent"]).mean())
    prudent_equal_central_share = float(np.isclose(mu["mu_apt_prudent"], mu["mu_apt_central"], atol=1e-12).mean())
    rows = [
        {
            "Check": "Mean_Prudent_lt_Central_lt_Optimistic",
            "Result": bool(means["mu_apt_prudent"] < means["mu_apt_central"] < means["mu_apt_optimistic"]),
            "Value": f"{means['mu_apt_prudent']:.6f} < {means['mu_apt_central']:.6f} < {means['mu_apt_optimistic']:.6f}",
            "Threshold": "ordre strict",
            "Status": "PASSED" if means["mu_apt_prudent"] < means["mu_apt_central"] < means["mu_apt_optimistic"] else "FAILED",
            "Comment": "Les rendements moyens doivent refléter les trois régimes de marché.",
        },
        {
            "Check": "Asset_Level_Monotonicity_Share",
            "Result": bool(monotone_share >= 0.70),
            "Value": monotone_share,
            "Threshold": 0.70,
            "Status": "PASSED" if monotone_share >= 0.70 else "FAILED",
            "Comment": "Au moins 70 % des actifs doivent respecter prudent <= central <= optimiste.",
        },
        {
            "Check": "Optimistic_Central_Average_Gap",
            "Result": bool(opt_gap >= 0.005),
            "Value": opt_gap,
            "Threshold": 0.005,
            "Status": "PASSED" if opt_gap >= 0.005 else "FAILED",
            "Comment": "L'écart moyen optimiste-central doit être visible pour l'optimisation.",
        },
        {
            "Check": "Central_Prudent_Average_Gap",
            "Result": bool(prudent_gap >= 0.005),
            "Value": prudent_gap,
            "Threshold": 0.005,
            "Status": "PASSED" if prudent_gap >= 0.005 else "FAILED",
            "Comment": "L'écart moyen central-prudent doit être visible pour l'optimisation.",
        },
        {
            "Check": "APT_SCENARIO_DIFFERENTIATION_OK",
            "Result": bool(prudent_equal_central_share <= 0.30),
            "Value": prudent_equal_central_share,
            "Threshold": 0.30,
            "Status": "PASSED" if prudent_equal_central_share <= 0.30 else "FAILED",
            "Comment": "Le contrôle échoue si plus de 30 % des actifs ont mu_prudent égal à mu_central.",
        },
        {
            "Check": "Central_Selected_As_Reference",
            "Result": bool(np.allclose(mu["mu_apt_central"], expected["mu_expected_reference"].astype(float), atol=1e-12)) if "mu_expected_reference" in expected.columns else False,
            "Value": "mu_expected_reference = mu_apt_central",
            "Threshold": "égalité numérique",
            "Status": "PASSED" if "mu_expected_reference" in expected.columns and np.allclose(mu["mu_apt_central"], expected["mu_expected_reference"].astype(float), atol=1e-12) else "FAILED",
            "Comment": "Le scénario central reste la référence principale du notebook 02.",
        },
    ]
    return pd.DataFrame(rows)


def bound_saturation_by_asset(scenario_audit: pd.DataFrame, assets: pd.Series | list[str]) -> pd.Series:
    """Flag assets whose prudent lower or optimistic upper scenario bound is saturated."""

    flags = pd.Series("NONE", index=pd.Index(pd.Series(assets, dtype=str), name="asset_id"), dtype=object)
    if scenario_audit.empty:
        return flags
    audit = scenario_audit.copy()
    audit["Asset"] = audit["Asset"].astype(str)
    prudent_lower = audit.loc[
        audit["scenario"].eq("prudent") & audit["clip_direction"].eq("LOW"),
        "Asset",
    ]
    optimistic_upper = audit.loc[
        audit["scenario"].eq("optimistic") & audit["clip_direction"].eq("HIGH"),
        "Asset",
    ]
    flags.loc[flags.index.intersection(prudent_lower)] = "PRUDENT_LOWER"
    flags.loc[flags.index.intersection(optimistic_upper)] = "OPTIMISTIC_UPPER"
    return flags


def select_bounds(asset_type: str, params: APTParameters) -> tuple[float, float]:
    """Return prudential expected-return bounds by asset type."""

    if asset_type == "listed_equity":
        return params.equity_bounds
    if asset_type == "government_bond":
        return params.government_bounds
    if asset_type == "corporate_bond":
        return params.corporate_bounds
    return (-0.05, 0.18)


def class_name_from_asset_type(asset_type: object, fallback: object = None) -> str:
    """Map internal asset types to the French asset-class labels used in APT bounds."""

    aliases = {
        "Actions cotees": "Actions cotées",
        "Actions cotées": "Actions cotées",
        "Titres de l'Etat": "Titres de l'État",
        "Titres de l'État": "Titres de l'État",
        "Obligations corporate": "Emprunts obligataires",
        "Emprunts obligataires": "Emprunts obligataires",
    }
    if pd.notna(fallback) and str(fallback).strip():
        label = str(fallback).strip()
        return aliases.get(label, label)
    mapping = {
        "listed_equity": "Actions cotées",
        "government_bond": "Titres de l'État",
        "corporate_bond": "Emprunts obligataires",
    }
    return mapping.get(str(asset_type), str(asset_type))


def apply_class_bounds(
    mu_raw: pd.Series,
    asset_classes: pd.Series | dict[str, str],
    bounds_dict: dict[str, tuple[float, float]],
    scenario_name: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Applique des bornes de calibration par classe d'actifs aux rendements attendus APT bruts.

    Les bornes servent à limiter le risque d'estimation et non à représenter des contraintes réglementaires.
    La fonction retourne la série bornée et un audit ligne par ligne.
    """

    mu = pd.Series(mu_raw, dtype=float).copy()
    classes = pd.Series(asset_classes, dtype=object)
    classes.index = classes.index.astype(str)
    mu.index = mu.index.astype(str)
    missing = sorted(set(mu.index) - set(classes.index))
    if missing:
        raise ValueError(f"Classes d'actifs manquantes pour les rendements APT : {missing}")

    rows: list[dict[str, object]] = []
    bounded_values: dict[str, float] = {}
    for asset, raw_value in mu.items():
        asset_class = class_name_from_asset_type("", classes.loc[asset])
        if asset_class not in bounds_dict:
            raise ValueError(f"Borne APT absente pour la classe d'actifs : {asset_class}")
        lower, upper = bounds_dict[asset_class]
        if not np.isfinite(raw_value):
            raise ValueError(f"Rendement APT brut non fini pour {asset}")
        bounded = float(min(max(float(raw_value), lower), upper))
        if bounded < raw_value:
            direction = "HIGH"
        elif bounded > raw_value:
            direction = "LOW"
        else:
            direction = "NONE"
        bounded_values[asset] = bounded
        rows.append(
            {
                "Asset": asset,
                "Asset_Class": asset_class,
                "mu_apt_raw": float(raw_value),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "mu_apt_scenario": bounded,
                "was_clipped": bool(direction != "NONE"),
                "clip_direction": direction,
                "scenario": scenario_name,
            }
        )
    return pd.Series(bounded_values, name=f"mu_apt_{scenario_name}"), pd.DataFrame(rows)


def build_scenario_raw_expected_returns(
    expected: pd.DataFrame,
    scenario_name: str,
    factor_assumptions: dict[str, dict[str, float]] | None = None,
    class_adjustments: dict[str, dict[str, float]] | None = None,
) -> pd.Series:
    """Build raw APT expected returns for one scenario before class bounds.

    The central scenario keeps the current raw APT estimate. Prudent and
    optimistic scenarios reuse the same betas and base carry, then apply
    documented factor and class assumptions. This keeps the model structure
    unchanged while making sensitivity tests economically visible.
    """

    factor_assumptions = factor_assumptions or APT_SCENARIO_FACTOR_ASSUMPTIONS
    class_adjustments = class_adjustments or APT_SCENARIO_CLASS_ADJUSTMENTS
    base = expected.set_index("asset_id")
    if scenario_name == "central":
        return base["mu_apt_raw"].astype(float).rename("mu_apt_raw_central")

    required = ["base_return_annualized", "alpha_component_annualized", "asset_type"]
    beta_cols = {
        "mkt_excess_tunindex20": "beta_MKT",
        "delta_zc_5y": "beta_RATE",
        "delta_credit_spread_5y": "beta_CREDIT",
        "delta_inflation_yoy": "beta_INF",
    }
    if any(col not in base.columns for col in required + list(beta_cols.values())):
        return base["mu_apt_raw"].astype(float).rename(f"mu_apt_raw_{scenario_name}")

    assumptions = factor_assumptions[scenario_name]
    factor_component = pd.Series(0.0, index=base.index)
    for factor, beta_col in beta_cols.items():
        factor_component = factor_component + pd.to_numeric(base[beta_col], errors="coerce").fillna(0.0) * float(assumptions.get(factor, 0.0))

    asset_type = base["asset_type"].astype(str)
    class_shift = asset_type.map(class_adjustments.get(scenario_name, {})).fillna(0.0).astype(float)
    vol = pd.to_numeric(base.get("historical_volatility_annualized", 0.0), errors="coerce").fillna(0.0)
    volatility_penalty = pd.Series(0.0, index=base.index)
    if scenario_name == "prudent":
        volatility_penalty = np.minimum(np.maximum(vol - 0.06, 0.0) * 0.20, 0.020)

    scenario_raw = (
        pd.to_numeric(base["base_return_annualized"], errors="coerce").fillna(0.0)
        + pd.to_numeric(base["alpha_component_annualized"], errors="coerce").fillna(0.0)
        + factor_component
        + class_shift
        - volatility_penalty
    )
    return scenario_raw.astype(float).rename(f"mu_apt_raw_{scenario_name}")


def build_apt_scenarios(
    expected: pd.DataFrame,
    scenario_bounds: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> dict[str, object]:
    """Build prudent, central and optimistic APT expected-return scenarios."""

    scenario_bounds = scenario_bounds or APT_SCENARIO_BOUNDS
    mu_raw = expected.set_index("asset_id")["mu_apt_raw"].astype(float)
    asset_classes = expected.set_index("asset_id")["asset_class"].astype(str)
    scenarios: dict[str, pd.Series] = {}
    audits = []
    raw_scenarios: dict[str, pd.Series] = {}
    for scenario_name, bounds in scenario_bounds.items():
        scenario_raw = build_scenario_raw_expected_returns(expected, scenario_name).reindex(mu_raw.index)
        raw_scenarios[scenario_name] = scenario_raw
        series, audit = apply_class_bounds(scenario_raw, asset_classes, bounds, scenario_name)
        scenarios[scenario_name] = series.reindex(mu_raw.index)
        audits.append(audit)
    if {"prudent", "central", "optimistic"}.issubset(scenarios):
        central = scenarios["central"].copy()
        scenarios["prudent"] = pd.Series(
            {asset: float(min(scenarios["prudent"].loc[asset], central.loc[asset])) for asset in central.index},
            name="mu_apt_prudent",
        )
        scenarios["optimistic"] = pd.Series(
            {asset: float(max(scenarios["optimistic"].loc[asset], central.loc[asset])) for asset in central.index},
            name="mu_apt_optimistic",
        )
    audit_all = pd.concat(audits, ignore_index=True)
    weights = expected.set_index("asset_id")["current_weight_optimisable"].astype(float).reindex(mu_raw.index)
    class_by_asset = asset_classes.reindex(mu_raw.index).map(lambda value: class_name_from_asset_type("", value))
    summary_rows = []
    for scenario_name, series in scenarios.items():
        audit = audit_all.loc[audit_all["scenario"].eq(scenario_name)]
        class_means = pd.DataFrame({"asset_class": class_by_asset, "mu": series}).groupby("asset_class")["mu"].mean()
        summary_rows.append(
            {
                "Scenario": scenario_name,
                "Mean_Return": float(series.mean()),
                "Median_Return": float(series.median()),
                "Min_Return": float(series.min()),
                "Max_Return": float(series.max()),
                "Std_Return": float(series.std(ddof=1)),
                "Equity_Mean": float(class_means.get("Actions cotées", np.nan)),
                "Sovereign_Mean": float(class_means.get("Titres de l'État", np.nan)),
                "Corporate_Bond_Mean": float(class_means.get("Emprunts obligataires", np.nan)),
                "Mean_expected_return": float(series.mean()),
                "Median_expected_return": float(series.median()),
                "Min_expected_return": float(series.min()),
                "Max_expected_return": float(series.max()),
                "Number_of_assets": int(series.notna().sum()),
                "Number_clipped_low": int(audit["clip_direction"].eq("LOW").sum()),
                "Number_clipped_high": int(audit["clip_direction"].eq("HIGH").sum()),
                "Portfolio_expected_return_current_weights": float((weights * series).sum()),
            }
        )
    return {
        "mu_apt_raw": mu_raw,
        "apt_scenarios_raw": raw_scenarios,
        "apt_scenarios": scenarios,
        "apt_scenarios_audit": audit_all,
        "apt_scenarios_summary": pd.DataFrame(summary_rows),
        "apt_optimistic_factor_treatment": APT_OPTIMISTIC_FACTOR_TREATMENT.copy(),
    }


def estimate_weekly_apt(
    daily_returns: pd.DataFrame,
    portfolio_optimisable: pd.DataFrame,
    asset_metrics: pd.DataFrame,
    factors_weekly: pd.DataFrame,
    params: APTParameters | None = None,
) -> dict[str, pd.DataFrame]:
    """Estimate OLS-HAC APT, expected returns and factor covariance."""

    params = params or APTParameters()
    weekly_returns = weekly_compounded_returns(daily_returns)
    factors = factors_weekly.copy()
    factors["week_date"] = pd.to_datetime(factors["week_date"])
    factor_index = factors.set_index("week_date")
    available_factors = detect_available_apt_factors(factors)
    factor_diagnostics = build_apt_factor_diagnostics(factors, weekly_returns)
    rf = factor_index["rf_weekly"]
    x_core = factor_index[APT_FACTORS_CORE]
    all_weeks = weekly_returns.index.union(factor_index.index)
    aligned_factor_rows = factor_index.reindex(all_weeks)
    factor_missing = aligned_factor_rows[APT_FACTORS_CORE].isna().sum().to_dict()
    meta = portfolio_optimisable.set_index("asset_id")
    hist_map = asset_metrics.set_index("asset_id")["annualized_return_geometric"].to_dict()
    vol_hist_map = asset_metrics.set_index("asset_id")["annualized_volatility"].to_dict()
    rf_annual = float(factors["tmm_annual_decimal"].dropna().mean())
    assumptions_rows = [
        {
            "factor": "mkt_excess_tunindex20",
            "expected_periodic_value": (1.0 + params.expected_mkt_premium_annual) ** (1.0 / params.periods_per_year) - 1.0,
            "expected_annualized_value": params.expected_mkt_premium_annual,
            "source": "PRUDENTIAL_PARAMETER",
            "justification": "Prime actions prudente paramétrable ; la performance brute 2025 du TUNINDEX20 n'est pas utilisée comme prévision.",
        },
        {
            "factor": "delta_zc_5y",
            "expected_periodic_value": 0.0,
            "expected_annualized_value": 0.0,
            "source": "CENTRAL_NO_DIRECTIONAL_RATE_VIEW",
            "justification": "Scénario central sans prévision directionnelle sur le niveau des taux souverains 5 ans.",
        },
        {
            "factor": "delta_credit_spread_5y",
            "expected_periodic_value": 0.0,
            "expected_annualized_value": 0.0,
            "source": "CENTRAL_NO_DIRECTIONAL_CREDIT_VIEW",
            "justification": "Scénario central sans élargissement ou resserrement attendu du spread corporate.",
        },
        {
            "factor": "delta_inflation_yoy",
            "expected_periodic_value": 0.0,
            "expected_annualized_value": 0.0,
            "source": "CENTRAL_NO_DIRECTIONAL_INFLATION_VIEW",
            "justification": "Scénario central sans variation attendue de l'inflation en glissement annuel.",
        },
    ]
    assumptions = pd.DataFrame(assumptions_rows)
    factor_expected_annual = assumptions.set_index("factor")["expected_annualized_value"].to_dict()
    betas_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    expected_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    residual_variance_weekly: dict[str, float] = {}

    for asset_id in weekly_returns.columns:
        asset = meta.loc[asset_id]
        reg = pd.DataFrame({"asset_return": weekly_returns[asset_id]}).join(factor_index[["rf_weekly"] + APT_FACTORS_CORE], how="left")
        reg["excess_return"] = reg["asset_return"] - reg["rf_weekly"]
        reg_clean = reg.dropna(subset=["excess_return"] + APT_FACTORS_CORE)
        n_obs = int(len(reg_clean))
        coverage = n_obs / max(1, len(weekly_returns))
        obs_flag = "INSUFFICIENT_APT_OBSERVATIONS" if n_obs < params.min_obs_warning else ("LOW_APT_OBSERVATIONS" if n_obs < params.min_obs_ok else "OK")
        alignment_rows.append(
            {
                "asset_id": asset_id,
                "asset_name": asset.get("asset_name"),
                "total_weeks_available": int(len(weekly_returns)),
                "aligned_observations": n_obs,
                "coverage_ratio": coverage,
                "missing_factors": ";".join([f"{k}:{v}" for k, v in factor_missing.items()]),
                "observation_flag": obs_flag,
            }
        )
        beta = dict.fromkeys(APT_FACTORS_CORE, 0.0)
        alpha = np.nan
        alpha_shrunk = 0.0
        r2 = np.nan
        adj_r2 = np.nan
        resid_vol = np.nan
        resid_var_w = np.nan
        dw = np.nan
        jb_stat = np.nan
        jb_pvalue = np.nan
        bp_pvalue = np.nan
        tstats = dict.fromkeys(APT_FACTORS_CORE, np.nan)
        pvals = dict.fromkeys(APT_FACTORS_CORE, np.nan)
        regression_status = "INSUFFICIENT_APT_OBSERVATIONS"
        flags = [obs_flag] if obs_flag != "OK" else []
        if n_obs >= params.min_obs_warning:
            y = reg_clean["excess_return"].astype(float)
            x = sm.add_constant(reg_clean[APT_FACTORS_CORE].astype(float), has_constant="add")
            try:
                model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
                alpha = float(model.params.get("const", np.nan))
                alpha_annual = alpha * params.periods_per_year if pd.notna(alpha) else np.nan
                unstable_alpha = pd.notna(alpha_annual) and (alpha_annual > 0.20 or alpha_annual < -0.20)
                alpha_shrunk = 0.0 if unstable_alpha else alpha * params.alpha_shrinkage
                if unstable_alpha:
                    flags.append("UNSTABLE_ALPHA")
                for factor in APT_FACTORS_CORE:
                    beta[factor] = float(model.params.get(factor, 0.0))
                    tstats[factor] = float(model.tvalues.get(factor, np.nan))
                    pvals[factor] = float(model.pvalues.get(factor, np.nan))
                r2 = float(model.rsquared)
                adj_r2 = float(model.rsquared_adj)
                resid = pd.Series(model.resid, index=reg_clean.index)
                resid_var_w = float(resid.var(ddof=1))
                resid_vol = float(math.sqrt(max(resid_var_w, 0.0)) * math.sqrt(params.periods_per_year))
                dw = float(durbin_watson(resid))
                jb = jarque_bera(resid)
                jb_stat, jb_pvalue = float(jb.statistic), float(jb.pvalue)
                try:
                    bp_pvalue = float(het_breuschpagan(resid, x)[1])
                except Exception:
                    bp_pvalue = np.nan
                regression_status = "OK"
                if adj_r2 < 0:
                    flags.append("LOW_EXPLANATORY_POWER")
                if resid_vol <= 0 or not np.isfinite(resid_vol):
                    flags.append("INVALID_RESIDUAL_VOL")
                if max(abs(v) for v in beta.values()) > 10:
                    flags.append("EXTREME_FACTOR_LOADING")
                if obs_flag != "OK":
                    regression_status = "APT_WITH_WARNING"
                if any(f in flags for f in ["LOW_EXPLANATORY_POWER", "INVALID_RESIDUAL_VOL", "EXTREME_FACTOR_LOADING", "UNSTABLE_ALPHA"]):
                    regression_status = "APT_WITH_WARNING"
            except Exception as exc:
                flags.append(f"OLS_NUMERICAL_FAILURE:{type(exc).__name__}")
                regression_status = "OLS_FAILED"
        if not np.isfinite(resid_var_w):
            hist_var = float(weekly_returns[asset_id].dropna().var(ddof=1))
            resid_var_w = hist_var if np.isfinite(hist_var) and hist_var > 0 else 1e-8
            flags.append("RESIDUAL_VARIANCE_FALLBACK_HISTORICAL")
        residual_variance_weekly[asset_id] = resid_var_w
        asset_type = str(asset.get("asset_type"))
        base_annual = rf_annual
        base_source = "RF_BASE_EQUITY"
        latest_zc5 = float(factors["zc_5y"].dropna().iloc[-1]) if factors["zc_5y"].notna().any() else rf_annual
        latest_spread = float(factors["credit_spread_5y"].dropna().iloc[-1]) if factors["credit_spread_5y"].notna().any() else 0.0
        if asset_type == "government_bond":
            base_annual = latest_zc5
            base_source = "SOVEREIGN_5Y_CARRY_PIVOT"
        elif asset_type == "corporate_bond":
            base_annual = latest_zc5 + latest_spread
            base_source = "CORPORATE_5Y_ZC_PLUS_SPREAD_CARRY_PIVOT"
        factor_component = sum(beta[f] * factor_expected_annual.get(f, 0.0) for f in APT_FACTORS_CORE)
        raw_expected_annual = base_annual + alpha_shrunk * params.periods_per_year + factor_component
        expected_annual = raw_expected_annual
        lower, upper = select_bounds(asset_type, params)
        clipped = False
        if not np.isfinite(expected_annual):
            expected_annual = base_annual
            raw_expected_annual = base_annual
            flags.append("EXPECTED_RETURN_FALLBACK_BASE_CARRY")
        if expected_annual < lower or expected_annual > upper:
            expected_annual = float(np.minimum(np.maximum(expected_annual, lower), upper))
            clipped = True
            flags.append("MU_APT_CLIPPED_PRUDENTIAL_BOUND")
        model = "APT_WEEKLY_MACRO"
        if regression_status != "OK":
            model = "APT_WEEKLY_WITH_WARNING"
        if "INSUFFICIENT_APT_OBSERVATIONS" in flags:
            model = "CARRY_OR_RF_FALLBACK_WITH_APT_CONTEXT"
        quality_flag = ";".join(dict.fromkeys([f for f in flags if f and f != "OK"])) or "OK"
        betas_rows.append(
            {
                "asset_id": asset_id,
                "asset_name": asset.get("asset_name"),
                "short_name": short_name(asset.get("asset_name")),
                "asset_class": asset.get("asset_class_standardized"),
                "asset_type": asset_type,
                "sector": asset.get("sector"),
                "n_obs": n_obs,
                "alpha_weekly": alpha,
                "alpha_annual_shrunk": alpha_shrunk * params.periods_per_year,
                "beta_MKT": beta["mkt_excess_tunindex20"],
                "beta_RATE": beta["delta_zc_5y"],
                "beta_CREDIT": beta["delta_credit_spread_5y"],
                "beta_INF": beta["delta_inflation_yoy"],
                "tstat_MKT": tstats["mkt_excess_tunindex20"],
                "tstat_RATE": tstats["delta_zc_5y"],
                "tstat_CREDIT": tstats["delta_credit_spread_5y"],
                "tstat_INF": tstats["delta_inflation_yoy"],
                "pvalue_MKT": pvals["mkt_excess_tunindex20"],
                "pvalue_RATE": pvals["delta_zc_5y"],
                "pvalue_CREDIT": pvals["delta_credit_spread_5y"],
                "pvalue_INF": pvals["delta_inflation_yoy"],
                "r_squared": r2,
                "adj_r_squared": adj_r2,
                "residual_volatility_annual": resid_vol,
                "residual_variance_weekly": resid_var_w,
                "durbin_watson": dw,
                "jarque_bera_stat": jb_stat,
                "jarque_bera_pvalue": jb_pvalue,
                "breusch_pagan_pvalue": bp_pvalue,
                "regression_status": regression_status,
                "quality_flag": quality_flag,
            }
        )
        diag_rows.append(
            {
                "asset_id": asset_id,
                "asset_name": asset.get("asset_name"),
                "n_obs": n_obs,
                "r_squared": r2,
                "adj_r_squared": adj_r2,
                "max_abs_beta": max(abs(v) for v in beta.values()),
                "residual_volatility": resid_vol,
                "durbin_watson": dw,
                "jarque_bera_pvalue": jb_pvalue,
                "breusch_pagan_pvalue": bp_pvalue,
                "status": regression_status,
                "warning": quality_flag,
            }
        )
        expected_rows.append(
            {
                "asset_id": asset_id,
                "asset_name": asset.get("asset_name"),
                "short_name": short_name(asset.get("asset_name")),
                "asset_class": asset.get("asset_class_standardized"),
                "asset_type": asset_type,
                "sector": asset.get("sector"),
                "current_weight_optimisable": float(asset.get("optimisable_weight", np.nan)),
                "current_value": float(asset.get("market_value", np.nan)),
                "historical_return_annualized": hist_map.get(asset_id, np.nan),
                "historical_volatility_annualized": vol_hist_map.get(asset_id, np.nan),
                "rf_annual": rf_annual,
                "base_return_annualized": base_annual,
                "base_return_source": base_source,
                "alpha_component_annualized": alpha_shrunk * params.periods_per_year,
                "factor_component_annualized": factor_component,
                "beta_MKT": beta["mkt_excess_tunindex20"],
                "beta_RATE": beta["delta_zc_5y"],
                "beta_CREDIT": beta["delta_credit_spread_5y"],
                "beta_INF": beta["delta_inflation_yoy"],
                "mu_apt_raw": raw_expected_annual,
                "expected_return_annualized_apt_raw": raw_expected_annual,
                "expected_return_annualized_apt": raw_expected_annual,
                "expected_return_annualized_final": expected_annual,
                "expected_return_model": model,
                "model_status": regression_status,
                "prudential_bound_applied": clipped,
                "central_lower_bound": lower,
                "central_upper_bound": upper,
                "quality_flag": quality_flag,
            }
        )
    betas = pd.DataFrame(betas_rows)
    diagnostics = pd.DataFrame(diag_rows)
    expected = pd.DataFrame(expected_rows)
    scenario_results = build_apt_scenarios(expected)
    scenarios = scenario_results["apt_scenarios"]
    scenario_audit = scenario_results["apt_scenarios_audit"]
    scenario_summary = scenario_results["apt_scenarios_summary"]
    optimistic_factor_treatment = scenario_results["apt_optimistic_factor_treatment"]
    expected["mu_apt_prudent"] = expected["asset_id"].map(scenarios["prudent"])
    expected["mu_apt_central"] = expected["asset_id"].map(scenarios["central"])
    expected["mu_apt_optimistic"] = expected["asset_id"].map(scenarios["optimistic"])
    expected["mu_expected_reference"] = expected["mu_apt_central"]
    expected["expected_return_annualized_final"] = expected["mu_apt_central"]
    expected["prudential_bound_applied"] = ~np.isclose(expected["mu_apt_raw"], expected["mu_apt_central"], atol=1e-12)
    expected["bound_saturated"] = expected["asset_id"].map(
        bound_saturation_by_asset(scenario_audit, expected["asset_id"])
    ).fillna("NONE")
    bounds_by_asset_class = build_apt_bounds_by_asset_class()
    scenario_differentiation_check = build_scenario_differentiation_check(expected, scenario_summary)
    alignment = pd.DataFrame(alignment_rows)
    factor_corr = factor_index[APT_FACTORS_CORE].corr()
    vif = compute_vif(factors)
    b = betas.set_index("asset_id")[["beta_MKT", "beta_RATE", "beta_CREDIT", "beta_INF"]].rename(
        columns={
            "beta_MKT": "mkt_excess_tunindex20",
            "beta_RATE": "delta_zc_5y",
            "beta_CREDIT": "delta_credit_spread_5y",
            "beta_INF": "delta_inflation_yoy",
        }
    )
    omega_weekly = factor_index[APT_FACTORS_CORE].dropna().cov()
    omega_annual = omega_weekly * params.periods_per_year
    sigma = b.to_numpy(float) @ omega_annual.reindex(index=APT_FACTORS_CORE, columns=APT_FACTORS_CORE).to_numpy(float) @ b.to_numpy(float).T
    psi = np.diag([residual_variance_weekly[a] * params.periods_per_year for a in b.index])
    sigma = sigma + psi
    sigma = (sigma + sigma.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sigma)
    psd_repaired = False
    if eigvals.min() < -1e-8:
        eigvals = np.maximum(eigvals, 1e-10)
        sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        sigma = (sigma + sigma.T) / 2.0
        psd_repaired = True
    sigma_apt = pd.DataFrame(sigma, index=b.index, columns=b.index)
    sys_var = np.diag(b.to_numpy(float) @ omega_annual.reindex(index=APT_FACTORS_CORE, columns=APT_FACTORS_CORE).to_numpy(float) @ b.to_numpy(float).T)
    spec_var = np.diag(psi)
    total_var = np.diag(sigma)
    risk_dec = expected[["asset_id", "asset_name", "short_name", "asset_class", "asset_type", "sector"]].copy()
    risk_dec["apt_total_variance_annual"] = total_var
    risk_dec["apt_systematic_variance_annual"] = sys_var
    risk_dec["apt_specific_variance_annual"] = spec_var
    risk_dec["systematic_share"] = np.where(total_var > 0, sys_var / total_var, np.nan)
    risk_dec["specific_share"] = np.where(total_var > 0, spec_var / total_var, np.nan)
    controls = [
        ("APT_Factors_Weekly_non_empty", not factors.empty, f"{len(factors)} semaines", "BLOCKING"),
        ("core_factors_exist", all(f in factors.columns for f in APT_FACTORS_CORE), ",".join(APT_FACTORS_CORE), "BLOCKING"),
        ("rf_weekly_reasonable", factors["rf_weekly"].dropna().between(0, 0.01).all(), "rf hebdomadaire entre 0% et 1%", "WARNING"),
        ("core_factors_not_all_nan", all(not factors[f].isna().all() for f in APT_FACTORS_CORE), "aucun facteur central entièrement NaN", "BLOCKING"),
        ("majority_assets_30_obs", (alignment["aligned_observations"] >= params.min_obs_warning).mean() >= 0.5, "majorité des actifs avec >=30 obs", "WARNING"),
        ("Sigma_APT_square", sigma_apt.shape[0] == sigma_apt.shape[1], str(sigma_apt.shape), "BLOCKING"),
        ("Sigma_APT_symmetric", np.allclose(sigma_apt.values, sigma_apt.values.T, atol=1e-10), "symétrie numérique", "BLOCKING"),
        ("Sigma_APT_asset_index", set(sigma_apt.index) == set(expected["asset_id"]), "mêmes actifs que mu_APT", "BLOCKING"),
        ("expected_returns_finite", np.isfinite(expected["expected_return_annualized_final"]).all(), "rendements attendus finis", "BLOCKING"),
        ("no_optimisable_asset_lost", len(expected) == len(portfolio_optimisable), f"{len(expected)}/{len(portfolio_optimisable)} actifs", "BLOCKING"),
        ("Sigma_APT_psd", np.linalg.eigvalsh(sigma_apt.values).min() >= -1e-8, f"psd_repaired={psd_repaired}", "BLOCKING"),
        ("APT_scenarios_no_nan", expected[["mu_apt_prudent", "mu_apt_central", "mu_apt_optimistic"]].notna().all().all(), "trois scénarios renseignés", "BLOCKING"),
        ("APT_scenarios_assets_aligned", set(expected["asset_id"]) == set(sigma_apt.index), "mêmes actifs dans mu scénarios et Sigma_APT", "BLOCKING"),
        ("APT_scenario_bounds_applied", scenario_audit["mu_apt_scenario"].between(scenario_audit["lower_bound"], scenario_audit["upper_bound"]).all(), "bornes par classe respectées", "BLOCKING"),
        ("APT_scenario_order_prudent_central_optimistic", ((expected["mu_apt_prudent"] <= expected["mu_apt_central"] + 1e-12) & (expected["mu_apt_central"] <= expected["mu_apt_optimistic"] + 1e-12)).all(), "prudent <= central <= optimiste", "WARNING"),
        ("APT_SCENARIO_DIFFERENTIATION_OK", not scenario_differentiation_check.loc[scenario_differentiation_check["Check"].eq("APT_SCENARIO_DIFFERENTIATION_OK"), "Status"].eq("FAILED").any(), "moins de 30% des actifs avec mu_prudent = mu_central", "WARNING"),
        ("APT_scenario_differentiation_sufficient", scenario_differentiation_check["Status"].eq("PASSED").all(), "différenciation prudent / central / optimiste contrôlée", "WARNING"),
        ("APT_optimistic_factor_treatment_documented", not optimistic_factor_treatment.empty and {"Factor", "Optimistic_Treatment", "Economic_Justification"}.issubset(optimistic_factor_treatment.columns), "traitement des facteurs optimistes documenté", "WARNING"),
        ("APT_central_reference_selected", np.allclose(expected["mu_apt_central"], expected["mu_expected_reference"], atol=1e-12), "mu_expected_reference = mu_apt_central", "BLOCKING"),
        ("APT_expected_returns_annual_frequency", True, "rendements attendus en fréquence annuelle", "BLOCKING"),
    ]
    final_control = pd.DataFrame(
        {
            "check_name": [c[0] for c in controls],
            "status": ["PASSED" if c[1] else "FAILED" for c in controls],
            "details": [c[2] for c in controls],
            "severity": [c[3] for c in controls],
        }
    )
    blocking_failed = final_control["status"].eq("FAILED") & final_control["severity"].eq("BLOCKING")
    warning_failed = final_control["status"].eq("FAILED") & final_control["severity"].eq("WARNING")
    final_status = "FAILED" if blocking_failed.any() else ("PASSED_WITH_WARNINGS" if warning_failed.any() or diagnostics["warning"].ne("OK").any() or vif["quality_flag"].ne("OK").any() else "PASSED")
    final_control.loc[len(final_control)] = {
        "check_name": "APT_final_status",
        "status": final_status,
        "details": f"warnings={int(diagnostics['warning'].ne('OK').sum()) + int(vif['quality_flag'].ne('OK').sum())}",
        "severity": "SUMMARY",
    }
    return {
        "Weekly_Returns": weekly_returns,
        "APT_Factors_Weekly": factors,
        "APT_Alignment_Audit": alignment,
        "APT_Betas": betas,
        "APT_Diagnostics": diagnostics,
        "APT_Factor_Correlation": factor_corr,
        "APT_VIF": vif,
        "APT_Available_Factors": available_factors,
        "APT_Factor_Diagnostics": factor_diagnostics,
        "APT_Expected_Returns": expected,
        "APT_Mu_Raw": expected[["asset_id", "asset_name", "asset_class", "mu_apt_raw"]],
        "APT_Mu_Prudent": expected[["asset_id", "asset_name", "asset_class", "mu_apt_prudent"]],
        "APT_Mu_Central": expected[["asset_id", "asset_name", "asset_class", "mu_apt_central"]],
        "APT_Mu_Optimistic": expected[["asset_id", "asset_name", "asset_class", "mu_apt_optimistic"]],
        "APT_Scenarios_Audit": scenario_audit,
        "APT_Scenarios_Summary": scenario_summary,
        "APT_Optimistic_Factor_Treatment": optimistic_factor_treatment,
        "APT_Bounds_By_Asset_Class": bounds_by_asset_class,
        "APT_Scenario_Differentiation_Check": scenario_differentiation_check,
        "apt_scenarios": scenarios,
        "apt_scenarios_audit": scenario_audit,
        "APT_Covariance_Matrix": sigma_apt,
        "APT_Risk_Decomposition": risk_dec,
        "APT_Assumptions": assumptions,
        "APT_Final_Control": final_control,
        "APT_Omega_Factor_Annual": omega_annual,
    }


def apt_references_table() -> pd.DataFrame:
    """Return the exact reference list documented in the notebook."""

    return pd.DataFrame(
        [
            ("Ross, S. A. (1976). The Arbitrage Theory of Capital Asset Pricing. Journal of Economic Theory, 13(3), 341–360.", "Fondement théorique de l’APT comme modèle multifactoriel."),
            ("Chen, N. F., Roll, R., & Ross, S. A. (1986). Economic Forces and the Stock Market. Journal of Business, 59(3), 383–403.", "Justification des facteurs macroéconomiques tels que les taux, l’inflation, la structure par terme et les primes de risque obligataires."),
            ("CFA Institute. Quantitative Investment Analysis, 4th Edition, Chapter 12: Using Multifactor Models.", "Cadre praticien des modèles multifactoriels, des sensibilités factorielles et de leur usage en construction de portefeuille."),
            ("CFA Institute. Quantitative Investment Analysis, 4th Edition, expected-return calibration.", "Justification des bornes par classe d'actifs : limiter le risque d'estimation lorsque les rendements historiques 2025 sont extrêmes ou peu liquides."),
            ("Cadre prudentiel assurance : gouvernance interne des placements, diversification et limites de concentration.", "Justification assurance des bornes equity/gov/corporate : elles ne sont pas réglementaires, mais encadrent les hypothèses APT pour une allocation institutionnelle défendable."),
            ("Kempthorne, P. MIT 18.S096, Lecture 15: Factor Models.", "Formulation matricielle des modèles factoriels et covariance factorielle Sigma = B Omega B’ + Psi."),
            ("Bourse des Valeurs Mobilières de Tunis. Guide des indices boursiers / méthodologie TUNINDEX20.", "Justification du TUNINDEX20 comme indice des valeurs les plus grandes et les plus liquides."),
            ("Institut National de la Statistique de Tunisie. Indice des Prix à la Consommation Familiale.", "Source officielle de l’inflation."),
            ("Banque Centrale de Tunisie. Indicateurs monétaires et financiers.", "Source de référence du TMM."),
            ("Tunisie Clearing / Tunisia Yield Curve. Courbes zéro coupon souveraines et courbes corporate sectorielles.", "Source des taux zéro coupon souverains et des spreads corporate."),
            ("Hammami, Y., & Jilani, F. (2011). Testing Factor Pricing Models in Tunisia: Macroeconomic Factors vs. Fundamental Factors. Review of Middle East Economics and Finance, 7(2), 1–22.", "Référence empirique spécifique au marché tunisien sur les modèles factoriels."),
            ("Dächert, K., Grindel, R., Leoff, E., Mahnkopp, J., Schirra, F., & Wenzel, J. (2022). Multicriteria asset allocation in practice. OR Spectrum, 44, 349–373.", "Justification du cadre allocation institutionnelle / assurance avec plusieurs objectifs, risque, rendement, solvabilité et distance au portefeuille actuel."),
        ],
        columns=["reference", "role_in_notebook"],
    )
