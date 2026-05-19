"""Valorisation des BTA et Emprunts Nationaux."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cashflows import cashflow_paid_between, discount_cashflows, generate_fixed_income_cashflows, infer_nominal
from .config import ANALYSIS_END_DATE, ANALYSIS_START_DATE, BTA_NOMINAL, EN_NOMINAL, PERIODS_PER_YEAR_DAILY
from .curves import interpolate_sovereign_rate, interpolate_zero_rate
from .loaders import slugify


def price_government_bond_weekly(asset_row: pd.Series, zc_curves: pd.DataFrame, valuation_dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    """Valorise un titre souverain chaque semaine avec cash-flows futurs strictement postérieurs."""

    asset_id = str(asset_row.get("asset_id", slugify(asset_row.get("asset_name", ""))))
    cashflows = generate_fixed_income_cashflows(asset_row)
    nominal = infer_nominal(asset_row, cashflows)
    coupon = asset_row.get("coupon_rate")
    maturity = pd.to_datetime(asset_row.get("maturity_date"), errors="coerce")
    missing_terms = cashflows.empty or pd.isna(nominal) or pd.isna(coupon) or pd.isna(maturity)
    rows: list[dict[str, object]] = []
    previous_date: pd.Timestamp | None = None
    previous_price: float | None = None
    for date in pd.to_datetime(valuation_dates):
        if date < pd.Timestamp(ANALYSIS_START_DATE) or date > pd.Timestamp(ANALYSIS_END_DATE):
            continue
        flag = "MODEL_BASED_VALUATION"
        price = duration = dv01 = np.nan
        if missing_terms:
            future = pd.DataFrame()
            flag = "MISSING_BOND_TERMS"
        else:
            future = cashflows[cashflows["cashflow_date"] > date].copy()
        if not missing_terms and future.empty:
            flag = "DATA_MISSING"
        elif not missing_terms:
            future["tau"] = (future["cashflow_date"] - date).dt.days / 365.0
            pairs = [interpolate_zero_rate(zc_curves, date, tau) for tau in future["tau"]]
            rates = pd.Series([p[0] for p in pairs], index=future.index)
            flags = {p[1] for p in pairs if p[1] != "OK"}
            if flags:
                flag = ";".join(sorted(flags))
            else:
                price, duration = discount_cashflows(future, date, rates)
                dv01 = price * duration * 0.0001 if pd.notna(price) and pd.notna(duration) else np.nan
        paid = cashflow_paid_between(cashflows, previous_date, date)
        weekly_return = np.nan
        if previous_price is not None and pd.notna(previous_price) and previous_price != 0 and pd.notna(price):
            weekly_return = (price + paid - previous_price) / previous_price
        rows.append(
            {
                "date": date,
                "asset_id": asset_id,
                "asset_name": asset_row.get("asset_name"),
                "dirty_price_model": price,
                "cashflow_paid": paid,
                "weekly_return": weekly_return,
                "duration_modified": duration,
                "dv01": dv01,
                "pricing_flag": flag,
            }
        )
        previous_date = date
        if pd.notna(price):
            previous_price = price
    return pd.DataFrame(rows)


def build_bond_return_matrix(pricing_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Transforme les valorisations en matrice de rendements."""

    frames = [f for f in pricing_frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return pd.DataFrame()
    pricing = pd.concat(frames, ignore_index=True)
    return pricing.pivot_table(index="date", columns="asset_id", values="weekly_return", aggfunc="last").sort_index()


def _state_nominal(asset_name: object) -> float:
    name = str(asset_name).upper()
    if "BTA" in name:
        return float(BTA_NOMINAL)
    if "EMPRUNT NATIONAL" in name:
        return float(EN_NOMINAL)
    return np.nan


def price_government_bond_daily(
    asset_row: pd.Series,
    sovereign_curves: pd.DataFrame,
    valuation_dates: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Valorise BTA/EN et calcule le rendement avec cash-flow paye.

    La valeur est obtenue par DCF sur flux futurs strictement posterieurs a la
    date de valorisation. Le rendement ajoute le cash-flow paye depuis la date
    precedente afin qu'une date de coupon ne soit pas interpretee comme une
    baisse de prix economique.
    """

    asset_id = str(asset_row.get("asset_id", slugify(asset_row.get("asset_name", ""))))
    cashflows = generate_fixed_income_cashflows(asset_row)
    coupon = asset_row.get("coupon_rate")
    maturity = pd.to_datetime(asset_row.get("maturity_date"), errors="coerce")
    nominal = _state_nominal(asset_row.get("asset_name"))
    missing_terms = cashflows.empty or pd.isna(coupon) or pd.isna(maturity) or pd.isna(nominal)
    rows: list[dict[str, object]] = []
    previous_date: pd.Timestamp | None = None
    previous_price: float | None = None
    for date in pd.to_datetime(valuation_dates):
        if date < pd.Timestamp(ANALYSIS_START_DATE) or date > pd.Timestamp(ANALYSIS_END_DATE):
            continue
        price = duration = np.nan
        flag = "MODEL_BASED_VALUATION;TOTAL_RETURN_WITH_CASHFLOW"
        if missing_terms:
            flag = "MISSING_BOND_TERMS"
        else:
            future = cashflows[cashflows["cashflow_date"] > date].copy()
            if future.empty:
                flag = "DATA_MISSING"
            else:
                future["tau"] = (future["cashflow_date"] - date).dt.days / 365.0
                pairs = [interpolate_sovereign_rate(sovereign_curves, date, tau) for tau in future["tau"]]
                rates = pd.Series([p[0] for p in pairs], index=future.index)
                flags = {p[1] for p in pairs if p[1] != "OK"}
                if flags:
                    flag = ";".join(sorted(flags))
                else:
                    price, duration = discount_cashflows(future, date, rates)
        paid = cashflow_paid_between(cashflows, previous_date, date)
        if paid and pd.notna(paid):
            flag = f"{flag};CASHFLOW_INCLUDED_IN_TOTAL_RETURN"
        daily_return = np.nan
        if previous_price is not None and pd.notna(previous_price) and previous_price != 0 and pd.notna(price):
            paid_amount = float(paid) if pd.notna(paid) else 0.0
            daily_return = (price + paid_amount - previous_price) / previous_price
        rows.append(
            {
                "date": date,
                "asset_id": asset_id,
                "asset_name": asset_row.get("asset_name"),
                "asset_class": "government_bond",
                "dirty_price_model": price,
                "cashflow_paid": paid,
                "daily_return": daily_return,
                "duration_modified": duration,
                "pricing_flag": flag,
            }
        )
        previous_date = date
        if pd.notna(price):
            previous_price = price
    return pd.DataFrame(rows)


def build_daily_bond_return_matrix(pricing_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Matrice date x actif des rendements journaliers obligataires."""

    frames = [f for f in pricing_frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return pd.DataFrame()
    pricing = pd.concat(frames, ignore_index=True)
    return pricing.pivot_table(index="date", columns="asset_id", values="daily_return", aggfunc="last").sort_index()


def build_government_bond_check(portfolio_df: pd.DataFrame, pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Table concise de contrôle des titres de l'État."""

    rows = []
    for _, asset in portfolio_df.loc[portfolio_df["asset_type"].eq("government_bond")].iterrows():
        asset_id = str(asset["asset_id"])
        one = pricing_df.loc[pricing_df["asset_id"].eq(asset_id)].sort_values("date")
        last = one.dropna(subset=["dirty_price_model"]).tail(1)
        price = float(last["dirty_price_model"].iloc[0]) if not last.empty else np.nan
        duration = float(last["duration_modified"].iloc[0]) if not last.empty else np.nan
        nominal = _state_nominal(asset["asset_name"])
        quantity = float(asset.get("quantity") or 0.0)
        model_value = price * quantity if pd.notna(price) else np.nan
        portfolio_value = float(asset.get("market_value") or np.nan)
        relative_gap = (model_value - portfolio_value) / portfolio_value if pd.notna(model_value) and portfolio_value else np.nan
        rows.append(
            {
                "asset_id": asset_id,
                "asset_name": asset["asset_name"],
                "coupon": asset.get("coupon_rate"),
                "maturity_date": asset.get("maturity_date"),
                "nominal": nominal,
                "model_price_last_date": price,
                "portfolio_value": portfolio_value,
                "model_value": model_value,
                "relative_gap": relative_gap,
                "duration_modified": duration,
                "pricing_flag": str(one["pricing_flag"].dropna().iloc[-1]) if not one.empty else "DATA_MISSING",
            }
        )
    return pd.DataFrame(rows)


def build_government_bond_reconciliation(portfolio_df: pd.DataFrame, pricing_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Réconciliation modèle vs valeur portefeuille au 31/12/2025."""

    rows: list[dict[str, object]] = []
    for _, asset in portfolio_df.loc[portfolio_df["asset_type"].eq("government_bond")].iterrows():
        asset_id = str(asset["asset_id"])
        one = pricing_df.loc[pricing_df["asset_id"].eq(asset_id)].sort_values("date")
        last = one.loc[one["date"] <= pd.Timestamp(ANALYSIS_END_DATE)].tail(1)
        model_price = float(last["dirty_price_model"].iloc[0]) if not last.empty else np.nan
        duration = float(last["duration_modified"].iloc[0]) if not last.empty else np.nan
        nominal = float(asset.get("nominal")) if pd.notna(asset.get("nominal")) else np.nan
        quantity = float(asset.get("quantity") or 0.0)
        model_value = model_price * quantity if pd.notna(model_price) else np.nan
        portfolio_value = float(asset.get("market_value") or np.nan)
        abs_gap = model_value - portfolio_value if pd.notna(model_value) and pd.notna(portfolio_value) else np.nan
        rel_gap = abs_gap / portfolio_value if pd.notna(abs_gap) and portfolio_value else np.nan
        returns = returns_df[asset_id].dropna() if asset_id in returns_df else pd.Series(dtype=float)
        flag = str(last["pricing_flag"].iloc[0]) if not last.empty else "DATA_MISSING"
        if pd.notna(rel_gap) and abs(rel_gap) > 0.02:
            flag = f"{flag};RECONCILIATION_GAP_HIGH"
        rows.append(
            {
                "asset_id": asset_id,
                "asset_name": asset["asset_name"],
                "coupon": asset.get("coupon_rate"),
                "maturity_date": asset.get("maturity_date"),
                "nominal": nominal,
                "model_price_last_date": model_price,
                "portfolio_value": portfolio_value,
                "model_value": model_value,
                "relative_gap": rel_gap,
                "duration_modified": duration,
                "pricing_flag": flag,
            }
        )
    return pd.DataFrame(rows)
