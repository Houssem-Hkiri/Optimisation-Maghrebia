"""Valorisation des obligations corporate observées ou proxifiées."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cashflows import cashflow_paid_between, discount_cashflows, generate_fixed_income_cashflows, infer_nominal
from .config import ANALYSIS_END_DATE, ANALYSIS_START_DATE
from .corporate_actions import CORPORATE_BOND_METADATA_OVERRIDES, CORPORATE_BOND_SPREAD_OVERRIDES
from .curves import get_sector_spread, interpolate_sovereign_rate, interpolate_zero_rate, map_corporate_sector
from .loaders import coerce_rate_to_decimal, filter_date_window, standardize_columns, validate_required_columns


def detect_dirty_price_scale(prices: pd.Series) -> str:
    """Détecte si le dirty price est exprimé en base 1 ou en base 100."""

    clean = pd.to_numeric(prices, errors="coerce").dropna()
    if clean.empty:
        return "UNKNOWN"
    median = float(clean.median())
    if 0.8 <= median <= 1.3:
        return "BASE_1"
    if 80.0 <= median <= 130.0:
        return "BASE_100"
    return "UNKNOWN"


def scale_dirty_price_to_unit(dirty_price: float, outstanding_principal_unit: float, dirty_price_scale: str) -> float:
    """Convertit un dirty price observé en prix unitaire monétaire."""

    if pd.isna(dirty_price) or pd.isna(outstanding_principal_unit):
        return np.nan
    if dirty_price_scale == "BASE_1":
        return float(dirty_price) * float(outstanding_principal_unit)
    if dirty_price_scale == "BASE_100":
        return float(dirty_price) / 100.0 * float(outstanding_principal_unit)
    return np.nan


def get_outstanding_principal_unit(asset_row: pd.Series, valuation_date: pd.Timestamp) -> tuple[float, str]:
    """Détermine le capital restant unitaire sans nominal arbitraire."""

    explicit_cols = [
        "capital_restant_unitaire",
        "nominal_restant",
        "outstanding_principal",
        "capital_open_unit",
        "nominal_actuel",
    ]
    for col in explicit_cols:
        value = asset_row.get(col)
        if pd.notna(value) and float(value) > 0:
            return float(value), "OK"

    schedule = asset_row.get("cashflow_schedule")
    if isinstance(schedule, pd.DataFrame) and not schedule.empty and {"cashflow_date", "capital_begin"}.issubset(schedule.columns):
        sched = schedule.copy()
        sched["cashflow_date"] = pd.to_datetime(sched["cashflow_date"], errors="coerce")
        sched["capital_begin"] = pd.to_numeric(sched["capital_begin"], errors="coerce")
        future = sched.loc[(sched["cashflow_date"] > pd.Timestamp(valuation_date)) & sched["capital_begin"].gt(0)].sort_values("cashflow_date")
        if not future.empty:
            return float(future["capital_begin"].iloc[0]), "OK"

    return np.nan, "MISSING_OUTSTANDING_PRINCIPAL"


def _missing_bond_fields(asset_row: pd.Series, include_isin: bool = False) -> list[str]:
    missing: list[str] = []
    if include_isin:
        isin = asset_row.get("isin")
        if pd.isna(isin) or str(isin).strip() in {"", "-", "nan", "None"}:
            missing.append("isin")
    if pd.isna(asset_row.get("coupon_rate")):
        missing.append("coupon")
    if pd.isna(pd.to_datetime(asset_row.get("maturity_date"), errors="coerce")):
        missing.append("maturity_date")
    return missing


def load_corporate_curves(path: Path, start: str = ANALYSIS_START_DATE, end: str = ANALYSIS_END_DATE) -> pd.DataFrame:
    """Charge les courbes sectorielles corporate."""

    if not path.exists():
        raise FileNotFoundError(path)
    df = standardize_columns(pd.read_excel(path, sheet_name="corporate_sector_curves"))
    df = df.rename(columns={"requested_date": "date", "corporate_yield_decimal": "yield_or_rate", "spread_decimal": "spread"})
    validate_required_columns(df, ["date", "sector", "bond_type", "maturity_years", "spread"], "Courbes corporate")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sector"] = df["sector"].astype(str).str.upper().str.strip()
    df["bond_type"] = df["bond_type"].astype(str).str.upper().str.strip()
    df["maturity_years"] = pd.to_numeric(df["maturity_years"], errors="coerce")
    df["spread"] = coerce_rate_to_decimal(df["spread"])
    if "yield_or_rate" in df:
        df["yield_or_rate"] = coerce_rate_to_decimal(df["yield_or_rate"])
    return filter_date_window(df.dropna(subset=["date", "sector", "bond_type", "maturity_years", "spread"]), "date", start, end)


def load_existing_corporate_prices(path: Path, start: str = ANALYSIS_START_DATE, end: str = ANALYSIS_END_DATE) -> pd.DataFrame:
    """Charge les dirty prices observés depuis corporate_existing_2025 si disponible."""

    try:
        df = standardize_columns(pd.read_excel(path, sheet_name="corporate_existing_2025"))
    except ValueError:
        return pd.DataFrame()
    df = df.rename(columns={"requested_date": "date"})
    if not {"date", "isin", "dirty_price"}.issubset(df.columns):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["isin"] = df["isin"].astype(str).str.strip()
    df["dirty_price"] = pd.to_numeric(df["dirty_price"], errors="coerce")
    if "sector" in df:
        df["sector"] = df["sector"].astype(str).str.upper().str.strip()
    return filter_date_window(df.dropna(subset=["date", "isin", "dirty_price"]), "date", start, end)


def estimate_sector_spreads(corporate_curves: pd.DataFrame) -> pd.DataFrame:
    """Agrège les spreads par date, secteur, type et maturité."""

    return (
        corporate_curves.dropna(subset=["spread"])
        .groupby(["date", "sector", "bond_type", "maturity_years"], as_index=False)["spread"]
        .median()
        .sort_values(["date", "sector", "bond_type", "maturity_years"])
    )


def _interpolate_spread(
    spreads: pd.DataFrame,
    valuation_date: pd.Timestamp,
    sector: str,
    bond_type: str,
    tau_years: float,
) -> tuple[float, str]:
    sector = str(sector).upper().strip()
    bond_type = str(bond_type).upper().strip()
    one = spreads[(spreads["date"].eq(valuation_date)) & (spreads["sector"].eq(sector)) & (spreads["bond_type"].eq(bond_type))]
    flag = "OK"
    if one.empty:
        one = spreads[(spreads["date"].eq(valuation_date)) & (spreads["sector"].eq(sector))]
    if one.empty:
        past = spreads[(spreads["date"] < valuation_date) & (spreads["sector"].eq(sector)) & (spreads["bond_type"].eq(bond_type))]
        if past.empty:
            past = spreads[(spreads["date"] < valuation_date) & (spreads["sector"].eq(sector))]
        if past.empty:
            return np.nan, "SPREAD_SECTOR_MISSING"
        one = past[past["date"].eq(past["date"].max())]
        flag = "SPREAD_LAST_AVAILABLE_USED"
    maturities = one["maturity_years"].to_numpy(float)
    values = one["spread"].to_numpy(float)
    if len(maturities) == 0:
        return np.nan, "SPREAD_SECTOR_MISSING"
    order = np.argsort(maturities)
    maturities, values = maturities[order], values[order]
    if tau_years < maturities.min() or tau_years > maturities.max():
        return np.nan, "SPREAD_SECTOR_MISSING"
    return float(np.interp(tau_years, maturities, values)), flag


def _observed_price_series(asset_row: pd.Series, existing_prices: pd.DataFrame, nominal: float) -> pd.Series:
    isin = str(asset_row.get("isin") or asset_row.get("asset_id") or "")
    if existing_prices.empty or isin in {"", "nan", "None", "-"}:
        return pd.Series(dtype=float)
    obs = existing_prices.loc[existing_prices["isin"].eq(isin)].sort_values("date")
    if obs.empty:
        return pd.Series(dtype=float)
    prices = obs.set_index("date")["dirty_price"].astype(float)
    if prices.median(skipna=True) < 10:
        prices = prices * nominal
    return prices


def price_corporate_bond_weekly(
    asset_row: pd.Series,
    zc_curves: pd.DataFrame,
    sector_spreads: pd.DataFrame,
    valuation_dates: pd.Series | pd.DatetimeIndex,
    existing_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Valorise une obligation corporate selon dirty price observé puis proxy modèle."""

    asset_id = str(asset_row.get("asset_id"))
    cashflows = generate_fixed_income_cashflows(asset_row)
    nominal = infer_nominal(asset_row, cashflows)
    coupon = asset_row.get("coupon_rate")
    maturity = pd.to_datetime(asset_row.get("maturity_date"), errors="coerce")
    missing_terms = cashflows.empty or pd.isna(nominal) or pd.isna(coupon) or pd.isna(maturity)
    observed = _observed_price_series(asset_row, existing_prices if existing_prices is not None else pd.DataFrame(), nominal if pd.notna(nominal) else 1.0)
    sector = str(asset_row.get("sector", "OTHER")).upper()
    bond_type = str(asset_row.get("bond_type", "ORDINARY")).upper()
    pro_forma = pd.isna(asset_row.get("isin")) or str(asset_row.get("isin")).strip() in {"", "-", "nan"}
    rows: list[dict[str, object]] = []
    previous_date: pd.Timestamp | None = None
    previous_price: float | None = None
    for date in pd.to_datetime(valuation_dates):
        if date < pd.Timestamp(ANALYSIS_START_DATE) or date > pd.Timestamp(ANALYSIS_END_DATE):
            continue
        spread_used = np.nan
        duration = dv01 = np.nan
        if missing_terms:
            obs_asof = pd.Series(dtype=float)
        elif not observed.empty:
            obs_asof = observed.loc[observed.index <= date]
        else:
            obs_asof = pd.Series(dtype=float)
        if not obs_asof.empty and obs_asof.index.max() == date:
            price = float(obs_asof.iloc[-1])
            flag = "CORPORATE_DIRTY_PRICE_OBSERVED"
            duration = np.nan
        else:
            future = cashflows[cashflows["cashflow_date"] > date].copy() if not missing_terms else pd.DataFrame()
            flag = "MODEL_BASED_CORPORATE_VALUATION"
            price = np.nan
            if missing_terms:
                flag = "MISSING_BOND_TERMS"
            if pro_forma:
                flag += ";MISSING_SECURITY_IDENTIFIER"
            if pro_forma and missing_terms:
                flag += ";MODEL_NOT_RELIABLE_WITHOUT_TERMS"
            if not missing_terms and future.empty:
                flag += ";DATA_MISSING"
            elif not missing_terms:
                future["tau"] = (future["cashflow_date"] - date).dt.days / 365.0
                zc_pairs = [interpolate_zero_rate(zc_curves, date, tau) for tau in future["tau"]]
                spread_pairs = [_interpolate_spread(sector_spreads, date, sector, bond_type, tau) for tau in future["tau"]]
                zc_rates = pd.Series([p[0] for p in zc_pairs], index=future.index)
                spreads = pd.Series([p[0] for p in spread_pairs], index=future.index)
                extra_flags = {p[1] for p in zc_pairs + spread_pairs if p[1] != "OK"}
                if extra_flags or zc_rates.isna().any() or spreads.isna().any():
                    flag += ";" + ";".join(sorted(extra_flags or {"SPREAD_SECTOR_MISSING"}))
                else:
                    total_rates = zc_rates + spreads
                    spread_used = float(spreads.mean())
                    price, duration = discount_cashflows(future, date, total_rates)
                    dv01 = price * duration * 0.0001 if pd.notna(price) and pd.notna(duration) else np.nan
        paid = cashflow_paid_between(cashflows, previous_date, date)
        weekly_return = np.nan
        if previous_price is not None and pd.notna(previous_price) and previous_price != 0 and pd.notna(price):
            weekly_return = (price + paid - previous_price) / previous_price
        rows.append(
            {
                "date": date,
                "asset_id": asset_id,
                "isin": asset_row.get("isin"),
                "sector": sector,
                "bond_type": bond_type,
                "dirty_price_model": price,
                "cashflow_paid": paid,
                "weekly_return": weekly_return,
                "spread_used": spread_used,
                "duration_modified": duration,
                "dv01": dv01,
                "pricing_flag": flag,
                "source_used": "LEGACY_OBSERVED_PRICE_RECON_ONLY" if flag == "CORPORATE_DIRTY_PRICE_OBSERVED" else "MODEL_PROXY",
            }
        )
        previous_date = date
        if pd.notna(price):
            previous_price = price
    return pd.DataFrame(rows)


def build_corporate_bond_reconciliation(portfolio_df: pd.DataFrame, pricing_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Réconciliation des obligations corporate."""

    rows: list[dict[str, object]] = []
    for _, asset in portfolio_df.loc[portfolio_df["asset_type"].eq("corporate_bond")].iterrows():
        asset_id = str(asset["asset_id"])
        one = pricing_df.loc[pricing_df["asset_id"].eq(asset_id)].sort_values("date")
        last = one.loc[one["date"] <= pd.Timestamp(ANALYSIS_END_DATE)].tail(1)
        model_price = float(last["dirty_price_model"].iloc[0]) if not last.empty else np.nan
        nominal = float(asset.get("nominal")) if pd.notna(asset.get("nominal")) else np.nan
        quantity = float(asset.get("quantity") or 0.0)
        model_value = model_price * quantity if pd.notna(model_price) else np.nan
        portfolio_value = float(asset.get("market_value") or np.nan)
        abs_gap = model_value - portfolio_value if pd.notna(model_value) and pd.notna(portfolio_value) else np.nan
        rel_gap = abs_gap / portfolio_value if pd.notna(abs_gap) and portfolio_value else np.nan
        returns = returns_df[asset_id].dropna() if asset_id in returns_df else pd.Series(dtype=float)
        flag = str(last["pricing_flag"].iloc[0]) if not last.empty else "DATA_MISSING"
        if pd.notna(rel_gap) and abs(rel_gap) > 0.02:
            flag += ";RECONCILIATION_GAP_HIGH"
        rows.append(
            {
                "asset_id": asset_id,
                "isin": asset.get("isin"),
                "sector": asset.get("sector"),
                "coupon": asset.get("coupon_rate"),
                "maturity_date": asset.get("maturity_date"),
                "nominal": nominal,
                "source_used": last["source_used"].iloc[0] if not last.empty else "MODEL_PROXY",
                "model_price_last_date": model_price,
                "portfolio_value": portfolio_value,
                "model_value": model_value,
                "relative_gap": rel_gap,
                "spread_used": last["spread_used"].iloc[0] if not last.empty else np.nan,
                "pricing_flag": flag,
            }
        )
    return pd.DataFrame(rows)


def price_corporate_bond_daily(
    asset_row: pd.Series,
    sovereign_curves: pd.DataFrame,
    sector_spreads: pd.DataFrame,
    valuation_dates: pd.Series | pd.DatetimeIndex,
    corporate_clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """Valorise une obligation corporate et calcule un rendement avec cash-flow.

    Les prix observes sont utilises en priorite. A defaut, le prix proxy DCF
    utilise la courbe souveraine et le spread sectoriel, avec overrides
    individuels explicites si renseignes.
    """

    asset_id = str(asset_row.get("asset_id"))
    asset_name = str(asset_row.get("asset_name", asset_id))
    metadata_override = CORPORATE_BOND_METADATA_OVERRIDES.get(asset_id, {})
    isin = asset_row.get("isin")
    if (pd.isna(isin) or str(isin).strip() in {"", "-", "nan", "None"}) and metadata_override.get("isin"):
        isin = metadata_override.get("isin")
    isin = None if pd.isna(isin) or str(isin).strip() in {"", "-", "nan", "None"} else str(isin).strip()
    cashflows = generate_fixed_income_cashflows(asset_row)
    coupon = asset_row.get("coupon_rate")
    maturity = pd.to_datetime(asset_row.get("maturity_date"), errors="coerce")
    missing_terms = cashflows.empty or pd.isna(coupon) or pd.isna(maturity)
    sector = str(asset_row.get("sector", "OTHER")).upper().strip()
    observed = pd.DataFrame()
    raw_scale = asset_row.get("dirty_price_scale")
    if pd.isna(raw_scale) or str(raw_scale).strip() in {"", "-", "nan", "None"}:
        raw_scale = metadata_override.get("dirty_price_scale")
    configured_scale = str(raw_scale or "UNKNOWN").upper().strip()
    dirty_price_scale = configured_scale if configured_scale in {"BASE_1", "BASE_100"} else "UNKNOWN"
    if isin:
        observed = corporate_clean_df.loc[corporate_clean_df["isin"].eq(isin) & corporate_clean_df["dirty_price"].notna(), ["curve_date", "dirty_price", "duration_modified"]].copy()
        if dirty_price_scale == "UNKNOWN":
            dirty_price_scale = detect_dirty_price_scale(observed["dirty_price"]) if not observed.empty else "UNKNOWN"
    rows = []
    previous_price: float | None = None
    previous_date: pd.Timestamp | None = None
    for date in pd.to_datetime(valuation_dates):
        if date < pd.Timestamp(ANALYSIS_START_DATE) or date > pd.Timestamp(ANALYSIS_END_DATE):
            continue
        price = duration = spread_used = np.nan
        return_price = np.nan
        last_dirty_price = np.nan
        outstanding_unit, outstanding_flag = get_outstanding_principal_unit(asset_row, date)
        missing_fields = _missing_bond_fields(asset_row, include_isin=False)
        if outstanding_flag != "OK":
            missing_fields.append("outstanding_principal_unit")
        source_used = "MODEL_PROXY"
        if not observed.empty and date in set(observed["curve_date"]):
            obs = observed.loc[observed["curve_date"].eq(date)].tail(1)
            last_dirty_price = float(obs["dirty_price"].iloc[0])
            duration = float(obs["duration_modified"].iloc[0]) if pd.notna(obs["duration_modified"].iloc[0]) else np.nan
            price = scale_dirty_price_to_unit(last_dirty_price, outstanding_unit, dirty_price_scale)
            if dirty_price_scale == "BASE_1":
                return_price = last_dirty_price
            elif dirty_price_scale == "BASE_100":
                return_price = last_dirty_price / 100.0
            flag = "CORPORATE_DIRTY_PRICE_OBSERVED;TOTAL_RETURN_WITH_CASHFLOW"
            if dirty_price_scale == "UNKNOWN":
                flag += ";DIRTY_PRICE_SCALE_UNKNOWN"
                missing_fields.append("dirty_price_scale")
            if outstanding_flag != "OK":
                flag += f";{outstanding_flag}"
            source_used = "LEGACY_OBSERVED_PRICE_RECON_ONLY"
        elif not observed.empty:
            flag = "CORPORATE_DIRTY_PRICE_MISSING_ON_DATE;OBSERVED_SERIES_NOT_PROXY_FILLED"
            if outstanding_flag != "OK":
                flag += f";{outstanding_flag}"
            source_used = "LEGACY_OBSERVED_PRICE_RECON_ONLY"
        elif missing_terms:
            flag = "MISSING_BOND_TERMS;TOTAL_RETURN_WITH_CASHFLOW"
            if not isin:
                flag += ";MISSING_SECURITY_IDENTIFIER"
                missing_fields.append("isin")
            if outstanding_flag != "OK":
                flag += f";{outstanding_flag}"
        else:
            future = cashflows[cashflows["cashflow_date"] > date].copy()
            flag = "MODEL_BASED_CORPORATE_VALUATION;TOTAL_RETURN_WITH_CASHFLOW"
            if not isin:
                flag += ";MODEL_BASED_PROXY_CORPORATE_VALUATION;MISSING_SECURITY_IDENTIFIER"
                missing_fields.append("isin")
            if dirty_price_scale == "UNKNOWN":
                flag += ";PROXY_MISSING_FIELDS:dirty_price_scale"
                missing_fields.append("dirty_price_scale")
            if outstanding_flag != "OK":
                flag += f";{outstanding_flag}"
            if future.empty:
                flag += ";DATA_MISSING"
            else:
                future["tau"] = (future["cashflow_date"] - date).dt.days / 365.0
                zc_pairs = [interpolate_sovereign_rate(sovereign_curves, date, tau) for tau in future["tau"]]
                spread_override = (
                    CORPORATE_BOND_SPREAD_OVERRIDES.get(asset_id)
                    or (CORPORATE_BOND_SPREAD_OVERRIDES.get(isin) if isin else None)
                )
                if spread_override and spread_override.get("spread_decimal") is not None:
                    spread_value = float(spread_override["spread_decimal"])
                    spread_pairs = [(spread_value, "INDIVIDUAL_SPREAD_OVERRIDE") for _ in future["tau"]]
                else:
                    spread_pairs = [
                        get_sector_spread(
                            sector_spreads,
                            date,
                            sector,
                            tau,
                            allow_last_available=True,
                            allow_nearest_maturity=True,
                        )
                        for tau in future["tau"]
                    ]
                zc_rates = pd.Series([p[0] for p in zc_pairs], index=future.index)
                spreads = pd.Series([p[0] for p in spread_pairs], index=future.index)
                flags = {p[1] for p in zc_pairs + spread_pairs if p[1] != "OK"}
                non_blocking_flags = {"SPREAD_NEAREST_MATURITY_USED", "CORPORATE_CURVE_LAST_AVAILABLE_USED"}
                blocking_flags = {
                    f
                    for flag_text in flags
                    for f in str(flag_text).split(";")
                    if f and f not in non_blocking_flags
                }
                if zc_rates.isna().any() or spreads.isna().any() or blocking_flags:
                    flag += ";" + ";".join(sorted(blocking_flags or {"SPREAD_MISSING"}))
                    if spreads.isna().any() or blocking_flags:
                        missing_fields.append("spread")
                else:
                    if flags:
                        flag += ";" + ";".join(sorted(flags))
                    spread_used = float(spreads.mean())
                    price, duration = discount_cashflows(future, date, zc_rates + spreads)
                    if pd.notna(outstanding_unit) and outstanding_unit > 0 and pd.notna(price):
                        return_price = price / outstanding_unit
                    else:
                        return_price = price
        paid = cashflow_paid_between(cashflows, previous_date, date)
        if paid and pd.notna(paid):
            flag = f"{flag};CASHFLOW_INCLUDED_IN_TOTAL_RETURN"
        daily_return = np.nan
        if previous_price is not None and pd.notna(previous_price) and previous_price != 0 and pd.notna(price):
            paid_amount = float(paid) if pd.notna(paid) else 0.0
            daily_return = (price + paid_amount - previous_price) / previous_price
            if pd.notna(daily_return) and abs(daily_return) > 0.03:
                flag = f"{flag};SUSPICIOUS_CORPORATE_DAILY_RETURN"
        rows.append(
            {
                "date": date,
                "asset_id": asset_id,
                "isin": isin,
                "asset_name": asset_name,
                "sector": sector,
                "asset_class": "corporate_bond",
                "dirty_price": last_dirty_price,
                "dirty_price_scale": dirty_price_scale,
                "outstanding_principal_unit": outstanding_unit,
                "model_price_unit": price,
                "dirty_price_model": price,
                "return_price_basis": return_price,
                "cashflow_paid": paid,
                "daily_return": daily_return,
                "spread_used": spread_used,
                "duration_modified": duration,
                "source_used": source_used,
                "pricing_method": "LEGACY_DISABLED_NOT_PRIMARY" if source_used == "LEGACY_OBSERVED_PRICE_RECON_ONLY" else "PROXY_DCF_ZC_PLUS_SECTOR_SPREAD",
                "pricing_flag": flag,
                "missing_fields": ", ".join(sorted(set(missing_fields))),
            }
        )
        if pd.notna(price):
            previous_date = date
            previous_price = price
    return pd.DataFrame(rows)


def build_corporate_bond_check(portfolio_df: pd.DataFrame, pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Table de contrôle et réconciliation corporate."""

    rows = []
    for _, asset in portfolio_df.loc[portfolio_df["asset_type"].eq("corporate_bond")].iterrows():
        asset_id = str(asset["asset_id"])
        metadata_override = CORPORATE_BOND_METADATA_OVERRIDES.get(asset_id, {})
        isin = asset.get("isin")
        if (pd.isna(isin) or str(isin).strip() in {"", "-", "nan", "None"}) and metadata_override.get("isin"):
            isin = metadata_override.get("isin")
        one = pricing_df.loc[pricing_df["asset_id"].eq(asset_id)].sort_values("date")
        last_any = one.tail(1)
        last = one.dropna(subset=["model_price_unit"]).tail(1)
        price = float(last["model_price_unit"].iloc[0]) if not last.empty else np.nan
        quantity = float(asset.get("quantity") or 0.0)
        model_value = price * quantity if pd.notna(price) else np.nan
        portfolio_value = float(asset.get("market_value") or np.nan)
        absolute_gap = model_value - portfolio_value if pd.notna(model_value) and portfolio_value else np.nan
        relative_gap = absolute_gap / portfolio_value if pd.notna(absolute_gap) and portfolio_value else np.nan
        flag_row = last if not last.empty else last_any
        flag = str(flag_row["pricing_flag"].dropna().iloc[-1]) if not flag_row.empty and not flag_row["pricing_flag"].dropna().empty else "DATA_MISSING"
        if pd.notna(relative_gap) and abs(relative_gap) > 0.05:
            flag += ";CORPORATE_RECONCILIATION_GAP_HIGH"
        if pd.notna(relative_gap) and abs(relative_gap) > 0.20:
            flag += ";CORPORATE_RECONCILIATION_GAP_CRITICAL"
        rows.append(
            {
                "asset_id": asset_id,
                "isin": isin,
                "asset_name": asset.get("asset_name"),
                "sector": asset.get("sector"),
                "quantity": quantity,
                "dirty_price_scale": last["dirty_price_scale"].iloc[0] if not last.empty else (last_any["dirty_price_scale"].iloc[0] if not last_any.empty else np.nan),
                "outstanding_principal_unit": last["outstanding_principal_unit"].iloc[0] if not last.empty else (last_any["outstanding_principal_unit"].iloc[0] if not last_any.empty else np.nan),
                "last_dirty_price": last["dirty_price"].iloc[0] if not last.empty else np.nan,
                "model_price_unit_last_date": price,
                "model_value_last_date": model_value,
                "portfolio_value": portfolio_value,
                "absolute_gap": absolute_gap,
                "relative_gap": relative_gap,
                "source_used": last["source_used"].iloc[0] if not last.empty else (last_any["source_used"].iloc[0] if not last_any.empty else "MODEL_PROXY"),
                "pricing_method": last["pricing_method"].iloc[0] if not last.empty and "pricing_method" in last else (last_any["pricing_method"].iloc[0] if not last_any.empty and "pricing_method" in last_any else "PROXY_DCF_ZC_PLUS_SECTOR_SPREAD"),
                "spread_used": last["spread_used"].iloc[0] if not last.empty else np.nan,
                "missing_fields": flag_row["missing_fields"].iloc[0] if not flag_row.empty and "missing_fields" in flag_row else np.nan,
                "pricing_flag": flag,
            }
        )
    return pd.DataFrame(rows)


def build_corporate_returns_check(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Contrôle les rendements corporate journaliers."""

    rows: list[dict[str, object]] = []
    for asset_id in returns_df.columns:
        r = returns_df[asset_id].dropna()
        ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
        flags: list[str] = []
        if not r.empty and (r.abs() > 0.03).any():
            flags.append("SUSPICIOUS_CORPORATE_DAILY_RETURN")
        if not r.empty and (r.abs() > 0.10).any():
            flags.append("EXTREME_DAILY_RETURN")
        if pd.notna(ann_vol) and ann_vol > 0.50:
            flags.append("EXTREME_ANNUALIZED_VOLATILITY")
        if pd.notna(ann_vol) and ann_vol < 0.001:
            flags.append("LOW_VOLATILITY_WARNING")
        wealth = (1 + r).cumprod() if len(r) else pd.Series(dtype=float)
        rows.append(
            {
                "asset_id": asset_id,
                "number_observations": int(len(r)),
                "min_return": float(r.min()) if len(r) else np.nan,
                "max_return": float(r.max()) if len(r) else np.nan,
                "annualized_volatility": ann_vol,
                "max_drawdown": float((wealth / wealth.cummax() - 1).min()) if len(wealth) else np.nan,
                "suspicious_return_flag": ";".join(flags) if flags else "OK",
            }
        )
    return pd.DataFrame(rows)


def _sector_premium_asof(
    sector_premiums: pd.DataFrame,
    valuation_date: pd.Timestamp,
    sector: str,
) -> tuple[float, str, str, float, object]:
    """Retourne la prime finale sectorielle à la date ou la dernière disponible."""

    if sector_premiums.empty:
        return np.nan, "MISSING_SECTOR_PREMIUM", "NO_VALID_SPREAD", np.nan, pd.NaT
    date = pd.Timestamp(valuation_date)
    sector = str(sector).upper().strip()
    prem = sector_premiums.copy()
    if "curve_date" in prem.columns and "date" not in prem.columns:
        prem = prem.rename(columns={"curve_date": "date"})
    prem["date"] = pd.to_datetime(prem["date"], errors="coerce")
    one = prem.loc[prem["date"].eq(date) & prem["sector"].astype(str).str.upper().eq(sector)].sort_values("date")
    if not one.empty:
        row = one.tail(1).iloc[0]
        return (
            float(row["final_sector_spread_decimal"]),
            str(row.get("spread_source", "OFFICIAL_TC_BULLETIN")),
            str(row.get("quality_flag", "OK")),
            float(row.get("days_since_last_official_spread", 0) or 0),
            row.get("bulletin_date", pd.NaT),
        )
    past = prem.loc[prem["date"].lt(date) & prem["sector"].astype(str).str.upper().eq(sector)].sort_values("date")
    if not past.empty:
        row = past.tail(1).iloc[0]
        return (
            float(row["final_sector_spread_decimal"]),
            str(row.get("spread_source", "OFFICIAL_TC_BULLETIN_FORWARD_FILLED")),
            str(row.get("quality_flag", "OK")),
            float(row.get("days_since_last_official_spread", (date - pd.Timestamp(row["date"])).days)),
            row.get("bulletin_date", pd.NaT),
        )
    return np.nan, "MISSING_SECTOR_PREMIUM", "NO_VALID_SPREAD", np.nan, pd.NaT


def _observed_corporate_row(
    corporate_clean_df: pd.DataFrame,
    isin: str | None,
    valuation_date: pd.Timestamp,
) -> dict[str, float]:
    """Récupère les prix observés uniquement pour rapprochement informatif."""

    if not isin or corporate_clean_df.empty:
        return {"observed_dirty_price": np.nan, "observed_clean_price": np.nan, "observed_ytm": np.nan}
    one = corporate_clean_df.loc[
        corporate_clean_df["isin"].astype(str).eq(str(isin))
        & pd.to_datetime(corporate_clean_df["curve_date"], errors="coerce").eq(pd.Timestamp(valuation_date))
    ].tail(1)
    if one.empty:
        return {"observed_dirty_price": np.nan, "observed_clean_price": np.nan, "observed_ytm": np.nan}
    row = one.iloc[0]
    return {
        "observed_dirty_price": pd.to_numeric(row.get("dirty_price"), errors="coerce"),
        "observed_clean_price": pd.to_numeric(row.get("clean_price"), errors="coerce"),
        "observed_ytm": pd.to_numeric(row.get("ytm_decimal"), errors="coerce"),
    }


def price_corporate_bond_dcf_with_sector_spread_daily(
    bond_row: pd.Series,
    sovereign_curves: pd.DataFrame,
    sector_premiums: pd.DataFrame,
    valuation_dates: pd.Series | pd.DatetimeIndex,
    cashflow_schedule: pd.DataFrame | None = None,
    corporate_clean_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Valorise une obligation corporate par DCF ZC souverain + prime sectorielle.

    Le dirty price observé n'alimente jamais le prix principal. Il reste stocké
    dans les colonnes de réconciliation.
    """

    asset_id = str(bond_row.get("asset_id"))
    asset_name = str(bond_row.get("asset_name", asset_id))
    metadata_override = CORPORATE_BOND_METADATA_OVERRIDES.get(asset_id, {})
    isin_raw = bond_row.get("isin")
    isin = None if pd.isna(isin_raw) or str(isin_raw).strip() in {"", "-", "nan", "None"} else str(isin_raw).strip()
    if isin is None and metadata_override.get("isin"):
        isin = str(metadata_override["isin"])
    sector, sector_flag = map_corporate_sector({**bond_row.to_dict(), **metadata_override})
    if asset_id == "EO_ATL_2025_2" or "ATL 2025-2" in asset_name.upper():
        sector = "LEASING"
        sector_flag = "SECTOR_FORCED_LEASING_FOR_ATL_2025_2"

    cashflows = cashflow_schedule.copy() if isinstance(cashflow_schedule, pd.DataFrame) and not cashflow_schedule.empty else generate_fixed_income_cashflows(bond_row)
    cashflows["cashflow_date"] = pd.to_datetime(cashflows.get("cashflow_date"), errors="coerce") if not cashflows.empty else pd.Series(dtype="datetime64[ns]")
    quantity = float(bond_row.get("quantity") or 0.0)
    rows: list[dict[str, object]] = []
    previous_price: float | None = None
    previous_date: pd.Timestamp | None = None
    for date in pd.to_datetime(valuation_dates):
        if date < pd.Timestamp(ANALYSIS_START_DATE) or date > pd.Timestamp(ANALYSIS_END_DATE):
            continue
        flags = [
            "MODEL_BASED_CORPORATE_VALUATION",
            "ZC_PLUS_OFFICIAL_TC_SECTOR_SPREAD",
            "DIRTY_PRICE_NOT_USED_FOR_PRIMARY_VALUATION",
            "TOTAL_RETURN_WITH_CASHFLOWS",
        ]
        if isin is None:
            flags.append("ISIN_MISSING")
        if sector_flag != "OK":
            flags.append(sector_flag)
        price = duration = np.nan
        zc_min_used = zc_max_used = np.nan
        spread, spread_source, spread_quality, days_since_spread, bulletin_date = _sector_premium_asof(sector_premiums, date, sector)
        if spread_source == "OFFICIAL_TC_BULLETIN_FORWARD_FILLED":
            flags.append("OFFICIAL_SPREAD_FORWARD_FILLED")
        elif spread_source == "ROBUST_MEDIAN_FALLBACK":
            flags.append("MEDIAN_SPREAD_FALLBACK_USED")
        elif spread_source not in {"OFFICIAL_TC_BULLETIN"}:
            flags.append(spread_source)
        if spread_quality not in {"OK", "OK_OFFICIAL", "OK_FORWARD_FILLED", "LOW_OBSERVATION_COUNT", "OUTLIERS_REMOVED"}:
            flags.append(spread_quality)
        if sector == "UNKNOWN_SECTOR":
            flags.append("UNKNOWN_SECTOR")
        if pd.isna(spread):
            flags.append("MISSING_SECTOR_PREMIUM")
        future = cashflows.loc[cashflows["cashflow_date"].gt(date)].copy() if not cashflows.empty else pd.DataFrame()
        if future.empty:
            flags.append("DATA_MISSING")
        elif sector != "UNKNOWN_SECTOR" and pd.notna(spread):
            future["tau"] = (future["cashflow_date"] - date).dt.days / 365.0
            zc_pairs = [interpolate_sovereign_rate(sovereign_curves, date, tau) for tau in future["tau"]]
            zc_rates = pd.Series([p[0] for p in zc_pairs], index=future.index, dtype=float)
            zc_flags = [p[1] for p in zc_pairs if p[1] != "OK"]
            if zc_rates.isna().any() or zc_flags:
                flags.extend(zc_flags or ["ZC_CURVE_MISSING"])
            else:
                total_rates = zc_rates + float(spread)
                price, duration = discount_cashflows(future, date, total_rates)
                zc_min_used = float(zc_rates.min())
                zc_max_used = float(zc_rates.max())
        paid = cashflow_paid_between(cashflows, previous_date, date)
        if paid and pd.notna(paid):
            flags.append("CASHFLOW_INCLUDED_IN_TOTAL_RETURN")
        daily_return = np.nan
        if previous_price is not None and pd.notna(previous_price) and previous_price != 0 and pd.notna(price):
            paid_amount = float(paid) if pd.notna(paid) else 0.0
            daily_return = (price + paid_amount) / previous_price - 1.0
            if abs(daily_return) > 0.05 and not paid_amount:
                flags.append("SUSPICIOUS_CORPORATE_DAILY_RETURN")
            elif abs(daily_return) > 0.05 and paid_amount:
                flags.append("LARGE_RETURN_EXPLAINED_BY_CASHFLOW")
        observed = _observed_corporate_row(corporate_clean_df if corporate_clean_df is not None else pd.DataFrame(), isin, date)
        dcf_value = price * quantity if pd.notna(price) else np.nan
        observed_dirty = observed["observed_dirty_price"]
        outstanding_unit, _ = get_outstanding_principal_unit(bond_row, date)
        dirty_value = scale_dirty_price_to_unit(observed_dirty, outstanding_unit, detect_dirty_price_scale(pd.Series([observed_dirty]))) * quantity if pd.notna(observed_dirty) and pd.notna(outstanding_unit) else np.nan
        reconciliation_gap = dcf_value - dirty_value if pd.notna(dcf_value) and pd.notna(dirty_value) else np.nan
        reconciliation_gap_pct = reconciliation_gap / dirty_value if pd.notna(reconciliation_gap) and dirty_value else np.nan
        rows.append(
            {
                "date": date,
                "asset_id": asset_id,
                "asset_name": asset_name,
                "isin": isin,
                "sector": sector,
                "asset_class": "corporate_bond",
                "model_price_unit": price,
                "portfolio_value_scaled": dcf_value,
                "cashflow_paid_unit": float(paid) if pd.notna(paid) else 0.0,
                "cashflow_paid": float(paid) if pd.notna(paid) else 0.0,
                "daily_return": daily_return,
                "final_sector_spread_decimal": spread,
                "spread_used": spread,
                "spread_source": spread_source,
                "spread_quality_flag": spread_quality,
                "days_since_last_official_spread": days_since_spread,
                "bulletin_date": bulletin_date,
                "zc_min_used": zc_min_used,
                "zc_max_used": zc_max_used,
                "duration_modified": duration,
                "pricing_method": "CORPORATE_DCF_ZC_PLUS_OFFICIAL_TC_SECTOR_SPREAD",
                "pricing_flag": ";".join(dict.fromkeys([f for f in flags if f and f != "OK"])),
                "quality_flag": ";".join(dict.fromkeys([f for f in flags if f and f != "OK"])),
                "observed_dirty_price": observed_dirty,
                "observed_clean_price": observed["observed_clean_price"],
                "observed_ytm": observed["observed_ytm"],
                "dirty_price_implied_value": dirty_value,
                "dcf_model_value": dcf_value,
                "reconciliation_gap": reconciliation_gap,
                "reconciliation_gap_pct": reconciliation_gap_pct,
            }
        )
        if pd.notna(price):
            previous_date = date
            previous_price = price
    return pd.DataFrame(rows)


def price_corporate_bond_daily(
    asset_row: pd.Series,
    sovereign_curves: pd.DataFrame,
    sector_spreads: pd.DataFrame,
    valuation_dates: pd.Series | pd.DatetimeIndex,
    corporate_clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compatibilité notebook : valorisation corporate primaire en DCF sectoriel."""

    return price_corporate_bond_dcf_with_sector_spread_daily(
        bond_row=asset_row,
        sovereign_curves=sovereign_curves,
        sector_premiums=sector_spreads,
        valuation_dates=valuation_dates,
        cashflow_schedule=asset_row.get("cashflow_schedule"),
        corporate_clean_df=corporate_clean_df,
    )


def build_corporate_bond_check(portfolio_df: pd.DataFrame, pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Contrôle synthétique des obligations corporate DCF + prime sectorielle."""

    rows: list[dict[str, object]] = []
    for _, asset in portfolio_df.loc[portfolio_df["asset_type"].eq("corporate_bond")].iterrows():
        asset_id = str(asset["asset_id"])
        one = pricing_df.loc[pricing_df["asset_id"].eq(asset_id)].sort_values("date")
        ret = one["daily_return"].dropna() if "daily_return" in one else pd.Series(dtype=float)
        last = one.dropna(subset=["model_price_unit"]).tail(1)
        qflag = str(one["quality_flag"].dropna().iloc[-1]) if not one.empty and one["quality_flag"].notna().any() else "DATA_MISSING"
        rows.append(
            {
                "asset_id": asset_id,
                "asset_name": asset.get("asset_name"),
                "isin": one["isin"].dropna().iloc[-1] if not one.empty and one["isin"].notna().any() else asset.get("isin"),
                "sector": one["sector"].dropna().iloc[-1] if not one.empty and one["sector"].notna().any() else asset.get("sector"),
                "pricing_method": one["pricing_method"].dropna().iloc[-1] if not one.empty and one["pricing_method"].notna().any() else "DATA_MISSING",
                "first_valuation_date": one["date"].min() if not one.empty else pd.NaT,
                "last_valuation_date": one["date"].max() if not one.empty else pd.NaT,
                "n_valuation_dates": int(one["model_price_unit"].notna().sum()) if "model_price_unit" in one else 0,
                "n_return_observations": int(ret.count()),
                "min_return": float(ret.min()) if len(ret) else np.nan,
                "max_return": float(ret.max()) if len(ret) else np.nan,
                "annualized_volatility": float(ret.std(ddof=1) * np.sqrt(252)) if len(ret) > 1 else np.nan,
                "has_cashflow_in_2025": bool(one["cashflow_paid_unit"].fillna(0).gt(0).any()) if "cashflow_paid_unit" in one else False,
                "total_cashflow_paid_2025": float(one["cashflow_paid_unit"].fillna(0).sum()) if "cashflow_paid_unit" in one else 0.0,
                "model_price_unit_last_date": float(last["model_price_unit"].iloc[0]) if not last.empty else np.nan,
                "model_value_last_date": float(last["portfolio_value_scaled"].iloc[0]) if not last.empty and "portfolio_value_scaled" in last else np.nan,
                "portfolio_value": float(asset.get("market_value") or np.nan),
                "absolute_gap": (float(last["portfolio_value_scaled"].iloc[0]) - float(asset.get("market_value"))) if not last.empty and pd.notna(asset.get("market_value")) else np.nan,
                "relative_gap": ((float(last["portfolio_value_scaled"].iloc[0]) - float(asset.get("market_value"))) / float(asset.get("market_value"))) if not last.empty and pd.notna(asset.get("market_value")) and float(asset.get("market_value")) != 0 else np.nan,
                "quality_flag": qflag,
            }
        )
    return pd.DataFrame(rows)


def build_corporate_returns_check(returns_df: pd.DataFrame, pricing_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Contrôle les rendements corporate et distingue les sauts expliqués."""

    rows: list[dict[str, object]] = []
    pricing = pricing_df if isinstance(pricing_df, pd.DataFrame) else pd.DataFrame()
    for asset_id in returns_df.columns:
        r = returns_df[asset_id].dropna()
        ann_vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
        flags: list[str] = []
        unexplained = False
        if not r.empty and (r.abs() > 0.05).any():
            dates = set(r.index[r.abs() > 0.05])
            one = pricing.loc[pricing["asset_id"].eq(asset_id) & pricing["date"].isin(dates)] if not pricing.empty else pd.DataFrame()
            explained = not one.empty and (
                one.get("cashflow_paid_unit", pd.Series(dtype=float)).fillna(0).gt(0).any()
                or one.get("quality_flag", pd.Series(dtype=str)).astype(str).str.contains("LARGE_RETURN_EXPLAINED_BY_CASHFLOW").any()
            )
            if explained:
                flags.append("LARGE_RETURN_EXPLAINED_BY_CASHFLOW")
            else:
                flags.append("SUSPICIOUS_CORPORATE_DAILY_RETURN")
                unexplained = True
        if not r.empty and (r.abs() > 0.10).any() and unexplained:
            flags.append("EXTREME_DAILY_RETURN")
        if pd.notna(ann_vol) and ann_vol > 0.50:
            flags.append("EXTREME_ANNUALIZED_VOLATILITY")
        if pd.notna(ann_vol) and ann_vol < 0.001:
            flags.append("LOW_VOLATILITY_WARNING")
        wealth = (1 + r).cumprod() if len(r) else pd.Series(dtype=float)
        rows.append(
            {
                "asset_id": asset_id,
                "number_observations": int(len(r)),
                "min_return": float(r.min()) if len(r) else np.nan,
                "max_return": float(r.max()) if len(r) else np.nan,
                "annualized_volatility": ann_vol,
                "max_drawdown": float((wealth / wealth.cummax() - 1).min()) if len(wealth) else np.nan,
                "suspicious_return_flag": ";".join(dict.fromkeys(flags)) if flags else "OK",
            }
        )
    return pd.DataFrame(rows)


def build_corporate_cashflow_inclusion_check(pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Liste les cashflows effectivement inclus dans les rendements corporate."""

    cols = ["date", "asset_id", "asset_name", "cashflow_paid_unit", "quality_flag"]
    if pricing_df.empty:
        return pd.DataFrame(columns=["asset_id", "asset_name", "cashflow_date", "cashflow_amount_unit", "included_in_return_period", "return_date", "quality_flag"])
    paid = pricing_df.loc[pricing_df["cashflow_paid_unit"].fillna(0).gt(0), cols].copy()
    return paid.rename(
        columns={
            "date": "return_date",
            "cashflow_paid_unit": "cashflow_amount_unit",
        }
    ).assign(cashflow_date=lambda d: d["return_date"], included_in_return_period=True)[
        ["asset_id", "asset_name", "cashflow_date", "cashflow_amount_unit", "included_in_return_period", "return_date", "quality_flag"]
    ]


def build_corporate_spread_usage_check(pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Trace la prime sectorielle utilisée par date et actif."""

    cols = [
        "date",
        "asset_id",
        "sector",
        "final_sector_spread_decimal",
        "spread_source",
        "spread_quality_flag",
        "days_since_last_official_spread",
        "bulletin_date",
        "quality_flag",
    ]
    out = pricing_df[cols].copy() if not pricing_df.empty else pd.DataFrame(columns=cols)
    out["zc_curve_available"] = ~out["quality_flag"].astype(str).str.contains("ZC_CURVE_MISSING|ZC_MATURITY_OUT_OF_RANGE", na=False)
    return out[
        [
            "date",
            "asset_id",
            "sector",
            "final_sector_spread_decimal",
            "spread_source",
            "spread_quality_flag",
            "days_since_last_official_spread",
            "bulletin_date",
            "zc_curve_available",
            "quality_flag",
        ]
    ]


def build_corporate_dirty_price_reconciliation(pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Build an informative reconciliation between observed dirty price and DCF value.

    The dirty price is kept only for audit purposes. It is not used to compute
    the corporate valuation, returns, risk metrics, covariance or expected
    returns.
    """

    cols = [
        "date",
        "asset_id",
        "asset_name",
        "observed_dirty_price",
        "model_price_unit",
        "dirty_price_implied_value",
        "portfolio_value_scaled",
        "reconciliation_gap",
        "reconciliation_gap_pct",
    ]
    target_cols = [
        "date",
        "asset_id",
        "asset_name",
        "observed_dirty_price_base1",
        "dcf_model_price_unit",
        "dirty_price_implied_portfolio_value",
        "dcf_model_portfolio_value",
        "gap_value_dt",
        "gap_pct",
        "comment",
    ]
    if pricing_df.empty:
        return pd.DataFrame(columns=target_cols)

    available_cols = [col for col in cols if col in pricing_df.columns]
    out = pricing_df.loc[pricing_df.get("observed_dirty_price").notna(), available_cols].copy()
    out = out.rename(
        columns={
            "observed_dirty_price": "observed_dirty_price_base1",
            "model_price_unit": "dcf_model_price_unit",
            "dirty_price_implied_value": "dirty_price_implied_portfolio_value",
            "portfolio_value_scaled": "dcf_model_portfolio_value",
            "reconciliation_gap": "gap_value_dt",
            "reconciliation_gap_pct": "gap_pct",
        }
    )
    for col in target_cols:
        if col not in out.columns:
            out[col] = np.nan
    out["comment"] = "Dirty price conservé uniquement pour réconciliation informative."
    return out[target_cols]
