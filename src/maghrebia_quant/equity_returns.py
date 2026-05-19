"""Listed equity prices, strict corporate actions and return series."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import ANALYSIS_END_DATE, ANALYSIS_START_DATE
from .loaders import coerce_numeric, filter_date_window, normalize_text, read_csv_flexible, slugify, validate_required_columns

logger = logging.getLogger(__name__)

EQUITY_NAME_ALIASES = {"PGH": ["POULINA GP HOLDING", "POULINA"]}


def load_bvmt_prices(paths: list[Path], start: str = ANALYSIS_START_DATE, end: str = ANALYSIS_END_DATE) -> pd.DataFrame:
    """Read BVMT price files, clean close prices and keep the analysis window."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            logger.warning("Fichier BVMT absent: %s", path)
            continue
        df = read_csv_flexible(path, sep=";")
        validate_required_columns(df, ["seance", "code", "valeur", "cloture"], f"BVMT {path.name}")
        df = df[["seance", "code", "valeur", "cloture"]].copy()
        df["date"] = pd.to_datetime(df["seance"], errors="coerce", dayfirst=True)
        df["bvmt_code"] = df["code"].astype(str).str.strip()
        df["asset_name"] = df["valeur"].astype(str).str.strip()
        df["asset_name_norm"] = df["asset_name"].map(normalize_text)
        df["close_raw"] = coerce_numeric(df["cloture"])
        non_positive = df["close_raw"].le(0) & df["close_raw"].notna()
        if non_positive.any():
            logger.warning("Clotures BVMT non positives ignorees dans %s: %s", path.name, int(non_positive.sum()))
            df.loc[non_positive, "close_raw"] = np.nan
        frames.append(df[["date", "bvmt_code", "asset_name", "asset_name_norm", "close_raw"]])
    if not frames:
        raise FileNotFoundError("Aucun historique BVMT exploitable.")
    prices = pd.concat(frames, ignore_index=True).dropna(subset=["date", "bvmt_code", "close_raw"])
    duplicates = prices.duplicated(["date", "bvmt_code"], keep=False)
    if duplicates.any():
        logger.warning("Doublons BVMT date-code supprimes: %s", int(duplicates.sum()))
        prices = prices.sort_values(["date", "bvmt_code"]).drop_duplicates(["date", "bvmt_code"], keep="last")
    prices = filter_date_window(prices, "date", start, end)
    return prices.sort_values(["bvmt_code", "date"]).reset_index(drop=True)


def apply_corporate_actions(prices_df: pd.DataFrame, actions_config: Sequence[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exact-match corporate actions and produce an auditable event table.

    Dividends are not adjusted because the diagnostic notebook measures price
    returns. Splits and capital adjustments multiply pre-event prices only when
    the BVMT code or exact normalized name matches.
    """

    out = prices_df.copy()
    out["close_adjusted"] = out["close_raw"]
    out["adjustment_flag"] = False
    out["adjustment_factor_applied"] = 1.0
    out["adjustment_comment"] = ""
    audit_rows: list[dict[str, object]] = []

    def _audit_row(
        action: dict[str, object],
        *,
        asset_name: object,
        date: object,
        effective_date: pd.Timestamp | pd.NaT,
        factor: float,
        raw_price: float | object = np.nan,
        adjusted_price: float | object = np.nan,
        raw_price_before: float | object = np.nan,
        adjusted_price_before: float | object = np.nan,
        raw_price_after: float | object = np.nan,
        adjusted_price_after: float | object = np.nan,
        applied_flag: bool,
        quality_flag: str,
        comment: str | None = None,
    ) -> dict[str, object]:
        return {
            "asset_id": action.get("asset_id"),
            "asset_name": asset_name if pd.notna(asset_name) else action.get("asset_name_exact"),
            "ticker": action.get("bvmt_code"),
            "corporate_action_type": action.get("action_type", "price_adjustment"),
            "date": date,
            "effective_date": effective_date,
            "factor": factor,
            "adjustment_factor": factor,
            "adjustment_direction": "PRE_EVENT_PRICE_MULTIPLIED" if applied_flag else "NO_PRICE_ADJUSTMENT",
            "raw_price": raw_price,
            "adjusted_price": adjusted_price,
            "raw_price_before": raw_price_before,
            "adjusted_price_before": adjusted_price_before,
            "raw_price_after": raw_price_after,
            "adjusted_price_after": adjusted_price_after,
            "applied_flag": applied_flag,
            "flag": quality_flag,
            "quality_flag": quality_flag,
            "comment": comment if comment is not None else str(action.get("comment", "")),
        }

    for action in actions_config:
        exact_name = normalize_text(action.get("asset_name_exact", ""))
        bvmt_code = str(action.get("bvmt_code") or "").strip()
        effective_date = pd.to_datetime(action.get("effective_date"), errors="coerce")
        action_type = str(action.get("action_type", "price_adjustment")).lower().strip()
        factor = float(action.get("price_adjustment_factor", action.get("factor", 1.0)) or 1.0)

        code_match = out["bvmt_code"].eq(bvmt_code) if bvmt_code else pd.Series(False, index=out.index)
        name_match = out["asset_name_norm"].eq(exact_name) if exact_name else pd.Series(False, index=out.index)
        matched_asset = code_match | name_match
        matched_rows = out.loc[matched_asset].sort_values("date")
        display_name = matched_rows["asset_name"].iloc[0] if not matched_rows.empty else action.get("asset_name_exact")

        if pd.isna(effective_date):
            audit_rows.append(
                _audit_row(
                    action,
                    asset_name=display_name,
                    date=pd.NaT,
                    effective_date=pd.NaT,
                    factor=factor,
                    applied_flag=False,
                    quality_flag="CORPORATE_ACTION_DATE_MISSING",
                    comment="Date effective manquante : operation non appliquee.",
                )
            )
            continue

        if matched_rows.empty:
            audit_rows.append(
                _audit_row(
                    action,
                    asset_name=display_name,
                    date=effective_date,
                    effective_date=effective_date,
                    factor=factor,
                    applied_flag=False,
                    quality_flag="CORPORATE_ACTION_REQUIRES_MANUAL_VALIDATION",
                    comment="Aucun instrument BVMT exact ne correspond a cette operation.",
                )
            )
            continue

        pre_rows = matched_rows.loc[matched_rows["date"] < effective_date]
        post_rows = matched_rows.loc[matched_rows["date"] >= effective_date]
        raw_price_before = float(pre_rows["close_raw"].iloc[-1]) if not pre_rows.empty else np.nan
        raw_price_after = float(post_rows["close_raw"].iloc[0]) if not post_rows.empty else np.nan
        adjusted_price_before = raw_price_before if action_type == "dividend" else raw_price_before * factor
        adjusted_price_after = raw_price_after

        if action_type == "dividend":
            audit_rows.append(
                _audit_row(
                    action,
                    asset_name=display_name,
                    date=effective_date,
                    effective_date=effective_date,
                    factor=1.0,
                    raw_price_before=raw_price_before,
                    adjusted_price_before=adjusted_price_before,
                    raw_price_after=raw_price_after,
                    adjusted_price_after=adjusted_price_after,
                    applied_flag=False,
                    quality_flag="DIVIDEND_PRICE_RETURN_ONLY",
                )
            )
            continue

        mask = matched_asset & (out["date"] < effective_date)
        before = out.loc[mask, "close_adjusted"].copy()
        if before.empty:
            audit_rows.append(
                _audit_row(
                    action,
                    asset_name=display_name,
                    date=effective_date,
                    effective_date=effective_date,
                    factor=factor,
                    raw_price_before=raw_price_before,
                    adjusted_price_before=adjusted_price_before,
                    raw_price_after=raw_price_after,
                    adjusted_price_after=adjusted_price_after,
                    applied_flag=False,
                    quality_flag="CORPORATE_ACTION_REQUIRES_MANUAL_VALIDATION",
                    comment="Aucune observation pre-evenement a ajuster dans la fenetre chargee.",
                )
            )
            continue

        out.loc[mask, "close_adjusted"] = out.loc[mask, "close_adjusted"] * factor
        out.loc[mask, "adjustment_flag"] = True
        out.loc[mask, "adjustment_factor_applied"] *= factor
        out.loc[mask, "adjustment_comment"] = str(action.get("comment", "Ajustement corporate action"))
        for idx, raw_price in before.items():
            audit_rows.append(
                _audit_row(
                    action,
                    asset_name=out.at[idx, "asset_name"],
                    date=out.at[idx, "date"],
                    effective_date=effective_date,
                    factor=factor,
                    raw_price=raw_price,
                    adjusted_price=out.at[idx, "close_adjusted"],
                    raw_price_before=raw_price_before,
                    adjusted_price_before=adjusted_price_before,
                    raw_price_after=raw_price_after,
                    adjusted_price_after=adjusted_price_after,
                    applied_flag=True,
                    quality_flag="CORPORATE_ACTION_ADJUSTED",
                )
            )

    return out, pd.DataFrame(audit_rows)


def _match_equity_codes(prices_df: pd.DataFrame, portfolio_assets: pd.DataFrame) -> dict[str, str]:
    names = prices_df[["bvmt_code", "asset_name_norm"]].drop_duplicates()
    mapping: dict[str, str] = {}
    for _, asset in portfolio_assets.loc[portfolio_assets["asset_type"].eq("listed_equity")].iterrows():
        asset_name = normalize_text(asset["asset_name"])
        aliases = [normalize_text(a) for a in EQUITY_NAME_ALIASES.get(asset_name, [])]
        cleaned = asset_name.replace("HOLDING", "").strip()
        candidates: list[str] = []
        for _, price_row in names.iterrows():
            price_name = price_row["asset_name_norm"]
            if " DA " in f" {price_name} ":
                continue
            if price_name == asset_name or price_name == cleaned or price_name in aliases:
                candidates.insert(0, price_row["bvmt_code"])
            elif len(asset_name) > 3 and (asset_name in price_name or price_name in asset_name):
                candidates.append(price_row["bvmt_code"])
        if candidates:
            mapping[str(asset["asset_id"])] = candidates[0]
    return mapping


def build_weekly_equity_prices(prices_df: pd.DataFrame, portfolio_assets: pd.DataFrame) -> pd.DataFrame:
    """Build weekly adjusted close prices using the last available session."""

    mapping = _match_equity_codes(prices_df, portfolio_assets)
    if not mapping:
        return pd.DataFrame()
    reverse = {code: asset_id for asset_id, code in mapping.items()}
    filtered = prices_df.loc[prices_df["bvmt_code"].isin(reverse)].copy()
    filtered["asset_id"] = filtered["bvmt_code"].map(reverse)
    return (
        filtered.pivot_table(index="date", columns="asset_id", values="close_adjusted", aggfunc="last")
        .sort_index()
        .resample("W-FRI")
        .last()
        .loc[lambda x: x.index <= pd.Timestamp(ANALYSIS_END_DATE)]
        .dropna(how="all")
    )


def build_daily_equity_prices(prices_df: pd.DataFrame, portfolio_assets: pd.DataFrame) -> pd.DataFrame:
    """Build daily adjusted close prices on real quote dates only."""

    mapping = _match_equity_codes(prices_df, portfolio_assets)
    if not mapping:
        return pd.DataFrame()
    reverse = {code: asset_id for asset_id, code in mapping.items()}
    filtered = prices_df.loc[prices_df["bvmt_code"].isin(reverse)].copy()
    filtered["asset_id"] = filtered["bvmt_code"].map(reverse)
    matrix = filtered.pivot_table(index="date", columns="asset_id", values="close_adjusted", aggfunc="last").sort_index()
    return matrix.dropna(how="all")


def build_daily_equity_prices_real_quotes(prices_df: pd.DataFrame, portfolio_assets: pd.DataFrame) -> pd.DataFrame:
    """Build an equity price matrix using real quotation dates, without forward-fill."""

    price_matrix = build_daily_equity_prices(prices_df, portfolio_assets)
    all_nan_dates = price_matrix.index[price_matrix.isna().all(axis=1)]
    price_matrix.attrs["all_nan_dates_removed"] = int(len(all_nan_dates))
    if len(all_nan_dates):
        import warnings

        warnings.warn(
            "build_daily_equity_prices_real_quotes: "
            f"{len(all_nan_dates)} date(s) sans aucun prix d'action supprimee(s): {list(all_nan_dates)}",
            UserWarning,
            stacklevel=2,
        )
        price_matrix = price_matrix.drop(index=all_nan_dates)
    return price_matrix


def compute_daily_equity_returns_real_quotes(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute equity returns between consecutive real quotation dates."""

    return compute_daily_equity_returns(daily_prices)


def compute_daily_equity_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute arithmetic daily returns R_t = P_t / P_{t-1} - 1."""

    if daily_prices.empty:
        return daily_prices.copy()
    returns = daily_prices.sort_index().pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    minima = returns.min(skipna=True)
    bad = minima[minima <= -0.99]
    assert bad.empty, f"Rendements de -100% detectes : {bad.to_dict()}"
    return returns


def compute_weekly_equity_returns(weekly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly arithmetic returns."""

    if weekly_prices.empty:
        return weekly_prices.copy()
    returns = weekly_prices.sort_index().pct_change(fill_method=None)
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
