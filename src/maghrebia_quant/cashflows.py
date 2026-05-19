"""Extraction et génération des cash-flows obligataires."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import coerce_numeric, slugify


def extract_cashflow_schedules(path: Path, sheet_name: str) -> pd.DataFrame:
    """Extrait les blocs de cash-flows des onglets Excel du portefeuille."""

    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    records: list[dict[str, object]] = []
    current: str | None = None
    for _, row in raw.iterrows():
        first = row.iloc[0]
        second = str(row.iloc[1]).upper() if len(row) > 1 and pd.notna(row.iloc[1]) else ""
        if pd.notna(first) and "CAPITAL" in second:
            current = str(first).strip().lstrip("-* ")
            continue
        if current is None:
            continue
        cf_date = pd.to_datetime(first, errors="coerce", dayfirst=True)
        if pd.isna(cf_date):
            continue
        records.append(
            {
                "schedule_name": current,
                "schedule_key": slugify(current),
                "cashflow_date": cf_date,
                "capital_begin": coerce_numeric(pd.Series([row.iloc[1]])).iloc[0],
                "interest": coerce_numeric(pd.Series([row.iloc[2]])).iloc[0],
                "principal": coerce_numeric(pd.Series([row.iloc[3]])).iloc[0],
                "cashflow": coerce_numeric(pd.Series([row.iloc[4]])).iloc[0],
            }
        )
    return pd.DataFrame(records)


def attach_cashflow_schedule(asset_row: pd.Series, schedules: pd.DataFrame) -> pd.Series:
    """Attache le planning de cash-flows correspondant à une ligne portefeuille."""

    row = asset_row.copy()
    schedule = schedules.loc[schedules["schedule_key"].eq(row.get("schedule_key"))].copy() if not schedules.empty else pd.DataFrame()
    if not schedule.empty:
        row["cashflow_schedule"] = schedule
    return row


def generate_fixed_income_cashflows(asset_row: pd.Series) -> pd.DataFrame:
    """Retourne uniquement les cash-flows contractuels disponibles."""

    schedule = asset_row.get("cashflow_schedule")
    if isinstance(schedule, pd.DataFrame) and not schedule.empty:
        cols = [c for c in ["cashflow_date", "capital_begin", "interest", "principal", "cashflow"] if c in schedule.columns]
        cf = schedule[cols].copy()
    else:
        cf = pd.DataFrame(columns=["cashflow_date", "capital_begin", "interest", "principal", "cashflow"])
    if cf.empty:
        return pd.DataFrame(columns=["cashflow_date", "capital_begin", "interest", "principal", "cashflow"])
    cf["cashflow_date"] = pd.to_datetime(cf["cashflow_date"], errors="coerce")
    for col in ["capital_begin", "interest", "principal", "cashflow"]:
        if col not in cf:
            cf[col] = 0.0
        cf[col] = pd.to_numeric(cf[col], errors="coerce").fillna(0.0)
    return cf.dropna(subset=["cashflow_date"]).sort_values("cashflow_date").reset_index(drop=True)


def infer_nominal(asset_row: pd.Series, cashflows: pd.DataFrame | None = None) -> float:
    """Infère le nominal unitaire uniquement depuis les données disponibles."""

    capital = asset_row.get("nominal")
    if pd.notna(capital) and float(capital) > 0:
        return float(capital)
    if cashflows is not None and not cashflows.empty:
        principal_max = cashflows["principal"].max()
        if principal_max > 0:
            return float(max(principal_max, cashflows.get("capital_begin", pd.Series([0])).max()))
    return np.nan


def cashflow_paid_between(cashflows: pd.DataFrame, previous_date: pd.Timestamp | None, valuation_date: pd.Timestamp) -> float:
    """Cash-flow payé dans l'intervalle (t-1, t]."""

    if previous_date is None or cashflows.empty:
        return 0.0
    paid = cashflows[(cashflows["cashflow_date"] > previous_date) & (cashflows["cashflow_date"] <= valuation_date)]
    return float(paid["cashflow"].sum())


def discount_cashflows(cashflows: pd.DataFrame, valuation_date: pd.Timestamp, rates: pd.Series | np.ndarray) -> tuple[float, float]:
    """Actualise une série de flux avec des taux par maturité et retourne prix, duration modifiée."""

    future = cashflows[cashflows["cashflow_date"] > valuation_date].copy()
    if future.empty:
        return np.nan, np.nan
    future["tau"] = (future["cashflow_date"] - valuation_date).dt.days / 365.0
    rate = np.asarray(rates, dtype=float)
    discounts = (1.0 + rate) ** future["tau"].to_numpy(float)
    pv_flows = future["cashflow"].to_numpy(float) / discounts
    price = float(pv_flows.sum())
    duration = float((future["tau"].to_numpy(float) * pv_flows).sum() / price / (1.0 + np.nanmean(rate))) if price > 0 else np.nan
    return price, duration
