"""Courbes souveraines et corporate journalières 2025."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANALYSIS_END_DATE, ANALYSIS_START_DATE
from .loaders import coerce_rate_to_decimal, filter_date_window, standardize_columns, validate_required_columns

LAST_SOVEREIGN_ZC_OUTLIERS = pd.DataFrame(columns=["curve_date", "reason", "n_affected_points"])


def _normalize_sector_label(value: object) -> str:
    """Normalise les libelles sectoriels corporate vers les trois secteurs TC."""

    text = str(value or "").upper().strip()
    if any(token in text for token in ["BANK", "BANQ", "BANCAIRE"]):
        return "BANCAIRE"
    if "LEAS" in text:
        return "LEASING"
    if "MICRO" in text:
        return "MICROFINANCE"
    return "UNKNOWN_SECTOR"


def _parse_spread_decimal(value: object) -> float:
    """Convertit une prime lue en decimal, que la source soit en %, bps ou decimal."""

    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        text = value.strip().replace("\u00a0", "").replace(" ", "").replace(",", ".")
        is_pct = text.endswith("%")
        text = text.rstrip("%")
        try:
            parsed = float(text)
        except ValueError:
            return np.nan
        if is_pct:
            return parsed / 100.0
        value = parsed
    value = float(value)
    if abs(value) > 5.0:
        return value / 10_000.0
    if abs(value) > 0.05:
        return value / 100.0
    return value


def _clean_sovereign_curve_outliers(
    df: pd.DataFrame,
    rate_col: str = "zero_coupon_rate_decimal",
    date_col: str = "curve_date",
    max_rate: float = 0.50,
    min_rate: float = -0.05,
    max_daily_shift_bps: float = 200.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Supprime les dates de courbe souveraine aberrantes et retourne le rapport."""

    outlier_rows: list[pd.DataFrame] = []
    clean = df.copy()
    bad_abs = clean[(clean[rate_col] > max_rate) | (clean[rate_col] < min_rate)]
    if not bad_abs.empty:
        bad_dates_abs = pd.Index(bad_abs[date_col].dropna().unique())
        outlier_rows.append(
            pd.DataFrame(
                {
                    "curve_date": bad_dates_abs,
                    "reason": f"RATE_OUT_OF_BOUNDS [{min_rate:.0%}, {max_rate:.0%}]",
                    "n_affected_points": [int((clean[date_col] == date).sum()) for date in bad_dates_abs],
                }
            )
        )
        clean = clean.loc[~clean[date_col].isin(bad_dates_abs)].copy()

    pivot = (
        clean.groupby([date_col, "maturity_years"])[rate_col]
        .mean()
        .unstack("maturity_years")
        .sort_index()
    )
    median_shift = pivot.diff().abs().median(axis=1)
    bad_shift_dates = median_shift[median_shift > max_daily_shift_bps / 10_000].index
    if len(bad_shift_dates):
        outlier_rows.append(
            pd.DataFrame(
                {
                    "curve_date": bad_shift_dates,
                    "reason": f"DAILY_SHIFT_EXCEEDS_{max_daily_shift_bps:g}BPS",
                    "n_affected_points": [int((clean[date_col] == date).sum()) for date in bad_shift_dates],
                }
            )
        )
        clean = clean.loc[~clean[date_col].isin(bad_shift_dates)].copy()

    outlier_report = (
        pd.concat(outlier_rows, ignore_index=True)
        if outlier_rows
        else pd.DataFrame(columns=["curve_date", "reason", "n_affected_points"])
    )
    return clean.reset_index(drop=True), outlier_report


def get_last_sovereign_zc_outliers() -> pd.DataFrame:
    """Retourne le dernier rapport d'outliers souverains calculÃ© au chargement."""

    return LAST_SOVEREIGN_ZC_OUTLIERS.copy()


def _read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier de courbe introuvable: {path}")
    return standardize_columns(pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig"))


def load_sovereign_curves_daily(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge et contrôle la courbe souveraine journalière 2025."""

    global LAST_SOVEREIGN_ZC_OUTLIERS
    df = _read_csv_auto(path)
    required = ["curve_date", "maturity_years", "zero_coupon_rate_decimal", "extraction_status"]
    validate_required_columns(df, required, "Courbe souveraine daily")
    df["curve_date"] = pd.to_datetime(df["curve_date"], errors="coerce")
    df["maturity_years"] = pd.to_numeric(df["maturity_years"], errors="coerce")
    df["zero_coupon_rate_decimal"] = coerce_rate_to_decimal(df["zero_coupon_rate_decimal"])
    df["extraction_status"] = df["extraction_status"].astype(str).str.upper().str.strip()
    df = filter_date_window(df, "curve_date")
    df = df.loc[df["extraction_status"].eq("EXTRACTED"), ["curve_date", "maturity_years", "zero_coupon_rate_decimal"]].copy()
    df, LAST_SOVEREIGN_ZC_OUTLIERS = _clean_sovereign_curve_outliers(df, max_daily_shift_bps=30.0)
    dup_count = int(df.duplicated(["curve_date", "maturity_years"]).sum())
    missing_count = int(df[["curve_date", "maturity_years", "zero_coupon_rate_decimal"]].isna().sum().sum())
    maturities_per_date = df.groupby("curve_date")["maturity_years"].nunique()
    rate_min = df["zero_coupon_rate_decimal"].min()
    rate_max = df["zero_coupon_rate_decimal"].max()
    check = pd.DataFrame(
        [
            {
                "rows": len(df),
                "number_dates": df["curve_date"].nunique(),
                "missing_values": missing_count,
                "duplicate_curve_date_maturity": dup_count,
                "min_maturities_per_date": int(maturities_per_date.min()) if not maturities_per_date.empty else 0,
                "max_maturities_per_date": int(maturities_per_date.max()) if not maturities_per_date.empty else 0,
                "rate_min": rate_min,
                "rate_max": rate_max,
                "zc_outlier_dates_removed": int(LAST_SOVEREIGN_ZC_OUTLIERS["curve_date"].nunique()),
                "is_usable": bool(missing_count == 0 and dup_count == 0 and not maturities_per_date.empty and maturities_per_date.min() >= 2 and 0 <= rate_min <= rate_max < 1),
            }
        ]
    )
    if dup_count:
        df = df.drop_duplicates(["curve_date", "maturity_years"], keep="last")
    return df.sort_values(["curve_date", "maturity_years"]).reset_index(drop=True), check


def interpolate_sovereign_rate(sovereign_df: pd.DataFrame, curve_date: pd.Timestamp, maturity_years: float) -> tuple[float, str]:
    """Interpôle linéairement le taux souverain sans extrapolation."""

    if pd.isna(maturity_years) or maturity_years <= 0:
        return np.nan, "ZC_MATURITY_OUT_OF_RANGE"
    curve = sovereign_df.loc[pd.to_datetime(sovereign_df["curve_date"]).eq(pd.Timestamp(curve_date)), ["maturity_years", "zero_coupon_rate_decimal"]].dropna()
    if curve.empty:
        return np.nan, "ZC_CURVE_MISSING"
    maturities = curve["maturity_years"].to_numpy(float)
    rates = curve["zero_coupon_rate_decimal"].to_numpy(float)
    order = np.argsort(maturities)
    maturities, rates = maturities[order], rates[order]
    if maturity_years < maturities.min() or maturity_years > maturities.max():
        return np.nan, "ZC_MATURITY_OUT_OF_RANGE"
    return float(np.interp(maturity_years, maturities, rates)), "OK"


def load_corporate_curves_daily(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les courbes corporate instrument-level journalières 2025."""

    df = _read_csv_auto(path)
    required = [
        "curve_date",
        "sector",
        "maturity_years",
        "corporate_rate_decimal",
        "ytm_decimal",
        "dirty_price",
        "clean_price",
        "raw_value",
        "extraction_status",
    ]
    validate_required_columns(df, required, "Courbes corporate daily")
    df["curve_date"] = pd.to_datetime(df["curve_date"], errors="coerce")
    df["sector"] = df["sector"].astype(str).str.upper().str.strip()
    for col in ["maturity_years", "corporate_rate_decimal", "ytm_decimal", "dirty_price", "clean_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["corporate_rate_decimal"] = coerce_rate_to_decimal(df["corporate_rate_decimal"])
    df["ytm_decimal"] = coerce_rate_to_decimal(df["ytm_decimal"])
    df["extraction_status"] = df["extraction_status"].astype(str).str.upper().str.strip()
    df = filter_date_window(df, "curve_date")
    df = df.loc[df["extraction_status"].eq("EXTRACTED"), required[:-1]].copy()
    extracted = df["raw_value"].astype(str)
    df["isin"] = extracted.str.extract(r"\b(TN[A-Z0-9]{10})\b", expand=False)
    df["label"] = extracted.str.replace(r"\bTN[A-Z0-9]{10}\b", "", regex=True).str.strip()
    df["type_instrument"] = extracted.str.extract(r"\b(TF|TV|SUB|ORDINARY|SUBORDINATED)\b", expand=False)
    df["duration_modified"] = extracted.str.extract(r"(?:duration|durée|duree)[^0-9-]*([0-9]+(?:[.,][0-9]+)?)", expand=False)
    df["duration_modified"] = pd.to_numeric(df["duration_modified"].str.replace(",", ".", regex=False), errors="coerce")
    quality = pd.DataFrame(
        [
            {
                "rows": len(df),
                "number_dates": df["curve_date"].nunique(),
                "sectors": ", ".join(sorted(s for s in df["sector"].dropna().unique() if s != "NAN")),
                "unique_isin": df["isin"].nunique(dropna=True),
                "missing_values": int(df.isna().sum().sum()),
                "duplicate_curve_date_isin": int(df.dropna(subset=["isin"]).duplicated(["curve_date", "isin"]).sum()),
                "outlier_rows": int(((df["corporate_rate_decimal"] > 0.30) | (df["corporate_rate_decimal"] <= 0) | (df["dirty_price"] <= 0) | (df["clean_price"] <= 0) | (df["maturity_years"] <= 0)).sum()),
                "dates_without_sector": int(df.loc[df["sector"].isin(["", "NAN"]) | df["sector"].isna(), "curve_date"].nunique()),
            }
        ]
    )
    return df.sort_values(["curve_date", "sector", "maturity_years"]).reset_index(drop=True), quality


def clean_corporate_curves_daily(corporate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nettoie les lignes corporate et conserve les outliers exclus."""

    df = corporate_df.copy()
    reasons = pd.Series("", index=df.index, dtype=object)
    masks = {
        "NON_POSITIVE_CORPORATE_RATE": df["corporate_rate_decimal"] <= 0,
        "NON_POSITIVE_DIRTY_PRICE": df["dirty_price"] <= 0,
        "NON_POSITIVE_MATURITY": df["maturity_years"] <= 0,
        "EXTREME_CORPORATE_RATE": df["corporate_rate_decimal"] > 0.30,
        "NON_POSITIVE_CLEAN_PRICE": df["clean_price"] <= 0,
    }
    for reason, mask in masks.items():
        reasons.loc[mask.fillna(False)] = reasons.loc[mask.fillna(False)].where(reasons.loc[mask.fillna(False)].eq(""), reasons.loc[mask.fillna(False)] + ";") + reason
    outlier_mask = reasons.ne("")
    outliers = df.loc[outlier_mask, ["curve_date", "sector", "isin", "label", "maturity_years", "corporate_rate_decimal", "dirty_price", "clean_price"]].copy()
    outliers["reason"] = reasons.loc[outlier_mask].values
    outliers["flag"] = "CORPORATE_CURVE_OUTLIER_REMOVED"
    clean = df.loc[~outlier_mask].copy()
    return clean.reset_index(drop=True), outliers.reset_index(drop=True)


def build_sector_spreads_daily(corporate_clean_df: pd.DataFrame, sovereign_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcule les spreads corporate instrument-level et prépare l'agrégation robuste."""

    sovereign_by_date = {
        pd.Timestamp(date): group.sort_values("maturity_years")[["maturity_years", "zero_coupon_rate_decimal"]].to_numpy(float)
        for date, group in sovereign_df.groupby("curve_date", sort=False)
    }
    rows = []
    flags = []
    for _, row in corporate_clean_df.iterrows():
        curve = sovereign_by_date.get(pd.Timestamp(row["curve_date"]))
        if curve is None or len(curve) < 2 or pd.isna(row["maturity_years"]):
            zc, zc_flag = np.nan, "ZC_CURVE_MISSING"
        else:
            maturities = curve[:, 0]
            rates = curve[:, 1]
            maturity = float(row["maturity_years"])
            if maturity < maturities.min() or maturity > maturities.max():
                zc, zc_flag = np.nan, "ZC_MATURITY_OUT_OF_RANGE"
            else:
                zc, zc_flag = float(np.interp(maturity, maturities, rates)), "OK"
        spread = row["corporate_rate_decimal"] - zc if pd.notna(zc) else np.nan
        flag = zc_flag
        if pd.notna(spread) and spread < -0.02:
            flag = "NEGATIVE_SPREAD_WARNING"
        if pd.notna(spread) and spread > 0.15:
            flag = "EXTREME_SPREAD_WARNING"
        rows.append({**row.to_dict(), "sovereign_zc_decimal_interpolated": zc, "spread_decimal": spread, "spread_flag": flag})
        if flag != "OK":
            flags.append({"date": row["curve_date"], "asset_id": row.get("isin"), "flag": flag, "message": "Contrôle spread corporate.", "severity": 2})
    spreads = pd.DataFrame(rows)
    spreads_clean = spreads.loc[spreads["spread_decimal"].notna() & spreads["spread_flag"].eq("OK")].copy()
    sector_spreads = (
        spreads_clean.groupby(["curve_date", "sector", "maturity_years"], as_index=False)
        .agg(spread_decimal=("spread_decimal", "median"), n_instruments=("spread_decimal", "count"))
        .sort_values(["curve_date", "sector", "maturity_years"])
    )
    sector_spreads["spread_decimal_raw"] = sector_spreads["spread_decimal"]
    sector_spreads["spread_smoothing_flag"] = "OK"
    max_daily_change = 0.0050
    flagged_parts: list[pd.DataFrame] = []
    for (_, _), group in sector_spreads.groupby(["sector", "maturity_years"], sort=False):
        group = group.sort_values("curve_date").copy()
        changes = group["spread_decimal"].diff()
        jump_mask = changes.abs().gt(max_daily_change).fillna(False)
        if jump_mask.any():
            group.loc[jump_mask, "spread_smoothing_flag"] = "SPREAD_DAILY_CHANGE_FLAGGED_NO_SMOOTHING"
            for _, flagged_row in group.loc[jump_mask].iterrows():
                flags.append(
                    {
                        "date": flagged_row["curve_date"],
                        "asset_id": f"{flagged_row['sector']}_{flagged_row['maturity_years']}",
                        "flag": "SPREAD_DAILY_CHANGE_FLAGGED_NO_SMOOTHING",
                        "message": "Variation quotidienne du spread sectoriel superieure a 50 bps, conservee sans lissage.",
                        "severity": 1,
                    }
                )
        flagged_parts.append(group)
    if flagged_parts:
        sector_spreads = pd.concat(flagged_parts, ignore_index=True).sort_values(["curve_date", "sector", "maturity_years"])
    return sector_spreads, pd.DataFrame(flags)


def map_corporate_sector(row: pd.Series | dict) -> tuple[str, str]:
    """Mappe un instrument corporate vers un secteur exploitable.

    Le mapping reste déterministe et documenté : on utilise d'abord le secteur
    fourni, puis le nom de l'actif / émetteur / libellé. Aucun rapprochement
    approximatif n'est appliqué.
    """

    data = dict(row)
    raw_sector = str(data.get("sector", "") or "").upper().strip()
    text = " ".join(
        str(data.get(col, "") or "").upper()
        for col in ["asset_name", "issuer_name", "issuer", "label", "raw_value", "isin", "asset_id"]
    )
    normalized = {
        "BANKING": "BANCAIRE",
        "BANK": "BANCAIRE",
        "BANCAIRE": "BANCAIRE",
        "LEASING": "LEASING",
        "MICROFINANCE": "MICROFINANCE",
    }.get(raw_sector)
    if normalized:
        return normalized, "OK"
    if "ATL" in text or "LEASING" in text or "HL " in text or "HANNIBAL" in text or "BH LEASING" in text:
        return "LEASING", "SECTOR_INFERRED_FROM_NAME"
    if "ADVANS" in text or "MICROCRED" in text or "BAOBAB" in text or "MICRO" in text:
        return "MICROFINANCE", "SECTOR_INFERRED_FROM_NAME"
    if "BNA" in text or "AMEN" in text or "AB " in text or "BANQUE" in text or "BANK" in text or "SUB" in text:
        return "BANCAIRE", "SECTOR_INFERRED_FROM_NAME"
    return "UNKNOWN_SECTOR", "UNKNOWN_SECTOR"


def build_sector_final_premiums_daily(corporate_curves_clean: pd.DataFrame, sovereign_curves: pd.DataFrame) -> pd.DataFrame:
    """Construit la prime finale sectorielle daily, style Tunisie Clearing.

    La prime est unique par date et secteur. Elle correspond à la médiane des
    spreads instrument-level : ytm corporate - ZC souverain interpolé à la
    maturité résiduelle. Elle n'est ni bucketisée ni interpolée par maturité.
    """

    if corporate_curves_clean.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "sector",
                "final_sector_spread_decimal",
                "n_instruments",
                "median_spread_decimal",
                "mean_spread_decimal",
                "std_spread_decimal",
                "min_spread_decimal",
                "max_spread_decimal",
                "quality_flag",
                "method",
            ]
        )

    sovereign_by_date = {
        pd.Timestamp(date): group.sort_values("maturity_years")[["maturity_years", "zero_coupon_rate_decimal"]].to_numpy(float)
        for date, group in sovereign_curves.groupby("curve_date", sort=False)
    }
    rows: list[dict[str, object]] = []
    for _, row in corporate_curves_clean.iterrows():
        sector, sector_flag = map_corporate_sector(row)
        curve = sovereign_by_date.get(pd.Timestamp(row["curve_date"]))
        maturity = pd.to_numeric(row.get("maturity_years"), errors="coerce")
        zc = np.nan
        zc_flag = "OK"
        if curve is None or len(curve) < 2 or pd.isna(maturity):
            zc_flag = "ZC_CURVE_MISSING"
        else:
            maturities = curve[:, 0]
            rates = curve[:, 1]
            if float(maturity) < maturities.min() or float(maturity) > maturities.max():
                zc_flag = "ZC_MATURITY_OUT_OF_RANGE"
            else:
                zc = float(np.interp(float(maturity), maturities, rates))
        ytm = pd.to_numeric(row.get("ytm_decimal"), errors="coerce")
        dirty = pd.to_numeric(row.get("dirty_price"), errors="coerce") if "dirty_price" in row else np.nan
        clean = pd.to_numeric(row.get("clean_price"), errors="coerce") if "clean_price" in row else np.nan
        spread = float(ytm - zc) if pd.notna(ytm) and pd.notna(zc) else np.nan
        invalid_reasons: list[str] = []
        if sector == "UNKNOWN_SECTOR":
            invalid_reasons.append("UNKNOWN_SECTOR")
        if not (pd.notna(ytm) and 0.0 < float(ytm) < 0.25):
            invalid_reasons.append("YTM_OUT_OF_RANGE")
        if not (pd.notna(maturity) and float(maturity) > 0.05):
            invalid_reasons.append("MATURITY_TOO_SHORT")
        if "dirty_price" in row and not (pd.notna(dirty) and float(dirty) > 0):
            invalid_reasons.append("NON_POSITIVE_DIRTY_PRICE")
        if "clean_price" in row and pd.notna(clean) and float(clean) < 0:
            invalid_reasons.append("NEGATIVE_CLEAN_PRICE")
        if zc_flag != "OK":
            invalid_reasons.append(zc_flag)
        if not (pd.notna(spread) and 0.0 <= spread <= 0.05):
            invalid_reasons.append("SPREAD_OUT_OF_RANGE")
        rows.append(
            {
                "date": pd.Timestamp(row["curve_date"]),
                "sector": sector,
                "isin": row.get("isin"),
                "label": row.get("label"),
                "maturity_years": maturity,
                "ytm_decimal": ytm,
                "sovereign_zc_decimal_interpolated": zc,
                "spread_decimal": spread,
                "is_valid_spread": len(invalid_reasons) == 0,
                "spread_quality_flag": "OK" if len(invalid_reasons) == 0 else ";".join(dict.fromkeys(invalid_reasons)),
                "sector_mapping_flag": sector_flag,
                "volume": pd.to_numeric(row.get("volume", np.nan), errors="coerce"),
            }
        )
    instrument_spreads = pd.DataFrame(rows)
    if instrument_spreads.empty:
        return pd.DataFrame()
    valid = instrument_spreads.loc[instrument_spreads["is_valid_spread"]].copy()
    if valid.empty:
        out = (
            instrument_spreads.groupby(["date", "sector"], as_index=False)
            .size()
            .rename(columns={"size": "n_instruments"})
        )
        out["final_sector_spread_decimal"] = np.nan
        out["median_spread_decimal"] = np.nan
        out["mean_spread_decimal"] = np.nan
        out["std_spread_decimal"] = np.nan
        out["min_spread_decimal"] = np.nan
        out["max_spread_decimal"] = np.nan
        out["quality_flag"] = "NO_VALID_SPREAD"
    else:
        out = (
            valid.groupby(["date", "sector"], as_index=False)
            .agg(
                final_sector_spread_decimal=("spread_decimal", "median"),
                n_instruments=("spread_decimal", "count"),
                median_spread_decimal=("spread_decimal", "median"),
                mean_spread_decimal=("spread_decimal", "mean"),
                std_spread_decimal=("spread_decimal", "std"),
                min_spread_decimal=("spread_decimal", "min"),
                max_spread_decimal=("spread_decimal", "max"),
            )
            .sort_values(["date", "sector"])
        )
        raw_counts = instrument_spreads.groupby(["date", "sector"]).size().rename("raw_count")
        out = out.merge(raw_counts, on=["date", "sector"], how="left")
        out["quality_flag"] = np.select(
            [
                out["n_instruments"].lt(3),
                out["raw_count"].gt(out["n_instruments"]),
            ],
            ["LOW_OBSERVATION_COUNT", "OUTLIERS_REMOVED"],
            default="OK",
        )
        out = out.drop(columns=["raw_count"])
    out["method"] = "ROBUST_MEDIAN_SECTOR_PREMIUM_APPROXIMATION"
    out.attrs["instrument_spreads"] = instrument_spreads
    return out[
        [
            "date",
            "sector",
            "final_sector_spread_decimal",
            "n_instruments",
            "median_spread_decimal",
            "mean_spread_decimal",
            "std_spread_decimal",
            "min_spread_decimal",
            "max_spread_decimal",
            "quality_flag",
            "method",
        ]
    ].reset_index(drop=True)


def load_official_sector_spreads_2025(path: str | Path) -> pd.DataFrame:
    """Charge les primes sectorielles officielles extraites des bulletins TC 2025."""

    path = Path(path)
    columns = [
        "date",
        "sector",
        "official_sector_spread_decimal",
        "official_sector_spread_pct",
        "bulletin_date",
        "source_file",
        "volume_mdt",
        "n_operations",
        "quality_flag",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    xls = pd.ExcelFile(path)
    sheet_name = "spreads_long" if "spreads_long" in xls.sheet_names else xls.sheet_names[0]
    raw = standardize_columns(pd.read_excel(path, sheet_name=sheet_name))
    if raw.empty:
        return pd.DataFrame(columns=columns)

    date_col = next((c for c in ["spread_reference_date", "reference_date", "date", "market_date"] if c in raw.columns), None)
    bulletin_col = next((c for c in ["bulletin_date", "publication_date", "source_date"] if c in raw.columns), None)
    sector_col = next((c for c in ["sector", "secteur", "sector_name"] if c in raw.columns), None)
    spread_col = next(
        (
            c
            for c in [
                "spread_decimal",
                "official_sector_spread_decimal",
                "prime_finale",
                "prime_finale_decimal",
                "spread",
                "prime",
                "spread_pct",
                "spread_bps",
            ]
            if c in raw.columns
        ),
        None,
    )
    if date_col is None or sector_col is None or spread_col is None:
        raise ValueError("Fichier spreads TC: colonnes date/sector/spread introuvables.")

    df = raw.copy()
    if "bond_type" in df.columns:
        final_mask = df["bond_type"].astype(str).str.upper().str.contains("FINAL_SECTOR|PRIME_FINALE|FINAL", na=False)
        if final_mask.any():
            df = df.loc[final_mask].copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["bulletin_date"] = pd.to_datetime(df[bulletin_col], errors="coerce") if bulletin_col else df["date"]
    df["sector"] = df[sector_col].map(_normalize_sector_label)
    if spread_col == "spread_bps":
        df["official_sector_spread_decimal"] = pd.to_numeric(df[spread_col], errors="coerce") / 10_000.0
    else:
        df["official_sector_spread_decimal"] = df[spread_col].map(_parse_spread_decimal)
    df["official_sector_spread_pct"] = df["official_sector_spread_decimal"] * 100.0
    df["source_file"] = df["source_file"] if "source_file" in df.columns else ""
    df["volume_mdt"] = pd.to_numeric(df["volume_mdt"], errors="coerce") if "volume_mdt" in df.columns else np.nan
    df["n_operations"] = pd.to_numeric(df["n_operations"], errors="coerce") if "n_operations" in df.columns else np.nan
    df["quality_flag"] = np.select(
        [
            df["date"].isna(),
            df["sector"].eq("UNKNOWN_SECTOR"),
            df["official_sector_spread_decimal"].isna(),
            df["official_sector_spread_decimal"].lt(0),
            df["official_sector_spread_decimal"].gt(0.05),
        ],
        ["INVALID_DATE", "UNKNOWN_SECTOR", "MISSING_SPREAD", "NEGATIVE_SPREAD", "SPREAD_GT_5PCT"],
        default="OK",
    )
    out = df.loc[df["quality_flag"].eq("OK"), columns].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = (
        out.sort_values(["date", "sector", "bulletin_date"])
        .groupby(["date", "sector"], as_index=False)
        .agg(
            official_sector_spread_decimal=("official_sector_spread_decimal", "last"),
            official_sector_spread_pct=("official_sector_spread_pct", "last"),
            bulletin_date=("bulletin_date", "last"),
            source_file=("source_file", "last"),
            volume_mdt=("volume_mdt", "sum"),
            n_operations=("n_operations", "sum"),
            quality_flag=("quality_flag", "last"),
        )
    )
    return out[columns]


def build_official_sector_premiums_daily(
    official_spreads: pd.DataFrame,
    estimated_premiums: pd.DataFrame,
    valuation_dates: pd.Series | pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit la prime finale secteur/date avec priorite aux bulletins TC."""

    sectors = ["BANCAIRE", "LEASING", "MICROFINANCE"]
    dates = pd.DatetimeIndex(pd.to_datetime(valuation_dates)).sort_values().unique()
    official = official_spreads.copy()
    if not official.empty:
        official["date"] = pd.to_datetime(official["date"], errors="coerce")
        official["sector"] = official["sector"].map(_normalize_sector_label)
    estimated = estimated_premiums.copy()
    if not estimated.empty:
        estimated["date"] = pd.to_datetime(estimated["date"], errors="coerce")
        estimated["sector"] = estimated["sector"].map(_normalize_sector_label)
        estimated = estimated.rename(columns={"final_sector_spread_decimal": "estimated_sector_spread_decimal"})

    rows: list[dict[str, object]] = []
    for date in dates:
        for sector in sectors:
            exact = official.loc[official["date"].eq(date) & official["sector"].eq(sector)].tail(1) if not official.empty else pd.DataFrame()
            past = official.loc[official["date"].lt(date) & official["sector"].eq(sector)].sort_values("date") if not official.empty else pd.DataFrame()
            est = estimated.loc[estimated["date"].eq(date) & estimated["sector"].eq(sector)].tail(1) if not estimated.empty else pd.DataFrame()
            official_value = np.nan
            bulletin_date = pd.NaT
            source_file = ""
            days_since = np.nan
            n_official = int(official.loc[official["sector"].eq(sector)].shape[0]) if not official.empty else 0
            if not exact.empty:
                row = exact.iloc[-1]
                final = float(row["official_sector_spread_decimal"])
                official_value = final
                bulletin_date = row.get("bulletin_date", pd.NaT)
                source_file = row.get("source_file", "")
                source = "OFFICIAL_TC_BULLETIN"
                days_since = 0
                quality = "OK_OFFICIAL"
                method = "OFFICIAL_TC_BULLETIN_SECTOR_PREMIUM"
            elif not past.empty:
                row = past.iloc[-1]
                final = float(row["official_sector_spread_decimal"])
                official_value = final
                bulletin_date = row.get("bulletin_date", pd.NaT)
                source_file = row.get("source_file", "")
                source = "OFFICIAL_TC_BULLETIN_FORWARD_FILLED"
                days_since = int((pd.Timestamp(date) - pd.Timestamp(row["date"])).days)
                quality = "OK_FORWARD_FILLED" if days_since <= 10 else "STALE_OFFICIAL_SPREAD"
                method = "OFFICIAL_TC_BULLETIN_SECTOR_PREMIUM"
            elif not est.empty:
                final = float(est.iloc[-1]["estimated_sector_spread_decimal"])
                source = "ROBUST_MEDIAN_FALLBACK"
                quality = "MEDIAN_FALLBACK_USED"
                method = "ROBUST_MEDIAN_FALLBACK"
            else:
                final = np.nan
                source = "MISSING_SPREAD"
                quality = "MISSING_SPREAD_FAILED"
                method = "ROBUST_MEDIAN_FALLBACK"
            estimated_value = float(est.iloc[-1]["estimated_sector_spread_decimal"]) if not est.empty else np.nan
            if pd.isna(final):
                quality = "MISSING_SPREAD_FAILED"
            elif final < 0:
                quality = "NEGATIVE_SPREAD_FAILED"
            elif final > 0.05:
                quality = "SPREAD_GT_5PCT_FAILED"
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "sector": sector,
                    "final_sector_spread_decimal": final,
                    "final_sector_spread_pct": final * 100 if pd.notna(final) else np.nan,
                    "official_sector_spread_decimal": official_value,
                    "estimated_sector_spread_decimal": estimated_value,
                    "spread_source": source,
                    "days_since_last_official_spread": days_since,
                    "n_official_observations_available": n_official,
                    "bulletin_date": bulletin_date,
                    "source_file": source_file,
                    "quality_flag": quality,
                    "method": method,
                }
            )
    out = pd.DataFrame(rows)
    check = out[
        [
            "date",
            "sector",
            "final_sector_spread_decimal",
            "spread_source",
            "days_since_last_official_spread",
            "quality_flag",
        ]
    ].copy()
    return out, check


def get_sector_spread(
    sector_spreads_df: pd.DataFrame,
    curve_date: pd.Timestamp,
    sector: str,
    maturity_years: float,
    allow_last_available: bool = False,
    allow_nearest_maturity: bool = False,
) -> tuple[float, str]:
    """Retourne un spread sectoriel interpolé sans extrapolation silencieuse."""

    if pd.isna(maturity_years) or maturity_years <= 0:
        return np.nan, "SPREAD_MISSING"
    date = pd.Timestamp(curve_date)
    sector_raw = str(sector).upper().strip()
    sector = {
        "BANKING": "BANCAIRE",
        "BANK": "BANCAIRE",
        "BANCAIRE": "BANCAIRE",
        "LEASING": "LEASING",
        "MICROFINANCE": "MICROFINANCE",
    }.get(sector_raw, sector_raw)
    one = sector_spreads_df.loc[sector_spreads_df["curve_date"].eq(date) & sector_spreads_df["sector"].eq(sector)]
    flag = "OK"
    if one.empty and allow_last_available:
        past = sector_spreads_df.loc[(sector_spreads_df["curve_date"] < date) & sector_spreads_df["sector"].eq(sector)]
        if not past.empty:
            one = past.loc[past["curve_date"].eq(past["curve_date"].max())]
            flag = "CORPORATE_CURVE_LAST_AVAILABLE_USED"
    if one.empty:
        return np.nan, "SPREAD_SECTOR_MISSING"
    maturities = one["maturity_years"].to_numpy(float)
    spreads = one["spread_decimal"].to_numpy(float)
    order = np.argsort(maturities)
    maturities, spreads = maturities[order], spreads[order]
    if maturity_years < maturities.min() or maturity_years > maturities.max():
        if allow_nearest_maturity:
            idx = int(np.argmin(np.abs(maturities - maturity_years)))
            return float(spreads[idx]), "SPREAD_NEAREST_MATURITY_USED" if flag == "OK" else f"{flag};SPREAD_NEAREST_MATURITY_USED"
        return np.nan, "SPREAD_MISSING"
    return float(np.interp(maturity_years, maturities, spreads)), flag


def load_zc_curves(path: Path, start: str = ANALYSIS_START_DATE, end: str = ANALYSIS_END_DATE) -> pd.DataFrame:
    """Charge les courbes ZC et filtre la fenêtre d'analyse."""

    if not path.exists():
        raise FileNotFoundError(path)
    df = standardize_columns(pd.read_excel(path, sheet_name="zc_weekly_standardized"))
    df = df.rename(columns={"requested_date": "date", "curve_date_used": "curve_date", "zc_actuarial_decimal": "zero_rate"})
    validate_required_columns(df, ["date", "maturity_years", "zero_rate"], "Courbes zéro-coupon")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["maturity_years"] = pd.to_numeric(df["maturity_years"], errors="coerce")
    df["zero_rate"] = coerce_rate_to_decimal(df["zero_rate"])
    df = filter_date_window(df.dropna(subset=["date", "maturity_years", "zero_rate"]), "date", start, end)
    counts = df.groupby("date")["maturity_years"].nunique()
    if (counts < 2).any():
        raise ValueError("Courbe zéro-coupon insuffisante sur au moins une date.")
    return df.sort_values(["date", "maturity_years"]).reset_index(drop=True)


def interpolate_zero_rate(curve_df: pd.DataFrame, valuation_date: pd.Timestamp, tau_years: float) -> tuple[float, str]:
    """Interpôle sans extrapoler. Retourne taux et flag."""

    if pd.isna(tau_years) or tau_years <= 0:
        return np.nan, "ZC_MATURITY_OUT_OF_RANGE"
    date = pd.Timestamp(valuation_date)
    curve = curve_df.loc[curve_df["date"].eq(date), ["maturity_years", "zero_rate"]].dropna()
    if curve.empty:
        return np.nan, "DATA_MISSING"
    maturities = curve["maturity_years"].to_numpy(float)
    rates = curve["zero_rate"].to_numpy(float)
    order = np.argsort(maturities)
    maturities, rates = maturities[order], rates[order]
    if tau_years < maturities.min() or tau_years > maturities.max():
        return np.nan, "ZC_MATURITY_OUT_OF_RANGE"
    return float(np.interp(tau_years, maturities, rates)), "OK"
