"""Métriques de risque, taux sans risque, covariance et portefeuille."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_END_DATE,
    ANALYSIS_START_DATE,
    MIN_OBSERVATIONS_OPTIMIZATION,
    PERIODS_PER_YEAR_DAILY,
)
from .loaders import filter_date_window, standardize_columns
from .validation import max_drawdown


def load_bct_short_rate_daily(
    bct_path: Path,
    target_dates: pd.Series | pd.DatetimeIndex,
    periods_per_year: int = PERIODS_PER_YEAR_DAILY,
) -> pd.DataFrame:
    """Charge le taux court BCT/TAO et l'aligne sur le calendrier de rendement.

    Le taux est observé à fréquence hebdomadaire dans la source disponible. Il
    est propagé jusqu'à la prochaine observation BCT afin de produire un taux
    périodique daté pour chaque journée finale.
    """

    target = pd.DatetimeIndex(pd.to_datetime(target_dates)).sort_values()
    columns = ["date", "rf_annual_decimal", "rf_periodic_decimal", "source"]
    if target.empty:
        return pd.DataFrame(columns=columns)
    if not bct_path.exists():
        return pd.DataFrame(
            {
                "date": target,
                "rf_annual_decimal": np.nan,
                "rf_periodic_decimal": np.nan,
                "source": "BCT_SHORT_RATE_FILE_MISSING",
            }
        )
    df = standardize_columns(pd.read_excel(bct_path, sheet_name="short_rates"))
    if "date" not in df.columns:
        raise ValueError("Fichier BCT: colonne date manquante.")
    rate_col = "tao_decimal" if "tao_decimal" in df.columns else None
    if rate_col is None:
        candidates = [c for c in df.columns if "decimal" in c or "rate" in c or "taux" in c]
        if not candidates:
            raise ValueError("Fichier BCT: aucune colonne de taux exploitable.")
        rate_col = candidates[0]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df = df.dropna(subset=["date", rate_col]).sort_values("date")
    aligned = pd.DataFrame({"date": target})
    aligned = pd.merge_asof(aligned, df[["date", rate_col]], on="date", direction="backward")
    if aligned[rate_col].isna().any():
        aligned = pd.merge_asof(aligned[["date"]], df[["date", rate_col]], on="date", direction="forward")
    aligned = aligned.rename(columns={rate_col: "rf_annual_decimal"})
    aligned["rf_periodic_decimal"] = (1.0 + aligned["rf_annual_decimal"]) ** (1.0 / periods_per_year) - 1.0
    aligned["source"] = f"BCT_SHORT_RATE_{rate_col}"
    return aligned[columns]


def _assign_quality_flag(row: dict[str, float | int | object]) -> str:
    flags: list[str] = []
    observations = row.get("observations", 0)
    if pd.notna(observations) and int(observations) < MIN_OBSERVATIONS_OPTIMIZATION:
        flags.append("SHORT_SERIES_WARNING")
    kurtosis = row.get("kurtosis", np.nan)
    if pd.notna(kurtosis):
        if float(kurtosis) > 20:
            flags.append("EXTREME_KURTOSIS")
        elif float(kurtosis) > 10:
            flags.append("HIGH_KURTOSIS")
    ann_vol = row.get("annualized_volatility", np.nan)
    if pd.notna(ann_vol) and float(ann_vol) > 1.0:
        flags.append("EXTREME_VOLATILITY")
    skewness = row.get("skewness", np.nan)
    if pd.notna(skewness) and abs(float(skewness)) > 5:
        flags.append("EXTREME_SKEWNESS")
    drawdown = row.get("max_drawdown", np.nan)
    if pd.notna(drawdown) and float(drawdown) < -0.50:
        flags.append("EXTREME_DRAWDOWN")
    min_return = row.get("min_return", np.nan)
    if pd.notna(min_return) and abs(float(min_return)) > 0.20:
        flags.append("SUSPICIOUS_MIN_RETURN")
    return ";".join(dict.fromkeys(flags)) if flags else "RETURN_SERIES_OK"


def align_daily_returns_diagnostic(*matrices: pd.DataFrame) -> pd.DataFrame:
    """Assemble les séries journalières disponibles sur la fenêtre, avec NaN possibles."""

    frames = [m for m in matrices if isinstance(m, pd.DataFrame) and not m.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1, join="outer").sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    start, end = pd.Timestamp(ANALYSIS_START_DATE), pd.Timestamp(ANALYSIS_END_DATE)
    out = out.loc[(out.index >= start) & (out.index <= end)]
    return out.dropna(axis=1, how="all")


def build_daily_returns_model(daily_returns_diagnostic: pd.DataFrame, min_observations: int = MIN_OBSERVATIONS_OPTIMIZATION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit la matrice journalière finale, sans colonnes vides ni dates hors fenêtre."""

    diag = align_daily_returns_diagnostic(daily_returns_diagnostic)
    records: list[dict[str, object]] = []
    keep = []
    for col in diag.columns:
        obs = int(diag[col].count())
        if obs >= min_observations:
            keep.append(col)
            reason = "included"
        else:
            reason = "SHORT_SERIES_WARNING"
        records.append({"asset_id": col, "observations": obs, "decision": reason})
    model = diag[keep].dropna(how="any").copy()
    model = model.loc[(model.index >= pd.Timestamp(ANALYSIS_START_DATE)) & (model.index <= pd.Timestamp(ANALYSIS_END_DATE))]
    return model, pd.DataFrame(records)


def build_complete_daily_returns_model(
    daily_returns_diagnostic: pd.DataFrame,
    required_assets: list[str] | pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit une matrice finale contenant tous les actifs requis.

    Les valeurs manquantes isolées sont traitées par remplissage contrôlé à 0,
    ce qui signifie "pas de variation de prix exploitable ce jour" et doit être
    documenté dans l'audit de couverture.
    """

    diag = align_daily_returns_diagnostic(daily_returns_diagnostic)
    required = [str(asset) for asset in required_assets]
    records: list[dict[str, object]] = []
    for asset in required:
        if asset not in diag.columns:
            diag[asset] = np.nan
            records.append({"asset_id": asset, "observations": 0, "missing_returns": np.nan, "status": "MISSING_REQUIRED_RETURN_SERIES"})
        else:
            observations = int(diag[asset].count())
            missing = int(diag[asset].isna().sum())
            records.append({"asset_id": asset, "observations": observations, "missing_returns": missing, "status": "OK" if observations else "MISSING_REQUIRED_RETURN_SERIES"})
    model = diag[required].sort_index().copy()
    model = model.loc[(model.index >= pd.Timestamp(ANALYSIS_START_DATE)) & (model.index <= pd.Timestamp(ANALYSIS_END_DATE))]
    model = model.dropna(how="all")
    return model, pd.DataFrame(records)


def align_weekly_returns_diagnostic(*matrices: pd.DataFrame) -> pd.DataFrame:
    """Alias de compatibilité ; utiliser align_daily_returns_diagnostic pour le notebook 2025."""

    return align_daily_returns_diagnostic(*matrices)


def build_weekly_returns_model(weekly_returns_diagnostic: pd.DataFrame, min_observations: int = MIN_OBSERVATIONS_OPTIMIZATION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alias de compatibilité ; utiliser build_daily_returns_model pour le notebook 2025."""

    return build_daily_returns_model(weekly_returns_diagnostic, min_observations=min_observations)


def _downside_deviation(returns: pd.Series) -> float:
    downside = returns.dropna().clip(upper=0.0)
    return float(np.sqrt((downside**2).mean()) * np.sqrt(PERIODS_PER_YEAR_DAILY)) if not downside.empty else np.nan


def compute_asset_metrics(returns_df: pd.DataFrame, annual_rf: float) -> pd.DataFrame:
    """Calcule les métriques par actif avec Sharpe basé sur rf non nul."""

    rows: list[dict[str, object]] = []
    for asset_id in returns_df.columns:
        r = returns_df[asset_id].dropna()
        n = len(r)
        cumulative = float((1 + r).prod() - 1) if n else np.nan
        annualized_return = float(r.mean() * PERIODS_PER_YEAR_DAILY) if n else np.nan
        annualized_return_normalized = float((1 + cumulative) ** (PERIODS_PER_YEAR_DAILY / n) - 1) if n else np.nan
        weekly_vol = float(r.std(ddof=1)) if n > 1 else np.nan
        ann_vol = weekly_vol * np.sqrt(PERIODS_PER_YEAR_DAILY) if pd.notna(weekly_vol) else np.nan
        return_var = float(r.quantile(0.05)) if n else np.nan
        return_cvar = float(r[r <= return_var].mean()) if n and pd.notna(return_var) else np.nan
        losses = -r
        loss_var = max(0.0, float(losses.quantile(0.95))) if n else np.nan
        loss_cvar = float(losses[losses >= loss_var].mean()) if n and pd.notna(loss_var) else np.nan
        skewness = float(r.skew()) if n > 2 else np.nan
        kurtosis = float(r.kurtosis()) if n > 3 else np.nan
        min_return = float(r.min()) if n else np.nan
        max_return = float(r.max()) if n else np.nan
        drawdown = max_drawdown(r)
        flag = _assign_quality_flag(
            {
                "observations": n,
                "annualized_volatility": ann_vol,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "min_return": min_return,
                "max_drawdown": drawdown,
            }
        )
        rows.append(
            {
                "asset_id": asset_id,
                "observations": n,
                "start_date": r.index.min() if n else pd.NaT,
                "end_date": r.index.max() if n else pd.NaT,
                "cumulative_return": cumulative,
                "annualized_return": annualized_return,
                "annualized_return_arithmetic": annualized_return,
                "annualized_return_geometric": annualized_return_normalized,
                "annualized_return_normalized": annualized_return_normalized,
                "weekly_mean": float(r.mean()) if n else np.nan,
                "weekly_volatility": weekly_vol,
                "daily_volatility": weekly_vol,
                "annualized_volatility": ann_vol,
                "weekly_variance": float(r.var(ddof=1)) if n > 1 else np.nan,
                "annualized_variance": ann_vol**2 if pd.notna(ann_vol) else np.nan,
                "downside_deviation": _downside_deviation(r),
                "sharpe_ratio": (annualized_return - annual_rf) / ann_vol if pd.notna(ann_vol) and ann_vol != 0 else np.nan,
                "raw_return_to_volatility": annualized_return / ann_vol if pd.notna(ann_vol) and ann_vol != 0 else np.nan,
                "return_var_95": return_var,
                "return_cvar_95": return_cvar,
                "loss_var_95": loss_var,
                "loss_cvar_95": loss_cvar,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "min_return": min_return,
                "max_return": max_return,
                "min_weekly_return": min_return,
                "max_weekly_return": max_return,
                "max_drawdown": drawdown,
                "quality_flag": flag,
            }
        )
    return pd.DataFrame(rows)


def compute_asset_metrics_with_daily_rf(
    returns_df: pd.DataFrame,
    rf_daily: pd.Series,
    rf_source: str,
    rf_maturity_used: float | int | None = None,
) -> pd.DataFrame:
    """Calcule les mÃ©triques par actif avec un taux sans risque journalier alignÃ©."""

    rf = pd.Series(rf_daily, dtype=float).rename("rf_daily_decimal")
    aligned_returns = returns_df.loc[returns_df.index.intersection(rf.dropna().index)].copy()
    rf = rf.reindex(aligned_returns.index)
    if rf.isna().any():
        raise ValueError("rf_daily contient des valeurs manquantes sur les dates de rendements.")
    annual_rf_mean = float((1.0 + rf).prod() ** (PERIODS_PER_YEAR_DAILY / len(rf)) - 1.0) if len(rf) else np.nan
    metrics = compute_asset_metrics(aligned_returns, annual_rf=annual_rf_mean)
    for idx, row in metrics.iterrows():
        asset_id = str(row["asset_id"])
        r = aligned_returns[asset_id].dropna()
        excess = r - rf.reindex(r.index)
        daily_vol = float(r.std(ddof=1)) if len(r) > 1 else np.nan
        excess_ann = float(excess.mean() * PERIODS_PER_YEAR_DAILY) if len(excess) else np.nan
        sharpe = excess.mean() / daily_vol * np.sqrt(PERIODS_PER_YEAR_DAILY) if pd.notna(daily_vol) and daily_vol != 0 else np.nan
        metrics.loc[idx, "rf_source"] = rf_source
        metrics.loc[idx, "rf_maturity_used"] = rf_maturity_used
        metrics.loc[idx, "rf_annual_mean"] = annual_rf_mean
        metrics.loc[idx, "excess_return_annualised"] = excess_ann
        metrics.loc[idx, "sharpe_rf_short_term"] = sharpe
        metrics.loc[idx, "sharpe_ratio"] = sharpe
    return metrics


def load_risk_free_rate(bct_path: Path | None = None, zc_curves: pd.DataFrame | None = None) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Charge le taux sans risque annuel et hebdomadaire."""

    flag = "OK"
    source = "missing"
    annual_rf = np.nan
    details = pd.DataFrame()
    if bct_path is not None and bct_path.exists():
        df = standardize_columns(pd.read_excel(bct_path, sheet_name="short_rates"))
        if {"date", "tao_decimal"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["tao_decimal"] = pd.to_numeric(df["tao_decimal"], errors="coerce")
            details = filter_date_window(df.dropna(subset=["date", "tao_decimal"]), "date")
            if not details.empty:
                annual_rf = float(details["tao_decimal"].mean())
                source = "bct_short_rates"
    if source == "missing" and zc_curves is not None and not zc_curves.empty:
        zc1 = zc_curves.iloc[(zc_curves["maturity_years"] - 1.0).abs().argsort()].head(len(zc_curves["date"].unique()))
        if not zc1.empty:
            annual_rf = float(zc1["zero_rate"].mean())
            source = "zero_coupon_1y_average"
    if source == "missing":
        flag = "RISK_FREE_RATE_MISSING"
    weekly_rf = (1 + annual_rf) ** (1 / PERIODS_PER_YEAR_DAILY) - 1 if pd.notna(annual_rf) else np.nan
    return {"annual_rf": annual_rf, "weekly_rf": weekly_rf, "source": source, "flag": flag}, details


def compute_clean_covariance_matrix(weekly_returns_model: pd.DataFrame, method: str = "ledoit_wolf") -> pd.DataFrame:
    """Calcule la covariance annualisee avec shrinkage ou clipping spectral."""

    if weekly_returns_model.isna().all(axis=None):
        raise ValueError("weekly_returns_model ne contient aucune donn?e.")
    if weekly_returns_model.isna().any().any():
        raise ValueError("weekly_returns_model contient des NaN.")
    if weekly_returns_model.index.max() > pd.Timestamp(ANALYSIS_END_DATE) or weekly_returns_model.index.min() < pd.Timestamp(ANALYSIS_START_DATE):
        raise ValueError("weekly_returns_model contient des dates hors fen?tre.")
    returns = weekly_returns_model.astype(float)
    cov_daily: pd.DataFrame
    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf

            estimator = LedoitWolf().fit(returns.to_numpy())
            cov_daily = pd.DataFrame(estimator.covariance_, index=returns.columns, columns=returns.columns)
        except Exception:
            method = "spectral_clip"
    if method == "spectral_clip":
        sample = returns.cov()
        values = ((sample + sample.T) / 2).to_numpy(float)
        eigenvalues, eigenvectors = np.linalg.eigh(values)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        cov_daily = pd.DataFrame(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T, index=returns.columns, columns=returns.columns)
    elif method == "sample":
        cov_daily = returns.cov()
    cov = cov_daily * PERIODS_PER_YEAR_DAILY
    cov = (cov + cov.T) / 2
    cov.attrs["method"] = method
    return cov


def validate_covariance_matrix(cov_matrix: pd.DataFrame, number_observations: int | None = None) -> dict[str, object]:
    """Valide symétrie, PSD et conditionnement."""

    values = cov_matrix.to_numpy(float)
    eig = np.linalg.eigvalsh(values)
    return {
        "min_eigenvalue": float(eig.min()),
        "is_psd": bool(eig.min() >= -1e-10),
        "condition_number": float(np.linalg.cond(values)) if values.size else np.nan,
        "number_assets": int(cov_matrix.shape[0]),
        "number_observations": int(number_observations) if number_observations is not None else np.nan,
    }


def nearest_psd_covariance(cov_matrix: pd.DataFrame, epsilon: float = 1e-10) -> pd.DataFrame:
    """Projette une covariance symétrique vers une matrice PSD simple."""

    values = ((cov_matrix + cov_matrix.T) / 2).to_numpy(float)
    eigvals, eigvecs = np.linalg.eigh(values)
    eigvals = np.maximum(eigvals, epsilon)
    adjusted = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return pd.DataFrame((adjusted + adjusted.T) / 2, index=cov_matrix.index, columns=cov_matrix.columns)


def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Corrélation Pearson."""

    return returns_df.corr()


def compute_constant_weight_current_portfolio_returns(returns_df: pd.DataFrame, weights: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Rendement du portefeuille actuel à pondérations constantes."""

    original = weights.astype(float) / weights.astype(float).sum()
    common = [c for c in returns_df.columns if c in original.index]
    used = original.loc[common].copy()
    if (used < -1e-12).any():
        raise ValueError("Poids négatif détecté.")
    used = used / used.sum()
    audit = pd.DataFrame({"asset_id": original.index, "original_weight": original.values})
    audit["used_weight"] = audit["asset_id"].map(used).fillna(0.0)
    audit["exclusion_reason"] = np.where(audit["asset_id"].isin(common), "", "excluded_from_weekly_returns_model")
    returns = returns_df[common].mul(used, axis=1).sum(axis=1)
    returns.name = "constant_weight_current_portfolio_returns"
    return returns, audit


def compute_portfolio_metrics(portfolio_returns: pd.Series, annual_rf: float) -> dict[str, float]:
    """Métriques du portefeuille courant."""

    return compute_asset_metrics(portfolio_returns.to_frame("portfolio"), annual_rf).iloc[0].drop("asset_id").to_dict()


def compute_portfolio_metrics_with_daily_rf(
    portfolio_returns: pd.Series,
    rf_daily: pd.Series,
    rf_source: str,
    rf_maturity_used: float | int | None = None,
) -> dict[str, float | str]:
    """MÃ©triques portefeuille avec Sharpe basÃ© sur un taux sans risque court terme."""

    metrics = compute_asset_metrics_with_daily_rf(
        portfolio_returns.to_frame("portfolio"),
        rf_daily=rf_daily,
        rf_source=rf_source,
        rf_maturity_used=rf_maturity_used,
    ).iloc[0].drop("asset_id").to_dict()
    metrics["portfolio_excess_return_annualised"] = metrics.get("excess_return_annualised", np.nan)
    return metrics


def compute_risk_contribution(weights: pd.Series, covariance_matrix: pd.DataFrame) -> pd.DataFrame:
    """Contribution au risque variance."""

    common = [c for c in covariance_matrix.columns if c in weights.index]
    w = weights.loc[common].astype(float)
    if (w < -1e-12).any():
        raise ValueError("Poids négatif détecté.")
    w = w / w.sum()
    sigma = covariance_matrix.loc[common, common]
    marginal = sigma.dot(w)
    portfolio_variance = float(w.T.dot(sigma).dot(w))
    risk_contrib = w * marginal / portfolio_variance if portfolio_variance > 0 else np.nan
    out = pd.DataFrame(
        {
            "asset_id": common,
            "weight": w.values,
            "marginal_contribution": marginal.values,
            "risk_contribution": risk_contrib.values,
        }
    )
    out["quality_flag"] = np.where(out["risk_contribution"] < 0, "NEGATIVE_RISK_CONTRIBUTION", "OK")
    return out


def coverage_table(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic de couverture des rendements."""

    total = len(returns_df)
    return pd.DataFrame(
        {
            "asset_id": returns_df.columns,
            "observations": returns_df.count().values,
            "coverage_rate": (returns_df.count() / total).values if total else np.nan,
            "missing_values": returns_df.isna().sum().values,
        }
    )
