import numpy as np
import pandas as pd

from maghrebia_quant.risk_metrics import (
    build_complete_daily_returns_model,
    compute_asset_metrics,
    compute_clean_covariance_matrix,
)


def test_loss_var_positive():
    idx = pd.date_range("2025-01-03", periods=52, freq="W-FRI")
    returns = pd.DataFrame({"A": np.r_[np.repeat(-0.02, 10), np.repeat(0.01, 42)]}, index=idx)
    metrics = compute_asset_metrics(returns, annual_rf=0.02)
    assert metrics.loc[0, "loss_var_95"] > 0


def test_loss_cvar_positive():
    idx = pd.date_range("2025-01-03", periods=52, freq="W-FRI")
    returns = pd.DataFrame({"A": np.r_[np.repeat(-0.02, 10), np.repeat(0.01, 42)]}, index=idx)
    metrics = compute_asset_metrics(returns, annual_rf=0.02)
    assert metrics.loc[0, "loss_cvar_95"] > 0


def test_covariance_without_nan():
    idx = pd.date_range("2025-01-03", periods=52, freq="W-FRI")
    returns = pd.DataFrame({"A": np.linspace(-0.01, 0.02, 52), "B": np.linspace(0.02, -0.01, 52)}, index=idx)
    cov = compute_clean_covariance_matrix(returns)
    assert not cov.isna().any().any()


def test_asset_metrics_flags_high_kurtosis():
    idx = pd.date_range("2025-01-03", periods=80, freq="B")
    values = np.r_[np.repeat(0.001, 79), -0.20]
    metrics = compute_asset_metrics(pd.DataFrame({"A": values}, index=idx), annual_rf=0.02)
    assert "HIGH_KURTOSIS" in metrics.loc[0, "quality_flag"] or "EXTREME_KURTOSIS" in metrics.loc[0, "quality_flag"]


def test_ledoit_wolf_covariance_is_well_conditioned_for_collinear_assets():
    idx = pd.date_range("2025-01-03", periods=120, freq="B")
    base = np.linspace(-0.002, 0.002, len(idx))
    returns = pd.DataFrame({"A": base, "B": base * 1.000001 + 1e-9, "C": -base}, index=idx)
    cov = compute_clean_covariance_matrix(returns, method="ledoit_wolf")
    assert np.linalg.cond(cov.to_numpy()) < 10000


def test_complete_returns_model_keeps_required_assets():
    idx = pd.date_range("2025-01-02", periods=3, freq="B")
    diagnostic = pd.DataFrame({"A": [0.01, np.nan, 0.02]}, index=idx)
    model, audit = build_complete_daily_returns_model(diagnostic, ["A", "B"])
    assert list(model.columns) == ["A", "B"]
    assert model["B"].isna().all()
    assert audit.loc[audit["asset_id"].eq("B"), "status"].iloc[0] == "MISSING_REQUIRED_RETURN_SERIES"
