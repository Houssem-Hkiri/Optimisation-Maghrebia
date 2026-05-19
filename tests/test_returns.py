import numpy as np
import pandas as pd

from maghrebia_quant.config import ANALYSIS_END_DATE, ANALYSIS_START_DATE
from maghrebia_quant.equity_returns import apply_corporate_actions, build_daily_equity_prices, compute_daily_equity_returns, compute_weekly_equity_returns
from maghrebia_quant.risk_metrics import build_weekly_returns_model, compute_asset_metrics


def test_simple_return():
    prices = pd.DataFrame({"A": [100.0, 105.0, 102.9]}, index=pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-17"]))
    returns = compute_weekly_equity_returns(prices)
    assert np.isclose(returns.loc[pd.Timestamp("2025-01-10"), "A"], 0.05)
    assert np.isclose(returns.loc[pd.Timestamp("2025-01-17"), "A"], -0.02)


def test_corporate_action_adjusts_pre_event_prices_only():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-29", "2025-04-30"]),
            "bvmt_code": ["TN0001600154", "TN0001600154"],
            "asset_name": ["ATTIJARI BANK", "ATTIJARI BANK"],
            "asset_name_norm": ["ATTIJARI BANK", "ATTIJARI BANK"],
            "close_raw": [70.25, 61.0],
        }
    )
    adjusted, audit = apply_corporate_actions(
        prices,
        [
            {
                "asset_id": "ATTIJARI_BANK",
                "bvmt_code": "TN0001600154",
                "asset_name_exact": "ATTIJARI BANK",
                "effective_date": "2025-04-30",
                "price_adjustment_factor": 21 / 25,
                "comment": "test",
            }
        ],
    )
    assert np.isclose(adjusted.loc[0, "close_adjusted"], 70.25 * 21 / 25)
    assert np.isclose(adjusted.loc[1, "close_adjusted"], 61.0)
    assert set(audit["flag"]) == {"CORPORATE_ACTION_ADJUSTED"}


def test_geometric_annualized_return():
    idx = pd.date_range("2025-01-03", periods=52, freq="W-FRI")
    returns = pd.DataFrame({"A": [0.01] * 52}, index=idx)
    metrics = compute_asset_metrics(returns, annual_rf=0.0)
    expected = (1.01**252) - 1
    assert np.isclose(metrics.loc[0, "annualized_return_normalized"], expected)


def test_no_dates_outside_analysis_window():
    idx = pd.to_datetime(["2024-12-27", "2025-01-03", "2025-12-26", "2026-01-02"])
    returns = pd.DataFrame({"A": [0.1, 0.1, 0.1, 0.1]}, index=idx)
    model, _ = build_weekly_returns_model(returns, min_observations=1)
    assert model.index.min() >= pd.Timestamp(ANALYSIS_START_DATE)
    assert model.index.max() <= pd.Timestamp(ANALYSIS_END_DATE)


def test_daily_equity_returns_do_not_convert_missing_prices_to_minus_100():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "bvmt_code": ["AAA", "AAA", "AAA"],
            "asset_name": ["TEST", "TEST", "TEST"],
            "asset_name_norm": ["TEST", "TEST", "TEST"],
            "close_adjusted": [100.0, np.nan, 101.0],
        }
    )
    portfolio = pd.DataFrame({"asset_type": ["listed_equity"], "asset_name": ["TEST"], "asset_id": ["TEST"]})
    matrix = build_daily_equity_prices(prices, portfolio)
    returns = compute_daily_equity_returns(matrix)
    assert returns.min().min() > -0.99
