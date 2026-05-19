import numpy as np
import pandas as pd
import pytest

from maghrebia_quant.corporate_bond_valuation import (
    build_corporate_returns_check,
    detect_dirty_price_scale,
    get_outstanding_principal_unit,
    scale_dirty_price_to_unit,
)


def test_dirty_price_base_1_multiplies_outstanding_principal():
    scale = detect_dirty_price_scale(pd.Series([1.02, 1.05, 0.99]))
    assert scale == "BASE_1"
    assert scale_dirty_price_to_unit(1.05, 60.0, scale) == pytest.approx(63.0)


def test_dirty_price_base_100_multiplies_outstanding_principal():
    scale = detect_dirty_price_scale(pd.Series([102.0, 105.0, 99.0]))
    assert scale == "BASE_100"
    assert scale_dirty_price_to_unit(105.0, 60.0, scale) == pytest.approx(63.0)


def test_outstanding_principal_from_cashflow_schedule():
    row = pd.Series(
        {
            "cashflow_schedule": pd.DataFrame(
                {
                    "cashflow_date": pd.to_datetime(["2025-03-27", "2026-03-27"]),
                    "capital_begin": [80.0, 60.0],
                }
            )
        }
    )
    value, flag = get_outstanding_principal_unit(row, pd.Timestamp("2025-12-31"))
    assert value == pytest.approx(60.0)
    assert flag == "OK"


def test_missing_outstanding_principal_is_flagged():
    value, flag = get_outstanding_principal_unit(pd.Series({}), pd.Timestamp("2025-12-31"))
    assert np.isnan(value)
    assert flag == "MISSING_OUTSTANDING_PRINCIPAL"


def test_corporate_returns_check_flags_unexplained_five_percent_move():
    returns = pd.DataFrame({"BOND": [0.001, -0.051, 0.002]}, index=pd.date_range("2025-01-02", periods=3))
    check = build_corporate_returns_check(returns)
    assert "SUSPICIOUS_CORPORATE_DAILY_RETURN" in check.loc[0, "suspicious_return_flag"]
