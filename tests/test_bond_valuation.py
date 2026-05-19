import numpy as np
import pandas as pd

from maghrebia_quant.cashflows import cashflow_paid_between, discount_cashflows
from maghrebia_quant.curves import interpolate_zero_rate


def test_discount_single_cashflow():
    valuation_date = pd.Timestamp("2025-01-03")
    cashflows = pd.DataFrame({"cashflow_date": [pd.Timestamp("2026-01-03")], "cashflow": [110.0]})
    price, _ = discount_cashflows(cashflows, valuation_date, np.array([0.10]))
    assert np.isclose(price, 100.0)


def test_cashflow_paid_between_dates():
    cashflows = pd.DataFrame(
        {
            "cashflow_date": pd.to_datetime(["2025-01-10", "2025-01-17", "2025-01-24"]),
            "cashflow": [5.0, 7.0, 11.0],
        }
    )
    paid = cashflow_paid_between(cashflows, pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-17"))
    assert paid == 7.0


def test_no_silent_extrapolation():
    curve = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-03")],
            "maturity_years": [1.0, 5.0],
            "zero_rate": [0.07, 0.09],
        }
    )
    rate, flag = interpolate_zero_rate(curve, pd.Timestamp("2025-01-03"), 10.0)
    assert np.isnan(rate)
    assert flag == "ZC_MATURITY_OUT_OF_RANGE"
