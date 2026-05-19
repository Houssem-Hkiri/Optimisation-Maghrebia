from pathlib import Path

import pytest

from maghrebia_quant.portfolio import (
    build_portfolio_classification_check,
    get_non_optimisable_portfolio,
    get_optimisable_portfolio,
    load_and_prepare_portfolio,
    portfolio_summary,
)


PORTFOLIO_PATH = Path("data/raw/Maghrebia Portfolio.xlsx")


@pytest.mark.skipif(not PORTFOLIO_PATH.exists(), reason="portfolio input not available")
def test_portfolio_keeps_non_optimisable_classes():
    portfolio = load_and_prepare_portfolio(PORTFOLIO_PATH)
    classes = set(portfolio.loc[portfolio["is_position"], "asset_class_standardized"])
    assert {"OPCVM", "SICAR", "Actions non cotées", "Placements monétaires", "Dépôts"}.issubset(classes)


@pytest.mark.skipif(not PORTFOLIO_PATH.exists(), reason="portfolio input not available")
def test_total_equals_optimisable_plus_non_optimisable():
    portfolio = load_and_prepare_portfolio(PORTFOLIO_PATH)
    summary, _ = portfolio_summary(portfolio)
    optimisable = get_optimisable_portfolio(portfolio)["market_value"].sum()
    non_optimisable = get_non_optimisable_portfolio(portfolio)["market_value"].sum()
    assert summary["total_portfolio_value"] == pytest.approx(optimisable + non_optimisable)
    assert summary["total_portfolio_value"] == pytest.approx(509_000_000, rel=0.01)


@pytest.mark.skipif(not PORTFOLIO_PATH.exists(), reason="portfolio input not available")
def test_non_optimisable_assets_are_not_dropped():
    portfolio = load_and_prepare_portfolio(PORTFOLIO_PATH)
    check = build_portfolio_classification_check(portfolio)
    non_optimisable_value = check.loc[~check["is_optimisable"], "total_value"].sum()
    assert non_optimisable_value > 0
    assert len(get_non_optimisable_portfolio(portfolio)) >= 8
