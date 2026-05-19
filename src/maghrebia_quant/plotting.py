"""Graphiques Plotly du diagnostic."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"


def clean_plotly_layout(fig: go.Figure, title: str, height: int = 700, width: int = 1200) -> go.Figure:
    """Applique une mise en forme Plotly sobre et lisible."""

    fig.update_layout(
        template=TEMPLATE,
        title={"text": title, "x": 0.02, "xanchor": "left"},
        title_font_size=20,
        font={"size": 13},
        height=height,
        width=width,
        margin={"l": 90, "r": 60, "t": 95, "b": 110},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    return fig


def plot_portfolio_breakdown(portfolio_df: pd.DataFrame) -> go.Figure:
    values = portfolio_df.groupby("is_optimisable")["market_value"].sum().rename(index={True: "Optimisable", False: "Non optimisable"})
    fig = px.pie(values.reset_index(), names="is_optimisable", values="market_value", hole=0.55, title="Portefeuille : optimisable vs non optimisable", template=TEMPLATE)
    fig.update_traces(texttemplate="%{label}<br>%{percent:.1%}", textposition="inside")
    return fig


def plot_asset_class_breakdown(optimisable_df: pd.DataFrame) -> go.Figure:
    data = optimisable_df.groupby("asset_type", as_index=False)["market_value"].sum().sort_values("market_value")
    return px.bar(data, x="market_value", y="asset_type", orientation="h", title="Poche optimisable par type d'actif", labels={"market_value": "Valeur actuelle (TND)", "asset_type": "Type d'actif"}, template=TEMPLATE)


def plot_top_positions(optimisable_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    data = optimisable_df.nlargest(top_n, "market_value").sort_values("market_value")
    return px.bar(data, x="market_value", y="asset_name", orientation="h", title=f"Top {top_n} positions optimisables", labels={"market_value": "Valeur actuelle (TND)", "asset_name": "Titre"}, template=TEMPLATE)


def plot_base100(df: pd.DataFrame, title: str) -> go.Figure:
    data = df.dropna(how="all")
    base = data.divide(data.bfill().iloc[0]).multiply(100.0)
    return px.line(base, title=title, labels={"value": "Indice normalisé", "date": "Date", "variable": "Actif"}, template=TEMPLATE)


def plot_weekly_returns(returns_df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.line(returns_df, title=title, labels={"value": "Rendement hebdomadaire", "date": "Date", "variable": "Actif"}, template=TEMPLATE)
    fig.update_yaxes(tickformat=".1%")
    return fig


def plot_asset_risk_return_scatter(metrics_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(metrics_df, x="annualized_volatility", y="annualized_return", text="asset_id", title="Rendement annualisé vs volatilité annualisée", template=TEMPLATE)
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    return px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Matrice de corrélation", template=TEMPLATE)


def plot_covariance_heatmap(cov_df: pd.DataFrame) -> go.Figure:
    return px.imshow(cov_df, text_auto=".3f", color_continuous_scale="Blues", title="Matrice de covariance annualisée", template=TEMPLATE)


def plot_current_portfolio_base100(portfolio_returns: pd.Series) -> go.Figure:
    base100 = (1.0 + portfolio_returns.dropna()).cumprod() * 100.0
    return px.line(base100, title="Portefeuille actuel à pondérations constantes - indice normalisé", labels={"value": "Indice normalisé", "date": "Date"}, template=TEMPLATE)


def plot_risk_contribution(risk_contrib_df: pd.DataFrame) -> go.Figure:
    data = risk_contrib_df.melt(id_vars="asset_id", value_vars=["weight", "risk_contribution"], var_name="Mesure", value_name="Contribution")
    data["Mesure"] = data["Mesure"].replace({"weight": "Poids", "risk_contribution": "Contribution au risque"})
    fig = px.bar(data, x="asset_id", y="Contribution", color="Mesure", barmode="group", title="Poids vs contribution au risque", template=TEMPLATE)
    fig.update_yaxes(tickformat=".1%")
    return fig
