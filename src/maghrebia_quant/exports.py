"""Exports Excel et HTML du diagnostic corrigé."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_diagnostic_outputs(
    output_path: Path,
    sheets: dict[str, pd.DataFrame | pd.Series | dict],
) -> Path:
    """Exporte toutes les feuilles obligatoires vers Excel."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        used_names: set[str] = set()
        for sheet_name, obj in sheets.items():
            safe_name = str(sheet_name)[:31]
            if safe_name in used_names:
                suffix = f"_{len(used_names)}"
                safe_name = f"{safe_name[:31-len(suffix)]}{suffix}"
            used_names.add(safe_name)
            if isinstance(obj, pd.Series):
                df = obj.to_frame()
            elif isinstance(obj, dict):
                df = pd.DataFrame([obj])
            else:
                df = obj.copy()
            df.to_excel(writer, sheet_name=safe_name, index=not isinstance(df.index, pd.RangeIndex))
    return output_path


def export_figures(figures: dict[str, object], figures_dir: Path) -> list[Path]:
    """Exporte les figures Plotly en HTML."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, fig in figures.items():
        path = figures_dir / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        paths.append(path)
    return paths
