"""Contraintes élémentaires utilisées par les tests et futures optimisations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_no_short_selling(weights: pd.Series | np.ndarray) -> bool:
    """Vérifie l'absence de vente à découvert."""

    values = np.asarray(weights, dtype=float)
    return bool(np.all(values >= -1e-12))


def validate_weights_sum_to_one(weights: pd.Series | np.ndarray, tolerance: float = 1e-8) -> bool:
    """Vérifie que les poids somment à 1."""

    values = np.asarray(weights, dtype=float)
    return bool(abs(values.sum() - 1.0) <= tolerance)
