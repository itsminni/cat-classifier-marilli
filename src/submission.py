"""Costruzione e validazione della submission della challenge."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd

from src.data import ID_COLUMN

PREDICTION_COLUMN = "razza_prevista"


def build_submission(
    ids: pd.Series,
    predictions: Sequence[object] | np.ndarray,
    allowed_labels: Collection[str],
) -> pd.DataFrame:
    """Crea una submission a due colonne preservando l'ordine degli ID forniti."""
    predicted = np.asarray(predictions).reshape(-1)
    if len(predicted) != len(ids):
        raise ValueError("There must be exactly one prediction for every test ID")

    submission = pd.DataFrame({ID_COLUMN: ids.reset_index(drop=True), PREDICTION_COLUMN: predicted})

    ids_are_missing = bool(submission[ID_COLUMN].isna().to_numpy(dtype=bool).any())
    ids_are_duplicated = bool(submission[ID_COLUMN].duplicated().to_numpy(dtype=bool).any())
    if ids_are_missing or ids_are_duplicated:
        raise ValueError("Submission IDs must be present and unique")
    if not submission[ID_COLUMN].equals(ids.reset_index(drop=True)):
        raise ValueError("Submission ID order differs from the test data")
    if bool(submission[PREDICTION_COLUMN].isna().to_numpy(dtype=bool).any()):
        raise ValueError("Predictions must not be missing")

    unexpected = set(submission[PREDICTION_COLUMN]) - set(allowed_labels)
    if unexpected:
        raise ValueError(f"Unexpected predicted labels: {sorted(unexpected)}")
    return submission
