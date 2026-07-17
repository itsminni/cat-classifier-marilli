from pathlib import Path

import pandas as pd
import pytest

from src.data import ID_COLUMN, TARGET, load_dataset
from src.submission import PREDICTION_COLUMN, build_submission

ROOT = Path(__file__).resolve().parents[1]


def test_submission_preserves_exact_schema_and_order():
    ids = pd.Series([3, 1, 2], name="ID")
    result = build_submission(ids, ["Siamese", "Bengala", "Siamese"], {"Siamese", "Bengala"})

    assert result.columns.tolist() == ["ID", "razza_prevista"]
    assert result["ID"].tolist() == [3, 1, 2]


def test_submission_rejects_unknown_labels_and_duplicate_ids():
    with pytest.raises(ValueError, match="Unexpected predicted labels"):
        build_submission(pd.Series([1]), ["Alien"], {"Siamese"})
    with pytest.raises(ValueError, match="unique"):
        build_submission(pd.Series([1, 1]), ["Siamese", "Siamese"], {"Siamese"})


def test_saved_predictions_match_the_real_test_set():
    """Controlla direttamente il file finale incluso nella repository."""
    train, test, _ = load_dataset(ROOT / "data")
    submission = pd.read_csv(ROOT / "predictions.csv")

    assert submission.columns.tolist() == [ID_COLUMN, PREDICTION_COLUMN]
    assert len(submission) == len(test)
    assert submission[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()
    assert not bool(submission[ID_COLUMN].isna().to_numpy(dtype=bool).any())
    assert not bool(submission[ID_COLUMN].duplicated().to_numpy(dtype=bool).any())
    assert not bool(submission[PREDICTION_COLUMN].isna().to_numpy(dtype=bool).any())
    assert set(submission[PREDICTION_COLUMN]).issubset(set(train[TARGET]))


def test_submission_rejects_a_prediction_count_mismatch():
    with pytest.raises(ValueError, match="one prediction"):
        build_submission(pd.Series([1, 2]), ["Siamese"], {"Siamese"})
