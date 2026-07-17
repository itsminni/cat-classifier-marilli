"""Modelli, validazione incrociata e selezione degli iperparametri."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

MODEL_FEATURES = FEATURE_COLUMNS
NUMERIC_MODEL_FEATURES = NUMERIC_COLUMNS


@dataclass(frozen=True)
class ModelSpec:
    """Nome e costruttore di un modello non addestrato."""

    name: str
    factory: Callable[[], Any]


@dataclass
class CVResult:
    """Metriche e predizioni out-of-fold di un modello."""

    spec: ModelSpec
    fold_accuracy: list[float]
    fold_balanced_accuracy: list[float]
    fold_macro_f1: list[float]
    fold_weighted_f1: list[float]
    oof_predictions: np.ndarray

    @property
    def summary(self) -> dict[str, object]:
        return {
            "modello": self.spec.name,
            "accuracy_media": np.mean(self.fold_accuracy),
            "accuracy_std": np.std(self.fold_accuracy),
            "balanced_accuracy_media": np.mean(self.fold_balanced_accuracy),
            "macro_f1_media": np.mean(self.fold_macro_f1),
            "macro_f1_std": np.std(self.fold_macro_f1),
            "weighted_f1_media": np.mean(self.fold_weighted_f1),
        }

    def fold_table(self, split_seed: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "modello": self.spec.name,
                "split_seed": split_seed,
                "fold": range(1, len(self.fold_accuracy) + 1),
                "accuracy": self.fold_accuracy,
                "balanced_accuracy": self.fold_balanced_accuracy,
                "macro_f1": self.fold_macro_f1,
                "weighted_f1": self.fold_weighted_f1,
            }
        )


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Preprocessing appreso esclusivamente sui dati del singolo fold."""
    numeric_steps: list[tuple[str, BaseEstimator]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_MODEL_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def _catboost(**overrides: object) -> CatBoostClassifier:
    parameters: dict[str, object] = {
        "cat_features": CATEGORICAL_COLUMNS,
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        "iterations": 650,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
        "random_strength": 0.5,
        "allow_writing_files": False,
        "verbose": False,
        "random_seed": 42,
        "thread_count": -1,
    }
    parameters.update(overrides)
    return CatBoostClassifier(**parameters)


def candidate_models() -> list[ModelSpec]:
    """Baseline usate nel confronto con il CatBoost selezionato."""
    return [
        ModelSpec(
            "Dummy (classe più frequente)",
            lambda: Pipeline(
                [
                    ("preprocess", make_preprocessor()),
                    ("model", DummyClassifier(strategy="most_frequent")),
                ]
            ),
        ),
        ModelSpec(
            "Regressione logistica",
            lambda: Pipeline(
                [
                    ("preprocess", make_preprocessor(scale_numeric=True)),
                    ("model", LogisticRegression(C=1.0, max_iter=3000, random_state=42)),
                ]
            ),
        ),
        ModelSpec(
            "ExtraTrees",
            lambda: Pipeline(
                [
                    ("preprocess", make_preprocessor()),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=700,
                            max_features=0.8,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]


def catboost_candidates() -> list[ModelSpec]:
    """Ricerca compatta su capacità, regolarizzazione e velocità di apprendimento."""
    return [
        ModelSpec("CatBoost (depth 4)", lambda: _catboost(iterations=500, depth=4)),
        ModelSpec("CatBoost (depth 5)", lambda: _catboost(iterations=550, depth=5)),
        ModelSpec("CatBoost (depth 6)", _catboost),
        ModelSpec(
            "CatBoost (depth 7)",
            lambda: _catboost(iterations=700, depth=7, l2_leaf_reg=7),
        ),
        ModelSpec(
            "CatBoost (learning rate 0,03)",
            lambda: _catboost(iterations=900, learning_rate=0.03),
        ),
        ModelSpec(
            "CatBoost (learning rate 0,08)",
            lambda: _catboost(iterations=450, learning_rate=0.08),
        ),
        ModelSpec("CatBoost (L2 10)", lambda: _catboost(l2_leaf_reg=10)),
        ModelSpec("CatBoost (depth 6, CTR 2)", lambda: _catboost(max_ctr_complexity=2)),
    ]


def _scores(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return (
            float(accuracy_score(y_true, y_pred)),
            float(balanced_accuracy_score(y_true, y_pred)),
            float(f1_score(y_true, y_pred, average="macro")),
            float(f1_score(y_true, y_pred, average="weighted")),
        )


def cross_validate_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    split_seed: int = 42,
    n_splits: int = 5,
) -> CVResult:
    """Produce una predizione out-of-fold per ogni riga."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    predictions = np.empty(len(y), dtype=object)
    accuracy, balanced_accuracy, macro_f1, weighted_f1 = [], [], [], []
    for train_idx, valid_idx in splitter.split(X, y):
        model = spec.factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_predictions = np.asarray(model.predict(X.iloc[valid_idx])).reshape(-1)
        predictions[valid_idx] = fold_predictions
        scores = _scores(y.iloc[valid_idx], fold_predictions)
        accuracy.append(scores[0])
        balanced_accuracy.append(scores[1])
        macro_f1.append(scores[2])
        weighted_f1.append(scores[3])
    return CVResult(
        spec,
        accuracy,
        balanced_accuracy,
        macro_f1,
        weighted_f1,
        predictions,
    )
