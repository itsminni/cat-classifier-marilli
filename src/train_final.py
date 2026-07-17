"""Esegue analisi, validazione, addestramento e creazione della submission."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report

from src.data import ID_COLUMN, TARGET, load_dataset
from src.models import (
    MODEL_FEATURES,
    candidate_models,
    catboost_candidates,
    cross_validate_model,
)
from src.plots import (
    plot_class_metrics,
    plot_coat_target_relation,
    plot_confusion,
    plot_feature_importance,
    plot_model_comparison,
    plot_numeric_correlation,
    plot_numeric_relationship,
    plot_target_distribution,
)
from src.submission import build_submission

RANDOM_SEED = 42


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    charts = root / "grafici"
    reports = root / "reports"
    charts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    train, test, _ = load_dataset(root / "data")
    X = cast(pd.DataFrame, train[MODEL_FEATURES])
    y = cast(pd.Series, train[TARGET])

    plot_target_distribution(train, charts / "distribuzione_razze.png")
    plot_numeric_correlation(train, charts / "correlazioni_numeriche.png")
    plot_coat_target_relation(train, charts / "relazione_colore_razza.png")
    plot_numeric_relationship(train, charts / "relazione_eta_peso.png")

    baseline_results = [
        cross_validate_model(spec, X, y, split_seed=RANDOM_SEED) for spec in candidate_models()
    ]
    coarse_results = [
        cross_validate_model(spec, X, y, split_seed=RANDOM_SEED) for spec in catboost_candidates()
    ]
    coarse_tuning = pd.DataFrame(result.summary for result in coarse_results).sort_values(
        ["accuracy_media", "macro_f1_media", "accuracy_std"],
        ascending=[False, False, True],
    )
    coarse_tuning.to_csv(reports / "ricerca_catboost.csv", index=False)
    shortlist = coarse_tuning.head(2)["modello"].tolist()

    tuning_results = [
        (RANDOM_SEED, result) for result in coarse_results if result.spec.name in shortlist
    ]
    tuning_results.extend(
        (seed, cross_validate_model(spec, X, y, split_seed=seed))
        for seed in (123, 2026)
        for spec in catboost_candidates()
        if spec.name in shortlist
    )
    tuning_table = pd.concat(
        [result.fold_table(seed) for seed, result in tuning_results], ignore_index=True
    )
    tuning_table.to_csv(reports / "stabilita_tuning_catboost.csv", index=False)
    tuning_summary = (
        tuning_table.groupby("modello", as_index=False)
        .agg(
            accuracy_media=("accuracy", "mean"),
            accuracy_std=("accuracy", lambda values: values.std(ddof=0)),
            balanced_accuracy_media=("balanced_accuracy", "mean"),
            macro_f1_media=("macro_f1", "mean"),
            weighted_f1_media=("weighted_f1", "mean"),
        )
        .sort_values(
            ["accuracy_media", "macro_f1_media", "accuracy_std"],
            ascending=[False, False, True],
        )
    )
    tuning_summary.to_csv(reports / "tuning_catboost.csv", index=False)

    final_name = tuning_summary.iloc[0]["modello"]
    final_spec = next(spec for spec in catboost_candidates() if spec.name == final_name)
    final_cv = next(result for result in coarse_results if result.spec.name == final_name)

    comparison_results = [*baseline_results, final_cv]
    comparison = pd.DataFrame(result.summary for result in comparison_results).sort_values(
        "accuracy_media", ascending=False
    )
    comparison.to_csv(reports / "confronto_modelli.csv", index=False)
    pd.concat(
        [result.fold_table(RANDOM_SEED) for result in comparison_results], ignore_index=True
    ).to_csv(reports / "metriche_per_fold.csv", index=False)
    plot_model_comparison(comparison, charts / "confronto_modelli.png")

    plot_confusion(y, final_cv.oof_predictions, charts / "confusion_matrix.png")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        report = classification_report(y, final_cv.oof_predictions, output_dict=True)
    if isinstance(report, str):
        raise TypeError("classification_report did not return the requested dictionary")
    accuracy = float(report.pop("accuracy"))
    class_report = pd.DataFrame(report).transpose()
    class_report.loc["accuracy"] = {
        "precision": np.nan,
        "recall": np.nan,
        "f1-score": accuracy,
        "support": len(y),
    }
    class_report.to_csv(reports / "classification_report.csv", index_label="classe")
    plot_class_metrics(class_report, charts / "metriche_per_classe.png")

    oof = pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN],
            "razza_reale": y,
            "razza_prevista": final_cv.oof_predictions,
            "corretta": np.asarray(y) == final_cv.oof_predictions,
        }
    )
    oof.to_csv(reports / "predizioni_oof.csv", index=False)

    final_model = final_spec.factory()
    final_model.fit(X, y)
    test_features = cast(pd.DataFrame, test[MODEL_FEATURES])
    test_ids = cast(pd.Series, test[ID_COLUMN])
    predictions = final_model.predict(test_features)
    build_submission(test_ids, predictions, set(y)).to_csv(root / "predictions.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importanza": final_model.get_feature_importance(),
        }
    ).sort_values("importanza", ascending=False)
    importance.to_csv(reports / "importanza_feature.csv", index=False)
    plot_feature_importance(importance, charts / "importanza_feature.png")

    print(f"Modello finale: {final_spec.name}")
    print(f"Accuracy media su 15 fold: {tuning_summary.iloc[0]['accuracy_media']:.2%}")
    print(f"Creato: {root / 'predictions.csv'}")


if __name__ == "__main__":
    main()
