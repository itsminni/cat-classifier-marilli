import numpy as np
from sklearn.dummy import DummyClassifier

from src.data import TARGET, load_dataset
from src.models import (
    MODEL_FEATURES,
    ModelSpec,
    candidate_models,
    catboost_candidates,
    cross_validate_model,
)


def test_cross_validation_is_reproducible_and_covers_every_row():
    train, _, _ = load_dataset()
    spec = ModelSpec("dummy", DummyClassifier)
    first = cross_validate_model(spec, train[MODEL_FEATURES], train[TARGET], split_seed=7)
    second = cross_validate_model(spec, train[MODEL_FEATURES], train[TARGET], split_seed=7)

    assert np.array_equal(first.oof_predictions, second.oof_predictions)
    assert len(first.oof_predictions) == len(train)
    assert len(first.fold_accuracy) == 5
    assert not any(value is None for value in first.oof_predictions)


def test_sklearn_pipeline_handles_real_cleaned_data():
    train, _, _ = load_dataset()
    logistic = next(spec for spec in candidate_models() if spec.name == "Regressione logistica")
    result = cross_validate_model(
        logistic,
        train[MODEL_FEATURES],
        train[TARGET],
        split_seed=7,
        n_splits=2,
    )

    assert len(result.oof_predictions) == len(train)
    assert min(result.fold_accuracy) > 0.9


def test_catboost_can_fit_and_predict_cleaned_data():
    train, _, _ = load_dataset()
    sample = train.iloc[:100]
    model = catboost_candidates()[0].factory()
    model.set_params(iterations=3, thread_count=1)

    model.fit(sample[MODEL_FEATURES], sample[TARGET])
    predictions = np.asarray(model.predict(sample[MODEL_FEATURES])).reshape(-1)

    assert len(predictions) == len(sample)
    assert set(predictions).issubset(set(sample[TARGET]))
