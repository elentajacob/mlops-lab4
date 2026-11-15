import json

import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_pipeline import (
    bootstrap_accuracy_ci,
    build_model,
    cross_validate,
    dataset_statistics,
    decide_promotion,
    evaluate_model,
    load_data,
    split_data,
    train_model,
)


@pytest.fixture(scope="module")
def data():
    X, y = load_data()
    return X, y, split_data(X, y)


@pytest.fixture(scope="module")
def trained(data):
    _, _, (X_train, X_test, y_train, y_test) = data
    return train_model(X_train, y_train), X_test, y_test


# ----------------- Data ----------------- #
def test_load_data_shape(data):
    X, y, _ = data
    assert X.shape == (178, 13)
    assert sorted(y.unique()) == [0, 1, 2]


def test_split_is_stratified_and_reproducible(data):
    X, y, (X_train, X_test, y_train, y_test) = data
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == 36
    for cls in (0, 1, 2):
        assert abs((y_train == cls).mean() - (y_test == cls).mean()) < 0.03
    _, X_test_again, _, _ = split_data(X, y)
    assert list(X_test.index) == list(X_test_again.index)


def test_dataset_statistics(data):
    X, y, _ = data
    stats = dataset_statistics(X, y)
    assert stats["n_samples"] == 178
    assert stats["n_classes"] == 3
    assert stats["missing_values"] == 0
    assert sum(stats["class_counts"].values()) == 178
    assert 0 < stats["class_balance_ratio"] <= 1
    assert stats["largest_feature_std"] == "proline"
    assert stats["feature_std_ratio"] > 100
    json.dumps(stats)  # must be JSON-serializable for the GCS upload


# ----------------- Model ----------------- #
def test_build_model_standardizes_first():
    model = build_model(c=0.5)
    assert isinstance(model, Pipeline)
    assert isinstance(model.steps[0][1], StandardScaler)
    assert model.named_steps["clf"].C == 0.5


def test_evaluate_model(trained):
    model, X_test, y_test = trained
    m = evaluate_model(model, X_test, y_test)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert 0.0 <= m[key] <= 1.0
    assert m["accuracy"] > 0.9
    assert m["log_loss"] > 0
    assert set(m["f1_per_class"]) == {"0", "1", "2"}
    assert sum(sum(row) for row in m["confusion_matrix"]) == len(y_test)
    assert m["accuracy_ci_low"] <= m["accuracy"] <= m["accuracy_ci_high"]
    json.dumps(m)


def test_cross_validate(data):
    _, _, (X_train, _, y_train, _) = data
    cv = cross_validate(X_train, y_train)
    assert cv["cv_folds"] == 5
    assert 0.0 <= cv["cv_accuracy_mean"] <= 1.0
    assert cv["cv_accuracy_std"] >= 0.0


# ----------------- Statistics ----------------- #
def test_bootstrap_ci_perfect_predictions():
    ci = bootstrap_accuracy_ci([0, 1, 2] * 10, [0, 1, 2] * 10)
    assert ci["accuracy_ci_low"] == ci["accuracy_ci_high"] == 1.0


def test_bootstrap_ci_is_reproducible_and_ordered():
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 1, 0, 1, 2, 2, 1, 2, 0]
    first = bootstrap_accuracy_ci(y_true, y_pred)
    assert first == bootstrap_accuracy_ci(y_true, y_pred)
    assert first["accuracy_ci_low"] < 0.8 < first["accuracy_ci_high"]


# ----------------- Quality gate ----------------- #
def test_gate_passes_first_model():
    decision = decide_promotion({"accuracy": 0.95, "macro_f1": 0.95}, None)
    assert decision["promote"] is True


def test_gate_rejects_low_accuracy():
    decision = decide_promotion({"accuracy": 0.85, "macro_f1": 0.85}, None, min_accuracy=0.90)
    assert decision["promote"] is False
    assert "below minimum" in decision["reasons"][0]


def test_gate_rejects_regression():
    decision = decide_promotion(
        {"accuracy": 0.93, "macro_f1": 0.92}, {"macro_f1": 0.98}, max_regression=0.02
    )
    assert decision["promote"] is False
    assert "regressed" in decision["reasons"][1]


def test_gate_allows_drop_within_tolerance():
    decision = decide_promotion(
        {"accuracy": 0.97, "macro_f1": 0.97}, {"macro_f1": 0.98}, max_regression=0.02
    )
    assert decision["promote"] is True
