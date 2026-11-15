"""Data, model, statistics and promotion rules for the Wine classifier.

Everything here is pure Python with no cloud calls, so it can be unit tested
on any machine without GCP credentials.
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ----------------- Data ----------------- #
# Wine dataset: 178 samples, 13 chemical measurements, 3 cultivars (classes 0, 1, 2)
def load_data():
    wine = load_wine(as_frame=True)
    return wine.data, wine.target


def class_names():
    return [str(name) for name in load_wine().target_names]


# Stratified split keeps the class proportions the same in train and test.
# Fixed random_state so training and verification see the identical test set.
def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


# Descriptive statistics of the dataset, computed before training
def dataset_statistics(X, y):
    counts = y.value_counts().sort_index()
    stds = X.std()
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(counts.shape[0]),
        "missing_values": int(X.isnull().sum().sum()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_balance_ratio": round(float(counts.min() / counts.max()), 4),
        # Features are on very different scales (proline in the hundreds, hue
        # around 1). This ratio is why the model standardizes features first.
        "largest_feature_std": str(stds.idxmax()),
        "smallest_feature_std": str(stds.idxmin()),
        "feature_std_ratio": round(float(stds.max() / stds.min()), 1),
    }


# ----------------- Model ----------------- #
# Standardize, then multinomial logistic regression.
# C is the inverse regularization strength: smaller C = simpler model.
def build_model(c=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=1000, random_state=RANDOM_STATE)),
    ])


def train_model(X_train, y_train, c=1.0):
    model = build_model(c)
    model.fit(X_train, y_train)
    return model


# 5-fold stratified cross-validation on the training set only
def cross_validate(X_train, y_train, c=1.0, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(build_model(c), X_train, y_train, cv=cv, scoring="accuracy")
    return {
        "cv_folds": folds,
        "cv_accuracy_mean": round(float(scores.mean()), 4),
        "cv_accuracy_std": round(float(scores.std()), 4),
    }


# ----------------- Statistics ----------------- #
# Bootstrap 95% confidence interval for test accuracy. The test set is small
# (36 samples), so a single accuracy number hides a lot of uncertainty.
def bootstrap_accuracy_ci(y_true, y_pred, n_boot=1000, confidence=0.95, seed=RANDOM_STATE):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = np.mean(y_true[idx] == y_pred[idx])
    tail = (1 - confidence) / 2 * 100
    return {
        "ci_level": confidence,
        "accuracy_ci_low": round(float(np.percentile(scores, tail)), 4),
        "accuracy_ci_high": round(float(np.percentile(scores, 100 - tail)), 4),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "log_loss": round(float(log_loss(y_test, y_proba, labels=model.classes_)), 4),
        "f1_per_class": {str(c): round(float(s), 4) for c, s in zip(model.classes_, f1_per_class)},
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test": int(len(y_test)),
    }
    metrics.update(bootstrap_accuracy_ci(y_test, y_pred))
    return metrics


# ----------------- Quality gate ----------------- #
# Decide whether a newly trained candidate replaces the model currently live in GCS.
# Rule 1: test accuracy must reach min_accuracy.
# Rule 2: macro F1 must not drop more than max_regression below the live model.
def decide_promotion(candidate, champion, min_accuracy=0.90, max_regression=0.02):
    reasons = []
    passed = True

    if candidate["accuracy"] >= min_accuracy:
        reasons.append(f"accuracy {candidate['accuracy']:.4f} >= minimum {min_accuracy:.4f}")
    else:
        passed = False
        reasons.append(f"accuracy {candidate['accuracy']:.4f} is below minimum {min_accuracy:.4f}")

    if champion is None:
        reasons.append("no live model yet, so there is nothing to compare against")
    else:
        floor = champion["macro_f1"] - max_regression
        if candidate["macro_f1"] >= floor:
            reasons.append(
                f"macro F1 {candidate['macro_f1']:.4f} >= live model "
                f"{champion['macro_f1']:.4f} minus tolerance {max_regression:.4f}"
            )
        else:
            passed = False
            reasons.append(
                f"macro F1 {candidate['macro_f1']:.4f} regressed more than {max_regression:.4f} "
                f"below the live model ({champion['macro_f1']:.4f})"
            )

    return {"promote": passed, "reasons": reasons}
