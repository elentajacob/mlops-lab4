"""Check that the model now live in GCS is the one this run trained, and that it works.

1. models/latest/metrics.json must point at this run's folder (EXPECTED_RUN_PREFIX).
2. Download models/latest/model.joblib and re-score the same held-out test set.
3. The re-scored accuracy must match the accuracy recorded at training time.
"""
import os
import sys

from sklearn.metrics import accuracy_score

from src import gcs_utils
from src.model_pipeline import load_data, split_data


def verify(bucket, expected_run_prefix):
    latest = gcs_utils.read_json(bucket, f"{gcs_utils.LATEST_PREFIX}/{gcs_utils.METRICS_FILE}")
    if latest is None:
        raise AssertionError("No models/latest/metrics.json found in the bucket")
    if latest["run_prefix"] != expected_run_prefix:
        raise AssertionError(
            f"Live model came from {latest['run_prefix']}, expected {expected_run_prefix}"
        )

    model = gcs_utils.download_model(bucket, f"{gcs_utils.LATEST_PREFIX}/{gcs_utils.MODEL_FILE}")
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    rescored = round(float(accuracy_score(y_test, model.predict(X_test))), 4)
    recorded = latest["metrics"]["accuracy"]
    if rescored != recorded:
        raise AssertionError(f"Re-scored accuracy {rescored} does not match recorded {recorded}")

    names = latest["class_names"]
    samples = [names[int(p)] for p in model.predict(X_test.head(3))]
    return {"version": latest["version"], "accuracy": rescored, "sample_predictions": samples}


def main():
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    expected = os.getenv("EXPECTED_RUN_PREFIX")
    if not bucket_name or not expected:
        sys.exit("GCS_BUCKET_NAME and EXPECTED_RUN_PREFIX must both be set.")

    result = verify(gcs_utils.get_bucket(bucket_name), expected)
    print(f"Verified live model version {result['version']}")
    print(f"Re-scored test accuracy: {result['accuracy']:.4f} (matches training)")
    print(f"Sample predictions: {result['sample_predictions']}")

    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(
                "## Deployment check\n"
                f"Live model version {result['version']} downloaded from GCS and re-scored: "
                f"accuracy {result['accuracy']:.4f}, matching the training run.\n"
                f"Sample predictions: {', '.join(result['sample_predictions'])}\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
