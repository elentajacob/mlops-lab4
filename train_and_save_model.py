"""Train the Wine classifier, upload it to GCS, and promote it if it passes the quality gate.

Usage:
  python train_and_save_model.py --local   train and evaluate only, save to ./artifacts (no GCP needed)
  python train_and_save_model.py           full run used by GitHub Actions (needs GCP credentials)

Settings come from command-line flags or environment variables:
  GCS_BUCKET_NAME   bucket to upload to (required for a full run)
  MODEL_C           logistic regression C (default 1.0)
  MIN_ACCURACY      quality gate: minimum test accuracy (default 0.90)
  MAX_REGRESSION    quality gate: allowed macro-F1 drop vs the live model (default 0.02)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.model_pipeline import (
    class_names,
    cross_validate,
    dataset_statistics,
    decide_promotion,
    evaluate_model,
    load_data,
    split_data,
    train_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local", action="store_true", help="skip GCS, save to ./artifacts")
    parser.add_argument("--c", type=float, default=float(os.getenv("MODEL_C", "1.0")))
    parser.add_argument("--min-accuracy", type=float, default=float(os.getenv("MIN_ACCURACY", "0.90")))
    parser.add_argument("--max-regression", type=float, default=float(os.getenv("MAX_REGRESSION", "0.02")))
    return parser.parse_args(argv)


def run_context():
    """Where this run came from, recorded with every model for traceability."""
    return {
        "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
        "commit_sha": os.getenv("GITHUB_SHA", "local")[:7],
        "trigger": os.getenv("GITHUB_EVENT_NAME", "local"),
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_report(args):
    X, y = load_data()
    stats = dataset_statistics(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train, c=args.c)
    metrics = evaluate_model(model, X_test, y_test)
    metrics.update(cross_validate(X_train, y_train, c=args.c))

    report = {
        "dataset": "sklearn wine",
        "model": "StandardScaler + LogisticRegression",
        "hyperparameters": {"C": args.c},
        "class_names": class_names(),
        "dataset_statistics": stats,
        "metrics": metrics,
        "context": run_context(),
    }
    return model, report


def write_github_output(values):
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            for key, value in values.items():
                f.write(f"{key}={value}\n")


def write_summary(report, champion):
    """Markdown table on the GitHub Actions run page."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    m = report["metrics"]
    gate = report["gate"]
    old = champion["metrics"] if champion else None

    def row(label, key):
        before = f"{old[key]:.4f}" if old else "none"
        return f"| {label} | {m[key]:.4f} | {before} |"

    lines = [
        "## Training run",
        f"**Decision:** {'PROMOTED to models/latest' if gate['promote'] else 'REJECTED, live model unchanged'}",
        f"**Version:** {report['version']}  |  **C:** {report['hyperparameters']['C']}  |  "
        f"**Commit:** {report['context']['commit_sha']}  |  **Trigger:** {report['context']['trigger']}",
        "",
        "| Metric | This run | Live model before this run |",
        "|---|---|---|",
        row("Test accuracy", "accuracy"),
        row("Macro F1", "macro_f1"),
        row("Log loss", "log_loss"),
        row("CV accuracy (mean)", "cv_accuracy_mean"),
        "",
        f"95% bootstrap CI for test accuracy: {m['accuracy_ci_low']:.4f} to {m['accuracy_ci_high']:.4f}",
        "",
        "**Quality gate checks:**",
        *[f"- {reason}" for reason in gate["reasons"]],
        "",
        f"Uploaded to `{report['run_prefix']}/`",
    ]
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not args.local and not bucket_name:
        sys.exit("GCS_BUCKET_NAME is not set. Use --local to run without GCP.")

    model, report = build_report(args)
    print("Dataset statistics:", json.dumps(report["dataset_statistics"], indent=2))
    print("Evaluation metrics:", json.dumps(report["metrics"], indent=2))

    if args.local:
        gate = decide_promotion(report["metrics"], None, args.min_accuracy, args.max_regression)
        report["gate"] = gate
        out = Path("artifacts")
        out.mkdir(exist_ok=True)
        joblib.dump(model, out / "model.joblib")
        (out / "metrics.json").write_text(json.dumps(report, indent=2))
        print("Quality gate:", "PASS" if gate["promote"] else "FAIL")
        for reason in gate["reasons"]:
            print(" -", reason)
        print(f"Saved model and metrics to {out.resolve()}")
        return 0

    from src import gcs_utils  # imported here so --local works without GCP

    bucket = gcs_utils.get_bucket(bucket_name)

    champion = gcs_utils.read_json(bucket, f"{gcs_utils.LATEST_PREFIX}/{gcs_utils.METRICS_FILE}")
    gate = decide_promotion(
        report["metrics"],
        champion["metrics"] if champion else None,
        args.min_accuracy,
        args.max_regression,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_prefix = f"{gcs_utils.RUNS_PREFIX}/{timestamp}-run{report['context']['github_run_id']}"
    report["version"] = (champion["version"] + 1) if champion else 1
    report["gate"] = gate
    report["run_prefix"] = run_prefix

    # Every candidate is uploaded, promoted or not, so rejected models stay auditable
    model_uri = gcs_utils.upload_model(
        bucket,
        f"{run_prefix}/{gcs_utils.MODEL_FILE}",
        model,
        metadata={
            "version": report["version"],
            "accuracy": report["metrics"]["accuracy"],
            "commit_sha": report["context"]["commit_sha"],
            "github_run_id": report["context"]["github_run_id"],
        },
    )
    gcs_utils.upload_json(bucket, f"{run_prefix}/{gcs_utils.METRICS_FILE}", report)
    print(f"Uploaded candidate to {model_uri}")

    if gate["promote"]:
        gcs_utils.promote(bucket, run_prefix)
        print(f"Quality gate PASSED. Version {report['version']} is now live in {gcs_utils.LATEST_PREFIX}/")
    else:
        print("Quality gate FAILED. The live model was not changed.")
    for reason in gate["reasons"]:
        print(" -", reason)

    write_github_output({
        "promoted": str(gate["promote"]).lower(),
        "version": report["version"],
        "run_prefix": run_prefix,
    })
    write_summary(report, champion)

    return 0 if gate["promote"] else 1


if __name__ == "__main__":
    sys.exit(main())
