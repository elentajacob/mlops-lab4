"""Google Cloud Storage helpers.

Bucket layout:
  models/runs/<timestamp>-run<id>/model.joblib    every trained candidate (audit trail)
  models/runs/<timestamp>-run<id>/metrics.json
  models/latest/model.joblib                      the live model (only if it passed the gate)
  models/latest/metrics.json

Unlike the original lab, upload errors are NOT caught here: a failed upload
must fail the GitHub Actions run instead of printing an error and passing.
"""
import io
import json

import joblib
from google.cloud import storage

RUNS_PREFIX = "models/runs"
LATEST_PREFIX = "models/latest"
MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"


def get_bucket(bucket_name):
    return storage.Client().bucket(bucket_name)


def upload_model(bucket, blob_name, model, metadata=None):
    """Serialize the model in memory and upload it, with optional string metadata."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    blob = bucket.blob(blob_name)
    if metadata:
        blob.metadata = {str(k): str(v) for k, v in metadata.items()}
    blob.upload_from_file(buffer, content_type="application/octet-stream")
    return f"gs://{bucket.name}/{blob_name}"


def upload_json(bucket, blob_name, data):
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
    return f"gs://{bucket.name}/{blob_name}"


def read_json(bucket, blob_name):
    """Return the parsed JSON file, or None if it does not exist yet."""
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def download_model(bucket, blob_name):
    data = bucket.blob(blob_name).download_as_bytes()
    return joblib.load(io.BytesIO(data))


def promote(bucket, run_prefix):
    """Copy a candidate's model and metrics into models/latest/."""
    for name in (MODEL_FILE, METRICS_FILE):
        source = bucket.blob(f"{run_prefix}/{name}")
        bucket.copy_blob(source, bucket, f"{LATEST_PREFIX}/{name}")
