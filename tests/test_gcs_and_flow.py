"""GCS helpers and the full train -> gate -> promote -> verify flow, on a fake bucket."""
from unittest.mock import patch

import pytest

import train_and_save_model
import verify_model
from src import gcs_utils
from src.model_pipeline import build_model

LATEST_METRICS = f"{gcs_utils.LATEST_PREFIX}/{gcs_utils.METRICS_FILE}"
LATEST_MODEL = f"{gcs_utils.LATEST_PREFIX}/{gcs_utils.MODEL_FILE}"


# ----------------- GCS helpers ----------------- #
def test_get_bucket_uses_client():
    with patch("src.gcs_utils.storage.Client") as client:
        gcs_utils.get_bucket("my-bucket")
        client.return_value.bucket.assert_called_once_with("my-bucket")


def test_json_round_trip(fake_bucket):
    assert gcs_utils.read_json(fake_bucket, "missing.json") is None
    gcs_utils.upload_json(fake_bucket, "a/metrics.json", {"accuracy": 0.97})
    assert gcs_utils.read_json(fake_bucket, "a/metrics.json") == {"accuracy": 0.97}


def test_model_round_trip_with_metadata(fake_bucket):
    model = build_model(c=0.3)
    uri = gcs_utils.upload_model(fake_bucket, "a/model.joblib", model, metadata={"version": 2})
    assert uri == "gs://test-bucket/a/model.joblib"
    assert fake_bucket.meta["a/model.joblib"] == {"version": "2"}  # GCS metadata is strings
    assert gcs_utils.download_model(fake_bucket, "a/model.joblib").named_steps["clf"].C == 0.3


def test_upload_errors_are_not_swallowed(fake_bucket):
    def broken_upload(*args, **kwargs):
        raise RuntimeError("upload failed")

    with patch.object(type(fake_bucket.blob("x")), "upload_from_file", broken_upload):
        with pytest.raises(RuntimeError):
            gcs_utils.upload_model(fake_bucket, "a/model.joblib", build_model())


def test_promote_copies_model_and_metrics(fake_bucket):
    fake_bucket.store["models/runs/r1/model.joblib"] = b"model"
    fake_bucket.store["models/runs/r1/metrics.json"] = b"{}"
    gcs_utils.promote(fake_bucket, "models/runs/r1")
    assert fake_bucket.store[LATEST_MODEL] == b"model"
    assert fake_bucket.store[LATEST_METRICS] == b"{}"


# ----------------- End-to-end flow ----------------- #
@pytest.fixture
def ci_env(monkeypatch, tmp_path, fake_bucket):
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("GITHUB_RUN_ID", "101")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    with patch("src.gcs_utils.get_bucket", return_value=fake_bucket):
        yield tmp_path


def test_first_run_promotes_and_verifies(ci_env, fake_bucket):
    assert train_and_save_model.main([]) == 0
    latest = gcs_utils.read_json(fake_bucket, LATEST_METRICS)
    assert latest["version"] == 1
    assert latest["gate"]["promote"] is True

    outputs = (ci_env / "output").read_text()
    assert "promoted=true" in outputs
    run_prefix = latest["run_prefix"]
    assert f"run_prefix={run_prefix}" in outputs
    assert "PROMOTED" in (ci_env / "summary").read_text()

    result = verify_model.verify(fake_bucket, run_prefix)
    assert result["version"] == 1
    assert result["accuracy"] == latest["metrics"]["accuracy"]


def test_rejected_run_keeps_live_model(ci_env, fake_bucket, monkeypatch):
    train_and_save_model.main([])
    live_before = fake_bucket.store[LATEST_METRICS]

    monkeypatch.setenv("GITHUB_RUN_ID", "102")
    assert train_and_save_model.main(["--min-accuracy", "1.01"]) == 1
    assert fake_bucket.store[LATEST_METRICS] == live_before          # live model untouched
    runs = [k for k in fake_bucket.store if k.endswith("run102/metrics.json")]
    assert len(runs) == 1                                             # candidate still audited
    assert "promoted=false" in (ci_env / "output").read_text()
    assert "REJECTED" in (ci_env / "summary").read_text()


def test_second_good_run_bumps_version(ci_env, fake_bucket, monkeypatch):
    train_and_save_model.main([])
    monkeypatch.setenv("GITHUB_RUN_ID", "103")
    assert train_and_save_model.main([]) == 0
    assert gcs_utils.read_json(fake_bucket, LATEST_METRICS)["version"] == 2


def test_verify_detects_wrong_run(ci_env, fake_bucket):
    train_and_save_model.main([])
    with pytest.raises(AssertionError):
        verify_model.verify(fake_bucket, "models/runs/some-other-run")


def test_missing_bucket_name_exits(monkeypatch):
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    with pytest.raises(SystemExit):
        train_and_save_model.main([])


def test_local_mode_needs_no_gcp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert train_and_save_model.main(["--local"]) == 0
    assert (tmp_path / "artifacts" / "model.joblib").exists()
    assert (tmp_path / "artifacts" / "metrics.json").exists()
