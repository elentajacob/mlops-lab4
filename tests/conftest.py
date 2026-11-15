"""An in-memory stand-in for a GCS bucket, so tests never touch real cloud storage."""
import pytest


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name
        self.metadata = None

    def exists(self):
        return self.name in self.bucket.store

    def upload_from_file(self, file_obj, content_type=None):
        self.bucket.store[self.name] = file_obj.read()
        self.bucket.meta[self.name] = self.metadata

    def upload_from_string(self, data, content_type=None):
        self.bucket.store[self.name] = data.encode() if isinstance(data, str) else data

    def download_as_bytes(self):
        return self.bucket.store[self.name]

    def download_as_text(self):
        return self.bucket.store[self.name].decode()


class FakeBucket:
    def __init__(self, name="test-bucket"):
        self.name = name
        self.store = {}
        self.meta = {}

    def blob(self, name):
        return FakeBlob(self, name)

    def copy_blob(self, blob, destination_bucket, new_name):
        destination_bucket.store[new_name] = self.store[blob.name]


# Tests must not depend on run settings in the environment (for example a manual
# run with MODEL_C=0.001). Clear them so every test starts from the defaults.
@pytest.fixture(autouse=True)
def clean_run_settings(monkeypatch):
    for name in ("MODEL_C", "MIN_ACCURACY", "MAX_REGRESSION", "GCS_BUCKET_NAME"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_bucket():
    return FakeBucket()
