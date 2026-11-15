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


@pytest.fixture
def fake_bucket():
    return FakeBucket()
