import pathlib
import concurrent.futures
import itertools
import threading

import pytest

import openpi.shared.download as download


@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


def test_concurrent_trainers_share_one_download(tmp_path, monkeypatch):
    (tmp_path / "example.com").mkdir()
    monkeypatch.setattr(download, "get_cache_dir", lambda: tmp_path)
    started, second_lock, release = threading.Event(), threading.Event(), threading.Event()
    real_lock = download.filelock.FileLock
    lock_calls = itertools.count()
    transfers = []

    def observed_lock(*args, **kwargs):
        if next(lock_calls) == 1:
            second_lock.set()
        return real_lock(*args, **kwargs)

    def transfer(url, path, **kwargs):
        transfers.append(url)
        started.set()
        assert release.wait(10)
        path.write_bytes(b"weights")

    monkeypatch.setattr(download.filelock, "FileLock", observed_lock)
    monkeypatch.setattr(download, "_download_fsspec", transfer)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(download.maybe_download, "gs://example.com/weights.bin")
        assert started.wait(10)
        second = pool.submit(download.maybe_download, "gs://example.com/weights.bin")
        try:
            assert second_lock.wait(10)
        finally:
            release.set()
        assert first.result() == second.result()
    assert len(transfers) == 1
    assert first.result().read_bytes() == b"weights"


def test_failed_transfer_does_not_publish_cache(tmp_path, monkeypatch):
    (tmp_path / "example.com").mkdir()
    monkeypatch.setattr(download, "get_cache_dir", lambda: tmp_path)

    class FailingFS:
        def info(self, url):
            return {"type": "file", "size": 7, "name": url}

        def get(self, url, path, **kwargs):
            path.write_bytes(b"partial")
            raise OSError("transfer failed")

    monkeypatch.setattr(download.fsspec.core, "url_to_fs", lambda *args, **kwargs: (FailingFS(), ""))
    with pytest.raises(OSError, match="transfer failed"):
        download.maybe_download("gs://example.com/weights.bin")
    assert not (tmp_path / "example.com/weights.bin").exists()


def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path
