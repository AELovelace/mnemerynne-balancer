import time

import httpx
import pytest

from conftest import sha
from mnemerynne_balancer.downloads import Downloader, DownloadError, resolve
from mnemerynne_balancer.models import Models, check_repo_file

DATA = bytes(range(256)) * 4096  # 1 MiB of known content.
REPO, FILE = "LiquidAI/LFM2.5-2.6B-GGUF", "LFM2.5-2.6B-Q8_0.gguf"


def hf(data=DATA, digest=None, gated=False, cut_at=None, honour_range=True, seen=None):
    """A fake Hugging Face: the tree API and resolve/main with Range support."""
    def handler(request: httpx.Request):
        if seen is not None:
            seen.append(request)
        if gated:
            return httpx.Response(401)
        if request.url.path == f"/api/models/{REPO}/tree/main":
            return httpx.Response(200, json=[
                {"type": "file", "path": FILE, "size": len(data), "lfs": {"oid": digest or sha(data), "size": len(data)}},
                {"type": "file", "path": "README.md", "size": 10},
                {"type": "directory", "path": "qad"}])
        if request.url.path == f"/{REPO}/resolve/main/{FILE}":
            start = 0
            rng = request.headers.get("range")
            if rng and honour_range:
                start = int(rng.split("=")[1].rstrip("-"))
            body = data[start:cut_at] if cut_at else data[start:]
            return httpx.Response(206 if start else 200, content=body)
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def wait(job, timeout=10):
    deadline = time.time() + timeout
    while job.state in ("queued", "downloading", "verifying") and time.time() < deadline:
        time.sleep(0.02)
    return job


@pytest.fixture
def entry(settings):
    return Models(settings.models_dir, settings.custom_models_file).get("lfm2.5-2.6b-q8")


def test_resolve_takes_size_and_hash_from_the_api():
    with httpx.Client(transport=hf()) as client:
        spec = resolve(client, REPO, FILE)
    assert spec["size"] == len(DATA) and spec["sha256"] == sha(DATA)


def test_download_verifies_and_renames_in(settings, entry):
    d = Downloader(settings.models_dir, transport=hf())
    job = wait(d.start(entry))
    assert job.state == "done", job.error
    target = settings.models_dir / REPO / FILE
    assert target.read_bytes() == DATA
    assert not d.partial_path(REPO, FILE).exists()


def test_interrupted_download_resumes_with_range(settings, entry):
    partial_first = Downloader(settings.models_dir, transport=hf(cut_at=300_000))
    job = wait(partial_first.start(entry))
    assert job.state == "failed"                                # Verification fails on the short file...
    # ...which is deleted, so simulate a genuine interruption: leave part of the file in place.
    partial = partial_first.partial_path(REPO, FILE)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(DATA[:300_000])
    seen = []
    job = wait(Downloader(settings.models_dir, transport=hf(seen=seen)).start(entry))
    assert job.state == "done", job.error
    assert any(r.headers.get("range") == "bytes=300000-" for r in seen)
    assert (settings.models_dir / REPO / FILE).read_bytes() == DATA


def test_server_ignoring_range_starts_over(settings, entry):
    d = Downloader(settings.models_dir, transport=hf(honour_range=False))
    partial = d.partial_path(REPO, FILE)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(DATA[:1000])
    assert wait(d.start(entry)).state == "done"
    assert (settings.models_dir / REPO / FILE).read_bytes() == DATA


def test_hash_mismatch_deletes_the_file(settings, entry):
    d = Downloader(settings.models_dir, transport=hf(digest="0" * 64))
    job = wait(d.start(entry))
    assert job.state == "failed" and "SHA-256" in job.error
    assert not (settings.models_dir / REPO / FILE).exists()
    assert not d.partial_path(REPO, FILE).exists()


def test_gated_repo_explains_the_token(settings, entry):
    job = wait(Downloader(settings.models_dir, transport=hf(gated=True)).start(entry))
    assert job.state == "failed" and "HF token" in job.error


@pytest.mark.parametrize("repo,file", [("../etc", "x.gguf"), ("a/b", "../../x.gguf"), ("a/b", "x.bin"),
                                       ("a", "x.gguf"), ("a/b/c", "x.gguf")])
def test_names_are_checked(repo, file):
    with pytest.raises(ValueError):
        check_repo_file(repo, file)


def test_custom_models_get_unique_ids(settings):
    models = Models(settings.models_dir, settings.custom_models_file)
    a = models.add_custom("someone/Thing-GGUF", "Thing-Q4_K_M.gguf")
    with pytest.raises(ValueError):
        models.add_custom("someone/Thing-GGUF", "Thing-Q4_K_M.gguf")
    b = models.add_custom("other/Thing-GGUF", "Thing-Q4_K_M.gguf")
    assert a["id"] == "thing-q4_k_m" and b["id"] == "thing-q4_k_m-2"
    assert models.get(b["id"])["source"] == "custom"
    models.remove_custom(a["id"])
    assert models.get(a["id"]) is None


def test_resolve_unknown_file():
    with httpx.Client(transport=hf()) as client, pytest.raises(DownloadError):
        resolve(client, REPO, "nope.gguf")
