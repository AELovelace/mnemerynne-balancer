"""Hugging Face downloads: size and SHA-256 from the HF API, a resumable stream into models/.partial/, a full
check, then a rename into models/<repo>/<file>. Only verified files ever appear under their real name."""
from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from mnemerynne_balancer.models import check_repo_file, local_path

LOG = logging.getLogger("balancer.downloads")
HF = "https://huggingface.co"
CHUNK = 1024 * 1024
DISK_SPARE = 256 * 2**20  # Never fill the disk to the last byte.


class DownloadError(Exception):
    pass


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def repo_files(client: httpx.Client, repo: str, token: str = "") -> list[dict]:
    """Every .gguf in the repo (subfolders too): path, size and SHA-256."""
    check_repo_file(repo, "x.gguf")  # Validates the repo name; the file name is a placeholder.
    r = client.get(f"{HF}/api/models/{repo}/tree/main", params={"recursive": "true"}, headers=_headers(token))
    if r.status_code in (401, 403):
        raise DownloadError("Hugging Face refused access: the repo is gated or private (set an HF token).")
    if r.status_code == 404:
        raise DownloadError("No such repository on Hugging Face.")
    r.raise_for_status()
    out = []
    for f in r.json():
        if f.get("type") == "file" and f.get("path", "").endswith(".gguf"):
            lfs = f.get("lfs") or {}
            out.append({"path": f["path"], "size": int(lfs.get("size") or f.get("size") or 0), "sha256": lfs.get("oid", "")})
    return sorted(out, key=lambda f: f["path"])


def resolve(client: httpx.Client, repo: str, file: str, token: str = "") -> dict:
    """Size and SHA-256 of one file, from the HF API (never from the download itself)."""
    check_repo_file(repo, file)
    for f in repo_files(client, repo, token):
        if f["path"] == file:
            if not f["sha256"] or not f["size"]:
                raise DownloadError("Hugging Face didn't give a size and SHA-256 for that file, so it can't be verified.")
            return f
    raise DownloadError(f"{file} isn't in {repo}.")


def sha256_of(path: Path, cancel: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * CHUNK), b""):
            if cancel is not None and cancel.is_set():
                raise DownloadError("cancelled")
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Job:
    model_id: str
    repo: str
    file: str
    state: str = "queued"          # queued | downloading | verifying | done | failed | cancelled
    received: int = 0
    total: int = 0
    speed: float = 0.0             # bytes/s over the last few seconds
    error: str = ""
    started: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def view(self) -> dict:
        eta = (self.total - self.received) / self.speed if self.speed > 0 and self.total else None
        return {"model": self.model_id, "state": self.state, "received": self.received, "total": self.total,
                "speed": round(self.speed), "eta_s": round(eta) if eta is not None else None, "error": self.error}


class Downloader:
    """One background thread per download; the control page polls view()."""

    def __init__(self, models_dir: Path, token=lambda: "", transport: httpx.BaseTransport | None = None) -> None:
        self.models_dir = models_dir
        self.token = token
        self.transport = transport
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport, follow_redirects=True,
                            timeout=httpx.Timeout(30, read=120), headers={"User-Agent": "mnemerynne-balancer"})

    def partial_path(self, repo: str, file: str) -> Path:
        return self.models_dir / ".partial" / (repo.replace("/", "__") + "__" + file.replace("/", "__"))

    def start(self, entry: dict) -> Job:
        check_repo_file(entry["repo"], entry["file"])
        with self._lock:
            job = self.jobs.get(entry["id"])
            if job and job.state in ("queued", "downloading", "verifying"):
                return job
            job = self.jobs[entry["id"]] = Job(entry["id"], entry["repo"], entry["file"])
        threading.Thread(target=self.run, args=(job,), name=f"download-{entry['id']}", daemon=True).start()
        return job

    def cancel(self, model_id: str) -> bool:
        with self._lock:
            job = self.jobs.get(model_id)
        if job and job.state in ("queued", "downloading", "verifying"):
            job.cancel.set()
            return True
        return False

    def view(self) -> list[dict]:
        with self._lock:
            return [j.view() for j in self.jobs.values()]

    def active(self, model_id: str) -> bool:
        with self._lock:
            job = self.jobs.get(model_id)
        return bool(job and job.state in ("queued", "downloading", "verifying"))

    def run(self, job: Job) -> None:
        try:
            with self.client() as client:
                self._run(client, job)
            job.state = "done"
            LOG.info("Downloaded %s/%s", job.repo, job.file)
        except DownloadError as error:
            job.state = "cancelled" if job.cancel.is_set() else "failed"
            job.error = "" if job.cancel.is_set() else str(error)
        except (httpx.HTTPError, OSError) as error:
            job.state = "cancelled" if job.cancel.is_set() else "failed"
            job.error = f"{type(error).__name__}: {error}"[:300]
            LOG.warning("Download of %s failed: %s", job.model_id, job.error)

    def _run(self, client: httpx.Client, job: Job) -> None:
        spec = resolve(client, job.repo, job.file, self.token())
        target = local_path(self.models_dir, job.repo, job.file)
        job.total = spec["size"]
        if target.is_file() and target.stat().st_size == spec["size"]:
            job.received = job.total
            return  # Already here (it was verified before it was renamed in).
        partial = self.partial_path(job.repo, job.file)
        partial.parent.mkdir(parents=True, exist_ok=True)
        have = partial.stat().st_size if partial.is_file() else 0
        if have > spec["size"]:
            partial.unlink()
            have = 0
        free = shutil.disk_usage(partial.parent).free
        if free < spec["size"] - have + DISK_SPARE:
            raise DownloadError(f"Not enough disk space: {(spec['size'] - have) / 2**30:.1f} GB needed, "
                                f"{free / 2**30:.1f} GB free.")
        job.state = "downloading"
        job.received = have
        if have < spec["size"]:
            self._stream(client, job, partial, have, spec["size"])
        job.state = "verifying"
        if partial.stat().st_size != spec["size"] or sha256_of(partial, job.cancel) != spec["sha256"]:
            partial.unlink(missing_ok=True)  # A bad file must not be resumed: start clean next time.
            raise DownloadError("The download didn't match Hugging Face's SHA-256; it was deleted. Try again.")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(target)

    def _stream(self, client: httpx.Client, job: Job, partial: Path, have: int, size: int) -> None:
        url = f"{HF}/{job.repo}/resolve/main/{job.file}"
        headers = {**_headers(self.token()), **({"Range": f"bytes={have}-"} if have else {})}
        window: list[tuple[float, int]] = [(time.monotonic(), have)]
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code in (401, 403):
                raise DownloadError("Hugging Face refused the download: the repo is gated (set an HF token).")
            response.raise_for_status()
            if have and response.status_code != 206:
                have = 0  # The server ignored the Range header: start over.
            job.received = have
            with partial.open("ab" if have else "wb") as out:
                for chunk in response.iter_bytes(CHUNK):
                    if job.cancel.is_set():
                        raise DownloadError("cancelled")
                    job.received += len(chunk)
                    if job.received > size:
                        raise DownloadError("The download is bigger than Hugging Face said it would be.")
                    out.write(chunk)
                    now = time.monotonic()
                    window.append((now, job.received))
                    while len(window) > 2 and now - window[0][0] > 5:
                        window.pop(0)
                    span = now - window[0][0]
                    job.speed = (job.received - window[0][1]) / span if span > 0 else 0.0
