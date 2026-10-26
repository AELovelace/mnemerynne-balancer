"""Which models exist: the curated catalog, models added from Hugging Face, and what is on disk."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

CATALOG = Path(__file__).with_name("catalog.json")
TEMPLATES = Path(__file__).with_name("templates")   # Chat templates shipped with the balancer (*.jinja).
TEMPLATE_NAME = re.compile(r"^[\w.-]+\.jinja$")
REPO = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}$")
FILE = re.compile(r"^(?:[\w.-]+/){0,3}[\w.-]+\.gguf$")
# A model split into shards: <stem>-00001-of-00003.gguf, -00002-of-00003.gguf, ... llama-server is given the first.
SHARD = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$")
ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
DEFAULT_KV_MIB_PER_1K = 160   # Unknown architecture: assume a mid-size dense model (cautious).
COMPUTE_OVERHEAD_MIB = 400    # llama-server's compute buffers, on top of weights and KV.


def check_repo_file(repo: str, file: str) -> None:
    """Refuse anything that isn't a plain HF repo name and a .gguf path inside it."""
    if not REPO.match(repo or "") or ".." in repo:
        raise ValueError("Repository must look like owner/name.")
    if not FILE.match(file or "") or ".." in file:
        raise ValueError("File must be a .gguf path inside the repository.")


def chat_template_path(entry: dict) -> Path | None:
    """The shipped chat template a catalog entry names (chat_template), passed to llama-server as
    --chat-template-file; None when the entry doesn't name one. Mnemerynne's lfm2-no-think.jinja is
    LFM2.5's own template except that it thinks only when the request asks, so JSON calls (memory
    extraction, follow-up questions) keep to their grammar."""
    name = entry.get("chat_template")
    if not name:
        return None
    if not TEMPLATE_NAME.match(str(name)) or ".." in str(name):
        raise ValueError(f"Chat template {name!r} isn't a plain .jinja file name.")
    path = TEMPLATES / str(name)
    if not path.is_file():
        raise ValueError(f"Chat template {name} isn't shipped with the balancer ({TEMPLATES}).")
    return path


def shard_names(file: str) -> list[str]:
    """Every file of a model: the shards when `file` is the first of a set, otherwise just `file`."""
    m = SHARD.match(file or "")
    if not m or int(m.group(2)) != 1 or int(m.group(3)) < 2:
        return [file]
    return [f"{m.group(1)}-{k:05d}-of-{m.group(3)}.gguf" for k in range(1, int(m.group(3)) + 1)]


def local_path(models_dir: Path, repo: str, file: str) -> Path:
    check_repo_file(repo, file)
    return models_dir / repo / file


def estimate_mib(size_bytes: int, kv_mib_per_1k: float, ctx_per_slot: int, slots: int) -> int:
    """VRAM a model needs on its card: weights + KV for every slot's context + compute buffers."""
    return round(size_bytes / 2**20 + kv_mib_per_1k * ctx_per_slot * slots / 1000 + COMPUTE_OVERHEAD_MIB)


class Models:
    """The catalog plus custom entries (data/custom-models.json); each has an id, repo, file and launch defaults."""

    def __init__(self, models_dir: Path, custom_file: Path, catalog: Path = CATALOG) -> None:
        self.models_dir = models_dir
        self.custom_file = custom_file
        self._lock = threading.Lock()
        self._catalog = [dict(m, source="catalog") for m in json.loads(catalog.read_text(encoding="utf-8"))["models"]]

    def _custom(self) -> list[dict]:
        try:
            return [dict(m, source="custom") for m in json.loads(self.custom_file.read_text(encoding="utf-8"))]
        except (OSError, ValueError):
            return []

    def all(self) -> list[dict]:
        with self._lock:
            return self._catalog + self._custom()

    def get(self, model_id: str) -> dict | None:
        return next((m for m in self.all() if m["id"] == model_id), None)

    def path(self, entry: dict) -> Path:
        """The file llama-server is given: the model, or the first shard of a sharded one."""
        return local_path(self.models_dir, entry["repo"], entry["file"])

    def paths(self, entry: dict) -> list[Path]:
        return [local_path(self.models_dir, entry["repo"], f) for f in shard_names(entry["file"])]

    def installed(self, entry: dict) -> int | None:
        """Size in bytes when the model is downloaded (and verified: only verified files are ever renamed in).
        A sharded model counts only when every shard is there, and its size is theirs added up."""
        paths = self.paths(entry)
        if not all(p.is_file() for p in paths):
            return None
        return sum(p.stat().st_size for p in paths)

    def add_custom(self, repo: str, file: str, name: str = "", kv_mib_per_1k: float | None = None) -> dict:
        check_repo_file(repo, file)
        stem = SHARD.sub(r"\1", Path(file).name).removesuffix(".gguf")  # A sharded model is named without -00001-of-N.
        base = re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-")[:60] or "model"
        with self._lock:
            custom = self._custom()
            taken = {m["id"] for m in self._catalog + custom}
            if any(m["repo"] == repo and m["file"] == file for m in self._catalog + custom):
                raise ValueError("That file is already in the list.")
            model_id, n = base, 2
            while model_id in taken:
                model_id, n = f"{base}-{n}", n + 1
            entry = {"id": model_id, "name": (name or stem)[:80], "repo": repo, "file": file,
                     "kv_mib_per_1k": float(kv_mib_per_1k or DEFAULT_KV_MIB_PER_1K), "ctx": 4096, "slots": 2,
                     "args": "--jinja", "notes": "Added from Hugging Face."}
            self._save([{k: v for k, v in m.items() if k != "source"} for m in custom] + [entry])
        return dict(entry, source="custom")

    def remove_custom(self, model_id: str) -> None:
        with self._lock:
            custom = [{k: v for k, v in m.items() if k != "source"} for m in self._custom() if m["id"] != model_id]
            self._save(custom)

    def _save(self, entries: list[dict]) -> None:
        self.custom_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.custom_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=1), encoding="utf-8")
        tmp.replace(self.custom_file)
