"""Settings, read from environment variables and an optional .env file (real variables win)."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # The checkout: .env and (by default) data/ live here.


def _load_dotenv(path: Path) -> None:
    """Copy KEY=value lines from .env into os.environ without overriding variables that are already set."""
    if not path.is_file():
        return  # .env is optional; plain environment variables work on their own.
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    llama_server: str
    admin_password: str
    api_key: str            # From .env; the control page can override it (Secrets).
    hf_token: str
    card_ports: tuple[int, ...]
    vram_margin_mb: int
    instance_host: str = "127.0.0.1"  # The cards' llama-servers are never reachable from the LAN.
    role: str = "balancer"  # balancer (one llama-server per card) | bfgguf (one across all cards). Only until state.json says.

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def state_file(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def secrets_file(self) -> Path:
        return self.data_dir / "secrets.json"

    @property
    def custom_models_file(self) -> Path:
        return self.data_dir / "custom-models.json"

    @property
    def audit_file(self) -> Path:
        return self.data_dir / "audit.jsonl"

    @property
    def pid_file(self) -> Path:
        return self.data_dir / "pids.json"


def load_settings(env_file: str | Path | None = None) -> Settings:
    _load_dotenv(Path(env_file) if env_file else Path(_env("BALANCER_ENV_FILE", str(ROOT / ".env"))))
    ports = tuple(int(p) for p in _env("CARD_PORTS", "9181,9182").split(",") if p.strip())
    if len(ports) < 1 or len(set(ports)) != len(ports):
        raise ValueError("CARD_PORTS must list one distinct port per card, e.g. 9181,9182")
    data = _env("BALANCER_DATA_DIR")
    role = _env("BALANCER_ROLE", "balancer").lower() or "balancer"
    if role not in ("balancer", "bfgguf"):
        raise ValueError("BALANCER_ROLE must be balancer or bfgguf")
    return Settings(
        host=_env("BALANCER_HOST", "0.0.0.0"),
        port=int(_env("BALANCER_PORT", "9180")),
        data_dir=Path(data) if data else ROOT / "data",
        llama_server=_env("LLAMA_SERVER", "/opt/llama.cpp/hip/bin/llama-server"),
        admin_password=_env("ADMIN_PASSWORD"),
        api_key=_env("BALANCER_API_KEY"),
        hf_token=_env("HF_TOKEN"),
        card_ports=ports,
        vram_margin_mb=int(_env("VRAM_MARGIN_MB", "512")),
        role=role,
    )


class Secrets:
    """The API key and HF token: .env values, replaced from the control page into
    data/secrets.json (owner-only). Never sent back to the page."""

    KEYS = ("api_key", "hf_token")

    def __init__(self, settings: Settings) -> None:
        self.path = settings.secrets_file
        self._defaults = {"api_key": settings.api_key, "hf_token": settings.hf_token}
        self._lock = threading.Lock()
        self._values = dict(self._defaults)
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self._values.update({k: str(v) for k, v in saved.items() if k in self.KEYS and v})
        except (OSError, ValueError):
            pass

    def get(self, key: str) -> str:
        with self._lock:
            return self._values[key]

    def set(self, key: str, value: str) -> None:
        if key not in self.KEYS:
            raise KeyError(key)
        with self._lock:
            self._values[key] = value.strip() or self._defaults[key]
            saved = {k: v for k, v in self._values.items() if v and v != self._defaults[k]}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(saved, f)
            os.replace(tmp, self.path)

    def status(self) -> dict:
        with self._lock:
            return {k: bool(v) for k, v in self._values.items()}
