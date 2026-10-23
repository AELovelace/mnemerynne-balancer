"""One llama-server per card: start, stop, watch. The pattern is npc-rag's student_fleet: every layer on the
card (--device ROCmN), /health polled until ready, a watchdog with backoff, children that die with us."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from mnemerynne_balancer.config import Settings
from mnemerynne_balancer.devices import child_env
from mnemerynne_balancer.models import Models, estimate_mib

LOG = logging.getLogger("balancer.instances")
LINUX = os.name == "posix" and Path("/proc").is_dir()
WINDOWS = os.name == "nt"
BACKOFF_START, BACKOFF_MAX = 5.0, 60.0
GIVE_UP_FAILURES, GIVE_UP_WINDOW = 5, 600.0
STOP_GRACE = 15.0
# Flags the balancer sets itself (or that would expose the card to the LAN): never taken from "extra args".
OWN_FLAGS = {"-m", "--model", "-mu", "--model-url", "-hf", "--hf-repo", "--host", "--port", "-dev", "--device",
             "--path", "--api-key", "--api-key-file", "-c", "--ctx-size", "-np", "--parallel"}
METRIC = re.compile(r"^llamacpp:predicted_tokens_seconds\s+([0-9.eE+-]+)", re.M)


def check_args(args: str) -> list[str]:
    try:
        parts = shlex.split(args or "")
    except ValueError as error:
        raise ValueError(f"Extra args: {error}") from None
    bad = sorted({p.split("=", 1)[0] for p in parts if p.split("=", 1)[0] in OWN_FLAGS})
    if bad:
        raise ValueError(f"Extra args can't include {', '.join(bad)}: the balancer sets those itself.")
    return parts


@dataclass
class CardConfig:
    model: str | None = None
    ctx: int = 8192          # Per slot.
    slots: int = 4
    args: str = ""           # Extra flags on top of the model entry's own.
    running: bool = False    # Wanted running: restored at boot, restarted after crashes.


@dataclass
class Card:
    index: int
    device: str
    description: str
    vram_mib: int
    port: int
    config: CardConfig = field(default_factory=CardConfig)
    state: str = "stopped"   # stopped | loading | ready | failed | stopping
    process: subprocess.Popen | None = None
    started_at: float | None = None
    restarts: int = 0
    failures: deque = field(default_factory=deque)
    backoff: float = BACKOFF_START
    next_start: float | None = None
    last_exit: int | None = None
    error: str = ""
    tokens_per_s: float | None = None
    # Kept by the proxy.
    in_flight: int = 0
    requests: int = 0
    errors: int = 0
    last_error: str = ""

    def view(self) -> dict:
        return {"index": self.index, "device": self.device, "description": self.description, "vram_mib": self.vram_mib,
                "port": self.port, "config": asdict(self.config), "state": self.state,
                "uptime_s": round(time.time() - self.started_at) if self.started_at and self.state == "ready" else None,
                "restarts": self.restarts, "last_exit": self.last_exit, "error": self.error,
                "tokens_per_s": self.tokens_per_s, "in_flight": self.in_flight, "requests": self.requests,
                "errors": self.errors, "last_error": self.last_error}


def _die_with_parent():  # Linux: llama-server gets SIGTERM if the balancer dies, even outside systemd.
    import ctypes
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 15)  # PR_SET_PDEATHSIG = 1, SIGTERM = 15


_job = None


def _windows_job():
    """Windows' PDEATHSIG: a job object with KILL_ON_JOB_CLOSE. Its handle closes when the balancer exits, however it
    exits, and Windows then ends every llama-server in it, so a crash never leaves one holding VRAM."""
    global _job
    if _job is None:
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
            _fields_ = [("Basic", Basic), ("Io", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        info = Extended()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # 9: extended
            raise OSError(ctypes.get_last_error(), "couldn't set up the job object")
        _job = (k32, job)
    return _job


def _join_job(proc) -> None:
    handle = getattr(proc, "_handle", None)  # Popen's process handle (tests' stand-ins have none).
    if WINDOWS and handle is not None:
        try:
            k32, job = _windows_job()
            if not k32.AssignProcessToJobObject(job, int(handle)):
                raise OSError("AssignProcessToJobObject failed")
        except OSError as error:
            LOG.warning("llama-server %s won't stop if the balancer crashes: %s", proc.pid, error)


class Fleet:
    def __init__(self, settings: Settings, models: Models, devices: list[dict], *, popen=subprocess.Popen,
                 transport: httpx.AsyncBaseTransport | None = None, clock=time.monotonic) -> None:
        self.settings = settings
        self.models = models
        self.popen = popen
        self.clock = clock
        self.http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(3.0))
        self.cards = []
        for i, port in enumerate(settings.card_ports):
            d = devices[i] if i < len(devices) else {"name": f"ROCm{i}", "description": "not detected", "mib": 0}
            self.cards.append(Card(i, d["name"], d["description"], d["mib"], port))
        self.locks = [asyncio.Lock() for _ in self.cards]
        self._task: asyncio.Task | None = None
        self._last_metrics = 0.0
        # Set when shutdown begins (SIGTERM): systemd stops the llama-servers at the same moment, so their exits
        # are not crashes, and nothing may start while connections drain.
        self.closing = False

    # ------------------------------------------------------------ state file
    def load_state(self) -> None:
        try:
            saved = json.loads(self.settings.state_file.read_text(encoding="utf-8")).get("cards", [])
        except (OSError, ValueError):
            return
        for card, cfg in zip(self.cards, saved):
            try:
                card.config = CardConfig(**{k: cfg[k] for k in asdict(CardConfig()) if k in cfg})
            except TypeError:
                continue
            if card.config.running:
                card.next_start = self.clock()  # Back as it was before the reboot.

    def save_state(self) -> None:
        path = self.settings.state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cards": [asdict(c.config) for c in self.cards]}, indent=1), encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------ processes
    def command(self, card: Card, entry: dict) -> list[str]:
        cfg = card.config
        return [self.settings.llama_server, "-m", str(self.models.path(entry)), "--host", self.settings.instance_host,
                "--port", str(card.port), "-c", str(cfg.ctx * cfg.slots), "-np", str(cfg.slots), "-ngl", "999",
                "--device", card.device, "--metrics", "--alias", entry["id"],
                *check_args(entry.get("args", "")), *check_args(cfg.args)]

    def log_path(self, card: Card) -> Path:
        return self.settings.logs_dir / f"card{card.index}.log"

    def _spawn(self, card: Card) -> None:
        entry = self.models.get(card.config.model or "")
        if entry is None or self.models.installed(entry) is None:
            card.state, card.error = "failed", f"Model {card.config.model} isn't downloaded."
            card.config.running = False
            card.next_start = None
            return
        cmd = self.command(card, entry)
        log = self.log_path(card)
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.is_file() and log.stat().st_size > 20 * 2**20:
            log.replace(log.with_suffix(".log.1"))  # Keep one old log; never grow without bound.
        with log.open("ab") as out:
            out.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(cmd)}\n".encode())
            out.flush()
            kwargs = {"preexec_fn": _die_with_parent} if LINUX else {}
            card.process = self.popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      env=child_env(), **kwargs)
        _join_job(card.process)
        card.state, card.error, card.started_at, card.next_start = "loading", "", time.time(), None
        card.tokens_per_s = None
        self._write_pids()
        LOG.info("Card %d (%s): starting %s on port %d", card.index, card.device, entry["id"], card.port)

    async def _kill(self, card: Card) -> None:
        proc = card.process
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            deadline = self.clock() + STOP_GRACE
            while proc.poll() is None and self.clock() < deadline:
                await asyncio.sleep(0.2)
            if proc.poll() is None:
                proc.kill()
                await asyncio.to_thread(proc.wait)
        card.last_exit = proc.returncode
        card.process = None
        self._write_pids()

    def _write_pids(self) -> None:
        pids = {str(c.port): c.process.pid for c in self.cards if c.process is not None and c.process.poll() is None}
        try:
            self.settings.pid_file.parent.mkdir(parents=True, exist_ok=True)
            self.settings.pid_file.write_text(json.dumps(pids), encoding="utf-8")
        except OSError:
            pass

    def cleanup_leftovers(self) -> int:
        """Kill llama-servers a previous run left behind (a crash without systemd), so their VRAM and ports free up.
        Only processes whose command line is llama-server on one of our ports are touched."""
        try:
            pids = json.loads(self.settings.pid_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        killed = 0
        for port, pid in pids.items():
            try:
                cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
            except (OSError, ValueError):
                continue
            if any(b"llama-server" in p for p in cmdline[:1]) and str(port).encode() in cmdline:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    killed += 1
                except OSError:
                    pass
        self.settings.pid_file.unlink(missing_ok=True)
        return killed

    # ------------------------------------------------------------ control
    def card(self, index: int) -> Card:
        if not 0 <= index < len(self.cards):
            raise KeyError(index)
        return self.cards[index]

    def estimate(self, entry: dict, ctx: int, slots: int) -> int:
        size = self.models.installed(entry) or 0
        return estimate_mib(size, float(entry.get("kv_mib_per_1k", 160)), ctx, slots)

    def fit(self, index: int, entry: dict, ctx: int, slots: int, reading: dict | None) -> dict:
        """Needed vs available VRAM on the card. What our own llama-server holds now counts as available, since it
        is stopped first; npc-rag's copies (or anything else) count as used."""
        card = self.card(index)
        needed = self.estimate(entry, ctx, slots)
        if reading is None:
            return {"needed_mib": needed, "available_mib": None, "fits": True, "known": False}
        available = reading["vram_free_mib"] - self.settings.vram_margin_mb
        current = self.models.get(card.config.model or "")
        if current is not None and card.process is not None:
            available += self.estimate(current, card.config.ctx, card.config.slots)
        return {"needed_mib": needed, "available_mib": available, "fits": needed <= available, "known": True}

    async def start(self, index: int, model_id: str, ctx: int, slots: int, args: str) -> Card:
        card = self.card(index)
        entry = self.models.get(model_id)
        if entry is None:
            raise ValueError("Unknown model.")
        if self.models.installed(entry) is None:
            raise ValueError("Download the model first.")
        if not (256 <= ctx <= 262144) or not (1 <= slots <= 64):
            raise ValueError("Context must be 256–262144 per slot and slots 1–64.")
        check_args(args)
        async with self.locks[index]:
            if self.closing:
                raise ValueError("The balancer is shutting down.")
            if card.process is not None:
                card.state = "stopping"
                await self._kill(card)
            card.config = CardConfig(model_id, ctx, slots, args, True)
            card.failures.clear()
            card.backoff = BACKOFF_START
            self.save_state()
            self._spawn(card)
        return card

    async def stop(self, index: int) -> Card:
        card = self.card(index)
        async with self.locks[index]:
            card.config.running = False
            card.next_start = None
            self.save_state()
            card.state = "stopping"
            await self._kill(card)
            card.state, card.error, card.started_at = "stopped", "", None
        return card

    async def stop_all(self) -> None:
        """Shutdown: stop every llama-server, all at once so two slow cards still fit in systemd's TimeoutStopSec.
        config.running is kept, so the next boot brings them back."""
        self.closing = True

        async def one(card: Card) -> None:
            async with self.locks[card.index]:
                await self._kill(card)
                card.state = "stopped"
        await asyncio.gather(*(one(c) for c in self.cards))

    def log_tail(self, index: int, lines: int = 200) -> str:
        path = self.log_path(self.card(index))
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 256 * 1024))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    # ------------------------------------------------------------ watchdog
    async def tick(self) -> None:
        """One pass: notice exits, promote loading cards that answer /health, restart crashed ones after backoff."""
        if self.closing:
            return
        now = self.clock()
        for card in self.cards:
            if self.locks[card.index].locked():
                continue  # A start or stop is in progress.
            proc = card.process
            if proc is not None and proc.poll() is not None:
                card.last_exit, card.process = proc.returncode, None
                self._write_pids()
                card.failures.append(now)
                while card.failures and now - card.failures[0] > GIVE_UP_WINDOW:
                    card.failures.popleft()
                tail = self.log_tail(card.index, 40)
                if len(card.failures) >= GIVE_UP_FAILURES:
                    card.state, card.next_start = "failed", None
                    card.error = f"llama-server exited {len(card.failures)} times in 10 minutes; gave up.\n{tail}"
                    card.config.running = False
                    self.save_state()
                    LOG.error("Card %d: giving up after repeated crashes", card.index)
                else:
                    card.state = "failed"
                    card.error = f"llama-server exited with code {card.last_exit}; restarting in {card.backoff:.0f} s.\n{tail}"
                    card.next_start = now + card.backoff
                    card.backoff = min(card.backoff * 2, BACKOFF_MAX)
                    LOG.warning("Card %d: llama-server exited (%s)", card.index, card.last_exit)
                continue
            if card.process is None and card.config.running and card.next_start is not None and now >= card.next_start:
                if card.state == "failed":
                    card.restarts += 1
                self._spawn(card)
                continue
            if card.state == "loading" and card.process is not None:
                try:
                    r = await self.http.get(f"http://{self.settings.instance_host}:{card.port}/health")
                    if r.status_code == 200:
                        card.state = "ready"
                        card.backoff = BACKOFF_START
                        LOG.info("Card %d: ready", card.index)
                except httpx.HTTPError:
                    pass  # Still loading (503) or not listening yet.
        if now - self._last_metrics >= 5:
            self._last_metrics = now
            await self._metrics()

    async def _metrics(self) -> None:
        for card in self.cards:
            if card.state != "ready":
                continue
            try:
                r = await self.http.get(f"http://{self.settings.instance_host}:{card.port}/metrics")
                m = METRIC.search(r.text) if r.status_code == 200 else None
                card.tokens_per_s = round(float(m.group(1)), 1) if m else card.tokens_per_s
            except (httpx.HTTPError, ValueError):
                pass

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:  # The watchdog must never die.
                LOG.exception("Watchdog pass failed")
            await asyncio.sleep(1.0)

    def view(self) -> list[dict]:
        return [c.view() for c in self.cards]
