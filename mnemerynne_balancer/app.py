"""The control page and its JSON API (cookie sign-in), plus the /v1 balancer (API key). Auth follows npc-rag's
control page: one password, an HMAC-signed cookie, a custom header on every POST, a login rate limit, an audit log."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import shutil
import socket
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from mnemerynne_balancer import devices as dev
from mnemerynne_balancer import proxy
from mnemerynne_balancer.config import Secrets, Settings
from mnemerynne_balancer.downloads import Downloader, DownloadError, repo_files
from mnemerynne_balancer.instances import Fleet
from mnemerynne_balancer.models import Models

LOG = logging.getLogger("balancer.app")
PAGE = Path(__file__).with_name("page.html")
COOKIE = "balancer_session"
CSRF_HEADER = "X-Balancer-Admin"
SESSION_S = 12 * 3600
FAILS, FAIL_WINDOW_S = 5, 300


def create_app(s: Settings, *, devices: list[dict] | None = None, popen=None, readings=None,
               instance_transport=None, proxy_transport=None, download_transport=None, watchdog: bool = True) -> FastAPI:
    """devices/popen/readings/transports are for tests; normally they come from the machine."""
    if len(s.admin_password) < 8:
        raise ValueError("ADMIN_PASSWORD must be set (8+ characters) before the control page can start")
    if devices is None:
        devices, raw = dev.probe_devices(s.llama_server)
        if not devices:
            LOG.warning("llama-server listed no GPUs:\n%s", raw)
    read_cards = readings or dev.card_readings
    secret_store = Secrets(s)
    models = Models(s.models_dir, s.custom_models_file)
    fleet = Fleet(s, models, devices, **({"popen": popen} if popen else {}), transport=instance_transport)
    downloader = Downloader(s.models_dir, lambda: secret_store.get("hf_token"), download_transport)
    v1 = proxy.router(fleet, lambda: secret_store.get("api_key"), proxy_transport)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        killed = fleet.cleanup_leftovers()
        if killed:
            LOG.info("Stopped %d llama-server(s) left over from a previous run", killed)
        fleet.load_state()
        task = asyncio.create_task(fleet.run()) if watchdog else None
        try:
            yield
        finally:
            if task:
                task.cancel()
            await fleet.stop_all()
            await fleet.http.aclose()
            await v1.client.aclose()

    app = FastAPI(title="mnemerynne-balancer", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.fleet, app.state.models, app.state.downloader, app.state.secrets = fleet, models, downloader, secret_store
    app.include_router(v1)
    key = secrets.token_bytes(32)  # Per process: restarting signs everyone out.
    fails: dict[str, deque] = defaultdict(deque)

    def audit(request: Request, action: str, **extra) -> None:
        """One line per action: when, from where, what (never secret values)."""
        s.audit_file.parent.mkdir(parents=True, exist_ok=True)
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ip": request.client.host if request.client else "",
               "action": action, **extra}
        with s.audit_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def sign(expires: int) -> str:
        return f"{expires}.{hmac.new(key, str(expires).encode(), hashlib.sha256).hexdigest()}"

    def require(request: Request, post: bool = False) -> None:
        token = request.cookies.get(COOKIE, "")
        expires, _, _ = token.partition(".")
        if not (expires.isdigit() and int(expires) > time.time() and hmac.compare_digest(token, sign(int(expires)))):
            raise HTTPException(401, "sign in first")
        if post and request.headers.get(CSRF_HEADER) != "1":
            raise HTTPException(403, "missing control page header")

    async def body(request: Request) -> dict:
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(400, "expected JSON") from None
        if not isinstance(data, dict):
            raise HTTPException(400, "expected a JSON object")
        return data

    def entry_or_404(model_id) -> dict:
        entry = models.get(str(model_id or ""))
        if entry is None:
            raise HTTPException(404, "unknown model")
        return entry

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        if not request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Content-Security-Policy"] = ("default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                                           "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        return response

    @app.get("/", response_class=HTMLResponse)
    async def page():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "ready_cards": sum(c.state == "ready" for c in fleet.cards)}

    @app.post("/api/login")
    async def login(request: Request):
        ip = request.client.host if request.client else ""
        cutoff = time.time() - FAIL_WINDOW_S
        for stale in [k for k, v in fails.items() if not v or v[-1] < cutoff]:
            del fails[stale]
        recent = fails[ip]
        while recent and recent[0] < cutoff:
            recent.popleft()
        if len(recent) >= FAILS:
            raise HTTPException(429, "too many wrong passwords; wait a few minutes")
        if request.headers.get(CSRF_HEADER) != "1":
            raise HTTPException(403, "missing control page header")
        password = str((await body(request)).get("password", ""))
        if not hmac.compare_digest(password.encode(), s.admin_password.encode()):
            recent.append(time.time())
            audit(request, "login_failed")
            await asyncio.sleep(0.5)
            raise HTTPException(401, "wrong password")
        recent.clear()
        audit(request, "login")
        response = JSONResponse({"ok": True})
        response.set_cookie(COOKIE, sign(int(time.time()) + SESSION_S), max_age=SESSION_S, httponly=True,
                            samesite="strict", secure=request.url.scheme == "https")
        return response

    @app.post("/api/logout")
    async def logout():
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE)
        return response

    @app.get("/api/status")
    async def status(request: Request):
        require(request)
        readings = await asyncio.to_thread(read_cards)
        matched = dev.match_readings([{"mib": c.vram_mib} for c in fleet.cards], readings)
        cards = [dict(c.view(), reading=r) for c, r in zip(fleet.cards, matched)]
        jobs = {j["model"]: j for j in downloader.view()}
        listed = []
        for m in models.all():
            listed.append({**{k: m.get(k) for k in ("id", "name", "repo", "file", "kv_mib_per_1k", "ctx", "slots",
                                                     "args", "notes", "source")},
                           "installed_bytes": models.installed(m), "download": jobs.get(m["id"]),
                           "in_use": [c.index for c in fleet.cards if c.config.model == m["id"] and c.process is not None]})
        disk = shutil.disk_usage(s.data_dir if s.data_dir.exists() else Path("."))
        return {"node": socket.gethostname(), "cards": cards, "models": listed, "secrets": secret_store.status(),
                "disk_free_gb": round(disk.free / 2**30, 1), "llama_server": s.llama_server,
                "port": s.port, "vram_margin_mb": s.vram_margin_mb}

    @app.post("/api/cards/{index}/start")
    async def start(index: int, request: Request):
        require(request, post=True)
        data = await body(request)
        entry = entry_or_404(data.get("model"))
        try:
            fleet.card(index)
            ctx, slots = int(data.get("ctx") or entry["ctx"]), int(data.get("slots") or entry["slots"])
        except (KeyError, ValueError):
            raise HTTPException(400, "bad card, context or slots") from None
        args = str(data.get("args", ""))
        readings = await asyncio.to_thread(read_cards)
        reading = dev.match_readings([{"mib": c.vram_mib} for c in fleet.cards], readings)[index]
        fit = fleet.fit(index, entry, ctx, slots, reading)
        if not fit["fits"] and not data.get("force"):
            return JSONResponse({"error": "doesn't fit", "fit": fit}, status_code=409)
        try:
            await fleet.start(index, entry["id"], ctx, slots, args)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        audit(request, "card_start", card=index, model=entry["id"], ctx=ctx, slots=slots, forced=bool(data.get("force")))
        return {"ok": True, "fit": fit}

    @app.post("/api/cards/{index}/stop")
    async def stop(index: int, request: Request):
        require(request, post=True)
        try:
            await fleet.stop(index)
        except KeyError:
            raise HTTPException(404, "no such card") from None
        audit(request, "card_stop", card=index)
        return {"ok": True}

    @app.get("/api/cards/{index}/log")
    async def log(index: int, request: Request, lines: int = 200):
        require(request)
        try:
            return {"text": await asyncio.to_thread(fleet.log_tail, index, max(20, min(2000, lines)))}
        except KeyError:
            raise HTTPException(404, "no such card") from None

    @app.post("/api/models/download")
    async def download(request: Request):
        require(request, post=True)
        entry = entry_or_404((await body(request)).get("model"))
        downloader.start(entry)
        audit(request, "download", model=entry["id"])
        return {"ok": True}

    @app.post("/api/models/cancel")
    async def cancel(request: Request):
        require(request, post=True)
        entry = entry_or_404((await body(request)).get("model"))
        return {"ok": downloader.cancel(entry["id"])}

    @app.post("/api/models/delete")
    async def delete(request: Request):
        require(request, post=True)
        data = await body(request)
        entry = entry_or_404(data.get("model"))
        if any(c.config.model == entry["id"] and (c.process is not None or c.config.running) for c in fleet.cards):
            raise HTTPException(409, "A card is using that model; stop it first.")
        if downloader.active(entry["id"]):
            raise HTTPException(409, "It's downloading; cancel that first.")
        path = models.path(entry)
        path.unlink(missing_ok=True)
        downloader.partial_path(entry["repo"], entry["file"]).unlink(missing_ok=True)
        if data.get("forget") and entry.get("source") == "custom":
            models.remove_custom(entry["id"])
        audit(request, "model_delete", model=entry["id"], forget=bool(data.get("forget")))
        return {"ok": True}

    @app.post("/api/hf/files")
    async def hf_files(request: Request):
        require(request, post=True)
        repo = str((await body(request)).get("repo", "")).strip()

        def fetch():
            with downloader.client() as client:
                return repo_files(client, repo, secret_store.get("hf_token"))
        try:
            return {"files": await asyncio.to_thread(fetch)}
        except (ValueError, DownloadError) as error:
            raise HTTPException(400, str(error)) from None
        except httpx.HTTPError as error:
            raise HTTPException(502, f"Hugging Face didn't answer: {type(error).__name__}") from None

    @app.post("/api/hf/add")
    async def hf_add(request: Request):
        require(request, post=True)
        data = await body(request)
        try:
            kv = float(data["kv_mib_per_1k"]) if data.get("kv_mib_per_1k") not in (None, "") else None
            entry = models.add_custom(str(data.get("repo", "")).strip(), str(data.get("file", "")).strip(),
                                      str(data.get("name", "")).strip(), kv)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        audit(request, "model_add", model=entry["id"], repo=entry["repo"], file=entry["file"])
        return {"ok": True, "model": entry}

    @app.post("/api/settings")
    async def settings(request: Request):
        require(request, post=True)
        data = await body(request)
        changed = []
        for name in Secrets.KEYS:
            if name in data and data[name] is not None:
                value = str(data[name]).strip()
                if name == "api_key" and value and len(value) < 16:
                    raise HTTPException(400, "Use an API key of 16+ characters.")
                secret_store.set(name, value)
                changed.append(name)
        audit(request, "settings", changed=changed)  # Names only, never values.
        return {"ok": True, "secrets": secret_store.status()}

    @app.get("/api/audit")
    async def audit_log(request: Request, lines: int = 100):
        require(request)
        try:
            rows = s.audit_file.read_text(encoding="utf-8").splitlines()[-max(1, min(1000, lines)):]
        except OSError:
            rows = []
        return {"rows": [json.loads(r) for r in reversed(rows) if r.strip()]}

    return app
