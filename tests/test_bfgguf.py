"""The BFGguf role: one llama-server across every card, switching to and from it, its routing, and sharded models."""
import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import DEVICES, KEY, PASSWORD, FakePopen, install_model, sha
from mnemerynne_balancer import instances
from mnemerynne_balancer.app import create_app
from mnemerynne_balancer.downloads import Downloader, repo_files
from mnemerynne_balancer.instances import Fleet, RoleRefused
from mnemerynne_balancer.models import Models

BIG = "qwen3.6-35b-a3b"
BIG_REPO, BIG_FILE = "unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
HDR = {"X-Balancer-Admin": "1"}
AUTH = {"Authorization": f"Bearer {KEY}"}
FREE = [{"vram_total_mib": 16384, "vram_free_mib": 7500, "vram_used_mib": 8884, "pci": "a"},
        {"vram_total_mib": 16384, "vram_free_mib": 7500, "vram_used_mib": 8884, "pci": "b"}]
TIGHT = [dict(r, vram_free_mib=600) for r in FREE]   # Another service holds nearly everything.


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Watching(FakePopen):
    """After watch(): fails the test if a llama-server starts while one of those running at watch() hasn't exited."""
    old: list = []

    def watch(self):
        self.old = list(self.procs)

    def __call__(self, cmd, **kwargs):
        assert all(p.returncode is not None for p in self.old), "started before the old llama-servers exited"
        return super().__call__(cmd, **kwargs)


def health(ready: set):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200 if request.url.port in ready else 503)
        if request.url.path == "/metrics":
            return httpx.Response(200, text="llamacpp:predicted_tokens_seconds 31.5\n")
        return httpx.Response(200, json={"port": request.url.port, "path": request.url.path})
    return httpx.MockTransport(handler)


def fleet(settings, ready=(), clock=None, big=True):
    install_model(settings)
    if big:
        install_model(settings, BIG_REPO, BIG_FILE, size=2000)
    popen = Watching()
    f = Fleet(settings, Models(settings.models_dir, settings.custom_models_file), DEVICES, popen=popen,
              transport=health(set(ready)), clock=clock or Clock())
    return f, popen


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


# ------------------------------------------------------------ Fleet
def test_the_bfgguf_command_spans_every_card(settings):
    async def go():
        f, popen = fleet(settings)
        assert f.role == "balancer" and f.bfgguf.config.model == BIG     # The catalog's BFGguf entry, by default.
        await f.set_role("bfgguf")
        cmd = popen.procs[0].cmd
        assert flag(cmd, "-m").endswith(BIG_FILE)
        for name, value in (("--device", "ROCm0,ROCm1"), ("-sm", "layer"), ("-ts", "1,1"), ("--port", "9181"),
                            ("-c", "16384"), ("-np", "2"), ("-ngl", "999"), ("--alias", BIG), ("--host", "127.0.0.1")):
            assert flag(cmd, name) == value
        assert "--metrics" in cmd and cmd[-3:] == ["--jinja", "-fa", "off"]
        assert f.bfgguf.state == "loading" and f.log_path(f.bfgguf).name == "bfgguf.log"
        assert "--device ROCm0,ROCm1" in f.log_path(f.bfgguf).read_text()
        await f.http.aclose()
    asyncio.run(go())


def test_switching_stops_the_old_set_first_and_brings_the_cards_back(settings):
    async def go():
        f, popen = fleet(settings)
        await f.start(0, "lfm2.5-2.6b-q8", 8192, 4, "")
        await f.start(1, "lfm2.5-2.6b-q8", 4096, 2, "-fa off")
        popen.watch()
        await f.set_role("bfgguf")
        assert [p.returncode for p in popen.procs[:2]] == [-15, -15] and len(popen.procs) == 3
        assert [c.state for c in f.cards] == ["stopped", "stopped"] and f.active() == [f.bfgguf]
        assert [c.config.running for c in f.cards] == [True, True]       # Kept for the way back.
        saved = json.loads(settings.state_file.read_text())
        assert saved["role"] == "bfgguf" and saved["bfgguf"]["model"] == BIG and saved["cards"][1]["args"] == "-fa off"
        assert json.loads(settings.pid_file.read_text()) == {"9181": popen.procs[2].pid}

        popen.watch()
        await f.set_role("balancer")
        assert popen.procs[2].returncode == -15 and f.bfgguf.state == "stopped"
        back = popen.procs[3:]
        assert [flag(p.cmd, "--device") for p in back] == ["ROCm0", "ROCm1"]
        assert [flag(p.cmd, "-c") for p in back] == ["32768", "8192"] and back[1].cmd[-2:] == ["-fa", "off"]
        assert json.loads(settings.state_file.read_text())["role"] == "balancer"
        await f.set_role("balancer")                                     # Already there: nothing happens.
        assert len(popen.procs) == 5
        await f.http.aclose()
    asyncio.run(go())


def test_the_role_survives_a_restart_and_the_bfgguf_comes_back_after_a_crash(settings):
    async def go():
        f, _ = fleet(settings)
        await f.start(1, "lfm2.5-2.6b-q8", 4096, 2, "")
        await f.set_role("bfgguf")
        await f.stop_all()
        await f.http.aclose()

        clock = Clock()
        g, popen = fleet(settings, ready={9181}, clock=clock)
        g.load_state()
        assert g.role == "bfgguf" and g.cards[1].config.model == "lfm2.5-2.6b-q8"
        await g.tick()
        assert len(popen.procs) == 1 and flag(popen.procs[0].cmd, "--device") == "ROCm0,ROCm1"
        await g.tick()
        assert g.bfgguf.state == "ready" and g.cards[1].state == "stopped"
        g._last_metrics, clock.t = 0, clock.t + 10
        await g.tick()
        assert g.bfgguf.tokens_per_s == 31.5
        popen.procs[0].crash(139)
        await g.tick()
        assert g.bfgguf.state == "failed" and "code 139" in g.bfgguf.error and g.bfgguf.next_start == clock.t + 5
        clock.t += 5
        await g.tick()
        assert g.bfgguf.state == "loading" and g.bfgguf.restarts == 1 and len(popen.procs) == 2
        assert flag(popen.procs[1].cmd, "-ts") == "1,1"
        await g.http.aclose()
    asyncio.run(go())


def test_balancer_role_env_counts_only_without_saved_state(settings):
    async def go():
        f, popen = fleet(replace(settings, role="bfgguf"))
        f.load_state()                                                   # A new node: no state.json yet.
        assert f.role == "bfgguf" and f.bfgguf.next_start is not None
        await f.tick()
        assert len(popen.procs) == 1 and f.bfgguf.state == "loading"
        await f.stop_all()
        await f.http.aclose()
        settings.state_file.write_text(json.dumps({"role": "balancer", "cards": []}))
        g, _ = fleet(replace(settings, role="bfgguf"))
        g.load_state()
        assert g.role == "balancer" and g.bfgguf.next_start is None
        await g.http.aclose()
        h, _ = fleet(settings)                                           # The default stays as it always was.
        h.load_state()
        assert h.role == "balancer"
        await h.http.aclose()
    asyncio.run(go())


def test_switching_is_refused_when_the_model_is_missing_or_too_big(settings):
    async def go():
        f, popen = fleet(settings, big=False)
        with pytest.raises(RoleRefused, match="isn't downloaded"):
            await f.set_role("bfgguf")
        install_model(settings, BIG_REPO, BIG_FILE)
        with pytest.raises(RoleRefused, match="free across the cards") as refused:
            await f.set_role("bfgguf", readings=TIGHT)
        assert refused.value.fit == {"needed_mib": 328 + 400, "available_mib": 2 * (600 - 512),   # KV for 2 x 8k + compute.
                                     "fits": False, "known": True}
        assert f.role == "balancer" and popen.procs == []
        fit = await f.set_role("bfgguf", anyway=True, readings=TIGHT)
        assert fit["fits"] is False and f.role == "bfgguf" and len(popen.procs) == 1
        with pytest.raises(ValueError, match="must be"):
            await f.set_role("other")
        await f.http.aclose()
    asyncio.run(go())


def test_bfgguf_fit_counts_our_own_running_cards_as_free(settings):
    async def go():
        f, _ = fleet(settings)
        entry = f.models.get(BIG)
        before = f.fit_bfgguf(entry, 8192, 2, FREE)["available_mib"]
        await f.start(0, "lfm2.5-2.6b-q8", 8192, 4, "")
        assert f.fit_bfgguf(entry, 8192, 2, FREE)["available_mib"] == before + f.estimate(f.models.get("lfm2.5-2.6b-q8"), 8192, 4)
        assert f.fit_bfgguf(entry, 8192, 2, [FREE[0], None])["known"] is False
        await f.http.aclose()
    asyncio.run(go())


def test_card_starts_and_stops_are_refused_on_a_bfgguf_node(settings):
    async def go():
        f, _ = fleet(settings)
        await f.set_role("bfgguf")
        with pytest.raises(ValueError, match="BFGguf node"):
            await f.start(0, "lfm2.5-2.6b-q8", 8192, 4, "")
        with pytest.raises(ValueError, match="BFGguf node"):
            await f.stop(1)
        await f.http.aclose()
    asyncio.run(go())


def test_changing_the_bfgguf_restarts_only_it(settings):
    async def go():
        f, popen = fleet(settings)
        await f.start_bfgguf(BIG, 4096, 4, "")                           # Balancer node: only saved.
        assert popen.procs == [] and f.bfgguf.config.slots == 4
        await f.set_role("bfgguf")
        assert flag(popen.procs[0].cmd, "-c") == "16384"
        await f.start_bfgguf(BIG, 8192, 1, "--top-k 20")
        assert popen.procs[0].returncode == -15 and len(popen.procs) == 2
        assert flag(popen.procs[1].cmd, "-np") == "1" and popen.procs[1].cmd[-2:] == ["--top-k", "20"]
        with pytest.raises(ValueError):
            await f.start_bfgguf(BIG, 8192, 1, "-ts 3,1")                 # The balancer sets the split itself.
        await f.http.aclose()
    asyncio.run(go())


# ------------------------------------------------------------ the app: page, /v1/balancer/role, routing
@pytest.fixture
def make(settings):
    def build(readings=None, big=True):
        install_model(settings)
        if big:
            install_model(settings, BIG_REPO, BIG_FILE)
        transport = health({9181, 9182})
        return create_app(settings, devices=DEVICES, popen=FakePopen(), readings=readings or (lambda: []),
                          instance_transport=transport, proxy_transport=transport, watchdog=False)
    return build


def signed_in(app):
    c = TestClient(app)
    c.__enter__()
    assert c.post("/api/login", json={"password": PASSWORD}, headers=HDR).status_code == 200
    return c


def test_v1_role_switch_needs_the_key_and_answers_with_the_load(make, settings):
    app = make()
    with TestClient(app) as c:
        assert c.post("/v1/balancer/role", json={"role": "bfgguf"}).status_code == 401
        assert c.post("/v1/balancer/role", json={"role": "bfgguf"}, headers={"Authorization": "Bearer nope"}).status_code == 401
        assert app.state.fleet.role == "balancer"
        r = c.post("/v1/balancer/role", json={"role": "sideways"}, headers=AUTH)
        assert r.status_code == 422 and "detail" in r.json()
        assert c.post("/v1/balancer/role", content=b"nope", headers=AUTH).status_code == 422
        r = c.post("/v1/balancer/role", json={"role": "bfgguf", "anyway": False}, headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"node", "role", "switching", "models"}
        assert body["role"] == "bfgguf" and body["switching"] is False
        assert body["models"] == [{"id": BIG, "ready": False, "in_flight": 0, "slots": 0, "cards": [0, 1]}]
        rows = [json.loads(line) for line in settings.audit_file.read_text().splitlines()]
        assert rows[-1]["action"] == "role" and rows[-1]["role"] == "bfgguf" and rows[-1]["via"] == "api"
        r = c.post("/v1/balancer/role", json={"role": "balancer"}, headers=AUTH)
        assert r.status_code == 200 and r.json()["role"] == "balancer"


def test_v1_role_switch_refuses_a_missing_or_oversized_model(make, settings):
    app = make(readings=lambda: TIGHT, big=False)
    with TestClient(app) as c:
        r = c.post("/v1/balancer/role", json={"role": "bfgguf"}, headers=AUTH)
        assert r.status_code == 409 and "isn't downloaded" in r.json()["detail"] and set(r.json()) == {"detail"}
        install_model(settings, BIG_REPO, BIG_FILE)
        r = c.post("/v1/balancer/role", json={"role": "bfgguf"}, headers=AUTH)
        assert r.status_code == 409 and "switch anyway" in r.json()["detail"]
        assert app.state.fleet.role == "balancer"
        r = c.post("/v1/balancer/role", json={"role": "bfgguf", "anyway": True}, headers=AUTH)
        assert r.status_code == 200 and r.json()["role"] == "bfgguf"


def test_a_bfgguf_node_serves_only_its_model(make):
    app = make()
    with TestClient(app) as c:
        fleet = app.state.fleet
        assert c.post("/v1/balancer/role", json={"role": "bfgguf"}, headers=AUTH).status_code == 200
        fleet.bfgguf.state = "ready"
        r = c.post("/v1/chat/completions", json={"model": "local"}, headers=AUTH)
        assert r.status_code == 404 and r.json()["detail"] == f"This is a BFGguf node; it only serves {BIG}."
        assert c.post("/v1/chat/completions", json={}, headers=AUTH).status_code == 404     # No model: no catch-all.
        for name in (BIG, "Qwen3.6 35B-A3B (BFGguf)", BIG_FILE.removesuffix(".gguf")):
            r = c.post("/v1/chat/completions", json={"model": name}, headers=AUTH)
            assert r.status_code == 200 and r.json()["port"] == 9181 and r.headers["x-balancer-card"] == "0,1"
        assert fleet.bfgguf.requests == 3 and fleet.bfgguf.in_flight == 0
        load = c.get("/v1/balancer/load", headers=AUTH).json()
        assert load["role"] == "bfgguf" and load["switching"] is False
        assert load["models"] == [{"id": BIG, "ready": True, "in_flight": 0, "slots": 2, "cards": [0, 1]}]
        assert c.get("/v1/models", headers=AUTH).json()["data"][0]["cards"] == [0, 1]
        fleet.bfgguf.state = "loading"
        r = c.post("/v1/chat/completions", json={"model": BIG}, headers=AUTH)
        assert r.status_code == 503 and "loading" in r.json()["detail"]


def test_load_always_has_role_and_switching_and_a_switch_answers_503(make):
    app = make()
    with TestClient(app) as c:
        card = app.state.fleet.cards[0]
        card.config.model, card.state = "lfm2.5-2.6b-q8", "ready"
        load = c.get("/v1/balancer/load", headers=AUTH).json()
        assert load["role"] == "balancer" and load["switching"] is False and load["node"]
        assert load["models"][0]["cards"] == [0]
        app.state.fleet.switching = True
        assert c.get("/v1/balancer/load", headers=AUTH).json()["switching"] is True
        r = c.post("/v1/chat/completions", json={"model": "local"}, headers=AUTH)
        assert r.status_code == 503 and "switching role" in r.json()["detail"]


def test_card_routes_are_refused_on_a_bfgguf_node(make):
    app = make()
    c = signed_in(app)
    r = c.post("/api/role", json={"role": "bfgguf"}, headers=HDR)
    assert r.status_code == 200 and r.json()["role"] == "bfgguf"
    for path, body in (("/api/cards/0/start", {"model": "lfm2.5-2.6b-q8"}), ("/api/cards/1/stop", None)):
        r = c.post(path, json=body, headers=HDR)
        assert r.status_code == 409 and r.json()["detail"] == "This is a BFGguf node right now."
    s = c.get("/api/status").json()
    assert s["role"] == "bfgguf" and s["bfgguf"]["cards"] == [0, 1] and s["bfgguf"]["state"] == "loading"
    assert next(m for m in s["models"] if m["id"] == BIG)["bfgguf"] is True
    assert next(m for m in s["models"] if m["id"] == BIG)["in_use"] == [0, 1]
    assert c.post("/api/models/delete", json={"model": BIG}, headers=HDR).status_code == 409
    assert c.get("/api/bfgguf/log").json()["text"].count("-sm layer") == 1
    c.__exit__(None, None, None)


def test_the_page_switches_with_a_fit_check_and_changes_the_bfgguf(make, settings):
    readings = [dict(r) for r in TIGHT]
    app = make(readings=lambda: readings)
    c = signed_in(app)
    r = c.post("/api/role", json={"role": "bfgguf"}, headers=HDR)
    assert r.status_code == 409 and r.json()["fit"]["fits"] is False
    assert c.post("/api/role", json={"role": "bfgguf", "anyway": True}, headers=HDR).status_code == 200
    procs = app.state.fleet.popen.procs
    for r in readings:                                   # Our own BFGguf counts as free, so squeeze the cards harder.
        r["vram_free_mib"] = 100
    r = c.post("/api/bfgguf", json={"model": BIG, "ctx": 4096, "slots": 1}, headers=HDR)
    assert r.status_code == 409 and r.json()["fit"]["fits"] is False
    assert c.post("/api/bfgguf", json={"model": BIG, "ctx": 4096, "slots": 1, "force": True}, headers=HDR).status_code == 200
    assert [p.returncode for p in procs] == [-15, None] and flag(procs[1].cmd, "-np") == "1"
    assert c.post("/api/role", json={"role": "balancer"}, headers=HDR).status_code == 200
    assert procs[1].returncode == -15
    rows = [json.loads(line)["action"] for line in settings.audit_file.read_text().splitlines()]
    assert rows.count("role") == 2 and "bfgguf_change" in rows
    c.__exit__(None, None, None)


def test_a_restarted_app_comes_back_as_a_bfgguf_node(make):
    app = make()
    with TestClient(app) as c:
        assert c.post("/v1/balancer/role", json={"role": "bfgguf"}, headers=AUTH).status_code == 200
    again = make()
    with TestClient(again) as c:
        fleet = again.state.fleet
        assert fleet.role == "bfgguf" and fleet.bfgguf.next_start is not None   # The watchdog starts it first thing.
        assert c.get("/v1/balancer/load", headers=AUTH).json()["role"] == "bfgguf"


# ------------------------------------------------------------ sharded models
SHARDS = {"Big-Q4/Big-Q4-00001-of-00002.gguf": bytes(range(256)) * 1024,
          "Big-Q4/Big-Q4-00002-of-00002.gguf": bytes(range(255, -1, -1)) * 512}
REPO = "someone/Big-GGUF"


def sharded_hf(seen=None):
    def handler(request: httpx.Request):
        if seen is not None:
            seen.append(request)
        if request.url.path == f"/api/models/{REPO}/tree/main":
            listing = [{"type": "file", "path": path, "size": len(data), "lfs": {"oid": sha(data), "size": len(data)}}
                       for path, data in SHARDS.items()]
            return httpx.Response(200, json=listing + [
                {"type": "file", "path": "Small.gguf", "size": 5, "lfs": {"oid": sha(b"small"), "size": 5}},
                {"type": "file", "path": "Gone-00001-of-00003.gguf", "size": 5, "lfs": {"oid": sha(b"gone!"), "size": 5}}])
        for path, data in SHARDS.items():
            if request.url.path == f"/{REPO}/resolve/main/{path}":
                rng = request.headers.get("range")
                start = int(rng.split("=")[1].rstrip("-")) if rng else 0
                return httpx.Response(206 if start else 200, content=data[start:])
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_repo_files_lists_a_sharded_model_once():
    with httpx.Client(transport=sharded_hf()) as client:
        files = repo_files(client, REPO)
    assert [f["path"] for f in files] == ["Big-Q4/Big-Q4-00001-of-00002.gguf", "Small.gguf"]   # Incomplete set left out.
    big = files[0]
    assert big["size"] == sum(len(d) for d in SHARDS.values())
    assert [(s["path"], s["sha256"]) for s in big["shards"]] == [(p, sha(d)) for p, d in SHARDS.items()]
    assert "shards" not in files[1]


def test_a_sharded_download_fetches_resumes_and_deletes_every_shard(settings):
    import time
    models = Models(settings.models_dir, settings.custom_models_file)
    entry = models.add_custom(REPO, "Big-Q4/Big-Q4-00001-of-00002.gguf")
    assert entry["id"] == "big-q4" and entry["name"] == "Big-Q4"
    d = Downloader(settings.models_dir, transport=sharded_hf())
    second = "Big-Q4/Big-Q4-00002-of-00002.gguf"
    partial = d.partial_path(REPO, second)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(SHARDS[second][:1000])                       # An interrupted earlier try.
    seen = []
    d.transport = sharded_hf(seen)
    job = d.start(entry)
    deadline = time.time() + 10
    while job.state in ("queued", "downloading", "verifying") and time.time() < deadline:
        time.sleep(0.02)
    assert job.state == "done", job.error
    assert job.received == job.total == sum(len(x) for x in SHARDS.values())
    assert any(r.headers.get("range") == "bytes=1000-" and r.url.path.endswith(second) for r in seen)
    for path, data in SHARDS.items():
        assert (settings.models_dir / REPO / path).read_bytes() == data
    assert models.installed(entry) == job.total
    (settings.models_dir / REPO / second).unlink()
    assert models.installed(entry) is None                          # Every shard, or it isn't installed.
    (settings.models_dir / REPO / second).write_bytes(SHARDS[second])

    f = Fleet(settings, models, DEVICES, popen=FakePopen(), transport=health(set()))
    f.cards[0].config = instances.CardConfig(entry["id"], 4096, 1)
    assert flag(f.command(f.cards[0], entry), "-m").endswith("Big-Q4-00001-of-00002.gguf")
    asyncio.run(f.http.aclose())

    app = create_app(settings, devices=DEVICES, popen=FakePopen(), readings=lambda: [], watchdog=False)
    c = signed_in(app)
    partial.write_bytes(b"leftover")
    assert c.post("/api/models/delete", json={"model": entry["id"]}, headers=HDR).status_code == 200
    assert not any((settings.models_dir / REPO / p).exists() for p in SHARDS) and not partial.exists()
    c.__exit__(None, None, None)
