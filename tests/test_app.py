import json

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import DEVICES, KEY, PASSWORD, FakePopen, install_model
from mnemerynne_balancer.app import create_app

HDR = {"X-Balancer-Admin": "1"}
AUTH = {"Authorization": f"Bearer {KEY}"}


def upstream(seen=None, down=()):
    """Fake llama-servers: echo which port answered; ports in `down` refuse connections."""
    def handler(request: httpx.Request):
        if seen is not None:
            seen.append(request)
        if request.url.port in down:
            raise httpx.ConnectError("refused", request=request)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/chat/completions" and json.loads(request.content).get("stream"):
            return httpx.Response(200, content=b"data: {\"x\":1}\n\ndata: [DONE]\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"port": request.url.port, "path": request.url.path})
    return httpx.MockTransport(handler)


@pytest.fixture
def make(settings):
    def build(readings=None, down=(), seen=None):
        install_model(settings)
        transport = upstream(seen, down)
        app = create_app(settings, devices=DEVICES, popen=FakePopen(), readings=readings or (lambda: []),
                         instance_transport=transport, proxy_transport=transport, watchdog=False)
        return app
    return build


def signed_in(app):
    c = TestClient(app)
    c.__enter__()
    assert c.post("/api/login", json={"password": PASSWORD}, headers=HDR).status_code == 200
    return c


def ready(app, index, model="lfm2.5-2.6b-q8", slots=4):
    card = app.state.fleet.cards[index]
    card.config.model, card.config.slots, card.state = model, slots, "ready"
    return card


def test_login_rules(make):
    with TestClient(make()) as c:
        assert c.get("/api/status").status_code == 401
        assert c.post("/api/login", json={"password": PASSWORD}).status_code == 403          # No page header.
        for _ in range(5):
            assert c.post("/api/login", json={"password": "wrong"}, headers=HDR).status_code == 401
        assert c.post("/api/login", json={"password": PASSWORD}, headers=HDR).status_code == 429


def test_posts_need_the_page_header(make):
    c = signed_in(make())
    assert c.post("/api/cards/0/stop").status_code == 403
    assert c.post("/api/cards/0/stop", headers=HDR).status_code == 200
    c.__exit__(None, None, None)


def test_secrets_are_write_only_and_never_logged(make, settings):
    app = make()
    c = signed_in(app)
    assert c.post("/api/settings", json={"api_key": "short"}, headers=HDR).status_code == 400
    r = c.post("/api/settings", json={"api_key": "n" * 20, "hf_token": "hf_secretvalue"}, headers=HDR)
    assert r.json()["secrets"] == {"api_key": True, "hf_token": True}
    status = c.get("/api/status").text
    assert "n" * 20 not in status and "hf_secretvalue" not in status and KEY not in status
    assert "hf_secretvalue" not in settings.audit_file.read_text()
    ready(app, 0)
    assert c.get("/v1/models", headers=AUTH).status_code == 401                                 # Old key replaced.
    assert c.get("/v1/models", headers={"Authorization": "Bearer " + "n" * 20}).status_code == 200
    c.__exit__(None, None, None)


def test_start_refuses_what_doesnt_fit_unless_forced(make, settings):
    app = make(readings=lambda: [{"vram_total_mib": 16384, "vram_free_mib": 600, "vram_used_mib": 15784, "pci": "a"},
                                 {"vram_total_mib": 16384, "vram_free_mib": 8000, "vram_used_mib": 8384, "pci": "b"}])
    c = signed_in(app)
    body = {"model": "lfm2.5-2.6b-q8", "ctx": 8192, "slots": 4}
    r = c.post("/api/cards/0/start", json=body, headers=HDR)
    assert r.status_code == 409 and r.json()["fit"]["fits"] is False
    assert c.post("/api/cards/1/start", json=body, headers=HDR).status_code == 200
    assert c.post("/api/cards/0/start", json={**body, "force": True}, headers=HDR).status_code == 200
    assert c.post("/api/cards/0/start", json={**body, "args": "--port 1"}, headers=HDR).status_code == 400
    rows = [json.loads(line) for line in settings.audit_file.read_text().splitlines()]
    assert [r["action"] for r in rows if r["action"] == "card_start"] == ["card_start", "card_start"]
    # The model is in use: it can't be deleted.
    assert c.post("/api/models/delete", json={"model": "lfm2.5-2.6b-q8"}, headers=HDR).status_code == 409
    c.__exit__(None, None, None)


def test_v1_requires_the_key(make):
    app = make()
    with TestClient(app) as c:
        ready(app, 0)
        assert c.get("/v1/models").status_code == 401
        assert c.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/models", headers=AUTH).json()["data"][0]["id"] == "lfm2.5-2.6b-q8"


def test_least_busy_card_wins_and_any_name_works_with_one_model(make):
    app = make()
    with TestClient(app) as c:
        a, b = ready(app, 0), ready(app, 1)
        a.in_flight = 2                                       # Card 0 is busier: card 1 answers.
        r = c.post("/v1/chat/completions", json={"model": "local", "messages": []}, headers=AUTH)
        assert r.status_code == 200 and r.json()["port"] == 9182 and r.headers["x-balancer-card"] == "1"
        assert b.in_flight == 0 and b.requests == 1           # Released after the answer.
        a.in_flight = 0
        r = c.post("/v1/chat/completions", json={"model": "LFM2.5-2.6B-Q8_0", "messages": []}, headers=AUTH)
        assert r.json()["port"] == 9181                       # Tie: the lower card.


def test_model_routing_with_two_models(make, settings):
    install_model(settings, "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf")
    app = make()
    with TestClient(app) as c:
        ready(app, 0, "lfm2.5-2.6b-q8")
        ready(app, 1, "qwen3-4b-q4")
        assert c.post("/v1/chat/completions", json={"model": "qwen3-4b-q4"}, headers=AUTH).json()["port"] == 9182
        r = c.post("/v1/chat/completions", json={"model": "local"}, headers=AUTH)
        assert r.status_code == 404 and "qwen3-4b-q4" in r.json()["detail"]


def test_a_dead_card_is_retried_on_the_other(make):
    app = make(down=(9181,))
    with TestClient(app) as c:
        a, b = ready(app, 0), ready(app, 1)
        r = c.post("/v1/chat/completions", json={"model": "local"}, headers=AUTH)
        assert r.json()["port"] == 9182
        assert a.errors == 1 and a.in_flight == 0 and "ConnectError" in a.last_error


def test_nothing_running_or_loading(make):
    app = make()
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json={}, headers=AUTH).status_code == 503
        app.state.fleet.cards[0].config.model, app.state.fleet.cards[0].state = "lfm2.5-2.6b-q8", "loading"
        r = c.post("/v1/chat/completions", json={}, headers=AUTH)
        assert r.status_code == 503 and "loading" in r.json()["detail"]
        assert c.post("/v1/chat/completions", content=b"not json", headers=AUTH).status_code == 400


def test_streaming_passes_through(make):
    app = make()
    with TestClient(app) as c:
        ready(app, 0)
        with c.stream("POST", "/v1/chat/completions", json={"stream": True}, headers=AUTH) as r:
            body = b"".join(r.iter_raw())
        assert r.headers["content-type"].startswith("text/event-stream") and body.endswith(b"data: [DONE]\n\n")
        assert app.state.fleet.cards[0].in_flight == 0


def test_load_reports_this_node(make):
    app = make()
    with TestClient(app) as c:
        ready(app, 0, slots=4).in_flight = 3
        ready(app, 1, slots=2)
        load = c.get("/v1/balancer/load", headers=AUTH).json()
        assert load["models"] == [{"id": "lfm2.5-2.6b-q8", "ready": True, "in_flight": 3, "slots": 6, "cards": [0, 1]}]
        assert c.get("/v1/balancer/load").status_code == 401


def test_page_and_status(make):
    app = make()
    c = signed_in(app)
    assert "<title>Balancer</title>" in c.get("/").text
    s = c.get("/api/status").json()
    assert [card["device"] for card in s["cards"]] == ["ROCm0", "ROCm1"]
    m = next(m for m in s["models"] if m["id"] == "lfm2.5-2.6b-q8")
    assert m["installed_bytes"] == 1000 and m["in_use"] == []
    c.__exit__(None, None, None)


def test_no_password_no_app(settings):
    from dataclasses import replace
    with pytest.raises(ValueError):
        create_app(replace(settings, admin_password="short"), devices=DEVICES)
