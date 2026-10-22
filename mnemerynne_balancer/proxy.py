"""The OpenAI-compatible front door: /v1 requests go to the card serving the requested model, the less busy one
when both do. /v1/balancer/load says how busy this node is (Mnemerynne picks between nodes with it)."""
from __future__ import annotations

import hmac
import json
import socket

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from mnemerynne_balancer.instances import Card, Fleet

FORWARDED_REQUEST_HEADERS = ("content-type", "accept")
FORWARDED_RESPONSE_HEADERS = ("content-type", "cache-control")


def _busy(card: Card) -> float:
    return card.in_flight / max(1, card.config.slots)


def _names(fleet: Fleet, card: Card) -> set[str]:
    """Lower-cased names a client may use for the card's model: its id, display name, file name with or without .gguf."""
    entry = fleet.models.get(card.config.model or "") or {}
    file = entry.get("file", "").rsplit("/", 1)[-1].lower()
    return {n for n in ((card.config.model or "").lower(), entry.get("name", "").lower(), file,
                        file.removesuffix(".gguf")) if n}


def candidates(fleet: Fleet, model: str | None) -> list[Card]:
    """Ready cards that can answer, least busy first (ties: lower card number). A model name that matches nothing
    still works while only one model is running, so clients that send "local" or nothing keep working."""
    ready = [c for c in fleet.cards if c.state == "ready" and c.config.model]
    chosen = [c for c in ready if model and model.strip().lower() in _names(fleet, c)]
    if not chosen and len({c.config.model for c in ready}) == 1:
        chosen = ready
    return sorted(chosen, key=lambda c: (_busy(c), c.index))


def router(fleet: Fleet, api_key, transport: httpx.AsyncBaseTransport | None = None) -> APIRouter:
    """api_key: a callable returning the current key (the control page can change it)."""
    api = APIRouter(prefix="/v1")
    client = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(None, connect=5.0, write=60.0))

    def authorize(request: Request) -> None:
        key = api_key()
        if not key:
            raise HTTPException(503, "No API key is set on this balancer (BALANCER_API_KEY or the control page).")
        given = request.headers.get("authorization", "")
        if not given.lower().startswith("bearer ") or not hmac.compare_digest(given[7:].strip().encode(), key.encode()):
            raise HTTPException(401, "Wrong or missing API key.", headers={"WWW-Authenticate": "Bearer"})

    async def forward(request: Request, path: str):
        authorize(request)
        body = await request.body()
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            raise HTTPException(400, "The request body must be JSON.") from None
        model = payload.get("model") if isinstance(payload, dict) else None
        cards = candidates(fleet, model if isinstance(model, str) else None)
        if not cards:
            running = sorted({c.config.model for c in fleet.cards if c.state == "ready" and c.config.model})
            loading = [c.config.model for c in fleet.cards if c.state == "loading"]
            if loading:
                raise HTTPException(503, f"{', '.join(loading)} is still loading; try again shortly.")
            if running:
                raise HTTPException(404, f"No card is running {model!r}. Running: {', '.join(running)}.")
            raise HTTPException(503, "No model is running on this balancer.")
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARDED_REQUEST_HEADERS}
        failures = []
        for card in cards[:2]:  # The best card, then (once) the other one serving the same model.
            card.in_flight += 1
            card.requests += 1
            released = False

            def release(card=card):
                nonlocal released
                if not released:
                    released = True
                    card.in_flight -= 1

            try:
                upstream = await client.send(client.build_request(
                    request.method, f"http://{fleet.settings.instance_host}:{card.port}{path}",
                    content=body, headers=headers), stream=True)
            except httpx.TransportError as error:
                release()
                card.errors += 1
                card.last_error = f"{type(error).__name__}: {error}"[:200]
                failures.append(f"card {card.index}: {type(error).__name__}")
                continue

            async def relay(upstream=upstream, release=release):
                try:
                    async for chunk in upstream.aiter_bytes():  # Decoded: Content-Encoding isn't passed on.
                        yield chunk
                finally:
                    await upstream.aclose()
                    release()

            async def cleanup(upstream=upstream, release=release):
                await upstream.aclose()
                release()  # Covers a client that went away before the body was read.

            if upstream.status_code >= 500:
                card.errors += 1
                card.last_error = f"HTTP {upstream.status_code}"
            out = {k: v for k, v in upstream.headers.items() if k.lower() in FORWARDED_RESPONSE_HEADERS}
            out["X-Balancer-Card"] = str(card.index)
            return StreamingResponse(relay(), status_code=upstream.status_code, headers=out,
                                     background=BackgroundTask(cleanup))
        raise HTTPException(503, "The card(s) serving that model didn't answer: " + "; ".join(failures))

    @api.post("/chat/completions")
    async def chat(request: Request):
        return await forward(request, "/v1/chat/completions")

    @api.post("/completions")
    async def completions(request: Request):
        return await forward(request, "/v1/completions")

    @api.post("/embeddings")
    async def embeddings(request: Request):
        return await forward(request, "/v1/embeddings")

    @api.get("/models")
    async def models(request: Request):
        authorize(request)
        seen: dict[str, list[int]] = {}
        for c in fleet.cards:
            if c.state == "ready" and c.config.model:
                seen.setdefault(c.config.model, []).append(c.index)
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "mnemerynne-balancer",
                                            "cards": cards} for m, cards in seen.items()]}

    @api.get("/balancer/load")
    async def load(request: Request):
        """This node's load per model: requests in flight against parallel slots, and whether it's ready."""
        authorize(request)
        out: dict[str, dict] = {}
        for c in fleet.cards:
            if not c.config.model or c.state not in ("ready", "loading"):
                continue
            m = out.setdefault(c.config.model, {"id": c.config.model, "ready": False, "in_flight": 0, "slots": 0, "cards": []})
            m["cards"].append(c.index)
            if c.state == "ready":
                m["ready"] = True
                m["in_flight"] += c.in_flight
                m["slots"] += c.config.slots
        return JSONResponse({"node": socket.gethostname(), "models": list(out.values())})

    api.client = client  # So the app can close it on shutdown.
    return api
