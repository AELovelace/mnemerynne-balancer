import asyncio
import json

import httpx
import pytest

from conftest import DEVICES, FakePopen, install_model
from mnemerynne_balancer import instances
from mnemerynne_balancer.instances import Fleet, check_args
from mnemerynne_balancer.models import Models


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def health(ready: set):
    """Card ports in `ready` answer /health 200; others 503 (still loading)."""
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200 if request.url.port in ready else 503)
        if request.url.path == "/metrics":
            return httpx.Response(200, text="llamacpp:predicted_tokens_seconds 87.5\n")
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def fleet(settings, ready=None, clock=None):
    install_model(settings)
    popen = FakePopen()
    f = Fleet(settings, Models(settings.models_dir, settings.custom_models_file), DEVICES, popen=popen,
              transport=health(ready if ready is not None else {9181, 9182}), clock=clock or Clock())
    return f, popen


def test_start_builds_the_command_and_becomes_ready(settings):
    async def go():
        f, popen = fleet(settings, ready=set())
        await f.start(0, "lfm2.5-2.6b-q8", 8192, 4, "--flash-attn on")
        cmd = popen.procs[0].cmd
        assert cmd[:2] == [settings.llama_server, "-m"] and cmd[2].endswith("LFM2.5-2.6B-Q8_0.gguf")
        for flag, value in (("--port", "9181"), ("-c", "32768"), ("-np", "4"), ("--device", "ROCm0"), ("--host", "127.0.0.1")):
            assert cmd[cmd.index(flag) + 1] == value
        assert cmd[-3:] == ["--jinja", "--flash-attn", "on"]
        await f.tick()
        assert f.cards[0].state == "loading"
        f.http._transport = health({9181})
        await f.tick()
        assert f.cards[0].state == "ready"
        f._last_metrics = 0
        f.clock.t += 10
        await f.tick()
        assert f.cards[0].tokens_per_s == 87.5
        saved = json.loads(settings.state_file.read_text())
        assert saved["cards"][0] == {"model": "lfm2.5-2.6b-q8", "ctx": 8192, "slots": 4, "args": "--flash-attn on", "running": True}
        await f.stop(0)
        assert popen.procs[0].returncode == -15 and f.cards[0].state == "stopped"
        assert json.loads(settings.state_file.read_text())["cards"][0]["running"] is False
        await f.http.aclose()
    asyncio.run(go())


def test_crash_restarts_with_backoff_then_gives_up(settings):
    async def go():
        clock = Clock()
        f, popen = fleet(settings, clock=clock)
        await f.start(1, "lfm2.5-2.6b-q8", 4096, 2, "")
        card = f.cards[1]
        delays = []
        for _ in range(instances.GIVE_UP_FAILURES):
            popen.procs[-1].crash(139)
            await f.tick()
            if card.next_start is None:
                break
            delays.append(card.next_start - clock.t)
            assert card.state == "failed" and "code 139" in card.error
            clock.t = card.next_start
            await f.tick()                       # Restarts once the backoff is over.
            assert card.state == "loading" and card.process is not None
        assert delays == [5, 10, 20, 40]
        assert card.state == "failed" and "gave up" in card.error and card.config.running is False
        assert card.restarts == 4
        await f.http.aclose()
    asyncio.run(go())


def test_state_is_restored_at_boot(settings):
    async def go():
        f, popen = fleet(settings)
        await f.start(0, "lfm2.5-2.6b-q8", 8192, 4, "")
        await f.http.aclose()
        g, popen2 = fleet(settings)
        g.load_state()
        await g.tick()
        assert g.cards[0].config.model == "lfm2.5-2.6b-q8" and g.cards[0].state == "loading"
        assert len(popen2.procs) == 1 and g.cards[1].state == "stopped"
        await g.http.aclose()
    asyncio.run(go())


def test_missing_model_fails_clearly(settings):
    async def go():
        f, _ = fleet(settings)
        with pytest.raises(ValueError, match="Download"):
            await f.start(0, "qwen3-8b-q4", 4096, 2, "")
        with pytest.raises(ValueError, match="Unknown"):
            await f.start(0, "nope", 4096, 2, "")
        await f.http.aclose()
    asyncio.run(go())


@pytest.mark.parametrize("args", ["--port 1", "--host=0.0.0.0", "-m other.gguf", "--device ROCm1", "--path /", '"unclosed'])
def test_extra_args_cannot_override_our_flags(args):
    with pytest.raises(ValueError):
        check_args(args)


def test_vram_fit_counts_other_processes_and_frees_our_own(settings):
    async def go():
        f, _ = fleet(settings)
        install_model(settings, size=3 * 2**30)                           # A 3 GiB model file.
        entry = f.models.get("lfm2.5-2.6b-q8")
        reading = {"vram_free_mib": 8000}
        fit = f.fit(0, entry, 8192, 4, reading)                          # 3072 + 16*32.768 + 400 = ~3996
        assert fit["fits"] and fit["available_mib"] == 8000 - 512
        assert not f.fit(0, entry, 8192, 4, {"vram_free_mib": 3000})["fits"]
        await f.start(0, "lfm2.5-2.6b-q8", 8192, 4, "")
        assert f.fit(0, entry, 8192, 4, {"vram_free_mib": 1000})["fits"]   # Our own copy is stopped first.
        assert f.fit(0, entry, 8192, 4, None) == {"needed_mib": fit["needed_mib"], "available_mib": None,
                                                  "fits": True, "known": False}
        await f.http.aclose()
    asyncio.run(go())


def test_shutdown_stops_every_card_and_the_next_boot_brings_them_back(settings):
    async def go():
        f, popen = fleet(settings)
        await f.start(0, "lfm2.5-2.6b-q8", 8192, 4, "")
        await f.start(1, "lfm2.5-2.6b-q8", 4096, 2, "-fa off")
        await f.stop_all()
        assert [p.returncode for p in popen.procs] == [-15, -15]
        assert [c.state for c in f.cards] == ["stopped", "stopped"] and f.closing
        assert json.loads(settings.pid_file.read_text()) == {}
        saved = json.loads(settings.state_file.read_text())["cards"]
        assert [c["running"] for c in saved] == [True, True]   # Wanted running: a restart isn't a stop.
        await f.http.aclose()
        g, popen2 = fleet(settings)
        g.load_state()
        await g.tick()
        assert [c.state for c in g.cards] == ["loading", "loading"]
        assert [p.cmd[p.cmd.index("--device") + 1] for p in popen2.procs] == ["ROCm0", "ROCm1"]
        assert popen2.procs[1].cmd[-2:] == ["-fa", "off"] and g.cards[1].config.ctx == 4096
        await g.http.aclose()
    asyncio.run(go())


def test_no_restarts_once_shutdown_begins(settings):
    async def go():
        clock = Clock()
        f, popen = fleet(settings, clock=clock)
        await f.start(0, "lfm2.5-2.6b-q8", 8192, 4, "")
        f.closing = True                 # SIGTERM arrived; systemd stops the llama-servers too.
        popen.procs[0].terminate()
        for _ in range(3):
            clock.t += 120
            await f.tick()
        assert len(popen.procs) == 1 and not f.cards[0].failures and f.cards[0].config.running
        with pytest.raises(ValueError, match="shutting down"):
            await f.start(1, "lfm2.5-2.6b-q8", 8192, 4, "")
        await f.stop_all()
        assert len(popen.procs) == 1
        await f.http.aclose()
    asyncio.run(go())
