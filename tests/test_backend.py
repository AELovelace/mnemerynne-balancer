import json
import subprocess
from dataclasses import replace

from fastapi.testclient import TestClient

from conftest import DEVICES, PASSWORD, FakePopen, install_model
from mnemerynne_balancer import config
from mnemerynne_balancer import devices as dev
from mnemerynne_balancer.app import create_app
from mnemerynne_balancer.config import server_for

HDR = {"X-Balancer-Admin": "1"}
CUDA = [{"name": "CUDA0", "description": "NVIDIA GeForce GTX TITAN X", "mib": 12207},
        {"name": "CUDA1", "description": "NVIDIA GeForce GTX TITAN X", "mib": 12207}]
# A HIP build on a Titan X box with an AMD integrated GPU: the iGPU is the only ROCm device.
MIXED = """Available devices:
  ROCm0: AMD Radeon Graphics (2048 MiB, 1900 MiB free)
  CUDA0: NVIDIA GeForce GTX TITAN X (12207 MiB, 12000 MiB free)
  CUDA1: NVIDIA GeForce GTX TITAN X (12207 MiB, 12000 MiB free)
"""


def test_a_backend_keeps_only_its_own_devices():
    run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, MIXED, "")  # noqa: E731
    assert [d["name"] for d in dev.probe_devices("llama-server", runner=run)[0]] == ["ROCm0", "CUDA0", "CUDA1"]
    assert [d["name"] for d in dev.probe_devices("llama-server", runner=run, backend="cuda")[0]] == ["CUDA0", "CUDA1"]
    assert [d["name"] for d in dev.probe_devices("llama-server", runner=run, backend="rocm")[0]] == ["ROCm0"]


def test_server_for_prefers_the_env_then_the_build_then_llama_server(settings, tmp_path, monkeypatch):
    built = tmp_path / "cuda-llama-server"
    monkeypatch.setattr(config, "BUILDS", {"rocm": str(tmp_path / "missing"), "cuda": str(built)})
    assert server_for(settings, "auto") == settings.llama_server
    assert server_for(settings, "cuda") == settings.llama_server     # No build yet.
    built.write_text("")
    assert server_for(settings, "cuda") == str(built)
    assert server_for(replace(settings, llama_server_cuda="/x/llama-server"), "cuda") == "/x/llama-server"
    assert server_for(settings, "rocm") == settings.llama_server


def make(settings, probed):
    """An app whose probe answers from `probed` (backend -> devices) and records what it was asked."""
    calls = []

    def probe(server, backend="auto"):
        calls.append((server, backend))
        return probed.get(backend, []), f"{server}: nothing for {backend}"

    install_model(settings)
    app = create_app(settings, popen=FakePopen(), readings=lambda: [], probe=probe, watchdog=False)
    return app, calls


def signed_in(app):
    c = TestClient(app)
    c.__enter__()
    assert c.post("/api/login", json={"password": PASSWORD}, headers=HDR).status_code == 200
    return c


def test_switching_backend_restarts_the_cards_on_the_new_build(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDS", {"rocm": str(tmp_path / "missing"), "cuda": str(tmp_path / "missing")})
    settings = replace(settings, llama_server_cuda="/opt/llama.cpp/cuda/bin/llama-server")
    app, calls = make(settings, {"auto": DEVICES, "cuda": CUDA})
    c = signed_in(app)
    assert calls == [(settings.llama_server, "auto")]
    assert c.post("/api/cards/0/start", json={"model": "lfm2.5-2.6b-q8"}, headers=HDR).status_code == 200
    old = app.state.fleet.popen.procs[0]

    r = c.post("/api/settings", json={"backend": "cuda"}, headers=HDR)
    assert r.status_code == 200 and r.json()["backend"] == "cuda"
    assert old.returncode == -15                                  # Stopped before the new one started.
    new = app.state.fleet.popen.procs[-1]
    assert new.cmd[0] == "/opt/llama.cpp/cuda/bin/llama-server"
    assert new.cmd[new.cmd.index("--device") + 1] == "CUDA0"
    assert [card.device for card in app.state.fleet.cards] == ["CUDA0", "CUDA1"]
    assert not app.state.fleet.cards[1].config.running           # Only what was running comes back.
    status = c.get("/api/status").json()
    assert status["backend"] == "cuda" and status["llama_server"] == "/opt/llama.cpp/cuda/bin/llama-server"
    assert json.loads(settings.state_file.read_text())["backend"] == "cuda"
    assert '"backend": "cuda"' in settings.audit_file.read_text()
    c.__exit__(None, None, None)

    again, calls = make(settings, {"cuda": CUDA})                  # A restart probes the saved backend's build.
    assert calls == [("/opt/llama.cpp/cuda/bin/llama-server", "cuda")]
    assert again.state.fleet.backend == "cuda"


def test_a_backend_with_no_devices_is_refused_and_nothing_changes(settings):
    app, _ = make(settings, {"auto": DEVICES})
    c = signed_in(app)
    assert c.post("/api/cards/0/start", json={"model": "lfm2.5-2.6b-q8"}, headers=HDR).status_code == 200
    r = c.post("/api/settings", json={"backend": "cuda"}, headers=HDR)
    assert r.status_code == 409 and "lists no CUDA devices" in r.json()["detail"]
    assert app.state.fleet.backend == "auto" and app.state.fleet.popen.procs[0].returncode is None
    assert c.post("/api/settings", json={"backend": "vulkan"}, headers=HDR).status_code == 400


def test_balancer_backend_env_counts_only_without_saved_state(settings):
    app, calls = make(replace(settings, backend="rocm"), {"rocm": DEVICES})
    assert app.state.fleet.backend == "rocm" and calls[0][1] == "rocm"
    settings.state_file.parent.mkdir(parents=True, exist_ok=True)
    settings.state_file.write_text(json.dumps({"backend": "cuda"}))
    app, calls = make(replace(settings, backend="rocm"), {"cuda": CUDA})
    assert app.state.fleet.backend == "cuda" and calls[0][1] == "cuda"


def test_cuda_reads_nvidia_smi_even_with_an_amd_card(monkeypatch):
    monkeypatch.setattr(dev, "gpu_readings", lambda: [{"pci": "igpu"}])
    monkeypatch.setattr(dev, "nvidia_readings", lambda: [{"pci": "titan"}])
    assert dev.readings_for("auto") == [{"pci": "igpu"}]
    assert dev.readings_for("cuda") == [{"pci": "titan"}]
    assert dev.readings_for("rocm") == [{"pci": "igpu"}]
