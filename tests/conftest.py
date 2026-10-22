import hashlib

import pytest

from mnemerynne_balancer.config import Settings

PASSWORD = "correct horse battery"
KEY = "k" * 24


def make_settings(tmp_path, **over):
    base = dict(host="127.0.0.1", port=9180, data_dir=tmp_path / "data", llama_server="/opt/llama.cpp/hip/bin/llama-server",
                admin_password=PASSWORD, api_key=KEY, hf_token="", card_ports=(9181, 9182), vram_margin_mb=512)
    base.update(over)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


DEVICES = [{"name": "ROCm0", "description": "AMD Radeon RX 6800 XT", "mib": 16368},
           {"name": "ROCm1", "description": "AMD Radeon RX 6800 XT", "mib": 16368}]


class FakeProc:
    """A llama-server stand-in: running until terminate()/kill() or crash()."""
    next_pid = 4000

    def __init__(self, cmd):
        self.cmd = cmd
        self.returncode = None
        FakeProc.next_pid += 1
        self.pid = FakeProc.next_pid

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode

    def crash(self, code=1):
        self.returncode = code


class FakePopen:
    def __init__(self):
        self.procs = []

    def __call__(self, cmd, **kwargs):
        proc = FakeProc(cmd)
        self.procs.append(proc)
        return proc


def install_model(settings, repo="LiquidAI/LFM2.5-2.6B-GGUF", file="LFM2.5-2.6B-Q8_0.gguf", size=1000):
    path = settings.models_dir / repo / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
