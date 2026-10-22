import subprocess

from mnemerynne_balancer import devices as dev

LIST = """ggml_cuda_init: found 2 ROCm devices:
Available devices:
  ROCm0: AMD Radeon RX 6800 XT (16368 MiB, 16112 MiB free)
  ROCm1: AMD Radeon RX 6800 XT (16368 MiB, 16112 MiB free)
"""


def test_list_devices_parses_rocm_cards():
    run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, LIST, "")  # noqa: E731
    found, raw = dev.probe_devices("llama-server", runner=run)
    assert [d["name"] for d in found] == ["ROCm0", "ROCm1"]
    assert found[0] == {"name": "ROCm0", "description": "AMD Radeon RX 6800 XT", "mib": 16368}


def test_missing_binary_is_reported_not_raised():
    def run(cmd, **kw):
        raise FileNotFoundError(cmd[0])
    found, raw = dev.probe_devices("/nope/llama-server", runner=run)
    assert found == [] and "couldn't run" in raw


def _card(root, n, pci, used_mib, vendor="0x1002", temps=("28000", "34000"), power="34000000"):
    pcidir = root / "drm" / f"card{n}" / "device"   # A symlink to the PCI device on a real system.
    hw = pcidir / "hwmon" / "hwmon0"
    hw.mkdir(parents=True)
    (pcidir / "uevent").write_text(f"DRIVER=amdgpu\nPCI_SLOT_NAME={pci}\n")
    (pcidir / "vendor").write_text(vendor)
    (pcidir / "mem_info_vram_total").write_text(str(16 * 2**30))
    (pcidir / "mem_info_vram_used").write_text(str(used_mib * 2**20))
    (pcidir / "gpu_busy_percent").write_text("99")
    (hw / "temp1_input").write_text(temps[0])
    (hw / "temp1_label").write_text("edge")
    (hw / "temp2_input").write_text(temps[1])
    (hw / "temp2_label").write_text("junction")
    (hw / "power1_average").write_text(power)


def test_sysfs_readings_in_pci_order(tmp_path):
    _card(tmp_path, 1, "0000:03:00.0", 8100)    # PCI order, not card number, decides.
    _card(tmp_path, 0, "0000:0b:00.0", 8090)
    _card(tmp_path, 2, "0000:0f:00.0", 10, vendor="0x10de")   # An NVIDIA card is ignored.
    (tmp_path / "drm" / "card0-DP-1").mkdir()                  # Connector entries are skipped.
    cards = dev.gpu_readings(tmp_path / "drm")
    assert [c["pci"] for c in cards] == ["0000:03:00.0", "0000:0b:00.0"]
    c = cards[0]
    assert c["vram_total_mib"] == 16384 and c["vram_used_mib"] == 8100 and c["vram_free_mib"] == 8284
    assert c["temp_c"] == 28 and c["hotspot_c"] == 34 and c["power_w"] == 34.0 and c["busy_pct"] == 99


def test_readings_pair_by_position_or_vram():
    devices = [{"mib": 16368}, {"mib": 16368}]
    readings = [{"vram_total_mib": 16384, "pci": "a"}, {"vram_total_mib": 16384, "pci": "b"}]
    assert dev.match_readings(devices, readings) == readings
    with_igpu = [{"vram_total_mib": 512, "pci": "igpu"}] + readings
    assert [r["pci"] for r in dev.match_readings(devices, with_igpu)] == ["a", "b"]
    assert dev.match_readings(devices, []) == [None, None]
