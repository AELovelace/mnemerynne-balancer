"""The cards: names and VRAM as llama-server sees them, live readings from amdgpu's sysfs files (AMD on Linux) or
nvidia-smi (NVIDIA, on Linux or Windows)."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

DEVICE_LINE = re.compile(r"^\s*([A-Za-z]+\d+)\s*:\s*(.+?)\s*\((\d+)\s*MiB")  # "  ROCm0: AMD Radeon RX 6800 XT (16368 MiB, ...)"
SYSFS = Path("/sys/class/drm")
AMD_VENDOR = "0x1002"
NVIDIA_QUERY = "pci.bus_id,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw"


def child_env() -> dict:
    """Environment for llama-server: CUDA numbers cards fastest-first by default, nvidia-smi in PCI order. PCI order
    for both, so CUDA0 is the first card in the readings. (ROCm already numbers in PCI order and ignores this.)"""
    return {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def probe_devices(server: str, runner=subprocess.run) -> tuple[list[dict], str]:
    """llama-server --list-devices: its GPUs (CPU entries skipped) and the raw output, which says why when there are none."""
    try:
        out = runner([server, "--list-devices"], capture_output=True, text=True, timeout=120, env=child_env())
    except (OSError, subprocess.SubprocessError) as error:
        return [], f"couldn't run {server}: {error}"
    text = f"{out.stdout}\n{out.stderr}"
    devices = []
    for line in text.splitlines():
        m = DEVICE_LINE.match(line)
        if m and not m.group(1).upper().startswith("CPU"):
            devices.append({"name": m.group(1), "description": m.group(2), "mib": int(m.group(3))})
    return devices, text.strip()


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _num(path: Path, scale: float = 1.0) -> float | None:
    value = _read(path)
    try:
        return float(value) / scale if value is not None else None
    except ValueError:
        return None


def _hwmon(device: Path) -> dict:
    """Edge and junction temperature (°C) and board power (W) from the card's hwmon files."""
    out: dict = {}
    for hw in sorted((device / "hwmon").glob("hwmon*")):
        for temp in sorted(hw.glob("temp*_input")):
            label = (_read(temp.with_name(temp.name.replace("_input", "_label"))) or "").lower()
            key = {"edge": "temp_c", "junction": "hotspot_c"}.get(label, "temp_c" if temp.name == "temp1_input" else None)
            if key and key not in out:
                out[key] = _num(temp, 1000)
        for name in ("power1_average", "power1_input"):  # Microwatts; which one exists depends on the card.
            watts = _num(hw / name, 1_000_000)
            if watts is not None:
                out.setdefault("power_w", round(watts, 1))
    return out


def gpu_readings(root: Path = SYSFS) -> list[dict]:
    """Every AMD card with VRAM, in PCI order (the order ROCm numbers them): VRAM in MiB, busy %, temperatures, power.
    Busy is whatever the driver reports; on these cards it can read 99% when idle, so it is never used for load."""
    cards = []
    for card in sorted(root.glob("card[0-9]*")):
        if "-" in card.name:
            continue  # card0-DP-1 and friends are connectors, not cards.
        device = card / "device"
        if _read(device / "vendor") != AMD_VENDOR:
            continue
        total = _num(device / "mem_info_vram_total", 2**20)
        if not total:
            continue
        uevent = dict(line.split("=", 1) for line in (_read(device / "uevent") or "").splitlines() if "=" in line)
        pci = uevent.get("PCI_SLOT_NAME") or card.name
        used = _num(device / "mem_info_vram_used", 2**20) or 0.0
        cards.append({"pci": pci, "vram_total_mib": round(total), "vram_used_mib": round(used),
                      "vram_free_mib": round(total - used), "busy_pct": _num(device / "gpu_busy_percent"),
                      **_hwmon(device)})
    return sorted(cards, key=lambda c: c["pci"])


def _float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]": the card doesn't report it (Maxwell boards often lack power.draw).


def nvidia_readings(runner=subprocess.run) -> list[dict]:
    """Every NVIDIA card from nvidia-smi, in PCI order: the same readings as gpu_readings. [] without nvidia-smi."""
    try:
        out = runner(["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for line in out.stdout.splitlines() if out.returncode == 0 else []:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 6:
            continue
        total, used = _float(fields[1]), _float(fields[2]) or 0.0
        if not total:
            continue
        power = _float(fields[5])
        cards.append({"pci": fields[0].lower(), "vram_total_mib": round(total), "vram_used_mib": round(used),
                      "vram_free_mib": round(total - used), "busy_pct": _float(fields[3]), "temp_c": _float(fields[4]),
                      "power_w": round(power, 1) if power is not None else None})
    return sorted(cards, key=lambda c: c["pci"])


def card_readings() -> list[dict]:
    """AMD cards from sysfs; if there are none (Windows, or an NVIDIA box), NVIDIA cards from nvidia-smi."""
    return gpu_readings() or nvidia_readings()


def match_readings(devices: list[dict], readings: list[dict]) -> list[dict | None]:
    """Pair each llama-server device with its sysfs card: by position when the counts agree, otherwise by the
    closest VRAM size (an integrated GPU can appear in one list and not the other)."""
    if len(devices) == len(readings):
        return list(readings)
    left, out = list(readings), []
    for device in devices:
        if not left:
            out.append(None)
            continue
        best = min(left, key=lambda r: abs(r["vram_total_mib"] - device["mib"]))
        left.remove(best)
        out.append(best)
    return out
