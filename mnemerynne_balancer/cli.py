"""python -m mnemerynne_balancer <command>: serve | devices | models | download MODEL_ID"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from mnemerynne_balancer import devices as dev
from mnemerynne_balancer.config import Secrets, load_settings
from mnemerynne_balancer.downloads import Downloader
from mnemerynne_balancer.models import Models


def _devices(s) -> int:
    found, raw = dev.probe_devices(s.llama_server)
    readings = dev.gpu_readings()
    if not found:
        print(f"{s.llama_server} listed no GPUs. Its output:\n{raw}")
        return 1
    matched = dev.match_readings(found, readings)
    for i, d in enumerate(found):
        r = matched[i]
        port = s.card_ports[i] if i < len(s.card_ports) else None
        line = f"{d['name']:8} {d['description']} ({d['mib']} MiB)"
        if r:
            line += f"  used {r['vram_used_mib']} / {r['vram_total_mib']} MiB, free {r['vram_free_mib']} MiB"
            if r.get("temp_c") is not None:
                line += f", {r['temp_c']:.0f} °C"
        line += f"  -> card port {port}" if port else "  (no CARD_PORTS entry: not used)"
        print(line)
    return 0


def _models(s) -> int:
    models = Models(s.models_dir, s.custom_models_file)
    for m in models.all():
        size = models.installed(m)
        print(f"{m['id']:28} {'downloaded' if size else '-':10} {m['repo']}/{m['file']}")
    return 0


def _download(s, model_id: str) -> int:
    models = Models(s.models_dir, s.custom_models_file)
    entry = models.get(model_id)
    if entry is None:
        print(f"Unknown model {model_id!r}; see `models`.")
        return 1
    downloader = Downloader(s.models_dir, lambda: Secrets(s).get("hf_token"))
    job = downloader.start(entry)
    while job.state in ("queued", "downloading", "verifying"):
        time.sleep(2)
        if job.total:
            print(f"\r{job.state:11} {job.received / 2**30:6.2f} / {job.total / 2**30:.2f} GB  {job.speed / 2**20:6.1f} MB/s",
                  end="", flush=True)
    print()
    print(job.state, job.error)
    return 0 if job.state == "done" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mnemerynne-balancer")
    parser.add_argument("--env-file", help=".env to read (default: the checkout's .env)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="the control page and /v1 balancer")
    sub.add_parser("devices", help="the cards llama-server sees, with VRAM in use")
    sub.add_parser("models", help="catalog and custom models, and which are downloaded")
    d = sub.add_parser("download", help="download one model by id")
    d.add_argument("model_id")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    s = load_settings(args.env_file)
    if args.command == "devices":
        return _devices(s)
    if args.command == "models":
        return _models(s)
    if args.command == "download":
        return _download(s, args.model_id)
    import uvicorn

    from mnemerynne_balancer.app import create_app
    try:
        app = create_app(s)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    uvicorn.run(app, host=s.host, port=s.port, log_level="info", proxy_headers=False)
    return 0
