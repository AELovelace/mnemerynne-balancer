# mnemerynne-balancer: plan

A small web control panel for llama.cpp on a Linux box with two AMD (ROCm)
cards. From the browser you:

1. pick a model from a short catalog (LFM2.5, Qwen3) and download it,
2. host it on card 0 or card 1 (one llama-server per card),
3. watch it load, see its health, VRAM and temperature, read its log, stop it
   or switch models.

It also gives one OpenAI-compatible address that forwards each request to
whichever card is serving the requested model, and spreads requests across
both cards when both serve the same one. That address is what Mnemerynne (and
anything else) points at.

It follows **npc-rag**'s patterns, which already solve most of this:

- `student_fleet.py`: download, verify, launch and watch llama-server copies
  per GPU.
- `admin.py` and `admin_page.html`: password, cookie, CSRF, an audit log, and
  a single HTML page with no build step.
- `linux/`: build script, install script, systemd unit.

This project is narrower: it only manages llama.cpp, and it owns the
llama-server processes itself.

## What it is not

- Not multi-host. One box, two cards. A second box would run its own copy.
- Not a model zoo. A curated catalog plus "add a GGUF from a Hugging Face
  repo"; no arbitrary URLs.
- Not exposed to the internet. Plain HTTP on the LAN, as npc-rag's control page
  is; put Caddy in front if it ever needs to leave the LAN.

## Layout (mirrors npc-rag)

```
mnemerynne_balancer/
  cli.py          argparse: serve, devices, download, check
  config.py       .env loader (real env vars win) -> frozen Settings dataclass
  catalog.json    curated models: HF repo, file, quant, context, size, notes
  downloads.py    HF download: resolve size and sha256 from the HF API, stream to
                  a temp file, verify, rename; progress for the page
  devices.py      `llama-server --list-devices` -> ROCm0/ROCm1 with VRAM;
                  rocm-smi for temperature, usage and power
  instances.py    one llama-server per card: launch, /health poll, watchdog with
                  backoff, log file, pidfile cleanup (student_fleet.py's pattern)
  proxy.py        OpenAI-compatible forwarding (/v1/chat/completions,
                  /v1/completions, /v1/embeddings, /v1/models), streaming
  app.py          FastAPI: page, JSON API, auth, CSRF, audit log
  page.html       the control panel (vanilla JS, inline CSS, dark mode)
linux/
  build-llama.sh  llama.cpp with HIP for the cards' gfx targets into
                  /opt/llama.cpp-rocm (pinned tag)
  install.sh      user, venv, .env, unit, ufw; re-runnable
  mnemerynne-balancer.service
tests/
.env.example  .gitattributes (LF for linux/**)  pyproject.toml  README.md
```

The Python package uses **uv** with exact pins (`uv.lock`), as Mnemerynne
does, rather than npc-rag's lower-bound `requirements.txt`. Dependencies:
fastapi, uvicorn, httpx. Nothing else is needed.

## Models

`catalog.json` lists a handful of known-good entries, for example:

| Entry | HF repo | Quant | Use |
|---|---|---|---|
| LFM2.5 2.6B | `LiquidAI/LFM2.5-…-GGUF` | Q4_K_M, Q8_0 | Erynne's generator |
| Qwen3 4B / 8B | `Qwen/Qwen3-…-GGUF` | Q4_K_M | general chat |
| Qwen3-Embedding 0.6B | `Qwen/Qwen3-Embedding-0.6B-GGUF` | Q8_0 | embeddings (`--embedding`) |

The exact repo and file names get checked against Hugging Face when the catalog
is written. Each entry also holds launch defaults: context size, `--jinja`
plus a chat template file if needed (Mnemerynne's `lfm2-no-think.jinja`),
`--embedding`, and `--pooling` for the embedder.

**Download:**

- The size and sha256 come from the HF API (`/api/models/{repo}/tree/main`
  lists each LFS file's `oid` = sha256 and `size`).
- The file streams to `models/.partial/<file>`, which resumes with an HTTP
  Range request if interrupted.
- It is checked against that size and sha256, then renamed into
  `models/<repo>/<file>`. That is npc-rag's `_matches` and `_install_lock`
  pattern, plus resume.
- The page polls `/api/downloads` for bytes, speed and ETA, and can cancel.
- Before starting it checks free disk space.
- `HF_TOKEN` (optional, write-only in `.env`) covers gated repos.
- Deleting a model is refused while a card is serving it.

**Custom models:** paste an HF repo; the page lists its `.gguf` files with
sizes; pick one. They are saved to `data/custom-models.json` and treated like
catalog entries with default launch settings that you can edit.

## Cards and instances

- `devices.py` runs `llama-server --list-devices` from the ROCm build and
  shows each card with its name and VRAM. Each instance gets
  `--device ROCmN`, as npc-rag does, so no `HIP_VISIBLE_DEVICES`.
  `rocm-smi --showtemp --showuse --showmemuse --showpower --json` feeds the
  tiles.
- **One instance per card.** Its settings are stored in `data/state.json`, so
  the same models come back after a reboot. The settings are:
  - the model
  - context
  - parallel slots
  - extra args
  - port (fixed per card, e.g. 127.0.0.1:9200 and :9201)
- **Before launching, the VRAM fit is checked**: the file size plus a KV-cache
  estimate from context × slots must fit the card's free VRAM. If it doesn't,
  the panel says so and asks you to confirm instead of letting the card fail.
- **Launch:** `-m … --host 127.0.0.1 --port … -c … -np … -ngl 999 --device
  ROCmN` plus the entry's flags. The panel polls `/health` until it returns
  200, showing "loading" while llama-server answers 503.
  - If the process exits, the panel shows the last 40 log lines.
  - A watchdog restarts a crashed instance with backoff (5 s doubling to 60 s).
    After 5 failures in 10 minutes it gives up and marks the card **failed**.
- **Switching models on a card** means stop, wait for the process to exit,
  then start the new one. The proxy answers 503 "loading" for that card
  meanwhile.
- **Ownership:** child processes live in the service's cgroup
  (`KillMode=control-group`) and get PDEATHSIG, so they never outlive the
  panel. A pidfile cleans up leftover copies at start.

## The balancer (proxy.py)

- A single address, `0.0.0.0:9190/v1/...` (LAN), with an optional bearer
  key (`BALANCER_API_KEY`).
- `/v1/models` lists the models that are running.
- A request's `model` field picks the instances serving that model:
  - one instance: the request goes there;
  - two instances: whichever has fewer requests in flight (counted by the
    proxy itself), with ties going to the lower card number;
  - no `model` field: whatever runs on card 0 (so simple clients work).
- Streaming (SSE) passes straight through. A client disconnecting cancels
  the upstream request.
- The proxy sets no timeout of its own beyond llama-server's.
- If the chosen instance is down, the request is tried once on the other
  instance serving the same model. Otherwise it gets 503 with a clear reason.
- The page shows per-card counters: requests, in flight, last error, and
  tokens/s from llama-server's `/metrics` (`--metrics`).

**Mnemerynne** then sets `MNEMERYNNE_LLM_URL` (and, if the embedder moves
here, its embed URL) to the balancer. For example, Erynne could get both
cards when both run LFM2.5.

## Control page

A single `page.html`, plain JS polling `/api/status` every 2 s:

- **Cards:** two tiles. Each shows the card name, VRAM bar, temperature and
  usage, then the model, state (stopped / loading / ready / failed), port,
  uptime, requests and tokens/s, with buttons for Start, Stop, Change model
  and Log.
- **Models:** what's downloaded (size, where it's in use, delete), the catalog
  with Download buttons, progress bars, and "add from Hugging Face".
- **Settings:** per-card context, slots and extra args; the API key
  (write-only); the HF token (write-only).
- **Log:** the last 200 lines of a card's llama-server log, refreshed live.
- **Activity:** the audit log (who started or stopped what, downloads,
  setting changes).

Auth is copied from npc-rag's `admin.py`:

- one `ADMIN_PASSWORD`, which the install script generates and prints once
- an HMAC-signed HttpOnly SameSite=Strict cookie lasting 12 h
- a custom-header CSRF check on every POST
- 5 failed logins per IP per 5 min
- CSP, no-store and frame-deny headers
- a JSONL audit log that never contains secrets

The proxy's `/v1` routes use the API key, not the cookie.

## Deploy (linux/)

- `build-llama.sh` builds llama.cpp with `-DGGML_HIP=ON`,
  `-DAMDGPU_TARGETS=<from rocminfo>` and a pinned `LLAMA_TAG` into
  `/opt/llama.cpp-rocm/bin`. It's safe to re-run, and a `--tag` flag upgrades.
- `install.sh`, run as root, does the following:
  - creates the `balancer` user (nologin; groups render, video)
  - installs the code in `/opt/mnemerynne-balancer` with `uv sync --frozen`
  - creates `/var/lib/mnemerynne-balancer/{models,logs,data}`
  - seeds `.env` with the password
  - installs and enables the unit
  - opens the ports with firewalld or ufw, LAN-only if a zone is given
  - prints the URL and password
- The unit runs `User=balancer`, `SupplementaryGroups=render video`,
  `KillMode=control-group`, `LimitMEMLOCK=infinity`, `ProtectSystem=strict`
  with the state directory writable, and `Restart=on-failure`.

## Tests

pytest with injectable fakes, as `test_student_fleet.py` does:

- a fake `popen` that "becomes healthy" after N polls or dies with a log
- a fake HF API and download server built on `httpx.MockTransport`, covering
  resume, sha256 mismatch and size cap
- `--list-devices` and `rocm-smi` output parsing from captured real samples
- the proxy: model routing, least-in-flight choice, retry on the other card,
  SSE pass-through, 503s
- auth: login rate limit, CSRF, secrets never echoed
- state restored after a restart

## Build order

1. **Skeleton:** pyproject and uv, config, cli `devices` (list cards), and
   `build-llama.sh`. Check: both cards appear as ROCm0/ROCm1.
2. **Downloads:** catalog, HF resolve, resumable verified download, and the
   CLI `download`.
3. **Instances:** start, stop and watch one model per card, saved in state,
   restored at boot.
4. **Page and auth:** status tiles, models, logs, settings, audit.
5. **Balancer:** `/v1` proxy with routing, retry and counters.
6. **Deploy:** install script and unit, README (reference-style, not a
   changelog). Then point Mnemerynne at it.

## Open questions

- **Which box and which cards?**
  - The Mnemerynne box's RX 580 is gfx803. Current ROCm doesn't support it
    (that was the old-HIP experiment), so it would stay on Vulkan.
  - If the two ROCm cards are in another machine (like the npc-rag nodes,
    gfx1030), build for their targets.
  - If one card can't run ROCm, the plan could allow a Vulkan build per card.
    It's a small addition, since npc-rag already falls back from HIP to
    Vulkan.
- **The name.** You said both "mnemerynne-lbalancer" and
  "mnemerynne-balancer"; the folder is `mnemerynne-balancer`. Rename before
  the first push if you meant the other.
- **Embeddings.** Should the embedder move here too, or stay on the 1050 Ti
  next to Mnemerynne?
- **Several models per card.** Should a card ever host two small models at
  once, such as LFM2.5 plus the embedder? The plan says one per card to keep
  VRAM accounting simple; two would need the fit check per card, not per
  instance.
