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

## Target: the npc-rag box

It runs on the npc-rag node: Ubuntu 26.04, two RX 6800-class cards (gfx1030),
ROCm from Ubuntu's archive.

**Reuse:** llama.cpp is already built there. `npc-rag/linux/build-llama.sh`
put a HIP build in `/opt/llama.cpp/hip/bin` (tag b9481). The balancer uses
that binary (`LLAMA_SERVER=/opt/llama.cpp/hip/bin/llama-server`) instead of
building its own, so the two projects stay on the same llama.cpp. A
`--llama-server` override is still there for trying a newer tag.

**Share the cards with npc-rag.** npc-rag's student fleet already runs
llama-server copies on both cards (`STUDENT_COPIES_PER_DEVICE=2`, ports
9100+). The balancer never touches those processes. It hosts *its* models in
the VRAM they leave free:

- The VRAM check uses the card's real free memory from `rocm-smi`, which
  includes npc-rag's copies, not only what the balancer itself started.
- Each card tile shows npc-rag's copies too, read-only, so it's clear who
  holds what.
- **Order at boot:** the unit starts `After=npc-rag.service`, so the student
  fleet takes its VRAM first. The balancer then fits around it instead of
  starving the game NPC.
- **Later, optionally:** npc-rag could stop running its own copies
  (`STUDENT_FLEET=off`) and point `LLM_URL` at the balancer. Then one panel
  manages every model on the box. That is a deliberate step, not part of
  this build.

**Ports.** These avoid npc-rag's 9090–9093 and its copies counting up from
9100:

| Port | What | Reachable from |
|---|---|---|
| 9180 | balancer: control page and `/v1` API | the LAN (ufw) |
| 9181, 9182 | the balancer's llama-server on card 0 and card 1 | this machine only |

**Mnemerynne** reaches it over the LAN. The Mnemerynne box's
`MNEMERYNNE_LLM_URL` becomes `http://<npc-rag box>:9180/v1`, with the
balancer's API key. Erynne's generator moves from the RX 580 (Vulkan) to the
6800s (HIP), which is the turbocharge. The RX 580 unit stays installed as a
fallback: if the balancer is unreachable, switching back is a one-line
change.

## What it is not

- Not a cluster manager. Each box runs its own balancer for its own two
  cards; balancers never talk to each other. Spreading requests across
  boxes is Mnemerynne's job (see "Several nodes").
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
- **One balancer instance per card**, next to npc-rag's copies. Its settings
  are stored in `data/state.json`, so the same models come back after a
  reboot. The settings are:
  - the model
  - context
  - parallel slots
  - extra args
  - port (fixed per card: 127.0.0.1:9181 and :9182)
- **Before launching, the VRAM fit is checked**: the file size plus a KV-cache
  estimate from context × slots must fit the card's *currently free* VRAM,
  minus a 512 MB margin. If it doesn't, the panel says so and asks you to
  confirm instead of letting the card fail.
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

- A single address, `0.0.0.0:9180/v1/...` (LAN), with a bearer key
  (`BALANCER_API_KEY`). The key is required, since Mnemerynne calls it from
  another box.
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

**Mnemerynne** then sets `MNEMERYNNE_LLM_URL` to the balancer. For example,
Erynne could get both cards when both run LFM2.5. The embedder stays on the
Mnemerynne box next to the files it indexes. Only the generator moves.

## Several nodes (balanced by Mnemerynne)

Balancers don't know about each other. Each one serves only its own cards.
**Mnemerynne** holds the list and does the balancing between nodes.

**What each balancer provides:** `GET /v1/balancer/load` (API key required)
returns, for each model it's running:
- requests in flight
- total parallel slots
- whether it's ready

The figures are this node's own: its two cards, nothing else.

**What Mnemerynne does** (a change in the Mnemerynne repo):

- **Settings → Balancer nodes:** a comma-separated list, for example
  `http://10.0.5.20:9180, http://10.0.5.21:9180`, plus one API key used for
  all of them. It's saved in Mnemerynne's settings and applied without a
  restart.
- **Least busy node.** Every node is polled every 2 s in the background
  (1 s timeout), so a request never waits on a poll. Each answer goes to
  the ready node with the lowest in-flight ÷ slots for Erynne's model. When
  Mnemerynne sends a request, it adds one to that node's count straight
  away, so a burst spreads out instead of piling onto the node that looked
  idle 2 s ago. Ties go to the earlier node in the list.
- **Failover.** A node that refuses the connection, times out, rejects the
  key or answers 5xx sits out for 30 s. A request that fails *before any
  output* is retried once on the next-best node. Once Erynne has started
  streaming an answer, it isn't retried.
- **Fallback.** With the list empty, or every node down, Mnemerynne uses
  `MNEMERYNNE_LLM_URL` (the RX 580) as today, and the Settings page says
  so.
- **The Settings page** shows one row per node: reachable or not, the model
  running, in flight / slots, and answers sent there in the last hour.
- **Checked on save.** Each URL must be http(s), and its
  `/v1/balancer/load` must answer with the key before the list is accepted.

The embedder stays on the Mnemerynne box. Only the generator goes to the
nodes.

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

- **No build script.** It uses npc-rag's `/opt/llama.cpp/hip` build. If
  that's missing, `install.sh` says to run `npc-rag/linux/build-llama.sh`.
- `install.sh`, run as root on Ubuntu 26.04 and safe to re-run, does the
  following:
  - creates the `balancer` user (nologin; groups render, video)
  - installs the code in `/opt/mnemerynne-balancer` with `uv sync --frozen`,
    using the `/opt/uv` npc-rag already installed or installing it the same
    way
  - creates `/var/lib/mnemerynne-balancer/{models,logs,data}`
  - seeds `.env` with a generated password and API key, both printed once
  - installs and enables the unit
  - allows 9180 in ufw
- The unit runs `User=balancer`, `SupplementaryGroups=render video`,
  `After=npc-rag.service`, `KillMode=control-group`, `LimitMEMLOCK=infinity`,
  `ProtectSystem=strict` with the state directory writable, and
  `Restart=on-failure`.
- **Models live on the balancer's own disk**
  (`/var/lib/mnemerynne-balancer/models`). The install script checks there's
  room: LFM2.5 Q8 is about 2.7 GB, and Qwen3-8B Q4 about 5 GB.

## Tests

pytest with injectable fakes, as `test_student_fleet.py` does:

- a fake `popen` that "becomes healthy" after N polls or dies with a log
- a fake HF API and download server built on `httpx.MockTransport`, covering
  resume, sha256 mismatch and size cap
- `--list-devices` and `rocm-smi` output parsing from captured real samples
- the proxy: model routing, least-in-flight choice, retry on the other card,
  SSE pass-through, 503s
- `/v1/balancer/load`: correct counts while requests are in flight,
  requires the key
- in the Mnemerynne repo: least-busy choice across fake nodes, local count
  bumps during a burst, a dead node sitting out, retry before the first byte
  only, falling back to `MNEMERYNNE_LLM_URL`, URL list validation
- auth: login rate limit, CSRF, secrets never echoed
- state restored after a restart

## Build order

1. **Skeleton:** pyproject and uv, config, and cli `devices` (list cards with
   free VRAM from `rocm-smi`). Check on the npc-rag box: ROCm0/ROCm1 appear
   with npc-rag's copies counted as used.
2. **Downloads:** catalog, HF resolve, resumable verified download, and the
   CLI `download`.
3. **Instances:** start, stop and watch one model per card, saved in state,
   restored at boot.
4. **Page and auth:** status tiles, models, logs, settings, audit.
5. **Balancer:** `/v1` proxy with routing, retry and counters.
6. **Load endpoint:** `/v1/balancer/load`.
7. **Deploy:** install script and unit, README (reference-style, not a
   changelog).
8. **Point Mnemerynne at it.** In the Mnemerynne repo, add the Balancer
   nodes setting with least-busy balancing and failover. On the box, enter
   the node URLs and the key. Compare Erynne's speed against the RX 580
   with the same questions, and keep the RX 580 unit as a fallback.

## Open questions

- **"Busy" isn't a usable load signal.** Both cards report 99% busy at
  34–40 W, which is idle power. That's a sysfs reading artifact, so the
  card tiles show power and temperature and label busy as "reported". Load
  for balancing always comes from request counts, never from GPU busy.
- **Several models per card.** Should a card ever host two small models at
  once? The plan says one per card to keep VRAM accounting simple; two would
  need the fit check per card, not per instance.

Settled: VRAM. With npc-rag's copies running, each card uses 7.9 of 16 GB
(measured 2026-10-22), leaving about 8 GB free per card, or about 7.5 GB
after the 512 MB margin. Per card, that fits LFM2.5 Q8 (about 2.7 GB plus
roughly 1–2 GB of KV for 4 slots × 8k) with room to spare, or Qwen3-8B
Q4_K_M (about 5 GB plus about 1.2 GB of KV at 8k context) more tightly.
npc-rag keeps two copies per card. The repo is `mnemerynne-balancer`. The embedder stays on the
Mnemerynne box. The node list and balancing between nodes live only in Mnemerynne.
