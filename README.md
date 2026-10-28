# mnemerynne-balancer

A web control panel for llama.cpp on a box with two AMD cards (ROCm), and an
OpenAI-compatible address in front of them. From the browser you download a
model (LFM2.5, Qwen3, or any GGUF on Hugging Face) and run it on card 0 or
card 1. Clients such as Mnemerynne send requests to one address, and each
request goes to the least busy card serving that model.

It runs on the npc-rag node, next to npc-rag's own copies, using npc-rag's
llama.cpp build. The plan and the reasons behind it are in
[PLAN.md](PLAN.md).

## Ports

| Port | What | Reachable from |
|---|---|---|
| 9180 | control page (`/`) and the OpenAI-compatible API (`/v1`) | the LAN (ufw or firewalld) |
| 9181, 9182 | the llama-server on card 0 and card 1 | this machine only |

npc-rag keeps 9090–9093 and its copies count up from 9100.

## Install (Ubuntu 26.04 or Fedora, as root)

```bash
git clone https://github.com/AELovelace/mnemerynne-balancer.git /opt/mnemerynne-balancer
/opt/mnemerynne-balancer/linux/build-llama.sh      # skip on an npc-rag node: its build is already there
/opt/mnemerynne-balancer/linux/install.sh          # ALLOW_FROM=172.27.0.0/16 to limit 9180 to the LAN
```

`build-llama.sh` builds llama-server from the same llama.cpp tag as
npc-rag's script. It picks the backend from the cards, NVIDIA first (an
NVIDIA box can also have an AMD integrated GPU):

- **AMD:** ROCm HIP into `/opt/llama.cpp/hip`, with ROCm from the distro's
  own packages. Set `GPU_TARGETS` if your cards aren't gfx1030
  (`rocminfo | grep gfx`).
- **NVIDIA:** CUDA into `/opt/llama.cpp/cuda`, built for the cards in the
  machine (or `CUDA_ARCHS`). Ubuntu gets nvcc from `nvidia-cuda-toolkit`.
  Fedora doesn't package nvcc, so install it from NVIDIA's CUDA repo first;
  if nvcc rejects Fedora's gcc, run with `CUDAHOSTCXX` set to an older g++.
  Maxwell cards need CUDA 12.
- **Vulkan (only with `BACKEND=vulkan`):** into `/opt/llama.cpp/vulkan`,
  for either kind of card. It needs no GPU SDK, only the card's driver and
  the Vulkan headers and `glslc` from the distro's own packages. Use it
  where CUDA is more trouble than it's worth, such as Maxwell cards on
  Fedora 44: CUDA 12 doesn't accept its gcc 16. On NVIDIA it needs NVIDIA's
  own driver, and for Maxwell that's the 580 branch (RPM Fusion's
  `akmod-nvidia-580xx`). Generation is usually somewhat slower than CUDA.

`BACKEND=hip`, `cuda` or `vulkan` overrides the choice. `install.sh` uses the
HIP build, else the CUDA one, else the Vulkan one, and writes it to
`LLAMA_SERVER` in `.env`. If the balancer ends up on the wrong one, pick it
in the Settings tab's **Backend** dropdown (below).

`install.sh` does the following, and is safe to run again (an existing
`.env` keeps its values):

1. creates the `balancer` user (in the render and video groups)
2. installs the exact Python packages from `uv.lock`, using npc-rag's
   `/opt/uv` and Python 3.12
3. fills in `.env`: a control page password and an API key, each printed
   once
4. installs and starts `mnemerynne-balancer.service`
5. opens 9180 in ufw, or on Fedora (found by `dnf`) in firewalld's default
   zone, as a rich rule when `ALLOW_FROM` is set

Check that it sees both cards:

```bash
sudo -u balancer /opt/mnemerynne-balancer/.venv/bin/python -m mnemerynne_balancer devices
# ROCm0    AMD Radeon RX 6800 XT (16368 MiB)  used 8100 / 16384 MiB, free 8284 MiB, 28 °C  -> card port 9181
```

## Windows with NVIDIA cards

The same code runs on Windows. VRAM and temperature come from `nvidia-smi`
instead of sysfs, and the llama-servers sit in a job object that Windows
closes when the balancer exits, so a crash never leaves one holding VRAM.

1. **llama.cpp:** download a `win-cuda-12.x-x64` release from
   github.com/ggml-org/llama.cpp/releases, plus the matching `cudart-…`
   zip, and unzip both into `C:\llama.cpp`. Maxwell cards (GTX Titan X,
   9xx) need CUDA **12** builds and driver 580 or older: CUDA 13 dropped
   them. The first start compiles kernels for the card and takes a minute.
2. **Balancer:**
   ```powershell
   git clone https://github.com/AELovelace/mnemerynne-balancer.git C:\mnemerynne-balancer
   cd C:\mnemerynne-balancer
   py -m uv sync
   copy .env.example .env    # then set the values below
   ```
   In `.env`: `LLAMA_SERVER=C:\llama.cpp\llama-server.exe`,
   `BALANCER_DATA_DIR=C:\mnemerynne-balancer\data`, `ADMIN_PASSWORD`, `BALANCER_API_KEY`.
3. **Check the cards:** `.venv\Scripts\python -m mnemerynne_balancer devices`
   should list `CUDA0` and `CUDA1` with their VRAM.
4. **Run it:** `.venv\Scripts\python -m mnemerynne_balancer serve`. To start
   it at boot, make a scheduled task that runs that command at startup
   ("Run whether user is logged on or not"), and open the port:
   `New-NetFirewallRule -DisplayName "mnemerynne-balancer" -Direction Inbound -Protocol TCP -LocalPort 9180 -RemoteAddress LocalSubnet -Action Allow`.

**Quick start for development:** `.\windows\start.ps1` from the checkout does
steps 2–4 in one go:
- runs `uv sync`;
- on the first run, makes `.env` with Windows values (llama-server at
  `C:\llama.cpp\llama-server.exe`, data in `.\data`) and a new password and
  API key, printed once;
- checks that llama-server is there;
- serves in that window, where Ctrl+C stops it and its llama-servers.

Options: `-Devices` lists the cards, `-LlamaServer <path>` points at another
llama-server.exe, and `-NoSync` skips `uv sync`.

On Windows, stopping a card ends its llama-server at once, not after its
current request; there is no SIGTERM.

## Fedora with NVIDIA cards and an older CUDA (as root)

`install-fedora-cuda.sh` wraps the two scripts above for a Fedora rig
whose cards want a **pinned, older CUDA** (Maxwell: CUDA 12 and driver 580)
and must not fall back to Vulkan (it contended for CPU cores with this
model). It adds RPM Fusion and NVIDIA's Fedora CUDA repo, installs the
driver if `nvidia-smi` is missing, installs `cuda-toolkit-12-8` and the
`gcc14` compat package nvcc 12 accepts, builds `/opt/llama.cpp/cuda` with
`build-llama.sh`, seeds `.env` with `BALANCER_BACKEND=cuda`, and runs
`install.sh`.

```bash
sudo git clone https://github.com/AELovelace/mnemerynne-balancer.git /opt/mnemerynne-balancer
sudo CUDA_VERSION=12.8 ALLOW_FROM=172.27.0.0/16 /opt/mnemerynne-balancer/linux/install-fedora-cuda.sh
```

- `CUDA_VERSION` (default `12.8`) picks the `cuda-toolkit-12-8` package and
  builds with its nvcc, whatever newer CUDA is also on the box.
- `CUDA_ARCHS` (default `52;61;75;86`): 52 Maxwell (GTX 9xx, Titan X), 61
  Pascal, 75 Turing, 86 Ampere; add `89` for Ada.
- `NVIDIA_DRIVER`: `auto` installs RPM Fusion's `akmod-nvidia` only when
  `nvidia-smi` is missing; `580xx` installs the legacy 580 branch (the last
  for Maxwell); `skip` leaves the driver alone. After a fresh driver install
  akmods still builds the module: `akmods --force && reboot`, then run the
  script again.
- `HOST_GCC` (default `14`) is the compat gcc nvcc builds with. If Fedora
  has no such package the build goes ahead with
  `-allow-unsupported-compiler`.
- Re-running is safe: repos, driver and toolkit are left alone once there.
- Flash attention works on CUDA, so a model added here doesn't need the
  `-fa off` the RX 6800 entries carry.
- **CPU contention:** every layer is on the card (`-ngl 999`), so
  llama-server only needs a few host threads. If it still fights other
  processes for cores, put `-t 2` in the card's "Extra args".

Then continue at [Everyday use](#everyday-use): download a model, start the
cards, and give Mnemerynne the node's URL and API key.

## Everyday use

Open `http://<node>:9180/` and sign in with the password.

- **Models tab.** Click **Download** on a catalog entry.
  - The size and SHA-256 come from Hugging Face. An interrupted download
    resumes, and a file that doesn't match is deleted, never used.
  - **Add from Hugging Face** lists a repository's `.gguf` files so you can
    add any of them.
  - Models live in `/var/lib/mnemerynne-balancer/models`.
  - **Chat templates.** A catalog entry can name a template shipped in
    `mnemerynne_balancer/templates/` (`"chat_template": "lfm2-no-think.jinja"`),
    which the card gets as `--chat-template-file`. The LFM2.5 2.6B entries use
    Mnemerynne's `lfm2-no-think.jinja` with `--reasoning-format deepseek`, the
    same as Mnemerynne's own generator: the model thinks only when a request
    asks, so JSON calls (memory extraction, follow-up questions) keep to their
    grammar. LFM2.5's own template thinks on every request and breaks them. A
    model added from Hugging Face gets no template; add `chat_template` to its
    entry in `data/custom-models.json` if it's an LFM2.5 fine-tune.
- **Cards tab.** Each card shows:
  - VRAM, including what npc-rag's copies use
  - temperature, hotspot and power
  - the model and its state (`stopped`, `loading`, `ready`, `failed`)
  - requests in flight per slot, and tokens/s

  **Start** or **Change model** picks a model, context per slot, slots and
  extra llama-server flags.
  - **VRAM check:** the balancer first estimates the VRAM needed (weights +
    KV cache + compute buffers) against what is free on the card now, minus
    `VRAM_MARGIN_MB`. If it won't fit, it says so and offers "Start anyway".
  - **Log** shows the card's llama-server output.
- **Crashes.** A crashed llama-server restarts after 5 s, then 10, 20, 40,
  up to 60 s. After 5 crashes in 10 minutes the card is marked `failed` with
  the end of its log.
- **Reboots.** Whatever was running before a reboot starts again, in the
  same role (see BFGguf below).
- **Settings tab.** Replace the API key (there's a **Generate** button) or
  set a Hugging Face token for gated repos. Both are write-only.
  - **Backend:** Auto, ROCm (AMD), CUDA (NVIDIA) or Vulkan. It picks the
    llama-server (`/opt/llama.cpp/hip`, `cuda` or `vulkan`, or
    `LLAMA_SERVER_ROCM`/`_CUDA`/`_VULKAN`), where VRAM and temperature come
    from (sysfs or `nvidia-smi`), and which devices count as cards. Vulkan
    also lists integrated GPUs, so when it sees NVIDIA cards it keeps only
    those and reads them from `nvidia-smi`. Auto uses
    `LLAMA_SERVER` and reads AMD cards first, which an AMD integrated GPU
    fools on an NVIDIA box. **Apply** checks that the build lists that
    backend's devices (or says why not and changes nothing), then restarts
    the running llama-servers on it with the same models. The choice
    survives restarts; `devices --backend cuda` tries one from the shell.
- **Activity tab.** Who signed in, started, stopped, downloaded or changed
  what, and from where. Secret values are never written.

Rough sizes on a 16 GB 6800 with npc-rag's two copies loaded (about 8 GB
free):
- LFM2.5 2.6B Q8 with 4 × 8k fits easily (about 3.9 GB).
- Qwen3 8B Q4 with 2 × 4k fits (about 6.6 GB).
- Bigger models need npc-rag to run one copy per card.

## BFGguf: one model across both cards

A node is either a **balancer** (one llama-server per card, the default) or
a **BFGguf** (one llama-server spanning every card, for a model too big for
one). The BFGguf runs `--device ROCm0,ROCm1 -sm layer -ts 1,1` (whole layers
split evenly) on the first card's port, with the same health checks, crash
restarts, metrics and log (`logs/bfgguf.log`) as a card.

- **Switching.** The **Role** switch at the top of the Cards tab, or
  `POST /v1/balancer/role` with the API key (below). The running set stops
  and its processes exit (VRAM freed) before the other set starts; the
  cards' own models come back when you switch back. New requests get 503
  "switching role" meanwhile. The role survives restarts and reboots;
  `BALANCER_ROLE` in `.env` only sets it on a node that has never saved one.
- **Refused (409)** when the BFGguf model isn't downloaded, or when its
  estimate doesn't fit the free VRAM of all cards added up, less
  `VRAM_MARGIN_MB` per card. "Switch anyway" (`"anyway": true`) skips the
  VRAM check.
- **While a BFGguf**, the card Start/Stop buttons and routes answer 409
  ("This is a BFGguf node right now."). Change its model, context or slots
  in its panel (or `POST /api/bfgguf`); that restarts only the BFGguf.
- **Routing.** A BFGguf node serves only requests that name its model (id,
  name or file name). `"local"` gets 404 "This is a BFGguf node; it only
  serves …", so Mnemerynne sends it elsewhere.
- **The model.** `qwen3.6-35b-a3b`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `Qwen3.6-35B-A3B-UD-IQ3_S.gguf`, 13.7 GB. Qwen doesn't publish its own
  GGUF of it. With npc-rag's copies holding about 8 GB per card, the two
  cards have about 15 GB free together, so this quant (2 slots × 8k) just
  fits; the Q3_K_S/Q3_K_M quants (15–17 GB) don't. It's a mixture of experts
  (about 3B active per token) with hybrid attention: only 10 of its 40 layers
  keep a KV cache, so context is cheap (about 20 MiB per 1,000 tokens).
  Download it from the Models tab or `python -m mnemerynne_balancer download
  qwen3.6-35b-a3b`. Catalog entries marked `"bfgguf": true` only appear in
  the BFGguf picker.
- **llama.cpp build.** The architecture is `qwen35moe` (Qwen3.5/3.6 MoE with
  Gated DeltaNet layers). npc-rag's HIP build must be new enough to know it;
  an older build fails to load the model ("unknown model architecture") and
  the log says so. Rebuild with `linux/build-llama.sh` (or npc-rag's). Flash
  attention stays off (`-fa off`), as for Qwen3 on the RX 6800s.
- **Sharded models.** GGUFs split as `name-00001-of-00003.gguf` are one
  model: listed once with the total size, every shard downloaded (resumed and
  SHA-256-checked one by one) and deleted together, and llama-server is given
  the first shard.

## The API

```bash
curl http://<node>:9180/v1/chat/completions \
  -H "Authorization: Bearer $BALANCER_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b-q8","messages":[{"role":"user","content":"Hello"}]}'
```

- **Routes:** `POST /v1/chat/completions`, `/v1/completions` and
  `/v1/embeddings`; `GET /v1/models`. Streaming (`"stream": true`) passes
  straight through.
- **Choosing the model:** `model` is a model id from the Models tab. The
  display name or file name also work. While only one model is running, any
  name works (Mnemerynne sends `"local"`), except on a BFGguf node.
- **Two cards, same model:** the card with fewer requests in flight per
  slot answers, and the response header `X-Balancer-Card` says which one.
- **A card that doesn't answer:** the request is retried once on the other
  card.
- **`GET /v1/balancer/load`:** this node's role and its load for each
  model. Mnemerynne uses it to pick between several balancer nodes. Each
  balancer only knows its own cards.
  ```json
  {"node": "npc-rag", "role": "balancer", "switching": false,
   "models": [{"id": "lfm2.5-2.6b-q8", "ready": true, "in_flight": 1, "slots": 8, "cards": [0, 1]}]}
  ```
  `role` is `"balancer"` or `"bfgguf"`; `switching` is true while a switch
  is under way. On a BFGguf node the one model lists every card in `cards`.
- **`POST /v1/balancer/role`:** `{"role": "bfgguf" | "balancer", "anyway":
  false}` with the API key. Answers with the load JSON above after the
  switch (the BFGguf is then still `"ready": false` while it loads), or 409
  / 422 with `{"detail": "..."}`. Switches are in the Activity tab.
- **Auth:** every `/v1` route needs the API key. Until one is set, `/v1`
  answers 503.

## Settings (.env)

| Key | Default | Meaning |
|---|---|---|
| `BALANCER_HOST`, `BALANCER_PORT` | `0.0.0.0`, `9180` | where the page and `/v1` listen |
| `ADMIN_PASSWORD` | (generated) | control page sign-in, 8+ characters |
| `BALANCER_API_KEY` | (generated) | bearer key for `/v1`; the page can replace it |
| `HF_TOKEN` | empty | only for gated repos; the page can set it |
| `LLAMA_SERVER` | `/opt/llama.cpp/hip/bin/llama-server` | npc-rag's HIP build |
| `CARD_PORTS` | `9181,9182` | one local port per card, in ROCm order |
| `VRAM_MARGIN_MB` | `512` | kept free on top of the estimate (per card) |
| `BALANCER_ROLE` | `balancer` | `balancer` or `bfgguf`; only for a node with no saved state yet |
| `BALANCER_BACKEND` | `auto` | `auto`, `rocm`, `cuda` or `vulkan`; only until the Settings tab saves one |
| `LLAMA_SERVER_ROCM`, `_CUDA`, `_VULKAN` | empty | each backend's llama-server; empty: build-llama.sh's build if it's there, else `LLAMA_SERVER` |
| `BALANCER_DATA_DIR` | `/var/lib/mnemerynne-balancer` | models, logs, state, audit log |

After changing `.env`: `sudo systemctl restart mnemerynne-balancer`. That
stops the cards' llama-servers; they start again with the same models.

## Operations

```bash
journalctl -u mnemerynne-balancer -f                          # the balancer
tail -f /var/lib/mnemerynne-balancer/logs/card0.log            # card 0's llama-server
cd /opt/mnemerynne-balancer && sudo git pull && sudo linux/install.sh   # update
```

`data/state.json` holds the role, what each card runs and the BFGguf's
settings. `data/secrets.json` holds
keys replaced from the page; it's owner-only. `data/audit.jsonl` is the
activity log.

## Security

- **Sign-in:** the page uses one password and an HMAC-signed, HttpOnly,
  SameSite=Strict cookie lasting 12 h. Restarting signs everyone out.
- **POSTs:** every POST also needs the page's own `X-Balancer-Admin` header,
  which a cross-site form can't send.
- **Rate limit:** 5 wrong passwords per address in 5 minutes gives a 429.
- **llama-server flags:** "Extra args" can't override the flags the
  balancer sets itself (`--host`, `--port`, `-m`, `--device`, `--path`, …),
  so a card can't be exposed to the LAN from the page.
- **Plain HTTP:** keep 9180 on the LAN (`ALLOW_FROM`), or put TLS in front.

## Development

```bash
py -3.14 -m uv sync          # or: uv sync
.venv/Scripts/python -m pytest -q
```

The tests use a fake `Popen`, fake llama-servers and a fake Hugging Face
(`httpx.MockTransport`), and a fake sysfs tree, so they run on Windows too.
