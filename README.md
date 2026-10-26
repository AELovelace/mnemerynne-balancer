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
| 9180 | control page (`/`) and the OpenAI-compatible API (`/v1`) | the LAN (ufw) |
| 9181, 9182 | the llama-server on card 0 and card 1 | this machine only |

npc-rag keeps 9090–9093 and its copies count up from 9100.

## Install (Ubuntu 26.04, as root)

npc-rag's llama.cpp build must be there first (`/opt/llama.cpp/hip`, from
`npc-rag/linux/build-llama.sh`).

```bash
git clone https://github.com/AELovelace/mnemerynne-balancer.git /opt/mnemerynne-balancer
/opt/mnemerynne-balancer/linux/install.sh          # ALLOW_FROM=172.27.0.0/16 to limit 9180 to the LAN
```

`install.sh` does the following, and is safe to run again (an existing
`.env` keeps its values):

1. creates the `balancer` user (in the render and video groups)
2. installs the exact Python packages from `uv.lock`, using npc-rag's
   `/opt/uv` and Python 3.12
3. fills in `.env`: a control page password and an API key, each printed
   once
4. installs and starts `mnemerynne-balancer.service`
5. opens 9180 in ufw

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

On Windows, stopping a card ends its llama-server at once, not after its
current request; there is no SIGTERM.

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
- **Reboots.** Whatever was running before a reboot starts again.
- **Settings tab.** Replace the API key (there's a **Generate** button) or
  set a Hugging Face token for gated repos. Both are write-only.
- **Activity tab.** Who signed in, started, stopped, downloaded or changed
  what, and from where. Secret values are never written.

Rough sizes on a 16 GB 6800 with npc-rag's two copies loaded (about 8 GB
free):
- LFM2.5 2.6B Q8 with 4 × 8k fits easily (about 3.9 GB).
- Qwen3 8B Q4 with 2 × 4k fits (about 6.6 GB).
- Bigger models need npc-rag to run one copy per card.

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
  name works (Mnemerynne sends `"local"`).
- **Two cards, same model:** the card with fewer requests in flight per
  slot answers, and the response header `X-Balancer-Card` says which one.
- **A card that doesn't answer:** the request is retried once on the other
  card.
- **`GET /v1/balancer/load`:** this node's load for each model:
  `{"id", "ready", "in_flight", "slots", "cards"}`. Mnemerynne will use it to
  pick between several balancer nodes. Each balancer only knows its own
  cards.
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
| `VRAM_MARGIN_MB` | `512` | kept free on top of the estimate |
| `BALANCER_DATA_DIR` | `/var/lib/mnemerynne-balancer` | models, logs, state, audit log |

After changing `.env`: `sudo systemctl restart mnemerynne-balancer`. That
stops the cards' llama-servers; they start again with the same models.

## Operations

```bash
journalctl -u mnemerynne-balancer -f                          # the balancer
tail -f /var/lib/mnemerynne-balancer/logs/card0.log            # card 0's llama-server
cd /opt/mnemerynne-balancer && sudo git pull && sudo linux/install.sh   # update
```

`data/state.json` holds what each card runs. `data/secrets.json` holds
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
