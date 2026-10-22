#!/usr/bin/env bash
# Install or update mnemerynne-balancer on an Ubuntu 26.04 npc-rag node. Run from the checkout as root:
#   sudo git clone https://github.com/AELovelace/mnemerynne-balancer.git /opt/mnemerynne-balancer
#   sudo /opt/mnemerynne-balancer/linux/install.sh
# Uses npc-rag's llama.cpp HIP build (/opt/llama.cpp/hip, from npc-rag/linux/build-llama.sh) and its uv (/opt/uv).
# Safe to run again: an existing .env keeps its values; only missing ones are filled in.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SERVICE_USER:-balancer}"
DATA_DIR="${DATA_DIR:-/var/lib/mnemerynne-balancer}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"      # npc-rag's install already put 3.12 under /opt/uv/python.
UV_HOME="${UV_HOME:-/opt/uv}"
PORT="${PORT:-9180}"
ALLOW_FROM="${ALLOW_FROM:-}"                  # e.g. 172.27.0.0/16: only that network may reach 9180 (default: any).

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo $0)"; exit 1; }
echo "mnemerynne-balancer checkout: $ROOT"

# 1. Packages.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3 python3-venv ufw curl

# 2. llama-server: npc-rag's HIP build.
LLAMA=$(grep -E '^LLAMA_SERVER=' "$ROOT/.env" 2>/dev/null | tail -n1 | cut -d= -f2- || true)
LLAMA="${LLAMA:-/opt/llama.cpp/hip/bin/llama-server}"
if [[ ! -x "$LLAMA" ]]; then
  echo "!! $LLAMA is missing. Build it with npc-rag's script first:  sudo /opt/npc-rag/linux/build-llama.sh"
  echo "   (or set LLAMA_SERVER in $ROOT/.env to another llama-server), then run this again."
  exit 1
fi

# 3. Service user: GPU access through render/video; home is the data directory.
if ! id "$SERVICE_USER" &>/dev/null; then
  useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi
usermod -aG render,video "$SERVICE_USER"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 750 "$DATA_DIR"

# 4. Python environment: uv with the exact versions in uv.lock.
if [[ ! -x "$UV_HOME/bootstrap/bin/uv" ]]; then
  python3 -m venv "$UV_HOME/bootstrap"
  "$UV_HOME/bootstrap/bin/pip" install --quiet --upgrade uv
fi
export UV_PYTHON_INSTALL_DIR="$UV_HOME/python"
"$UV_HOME/bootstrap/bin/uv" python install "$PYTHON_VERSION"
chmod -R a+rX "$UV_HOME"
(cd "$ROOT" && "$UV_HOME/bootstrap/bin/uv" sync --frozen --no-dev --python "$PYTHON_VERSION")

# 5. .env: from the example the first time; a password and API key when they're empty (printed once).
ENV="$ROOT/.env"
[[ -f "$ENV" ]] || cp "$ROOT/.env.example" "$ENV"
NEW_PASSWORD="" NEW_KEY=""
if ! grep -qE '^ADMIN_PASSWORD=.{8,}' "$ENV"; then
  NEW_PASSWORD=$("$ROOT/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(15))')
  sed -i "s|^ADMIN_PASSWORD=.*|ADMIN_PASSWORD=$NEW_PASSWORD|" "$ENV"
  grep -q '^ADMIN_PASSWORD=' "$ENV" || echo "ADMIN_PASSWORD=$NEW_PASSWORD" >> "$ENV"
fi
if ! grep -qE '^BALANCER_API_KEY=.{16,}' "$ENV"; then
  NEW_KEY=$("$ROOT/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))')
  sed -i "s|^BALANCER_API_KEY=.*|BALANCER_API_KEY=$NEW_KEY|" "$ENV"
  grep -q '^BALANCER_API_KEY=' "$ENV" || echo "BALANCER_API_KEY=$NEW_KEY" >> "$ENV"
fi
grep -q '^BALANCER_DATA_DIR=' "$ENV" || echo "BALANCER_DATA_DIR=$DATA_DIR" >> "$ENV"
chown root:"$SERVICE_USER" "$ENV"
chmod 640 "$ENV"

# 6. Disk: models are 1-9 GB each.
FREE_GB=$(df -BG --output=avail "$DATA_DIR" | tail -n1 | tr -dc '0-9')
if (( FREE_GB < 15 )); then
  echo "!! Only ${FREE_GB} GB free under $DATA_DIR; a couple of models need 5-15 GB."
fi

# 7. The unit (paths rewritten if the checkout isn't /opt/mnemerynne-balancer).
sed -e "s|/opt/mnemerynne-balancer|$ROOT|g" -e "s|/var/lib/mnemerynne-balancer|$DATA_DIR|g" \
    -e "s|^User=balancer|User=$SERVICE_USER|" -e "s|^Group=balancer|Group=$SERVICE_USER|" \
    "$ROOT/linux/mnemerynne-balancer.service" > /etc/systemd/system/mnemerynne-balancer.service
systemctl daemon-reload
systemctl enable mnemerynne-balancer.service
systemctl restart mnemerynne-balancer.service

# 8. Firewall: 9180 only; the cards' own ports (9181/9182) stay on 127.0.0.1.
if [[ -n "$ALLOW_FROM" ]]; then
  ufw allow from "$ALLOW_FROM" to any port "$PORT" proto tcp comment mnemerynne-balancer
else
  ufw allow "$PORT"/tcp comment mnemerynne-balancer
fi
ufw status | grep -q "Status: active" || echo "ufw is off. To turn it on without locking yourself out: ufw allow OpenSSH && ufw enable"

echo
echo "== mnemerynne-balancer is running: http://$(hostname -f 2>/dev/null || hostname):$PORT/"
echo "   Cards:  sudo -u $SERVICE_USER $ROOT/.venv/bin/python -m mnemerynne_balancer devices"
echo "   Logs:   journalctl -u mnemerynne-balancer -f"
[[ -n "$NEW_PASSWORD" ]] && echo "   Control page password (shown once, also in $ENV): $NEW_PASSWORD"
[[ -n "$NEW_KEY" ]] && echo "   /v1 API key for Mnemerynne (shown once, also in $ENV): $NEW_KEY"
exit 0
