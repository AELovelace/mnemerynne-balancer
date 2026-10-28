#!/usr/bin/env bash
# Install or update mnemerynne-balancer on an Ubuntu 26.04 npc-rag node, or on Fedora (dnf and firewalld instead of
# apt and ufw). Run from the checkout as root:
#   sudo git clone https://github.com/AELovelace/mnemerynne-balancer.git /opt/mnemerynne-balancer
#   sudo /opt/mnemerynne-balancer/linux/install.sh
# Uses llama.cpp's HIP build (/opt/llama.cpp/hip) or CUDA build (/opt/llama.cpp/cuda) from linux/build-llama.sh
# (npc-rag's own build-llama.sh makes the same HIP build), and npc-rag's uv (/opt/uv, made here if it's missing).
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

# 1. Packages. Fedora's python3 already has venv; firewalld stands in for ufw.
if command -v dnf &>/dev/null; then
  FIREWALL=firewalld
  dnf install -y -q git python3 firewalld curl
else
  FIREWALL=ufw
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq git python3 python3-venv ufw curl
fi

# 2. llama-server: the HIP build (AMD), or the CUDA build (NVIDIA) when there's no HIP one. Both from build-llama.sh.
LLAMA=$(grep -E '^LLAMA_SERVER=' "$ROOT/.env" 2>/dev/null | tail -n1 | cut -d= -f2- || true)
LLAMA="${LLAMA:-/opt/llama.cpp/hip/bin/llama-server}"
if [[ ! -x "$LLAMA" && "$LLAMA" == /opt/llama.cpp/hip/bin/llama-server && -x /opt/llama.cpp/cuda/bin/llama-server ]]; then
  LLAMA=/opt/llama.cpp/cuda/bin/llama-server
fi
if [[ ! -x "$LLAMA" ]]; then
  echo "!! $LLAMA is missing. Build it first:  sudo $ROOT/linux/build-llama.sh"
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
sed -i "s|^LLAMA_SERVER=.*|LLAMA_SERVER=$LLAMA|" "$ENV"     # The one found in step 2 (unchanged unless that was CUDA).
grep -q '^LLAMA_SERVER=' "$ENV" || echo "LLAMA_SERVER=$LLAMA" >> "$ENV"
grep -q '^BALANCER_DATA_DIR=' "$ENV" || echo "BALANCER_DATA_DIR=$DATA_DIR" >> "$ENV"
# balancer (one llama-server per card) or bfgguf (one across all cards). Only used until the node saves a role.
grep -q '^BALANCER_ROLE=' "$ENV" || echo "BALANCER_ROLE=${BALANCER_ROLE:-balancer}" >> "$ENV"
chown root:"$SERVICE_USER" "$ENV"
chmod 640 "$ENV"

# 6. Disk: card models are 1-9 GB each; a BFGguf model (one across both cards) is 14-20 GB.
FREE_GB=$(df -BG --output=avail "$DATA_DIR" | tail -n1 | tr -dc '0-9')
if (( FREE_GB < 40 )); then
  echo "!! Only ${FREE_GB} GB free under $DATA_DIR; a couple of card models need 5-15 GB, and a BFGguf model up to 20 GB more."
fi

# 7. The unit (paths rewritten if the checkout isn't /opt/mnemerynne-balancer).
sed -e "s|/opt/mnemerynne-balancer|$ROOT|g" -e "s|/var/lib/mnemerynne-balancer|$DATA_DIR|g" \
    -e "s|^User=balancer|User=$SERVICE_USER|" -e "s|^Group=balancer|Group=$SERVICE_USER|" \
    "$ROOT/linux/mnemerynne-balancer.service" > /etc/systemd/system/mnemerynne-balancer.service
systemctl daemon-reload
systemctl enable mnemerynne-balancer.service
systemctl restart mnemerynne-balancer.service

# 8. Firewall: 9180 only; the cards' own ports (9181/9182) stay on 127.0.0.1.
if [[ $FIREWALL == firewalld ]]; then
  # The default zone; firewall-offline-cmd edits the saved config while firewalld is stopped.
  if firewall-cmd --state &>/dev/null; then FWCMD=(firewall-cmd --permanent); else FWCMD=(firewall-offline-cmd); fi
  if [[ -n "$ALLOW_FROM" ]]; then
    [[ $ALLOW_FROM == *:* ]] && FAMILY=ipv6 || FAMILY=ipv4
    "${FWCMD[@]}" --add-rich-rule="rule family=$FAMILY source address=$ALLOW_FROM port port=$PORT protocol=tcp accept"
  else
    "${FWCMD[@]}" --add-port="$PORT"/tcp
  fi
  if firewall-cmd --state &>/dev/null; then
    firewall-cmd --reload
  else
    echo "firewalld is off. To turn it on (ssh stays allowed in Fedora's default zones): systemctl enable --now firewalld"
  fi
else
  if [[ -n "$ALLOW_FROM" ]]; then
    ufw allow from "$ALLOW_FROM" to any port "$PORT" proto tcp comment mnemerynne-balancer
  else
    ufw allow "$PORT"/tcp comment mnemerynne-balancer
  fi
  ufw status | grep -q "Status: active" || echo "ufw is off. To turn it on without locking yourself out: ufw allow OpenSSH && ufw enable"
fi

echo
echo "== mnemerynne-balancer is running: http://$(hostname -f 2>/dev/null || hostname):$PORT/"
echo "   Cards:  sudo -u $SERVICE_USER $ROOT/.venv/bin/python -m mnemerynne_balancer devices"
echo "   Logs:   journalctl -u mnemerynne-balancer -f"
[[ -n "$NEW_PASSWORD" ]] && echo "   Control page password (shown once, also in $ENV): $NEW_PASSWORD"
[[ -n "$NEW_KEY" ]] && echo "   /v1 API key for Mnemerynne (shown once, also in $ENV): $NEW_KEY"
exit 0
