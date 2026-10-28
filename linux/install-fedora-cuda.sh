#!/usr/bin/env bash
# One shot for a Fedora rig with NVIDIA cards: the NVIDIA driver, a PINNED (older) CUDA toolkit from NVIDIA's own
# Fedora repo, a compat gcc that nvcc accepts, then linux/build-llama.sh (CUDA only, never Vulkan: it contended for
# CPU cores with this model) and linux/install.sh. Run from the checkout as root:
#   sudo git clone https://github.com/AELovelace/mnemerynne-balancer.git /opt/mnemerynne-balancer
#   sudo /opt/mnemerynne-balancer/linux/install-fedora-cuda.sh
# Safe to run again: repos, driver and toolkit are left alone once there; build-llama.sh rebuilds llama-server;
# install.sh keeps an existing .env.
#
#   CUDA_VERSION=12.8          cuda-toolkit-12-8; 12.x is the last that knows Maxwell/Pascal (13 dropped them)
#   CUDA_ARCHS="52;61;75;86"   sm_ targets: 52 Maxwell (GTX 9xx, Titan X), 61 Pascal, 75 Turing, 86 Ampere, 89 Ada
#   NVIDIA_DRIVER=auto         auto: RPM Fusion's akmod-nvidia only if nvidia-smi is missing; 580xx: the legacy 580
#                              branch (the last for Maxwell); skip: leave the driver alone
#   HOST_GCC=14                the gcc major nvcc builds with (Fedora's gccNN compat package); empty: the default gcc
#   LLAMA_TAG, ALLOW_FROM      passed on to build-llama.sh and install.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VERSION="${CUDA_VERSION:-12.8}"
CUDA_ARCHS="${CUDA_ARCHS:-52;61;75;86}"
NVIDIA_DRIVER="${NVIDIA_DRIVER:-auto}"
HOST_GCC="${HOST_GCC-14}"
CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"
CUDA_PKG="cuda-toolkit-${CUDA_VERSION//./-}"   # 12.8 -> cuda-toolkit-12-8

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo $0)"; exit 1; }
. /etc/os-release
[[ "${ID:-}" == fedora ]] || { echo "!! This is for Fedora (/etc/os-release says ${ID:-?}); on Ubuntu use build-llama.sh + install.sh."; exit 1; }

# 1. Repos: RPM Fusion (driver) and NVIDIA's CUDA repo (the pinned toolkit; published per Fedora release, a little behind).
dnf install -y curl kernel-devel-"$(uname -r)" kernel-headers
dnf install -y "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-${VERSION_ID}.noarch.rpm" \
  "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-${VERSION_ID}.noarch.rpm" 2>/dev/null || true
if ! ls /etc/yum.repos.d/cuda-fedora*.repo &>/dev/null; then
  for v in "$VERSION_ID" 42 41 39; do
    url="https://developer.download.nvidia.com/compute/cuda/repos/fedora$v/x86_64/cuda-fedora$v.repo"
    if curl -fsI "$url" >/dev/null; then
      dnf config-manager addrepo --from-repofile="$url" 2>/dev/null || dnf config-manager --add-repo "$url"
      break
    fi
  done
fi

# 2. Driver. Maxwell (GTX 9xx, Titan X) stops at 580: with a newer akmod-nvidia the card simply isn't seen.
case "$NVIDIA_DRIVER" in
  skip)  ;;
  580xx) dnf install -y akmod-nvidia-580xx xorg-x11-drv-nvidia-580xx-cuda ;;
  auto)  command -v nvidia-smi >/dev/null || dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda ;;
  *)     echo "NVIDIA_DRIVER must be auto, 580xx or skip"; exit 1 ;;
esac
if ! nvidia-smi -L &>/dev/null; then
  echo "!! nvidia-smi doesn't see a card yet. After a fresh driver install akmods is still building the module:"
  echo "   akmods --force && reboot, then run this script again. (Maxwell: NVIDIA_DRIVER=580xx.)"
  exit 1
fi
nvidia-smi -L | sed 's/^/   /'

# 3. The pinned toolkit, and a gcc it accepts. nvcc 12.x stops at gcc 14; Fedora's default is newer.
dnf install -y "$CUDA_PKG"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "!! $CUDA_HOME/bin/nvcc is missing after installing $CUDA_PKG."; exit 1; }
echo "$CUDA_HOME/lib64" > /etc/ld.so.conf.d/cuda-${CUDA_VERSION//./-}.conf && ldconfig
export CC=gcc CXX=g++ CUDAFLAGS=""
if [[ -n "$HOST_GCC" ]]; then
  if dnf install -y "gcc$HOST_GCC" "gcc$HOST_GCC-c++" 2>/dev/null && command -v "g++-$HOST_GCC" >/dev/null; then
    CC="gcc-$HOST_GCC" CXX="g++-$HOST_GCC"
  else
    echo "!! No gcc$HOST_GCC compat package on this Fedora; building with -allow-unsupported-compiler."
    CUDAFLAGS="-allow-unsupported-compiler"
  fi
fi
export CUDACXX="$CUDA_HOME/bin/nvcc" CUDAHOSTCXX="$(command -v "$CXX")" PATH="$CUDA_HOME/bin:$PATH"
echo "== nvcc: $("$CUDACXX" --version | tail -1), host compiler $CUDAHOSTCXX ${CUDAFLAGS:+($CUDAFLAGS)}"

# 4. llama-server with CUDA (build-llama.sh builds the one backend asked for), then the balancer itself.
BACKEND=cuda CUDA_SOURCE=nvidia CUDA_ARCHS="$CUDA_ARCHS" bash "$HERE/build-llama.sh"
ENV="$HERE/../.env"
[[ -f "$ENV" ]] || { cp "$HERE/../.env.example" "$ENV"; sed -i 's|^BALANCER_BACKEND=.*|BALANCER_BACKEND=cuda|' "$ENV"; }
bash "$HERE/install.sh"
