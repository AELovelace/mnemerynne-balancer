#!/usr/bin/env bash
# One shot for a Fedora rig with NVIDIA cards: the NVIDIA driver, a PINNED (older) CUDA llama-server build, then
# linux/install.sh. CUDA only, never Vulkan (it contended for CPU cores with this model). Run from the checkout as root:
#   sudo git clone https://github.com/AELovelace/mnemerynne-balancer.git /opt/mnemerynne-balancer
#   sudo /opt/mnemerynne-balancer/linux/install-fedora-cuda.sh
# Two ways to build, picked by BUILD:
#   host       NVIDIA's Fedora CUDA repo, cuda-toolkit-<CUDA_VERSION>, and a gcc nvcc accepts (gcc14 compat package),
#              then build-llama.sh. Only works while Fedora still ships such a gcc.
#   container  build-llama-container.sh: NVIDIA's cuda-devel image (its gcc fits its nvcc), static binary. No toolkit
#              on the host; needs podman. The way out when the host's gcc is too new (Fedora 44: gcc 16 vs CUDA 12).
#   auto       host if the compat gcc is installable, else container.
# Safe to run again: repos, driver and toolkit are left alone once there; install.sh keeps an existing .env.
#
#   CUDA_VERSION=12.8          12.x is the last that knows Maxwell/Pascal (13 dropped them)
#   CUDA_ARCHS="52;61;75;86"   sm_ targets: 52 Maxwell (GTX 9xx, Titan X), 61 Pascal, 75 Turing, 86 Ampere, 89 Ada
#   NVIDIA_DRIVER=auto         auto: RPM Fusion's akmod-nvidia only if nvidia-smi is missing; 580xx: the legacy 580
#                              branch (the last for Maxwell); skip: leave the driver alone
#   HOST_GCC=14                the gcc major the host build uses (Fedora's gccNN compat package)
#   LLAMA_TAG, ALLOW_FROM      passed on to the build script and install.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${BUILD:-auto}"
CUDA_VERSION="${CUDA_VERSION:-12.8}"
CUDA_ARCHS="${CUDA_ARCHS:-52;61;75;86}"
NVIDIA_DRIVER="${NVIDIA_DRIVER:-auto}"
HOST_GCC="${HOST_GCC:-14}"
CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"
CUDA_PKG="cuda-toolkit-${CUDA_VERSION//./-}"   # 12.8 -> cuda-toolkit-12-8
export CUDA_VERSION CUDA_ARCHS

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo $0)"; exit 1; }
[[ "$BUILD" =~ ^(auto|host|container)$ ]] || { echo "BUILD must be auto, host or container"; exit 1; }
. /etc/os-release
[[ "${ID:-}" == fedora ]] || { echo "!! This is for Fedora (/etc/os-release says ${ID:-?}); on Ubuntu use build-llama.sh + install.sh."; exit 1; }

# 1. Repos: RPM Fusion (driver) and NVIDIA's CUDA repo (published per Fedora release, a little behind).
dnf install -y curl kernel-devel-"$(uname -r)" kernel-headers
dnf install -y "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-${VERSION_ID}.noarch.rpm" \
  "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-${VERSION_ID}.noarch.rpm" 2>/dev/null || true
add_cuda_repo() {  # add_cuda_repo <fedora release>: true when the repo exists (already, or now).
  local v="$1" url="https://developer.download.nvidia.com/compute/cuda/repos/fedora$1/x86_64/cuda-fedora$1.repo"
  [[ -f /etc/yum.repos.d/cuda-fedora$v.repo ]] && return 0
  curl -fsI "$url" >/dev/null || return 1
  dnf config-manager addrepo --from-repofile="$url" 2>/dev/null || dnf config-manager --add-repo "$url"
}
if ! ls /etc/yum.repos.d/cuda-fedora*.repo &>/dev/null; then
  for v in "$VERSION_ID" 42 41 39; do add_cuda_repo "$v" && break; done
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

# 3. Which build. nvcc 12.x stops at gcc 14; once Fedora drops the gcc14 compat package only the container works.
if [[ "$BUILD" != container ]]; then
  if command -v "g++-$HOST_GCC" >/dev/null || dnf install -y "gcc$HOST_GCC" "gcc$HOST_GCC-c++" 2>/dev/null; then
    BUILD=host
  elif [[ "$BUILD" == host ]]; then
    echo "!! No gcc$HOST_GCC compat package on Fedora $VERSION_ID, and nvcc $CUDA_VERSION rejects its gcc $(gcc -dumpversion). Use BUILD=container."
    exit 1
  else
    echo "== No gcc$HOST_GCC compat package on Fedora $VERSION_ID: building in NVIDIA's cuda-devel container instead."
    BUILD=container
  fi
fi

if [[ "$BUILD" == host ]]; then
  # 4a. The pinned toolkit from NVIDIA's repo. Each release's repo starts at the CUDA current then (fedora42 and
  # later: 13.x), so an older toolkit comes from an older release's repo. nvcc already at $CUDA_HOME (.run installer)
  # is used as it is.
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    if ! dnf list --available "$CUDA_PKG" &>/dev/null; then
      for v in 41 39; do
        add_cuda_repo "$v" || continue
        dnf list --available "$CUDA_PKG" &>/dev/null && { echo "== $CUDA_PKG comes from the fedora$v repo"; break; }
      done
    fi
    dnf list --available "$CUDA_PKG" &>/dev/null || {
      echo "!! No repo offers $CUDA_PKG. Available: $(dnf list --available 'cuda-toolkit-1*' 2>/dev/null | grep -oE '^cuda-toolkit-1[0-9]-[0-9]+' | sort -u | tr '\n' ' ')"
      echo "   Pick one of those with CUDA_VERSION=, install the toolkit with NVIDIA's .run file into $CUDA_HOME, or BUILD=container."
      exit 1
    }
    dnf install -y "$CUDA_PKG"
  fi
  [[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "!! $CUDA_HOME/bin/nvcc is missing after installing $CUDA_PKG."; exit 1; }
  echo "$CUDA_HOME/lib64" > /etc/ld.so.conf.d/cuda-${CUDA_VERSION//./-}.conf && ldconfig
  export CC="gcc-$HOST_GCC" CXX="g++-$HOST_GCC" CUDACXX="$CUDA_HOME/bin/nvcc" PATH="$CUDA_HOME/bin:$PATH"
  export CUDAHOSTCXX="$(command -v "$CXX")"
  echo "== nvcc: $("$CUDACXX" --version | tail -1), host compiler $CUDAHOSTCXX"
  BACKEND=cuda CUDA_SOURCE=nvidia bash "$HERE/build-llama.sh"
else
  # 4b. The container build: podman, NVIDIA's image, a static binary into /opt/llama.cpp/cuda.
  command -v podman >/dev/null || command -v docker >/dev/null || dnf install -y podman
  bash "$HERE/build-llama-container.sh"
fi

# 5. The balancer itself. install.sh finds /opt/llama.cpp/cuda when there's no HIP build and writes it to .env.
ENV="$HERE/../.env"
[[ -f "$ENV" ]] || { cp "$HERE/../.env.example" "$ENV"; sed -i 's|^BALANCER_BACKEND=.*|BALANCER_BACKEND=cuda|' "$ENV"; }
bash "$HERE/install.sh"
