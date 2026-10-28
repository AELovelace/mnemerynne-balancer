#!/usr/bin/env bash
# Builds llama.cpp's llama-server for the balancer's cards, on Ubuntu 26.04 or Fedora:
#   /opt/llama.cpp/hip/bin/llama-server     ROCm HIP, for AMD cards (install.sh's default LLAMA_SERVER)
#   /opt/llama.cpp/cuda/bin/llama-server    CUDA, for NVIDIA cards (install.sh uses it when there's no HIP build)
# Based on npc-rag/linux/build-llama.sh, same prefix and tag, so either script's HIP build serves both on a shared node.
# Usage: sudo bash build-llama.sh            (picks hip or cuda from the cards it finds; LLAMA_TAG=b9481)
#        sudo BACKEND=cuda bash build-llama.sh
#        sudo LLAMA_TAG=master bash build-llama.sh
#        sudo GPU_TARGETS="gfx1030;gfx1100" bash build-llama.sh    (AMD: rocminfo | grep gfx shows yours)
#        sudo CUDA_ARCHS="52;61" bash build-llama.sh              (NVIDIA: default is the cards in this machine)
#        sudo ROCM_SOURCE=amd bash build-llama.sh    (ROCm already installed from AMD's repo into /opt/rocm)
#        sudo CUDA_SOURCE=nvidia bash build-llama.sh (CUDA already installed from NVIDIA's repo into /usr/local/cuda)
# CUDA on Fedora: Fedora doesn't package nvcc, so install it from NVIDIA's CUDA repo first. If nvcc refuses Fedora's
# newer gcc, point it at an older one: CUDAHOSTCXX=/usr/bin/g++-14. Maxwell cards (9xx, Titan X) need CUDA 12: 13 dropped them.
set -euo pipefail

PREFIX="${PREFIX:-/opt/llama.cpp}"      # Install root.
LLAMA_TAG="${LLAMA_TAG:-b9481}"         # npc-rag's tested build; any llama.cpp tag or branch works.
BACKEND="${BACKEND:-}"                  # hip or cuda; empty: AMD cards -> hip, else NVIDIA cards -> cuda.
GPU_TARGETS="${GPU_TARGETS:-gfx1030}"   # RX 6800 / 6800 XT / 6900 XT.
CUDA_ARCHS="${CUDA_ARCHS:-}"            # Empty: llama.cpp builds for the cards in this machine.
ROCM_SOURCE="${ROCM_SOURCE:-archive}"   # archive = the distro's own ROCm packages; amd = leave /opt/rocm alone.
CUDA_SOURCE="${CUDA_SOURCE:-archive}"   # archive = Ubuntu's nvidia-cuda-toolkit; nvidia = leave /usr/local/cuda alone.
SRC="$PREFIX/src"

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo $0)"; exit 1; }

if [[ -z "$BACKEND" ]]; then
  if grep -qs 0x1002 /sys/class/drm/card[0-9]*/device/vendor; then
    BACKEND=hip
  elif nvidia-smi -L &>/dev/null; then
    BACKEND=cuda
  else
    BACKEND=hip
    echo "WARNING: no AMD card in sysfs and no working nvidia-smi; building HIP. Set BACKEND=cuda for NVIDIA."
  fi
fi
[[ "$BACKEND" == hip || "$BACKEND" == cuda ]] || { echo "unknown BACKEND '$BACKEND' (hip or cuda)"; exit 1; }
echo "== building llama-server with $BACKEND"

# Toolchain, and the GPU SDK llama.cpp compiles against.
if command -v dnf &>/dev/null; then
  dnf install -y git cmake ninja-build gcc-c++ libcurl-devel openssl-devel
  if [[ "$BACKEND" == hip && "$ROCM_SOURCE" == archive ]]; then
    dnf install -y hipcc rocm-hip-devel hipblas-devel rocblas-devel rocminfo rocm-smi || \
      echo "WARNING: some ROCm packages are missing; try ROCM_SOURCE=amd after AMD's installer."
  fi
else
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y git cmake ninja-build build-essential libcurl4-openssl-dev libssl-dev
  if [[ "$BACKEND" == hip && "$ROCM_SOURCE" == archive ]]; then
    apt-get install -y rocm || echo "WARNING: apt install rocm failed."
    apt-get install -y hipcc libhipblas-dev librocblas-dev rocminfo rocm-smi || \
      echo "WARNING: some ROCm -dev packages are missing from the archive; try ROCM_SOURCE=amd after AMD's installer."
  fi
  if [[ "$BACKEND" == cuda && "$CUDA_SOURCE" == archive ]]; then
    apt-get install -y nvidia-cuda-toolkit || \
      echo "WARNING: apt install nvidia-cuda-toolkit failed; try CUDA_SOURCE=nvidia after NVIDIA's installer."
  fi
fi
export PATH="$PATH:/opt/rocm/bin:/usr/local/cuda/bin"   # Where AMD's and NVIDIA's own repos put hipconfig and nvcc.

if [[ -d "$SRC/.git" ]]; then
  git -C "$SRC" fetch --tags origin
else
  git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" checkout --force "$LLAMA_TAG"
if [[ "$LLAMA_TAG" == master ]]; then
  git -C "$SRC" pull --ff-only   # A branch: bring it up to date.
fi

build() {  # build <name> <cmake flags...>: a static llama-server in $PREFIX/<name>/bin.
  local name="$1"; shift
  cmake -S "$SRC" -B "$SRC/build-$name" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "$@"
  cmake --build "$SRC/build-$name" --target llama-server -j "$(nproc)"
  install -D -m 755 "$SRC/build-$name/bin/llama-server" "$PREFIX/$name/bin/llama-server"
  echo "== $name: $("$PREFIX/$name/bin/llama-server" --version 2>&1 | head -1)"
  "$PREFIX/$name/bin/llama-server" --list-devices 2>&1 | sed 's/^/   /'
}

case "$BACKEND" in
  hip)  command -v hipconfig >/dev/null || { echo "!! No hipconfig: ROCm isn't installed, so there's nothing to build against."; exit 1; }
        HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" build hip -DGGML_HIP=ON -DGPU_TARGETS="$GPU_TARGETS" ;;
  cuda) command -v nvcc >/dev/null || { echo "!! No nvcc: install the CUDA toolkit (on Fedora, from NVIDIA's CUDA repo) first."; exit 1; }
        build cuda -DGGML_CUDA=ON ${CUDA_ARCHS:+-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"} ;;
esac
echo "Done. Next: sudo $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.sh"
