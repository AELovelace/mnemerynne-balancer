#!/usr/bin/env bash
# Builds a CUDA llama-server inside NVIDIA's own cuda-devel container, for a host whose gcc is too new for the pinned
# nvcc (Fedora 44's gcc 16 against CUDA 12.8, which stops at gcc 14). cudart, cublas, libstdc++ and libgcc are linked
# statically, so the binary needs only the driver's libcuda.so on the host. No CUDA toolkit on the host, and no GPU
# in the container: only podman (or docker). Output: /opt/llama.cpp/cuda/bin/llama-server, the same place
# build-llama.sh puts its CUDA build, so install.sh and the page's CUDA backend find it.
#   sudo bash build-llama-container.sh
#   sudo CUDA_VERSION=12.8 CUDA_ARCHS="52;61" LLAMA_TAG=b9481 bash build-llama-container.sh
#   sudo IMAGE=docker.io/nvidia/cuda:12.8.1-devel-ubuntu22.04 bash build-llama-container.sh
set -euo pipefail

PREFIX="${PREFIX:-/opt/llama.cpp}"
LLAMA_TAG="${LLAMA_TAG:-b9481}"
CUDA_VERSION="${CUDA_VERSION:-12.8}"          # 12.x is the last that knows Maxwell/Pascal.
CUDA_ARCHS="${CUDA_ARCHS:-52;61;75;86}"       # 52 Maxwell (GTX 9xx, Titan X), 61 Pascal, 75 Turing, 86 Ampere.
IMAGE="${IMAGE:-}"                            # Empty: nvidia/cuda:<CUDA_VERSION>.1-devel-ubuntu24.04, else .0.
ENGINE="${ENGINE:-}"                          # podman or docker; empty: whichever is installed (podman first).
SRC="$PREFIX/src"
OUT="$PREFIX/cuda"

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo $0)"; exit 1; }
if [[ -z "$ENGINE" ]]; then
  ENGINE=$(command -v podman || command -v docker || true)
  [[ -n "$ENGINE" ]] || { echo "!! Neither podman nor docker is installed (Fedora: dnf install -y podman)."; exit 1; }
fi

# llama.cpp source on the host (shared with build-llama.sh), at the tag.
if [[ -d "$SRC/.git" ]]; then git -C "$SRC" fetch --tags origin; else git clone https://github.com/ggml-org/llama.cpp "$SRC"; fi
git -C "$SRC" checkout --force "$LLAMA_TAG"
[[ "$LLAMA_TAG" == master ]] && git -C "$SRC" pull --ff-only
mkdir -p "$OUT/bin"

# The image: NVIDIA's devel image for that toolkit, whose gcc nvcc accepts.
if [[ -z "$IMAGE" ]]; then
  for patch in 1 0; do
    candidate="docker.io/nvidia/cuda:${CUDA_VERSION}.${patch}-devel-ubuntu24.04"
    if "$ENGINE" pull "$candidate"; then IMAGE="$candidate"; break; fi
  done
  [[ -n "$IMAGE" ]] || { echo "!! No nvidia/cuda ${CUDA_VERSION}.x devel image; set IMAGE= to one from hub.docker.com/r/nvidia/cuda/tags"; exit 1; }
else
  "$ENGINE" pull "$IMAGE"
fi
echo "== building llama-server $LLAMA_TAG with CUDA $CUDA_VERSION (sm $CUDA_ARCHS) in $IMAGE"

# :Z relabels the mounts for SELinux. The build dir is separate from build-llama.sh's build-cuda on the host.
"$ENGINE" run --rm -v "$SRC:/src:Z" -v "$OUT:/out:Z" -e CUDA_ARCHS="$CUDA_ARCHS" "$IMAGE" bash -euo pipefail -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq git cmake ninja-build build-essential >/dev/null
  # The image ships NCCL (multi-node collectives) and llama.cpp links it dynamically when it finds it; the host has
  # no libnccl.so.2 and the balancer never needs it, so it goes before cmake looks.
  apt-get remove -y -qq "libnccl*" >/dev/null 2>&1 || true
  rm -rf /usr/local/cuda/lib64/libnccl* /usr/local/cuda/include/nccl*.h /usr/lib/x86_64-linux-gnu/libnccl*
  git config --global --add safe.directory "*"
  rm -rf /src/build-cuda-container
  cmake -S /src -B /src/build-cuda-container -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" -DCMAKE_DISABLE_FIND_PACKAGE_NCCL=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++"
  cmake --build /src/build-cuda-container --target llama-server -j "$(nproc)"
  install -m 755 /src/build-cuda-container/bin/llama-server /out/bin/llama-server
'
echo "== cuda: $("$OUT/bin/llama-server" --version 2>&1 | head -1)"
# Everything beyond glibc, libgomp and the driver's libcuda is a library the host must have: say so here, not at start.
echo "   links: $(ldd "$OUT/bin/llama-server" | grep -vE 'linux-vdso|ld-linux|libc\.so|libm\.so|libpthread|libdl|librt|libgomp|libgcc_s' | awk '{print $1 ($3=="not" ? " (MISSING)" : "")}' | tr '\n' ' ')"
"$OUT/bin/llama-server" --list-devices 2>&1 | sed 's/^/   /'
echo "Done. Next: sudo $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install.sh"
