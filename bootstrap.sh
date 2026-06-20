#!/usr/bin/env bash
# Build the isolated Stable Audio 3 sidecar environment.
#
# Creates plugins/stable-audio-3/.venv with a CUDA torch + the stable-audio-3
# package, fully separate from the main OmniVoice venv (which pins a different
# torch and would otherwise conflict). Re-runnable / idempotent.
#
# Platforms
#   • Linux x86_64        → torch 2.7.1 (cu126) + flash-attn 2 prebuilt wheel.
#                            This matches what stable-audio-3 itself pins, so its
#                            deps resolve normally.
#   • Linux aarch64       → torch 2.10.0 (cu130). NVIDIA Grace-Blackwell parts
#     (GB10 "DGX Spark" /   (GB10 reports sm_121) have NO aarch64 CUDA wheels for
#      EdgeXpert, GH200)    torch 2.7.x — the earliest aarch64+CUDA torch is
#                            2.10.0 on the cu130 index. stable-audio-3 hard-pins
#                            torch==2.7.1, so we install torch ourselves and add
#                            the package with --no-deps (+ its non-torch deps),
#                            overriding that pin. flash-attn has no aarch64
#                            prebuilt wheel and 2.6.3 predates Blackwell, so it's
#                            skipped — SA3 falls back to torch SDPA (set
#                            SA3_BUILD_FLASH=1 to attempt a source build).
#
# Usage:
#   ./bootstrap.sh                 # build the env (no model download)
#   ./bootstrap.sh --with-model    # also download SA3 Medium (needs HF token)
#
# Overrides (env): SA3_CUDA=cu118|cu126|cu128|cu130  SA3_TORCH=2.7.1|2.10.0
#                  SA3_PYTHON=3.10  SA3_FLASH_WHEEL=<url>
#   flash-attn source build: SA3_BUILD_FLASH=1  SA3_FLASH_VER=2.8.3
#                  SA3_FLASH_ARCH=12.0 (sm_120, runs on GB10 sm_121; "9.0" GH200)
#                  MAX_JOBS=4  CUDA_HOME=/usr/local/cuda
#
# The SA3 Medium weights are gated on HuggingFace. Accept the license at
#   https://huggingface.co/stabilityai/stable-audio-3-medium
# then export a token before --with-model:
#   export HF_TOKEN=hf_xxx
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY="${SA3_PYTHON:-3.10}"

log() { printf '\n\033[1;36m[sa3-bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[sa3-bootstrap]\033[0m %s\n' "$*"; }

# ── Platform → torch build matrix ─────────────────────────────────────────────
ARCH="$(uname -m)"
# SA3_NO_DEPS=1 means torch != SA3's pinned 2.7.1, so install the package without
# letting it drag torch back, and provide its other runtime deps ourselves.
SA3_NO_DEPS=0
case "$ARCH" in
  x86_64)
    CUDA="${SA3_CUDA:-cu126}"
    TORCH_VER="${SA3_TORCH:-2.7.1}"
    ;;
  aarch64|arm64)
    CUDA="${SA3_CUDA:-cu130}"
    TORCH_VER="${SA3_TORCH:-2.10.0}"
    SA3_NO_DEPS=1
    log "Detected Linux ARM ($ARCH) — using torch $TORCH_VER ($CUDA) for NVIDIA Grace-Blackwell (DGX Spark / GH200)."
    ;;
  *)
    warn "Unrecognized arch '$ARCH' — defaulting to cu126 / torch 2.7.1. Override with SA3_CUDA / SA3_TORCH if this is wrong."
    CUDA="${SA3_CUDA:-cu126}"
    TORCH_VER="${SA3_TORCH:-2.7.1}"
    ;;
esac
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA"

# Prebuilt flash-attn wheel matching CUDA / torch / python (linux x86_64, cp310).
FLASH_WHEEL="${SA3_FLASH_WHEEL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl}"

# stable-audio-3's runtime deps minus torch/torchaudio (pyproject [project]),
# installed explicitly when we skip its dependency resolution on ARM.
SA3_DEPS=(
  "einops>=0.8.2" "einops-exts>=0.0.4" "numpy>=2.2.6" "packaging>=26.0"
  "safetensors>=0.7.0" "tqdm>=4.67.3" "huggingface-hub>=1.7.1"
  "transformers>=5.8.0" "soundfile>=0.13.1"
)

# ── Find a working uv (preferred) or fall back to stdlib venv + pip ───────────
# Note a pyenv shim named "uv" may sit on PATH but error out ("command not
# found"); we validate each candidate by actually running `uv --version`.
UV_BIN=""
_uv_ok() { [[ -n "$1" && -x "$1" ]] && "$1" --version >/dev/null 2>&1; }
for cand in "${UV:-}" "$HOME/.local/bin/uv" "$HOME/.pyenv/versions/3.10.13/bin/uv" "$(command -v uv 2>/dev/null || true)"; do
  if _uv_ok "$cand"; then UV_BIN="$cand"; break; fi
done

if [[ -n "$UV_BIN" ]]; then
  log "Using uv at $UV_BIN"
  # --no-config: this venv must NOT inherit the repo's [tool.uv] torch pin
  # (the whole point of isolating SA3 on its own torch).
  "$UV_BIN" venv --no-config --python "$PY" "$VENV"
  PIP=("$UV_BIN" pip install --no-config --python "$VENV/bin/python")
else
  log "uv not found — falling back to python -m venv + pip"
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
fi

log "Installing torch $TORCH_VER ($CUDA) ..."
"${PIP[@]}" "torch==$TORCH_VER" "torchaudio==$TORCH_VER" --index-url "$TORCH_INDEX"

if [[ "$SA3_NO_DEPS" == "1" ]]; then
  log "Installing stable-audio-3 (--no-deps; keeps our torch $TORCH_VER) ..."
  "${PIP[@]}" --no-deps "stable-audio-3 @ git+https://github.com/Stability-AI/stable-audio-3"
  log "Installing stable-audio-3 runtime deps (from PyPI) ..."
  "${PIP[@]}" "${SA3_DEPS[@]}"
else
  log "Installing stable-audio-3 (no torch reinstall) ..."
  "${PIP[@]}" "stable-audio-3 @ git+https://github.com/Stability-AI/stable-audio-3" \
    --no-build-isolation || \
    "${PIP[@]}" "stable-audio-3 @ git+https://github.com/Stability-AI/stable-audio-3"
fi

# ── flash-attn ───────────────────────────────────────────────────────────────
# Not a declared SA3 dependency; it speeds up attention and can avoid occasional
# glitch artifacts. Only an x86_64 prebuilt wheel exists. On NVIDIA ARM there are
# no aarch64+sm_121 wheels at all, and torch-native SDPA on Blackwell is as fast
# (often faster) and numerically identical to FA2 — so SDPA is the sane default.
# Opt into a source build with SA3_BUILD_FLASH=1 (Blackwell needs ~60–75 min on
# the Grace CPU; requires the CUDA toolkit / nvcc installed).
if [[ "${SA3_BUILD_FLASH:-0}" == "1" ]]; then
  # sm_121 (GB10) has no flash-attn kernels and the arch whitelist rejects 12.1;
  # build for sm_120 instead — it's binary-compatible with sm_121 via CUDA
  # forward compat. Override SA3_FLASH_ARCH for other parts (e.g. "9.0" GH200).
  FA_ARCH="${SA3_FLASH_ARCH:-12.0}"
  FA_VER="${SA3_FLASH_VER:-2.8.3}"
  log "Building flash-attn $FA_VER from source for sm_${FA_ARCH/./} (this can take 60–75 min) ..."
  "${PIP[@]}" ninja setuptools wheel packaging || true
  if TORCH_CUDA_ARCH_LIST="$FA_ARCH" \
     FLASH_ATTN_CUDA_ARCHS="${FA_ARCH/./}" \
     FLASH_ATTENTION_FORCE_BUILD=TRUE \
     MAX_JOBS="${MAX_JOBS:-4}" \
     CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
     "${PIP[@]}" "flash-attn==$FA_VER" --no-build-isolation --no-deps; then
    log "flash-attn $FA_VER built OK."
  else
    warn "flash-attn source build failed — SA3 will fall back to torch SDPA (fine on Blackwell). Check that nvcc / the CUDA toolkit is installed."
  fi
elif [[ "$ARCH" == "x86_64" ]]; then
  log "Installing flash-attn (prebuilt wheel) ..."
  "${PIP[@]}" "$FLASH_WHEEL" || \
    warn "Prebuilt flash-attn wheel failed — SA3 will fall back to torch SDPA. See README to build from source."
else
  warn "Skipping flash-attn on $ARCH (no prebuilt aarch64/sm_121 wheel) — SA3 uses torch SDPA, which is fast on Blackwell. Set SA3_BUILD_FLASH=1 to build it from source anyway."
fi

log "Verifying imports ..."
"$VENV/bin/python" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn NOT importable (SDPA fallback):", e)
import stable_audio_3
print("stable_audio_3 OK")
PYEOF

if [[ "${1:-}" == "--with-model" ]]; then
  log "Downloading SA3 Medium weights (gated — needs HF token) ..."
  HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
  "$VENV/bin/python" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or None
path = snapshot_download("stabilityai/stable-audio-3-medium", token=tok)
print("Downloaded to", path)
PYEOF
fi

log "Done. Env at $VENV"
