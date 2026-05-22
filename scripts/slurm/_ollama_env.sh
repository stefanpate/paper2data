#!/usr/bin/env bash
# Shared helpers: start an ollama server bound to this node and pull a model.
# Source from sbatch scripts after loading the ollama module.
#
# Required env going in:
#   OLLAMA_MODEL_TAG   e.g. "qwen2.5:7b-instruct"
# Set on exit:
#   OLLAMA_HOST        host:port the python client should hit
#   OLLAMA_PID         server pid (killed in cleanup)

set -euo pipefail

# Bind ollama to a free local port so it doesn't collide with other jobs on the node.
pick_free_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

start_ollama() {
  : "${OLLAMA_MODEL_TAG:?OLLAMA_MODEL_TAG must be set before start_ollama}"

  local port
  port="$(pick_free_port)"
  # Quest runs ollama inside Singularity; the server must bind 0.0.0.0 and the
  # env var must be mirrored as SINGULARITYENV_* to cross the container boundary.
  export OLLAMA_PORT="${port}"
  export OLLAMA_HOST="0.0.0.0:${port}"
  export SINGULARITYENV_OLLAMA_HOST="${OLLAMA_HOST}"

  # Keep model weights on $SCRATCH if available — they're big and node-local /tmp may be too small.
  export OLLAMA_MODELS="${OLLAMA_MODELS:-${SCRATCH:-$HOME}/ollama_models}"
  export SINGULARITYENV_OLLAMA_MODELS="${OLLAMA_MODELS}"
  mkdir -p "$OLLAMA_MODELS"

  echo "[ollama] starting server on $OLLAMA_HOST, models dir $OLLAMA_MODELS"
  ollama serve >"${SLURM_SUBMIT_DIR:-.}/ollama-${SLURM_JOB_ID:-local}.log" 2>&1 &
  OLLAMA_PID=$!
  export OLLAMA_PID

  # Wait for the server to accept connections (max ~60s).
  for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${port}/api/tags" >/dev/null 2>&1; then
      echo "[ollama] server up after ${i}s"
      break
    fi
    sleep 1
  done

  # Server bound to 0.0.0.0:port for the container; switch the CLIENT-facing
  # OLLAMA_HOST to 127.0.0.1:port so `ollama pull` and the python client connect
  # via loopback. The already-running server keeps its original bind.
  export OLLAMA_HOST="127.0.0.1:${port}"
  export SINGULARITYENV_OLLAMA_HOST="${OLLAMA_HOST}"

  echo "[ollama] pulling ${OLLAMA_MODEL_TAG}"
  ollama pull "${OLLAMA_MODEL_TAG}"
}

stop_ollama() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "[ollama] stopping server pid=$OLLAMA_PID"
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
  fi
}
