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
  export OLLAMA_HOST="127.0.0.1:${port}"

  # Keep model weights on $SCRATCH if available — they're big and node-local /tmp may be too small.
  export OLLAMA_MODELS="${OLLAMA_MODELS:-${SCRATCH:-$HOME}/ollama_models}"
  mkdir -p "$OLLAMA_MODELS"

  echo "[ollama] starting server on $OLLAMA_HOST, models dir $OLLAMA_MODELS"
  ollama serve >"${SLURM_SUBMIT_DIR:-.}/ollama-${SLURM_JOB_ID:-local}.log" 2>&1 &
  OLLAMA_PID=$!
  export OLLAMA_PID

  # Wait for the server to accept connections (max ~60s).
  for i in $(seq 1 60); do
    if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "[ollama] server up after ${i}s"
      break
    fi
    sleep 1
  done

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
