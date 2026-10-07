#!/usr/bin/env bash
# Start the host-native MLX OpenAI-compatible server (Apple Silicon).
# Usage: scripts/start-mlx.sh [hf-repo]   (default RadixArk/Muse-Glimmer-q4-MLX)
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}[mlx]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
die()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-RadixArk/Muse-Glimmer-q4-MLX}"
PORT=8081
URL="http://localhost:${PORT}/v1/models"
PID_FILE=/tmp/mlx.pid
LOG_FILE=/tmp/mlx.log

if curl -sf "$URL" >/dev/null 2>&1; then
    ok "MLX already running on :${PORT}"
    exit 0
fi

# Prefer the venv created by setup.sh, then whatever is on PATH.
if [[ -x "$ROOT_DIR/.venv-mlx/bin/mlx_lm.server" ]]; then
    MLX_SERVER="$ROOT_DIR/.venv-mlx/bin/mlx_lm.server"
elif command -v mlx_lm.server >/dev/null 2>&1; then
    MLX_SERVER="$(command -v mlx_lm.server)"
else
    die "mlx_lm.server not found. Run 'make setup' (choose MLX) or: python3 -m venv .venv-mlx && .venv-mlx/bin/pip install -r services/api/requirements-mlx.txt"
fi

info "Starting MLX server: $MODEL on :${PORT}"
"$MLX_SERVER" --model "$MODEL" --port "$PORT" >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

# First run downloads the model from Hugging Face, so allow a long-ish wait via
# MLX_START_TIMEOUT (default 15s once the weights are cached).
TIMEOUT="${MLX_START_TIMEOUT:-15}"
for _ in $(seq 1 "$TIMEOUT"); do
    sleep 1
    curl -sf "$URL" >/dev/null 2>&1 && { ok "MLX started (pid $(cat "$PID_FILE")) — $URL"; exit 0; }
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null || break
done
die "MLX failed to become ready within ${TIMEOUT}s — check $LOG_FILE (first run downloads the model; retry with MLX_START_TIMEOUT=300)"
