#!/usr/bin/env bash
# Gracefully stop the MLX server started by scripts/start-mlx.sh.
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${CYAN}[mlx]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

PID_FILE=/tmp/mlx.pid
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        info "Stopping MLX (pid $PID)..."
        kill "$PID"
        for _ in $(seq 1 10); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$PID" 2>/dev/null && { warn "MLX still running — sending SIGKILL"; kill -9 "$PID" || true; }
        ok "MLX stopped"
    else
        warn "MLX pid $PID not running — skipping"
    fi
    rm -f "$PID_FILE"
else
    warn "MLX was not started by this stack — leaving it running"
fi
