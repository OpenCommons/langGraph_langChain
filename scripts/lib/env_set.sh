#!/usr/bin/env bash
# Usage: scripts/lib/env_set.sh KEY VALUE  — set KEY=VALUE in .env (create if absent).
set -euo pipefail
KEY="${1:?usage: env_set.sh KEY VALUE}"; VALUE="${2:?usage: env_set.sh KEY VALUE}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
[[ -f "$ENV_FILE" ]] || cp "$ROOT_DIR/.env.example" "$ENV_FILE"
if grep -qE "^${KEY}=" "$ENV_FILE"; then
    tmp="$(mktemp)"
    sed "s|^${KEY}=.*|${KEY}=${VALUE}|" "$ENV_FILE" > "$tmp" && cat "$tmp" > "$ENV_FILE"
    rm -f "$tmp"
else
    echo "${KEY}=${VALUE}" >> "$ENV_FILE"
fi
echo "[env] ${KEY}=${VALUE}"
