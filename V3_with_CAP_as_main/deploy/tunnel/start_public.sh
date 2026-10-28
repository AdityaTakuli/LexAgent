#!/usr/bin/env bash
# Serve LexAgent from this machine to https://lexagent.velocitytechnologies.in
#
#   deploy/tunnel/start_public.sh        (from V3_with_CAP_as_main; Ctrl+C to stop)
#
# Starts the API (ui/server.py), opens a Cloudflare quick tunnel to it, and publishes the
# tunnel's address in tunnel.json on the Hugging Face dataset, where the public page reads it.
# On exit the address is cleared, so the page says the backend is offline.
# Needs: .env with LEXAGENT_LLM_API_KEY, `hf auth login`, cloudflared, and a built corpus in data/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PORT="${LEXAGENT_PORT:-8010}"
SITE="${LEXAGENT_SITE:-https://lexagent.velocitytechnologies.in}"
CONFIG_REPO="${LEXAGENT_CONFIG_REPO:-iamadityatakuli/lexagent-corpus}"
CLOUDFLARED="${CLOUDFLARED:-$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")}"
PY="$ROOT/.venv/bin/python"
HF="$ROOT/.venv/bin/hf"
LOGS="$ROOT/data/tunnel"
mkdir -p "$LOGS"

[ -x "$CLOUDFLARED" ] || { echo "cloudflared not found (looked at $CLOUDFLARED)" >&2; exit 1; }
grep -q '^LEXAGENT_LLM_API_KEY=.' .env 2>/dev/null || { echo "Put your FreeLLMAPI key in .env (LEXAGENT_LLM_API_KEY)" >&2; exit 1; }
if ss -ltn | grep -q ":$PORT "; then echo "Port $PORT is in use. Stop the other server first." >&2; exit 1; fi

publish() {  # publish <api base or empty> <commit message>
  local file="$LOGS/tunnel.json"
  printf '{"apiBase": "%s", "updated_at": "%s"}\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$file"
  "$HF" upload "$CONFIG_REPO" "$file" tunnel.json --repo-type dataset --commit-message "$2" > /dev/null
}

SERVER_PID="" TUNNEL_PID=""
cleanup() {
  trap - EXIT INT TERM
  echo; echo "Stopping. Marking the site offline..."
  publish "" "LexAgent backend offline" || echo "Could not update tunnel.json; the page will just fail to reach the old address."
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2> /dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2> /dev/null || true
  wait 2> /dev/null || true
  echo "Stopped."
}
trap cleanup EXIT INT TERM

echo "Starting the API on port $PORT (log: $LOGS/server.log)..."
HF_HUB_OFFLINE=1 LEXAGENT_ALLOWED_ORIGINS="$SITE" "$PY" ui/server.py --host 127.0.0.1 --port "$PORT" --preload > "$LOGS/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/api/status" > /dev/null && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/api/status" > /dev/null || { echo "The API did not start; see $LOGS/server.log" >&2; exit 1; }

echo "Opening the tunnel (log: $LOGS/cloudflared.log)..."
"$CLOUDFLARED" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$LOGS/cloudflared.log" 2>&1 &
TUNNEL_PID=$!
URL=""
for _ in $(seq 1 60); do
  URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOGS/cloudflared.log" | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "$URL" ] || { echo "The tunnel did not come up; see $LOGS/cloudflared.log" >&2; exit 1; }
for _ in $(seq 1 60); do curl -sf "$URL/api/status" > /dev/null && break; sleep 2; done

publish "$URL" "LexAgent backend online"
echo
echo "  LexAgent is live:  $SITE"
echo "  Backend address:   $URL"
echo "  The models finish loading about a minute after start. Ctrl+C to stop."
echo

# Run until Ctrl+C or until the server or the tunnel exits.
wait -n "$SERVER_PID" "$TUNNEL_PID" || true
echo "The server or the tunnel stopped (see $LOGS)."
