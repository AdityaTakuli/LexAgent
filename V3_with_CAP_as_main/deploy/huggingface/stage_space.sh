#!/usr/bin/env bash
# Copy what the Hugging Face Space needs into a clean folder, ready for `hf upload`.
# Usage: deploy/huggingface/stage_space.sh <out_dir>     (run from V3_with_CAP_as_main)
# Never copies .env, data/ or caches: the key is a Space secret, the corpus a dataset repo.
set -euo pipefail

OUT="${1:?usage: stage_space.sh <out_dir>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

[ -d "$ROOT/ui/web/dist" ] || { echo "Build the web app first: cd ui/web && npm run build" >&2; exit 1; }

rm -rf "$OUT" && mkdir -p "$OUT/ui/web" "$OUT/experiments"
cp "$ROOT/deploy/huggingface/Dockerfile" "$ROOT/deploy/huggingface/README.md" "$OUT/"
cp "$ROOT/config.py" "$ROOT/requirements-server.txt" "$OUT/"
cp -r "$ROOT/src" "$OUT/src"
cp "$ROOT"/experiments/*.py "$OUT/experiments/"
cp "$ROOT/ui/server.py" "$ROOT/ui/sample_transcript.json" "$OUT/ui/"
cp -r "$ROOT/ui/web/dist" "$OUT/ui/web/dist"

find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$OUT" -name "*.pyc" -delete
if find "$OUT" -name ".env*" | grep -q .; then echo "Refusing: a .env file was staged" >&2; exit 1; fi
echo "Staged Space in $OUT ($(du -sh "$OUT" | cut -f1))"
