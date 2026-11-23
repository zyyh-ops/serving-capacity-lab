#!/usr/bin/env bash
# Usage: bash scripts/run_probe.sh NAME RATE SECONDS INPUT OUTPUT SEED
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:?name}"; RATE="${2:?rate}"; DUR="${3:?seconds}"
P="${4:?input_tokens}"; O="${5:?output_tokens}"; SEED="${6:-2027}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
OUT="$ROOT/results/${NAME}"
if [[ -e "$OUT/summary.json" ]]; then echo "Already exists: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
curl -fsS "$BASE_URL/v1/models" > "$OUT/models.json"
curl -fsS "$BASE_URL/metrics" > "$OUT/metrics_pre.txt"
{
 date -u +%FT%TZ
 python -c 'import vllm; print("vllm", vllm.__version__)'
 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
 printf 'name=%s rate=%s duration=%s prompt=%s output=%s seed=%s\n' "$NAME" "$RATE" "$DUR" "$P" "$O" "$SEED"
} > "$OUT/contract.txt"
PYTHONUNBUFFERED=1 python "$ROOT/scripts/day5_probe.py" \
 --base-url "$BASE_URL" --rate "$RATE" --duration "$DUR" \
 --input-len "$P" --output-len "$O" --seed "$SEED" \
 --max-inflight "${MAX_INFLIGHT:-1024}" --timeout "${REQUEST_TIMEOUT:-90}" \
 --drain-timeout "${DRAIN_TIMEOUT:-90}" --out "$OUT" 2>&1 | tee "$OUT/probe.log"
curl -fsS "$BASE_URL/metrics" > "$OUT/metrics_post.txt"
echo "Saved $OUT"
