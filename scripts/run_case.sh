#!/usr/bin/env bash
# bash scripts/run_case.sh NAME PHASES POLICY SEED LONG_FRAC [extra flags...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:?NAME}"; PHASES="${2:?PHASES}"; POLICY="${3:?POLICY}"
SEED="${4:-2031}"; LONG_FRAC="${5:-0}"
shift $(( $# >= 5 ? 5 : $# ))
OUT="$ROOT/results/${NAME}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
if [[ -e "$OUT/summary.json" ]]; then echo "Refusing overwrite $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
curl -fsS --max-time 10 "$BASE_URL/v1/models" > "$OUT/models.json"
curl -fsS --max-time 10 "$BASE_URL/metrics" > "$OUT/metrics_pre.txt"
{
  date -u +%FT%TZ
  python -c 'import vllm; print("vllm_version", vllm.__version__)'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  printf 'name=%s phases=%s policy=%s seed=%s long_frac=%s\n' "$NAME" "$PHASES" "$POLICY" "$SEED" "$LONG_FRAC"
} > "$OUT/contract.txt"
PYTHONUNBUFFERED=1 python "$ROOT/scripts/capacity_probe.py" \
  --base-url "$BASE_URL" --model "${MODEL:-Qwen/Qwen3-0.6B}" \
  --out "$OUT" --phases "$PHASES" --policy "$POLICY" \
  --seed "$SEED" --long-frac "$LONG_FRAC" \
  --max-inflight-guard "${MAX_INFLIGHT_GUARD:-2048}" \
  --drain-timeout "${DRAIN_TIMEOUT:-120}" \
  "$@" 2>&1 | tee "$OUT/probe.log"
curl -fsS --max-time 10 "$BASE_URL/metrics" > "$OUT/metrics_post.txt"
echo "Saved to $OUT"
