#!/usr/bin/env bash
# Week17 Day6 - vLLM 0.10.2 serve benchmark runner
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
LOAD="${2:-}"
WORKLOAD="${3:-short_decode}"

if [[ "$MODE" != "concurrency" && "$MODE" != "rate" ]]; then
  echo "Usage: bash experiments/day6_suite.sh concurrency 4 short_decode" >&2
  echo "   or: bash experiments/day6_suite.sh rate 8 long_prefill" >&2
  exit 2
fi
if [[ ! "$LOAD" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$LOAD" == 0 ]]; then
  echo "LOAD must be positive" >&2; exit 2
fi

case "$WORKLOAD" in
  short_decode)  INPUT=64; OUTPUT=128 ;;
  long_prefill)  INPUT=512; OUTPUT=32 ;;
  balanced)      INPUT=256; OUTPUT=128 ;;
  *) echo "Unknown workload: $WORKLOAD" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/results"
NAME="${MODE}_${LOAD}_${WORKLOAD}"
METRICS="$ROOT/results/${NAME}_metrics.jsonl"
RESULT="$ROOT/results/${NAME}.json"
LOG="$ROOT/results/${NAME}.log"

COMMON=(
  --backend openai
  --host 127.0.0.1 --port 8000
  --endpoint /v1/completions
  --model Qwen/Qwen3-0.6B
  --dataset-name random
  --random-input-len "$INPUT"
  --random-output-len "$OUTPUT"
  --random-range-ratio 0.0
  --seed 2026 --ignore-eos
  --num-prompts "${DAY6_NUM_PROMPTS:-128}"
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,95,99
  --goodput ttft:200 tpot:15
  --save-result --save-detailed
  --result-dir "$ROOT/results"
  --result-filename "${NAME}.json"
  --disable-tqdm
)

if [[ "$MODE" == "concurrency" ]]; then
  # Infinite offer rate + client-side max concurrency. Not open-loop.
  COMMON+=(--request-rate inf --max-concurrency "$LOAD")
else
  # Finite request rate = Poisson process (burstiness=1); no concurrency gate.
  COMMON+=(--request-rate "$LOAD" --burstiness 1.0)
fi

python "$ROOT/experiments/day6_metrics_sampler.py" \
  --interval 0.5 --output "$METRICS" &
SAMPLER_PID=$!
cleanup() {
  kill -TERM "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true
}
trap cleanup EXIT

printf 'mode=%s load=%s workload=%s input=%s output=%s num_prompts=%s\n' \
  "$MODE" "$LOAD" "$WORKLOAD" "$INPUT" "$OUTPUT" "${DAY6_NUM_PROMPTS:-128}" | tee "$LOG"
vllm bench serve "${COMMON[@]}" 2>&1 | tee -a "$LOG"
printf 'Results: %s\nMetrics: %s\n' "$RESULT" "$METRICS"
