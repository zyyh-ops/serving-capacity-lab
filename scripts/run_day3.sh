#!/usr/bin/env bash
set -euo pipefail
# Run from week18_day3 directory. Discover first. 60 seconds are not capacity evidence.
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
OUT="${OUT:-results/day3}"
SECONDS="${SECONDS_OVERRIDE:-60}"
SEED="${SEED:-2027}"
mkdir -p "$OUT"
run_case() {
 local name="$1"; shift
 echo "[day3] $name"
 python scripts/workload_probe.py --model "$MODEL" --base-url "$BASE_URL" \
   --seed "$SEED" --duration "$SECONDS" \
   --out "$OUT/$name" "$@" | tee "$OUT/${name}.log" >/dev/null
 echo "[day3] completed $name"
}
# Shape comparison at the SAME rate: workload changes, offered request count comparable.
run_case shape_baseline --short-rate 10 --short-input 128 --short-output 128
run_case shape_long_prefill --short-rate 10 --short-input 512 --short-output 128
run_case shape_long_decode --short-rate 10 --short-input 128 --short-output 512
# Interference: identical short arrival schedule and prompts; add separate long stream.
run_case short_control --short-rate 20 --long-rate 0 --short-input 128 --short-output 128
run_case short_plus_long --short-rate 20 --long-rate 5 --short-input 128 --short-output 128 --long-input 512 --long-output 128
python scripts/analyze_day3.py --results "$OUT"
