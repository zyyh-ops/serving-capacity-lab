#!/usr/bin/env bash
set -euo pipefail
# Independent vllm bench serve cross-check. This run is not the same request sample as our probe.
RATE="${RATE:-20}"
N="${N:-256}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
OUT_DIR="${OUT_DIR:-results/day2_official}"
mkdir -p "$OUT_DIR"
vllm bench serve \
  --endpoint-type openai-chat \
  --backend openai-chat \
  --host 127.0.0.1 --port 8000 \
  --endpoint /v1/chat/completions \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 128 --random-output-len 128 \
  --random-range-ratio 0.0 \
  --num-prompts "$N" --request-rate "$RATE" --burstiness 1.0 \
  --seed 2026 --ignore-eos --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --goodput ttft:200 tpot:15 \
  --save-result --save-detailed \
  --result-dir "$OUT_DIR" --result-filename "official_rate_${RATE}.json" \
  --disable-tqdm
