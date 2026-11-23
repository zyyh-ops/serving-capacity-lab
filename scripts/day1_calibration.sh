#!/usr/bin/env bash
# Week18 Day1: vLLM 0.10.2 measurement/client throttling calibration.
# This is NOT a GPU capacity benchmark. It only validates measurement plumbing.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
OUT_DIR="${OUT_DIR:-results/day1}"
mkdir -p "$OUT_DIR"

# Must run against the existing Week17 model server, launched in another terminal.
curl --fail --silent --show-error "$BASE_URL/v1/models" > "$OUT_DIR/models.json"
curl --fail --silent --show-error "$BASE_URL/metrics" > "$OUT_DIR/metrics_before.prom"
python -c 'import vllm; print(vllm.__version__)' > "$OUT_DIR/vllm_version.txt"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT_DIR/gpu_info.csv"
fi

cat > "$OUT_DIR/calibration_contract.json" <<CONTRACT
{
  "purpose": "client-side throttling / metric plumbing calibration, not capacity measurement",
  "model": "$MODEL", "base_url": "$BASE_URL",
  "dataset_name": "random", "random_input_len": 128, "random_output_len": 128,
  "random_range_ratio": 0.0, "num_prompts": 64, "seed": 2026,
  "request_rate": 20, "burstiness": 1.0, "ignore_eos": true,
  "temperature": 0.0, "ttft_slo_ms": 200, "tpot_slo_ms": 15,
  "warmup_excluded": true, "tokenizer_and_chat_template": "same running model",
  "max_model_len": 2048, "gpu_memory_utilization": 0.7,
  "caveat": "Verify actual server arguments; run order, shared prefixes and cache state can confound two runs."
}
CONTRACT

COMMON=(
  --backend openai-chat
  --endpoint /v1/chat/completions
  --base-url "$BASE_URL"
  --model "$MODEL"
  --dataset-name random
  --random-input-len 128
  --random-output-len 128
  --random-range-ratio 0
  --num-prompts 64
  --request-rate 20
  --burstiness 1.0
  --seed 2026
  --temperature 0
  --ignore-eos
  --goodput ttft:200 tpot:15
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,95,99
  --save-result --save-detailed
  --result-dir "$OUT_DIR"
  --disable-tqdm
)

# A short WARMUP run excluded from statistics; it primes CUDA graph/compile paths.
vllm bench serve \
  --backend openai-chat --endpoint-type openai-chat --endpoint /v1/chat/completions \
  --base-url "$BASE_URL" --model "$MODEL" \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --random-range-ratio 0 --num-prompts 8 --request-rate 2 \
  --seed 2025 --temperature 0 --ignore-eos --disable-tqdm \
  > "$OUT_DIR/warmup.log" 2>&1

# Deliberate negative control: client semaphore=1 (not a server capacity point).
vllm bench serve "${COMMON[@]}" --max-concurrency 1 \
  --result-filename client_cap1.json \
  > "$OUT_DIR/client_cap1.log" 2>&1
curl --fail --silent --show-error "$BASE_URL/metrics" > "$OUT_DIR/metrics_after_cap1.prom"

# No effective client cap for this small diagnostic run.
vllm bench serve "${COMMON[@]}" --max-concurrency 128 \
  --result-filename client_cap128.json \
  > "$OUT_DIR/client_cap128.log" 2>&1
curl --fail --silent --show-error "$BASE_URL/metrics" > "$OUT_DIR/metrics_after_cap128.prom"

python "$(dirname "$0")/audit_bench.py" \
  "$OUT_DIR/client_cap1.json" "$OUT_DIR/client_cap128.json" \
  --output "$OUT_DIR/day1_audit.json"

echo "Outputs written to: $OUT_DIR"
echo "WARNING: DO NOT use the calibration rates as stable server capacity limits."
