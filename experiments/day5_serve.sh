#!/usr/bin/env bash
set -euo pipefail

cd /data/vllm_learning
mkdir -p results

MODE="${1:?Usage: bash experiments/day5_serve.sh [no_graph|piecewise|eager]}"
case "$MODE" in
  no_graph)
    EXTRA=(
      --compilation-config
      '{"level":3, "cudagraph_mode":0}'
    )
    ;;
  piecewise)
    EXTRA=(
      --compilation-config
      '{"level":3, "cudagraph_mode":1}'
    )
    ;;
  eager)
    EXTRA=(--enforce-eager)
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES=0
export VLLM__USE_V1=1

vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --dtype auto \
  --max-model-len 512 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.7 \
  --no-enable-prefix-caching \
  --generation-config vllm \
  "${EXTRA[@]}" \
  2>&1 | tee "results/day5_${MODE}_server.log"