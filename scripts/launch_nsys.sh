#!/usr/bin/env bash
# This command starts a *new* vLLM server underneath Nsight Systems.
# It is NOT an attach to an already running server. Use a fresh terminal.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-admission_256}"
case "$VARIANT" in
  admission_256) NUM_SEQS=256; BLOCKS=4096 ;;
  admission_512) NUM_SEQS=512; BLOCKS=4096 ;;
  *) echo 'Use admission_256 or admission_512 for GPU/Host profiling' >&2; exit 2;;
esac
command -v nsys >/dev/null || { echo 'nsys not installed' >&2; exit 1; }
mkdir -p "$ROOT/results/nsys"
OUT="$ROOT/results/nsys/${VARIANT}_delay${NSYS_DELAY:-60}_dur${NSYS_DURATION:-15}"
export VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# vLLM 0.10.2 profiling docs recommend process-tree capture and CUDA graph nodes.
# Delay starts from nsys launch, NOT server readiness.
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --delay="${NSYS_DELAY:-60}" --duration="${NSYS_DURATION:-15}" \
  --output="$OUT" \
  vllm serve "${MODEL:-Qwen/Qwen3-0.6B}" \
   --dtype auto --max-model-len 2048 --gpu-memory-utilization 0.7 \
   --max-num-seqs "$NUM_SEQS" --max-num-batched-tokens 2048 \
   --enable-chunked-prefill --no-enable-prefix-caching \
   --num-gpu-blocks-override "$BLOCKS"
