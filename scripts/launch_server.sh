#!/usr/bin/env bash
# Single server instance. Stop the old server and confirm :8000 free before relaunch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-admission_256}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
case "$VARIANT" in
  admission_256) SEQS=256 ;;
  admission_512) SEQS=512 ;;
  *) echo "variant must be admission_256 or admission_512" >&2; exit 2 ;;
esac
mkdir -p "$ROOT/results/servers"
LOG="$ROOT/results/servers/${VARIANT}.log"
if [[ -e "$LOG" ]]; then
  LOG="$ROOT/results/servers/${VARIANT}_$(date -u +%Y%m%dT%H%M%S).log"
fi
CMD=(vllm serve "$MODEL" --dtype auto --max-model-len 2048 \
    --gpu-memory-utilization 0.7 --max-num-seqs "$SEQS" \
    --max-num-batched-tokens 2048 --enable-chunked-prefill \
    --no-enable-prefix-caching --num-gpu-blocks-override 4096)
(printf 'VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=%s ' "${CUDA_VISIBLE_DEVICES:-0}";
 printf '%q ' "${CMD[@]}"; printf '\n') > "${LOG}.cmd"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" VLLM_USE_V1=1
"${CMD[@]}" 2>&1 | tee "$LOG"
