#!/usr/bin/env bash
# Generate a minimal reproducibility fingerprint. Deliberately excludes full env / API keys.
set -euo pipefail
OUT="${1:-results/week18_day7}"
mkdir -p "$OUT"
{
  echo "# Week18 Environment Fingerprint"
  echo
  echo "Captured UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Python: $(python --version 2>&1)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  else
    echo "nvidia-smi unavailable"
  fi
  python - <<'PY'
import importlib.metadata as m
for name in ('vllm','torch','triton','transformers','aiohttp'):
 try: print(f'{name}: {m.version(name)}')
 except m.PackageNotFoundError: print(f'{name}: not installed')
PY
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_branch: $(git branch --show-current)"
    if [ -z "$(git status --porcelain)" ]; then echo 'git_dirty: false'; else echo 'git_dirty: true'; fi
  else
    echo "git: not a repository"
  fi
} > "$OUT/environment_fingerprint.md"
echo "[OK] $OUT/environment_fingerprint.md"
