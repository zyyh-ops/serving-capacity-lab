#!/usr/bin/env bash
# Non-destructive release preflight; no server startup, no remote access.
set -uo pipefail
ROOT="${1:-.}"
cd "$ROOT" || exit 2
FAILED=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git diff --check; then echo '[PASS] git diff --check'; else echo '[FAIL] git diff --check'; FAILED=1; fi
  echo "Git HEAD: $(git rev-parse --short HEAD)"
  git status --short
else
  echo '[WARN] No Git repository found; Git closure not audited'
fi
python -m compileall -q scripts || FAILED=1
python scripts/test_audit.py || FAILED=1
python scripts/audit_week18.py --root . --out results/week18_day7 || FAILED=1
exit "$FAILED"
