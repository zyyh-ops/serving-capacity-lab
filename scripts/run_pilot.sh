#!/usr/bin/env bash
# Run one pilot experiment at a time, keeping the same live server running.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
case "${1:-}" in
  sweep)
    bash scripts/run_case.sh sweep_70 'reference:40:70' open 2031 0
    bash scripts/run_case.sh sweep_80 'reference:40:80' open 2031 0
    bash scripts/run_case.sh sweep_90 'reference:40:90' open 2031 0
    ;;
  recovery)
    bash scripts/run_case.sh recovery_open 'baseline:35:65,overload:25:105,recovery:60:65' open 2031 0
    bash scripts/run_case.sh recovery_gate 'baseline:35:65,overload:25:105,recovery:60:65' rps 2031 0 --admit-rate 78 --burst 20
    ;;
  mix)
    bash scripts/run_case.sh mix_open 'mixed:60:60' open 2031 0.25
    bash scripts/run_case.sh mix_cost 'mixed:60:60' cost 2031 0.25 --cost-rate 24000 --cost-burst 6000
    ;;
  *) echo 'Usage: bash scripts/run_pilot.sh sweep|recovery|mix' >&2; exit 2;;
esac
python scripts/analyze_day6.py --results results
