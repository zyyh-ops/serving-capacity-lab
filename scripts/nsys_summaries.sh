#!/usr/bin/env bash
# Usage: bash scripts/nsys_summaries.sh path/to/report.nsys-rep
set -euo pipefail
REPORT="${1:?nsys report path required}"
[[ -f "$REPORT" ]] || { echo "Report missing: $REPORT" >&2; exit 2; }
ROOT="${REPORT%.nsys-rep}"
LOG="${ROOT}.nsys_stats.log"

# `nsys stats` writes its own banner to stdout *before* the report table:
#   <blank line>
#   NOTICE: Existing SQLite export found: ...
#   It is assumed file was previously exported from: ...
#   Consider using --force-export=true if needed.
#   <blank line>
#   Processing [...] with [...]...
# Redirecting stdout straight into the .csv therefore puts those lines above the
# real header, and any strict CSV consumer sees "No CSV header". Capture the raw
# output, keep it in the .log for provenance, and strip the banner from the .csv.
strip_preamble() {  # $1 = raw capture, $2 = destination .csv
    if grep -q '^Processing \[' "$1"; then
        sed '1,/^Processing \[/d' "$1" > "$2"
    else
        cp "$1" "$2"
    fi
    sed -i '/./,$!d' "$2"   # drop any leading blank lines before the header
}

emit() {  # $1 = report name, $2 = csv suffix
    local raw
    raw="$(mktemp)"
    nsys stats --report "$1" --format csv "$REPORT" > "$raw" 2>&1 || true
    { printf '\n### %s\n' "$1"; cat "$raw"; } >> "$LOG"
    strip_preamble "$raw" "${ROOT}.$2.csv"
    rm -f "$raw"
}

: > "$LOG"
emit cuda_gpu_kern_sum kernel_summary
emit cuda_api_sum cuda_api_summary
emit cuda_kern_exec_sum kernel_exec_summary
nsys export --type sqlite --output "${ROOT}.sqlite" "$REPORT" || true
printf 'Saved summaries beside %s\n' "$REPORT"
