#!/usr/bin/env python3
"""Small audit of vLLM 0.10.2 'vllm bench serve --save-result' JSON.
Does not infer target vs actual injection rate from aggregate throughput.
Does not pretend that ITL event gaps always equal individual token gaps.
"""
import argparse
import json
import math
from pathlib import Path


def almost(a: float, b: float, rtol: float = 1e-3, atol: float = 1e-4) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)


def audit_file(filename: str) -> dict:
    d = json.loads(Path(filename).read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{filename}: expected a JSON object")
    failed, warnings = [], []
    duration = float(d.get("duration", 0))
    completed = int(d.get("completed", -1))
    rate = d.get("request_throughput", None)
    tok_rate = d.get("output_throughput", None)
    token_count = d.get("total_output_tokens", None)
    goodput = d.get("request_goodput", None)
    input_lens = d.get("input_lens")
    output_lens = d.get("output_lens")
    errors = d.get("errors")

    if duration <= 0:
        failed.append("duration must be positive")
    if completed < 0:
        failed.append("completed missing/negative")
    if duration > 0 and completed >= 0 and rate is not None:
        if not almost(rate, completed / duration):
            failed.append("request_throughput != completed/duration")
    if duration > 0 and token_count is not None and tok_rate is not None:
        if not almost(tok_rate, float(token_count) / duration):
            failed.append("output_throughput != total_output_tokens/duration")
    if goodput is not None and rate is not None:
        if not (-1e-8 <= float(goodput) <= float(rate) + 1e-8):
            failed.append("request_goodput outside [0, request_throughput]")
    if output_lens is not None:
        if completed > len(output_lens):
            failed.append("completed > number of output records")
        if token_count is not None and int(token_count) != sum(int(x) for x in output_lens):
            failed.append("total_output_tokens != sum(output_lens)")
        if input_lens is not None and len(input_lens) != len(output_lens):
            failed.append("input_lens/output_lens lengths differ")
        if errors is not None and len(errors) != len(output_lens):
            failed.append("errors/output_lens lengths differ")
    else:
        warnings.append("No detailed per-request output_lens. Re-run with --save-detailed")
    if errors is not None:
        num_failed = sum(bool(e) for e in errors)
        if completed >= 0 and num_failed + completed != len(errors):
            warnings.append("Error records do not align 1:1 with completed; inspect JSON schema")
    else:
        num_failed = None
        warnings.append("No detailed errors list; can't verify failure denominator")
    for metric in ("ttft", "tpot", "itl", "e2el"):
        p50 = d.get(f"p50_{metric}_ms")
        p95 = d.get(f"p95_{metric}_ms")
        p99 = d.get(f"p99_{metric}_ms")
        if all(x is not None for x in (p50, p95, p99)):
            if not (0 <= p50 <= p95 <= p99):
                failed.append(f"{metric} percentile order invalid")
        else:
            warnings.append(f"Missing one of p50/p95/p99 {metric}; check CLI flags")
    if completed < 100:
        warnings.append("Small completed sample: p99 unstable; calibration only")
    if d.get("max_concurrency") is None:
        warnings.append("No recorded max_concurrency metadata; verify logged CLI")

    return {
        "file": filename,
        "pass": not failed,
        "checks_failed": failed,
        "warnings": warnings,
        "completed": completed,
        "failed_request_records": num_failed,
        "duration_s": duration,
        "nominal_request_rate": d.get("request_rate"),
        "max_concurrency": d.get("max_concurrency"),
        "achieved_completed_per_wall_s": rate,
        "output_tok_per_wall_s": tok_rate,
        "slo_goodput_per_wall_s": goodput,
        "ttft_p99_ms": d.get("p99_ttft_ms"),
        "tpot_p99_ms": d.get("p99_tpot_ms"),
        "itl_p99_ms": d.get("p99_itl_ms"),
        "e2el_p99_ms": d.get("p99_e2el_ms"),
        "note": "Completed/duration includes tail-drain. Not actual network request injection rate."
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    results = [audit_file(f) for f in args.files]
    summary = {"all_pass": all(r["pass"] for r in results), "files": results}
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not summary["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
