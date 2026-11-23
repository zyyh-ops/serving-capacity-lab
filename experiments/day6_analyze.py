"""Recompute Day6 serving-benchmark numbers from results/*.json and *_metrics.jsonl.

Read-only analysis. No third-party deps beyond numpy.

Usage:
    python experiments/day6_analyze.py --results results \
        --output results/day6_analysis.json
"""
import argparse
import glob
import json
import math
import os
from datetime import datetime

import numpy as np

SLO_TTFT_MS = 200.0
SLO_TPOT_MS = 15.0
# Sensitivity grid used in the analysis doc (ttft_ms, tpot_ms).
SLO_GRID = [(200, 15), (50, 6), (30, 5.5), (30, 5), (25, 4.5),
            (20, 4.5), (20, 4)]


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def load_json_runs(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)[:-5]
        if not (name.startswith("concurrency_") or name.startswith("rate_")):
            continue
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        ttfts = np.array(raw["ttfts"], dtype=float) * 1000.0  # s -> ms
        out_lens = np.array(raw["output_lens"], dtype=int)
        itl_lists = [np.array(x, dtype=float) * 1000.0 for x in raw["itls"]]
        e2el = ttfts + np.array([
            sum(x) for x in itl_lists
        ]) if len(itl_lists) else np.array([])
        # TPOT as vLLM defines it: (latency - ttft) / (output_len - 1)
        tpot = np.array([
            (sum(itl) / (n - 1)) if n > 1 else 0.0
            for itl, n in zip(itl_lists, out_lens)
        ])
        itl = np.concatenate(itl_lists) if itl_lists else np.array([])
        good = (ttfts <= SLO_TTFT_MS) & (tpot <= SLO_TPOT_MS)
        runs[name] = {
            "has_json": True,
            "request_rate": raw["request_rate"],
            "max_concurrency": raw["max_concurrency"],
            "duration_s": raw["duration"],
            "completed": raw["completed"],
            "total_input_tokens": raw["total_input_tokens"],
            "total_output_tokens": raw["total_output_tokens"],
            "request_throughput": raw["request_throughput"],
            "request_goodput_reported": raw["request_goodput"],
            "output_throughput": raw["output_throughput"],
            "total_token_throughput": raw["total_token_throughput"],
            "input_len_values": {
                int(v): int((np.array(raw["input_lens"]) == v).sum())
                for v in sorted(set(raw["input_lens"]))
            },
            "output_len_values": {
                int(v): int((out_lens == v).sum())
                for v in sorted(set(out_lens.tolist()))
            },
            "errors": sum(1 for e in raw["errors"] if e),
            "ttft_p50": pct(ttfts, 50),
            "ttft_p95": pct(ttfts, 95),
            "ttft_p99": pct(ttfts, 99),
            "ttft_mean": float(ttfts.mean()),
            "ttft_std": float(ttfts.std()),
            "ttft_max": float(ttfts.max()),
            "tpot_p50": pct(tpot, 50),
            "tpot_p95": pct(tpot, 95),
            "tpot_p99": pct(tpot, 99),
            "tpot_mean": float(tpot.mean()),
            "tpot_std": float(tpot.std()),
            "tpot_max": float(tpot.max()),
            "itl_p50": pct(itl, 50),
            "itl_p95": pct(itl, 95),
            "itl_p99": pct(itl, 99),
            "itl_mean": float(itl.mean()),
            "itl_std": float(itl.std()),
            "itl_n": int(itl.size),
            "itl_n_expected": int((out_lens - 1).sum()),
            "itl_gt_2x_median": float((itl > 2 * np.median(itl)).mean()),
            "itl_frac_gt_6ms": float((itl > 6).mean()),
            "itl_frac_gt_8ms": float((itl > 8).mean()),
            "itl_frac_gt_10ms": float((itl > 10).mean()),
            "itl_frac_gt_12ms": float((itl > 12).mean()),
            "e2el_p50": pct(e2el, 50),
            "e2el_p95": pct(e2el, 95),
            "e2el_p99": pct(e2el, 99),
            "e2el_mean": float(e2el.mean()),
            "e2el_max": float(e2el.max()),
            "good_count_recomputed": int(good.sum()),
            "goodput_recomputed": float(good.sum() / raw["duration"]),
            "slo_violations_ttft": int((ttfts > SLO_TTFT_MS).sum()),
            "slo_violations_tpot": int((tpot > SLO_TPOT_MS).sum()),
            # Little's law: L = lambda * W (W uses per-request mean E2E)
            "mean_inflight_littles_law": float(e2el.mean() / 1000.0 *
                                               raw["request_throughput"]),
            # consistency check: tokens/s per in-flight request == 1/TPOT
            "tok_s_per_inflight_check": float(
                raw["output_throughput"] /
                max(e2el.mean() / 1000.0 * raw["request_throughput"], 1e-9)),
            "e2el_reconstructed_p50": pct(e2el, 50),
            "e2el_reported_p50": raw["p50_e2el_ms"],
            "goodput_grid": [
                {"ttft_ms": tt, "tpot_ms": tp,
                 "good": int(((ttfts <= tt) & (tpot <= tp)).sum()),
                 "goodput_req_s": float(
                     ((ttfts <= tt) & (tpot <= tp)).sum() / raw["duration"])}
                for tt, tp in SLO_GRID
            ],
        }
    return runs


def load_metrics(results_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_metrics.jsonl"))):
        base = os.path.basename(path)[:-len("_metrics.jsonl")]
        samples = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        if not samples:
            continue
        ok = [s for s in samples if "metrics" in s]
        if not ok:
            out[base] = {"samples": len(samples), "all_errors": True}
            continue
        t = np.array([s["monotonic"] for s in ok])
        order = np.argsort(t)
        t = t[order]
        ok = [ok[i] for i in order]
        m = {k: np.array([s["metrics"].get(k, np.nan) for s in ok])
             for k in ok[0]["metrics"]}
        running = m.get("vllm:num_requests_running", np.zeros_like(t))
        waiting = m.get("vllm:num_requests_waiting", np.zeros_like(t))
        kv = m.get("vllm:gpu_cache_usage_perc", np.zeros_like(t))
        preempt = m.get("vllm:num_preemptions_total", np.zeros_like(t))
        prompt = m.get("vllm:prompt_tokens_total", np.zeros_like(t))
        gen = m.get("vllm:generation_tokens_total", np.zeros_like(t))
        dt = np.diff(t)
        active = (running > 0) | (waiting > 0) | (np.r_[0, np.diff(gen)] > 0)
        idx = np.where(active)[0]
        if idx.size:
            a0, a1 = int(idx[0]), int(idx[-1])
        else:
            a0 = a1 = 0
        span = float(t[-1] - t[0])
        aspan = float(t[a1] - t[a0]) if a1 > a0 else 0.0
        # time-weighted mean over the active window (left-endpoint rectangles)
        if a1 > a0:
            w = dt[a0:a1]
            mean_run = float((running[a0:a1] * w).sum() / w.sum())
            mean_wait = float((waiting[a0:a1] * w).sum() / w.sum())
            mean_kv = float((kv[a0:a1] * w).sum() / w.sum())
            frac_wait = float((w[waiting[a0:a1] > 0]).sum() / w.sum())
        else:
            mean_run = mean_wait = mean_kv = frac_wait = float("nan")
        out[base] = {
            "samples": int(len(samples)),
            "parse_ok": int(len(ok)),
            "span_s": span,
            "active_span_s": aspan,
            "sample_interval_p50_s": float(np.median(dt)) if dt.size else None,
            "peak_running": float(np.nanmax(running)),
            "mean_running_active": mean_run,
            "peak_waiting": float(np.nanmax(waiting)),
            "mean_waiting_active": mean_wait,
            "frac_active_time_waiting_gt0": frac_wait,
            "peak_kv_perc": float(np.nanmax(kv)) * 100.0,
            "mean_kv_perc_active": mean_kv * 100.0 if mean_kv == mean_kv else None,
            "preemptions_delta": float(preempt[-1] - preempt[0]),
            "prompt_tokens_delta": float(prompt[-1] - prompt[0]),
            "generation_tokens_delta": float(gen[-1] - gen[0]),
            "server_gen_tok_s_active": float(
                (gen[-1] - gen[0]) / aspan) if aspan else None,
            "first_sample_utc": ok[0]["utc"],
            "last_sample_utc": ok[-1]["utc"],
            "sample_step_energy_running": float(running.sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default=None)
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "slo": {"ttft_ms": SLO_TTFT_MS, "tpot_ms": SLO_TPOT_MS},
        "slo_grid": [{"ttft_ms": t, "tpot_ms": p} for t, p in SLO_GRID],
        "runs": load_json_runs(args.results),
        "server_metrics": load_metrics(args.results),
    }
    text = json.dumps(payload, indent=args.indent, ensure_ascii=False,
                      allow_nan=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
