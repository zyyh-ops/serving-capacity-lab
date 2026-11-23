"""Sample vLLM 0.10.2 /metrics during one benchmark run.

Stores raw Prometheus metrics of interest (gauges, counters). These are
point-in-time samples, not per-request traces. No external dependencies.
"""
import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

KEEP = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
}
STOP = False


def stop(*_args):
    global STOP
    STOP = True


def parse_metrics(body):
    data = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        # Names may carry model_name and engine labels. Sum matching samples
        # for multi-engine counters, but this study uses one engine.
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        if name in KEEP:
            try:
                value = float(parts[1])
            except ValueError:
                continue
            data[name] = data.get(name, 0.0) + value
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.interval <= 0:
        ap.error("--interval must be positive")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        while not STOP:
            sample = {"utc": datetime.now(timezone.utc).isoformat(),
                      "monotonic": time.monotonic()}
            try:
                with urlopen(args.url, timeout=3) as resp:
                    sample["metrics"] = parse_metrics(
                        resp.read().decode("utf-8", errors="replace")
                    )
            except Exception as exc:
                sample["error"] = f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            fh.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
