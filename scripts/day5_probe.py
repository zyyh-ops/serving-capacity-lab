#!/usr/bin/env python3
"""Week 18 Day 4 KV/admission pressure probe for vLLM 0.10.2 (educational load test).

Produces request-level planned/dispatch/end timestamps plus server metric samples.
NOT identical to vllm bench serve: use the official bench for independent cross-check.
Requires aiohttp, transformers, and the installed vllm 0.10.2 package.
"""
import argparse
import asyncio
import hashlib
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

import aiohttp
from transformers import AutoTokenizer
from vllm.benchmarks.datasets import RandomDataset

METRICS = {
    "waiting": "vllm:num_requests_waiting",
    "running": "vllm:num_requests_running",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
    "generation_tokens": "vllm:generation_tokens_total",
}
METRIC_RE = re.compile(r'^([^\s{]+)(?:\{[^}]*\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')


def metric_snapshot(body):
    by_key = {key: [] for key in METRICS}
    for line in body.splitlines():
        if not line or line.startswith('#'):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name, val = match.group(1), match.group(2)
        for short, full in METRICS.items():
            if name == full:
                by_key[short].append(float(val))
    return {key: (max(vals) if key == 'kv_usage' else sum(vals)) if vals else None
            for key, vals in by_key.items()}


def quantile(vals, percentile):
    a = sorted(x for x in vals if x is not None and math.isfinite(x))
    if not a:
        return None
    h = (len(a) - 1) * percentile / 100
    lo, hi = math.floor(h), math.ceil(h)
    return a[lo] + (h - lo) * (a[hi] - a[lo])


def utc_now():
    return datetime.now(timezone.utc).isoformat()


async def collect_metrics(session, base_url, t0, stop, sink, interval):
    while not stop.is_set():
        ts = time.monotonic() - t0
        entry = {'t': round(ts, 6), 'utc': utc_now()}
        try:
            async with session.get(base_url + '/metrics', timeout=aiohttp.ClientTimeout(total=3)) as response:
                if response.status != 200:
                    raise RuntimeError(f'metrics status={response.status}')
                entry.update(metric_snapshot(await response.text()))
        except Exception as exc:
            entry['metrics_error'] = repr(exc)
        sink.append(entry)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def send_one(session, base_url, model, sample, index, planned, t0, timeout, results):
    started = time.monotonic()
    record = {
        'id': index, 'planned_t': round(planned, 6),
        'dispatch_t': round(started - t0, 6),
        'dispatch_lag_ms': round((started - t0 - planned) * 1000, 4),
        'status': None, 'success': False, 'error': None,
        'output_tokens': None, 'ttft_ms': None, 'tpot_ms': None,
        'e2e_ms': None, 'end_t': None, 'output_sha256': None, 'finish_reason': None,
    }
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': sample.prompt}],
        'temperature': 0.0,
        'max_completion_tokens': int(sample.expected_output_len),
        'ignore_eos': True,
        'stream': True,
        'stream_options': {'include_usage': True},
    }
    first_choice = None
    last_data = None
    tokens = None
    got_done = False
    completion_chunks = []
    try:
        async with session.post(
            base_url + '/v1/chat/completions', json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            record['status'] = response.status
            if response.status != 200:
                record['error'] = (await response.text())[:1200]
            else:
                parts = []
                async for raw_line in response.content:
                    line = raw_line.decode('utf-8', errors='replace').strip('\r\n')
                    if line == '':
                        if not parts:
                            continue
                        data = '\n'.join(parts)
                        parts = []
                        if data == '[DONE]':
                            got_done = True
                            continue
                        instant = time.monotonic()
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        last_data = instant
                        if obj.get('choices'):
                            if first_choice is None:
                                first_choice = instant
                            for choice in obj['choices']:
                                delta = choice.get('delta') or {}
                                if isinstance(delta.get('content'), str):
                                    completion_chunks.append(delta['content'])
                                if choice.get('finish_reason') is not None:
                                    record['finish_reason'] = choice['finish_reason']
                        if isinstance(obj.get('usage'), dict):
                            tokens = obj['usage'].get('completion_tokens')
                    elif line.startswith('data:'):
                        parts.append(line[5:].lstrip(' '))
                if parts and parts == ['[DONE]']:
                    got_done = True
                if first_choice is None:
                    record['error'] = 'no choices in SSE stream'
                elif tokens is None:
                    record['error'] = 'missing usage.completion_tokens'
                elif last_data is None:
                    record['error'] = 'no parsed SSE data'
                elif not got_done:
                    record['error'] = 'missing SSE [DONE]'
                else:
                    record['success'] = True
                    record['output_tokens'] = int(tokens)
                    record['output_sha256'] = hashlib.sha256(''.join(completion_chunks).encode('utf-8')).hexdigest()
                    record['ttft_ms'] = (first_choice - started) * 1000
                    record['e2e_ms'] = (last_data - started) * 1000
                    if tokens > 1:
                        record['tpot_ms'] = (record['e2e_ms'] - record['ttft_ms']) / (tokens - 1)
    except asyncio.CancelledError:
        record['error'] = 'client_drain_timeout_cancelled'
        raise
    except Exception as exc:
        record['error'] = repr(exc)
    finally:
        record['end_t'] = round(time.monotonic() - t0, 6)
        results.append(record)


def _counter_delta(metrics, name):
    values = [x[name] for x in metrics if x.get(name) is not None]
    if len(values) < 2:
        return None
    if any(b < a for a, b in zip(values, values[1:])):
        # A counter reset during the run would invalidate simple subtraction.
        return None
    return values[-1] - values[0]


def summarize(meta, requests, metrics):
    T = meta['arrival_duration_s']
    sent = [x for x in requests if x.get('status') != 'client_guard_drop' and x.get('dispatch_t') is not None and x['dispatch_t'] < T]
    sent_all = [x for x in requests if x.get('status') != 'client_guard_drop']
    completed_in = [x for x in sent_all if x['success'] and x['end_t'] < T]
    good = lambda x: x['success'] and x['ttft_ms'] is not None and x['tpot_ms'] is not None and x['ttft_ms'] <= meta['ttft_slo_ms'] and x['tpot_ms'] <= meta['tpot_slo_ms']
    good_in = [x for x in completed_in if good(x)]
    success = [x for x in requests if x['success']]
    dropped = [x for x in requests if x.get('status') == 'client_guard_drop']
    failures = [x for x in requests if not x['success'] and x.get('status') != 'client_guard_drop']
    queue = [x for x in metrics if x.get('waiting') is not None and 0 <= x['t'] < T]
    first = [x['waiting'] for x in queue if x['t'] < .25 * T]
    last = [x['waiting'] for x in queue if x['t'] >= .75 * T]
    return {
        'scheduled': len(requests), 'attempted': len(sent_all),
        'actual_dispatch_in_window': len(sent),
        'success_total_after_drain': len(success), 'failure_or_timeout': len(failures),
        'client_guard_drops': len(dropped),
        'completed_success_in_window': len(completed_in),
        'output_tokens_in_window': sum(x.get('output_tokens') or 0 for x in completed_in),
        'output_throughput_in_window': round(sum(x.get('output_tokens') or 0 for x in completed_in)/T, 4),
        'slo_good_completions_in_window': len(good_in),
        'offered_rate_realized': round(len(requests)/T, 4),
        'injected_rate_dispatch_proxy': round(len(sent)/T, 4),
        'completed_rate_in_window': round(len(completed_in)/T, 4),
        'goodput_in_window': round(len(good_in)/T, 4),
        'goodput_cohort_total': len([x for x in success if good(x)]),
        'outstanding_at_arrival_end': len(sent) - len([x for x in sent if x['end_t'] < T]),
        'dispatch_lag_p99_ms': quantile([x['dispatch_lag_ms'] for x in sent], 99),
        'ttft_p99_in_window_ms': quantile([x['ttft_ms'] for x in completed_in], 99),
        'tpot_p99_in_window_ms': quantile([x['tpot_ms'] for x in completed_in], 99),
        'ttft_p50_ms': quantile([x['ttft_ms'] for x in success], 50),
        'ttft_p95_ms': quantile([x['ttft_ms'] for x in success], 95),
        'ttft_p99_ms': quantile([x['ttft_ms'] for x in success], 99),
        'tpot_p50_ms': quantile([x['tpot_ms'] for x in success], 50),
        'tpot_p95_ms': quantile([x['tpot_ms'] for x in success], 95),
        'tpot_p99_ms': quantile([x['tpot_ms'] for x in success], 99),
        'e2e_p99_ms': quantile([x['e2e_ms'] for x in success], 99),
        'waiting_peak_sampled': max((x['waiting'] for x in queue), default=None),
        'waiting_first_quarter_avg': statistics.mean(first) if first else None,
        'waiting_last_quarter_avg': statistics.mean(last) if last else None,
        'kv_peak_sampled': max((x['kv_usage'] for x in metrics if x.get('kv_usage') is not None), default=None),
        'metrics_errors': sum('metrics_error' in x for x in metrics),
        'running_peak_sampled': max((x['running'] for x in metrics if x.get('running') is not None and 0 <= x['t'] < T), default=None),
        'preemptions_delta': _counter_delta(metrics, 'preemptions'),
        'kv_peak_window_sampled': max((x['kv_usage'] for x in metrics if x.get('kv_usage') is not None and 0 <= x['t'] < T), default=None),
        'finished_length_count': sum(x.get('finish_reason') == 'length' for x in success),
        'output_length_mismatch_count': sum(x.get('output_tokens') != meta['output_len_target'] for x in success),
        'http_status_counts': dict(Counter(str(x['status']) for x in requests)),
    }


async def run(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)
    schedule = []
    t = 0.0
    while len(schedule) < args.max_requests:
        t += rng.expovariate(args.rate)
        if t >= args.duration:
            break
        schedule.append(t)
    if len(schedule) == args.max_requests:
        raise ValueError('max-requests reached: shorten duration or raise bound explicitly')
    samples = RandomDataset(random_seed=args.seed).sample(
        tokenizer=tokenizer, num_requests=len(schedule),
        input_len=args.input_len, output_len=args.output_len, range_ratio=0.0,
    ) if schedule else []
    if len(samples) != len(schedule):
        raise RuntimeError(f'dataset generated {len(samples)} instead of {len(schedule)} prompts')
    info = {
        'utc_start': utc_now(), 'model': args.model,
        'base_url': args.base_url, 'rate_target': args.rate,
        'arrival_duration_s': args.duration, 'seed': args.seed,
        'input_len_target': args.input_len, 'output_len_target': args.output_len,
        'ttft_slo_ms': args.ttft_slo, 'tpot_slo_ms': args.tpot_slo,
        'max_inflight_guard': args.max_inflight,
        'drain_timeout_s': args.drain_timeout,
        'timeout_per_request_s': args.timeout,
        'metrics_interval_s': args.metrics_interval,
        'note': 'Workload intended for controlled admission/KV experiments. Dispatch timestamps are immediately before aiohttp session.post, NOT socket-level packet timestamps. Finite arrival window + drain; cannot establish steady state from one short run.'
    }
    results, metrics, pending = [], [], set()
    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        async with session.get(args.base_url + '/v1/models', timeout=aiohttp.ClientTimeout(total=10)) as res:
            if res.status != 200:
                raise RuntimeError(f'/v1/models returned {res.status}')
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        stop = asyncio.Event()
        mon = asyncio.create_task(collect_metrics(session, args.base_url, t0, stop, metrics, args.metrics_interval))
        for i, when in enumerate(schedule):
            await asyncio.sleep(max(0.0, t0 + when - loop.time()))
            if len(pending) >= args.max_inflight:
                results.append({
                    'id': i, 'planned_t': when, 'dispatch_t': None,
                    'dispatch_lag_ms': None, 'status': 'client_guard_drop',
                    'success': False, 'error': 'max_inflight_guard',
                    'end_t': loop.time()-t0,
                })
                continue
            task = asyncio.create_task(send_one(
                session, args.base_url, args.model, samples[i], i, when,
                t0, args.timeout, results,
            ))
            pending.add(task)
            task.add_done_callback(pending.discard)
        # Keep the full fixed observation window, even if the final Poisson arrival
        # occurred earlier. Metrics and completion counts use the same [t0,t0+T).
        await asyncio.sleep(max(0.0, t0 + args.duration - loop.time()))
        arrival_end_actual = loop.time()-t0
        done, remaining = await asyncio.wait(pending, timeout=args.drain_timeout) if pending else (set(), set())
        if remaining:
            for task in remaining:
                task.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)
        stop.set()
        await mon
        total_time = loop.time()-t0
    info['arrival_end_actual_s'] = round(arrival_end_actual, 6)
    info['total_wall_s'] = round(total_time, 6)
    info['utc_end'] = utc_now()
    results.sort(key=lambda x: x['id'])
    stats = summarize(info, results, metrics)
    (root/'request_records.json').write_text(json.dumps({'meta':info,'summary':stats,'requests':results}, indent=2, ensure_ascii=False), encoding='utf-8')
    (root/'metrics.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in metrics), encoding='utf-8')
    (root/'summary.json').write_text(json.dumps({'meta':info,'summary':stats}, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'out':str(root),'summary':stats}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B')
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--input-len', type=int, default=128)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--ttft-slo', type=float, default=200)
    parser.add_argument('--tpot-slo', type=float, default=15)
    parser.add_argument('--metrics-interval', type=float, default=.25)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--drain-timeout', type=float, default=90)
    parser.add_argument('--max-inflight', type=int, default=512)
    parser.add_argument('--max-requests', type=int, default=30000)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.rate <= 0 or args.duration <= 0 or args.max_inflight <= 0 or args.timeout <= 0:
        parser.error('rate, duration, timeout, max-inflight must all be positive')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
