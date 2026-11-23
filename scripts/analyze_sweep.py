#!/usr/bin/env python3
"""Merge Day 2 probe outputs into summary.csv and summary.md."""
import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    'label','target_rate','duration_s','planned','sent_in_window','injected_rate',
    'completed_rate','goodput_rate','completed_during_window','completed_after_drain',
    'failed_timeout','client_drops','outstanding_at_end',
    'dispatch_lag_p99_ms','ttft_p50_ms','ttft_p95_ms','ttft_p99_ms','ttft_p99_in_window_ms','output_throughput_in_window',
    'tpot_p95_ms','tpot_p99_ms','waiting_first_q','waiting_last_q',
    'waiting_peak_sampled','kv_peak_sampled','metrics_errors',
]


def flatten(root):
    data = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    s, m = data['summary'], data['meta']
    return dict(
        label=root.name,target_rate=m['rate_target'],duration_s=m['arrival_duration_s'],
        planned=s['scheduled'],sent_in_window=s['actual_dispatch_in_window'],
        injected_rate=s['injected_rate_dispatch_proxy'],completed_rate=s['completed_rate_in_window'],
        goodput_rate=s['goodput_in_window'],completed_during_window=s['completed_success_in_window'],
        completed_after_drain=s['success_total_after_drain'],failed_timeout=s['failure_or_timeout'],
        client_drops=s['client_guard_drops'],outstanding_at_end=s['outstanding_at_arrival_end'],
        dispatch_lag_p99_ms=s['dispatch_lag_p99_ms'],ttft_p50_ms=s['ttft_p50_ms'],
        ttft_p95_ms=s['ttft_p95_ms'],ttft_p99_ms=s['ttft_p99_ms'],
        ttft_p99_in_window_ms=s.get('ttft_p99_in_window_ms'),
        output_throughput_in_window=s.get('output_throughput_in_window'),
        tpot_p95_ms=s['tpot_p95_ms'],tpot_p99_ms=s['tpot_p99_ms'],
        waiting_first_q=s['waiting_first_quarter_avg'],waiting_last_q=s['waiting_last_quarter_avg'],
        waiting_peak_sampled=s['waiting_peak_sampled'],kv_peak_sampled=s['kv_peak_sampled'],
        metrics_errors=s['metrics_errors'],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/day2')
    args = p.parse_args()
    results = Path(args.results)
    runs = sorted([x for x in results.iterdir() if x.is_dir() and (x/'summary.json').exists()]) if results.exists() else []
    if not runs:
        raise SystemExit('No summary.json files found. Run openloop_probe.py first.')
    records = [flatten(x) for x in runs]
    records.sort(key=lambda x: float(x['target_rate']))
    with (results/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    lines = ['# Week 18 Day 2 — Sweep Summary', '',
             '这些数字按固定 arrival window 统计；`completed_after_drain` 属于额外的最终队列清空结果，不能作为该窗口内稳定吞吐。', '',
             '| Target req/s | Injected req/s | Completed req/s | Goodput req/s | TTFT P99 (ms) | Waiting peak (sampled) | End backlog | Client guard drops |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return '-' if x is None else f'{x:.2f}' if isinstance(x,float) else str(x)
    for r in records:
        lines.append('| '+ ' | '.join(fmt(r[k]) for k in ['target_rate','injected_rate','completed_rate','goodput_rate','ttft_p99_ms','waiting_peak_sampled','outstanding_at_end','client_drops'])+' |')
    lines += ['', '## Evidence limits',
              '- 每个负载点需要审查目标发起速率与实际 dispatch 速率是否一致；若发生 client guard drops，则该点不属于完整 Open-loop 注入。',
              '- P99 为样本估计，且本表包含所有最终成功请求的 P99；可能包括 arrival window 后 Drain 阶段的请求，不可当作纯稳态 P99。',
              '- Queue peak 是离散采样观察到的最大值，不等于精确真实峰值。',
              '- 只有同时观察注入率、稳态离开率、队列趋势和多次重复，才能判断最大可持续容量。',
              '- 默认单次短实验只能定位候选拐点；不能独立认定正式容量。',
              '']
    (results/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
