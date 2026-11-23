#!/usr/bin/env python3
"""Summarize all executed Day4 experiments. Never interpolate missing runs."""
import csv
import json
from pathlib import Path
import argparse

FIELDS = [
    'variant','target_req_s','duration_s','seed','prompt','output',
    'injected_req_s','completed_req_s','goodput_req_s','completed_total',
    'client_drops','failures','waiting_peak','running_peak','kv_peak',
    'preemptions_delta','ttft_p99_ms','tpot_p99_ms','dispatch_lag_p99_ms',
    'length_mismatch','source'
]

def to_row(p):
    obj = json.loads(p.read_text(encoding='utf-8'))
    meta = obj['meta']; s = obj['summary']
    return dict(zip(FIELDS, [
        p.parent.parent.name,meta['rate_target'],meta['arrival_duration_s'],meta['seed'],
        meta['input_len_target'],meta['output_len_target'],s['injected_rate_dispatch_proxy'],
        s['completed_rate_in_window'],s['goodput_in_window'],s['success_total_after_drain'],
        s['client_guard_drops'],s['failure_or_timeout'],s.get('waiting_peak_sampled'),
        s.get('running_peak_sampled'),s.get('kv_peak_window_sampled'),
        s.get('preemptions_delta'),s.get('ttft_p99_ms'),s.get('tpot_p99_ms'),
        s.get('dispatch_lag_p99_ms'),s.get('output_length_mismatch_count'),str(p.parent)
    ]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results/day4');a=ap.parse_args()
    root=Path(a.results);rows=[to_row(x) for x in root.glob('*/rate_*/summary.json')]
    rows.sort(key=lambda x:(x['variant'], x['target_req_s'], x['seed']))
    out=root/'comparison.csv';root.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    md=["# Week18 Day4 Results (observed only)","",'These rows combine independent runs. Different server variants must not be treated as a proof of GPU saturation.', '', '| Variant | λ target | Injected | Completed | Goodput | Waiting peak | Running peak | KV peak | Preemptions Δ | TTFT P99 | TPOT P99 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        f=lambda x: 'NA' if x is None else str(round(x,3) if isinstance(x,float) else x)
        md.append('| '+' | '.join(f(r[k]) for k in ['variant','target_req_s','injected_req_s','completed_req_s','goodput_req_s','waiting_peak','running_peak','kv_peak','preemptions_delta','ttft_p99_ms','tpot_p99_ms'])+' |')
    md += ['','**Quality gates:** inspect guard drops, HTTP/timeouts, metrics errors, preemption counter resets, and each server log before making causal claims.','']
    (root/'comparison.md').write_text('\n'.join(md),encoding='utf-8')
    print(f'{len(rows)} runs -> {out} and {root/"comparison.md"}')
if __name__=='__main__':main()
