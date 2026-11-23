#!/usr/bin/env python3
"""Summarize Day3 results without mixing class-specific metrics or drain boundaries."""
import argparse,csv,json
from pathlib import Path

def read_runs(root):
    for p in sorted(root.glob('*/summary.json')):
        data=json.loads(p.read_text(encoding='utf-8'))
        yield p.parent.name,data

def run(root):
    rows=[]
    for name,entry in read_runs(root):
        meta=entry['meta'];serv=entry['server_metrics']
        for group,d in entry['groups'].items():
            if d['scheduled']==0:continue
            rows.append({'run':name,'group':group,'duration_s':meta['duration_s'],
                 'target_short_rps':meta['short_rate'],'target_long_rps':meta['long_rate'],
                 'P_est':d['avg_prompt_tokens_est'],'O_avg':d['avg_output_tokens_observed'],
                 'injected_rps':d['injected_rps_proxy'],'completed_rps':d['completed_rps_window'],
                 'goodput_rps':d['goodput_rps_window'],'joint_slo_cohort':d['joint_slo_satisfaction_cohort'],
                 'ttft_p50_ms':d['ttft_p50_ms'],'ttft_p95_ms':d['ttft_p95_ms'],'ttft_p99_ms':d['ttft_p99_ms'],
                 'tpot_p50_ms':d['tpot_p50_ms'],'tpot_p95_ms':d['tpot_p95_ms'],'tpot_p99_ms':d['tpot_p99_ms'],
                 'success':d['success_after_drain'],'failed':d['failed_or_timed_out'],
                 'guard_drops':d['client_guard_drops'],'dispatch_lag_p99_ms':d['dispatch_lag_p99_ms'],
                 'waiting_peak_sampled':serv['waiting_peak_sampled'],'running_peak_sampled':serv['running_peak_sampled'],
                 'kv_peak_sampled':serv['kv_peak_sampled'],'metrics_errors':serv['metrics_scrape_errors']})
    if not rows:raise SystemExit('No */summary.json found in '+str(root))
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hdr=['run','group','injected_rps','completed_rps','goodput_rps','ttft_p99_ms','tpot_p99_ms','waiting_peak_sampled','guard_drops']
    fmt=lambda r,k:str(r[k]) if r[k] is not None else 'NA'
    lines=['# Week 18 Day 3 Comparison','','Percentiles are for eventual success cohort and may include Drain.','Client guard drops are UNSENT requests.','', '| '+' | '.join(hdr)+' |','| '+' | '.join(['---']*len(hdr))+' |']
    for r in rows:lines.append('| '+' | '.join(fmt(r,k) for k in hdr)+' |')
    lines.extend(['','**Interference check**: compare `short_control/short` vs `short_plus_long/short`.','Do not claim causality unless short schedules/prompt samples match, client health is good, and runs repeat in AB/BA order.'])
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(rows)} group rows -> {root}/comparison.csv and comparison.md')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results/day3'));run(p.parse_args().results)
