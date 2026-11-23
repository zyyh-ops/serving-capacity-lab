#!/usr/bin/env python3
"""Summarize Day6 evidence. Keeps arrival-cohort and completion-window metrics distinct."""
import argparse
import csv
import json
from pathlib import Path


def recovery_clear_time(case):
    meta=case['meta']; phases=meta['phases']
    r=next((p for p in phases if p['name']=='recovery'),None)
    o=next((p for p in phases if p['name']=='overload'),None)
    if not r or not o:
        return None
    path=Path(case['_dir'])/'metrics.jsonl'
    if not path.exists():
        return None
    samples=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    during=[x.get('waiting') for x in samples if o['start']<=x['t']<o['end'] and x.get('waiting') is not None]
    if not during or max(during)<1:
        return None  # No sampled queue: clearance has no meaningful start point.
    for i,x in enumerate(samples):
        if x['t']<r['start']:
            continue
        if x.get('waiting') != 0:
            continue
        # Require zero for a full consecutive second, not a single zero snapshot.
        next_zeros=[y for y in samples[i:] if y.get('waiting') is not None and y['t']<=x['t']+1.4]
        if next_zeros and next_zeros[-1]['t']>=x['t']+1.0 and all(y['waiting']==0 for y in next_zeros):
            return round(x['t']-r['start'],3)
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',default='results')
    args=ap.parse_args()
    root=Path(args.results)
    cases=[]
    for f in sorted(root.glob('*/summary.json')):
        obj=json.loads(f.read_text())
        if 'phases' not in obj.get('summary',{}):
            continue
        obj['_dir']=str(f.parent)
        cases.append(obj)
    if not cases:
        raise SystemExit('No Day6 summary.json under results/*/')
    rows=[]; notes=[]
    for case in cases:
        name=Path(case['_dir']).name
        sm=case['summary']; meta=case['meta']
        clear=recovery_clear_time(case)
        for p in sm['phases']:
            rows.append({'case':name,'policy':meta['policy'],'phase':p['phase'],
                         'rate_target':p['rate_target'],'offered':p['offered'],
                         'injected_rate_proxy':p['injected_rate_proxy'],
                         'completed_rate_window':p['completed_rate_window'],
                         'goodput_window':p['goodput_window'],
                         'cohort_good_over_offered':p['offered_good_fraction'],
                         'cohort_good_over_success':p['admitted_good_fraction'],
                         'local_reject':p['local_reject'],'guard_drop':p['guard_drop'],
                         'waiting_peak_sampled':p['waiting_peak_sampled'],
                         'ttft_p99_cohort_ms':p['ttft_p99_cohort_ms'],
                         'tpot_p99_cohort_ms':p['tpot_p99_cohort_ms'],
                         'recovery_clear_s':clear if p['phase']=='recovery' else ''})
        totals=sm['totals']
        notes.append((name,meta['policy'],totals['offered'],totals['client_local_reject'],
                     totals['client_guard_drop'],totals['http_error_or_timeout'],totals['good_over_offered']))
    csv_file=root/'day6_comparison.csv'
    with csv_file.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    md=['# Day6 Comparison (observed only)','',
        '> Completion-window Goodput ≠ arrival-cohort SLO satisfaction. Local rejects count as offered but are not HTTP 429.',
        '> Recovery clearance uses sampled scheduler Waiting=0 sustained for ≥1s; it is not proof that all request latency recovered.','',
        '| Case | Policy | Phase | Offered | Injected req/s | Completed req/s | Goodput req/s | Good / Offered | Local Reject | Waiting Peak |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        pct= '-' if row['cohort_good_over_offered'] is None else f"{100*row['cohort_good_over_offered']:.1f}%"
        md.append(f"| {row['case']} | {row['policy']} | {row['phase']} | {row['offered']} | {row['injected_rate_proxy']:.2f} | {row['completed_rate_window']:.2f} | {row['goodput_window']:.2f} | {pct} | {row['local_reject']} | {row['waiting_peak_sampled']} |")
    md+=['','## Quality and caution','',
         '| Case | Policy | Offered | Local reject | Guard drop | HTTP error/timeout | Good over offered |',
         '|---|---|---:|---:|---:|---:|---:|']
    for name,policy,off,rej,guard,err,rate in notes:
        md.append(f'| {name} | {policy} | {off} | {rej} | {guard} | {err} | {rate:.1%} |')
    md+=['','- If a guard drop occurred, treat that arm as a client-capped load, not faithful full open-loop.',
         '- A better accepted-request TTFT can be purchased by rejecting more requests. Always report Good/Offered and rejection rate.',
         '- These are short pilots with one seed, not production capacity certifications.',
         '- Comparing two arms requires checking identical planned arrivals and model/server config; AB/BA repeats are recommended.',
         '- Request statuses and sampled Waiting do not capture all frontend queues or network timings.',
         '- Workload cost weights are heuristic and must be calibrated; P+2O is not a GPU-time model.']
    (root/'day6_comparison.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f'Wrote {csv_file} and {root / "day6_comparison.md"}; {len(cases)} runs')

if __name__=='__main__':
    main()
