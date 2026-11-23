#!/usr/bin/env python3
"""Consistency audit for Day4 probe artifacts. Checks counting, not performance quality."""
import json
from pathlib import Path
import sys

def audit(root):
    obj=json.loads((root/'request_records.json').read_text(encoding='utf8'))
    rec=obj['requests']; summary=obj['summary']; meta=obj['meta']; T=meta['arrival_duration_s']
    scheduled=len(rec)
    drops=[r for r in rec if r.get('status')=='client_guard_drop']
    attempts=[r for r in rec if r.get('status')!='client_guard_drop']
    successes=[r for r in rec if r.get('success')]
    failures=[r for r in attempts if not r.get('success')]
    completed=[r for r in successes if r.get('end_t') is not None and r['end_t']<T]
    good=[r for r in completed if (r.get('ttft_ms') is not None and r.get('tpot_ms') is not None and
        r['ttft_ms']<=meta['ttft_slo_ms'] and r['tpot_ms']<=meta['tpot_slo_ms'])]
    checks={
        'unique_request_ids':len({r['id'] for r in rec})==scheduled,
        'scheduled_eq_attempted_plus_guard':scheduled==len(attempts)+len(drops),
        'all_attempts_end_with_success_or_failure':len(attempts)==len(successes)+len(failures),
        'summary_scheduled':scheduled==summary['scheduled'],
        'summary_drops':len(drops)==summary['client_guard_drops'],
        'summary_success':len(successes)==summary['success_total_after_drain'],
        'summary_failed':len(failures)==summary['failure_or_timeout'],
        'summary_completed_window':len(completed)==summary['completed_success_in_window'],
        'summary_good_window':len(good)==summary['slo_good_completions_in_window'],
        'completed_outputs_have_hash':all(isinstance(r.get('output_sha256'),str) and len(r['output_sha256'])==64 for r in successes),
    }
    warnings=[]
    if summary['client_guard_drops']:warnings.append('CLIENT_GUARD_DROPS: incomplete offered-load injection')
    if summary['failure_or_timeout']:warnings.append('FAILED_OR_TIMED_OUT_REQUESTS: report these, not necessarily an invalid overload run')
    if summary.get('preemptions_delta') is None:warnings.append('PREEMPTION_COUNTER_NOT_VERIFIED: metrics unavailable or reset')
    if summary.get('metrics_errors'):warnings.append('METRICS_SCRAPE_ERRORS')
    if summary.get('output_length_mismatch_count'):warnings.append('OUTPUT_LENGTH_MISMATCH: inspect finish reason, usage, timeout')
    report={'checks':checks,'all_pass':all(checks.values()),'warnings':warnings,'summary_subset':{k:summary.get(k) for k in ['running_peak_sampled','waiting_peak_sampled','kv_peak_window_sampled','preemptions_delta','output_length_mismatch_count']}}
    (root/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['all_pass'] else 2
if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: audit_day4.py DIRECTORY')
    raise SystemExit(audit(Path(sys.argv[1])))
