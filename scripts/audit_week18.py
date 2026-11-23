#!/usr/bin/env python3
"""Week 18 evidence auditor (offline, Python 3.9+, standard library only).

This is a consistency/evidence-presence auditor, NOT a capacity certificate.
Run from a repository containing Week18 results, e.g.:
 python week18_day7/scripts/audit_week18.py --root . --out results/week18_day7
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

SKIP_DIRS = {'.git','.venv','venv','node_modules','__pycache__','.cache','.mypy_cache','.pytest_cache','site-packages'}
DAY_RE = re.compile(r'(?:week[_-]?18[_-]?)?day[_-]?([1-6])(?!\d)',re.I)
LIMIT_ISSUES = 500


def record(issues, severity, code, source, message):
    if len(issues) < LIMIT_ISSUES:
        issues.append({'severity':severity,'code':code,'source':source,'detail':message})


def close(a,b,tol=1e-5):
    return abs(float(a)-float(b)) <= tol * max(1,abs(float(a)),abs(float(b)))


def as_num(row,key):
    value = row.get(key)
    return None if value is None or value == '' else float(value)


def collect(root, out):
    results=[]
    for base,dirs,files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            p=Path(base)/name
            if out == p or out in p.parents: continue
            if p.suffix.lower() in {'.csv','.json','.jsonl','.md','.txt','.sqlite','.nsys-rep','.log','.cmd','.sh','.py'}:
                results.append(p)
    return results


def audit_csv(path,issues):
    rel=str(path)
    try:
        with path.open('r',encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames:
                record(issues,'FAIL','CSV_HEADER',rel,'No CSV header'); return None
            columns=set(reader.fieldnames)
            count=0
            for row in reader:
                count+=1
                try:
                    if {'planned','sent_in_window','client_drops'} <= columns:
                        if not close(as_num(row,'planned'),as_num(row,'sent_in_window')+as_num(row,'client_drops')):
                            record(issues,'FAIL','DAY2_PLANNED',rel,f'row {count}: planned != sent + drops')
                    if {'completed_during_window','completed_after_drain'} <= columns:
                        if as_num(row,'completed_during_window') > as_num(row,'completed_after_drain'):
                            record(issues,'FAIL','DRAIN_ORDER',rel,f'row {count}: window > after-drain successes')
                    if {'sent_in_window','completed_during_window','outstanding_at_end','failed_timeout'} <= columns:
                        sent=as_num(row,'sent_in_window'); comp=as_num(row,'completed_during_window'); outstanding=as_num(row,'outstanding_at_end')
                        # outstanding may include failures completed in window; avoid strict equality here.
                        if outstanding < 0 or comp > sent or outstanding > sent:
                            record(issues,'FAIL','DAY2_COUNTS',rel,f'row {count}: invalid in-flight accounting')
                    if {'goodput_rate','completed_rate'} <= columns:
                        if as_num(row,'goodput_rate') > as_num(row,'completed_rate')+1e-4:
                            record(issues,'FAIL','GOOD_GT_COMPLETED',rel,f'row {count}: goodput > completed')
                    if {'goodput_req_s','completed_req_s'} <= columns:
                        if as_num(row,'goodput_req_s') > as_num(row,'completed_req_s')+1e-4:
                            record(issues,'FAIL','GOOD_GT_COMPLETED',rel,f'row {count}: goodput > completed')
                        t=as_num(row,'duration_s')
                        if t and as_num(row,'completed_total') is not None and as_num(row,'completed_req_s')*t>as_num(row,'completed_total')+0.1:
                            record(issues,'FAIL','WINDOW_GT_COHORT',rel,f'row {count}: window completions > total completed')
                    if {'goodput_window','completed_rate_window'} <= columns:
                        if as_num(row,'goodput_window')>as_num(row,'completed_rate_window')+1e-6:
                            record(issues,'FAIL','GOOD_GT_COMPLETED',rel,f'row {count}: completion-window goodput > completed rate')
                    if {'offered','local_reject','guard_drop'} <= columns:
                        if as_num(row,'local_reject')+as_num(row,'guard_drop')>as_num(row,'offered'):
                            record(issues,'FAIL','REJECT_GT_OFFERED',rel,f'row {count}: drops exceed offered')
                    for field in ['cohort_good_over_offered','cohort_good_over_success']:
                        val=as_num(row,field)
                        if val is not None and not 0<=val<=1:
                            record(issues,'FAIL','RATE_RANGE',rel,f'row {count}: {field} outside 0..1')
                except (ValueError,TypeError,KeyError) as e:
                    record(issues,'WARN','CSV_NONNUMERIC',rel,f'row {count}: {e}')
            return {'path':rel,'rows':count,'columns':list(reader.fieldnames)}
    except (OSError,UnicodeError,csv.Error) as e:
        record(issues,'FAIL','CSV_READ',rel,str(e)); return None


def audit_day6_json(path,issues):
    """Recompute Day6 counts from raw records when present; don't invent invariants from summaries."""
    source=str(path)
    try:
        report=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError) as e:
        record(issues,'FAIL','SUMMARY_READ',source,str(e)); return None
    if not isinstance(report,dict) or not {'meta','summary'} <= set(report): return None
    meta=report['meta']; summary=report['summary']
    if not (isinstance(summary,dict) and 'totals' in summary and 'phases' in summary): return None
    totals=summary['totals']
    offered=totals.get('offered'); sent=totals.get('sent'); rejected=totals.get('client_local_reject',0); guard=totals.get('client_guard_drop',0)
    if all(isinstance(x,(int,float)) for x in (offered,sent,rejected,guard)):
        if offered != sent+rejected+guard:
            record(issues,'FAIL','DAY6_OFFERED',source,f'{offered} != {sent}+{rejected}+{guard}')
    if totals.get('record_accounting_ok') is False:
        record(issues,'FAIL','DAY6_ACCOUNTING',source,'Probe reports record_accounting_ok=false')
    for p in summary['phases']:
        t=p.get('duration_s'); n=p.get('offered'); ng=p.get('cohort_good_after_drain'); nc=p.get('window_completed'); nw=p.get('window_good')
        if isinstance(n,int) and isinstance(ng,int) and ng>n:
            record(issues,'FAIL','DAY6_COHORT',source,f"{p.get('phase')}: good cohort > offered")
        if isinstance(nw,int) and isinstance(nc,int) and nw>nc:
            record(issues,'FAIL','DAY6_WINDOW',source,f"{p.get('phase')}: window good > window completed")
        if t and isinstance(nw,int) and not close(nw/t,p.get('goodput_window',-1),1e-6):
            record(issues,'FAIL','DAY6_RATE',source,f"{p.get('phase')}: window good count/rate inconsistent")
        # DO NOT require window completions <= same-phase injected in recovery phases.
    raw=path.with_name('request_records.json')
    if not raw.is_file():
        record(issues,'WARN','DAY6_RAW_MISSING',source,'request_records.json missing; summary-only consistency')
        return {'path':source,'raw_checked':False,'offered':offered}
    try:
        obj=json.loads(raw.read_text(encoding='utf-8'))
        requests=obj['requests']
    except (OSError,ValueError,KeyError,TypeError) as e:
        record(issues,'FAIL','DAY6_RAW_READ',str(raw),str(e)); return {'path':source,'raw_checked':False,'offered':offered}
    def good(r):
        return bool(r.get('success') and r.get('ttft_ms') is not None and r.get('tpot_ms') is not None and r['ttft_ms']<=meta['ttft_slo_ms'] and r['tpot_ms']<=meta['tpot_slo_ms'])
    if len(requests)!=offered:
        record(issues,'FAIL','DAY6_RAW_COUNT',source,f'records={len(requests)} summary offered={offered}')
    seen_ids=set()
    for r in requests:
        if r.get('id') in seen_ids:
            record(issues,'FAIL','DAY6_DUPLICATE_ID',source,f"Duplicate request id {r.get('id')}")
            break
        seen_ids.add(r.get('id'))
    actual={'offered':len(requests),'sent':sum(r.get('dispatch_t') is not None for r in requests),
            'client_local_reject':sum(r.get('status')=='local_reject' for r in requests),
            'client_guard_drop':sum(r.get('status')=='client_guard_drop' for r in requests),
            'success_after_drain':sum(bool(r.get('success')) for r in requests),
            'good_after_drain':sum(good(r) for r in requests)}
    for k,v in actual.items():
        if v!=totals.get(k): record(issues,'FAIL','DAY6_RAW_MISMATCH',source,f'{k}: raw {v} != summary {totals.get(k)}')
    for phase in summary['phases']:
        name=phase['phase']; cohort=[r for r in requests if r.get('arrival_phase')==name]
        for key,value in [('offered',len(cohort)),('local_reject',sum(r.get('status')=='local_reject' for r in cohort)),
                          ('guard_drop',sum(r.get('status')=='client_guard_drop' for r in cohort)),
                          ('cohort_good_after_drain',sum(good(r) for r in cohort))]:
            if phase.get(key)!=value: record(issues,'FAIL','DAY6_PHASE_MISMATCH',source,f'{name}: {key} raw={value} summary={phase.get(key)}')
        p_start=next((p['start'] for p in meta['phases'] if p['name']==name),None)
        p_end=next((p['end'] for p in meta['phases'] if p['name']==name),None)
        if p_start is not None and p_end is not None:
            completed=[r for r in requests if r.get('success') and r.get('end_t') is not None and p_start <= r['end_t'] <p_end]
            if phase.get('window_completed')!=len(completed): record(issues,'FAIL','DAY6_WINDOW_MISMATCH',source,f'{name}: window_completed')
            if phase.get('window_good')!=sum(good(r) for r in completed): record(issues,'FAIL','DAY6_WINDOW_MISMATCH',source,f'{name}: window_good')
    return {'path':source,'raw_checked':True,'offered':offered}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path,default=Path('results/week18_day7'))
    args=ap.parse_args()
    root=args.root.resolve(); out=args.out.resolve()
    if not root.is_dir(): ap.error(f'root not found: {root}')
    out.mkdir(parents=True,exist_ok=True)
    files=collect(root,out)
    issues=[]; days=defaultdict(lambda:Counter()); selected=[]
    for path in files:
        rel=path.relative_to(root)
        match=DAY_RE.search(str(rel))
        if match: days[match.group(1)][path.suffix.lower()]+=1
        if path.suffix=='.csv' and (match or 'comparison' in path.name):
            row=audit_csv(path,issues)
            if row: selected.append(row)
    day6=[]
    for path in files:
        if path.name=='summary.json' and (m:=DAY_RE.search(str(path))) and m.group(1)=='6':
            info=audit_day6_json(path,issues)
            if info: day6.append(info)
    for day in map(str,range(1,7)):
        if not days.get(day): record(issues,'WARN','DAY_NOT_FOUND',f'Day {day}','No Day-specific files discovered; check --root and file naming')
    if not day6: record(issues,'WARN','DAY6_RAW_NOT_FOUND','Day 6','No Day6 summary.json with associated raw records discovered')
    if not any(p.suffix=='.nsys-rep' or p.suffix=='.sqlite' for p in files):
        record(issues,'WARN','NSYS_MISSING','Day 5','No .nsys-rep/.sqlite found in scanned root; source report not independently auditable here')
    if not any(p.name=='release_audit.json' for p in files):
        record(issues,'INFO','RELEASE_NOT_FINAL','Release','Release closure checklist remains to be completed by user')
    counts=Counter(x['severity'] for x in issues)
    result={'tool':'week18_day7_auditor_v1','scope':str(root),'files_inspected':len(files),
            'days':{d:dict(days.get(d,{})) for d in map(str,range(1,7))},
            'csv_audits':selected,'day6_summary_audits':day6,'issues':issues,
            'issue_counts':dict(counts),'status':'FAIL' if counts['FAIL'] else ('WARN' if counts['WARN'] else 'PASS')}
    (out/'evidence_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Week 18 — Evidence Audit (automatically generated)',
           '',f"**Status: {result['status']}** (FAIL={counts['FAIL']}, WARN={counts['WARN']}, INFO={counts['INFO']})",
           '',f"Scanned: `{root}`, {len(files)} candidate files. No performance data was inferred.",'',
           '## Day file inventory','', '| Day | File types found |','|---|---|']
    for d in map(str,range(1,7)):
        groups=', '.join(f'{k or "(no ext)"}:{v}' for k,v in sorted(days.get(d,{}).items())) or 'None'
        lines.append(f'| {d} | {groups} |')
    lines.extend(['','## Raw consistency coverage','',f'- CSV files audited: {len(selected)}',
                  f'- Day6 summary.json audited: {len(day6)}',
                  f"- Day6 request-record raw audits: {sum(x['raw_checked'] for x in day6)}",'',
                  '## Findings',''])
    if not issues: lines.append('No findings in implemented invariant checks. This does NOT establish production capacity or scientific reproducibility.')
    for x in issues:
        lines.append(f"- **{x['severity']}** `{x['code']}` — `{x['source']}`: {x['detail']}")
    lines+=['','## Interpretation boundaries','',
            '- PASS means only that implemented invariants passed and expected evidence was discovered; it does not certify SLO capacity.',
            '- WARN means missing or insufficient evidence, not necessarily a failed experiment.',
            '- Window completions can exceed same-phase injections during overload recovery; this is NOT an error.',
            '- Cohort good/Offered and completion-window Goodput use different populations.',
            '- `.nsys-rep` and SQLite are inventoried but not parsed by this tool; open them in Nsight for the actual timeline.',
            '- Multi-seed confirmation, hardware fingerprints, output correctness and performance claims require manual review.']
    (out/'evidence_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'files':len(files),'csv':len(selected),'day6_raw':sum(x['raw_checked'] for x in day6),'counts':dict(counts),'output':str(out)},ensure_ascii=False,indent=2))
    return 2 if counts['FAIL'] else 0


if __name__=='__main__': raise SystemExit(main())
