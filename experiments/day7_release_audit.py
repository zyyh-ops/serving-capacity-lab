#!/usr/bin/env python3
"""Week 17 Day 7: read-only evidence audit for the vLLM 0.10.2 study project.

Usage: python day7_release_audit.py --root . --out release_audit
Requires Python 3.10+, standard library only. Never modifies source evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

DAY6 = [f'concurrency_{n}_short_decode' for n in (1, 2, 4, 8, 16, 32)] + [
    'concurrency_8_balanced', 'concurrency_8_long_prefill',
] + [f'rate_{n}_short_decode' for n in (2, 5, 10, 20)]

DOCS = {
    'day2_source_trace.md': ('day2_source_trace.md', 'day2_source_trace(1).md'),
    'day3_source_trace.md': ('day3_source_trace.md',),
    'day4_source_trace.md': ('day4_source_trace.md',),
    'day5_source_trace.md': ('day5_source_trace.md',),
    'day5_benchmark_analysis.md': ('day5_benchmark_analysis.md',),
    'day6_benchmark_analysis.md': ('day6_benchmark_analysis.md', 'day6_benchmark_analysis(1).md'),
    'day6_theory_answers.md': ('day6_theory_answers.md',),
}

EVIDENCE = ['day3_prefix.json', 'day3_pressure.json', 'day4_runner_test.json',
            'day5_no_graph.json', 'day5_piecewise.json']


def locate(root: Path, *names: str) -> Path | None:
    for name in names:
        for parent in (root / 'results', root / 'docs', root / 'experiments', root):
            p = parent / name
            if p.is_file():
                return p
    return None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(p: Path) -> dict:
    x = json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(x, dict):
        raise ValueError('Expected JSON object')
    return x


def check_day6(root, checks, warnings, stats, files):
    for base in DAY6:
        p = locate(root, base + '.json')
        if not p:
            checks.append({'name': f'Day6 {base}: client JSON', 'ok': False, 'reason': 'missing'})
            continue
        files[str(p.relative_to(root))] = sha256(p)
        try:
            d = load_json(p)
            nums = d['output_lens']
            completed = int(d['completed'])
            num_prompts = int(d['num_prompts'])
            total_output = int(d['total_output_tokens'])
            throughput = float(d['request_throughput'])
            goodput = float(d['request_goodput'])
            ok = (completed == num_prompts == 128 and len(nums) == completed
                  and sum(nums) == total_output and 0 <= goodput <= throughput + 1e-7
                  and total_output > 0 and float(d['duration']) > 0)
            checks.append({'name': f'Day6 {base}: length/throughput consistency', 'ok': ok,
                           'reason': f'completed={completed}, tokens={total_output}'})
            stats[base] = {'requests': completed,
                           'output_tokens_per_s': round(float(d['output_throughput']), 3),
                           'request_goodput': round(goodput, 3)}
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            checks.append({'name': f'Day6 {base}: JSON format', 'ok': False, 'reason': str(e)})

        m = locate(root, base + '_metrics.jsonl')
        if not m:
            warnings.append(f'Day6 {base}: no server metrics JSONL')
        else:
            files[str(m.relative_to(root))] = sha256(m)
            try:
                samples = [json.loads(s) for s in m.read_text(encoding='utf-8').splitlines() if s.strip()]
                if not samples:
                    warnings.append(f'Day6 {base}: metrics JSONL empty')
                else:
                    st = stats.setdefault(base, {})
                    st['metrics_samples'] = len(samples)
                    for metric, key in [('vllm:num_requests_waiting', 'waiting_peak'),
                                        ('vllm:num_requests_running', 'running_peak'),
                                        ('vllm:gpu_cache_usage_perc', 'kv_usage_peak')]:
                        vals = [float(row.get('metrics', {}).get(metric, 0)) for row in samples]
                        st[key] = round(max(vals), 6)
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                warnings.append(f'Day6 {base}: metrics parse error {e}')


def check_day5(root, checks, stats):
    p1 = locate(root, 'day5_no_graph.json')
    p2 = locate(root, 'day5_piecewise.json')
    if not p1 or not p2:
        checks.append({'name': 'Day5 paired benchmarks', 'ok': False, 'reason': 'missing one/both JSON'})
        return
    try:
        a, b = load_json(p1), load_json(p2)
        hashes_same = True
        req_count = 0
        for name in ('single_batch', 'concurrent_batch'):
            left, right = a[name], b[name]
            if len(left) != len(right):
                hashes_same = False
            for x, y in zip(left, right):
                if x['output_hashes'] != y['output_hashes'] or x['output_tokens'] != y['output_tokens']:
                    hashes_same = False
                req_count += x['batch_size']
        for x, y in zip(a['streaming'], b['streaming']):
            if x['hash'] != y['hash']:
                hashes_same = False
        if len(a['streaming']) != len(b['streaming']):
            hashes_same = False
        checks.append({'name': 'Day5 comparable result hashes', 'ok': hashes_same,
                       'reason': f'checked {req_count} non-streaming + {len(a["streaming"])} streaming requests'})
        s1, s2 = a['summary'], b['summary']
        stats['day5'] = {
            'single_e2e_speedup': round(s1['single_e2e_p50_ms'] / s2['single_e2e_p50_ms'], 3),
            'batch8_e2e_speedup': round(s1['batch8_e2e_p50_ms'] / s2['batch8_e2e_p50_ms'], 3),
        }
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
        checks.append({'name': 'Day5 JSON cross-check', 'ok': False, 'reason': str(e)})


def check_day4(root, checks, warnings):
    p = locate(root, 'day4_runner_trace.log')
    if not p:
        warnings.append('Day4 runner trace not archived')
        return
    count = 0
    for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
        if 'W17D4_RUNNER' not in line:
            continue
        try:
            d = json.loads(line.split('W17D4_RUNNER ', 1)[1])
            seq = d['query_start_loc']
            x = d['scheduled_tokens']
            n = d['total_scheduled_tokens']
            valid = (seq[0] == 0 and len(seq) == len(x) + 1 and seq[-1] == n
                     and sum(x) == n and all(seq[i + 1] - seq[i] == v
                                            for i, v in enumerate(x)))
        except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
            valid = False
        if not valid:
            checks.append({'name': f'Day4 trace line {count+1}', 'ok': False,
                           'reason': 'query cumulative sum invariant violation'})
        count += 1
    checks.append({'name': 'Day4 runner trace invariants', 'ok': count > 0,
                   'reason': f'{count} trace records scanned'})


def check_secrets(root, checks, warnings):
    try:
        r = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], capture_output=True,
                           check=True, timeout=8)
        names = [s.decode('utf-8', errors='replace') for s in r.stdout.split(b'\0') if s]
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        warnings.append('Not a git repo; git tracked-file security review skipped')
        return
    if not names:
        warnings.append('No tracked files yet; secret scan skipped')
    hits = []
    secret = re.compile(r'(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    suspicious = ('.env', '.pem', '.key', '.sqlite', '.nsys-rep')
    for name in names:
        p = root / name
        if not p.is_file():
            continue
        if p.name == '.env' or p.suffix.lower() in suspicious[1:]:
            hits.append(f'{name}: sensitive/large artifact tracked')
        if p.stat().st_size > 10 * 1024 * 1024:
            hits.append(f'{name}: >10 MiB tracked; consider Git LFS / external evidence')
        if p.stat().st_size <= 1024 * 1024 and p.suffix.lower() in ('.py', '.sh', '.md', '.txt', '.yaml', '.yml', '.json', '.toml', '.env'):
            data = p.read_text(encoding='utf-8', errors='replace')
            if secret.search(data):
                hits.append(f'{name}: possible secret token (value deliberately hidden)')
    checks.append({'name': 'Tracked-file sensitive content scan', 'ok': not hits,
                   'reason': '; '.join(hits[:12]) or f'{len(names)} tracked files inspected'})
    if hits:
        warnings.append('Do not commit/share until tracked-file findings are reviewed')


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--root', type=Path, default=Path('.'))
    pa.add_argument('--out', type=Path, default=Path('release_audit'))
    args = pa.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checks, warnings, stats, files = [], [], {}, {}

    for display, options in DOCS.items():
        p = locate(root, *options)
        checks.append({'name': 'Document: ' + display, 'ok': bool(p), 'reason': str(p.relative_to(root)) if p else 'missing'})
        if p:
            files[str(p.relative_to(root))] = sha256(p)

    for name in EVIDENCE:
        p = locate(root, name)
        checks.append({'name': 'Evidence: ' + name, 'ok': bool(p), 'reason': str(p.relative_to(root)) if p else 'missing'})
        if p:
            files[str(p.relative_to(root))] = sha256(p)

    check_day5(root, checks, stats)
    check_day6(root, checks, warnings, stats, files)
    check_day4(root, checks, warnings)
    check_secrets(root, checks, warnings)

    for fn in ('day3_server_trace.log', 'day5_no_graph_profile.nsys-rep',
               'day5_piecewise_profile.nsys-rep', 'day6_server.log'):
        if not locate(root, fn):
            warnings.append(f'Optional original evidence missing: {fn}')

    overall = all(row['ok'] for row in checks)
    report = {'status': 'PASS' if overall else 'REVIEW_REQUIRED',
              'root': str(root), 'checks': checks, 'warnings': warnings,
              'statistics': stats, 'sha256_manifest': files}
    (out / 'release_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = [f'# Week 17 Release Evidence Audit — {report["status"]}', '',
             '自动检查仅覆盖文件存在性、结构与部分数值不变量；不替代真实 GPU 回归、性能复现和人工源码审计。', '',
             '## Checks', '']
    lines += [f'- [{"x" if r["ok"] else " "}] {r["name"]} — {r["reason"]}' for r in checks]
    lines += ['', '## Warnings / limitations', '']
    lines += [f'- {w}' for w in warnings] or ['- 无额外警告']
    lines += ['', '## Metrics summary', '', '```json', json.dumps(stats, ensure_ascii=False, indent=2), '```', '']
    (out / 'release_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'passed': sum(x['ok'] for x in checks),
                      'total': len(checks), 'warnings': len(warnings), 'output': str(out)}, ensure_ascii=False))
    raise SystemExit(0 if overall else 1)

if __name__ == '__main__':
    main()
