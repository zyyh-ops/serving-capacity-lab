#!/usr/bin/env python3
"""Count repeated *scheduled logical positions* in a small KV pressure trace.

Limitations: assumes single text-only request stream, no speculative decode,
no prefix caching, no model parallel; scheduled != guaranteed executed if engine crashes.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(path):
    seen_high = defaultdict(int)
    preempted = defaultdict(int)
    by_kind = Counter()
    scheduled = unique = repeated = 0
    step_cnt = 0
    repeat_examples = []
    for line in Path(path).open(encoding='utf-8'):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # permit partial final line when process is stopped
        kind = row.get('kind')
        by_kind[kind] += 1
        if kind == 'step':
            step_cnt += 1
        elif kind == 'preempt':
            preempted[row['request_id']] += 1
        elif kind == 'scheduled':
            rid, start, n = row['request_id'], int(row['start']), int(row['n'])
            if n <= 0 or start < 0:
                continue
            end = start + n
            overlap = max(0, min(end, seen_high[rid]) - start)
            scheduled += n
            repeated += overlap
            unique += n - overlap
            if overlap and len(repeat_examples) < 20:
                repeat_examples.append({'request_id': rid, 'step': row.get('step'),
                                        'start': start, 'end': end,
                                        'reexecuted_logical_positions': overlap})
            seen_high[rid] = max(seen_high[rid], end)
    return {
        'events': dict(by_kind), 'steps': step_cnt,
        'n_requests_with_scheduled_work': len(seen_high),
        'n_requests_preempted': len(preempted),
        'preemption_events': sum(preempted.values()),
        'scheduled_logical_token_occurrences': scheduled,
        'unique_logical_token_positions': unique,
        'repeated_logical_token_positions': repeated,
        'repeat_fraction_of_scheduled': repeated / scheduled if scheduled else 0,
        'repeat_examples': repeat_examples,
        'limits': [
            'This is a lower bound for the captured trace window; truncation / partial lifetimes bias counts.',
            'Counted logical scheduling positions, not exact GPU FLOPs or time, and no guaranteed model execution on failure.',
            'Only meaningful with prefix cache OFF, single-engine, text-only, no speculative decoding.',
            'If scheduled=0, check patch installation, runtime log path, and scheduler implementation.',
        ]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('trace_jsonl')
    p.add_argument('--output', default='')
    args = p.parse_args()
    result = analyze(args.trace_jsonl)
    out = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(out+'\n', encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
