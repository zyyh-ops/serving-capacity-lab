#!/usr/bin/env python3
"""Reversible, *vLLM 0.10.2 only* Scheduler instrumentation. Never patch while server runs."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

MARKER = '# W18D5_SCHED_TRACE_V1'
HELPER = '''# W18D5_SCHED_TRACE_V1 (installed by scripts/patch_scheduler.py)
import json as _w18_json
import os as _w18_os
_W18_EVENT_COUNT = 0
_W18_STEP = 0

def _w18_emit(kind, **fields):
    global _W18_EVENT_COUNT
    path = _w18_os.environ.get("W18_SCHED_TRACE_FILE")
    if not path:
        return
    limit = int(_w18_os.environ.get("W18_SCHED_TRACE_MAX_EVENTS", "100000"))
    if _W18_EVENT_COUNT >= limit:
        return
    _W18_EVENT_COUNT += 1
    record = dict(kind=kind, t_ns=time.monotonic_ns(),
                  pid=_w18_os.getpid(), **fields)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_w18_json.dumps(record, ensure_ascii=False) + "\\n")

def _w18_log_step(scheduler, sched_out):
    global _W18_STEP
    if not _w18_os.environ.get("W18_SCHED_TRACE_FILE"):
        return
    _W18_STEP += 1
    _w18_emit("step", step=_W18_STEP,
              n_running=len(scheduler.running), n_waiting=len(scheduler.waiting),
              total_tokens=sum(sched_out.num_scheduled_tokens.values()))
    for req_id, num_tok in sched_out.num_scheduled_tokens.items():
        req = scheduler.requests.get(req_id)
        if req is None:
            continue
        # This is the scheduling *pre-update* position. With prefix cache OFF,
        # a resumed request restarts from position 0. No GPU synchronization.
        _w18_emit("scheduled", step=_W18_STEP, request_id=str(req_id),
                  start=int(req.num_computed_tokens), n=int(num_tok),
                  request_tokens=int(req.num_tokens),
                  prompt_tokens=int(req.num_prompt_tokens))

'''


def get_target():
    ver = importlib.metadata.version('vllm')
    if ver != '0.10.2':
        raise SystemExit(f'Refusing: installed vLLM {ver!r} != 0.10.2')
    spec = importlib.util.find_spec('vllm')
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit('vLLM installation not found')
    target = Path(next(iter(spec.submodule_search_locations))) / 'v1/core/sched/scheduler.py'
    if not target.exists():
        raise SystemExit(f'No Scheduler source at {target}')
    return target


def sha(data):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['check', 'install', 'restore'])
    args = ap.parse_args()
    target = get_target()
    backup = target.with_name(target.name + '.w18d5.orig')
    source = target.read_text(encoding='utf-8')
    print(f'target={target} sha256-prefix={sha(source)} backup={backup}')
    if args.action == 'check':
        print(f'patched={MARKER in source}; backup_exists={backup.exists()}')
        return
    if args.action == 'restore':
        if not backup.exists():
            raise SystemExit('No backup; nothing to restore (no change made)')
        if MARKER not in source:
            raise SystemExit('Refusing restore: current source lacks our marker. Inspect manually.')
        original = backup.read_text(encoding='utf-8')
        target.write_text(original, encoding='utf-8')
        backup.unlink()  # restoration complete; future install can create a fresh backup
        print('RESTORED original scheduler.py; backup cleared. Restart vLLM processes.')
        return
    if MARKER in source:
        raise SystemExit('Already patched. Use check or restore.')
    if backup.exists():
        raise SystemExit('Backup exists; refusing to overwrite. Restore/inspect first.')
    # Strict anchors avoid silently patching another vLLM version or an edited tree.
    anchors = [
        ('class Scheduler(SchedulerInterface):', HELPER + '\nclass Scheduler(SchedulerInterface):'),
        ('                    self.kv_cache_manager.free(preempted_req)',
         '                    _w18_emit("preempt", step=_W18_STEP + 1,\n'
         '                              request_id=str(preempted_req.request_id),\n'
         '                              previously_computed=int(preempted_req.num_computed_tokens),\n'
         '                              request_tokens=int(preempted_req.num_tokens))\n'
         '                    self.kv_cache_manager.free(preempted_req)'),
        ('        self._update_after_schedule(scheduler_output)',
         '        _w18_log_step(self, scheduler_output)\n'
         '        self._update_after_schedule(scheduler_output)'),
    ]
    for old, new in anchors:
        count = source.count(old)
        if count != 1:
            raise SystemExit(f'Refusing: expected one anchor, found {count}: {old!r}')
        source = source.replace(old, new, 1)
    compile(source, str(target), 'exec')
    backup.write_text(target.read_text(encoding='utf-8'), encoding='utf-8')
    target.write_text(source, encoding='utf-8')
    print('INSTALLED. Restart server with W18_SCHED_TRACE_FILE set. Non-profile runs only.')


if __name__ == '__main__':
    main()
