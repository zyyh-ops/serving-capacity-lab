#!/usr/bin/env python3
"""Week18 Day6: fixed-window SLO-aware load, admission and overload-recovery probe.

Designed for vLLM 0.10.2 + aiohttp + transformers + vllm.benchmarks.datasets.
This is an educational CLIENT-SIDE admission gate, NOT vLLM server admission.
Every planned request is accounted for. Phase endpoints belong to a single server run.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

METRICS = {
    "waiting": "vllm:num_requests_waiting",
    "running": "vllm:num_requests_running",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
    "generation_tokens": "vllm:generation_tokens_total",
}
RX = re.compile(r'^([^\s{]+)(?:\{[^}]*\})?\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)')


def utc():
    return datetime.now(timezone.utc).isoformat()


def percentile(values, p):
    arr = sorted(v for v in values if isinstance(v, (float, int)) and math.isfinite(v))
    if not arr:
        return None
    k = (len(arr) - 1) * p / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    return arr[lo] + (k-lo) * (arr[hi]-arr[lo])


def parse_phases(s):
    result, cur = [], 0.0
    names = set()
    for item in s.split(','):
        bits = item.strip().split(':')
        if len(bits) != 3:
            raise ValueError('phases must be NAME:SECONDS:REQ_PER_SEC comma-separated')
        name, dur, rate = bits[0], float(bits[1]), float(bits[2])
        if not name or name in names or dur <= 0 or rate <= 0 or not name.replace('_','').isalnum():
            raise ValueError('invalid phase, duplicate name, or nonpositive rate/duration')
        result.append(dict(name=name, start=cur, end=cur+dur, seconds=dur, rate=rate))
        names.add(name)
        cur += dur
    return result


def schedule_requests(phases, seed, long_frac, max_requests):
    rng = random.Random(seed)
    category_rng = random.Random(seed + 100003)
    req = []
    for phase in phases:
        t = phase['start']
        while True:
            t += rng.expovariate(phase['rate'])
            if t >= phase['end']:
                break
            if len(req) >= max_requests:
                raise ValueError('max_requests exceeded; use a shorter experiment or explicitly increase cap')
            req.append({'id':len(req), 'planned_t':t, 'arrival_phase':phase['name'],
                        'category':'long' if category_rng.random() < long_frac else 'short'})
    return req


def parse_metrics(body):
    values = {k: [] for k in METRICS}
    for line in body.splitlines():
        if not line or line[0] == '#':
            continue
        m = RX.match(line)
        if not m:
            continue
        for k, metric in METRICS.items():
            if m.group(1) == metric:
                values[k].append(float(m.group(2)))
    return {k: (max(v) if k == 'kv_usage' else sum(v)) if v else None for k,v in values.items()}


async def monitor(session, url, t0, stop, output, interval):
    while not stop.is_set():
        rec = {'t':time.monotonic()-t0, 'utc':utc()}
        try:
            import aiohttp
            async with session.get(url+'/metrics',timeout=aiohttp.ClientTimeout(total=4)) as res:
                if res.status != 200:
                    raise RuntimeError('metrics HTTP ' + str(res.status))
                rec.update(parse_metrics(await res.text()))
        except Exception as err:
            rec['metrics_error'] = str(err)
        output.append(rec)
        try:
            await asyncio.wait_for(stop.wait(),timeout=interval)
        except asyncio.TimeoutError:
            pass


class Bucket:
    def __init__(self,rate,capacity):
        if rate <= 0 or capacity <= 0:
            raise ValueError('bucket refill rate/capacity must be positive')
        self.rate=float(rate)
        self.capacity=float(capacity)
        self.balance=float(capacity)
        self.last=0.0

    def allow(self,t,cost=1.0):
        self.balance=min(self.capacity,self.balance+max(0.0,t-self.last)*self.rate)
        self.last=t
        if self.balance+1e-8 < cost:
            return False
        self.balance-=cost
        return True


def build_payload(sample,model):
    return {
        'model':model,'messages':[{'role':'user','content':sample.prompt}],
        'max_completion_tokens':int(sample.expected_output_len),'temperature':0.0,
        'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True},
    }


async def send_one(session,base_url,model,sample,original,t0,timeout,records):
    import aiohttp
    started=time.monotonic()
    rec={**original,'status':'sent','http_status':None,'success':False,
         'error':None,'dispatch_t':started-t0,
         'dispatch_lag_ms':(started-t0-original['planned_t'])*1000,
         'end_t':None,'ttft_ms':None,'tpot_ms':None,'e2e_ms':None,
         'output_tokens':None,'finish_reason':None,'output_sha256':None}
    first_choice=last_data=None
    usage_tokens=None
    done=False
    chunks=[]
    try:
        async with session.post(base_url+'/v1/chat/completions',json=build_payload(sample,model),
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            rec['http_status']=resp.status
            if resp.status != 200:
                rec['error']='HTTP '+str(resp.status)+': '+(await resp.text())[:400]
            else:
                lines=[]
                async for raw in resp.content:
                    line=raw.decode('utf-8',errors='replace').strip('\r\n')
                    if line == '':
                        if not lines:
                            continue
                        data='\n'.join(lines)
                        lines=[]
                        if data=='[DONE]':
                            done=True
                            continue
                        now=time.monotonic()
                        try:
                            obj=json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        last_data=now
                        if obj.get('choices'):
                            if first_choice is None:
                                first_choice=now
                            for choice in obj['choices']:
                                delta=choice.get('delta') or {}
                                if isinstance(delta.get('content'),str):
                                    chunks.append(delta['content'])
                                if choice.get('finish_reason') is not None:
                                    rec['finish_reason']=choice['finish_reason']
                        if isinstance(obj.get('usage'),dict):
                            usage_tokens=obj['usage'].get('completion_tokens')
                    elif line.startswith('data:'):
                        lines.append(line[5:].lstrip(' '))
                if first_choice is None or last_data is None or usage_tokens is None or not done:
                    rec['error']='incomplete SSE: choices=%s usage=%s done=%s' % (first_choice is not None,usage_tokens is not None,done)
                else:
                    rec['success']=True
                    rec['output_tokens']=int(usage_tokens)
                    rec['ttft_ms']=(first_choice-started)*1000
                    rec['e2e_ms']=(last_data-started)*1000
                    rec['tpot_ms']=(rec['e2e_ms']-rec['ttft_ms'])/(usage_tokens-1) if usage_tokens>1 else 0.0
                    rec['output_sha256']=hashlib.sha256(''.join(chunks).encode()).hexdigest()
    except asyncio.CancelledError:
        rec['error']='client_cancelled_at_drain_deadline'
        raise
    except Exception as err:
        rec['error']=repr(err)
    finally:
        rec['end_t']=time.monotonic()-t0
        records.append(rec)


def phase_for(t,phases):
    if t is None:
        return 'unknown'
    for phase in phases:
        if phase['start']<=t<phase['end']:
            return phase['name']
    if t>=phases[-1]['end']:
        return 'drain'
    return 'before'


def good(r,ttft_slo,tpot_slo):
    return (r.get('success') and r.get('ttft_ms') is not None and
            r.get('tpot_ms') is not None and r['ttft_ms']<=ttft_slo and r['tpot_ms']<=tpot_slo)


def metric_delta(rows,key):
    vals=[r[key] for r in rows if r.get(key) is not None]
    if len(vals)<2 or any(b<a for a,b in zip(vals,vals[1:])):
        return None
    return vals[-1]-vals[0]


def summary(meta,records,metrics):
    phases=meta['phases']
    ttft=meta['ttft_slo_ms']; tpot=meta['tpot_slo_ms']
    total_end=phases[-1]['end']
    phase_rows=[]
    for p in phases:
        arrivals=[r for r in records if r['arrival_phase']==p['name']]
        injected=[r for r in arrivals if r.get('dispatch_t') is not None and r['dispatch_t']<p['end']]
        successes_cohort=[r for r in arrivals if r.get('success')]
        cohort_good=[r for r in arrivals if good(r,ttft,tpot)]
        completed=[r for r in records if r.get('success') and p['start']<=r.get('end_t',float('inf'))<p['end']]
        window_good=[r for r in completed if good(r,ttft,tpot)]
        samples=[r for r in metrics if p['start']<=r['t']<p['end']]
        q=[r['waiting'] for r in samples if r.get('waiting') is not None]
        running=[r['running'] for r in samples if r.get('running') is not None]
        phase_rows.append({
            'phase':p['name'],'duration_s':p['seconds'],'rate_target':p['rate'],
            'offered':len(arrivals),'injected_in_phase':len(injected),
            'local_reject':sum(r.get('status')=='local_reject' for r in arrivals),
            'guard_drop':sum(r.get('status')=='client_guard_drop' for r in arrivals),
            'cohort_success_after_drain':len(successes_cohort),'cohort_good_after_drain':len(cohort_good),
            'offered_good_fraction':len(cohort_good)/len(arrivals) if arrivals else None,
            'admitted_good_fraction':len(cohort_good)/len(successes_cohort) if successes_cohort else None,
            'window_completed':len(completed),'window_good':len(window_good),
            'offered_rate_realized':len(arrivals)/p['seconds'],
            'injected_rate_proxy':len(injected)/p['seconds'],
            'completed_rate_window':len(completed)/p['seconds'],
            'goodput_window':len(window_good)/p['seconds'],
            'ttft_p99_cohort_ms':percentile([r['ttft_ms'] for r in successes_cohort],99),
            'tpot_p99_cohort_ms':percentile([r['tpot_ms'] for r in successes_cohort],99),
            'waiting_peak_sampled':max(q) if q else None,
            'waiting_first_half_mean':statistics.mean(q[:len(q)//2]) if len(q)>=2 else None,
            'waiting_last_half_mean':statistics.mean(q[len(q)//2:]) if q else None,
            'running_peak_sampled':max(running) if running else None,
            'kv_peak_sampled':max((r['kv_usage'] for r in samples if r.get('kv_usage') is not None),default=None),
            'metrics_samples':len(samples),
        })
    group_rows=[]
    for cat in ('short','long'):
        cohort=[r for r in records if r['category']==cat]
        if not cohort:
            continue
        succ=[r for r in cohort if r.get('success')]
        gc=[r for r in cohort if good(r,ttft,tpot)]
        group_rows.append({'category':cat,'offered':len(cohort),'success':len(succ),
                           'good':len(gc),'good_over_offered':len(gc)/len(cohort),
                           'ttft_p99_ms':percentile([r['ttft_ms'] for r in succ],99),
                           'tpot_p99_ms':percentile([r['tpot_ms'] for r in succ],99)})
    offered=len(records)
    sent=[r for r in records if r.get('dispatch_t') is not None]
    success=[r for r in records if r.get('success')]
    return {
        'totals':{'offered':offered,'sent':len(sent),'client_local_reject':sum(r['status']=='local_reject' for r in records),
                  'client_guard_drop':sum(r['status']=='client_guard_drop' for r in records),
                  'success_after_drain':len(success),'http_error_or_timeout':sum(r['status']=='sent' and not r['success'] for r in records),
                  'good_after_drain':sum(good(r,ttft,tpot) for r in records),
                  'good_over_offered':sum(good(r,ttft,tpot) for r in records)/offered if offered else None,
                  'incomplete_output_length':sum(r['success'] and r['output_tokens']!=r['output_len_target'] for r in records),
                  'end_arrival_window_s':total_end,'last_completion_s':max((r.get('end_t') or 0 for r in records),default=0),
                  'metrics_errors':sum('metrics_error' in r for r in metrics),
                  'preemptions_delta':metric_delta(metrics,'preemptions'),
                  'record_accounting_ok':len(sent)+sum(r['status'] in ('local_reject','client_guard_drop') for r in records)==offered},
        'phases':phase_rows,'categories':group_rows,
    }


def build_samples(schedule,args):
    from transformers import AutoTokenizer
    from vllm.benchmarks.datasets import RandomDataset
    tok=AutoTokenizer.from_pretrained(args.model)
    shorts=sum(r['category']=='short' for r in schedule)
    longs=len(schedule)-shorts
    sr=iter(RandomDataset(random_seed=args.seed+17).sample(tokenizer=tok, num_requests=shorts, input_len=args.short_input, output_len=args.short_output, range_ratio=0.0) if shorts else [])
    lr=iter(RandomDataset(random_seed=args.seed+29).sample(tokenizer=tok, num_requests=longs, input_len=args.long_input, output_len=args.long_output, range_ratio=0.0) if longs else [])
    return [next(sr) if r['category']=='short' else next(lr) for r in schedule]


async def run(args):
    import aiohttp
    phases=parse_phases(args.phases)
    schedule=schedule_requests(phases,args.seed,args.long_frac,args.max_requests)
    if not schedule:
        raise ValueError('zero requests generated')
    samples=build_samples(schedule,args)
    for r,s in zip(schedule,samples):
        r['input_len_target']=args.short_input if r['category']=='short' else args.long_input
        r['output_len_target']=int(s.expected_output_len)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        raise FileExistsError('refuse to overwrite existing summary: '+str(out))
    config={
        'created_utc':utc(),'base_url':args.base_url,'model':args.model,
        'seed':args.seed,'phases':phases,'policy':args.policy,
        'admit_rate':args.admit_rate,'burst':args.burst,
        'cost_rate':args.cost_rate,'cost_burst':args.cost_burst,
        'cost_prompt_weight':args.cost_prompt_weight,'cost_output_weight':args.cost_output_weight,
        'inflight_cap':args.inflight_cap,'max_inflight_guard':args.max_inflight_guard,
        'short_input':args.short_input,'short_output':args.short_output,
        'long_input':args.long_input,'long_output':args.long_output,'long_frac':args.long_frac,
        'ttft_slo_ms':args.ttft_slo,'tpot_slo_ms':args.tpot_slo,
        'request_timeout_s':args.request_timeout,'drain_timeout_s':args.drain_timeout,
        'measurement_warning':'Local reject is synthetic CLIENT-side rejection, not vLLM HTTP 429. Dispatch_t is not socket transmission. Arrival-phase cohorts differ from completion-phase rates; no claim of true steady state.'}
    bucket=None
    if args.policy=='rps':
        bucket=Bucket(args.admit_rate,args.burst)
    if args.policy=='cost':
        bucket=Bucket(args.cost_rate,args.cost_burst)
    records,metrics,pending=[],[],set()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0,limit_per_host=0)) as session:
        async with session.get(args.base_url+'/v1/models',timeout=aiohttp.ClientTimeout(total=10)) as res:
            if res.status!=200:
                raise RuntimeError('server not ready: '+str(res.status))
        loop=asyncio.get_running_loop()
        t0=loop.time()
        stop=asyncio.Event()
        sampler=asyncio.create_task(monitor(session,args.base_url,t0,stop,metrics,args.metrics_interval))
        for r,s in zip(schedule,samples):
            await asyncio.sleep(max(0.0,t0+r['planned_t']-loop.time()))
            t=loop.time()-t0
            reason=None
            # A stable deterministic bucket uses the planned arrival time, not the
            # event-loop dispatch time. This preserves admission decisions across repeats.
            if bucket is not None:
                cost=1 if args.policy=='rps' else (args.cost_prompt_weight*r['input_len_target']+
                                                    args.cost_output_weight*r['output_len_target'])
                if not bucket.allow(r['planned_t'],cost):
                    reason='local_reject'
            if reason is None and args.policy=='inflight' and len(pending)>=args.inflight_cap:
                reason='local_reject'
            if reason is None and len(pending)>=args.max_inflight_guard:
                reason='client_guard_drop'
            if reason is not None:
                records.append({**r,'status':reason,'success':False,'http_status':None,'error':reason,
                                'dispatch_t':None,'end_t':t,'ttft_ms':None,'tpot_ms':None})
                continue
            task=asyncio.create_task(send_one(session,args.base_url,args.model,s,r,t0,args.request_timeout,records))
            pending.add(task)
            task.add_done_callback(pending.discard)
        await asyncio.sleep(max(0.0,t0+phases[-1]['end']-loop.time()))
        if pending:
            _,left=await asyncio.wait(pending,timeout=args.drain_timeout)
            if left:
                for task in left:
                    task.cancel()
                await asyncio.gather(*left,return_exceptions=True)
        stop.set()
        await sampler
    config['end_utc']=utc()
    config['total_elapsed_s']=time.monotonic()-t0
    records.sort(key=lambda r:r['id'])
    report=summary(config,records,metrics)
    (out/'request_records.json').write_text(json.dumps({'meta':config,'requests':records},ensure_ascii=False,indent=2))
    (out/'metrics.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in metrics))
    (out/'summary.json').write_text(json.dumps({'meta':config,'summary':report},ensure_ascii=False,indent=2))
    print(json.dumps({'out':str(out),'summary':report},ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',default='http://127.0.0.1:8000')
    ap.add_argument('--model',default='Qwen/Qwen3-0.6B')
    ap.add_argument('--out',required=True)
    ap.add_argument('--phases',default='baseline:35:65,overload:25:105,recovery:60:65')
    ap.add_argument('--seed',type=int,default=2029)
    ap.add_argument('--long-frac',type=float,default=0)
    ap.add_argument('--short-input',type=int,default=128)
    ap.add_argument('--short-output',type=int,default=128)
    ap.add_argument('--long-input',type=int,default=512)
    ap.add_argument('--long-output',type=int,default=128)
    ap.add_argument('--policy',choices=['open','rps','cost','inflight'],default='open')
    ap.add_argument('--admit-rate',type=float,default=75)
    ap.add_argument('--burst',type=float,default=20)
    ap.add_argument('--cost-rate',type=float,default=20000)
    ap.add_argument('--cost-burst',type=float,default=6000)
    ap.add_argument('--cost-prompt-weight',type=float,default=1)
    ap.add_argument('--cost-output-weight',type=float,default=2)
    ap.add_argument('--inflight-cap',type=int,default=128)
    ap.add_argument('--max-inflight-guard',type=int,default=2048)
    ap.add_argument('--max-requests',type=int,default=30000)
    ap.add_argument('--request-timeout',type=float,default=150)
    ap.add_argument('--drain-timeout',type=float,default=120)
    ap.add_argument('--metrics-interval',type=float,default=.25)
    ap.add_argument('--ttft-slo',type=float,default=200)
    ap.add_argument('--tpot-slo',type=float,default=15)
    args=ap.parse_args()
    if not 0<=args.long_frac<=1 or args.max_requests<1:
        ap.error('long-frac must be in [0,1]; max-requests >= 1')
    if any(x<=0 for x in (args.short_input,args.short_output,args.long_input,args.long_output,
                             args.metrics_interval,args.max_inflight_guard,args.request_timeout,args.drain_timeout)):
        ap.error('lengths, intervals and timeouts must be positive')
    try:
        asyncio.run(run(args))
    except (ValueError,FileExistsError) as err:
        ap.error(str(err))

if __name__=='__main__':
    main()
