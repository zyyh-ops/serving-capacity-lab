#!/usr/bin/env python3
"""Week18 Day3: mixed-workload open-loop probe; requires vLLM 0.10.2 environment.

Same short-class RNG/time schedule and prompt seed in short-only and mixed runs.
Times use a monotonic clock. Client observations are not server queue timings.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

METRICS = {
    'waiting':'vllm:num_requests_waiting',
    'running':'vllm:num_requests_running',
    'kv_usage':'vllm:kv_cache_usage_perc',
    'preemptions':'vllm:num_preemptions_total',
    'prompt_tokens':'vllm:prompt_tokens_total',
    'generation_tokens':'vllm:generation_tokens_total',
}
METRIC_RE = re.compile(r'^([^\s{]+)(?:\{[^}]*\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')


def q(values, pct):
    a=sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not a: return None
    p=(len(a)-1)*pct/100
    lo=math.floor(p); hi=math.ceil(p)
    return round(a[lo]+(p-lo)*(a[hi]-a[lo]),4)


def arrival_times(rate, duration, seed, max_requests):
    if rate<=0: return []
    rng=random.Random(seed); t=0.; out=[]
    while True:
        t+=rng.expovariate(rate)
        if t>=duration: break
        out.append(t)
        if len(out)>max_requests: raise ValueError('Exceeded --max-requests')
    return out


def build_schedule(args):
    events=[]
    for group, rate, seed in [('short',args.short_rate,args.seed),('long',args.long_rate,args.seed+100003)]:
        events.extend({'group':group,'planned_t':round(t,8),'group_index':i}
                      for i,t in enumerate(arrival_times(rate,args.duration,seed,args.max_requests)))
    events.sort(key=lambda e:(e['planned_t'],e['group']))
    if len(events)>args.max_requests: raise ValueError('Combined events exceed max-requests')
    for i,e in enumerate(events): e['id']=i
    return events


def decode_prometheus(body):
    d={k:[] for k in METRICS}
    for ln in body.splitlines():
        if not ln or ln.startswith('#'): continue
        m=METRIC_RE.match(ln)
        if not m: continue
        for key,name in METRICS.items():
            if m.group(1)==name: d[key].append(float(m.group(2)))
    return {k:(max(a) if k=='kv_usage' else sum(a)) if a else None for k,a in d.items()}


async def scrape_metrics(session,base,t0,done,records,interval):
    import aiohttp
    while not done.is_set():
        row={'t':round(time.monotonic()-t0,6)}
        try:
            async with session.get(base+'/metrics',timeout=aiohttp.ClientTimeout(total=3)) as r:
                if r.status!=200: raise RuntimeError(f'/metrics HTTP {r.status}')
                row.update(decode_prometheus(await r.text()))
        except Exception as ex: row['error']=repr(ex)
        records.append(row)
        try: await asyncio.wait_for(done.wait(), timeout=interval)
        except asyncio.TimeoutError: pass


def event_data(data,instant,state):
    if data=='[DONE]': state['done']=True; return
    try: obj=json.loads(data)
    except Exception: return
    if not isinstance(obj,dict): return
    state['last_data']=instant
    if obj.get('choices'):
        state['choice_ts'].append(instant)
        if state['first_choice'] is None: state['first_choice']=instant
    if isinstance(obj.get('usage'),dict):
        tok=obj['usage'].get('completion_tokens')
        if tok is not None: state['tokens']=int(tok)


async def send_one(session,args,e,sample,t0,records):
    import aiohttp
    now=time.monotonic()
    r={'id':e['id'],'group':e['group'],'group_index':e['group_index'],
       'planned_t':e['planned_t'],'dispatch_t':round(now-t0,6),
       'dispatch_lag_ms':round(1000*(now-t0-e['planned_t']),3),
       'status':None,'success':False,'error':None,
       'prompt_tokens_est':sample['prompt_tokens_est'],
       'output_tokens':None,'ttft_ms':None,'tpot_ms':None,
       'e2e_ms':None,'choice_events':None,'sse_event_gap_p95_ms':None,'end_t':None}
    st={'first_choice':None,'last_data':None,'tokens':None,'done':False,'choice_ts':[]}
    payload={'model':args.model,'messages':[{'role':'user','content':sample['prompt']}],
             'temperature':0.0,'max_completion_tokens':sample['output_len'],
             'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}
    try:
        async with session.post(args.base_url+'/v1/chat/completions',json=payload,
                                timeout=aiohttp.ClientTimeout(total=args.request_timeout)) as response:
            r['status']=response.status
            if response.status!=200:
                r['error']=(await response.text())[:1000]
            else:
                parts=[]
                async for line in response.content:
                    s=line.decode('utf-8',errors='replace').rstrip('\r\n')
                    if s.startswith('data:'): parts.append(s[5:].lstrip())
                    elif not s:
                        if parts:
                            event_data('\n'.join(parts),time.monotonic(),st)
                            parts=[]
                if parts: event_data('\n'.join(parts),time.monotonic(),st)
                if st['first_choice'] is None: r['error']='no choices event'
                elif st['tokens'] is None: r['error']='missing usage.completion_tokens'
                elif not st['done']: r['error']='missing SSE [DONE]'
                elif st['last_data'] is None: r['error']='no SSE data'
                else:
                    r['success']=True
                    r['output_tokens']=st['tokens']
                    r['ttft_ms']=round((st['first_choice']-now)*1000,4)
                    r['e2e_ms']=round((st['last_data']-now)*1000,4)
                    if st['tokens']>1:
                        r['tpot_ms']=round((r['e2e_ms']-r['ttft_ms'])/(st['tokens']-1),4)
                    r['choice_events']=len(st['choice_ts'])
                    r['sse_event_gap_p95_ms']=q([1000*(b-a) for a,b in zip(st['choice_ts'],st['choice_ts'][1:])],95)
    except asyncio.CancelledError:
        r['error']='cancelled_during_drain'; raise
    except Exception as ex: r['error']=repr(ex)
    finally:
        r['end_t']=round(time.monotonic()-t0,6)
        records.append(r)


def summarize_one(records,metrics,meta,group):
    T=meta['duration_s']; sel=[r for r in records if group=='all' or r['group']==group]
    dispatched=[r for r in sel if r.get('dispatch_t') is not None and r['dispatch_t']<T]
    window=[r for r in sel if r['success'] and r['end_t']<T]
    succ=[r for r in sel if r['success']]
    good=lambda r: r['success'] and r['ttft_ms'] is not None and r['tpot_ms'] is not None and r['ttft_ms']<=meta['ttft_slo_ms'] and r['tpot_ms']<=meta['tpot_slo_ms']
    drops=[r for r in sel if r['status']=='client_guard_drop']
    return {'scheduled':len(sel),'dispatched_in_window':len(dispatched),
            'success_after_drain':len(succ),'completed_in_window':len(window),
            'good_in_window':sum(bool(good(r)) for r in window),
            'good_after_drain':sum(bool(good(r)) for r in succ),
            'failed_or_timed_out':sum(not r['success'] and r['status']!='client_guard_drop' for r in sel),
            'client_guard_drops':len(drops),
            'realized_offered_rps':round(len(sel)/T,4),
            'injected_rps_proxy':round(len(dispatched)/T,4),
            'completed_rps_window':round(len(window)/T,4),
            'goodput_rps_window':round(sum(bool(good(r)) for r in window)/T,4),
            'completed_output_toks_per_s':round(sum(r['output_tokens'] or 0 for r in window)/T,4),
            'joint_slo_satisfaction_cohort':round(sum(bool(good(r)) for r in succ)/len(succ),5) if succ else None,
            'ttft_p50_ms':q([r['ttft_ms'] for r in succ],50),
            'ttft_p95_ms':q([r['ttft_ms'] for r in succ],95),
            'ttft_p99_ms':q([r['ttft_ms'] for r in succ],99),
            'tpot_p50_ms':q([r['tpot_ms'] for r in succ],50),
            'tpot_p95_ms':q([r['tpot_ms'] for r in succ],95),
            'tpot_p99_ms':q([r['tpot_ms'] for r in succ],99),
            'e2e_p99_ms':q([r['e2e_ms'] for r in succ],99),
            'sse_choice_event_gap_p95_ms':q([r['sse_event_gap_p95_ms'] for r in succ],95),
            'dispatch_lag_p99_ms':q([r['dispatch_lag_ms'] for r in dispatched],99),
            'avg_prompt_tokens_est':round(statistics.mean(r['prompt_tokens_est'] for r in dispatched),1) if dispatched else None,
            'avg_output_tokens_observed':round(statistics.mean(r['output_tokens'] for r in succ),1) if succ else None,
            'http_status_counts':dict(Counter(str(r['status']) for r in sel))}


async def run(args):
    import aiohttp
    from transformers import AutoTokenizer
    from vllm.benchmarks.datasets import RandomDataset
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    events=build_schedule(args)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    specs={'short':(args.short_input,args.short_output,args.seed+11),
           'long':(args.long_input,args.long_output,args.seed+29)}
    samples={}
    for group,(inputlen,outlen,seed) in specs.items():
        n=sum(e['group']==group for e in events)
        if n==0: samples[group]=[];continue
        ds=RandomDataset(random_seed=seed).sample(tokenizer=tokenizer,
            num_requests=n,input_len=inputlen,output_len=outlen,range_ratio=0.0)
        if len(ds)!=n: raise RuntimeError(f'{group} dataset size mismatch: {len(ds)} != {n}')
        samples[group]=[]
        for sample in ds:
            message=[{'role':'user','content':sample.prompt}]
            try: promptlen=len(tokenizer.apply_chat_template(message,tokenize=True,add_generation_prompt=True))
            except Exception: promptlen=len(tokenizer.encode(sample.prompt))
            if promptlen+outlen>args.max_model_len:
                raise ValueError(f'{group}: prompt {promptlen}+output {outlen} > model len {args.max_model_len}')
            samples[group].append({'prompt':sample.prompt,'prompt_tokens_est':promptlen,'output_len':outlen})
    info={'created_utc':datetime.now(timezone.utc).isoformat(),
          'model':args.model,'base_url':args.base_url,'seed':args.seed,
          'duration_s':args.duration,'short_rate':args.short_rate,'long_rate':args.long_rate,
          'short_input':args.short_input,'short_output':args.short_output,
          'long_input':args.long_input,'long_output':args.long_output,
          'max_model_len':args.max_model_len,'ttft_slo_ms':args.ttft_slo,
          'tpot_slo_ms':args.tpot_slo,'max_inflight_guard':args.max_inflight,
          'request_timeout_s':args.request_timeout,'drain_timeout_s':args.drain_timeout,
          'metrics_interval_s':args.metrics_interval,
          'notes':'short schedule and prompts fixed by seed across short-only vs mixture; dispatch is before HTTP operation; SSE event gap != token ITL; percentiles on eventual successes include drain.'}
    records=[];metrics=[];pending=set()
    conn=aiohttp.TCPConnector(limit=0,limit_per_host=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        async with session.get(args.base_url+'/v1/models',timeout=aiohttp.ClientTimeout(total=10)) as res:
            if res.status!=200: raise RuntimeError(f'/v1/models returned {res.status}')
        # shape-specific warmup before t0; no trace contamination
        for group in ('short','long'):
            if not samples[group]:continue
            s=samples[group][0]
            data={'model':args.model,'messages':[{'role':'user','content':s['prompt']}],
                  'temperature':0.0,'max_completion_tokens':min(32,s['output_len']),
                  'ignore_eos':True,'stream':False}
            async with session.post(args.base_url+'/v1/chat/completions',json=data,
                                    timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status!=200: raise RuntimeError(f'warmup {group} HTTP {r.status}: {(await r.text())[:500]}')
                await r.read()
        t0=time.monotonic();stop=asyncio.Event()
        monitor=asyncio.create_task(scrape_metrics(session,args.base_url,t0,stop,metrics,args.metrics_interval))
        for e in events:
            await asyncio.sleep(max(0,t0+e['planned_t']-time.monotonic()))
            if len(pending)>=args.max_inflight:
                records.append({'id':e['id'],'group':e['group'],'group_index':e['group_index'],
                     'planned_t':e['planned_t'],'dispatch_t':None,'dispatch_lag_ms':None,
                     'status':'client_guard_drop','success':False,'error':'client max_inflight',
                     'end_t':round(time.monotonic()-t0,6)})
                continue
            sample=samples[e['group']][e['group_index']]
            task=asyncio.create_task(send_one(session,args,e,sample,t0,records))
            pending.add(task);task.add_done_callback(pending.discard)
        await asyncio.sleep(max(0,t0+args.duration-time.monotonic()))
        arrival_end_actual=time.monotonic()-t0
        if pending:
            done,remaining=await asyncio.wait(pending,timeout=args.drain_timeout)
            if remaining:
                for task in remaining:task.cancel()
                await asyncio.gather(*remaining,return_exceptions=True)
        stop.set();await monitor
        wall=time.monotonic()-t0
    info['arrival_window_end_actual_s']=round(arrival_end_actual,6)
    info['wall_elapsed_s']=round(wall,6)
    records.sort(key=lambda r:r['id'])
    groups={g:summarize_one(records,metrics,info,g) for g in ['all','short','long']}
    inwin=[m for m in metrics if 0<=m['t']<args.duration]
    first=[m['waiting'] for m in inwin if m.get('waiting') is not None and m['t']<args.duration*.25]
    last=[m['waiting'] for m in inwin if m.get('waiting') is not None and m['t']>=args.duration*.75]
    measures={'waiting_peak_sampled':max((m['waiting'] for m in inwin if m.get('waiting') is not None),default=None),
              'waiting_first_quarter_avg':round(statistics.mean(first),3) if first else None,
              'waiting_last_quarter_avg':round(statistics.mean(last),3) if last else None,
              'running_peak_sampled':max((m['running'] for m in inwin if m.get('running') is not None),default=None),
              'kv_peak_sampled':max((m['kv_usage'] for m in inwin if m.get('kv_usage') is not None),default=None),
              'metrics_scrape_errors':sum(bool(m.get('error')) for m in metrics)}
    summary={'meta':info,'groups':groups,'server_metrics':measures}
    (out/'request_records.json').write_text(json.dumps({'meta':info,'requests':records},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'metrics.jsonl').write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in metrics),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'out':str(out),'summary':summary},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--model',default='Qwen/Qwen3-0.6B')
    p.add_argument('--duration',type=float,default=30.)
    p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--short-rate',type=float,default=10.)
    p.add_argument('--long-rate',type=float,default=0.)
    p.add_argument('--short-input',type=int,default=128)
    p.add_argument('--short-output',type=int,default=128)
    p.add_argument('--long-input',type=int,default=512)
    p.add_argument('--long-output',type=int,default=128)
    p.add_argument('--max-model-len',type=int,default=2048)
    p.add_argument('--ttft-slo',type=float,default=200.)
    p.add_argument('--tpot-slo',type=float,default=15.)
    p.add_argument('--max-inflight',type=int,default=512)
    p.add_argument('--request-timeout',type=float,default=90.)
    p.add_argument('--drain-timeout',type=float,default=90.)
    p.add_argument('--metrics-interval',type=float,default=.25)
    p.add_argument('--max-requests',type=int,default=50000)
    p.add_argument('--out',required=True)
    p.add_argument('--dry-run',action='store_true',help='check Poisson schedule without server/model')
    a=p.parse_args()
    if a.duration<=0 or a.short_rate<0 or a.long_rate<0 or a.short_rate+a.long_rate<=0 or a.max_inflight<1 or a.max_model_len<1 or a.request_timeout<=0 or a.drain_timeout<=0 or a.metrics_interval<=0:
        p.error('invalid duration/rate/resource arguments')
    if a.short_input<=0 or a.short_output<=0 or a.long_input<=0 or a.long_output<=0:p.error('lengths must be positive')
    if a.dry_run:
        events=build_schedule(a)
        print(json.dumps({'total_events':len(events),'short':sum(x['group']=='short' for x in events),'long':sum(x['group']=='long' for x in events),'first_three':events[:3]},indent=2))
        return
    asyncio.run(run(a))

if __name__=='__main__':main()
