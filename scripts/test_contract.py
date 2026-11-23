#!/usr/bin/env python3
"""CPU-only invariants; no vLLM/GPU required."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import capacity_probe as w


def test():
    phases=w.parse_phases('baseline:4:20,overload:3:40,recovery:5:20')
    a=w.schedule_requests(phases,2029,.25,1000)
    b=w.schedule_requests(phases,2029,.25,1000)
    assert a==b
    assert phases[-1]['end']==12
    assert len({x['id'] for x in a})==len(a)
    assert all(0<=x['planned_t']<12 for x in a)
    assert all(w.phase_for(x['planned_t'],phases)==x['arrival_phase'] for x in a)
    assert w.percentile([1,2,3],50)==2
    bkt=w.Bucket(2,2)
    assert bkt.allow(0,1) and bkt.allow(0,1) and not bkt.allow(0,1) and bkt.allow(.5,1)
    # One success + one synthetic reject: aggregate report must NOT count 100% SLO success.
    rows=[{'id':0,'arrival_phase':'baseline','planned_t':.2,'category':'short','status':'sent',
           'dispatch_t':.2,'end_t':.8,'success':True,'output_tokens':128,'output_len_target':128,
           'ttft_ms':20,'tpot_ms':5},
          {'id':1,'arrival_phase':'baseline','planned_t':.3,'category':'short','status':'local_reject',
           'dispatch_t':None,'end_t':.3,'success':False,'ttft_ms':None,'tpot_ms':None}]
    result=w.summary({'phases':[{'name':'baseline','start':0,'end':1,'seconds':1,'rate':2}],
                      'ttft_slo_ms':200,'tpot_slo_ms':15},rows,[])
    assert result['totals']['offered']==2 and result['totals']['good_over_offered']==.5
    assert result['phases'][0]['goodput_window']==1
    assert result['totals']['record_accounting_ok']
    print('[PASS] scheduling determinism, buckets, phase boundaries, rejection denominator, accounting')


if __name__=='__main__':
    test()
