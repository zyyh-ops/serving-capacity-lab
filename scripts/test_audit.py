#!/usr/bin/env python3
"""CPU-only regression for the audit logic; no vLLM/GPU needed."""
import importlib.util
import json
import tempfile
from pathlib import Path

path=Path(__file__).with_name('audit_week18.py')
spec=importlib.util.spec_from_file_location('week18_auditor',path)
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
assert a.close(90,90)
assert not a.close(90,85)
with tempfile.TemporaryDirectory() as d:
    folder=Path(d)
    run=folder/'results'/'week18_day6'/'recovery_test'
    run.mkdir(parents=True)
    meta={'phases':[{'name':'overload','start':0,'end':10,'seconds':10,'rate':3},
                    {'name':'recovery','start':10,'end':20,'seconds':10,'rate':1}],
          'ttft_slo_ms':200,'tpot_slo_ms':15}
    recs=[]
    def r(i,phase,end,good=True,success=True,status='sent'):
        return {'id':i,'arrival_phase':phase,'dispatch_t':0.1 if status=='sent' else None,
                'end_t':end,'success':success,'status':status,'ttft_ms':50 if good else 250,'tpot_ms':5}
    recs=[r(1,'overload',4),r(2,'overload',12),r(3,'overload',5,False),r(4,'overload',6,False,False,'local_reject'),r(5,'recovery',14)]
    # Overload has a late completion in recovery -> valid completion-window > recovery arrivals.
    p1={'phase':'overload','duration_s':10,'offered':4,'local_reject':1,'guard_drop':0,'cohort_good_after_drain':2,'window_completed':2,'window_good':1,'goodput_window':0.1}
    p2={'phase':'recovery','duration_s':10,'offered':1,'local_reject':0,'guard_drop':0,'cohort_good_after_drain':1,'window_completed':2,'window_good':2,'goodput_window':0.2}
    t={'offered':5,'sent':4,'client_local_reject':1,'client_guard_drop':0,'success_after_drain':4,'good_after_drain':3,'record_accounting_ok':True}
    (run/'summary.json').write_text(json.dumps({'meta':meta,'summary':{'totals':t,'phases':[p1,p2],'categories':[]}}))
    (run/'request_records.json').write_text(json.dumps({'meta':meta,'requests':recs}))
    issues=[]
    result=a.audit_day6_json(run/'summary.json',issues)
    assert result['raw_checked'] and not [x for x in issues if x['severity']=='FAIL'],issues
    # Mutate to verify a failing summary is detectable
    t['offered']=6
    (run/'summary.json').write_text(json.dumps({'meta':meta,'summary':{'totals':t,'phases':[p1,p2],'categories':[]}}))
    issues=[]
    a.audit_day6_json(run/'summary.json',issues)
    assert any(x['severity']=='FAIL' for x in issues)
print('[PASS] Week18 audit synthetic tests (cohort/window/drain/reject, intentional mismatch)')
