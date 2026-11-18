from oversight_lab.runner import ARMS, eval_final, select_inspection, make_feedback, Meter
from oversight_lab.tasks import test_cases as generate_cases
from oversight_lab.mock import MOCK_SOLUTIONS
from oversight_lab.integration import evaluate_system

def test_treatments_and_probe_isolation():
    c=generate_cases('scheduling',123)
    for arm in ARMS:
        s=select_inspection(c,arm,seed=42,round_id=2)
        assert len(s)==(0 if arm=='control' else 3)
        if arm!='control':
            from oversight_lab.evaluator import score
            r=score('scheduling',MOCK_SOLUTIONS['scheduling'],s)
            r['cases']=s
            f=make_feedback(r,arm)
            assert 'expected' not in str(f)
            assert 'input' not in str(f)
    a,b=eval_final('scheduling',MOCK_SOLUTIONS['scheduling'],42)
    assert a['score']==b['score']==1

def test_system_reference_vs_mock():
    out=evaluate_system(MOCK_SOLUTIONS,123)
    ref=evaluate_system(MOCK_SOLUTIONS,123,reference=True)
    assert out['feasible'] and ref['feasible']
    assert out['fulfilled_units']==ref['fulfilled_units']

def test_meter_preflight(tmp_path):
    m=Meter(tmp_path/'usage.jsonl','mock',limit=50,guard=5)
    assert m.spent==0
    assert m.reserve([{'role':'user','content':'hey'}],1200)==0
