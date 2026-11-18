from oversight_lab.tasks import TASK_REGISTRY, test_cases as generate_cases
from oversight_lab.evaluator import score,execute,validate
from oversight_lab.mock import MOCK_SOLUTIONS

def test_registry():
    assert len(TASK_REGISTRY)==7
    for name in TASK_REGISTRY:
        tests=generate_cases(name)
        assert len(tests)>=6
        res=score(name,MOCK_SOLUTIONS[name],tests)
        assert res['score']==1.0, (name,res)

def test_restricted_code():
    for code in ['import os\ndef solve(x):return 1','def solve(x): return x.__class__',
                 'def solve(x):return open("/etc/passwd").read()']:
        try:validate(code)
        except (ValueError,SyntaxError):pass
        else:assert False,code
