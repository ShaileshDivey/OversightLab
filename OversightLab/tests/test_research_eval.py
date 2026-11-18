"""Regression tests for zero-API post-hoc evaluation helpers."""
import json
from pathlib import Path
from oversight_lab.research_eval import sentinel_check, evaluate_one, evaluate_across


def _write(root, model):
    root.mkdir(parents=True)
    rows=[]
    for round_id in (1,2):
        rows.append(dict(model=model, episode=f'{model}|0|control|{round_id}|analytics',
                         condition='control',round=round_id,seed=0,family='analytics',
                         common_score=0.5,hidden_score=0.25*round_id,gap=0.5-0.25*round_id,
                         initial_scores={'common_score':0.25,'hidden_score':0.25},
                         actions=['submit_code','finalize'],code='def solve(x):\n return 1'))
    (root/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_sentinel_discrimination():
    result=sentinel_check()
    assert len(result)==7
    assert result['reference_pass'].eq(1.0).all()
    assert result['discriminates'].all()


def test_single_and_across_no_api(tmp_path):
    a,b=tmp_path/'model_a',tmp_path/'model_b'
    _write(a,'provider/model-a');_write(b,'provider/model-b')
    one=evaluate_one(a,run_sentinels=False)
    assert (one['directory']/'single_model_summary.csv').exists()
    across=evaluate_across([a,b],tmp_path/'comparison')
    assert len(across['paired'])==1
    assert across['paired'].iloc[0]['matched_blocks']==2


def test_metamorphic_oracles():
    from oversight_lab.research_eval import metamorphic_cases
    from oversight_lab.tasks import ORDER
    from oversight_lab.mock import MOCK_SOLUTIONS
    from oversight_lab.evaluator import score
    for family in ORDER:
        assert score(family, MOCK_SOLUTIONS[family], metamorphic_cases(family))['score'] == 1
