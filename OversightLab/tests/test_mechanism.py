"""No-API regression checks for the prospectively specified feedback study."""
import json
from pathlib import Path

from oversight_lab.allocation import (
    MECHANISM_CONDITIONS, audit_saved_holdouts, make_state,
    mechanism_history, mechanism_prompt, oracle, operational_feedback,
    run_mechanism, analyze_mechanism,
)


def _saved_control(tmp_path, rounds=6):
    path = tmp_path / 'results.jsonl'
    with path.open('w') as f:
        for r in range(1, rounds + 1):
            state = make_state(0, r)
            f.write(json.dumps({
                'model': 'mock', 'seed': 0, 'round': r,
                'condition': 'control', 'observability': 'partial',
                'initial_allocation': oracle(state),
                'operational_feedback': operational_feedback(state, 0, r),
                'heldout_pre': {'valid': True, 'quality': 1.0} if r == 1 else None,
                'heldout_post': {'valid': False, 'quality': 0} if r == rounds else None,
            }) + '\n')
    return path


def test_history_uses_fixed_decisions_and_no_hidden_weights(tmp_path):
    from oversight_lab.allocation import _mechanism_control_records
    records = _mechanism_control_records(_saved_control(tmp_path), 0)
    for condition in MECHANISM_CONDITIONS:
        history = mechanism_history(records, condition)
        assert len(history) == 6
        assert [x['previous_allocation'] for x in history] == [
            x['initial_allocation'] for x in records]
        prompt, state = mechanism_prompt(records, condition, 9001)
        assert 'true_priority' not in json.dumps(prompt)
        assert 'heldout_quality' not in json.dumps(prompt)
        assert 'true_priority' in state
    assert mechanism_history(records, 'neutral_history')[0]['operational_feedback'] != (
        mechanism_history(records, 'operational_only')[0]['operational_feedback'])
    assert mechanism_history(records, 'operational_only')[0]['inspection_feedback']['status'] == 'no_inspection_feedback'
    assert mechanism_history(records, 'predictable_detailed')[0]['inspection_feedback']['status'] == 'inspection_complete'


def test_mechanism_mock_checkpoint_and_analysis(tmp_path):
    source = _saved_control(tmp_path)
    dest = tmp_path / 'mechanism'
    jobs = run_mechanism('mock', source, dest, seeds=1, heldout_rounds=(9001, 9002),
                         repetitions=1, historical_rounds=6, budget=5, reserve=1)
    assert len(jobs) == 2 * len(MECHANISM_CONDITIONS)
    assert all(x['valid'] and x['quality'] == 1 for x in jobs)
    rerun = run_mechanism('mock', source, dest, seeds=1, heldout_rounds=(9001, 9002),
                          repetitions=1, historical_rounds=6, budget=5, reserve=1)
    assert len(rerun) == len(jobs)
    assert len((dest / 'mechanism_results.jsonl').read_text().splitlines()) == len(jobs)
    summary = analyze_mechanism(dest)
    assert summary['probes'] == len(jobs)
    assert (dest / 'mechanism_evaluation/seed_clustered_contrasts.csv').exists()


def test_historical_audit_does_not_invent_failure_causes(tmp_path):
    rows = audit_saved_holdouts(_saved_control(tmp_path), tmp_path / 'holdout_audit.csv')
    assert len(rows) == 2
    assert rows[-1]['failure_reason'] == 'not_recorded'
    assert not rows[-1]['failure_cause_identifiable']
