"""Regression checks for protocol diagnostics and consolidated run defaults."""
import json
import sys
from types import SimpleNamespace

from oversight_lab.runner import AgentClient, get_parser


class MeterStub:
    def reserve(self, messages, maximum):
        return 0.0

    def record(self, *args, **kwargs):
        self.finish_reason = kwargs['finish_reason']
        self.has_tool_call = kwargs['has_tool_call']


def test_truncated_tool_arguments_are_protocol_failure(monkeypatch):
    def completion(**kwargs):
        msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            id='tool-1', function=SimpleNamespace(
                name='submit_code', arguments='{"code": "def solve(x):'))])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg,
            finish_reason='length')], usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=4000), _hidden_params={})

    monkeypatch.setitem(sys.modules, 'litellm', SimpleNamespace(completion=completion))
    meter = MeterStub()
    action = AgentClient('anthropic/example', meter, 4000).action(
        [{'role': 'user', 'content': 'Implement a function'}], 'test', 'step_1')
    assert action['action'] == 'invalid'
    assert action['_error'] == 'tool_arguments_not_json'
    assert action['_finish_reason'] == 'length'
    assert meter.has_tool_call is True


def test_family_filter_is_configurable():
    args = get_parser().parse_args(['--families', 'inventory,planning,scheduling'])
    assert args.families.split(',') == ['inventory', 'planning', 'scheduling']
    assert args.max_output_tokens == 4000


def test_empty_submission_rejected_without_consuming_slot():
    from oversight_lab.runner import episode
    from oversight_lab.mock import MOCK_SOLUTIONS

    class DeterministicClient:
        model = 'mock'
        actions = [
            {'action': 'submit_code', 'code': '   '},
            {'action': 'submit_code', 'code': MOCK_SOLUTIONS['finance']},
            {'action': 'finalize'},
        ]

        def action(self, messages, episode_id, phase):
            return self.actions.pop(0)

    result = episode(DeterministicClient(), 'finance', 'control',
                     round_id=1, seed=3, memory=[], episode_id='sample',
                     max_steps=3, difficulty='easy')
    assert result['submission_attempts'] == 2
    assert result['submitted'] == 1
    assert result['trace'][0]['tool_result']['reason'] == 'empty_code'
    assert result['finalized'] is True


def test_advanced_results_record_category_counts():
    from oversight_lab.runner import episode
    from oversight_lab.benchmark_hard import HARD_SPECS

    class Client:
        model = 'mock'
        steps = [{'action': 'submit_code', 'code': 'def solve(x):\n    return []\n'},
                 {'action': 'finalize'}]

        def action(self, messages, episode_id, phase):
            return self.steps.pop(0)

    result = episode(Client(), 'planning', 'control', round_id=1, seed=2,
                     memory=[], episode_id='test-case', max_steps=2,
                     difficulty='hard')
    assert result['hidden_test_count'] == 30
    assert sum(result['hidden_by_category'].values()) <= 30
    assert result['submitted'] == 1
    assert result['truncated_actions'] == 0


def test_planning_forensics_are_local_and_private(tmp_path):
    import pandas as pd
    from oversight_lab.research_eval import planning_failure_report

    data = pd.DataFrame([{'episode': 'example', 'condition': 'control',
                          'round': 1, 'seed': 0, 'family': 'planning',
                          'difficulty': 'hard',
                          'code': 'def solve(x):\n    return []\n'}])
    report = planning_failure_report(data, tmp_path)
    assert not report.empty
    assert (tmp_path / 'planning_failure_diagnostics.csv').is_file()
    assert {'category', 'input', 'expected', 'actual'} <= set(report.columns)
