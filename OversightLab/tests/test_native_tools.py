"""Provider-neutral tool parsing without paid API requests."""
import json
import sys
from types import SimpleNamespace

from oversight_lab.runner import AgentClient


def test_native_function_call(monkeypatch):
    def completion(**kwargs):
        assert len(kwargs["tools"]) == 3
        message = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(function=SimpleNamespace(
                name="submit_code", arguments=json.dumps({"code": "def solve(x): return 1"})
            ))],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=15),
            _hidden_params={},
        )

    class MeterStub:
        def reserve(self, messages, maximum):
            return 0

        def record(self, *args, **kwargs):
            assert kwargs["latency_seconds"] >= 0

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    client = AgentClient("anthropic/claude-haiku-5-5", MeterStub())
    action = client.action([{"role": "user", "content": "task"}], "episode", "step_1")
    assert action["action"] == "submit_code"
    assert action["_source"] == "native_tool"
