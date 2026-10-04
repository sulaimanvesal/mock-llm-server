"""Unit tests for scenario matching, rotation, and tool-call rendering."""
import json

from mockllm.scenarios import Scenario


def make_scenario():
    return Scenario(
        {
            "name": "test",
            "rules": [
                {"match": {"contains": "ping"}, "response": "pong"},
                {
                    "match": {"contains": ["foo", "bar"]},
                    "response": ["first", "second"],
                },
                {
                    "match": {"regex": r"weather in (?P<city>[A-Za-z ]+)"},
                    "response": None,
                    "tool_calls": [
                        {
                            "name": "get_weather",
                            "args_from_regex": r"weather in (?P<location>[A-Za-z ]+)",
                        }
                    ],
                },
            ],
            "default_response": "default-reply",
        },
        seed=7,
    )


def msgs(text):
    return [{"role": "user", "content": text}]


def test_contains_match():
    picked = make_scenario().pick_response(msgs("PING me"))
    assert picked["content"] == "pong"
    assert picked["finish_reason"] == "stop"


def test_contains_list_match():
    assert make_scenario().pick_response(msgs("say bar"))["content"] in ("first", "second")


def test_response_rotation_round_robin():
    s = make_scenario()
    assert s.pick_response(msgs("foo"))["content"] == "first"
    assert s.pick_response(msgs("foo"))["content"] == "second"
    assert s.pick_response(msgs("foo"))["content"] == "first"


def test_regex_match_with_tool_call():
    picked = make_scenario().pick_response(msgs("what's the weather in Paris?"))
    assert picked["content"] is None
    assert picked["finish_reason"] == "tool_calls"
    tc = picked["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"location": "Paris"}
    assert tc["id"].startswith("call_")


def test_default_response_when_no_rule_matches():
    picked = make_scenario().pick_response(msgs("something totally unrelated"))
    assert picked["content"] == "default-reply"
    assert picked["tool_calls"] == []


def test_last_user_message_wins():
    s = make_scenario()
    conversation = [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
        {"role": "user", "content": "nothing matching here"},
    ]
    assert s.pick_response(conversation)["content"] == "default-reply"


def test_failure_rolls():
    s = Scenario({"failure": {"error_rate": 1.0, "error_status": 429}})
    assert s.roll_failure() == {"status": 429}
    s2 = Scenario({"failure": {"error_rate": 0.0}})
    assert s2.roll_failure() == {}
    s3 = Scenario({"failure": {"malformed_json_rate": 1.0}})
    assert s3.roll_malformed() is True
    assert s2.roll_malformed() is False


def test_latency_property():
    assert Scenario({"failure": {"latency_ms": 250}}).latency_ms == 250
    assert Scenario({}).latency_ms == 0


def test_scenario_files_load(tmp_path=None):
    from pathlib import Path

    for path in Path("scenarios").glob("*.json"):
        s = Scenario.from_file(path)
        assert s.name
        # every scenario must answer *something*
        assert s.pick_response(msgs("hello world"))["content"] is not None or s.rules
