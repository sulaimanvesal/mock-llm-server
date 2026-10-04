"""End-to-end tests: real HTTP against the stub server on an ephemeral port."""
import json
import urllib.error
import urllib.request

import pytest

from mockllm.scenarios import Scenario
from mockllm.server import start_in_background


@pytest.fixture()
def base_url():
    scenario = Scenario(
        {
            "name": "test",
            "rules": [
                {"match": {"contains": "ping"}, "response": "pong"},
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
        seed=42,
    )
    server, thread, port = start_in_background("127.0.0.1", 0, scenario)
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join()


def post(base_url, path, payload, raw=False, timeout=10):
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, (body if raw else json.loads(body))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def chat(text, **kwargs):
    payload = {"model": "mock-gpt-1", "messages": [{"role": "user", "content": text}]}
    payload.update(kwargs)
    return payload


def test_chat_completion_shape(base_url):
    status, body = post(base_url, "/v1/chat/completions", chat("ping"))
    assert status == 200
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-mock-")
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "pong"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0


def test_default_response(base_url):
    _, body = post(base_url, "/v1/chat/completions", chat("unrelated question"))
    assert body["choices"][0]["message"]["content"] == "default-reply"


def test_tool_call_emitted(base_url):
    _, body = post(base_url, "/v1/chat/completions", chat("weather in Paris"))
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"location": "Paris"}


def test_streaming_sse(base_url):
    _, raw = post(base_url, "/v1/chat/completions", chat("ping", stream=True), raw=True)
    text = raw.decode()
    lines = [ln for ln in text.splitlines() if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    streamed = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks
    )
    assert streamed.strip() == "pong"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_models_endpoint(base_url):
    with urllib.request.urlopen(base_url + "/v1/models", timeout=10) as resp:
        body = json.loads(resp.read())
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "mock-gpt-1"


def test_health_endpoint(base_url):
    with urllib.request.urlopen(base_url + "/health", timeout=10) as resp:
        body = json.loads(resp.read())
    assert body == {"status": "ok", "scenario": "test"}


def test_unknown_route_404(base_url):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(base_url + "/v1/nope", timeout=10)
    assert excinfo.value.code == 404


def test_invalid_json_400(base_url):
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 400


def test_error_injection():
    scenario = Scenario({"failure": {"error_rate": 1.0, "error_status": 429}})
    server, thread, port = start_in_background("127.0.0.1", 0, scenario)
    try:
        status, body = post(f"http://127.0.0.1:{port}", "/v1/chat/completions", chat("ping"))
        assert status == 429
        assert body["error"]["type"] == "mock_error"
    finally:
        server.shutdown()
        thread.join()


def test_malformed_json_injection():
    scenario = Scenario({"failure": {"malformed_json_rate": 1.0}})
    server, thread, port = start_in_background("127.0.0.1", 0, scenario)
    try:
        status, raw = post(
            f"http://127.0.0.1:{port}", "/v1/chat/completions", chat("ping"), raw=True
        )
        assert status == 200
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)
    finally:
        server.shutdown()
        thread.join()


def test_request_recording(tmp_path):
    record = tmp_path / "requests.jsonl"
    scenario = Scenario.default()
    server, thread, port = start_in_background(
        "127.0.0.1", 0, scenario, record_path=str(record)
    )
    try:
        post(f"http://127.0.0.1:{port}", "/v1/chat/completions", chat("hello"))
    finally:
        server.shutdown()
        thread.join()
    lines = record.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["path"] == "/v1/chat/completions"
    assert entry["body"]["messages"][0]["content"] == "hello"


def test_latency_injection():
    import time

    scenario = Scenario({"failure": {"latency_ms": 300}})
    server, thread, port = start_in_background("127.0.0.1", 0, scenario)
    try:
        start = time.monotonic()
        post(f"http://127.0.0.1:{port}", "/v1/chat/completions", chat("ping"))
        assert time.monotonic() - start >= 0.25
    finally:
        server.shutdown()
        thread.join()
