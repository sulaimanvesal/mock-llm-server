"""Zero-API-key demo: boots the stub server and exercises chat, tools, streaming.

Run:  python demo.py
"""
import json
import urllib.request
from pathlib import Path

from mockllm.scenarios import Scenario
from mockllm.server import start_in_background

ROOT = Path(__file__).parent


def call(base, payload):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main():
    scenario = Scenario.from_file(ROOT / "scenarios" / "tool-calling.json", seed=1)
    server, thread, port = start_in_background("127.0.0.1", 0, scenario)
    base = f"http://127.0.0.1:{port}"
    try:
        print(f"mock-llm-server running at {base} (scenario={scenario.name!r})\n")

        # 1. plain chat completion
        body = call(base, {"model": "mock-gpt-1", "messages": [
            {"role": "user", "content": "hello there"}]})
        print("1) chat completion:")
        print("   ", body["choices"][0]["message"]["content"])

        # 2. tool call
        body = call(base, {"model": "mock-gpt-1", "messages": [
            {"role": "user", "content": "What is 12 + 30?"}]})
        choice = body["choices"][0]
        tc = choice["message"]["tool_calls"][0]["function"]
        print("\n2) tool call (finish_reason=%s):" % choice["finish_reason"])
        print(f"    {tc['name']}({tc['arguments']})")

        # 3. streaming
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({"model": "mock-gpt-1", "stream": True,
                             "messages": [{"role": "user", "content": "hello"}]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        streamed = []
        with urllib.request.urlopen(req, timeout=10) as resp:
            for line in resp:
                line = line.decode().strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    delta = json.loads(line[6:])["choices"][0]["delta"]
                    streamed.append(delta.get("content", ""))
        print("\n3) streaming (SSE):")
        print("   ", "".join(streamed).strip())

        # 4. models endpoint
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as resp:
            models = json.loads(resp.read())["data"]
        print("\n4) /v1/models:", [m["id"] for m in models])

        print("\nDemo complete -- no API keys were used. "
              "Run your own:  python -m mockllm --scenario scenarios/flaky.json")
    finally:
        server.shutdown()
        thread.join()


if __name__ == "__main__":
    main()
