"""HTTP server exposing an OpenAI-compatible chat-completions stub.

Endpoints:
    POST /v1/chat/completions   chat completions (JSON or SSE streaming)
    GET  /v1/models             list the mock model
    GET  /health                liveness probe + active scenario name
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .scenarios import Scenario


class MockLLMHandler(BaseHTTPRequestHandler):
    scenario: Scenario = Scenario.default()
    record_path: str | None = None
    _lock = threading.Lock()
    _request_count = 0

    # -- helpers ------------------------------------------------------------
    def _send_json(self, status: int, payload: dict, raw: bytes | None = None):
        body = raw if raw is not None else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return None

    def _record(self, body: dict):
        if not self.record_path:
            return
        with self._lock:
            MockLLMHandler._request_count += 1
            entry = {
                "n": MockLLMHandler._request_count,
                "path": urlparse(self.path).path,
                "body": body,
            }
            with open(self.record_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

    def log_message(self, *args):  # quieter than the stdlib default
        pass

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok", "scenario": self.scenario.name})
        elif path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.scenario.model,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "mock",
                        }
                    ],
                },
            )
        else:
            self._send_json(
                404, {"error": {"message": "not found", "type": "not_found"}}
            )

    def do_POST(self):
        path = urlparse(self.path).path
        if path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat()
        else:
            self._send_json(
                404, {"error": {"message": "not found", "type": "not_found"}}
            )

    # -- chat completions ---------------------------------------------------
    def _handle_chat(self):
        body = self._read_json()
        if body is None:
            self._send_json(
                400,
                {"error": {"message": "invalid JSON", "type": "invalid_request_error"}},
            )
            return
        self._record(body)

        if self.scenario.latency_ms:
            time.sleep(self.scenario.latency_ms / 1000.0)

        failure = self.scenario.roll_failure()
        if failure:
            self._send_json(
                failure["status"],
                {
                    "error": {
                        "message": f"injected mock failure (HTTP {failure['status']})",
                        "type": "mock_error",
                    }
                },
            )
            return

        messages = body.get("messages", [])
        picked = self.scenario.pick_response(messages)
        model = body.get("model", self.scenario.model)

        if self.scenario.roll_malformed():
            good = json.dumps(self._completion_payload(model, picked, messages)).encode()
            self._send_json(200, {}, raw=good[: len(good) // 2])  # truncated JSON
            return

        if body.get("stream"):
            self._send_sse(model, picked)
        else:
            self._send_json(200, self._completion_payload(model, picked, messages))

    @staticmethod
    def _usage(messages: list[dict], picked: dict) -> dict:
        prompt_words = sum(len(str(m.get("content", "")).split()) for m in messages)
        completion_words = len((picked.get("content") or "").split())
        return {
            "prompt_tokens": prompt_words,
            "completion_tokens": completion_words,
            "total_tokens": prompt_words + completion_words,
        }

    def _completion_payload(self, model: str, picked: dict, messages: list[dict]) -> dict:
        message = {"role": "assistant", "content": picked.get("content")}
        if picked.get("tool_calls"):
            message["tool_calls"] = picked["tool_calls"]
        return {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": picked.get("finish_reason", "stop"),
                }
            ],
            "usage": self._usage(messages, picked),
        }

    def _send_sse(self, model: str, picked: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        chunk_id = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def chunk(delta: dict, finish_reason=None) -> dict:
            return {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }

        self.wfile.write(
            f"data: {json.dumps(chunk({'role': 'assistant'}))}\n\n".encode()
        )
        for word in (picked.get("content") or "").split():
            self.wfile.write(
                f"data: {json.dumps(chunk({'content': word + ' '}))}\n\n".encode()
            )
        for tc in picked.get("tool_calls", []):
            self.wfile.write(
                f"data: {json.dumps(chunk({'tool_calls': [tc]}))}\n\n".encode()
            )
        self.wfile.write(
            f"data: {json.dumps(chunk({}, picked.get('finish_reason', 'stop')))}\n\n".encode()
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def make_handler(scenario: Scenario, record_path: str | None = None):
    """Build a handler class bound to one scenario (and optional record file)."""

    class _Handler(MockLLMHandler):
        pass

    _Handler.scenario = scenario
    _Handler.record_path = record_path
    return _Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 11434,
    scenario: Scenario | None = None,
    record_path: str | None = None,
):
    """Run the stub server in the foreground (blocks until Ctrl-C)."""
    handler = make_handler(scenario or Scenario.default(), record_path)
    httpd = ThreadingHTTPServer((host, port), handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def start_in_background(
    host: str = "127.0.0.1",
    port: int = 0,
    scenario: Scenario | None = None,
    record_path: str | None = None,
):
    """Start the stub server on a daemon thread; returns ``(server, thread, port)``."""
    handler = make_handler(scenario or Scenario.default(), record_path)
    httpd = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, httpd.server_address[1]
