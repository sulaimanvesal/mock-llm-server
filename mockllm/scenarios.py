"""Scenario definitions: canned responses, tool-call templates, failure injection.

A scenario is plain JSON, e.g.::

    {
      "name": "helpful-assistant",
      "model": "mock-gpt-1",
      "rules": [
        {"match": {"contains": ["hello", "hi"]}, "response": "Hello!"},
        {"match": {"regex": "weather in (?P<city>[A-Za-z ]+)"},
         "response": null,
         "tool_calls": [{"name": "get_weather",
                         "args_from_regex": "weather in (?P<location>[A-Za-z ]+)"}]}
      ],
      "default_response": "canned reply",
      "failure": {"latency_ms": 0, "error_rate": 0.0,
                  "error_status": 500, "malformed_json_rate": 0.0}
    }

Rules are tried in order; the first match wins. ``response`` may be a string,
``null`` (for pure tool-call turns), or a list (rotated round-robin).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path


class Scenario:
    """A named behaviour profile for the stub server."""

    def __init__(self, data: dict, seed: int | None = None):
        self.name = data.get("name", "default")
        self.model = data.get("model", "mock-gpt-1")
        self.rules = data.get("rules", [])
        self.default_response = data.get(
            "default_response", "This is a canned mock response."
        )
        self.failure = dict(data.get("failure", {}))
        self._rng = random.Random(seed)
        self._rotate_counters: dict[int, int] = {}

    @classmethod
    def from_file(cls, path: str | Path, seed: int | None = None) -> "Scenario":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), seed=seed)

    @classmethod
    def default(cls, seed: int | None = None) -> "Scenario":
        return cls({"name": "default"}, seed=seed)

    # -- response selection -------------------------------------------------
    @staticmethod
    def _last_user_text(messages: list[dict]) -> str:
        for msg in reversed(messages or []):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):  # multimodal content parts
                    content = " ".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict)
                    )
                return content or ""
        return ""

    def pick_response(self, messages: list[dict]) -> dict:
        """Return ``{"content", "tool_calls", "finish_reason"}`` for a request."""
        text = self._last_user_text(messages)
        for i, rule in enumerate(self.rules):
            if self._rule_matches(rule.get("match", {}), text):
                return self._render_rule(i, rule, text)
        return {
            "content": self.default_response,
            "tool_calls": [],
            "finish_reason": "stop",
        }

    def _rule_matches(self, match: dict, text: str) -> bool:
        if "contains" in match:
            needles = match["contains"]
            if isinstance(needles, str):
                needles = [needles]
            return any(n.lower() in text.lower() for n in needles)
        if "regex" in match:
            return re.search(match["regex"], text, re.IGNORECASE) is not None
        return False

    def _render_rule(self, idx: int, rule: dict, text: str) -> dict:
        resp = rule.get("response", self.default_response)
        if isinstance(resp, list):
            n = self._rotate_counters.get(idx, 0)
            self._rotate_counters[idx] = n + 1
            resp = resp[n % len(resp)]
        tool_calls = [self._render_tool_call(tc, text) for tc in rule.get("tool_calls", [])]
        return {
            "content": resp,
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls" if tool_calls else rule.get("finish_reason", "stop"),
        }

    def _render_tool_call(self, tc: dict, text: str) -> dict:
        args = tc.get("args", {})
        pattern = tc.get("args_from_regex")
        if pattern:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                args = m.groupdict() or {"match": m.group(0)}
        if not isinstance(args, str):
            args = json.dumps(args)
        call_id = tc.get("id") or f"call_{self._rng.randrange(10**9):09d}"
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": tc.get("name", "unknown_tool"), "arguments": args},
        }

    # -- failure injection ---------------------------------------------------
    def roll_failure(self) -> dict:
        """Return ``{"status": int}`` if this request should fail, else ``{}``."""
        f = self.failure
        if f.get("error_rate", 0) and self._rng.random() < f["error_rate"]:
            return {"status": int(f.get("error_status", 500))}
        return {}

    def roll_malformed(self) -> bool:
        rate = self.failure.get("malformed_json_rate", 0)
        return bool(rate) and self._rng.random() < rate

    @property
    def latency_ms(self) -> int:
        return int(self.failure.get("latency_ms", 0))
