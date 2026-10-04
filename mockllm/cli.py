"""CLI entry point: ``python -m mockllm`` / ``mock-llm-server``."""
from __future__ import annotations

import argparse
from pathlib import Path

from .scenarios import Scenario
from .server import serve


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="mock-llm-server",
        description=(
            "Local OpenAI-compatible stub LLM server for testing agent code. "
            "No API keys, no network access needed."
        ),
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11434)
    p.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help="JSON scenario file (see scenarios/). Defaults to a plain assistant.",
    )
    p.add_argument("--latency-ms", type=int, default=None,
                   help="Override: artificial latency per request.")
    p.add_argument("--error-rate", type=float, default=None,
                   help="Override: fraction of requests that fail (0.0-1.0).")
    p.add_argument("--record", type=Path, default=None,
                   help="Append every request body as JSONL to this file.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for response rotation and failure injection.")
    args = p.parse_args(argv)

    scenario = (
        Scenario.from_file(args.scenario, seed=args.seed)
        if args.scenario
        else Scenario.default(seed=args.seed)
    )
    if args.latency_ms is not None:
        scenario.failure["latency_ms"] = args.latency_ms
    if args.error_rate is not None:
        scenario.failure["error_rate"] = args.error_rate

    print(f"mock-llm-server  scenario={scenario.name!r}  ->  http://{args.host}:{args.port}")
    print("Endpoints: POST /v1/chat/completions | GET /v1/models | GET /health")
    serve(
        args.host,
        args.port,
        scenario,
        record_path=str(args.record) if args.record else None,
    )


if __name__ == "__main__":
    main()
