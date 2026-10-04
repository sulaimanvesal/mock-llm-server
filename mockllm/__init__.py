"""mock-llm-server: a local OpenAI-compatible stub LLM server for testing.

Spin up a fake ``/v1/chat/completions`` endpoint with canned responses,
tool-call emission, streaming, and failure injection -- so agent code can be
tested with zero API keys and zero network access.
"""

from .scenarios import Scenario
from .server import make_handler, serve, start_in_background

__all__ = ["Scenario", "make_handler", "serve", "start_in_background"]
__version__ = "0.1.0"
