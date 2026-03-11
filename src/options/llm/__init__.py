"""
LLM Integration: Natural Language → DSL translation via Ollama.

Provides NL-to-DSL translation with automatic error recovery.
"""

from .client import OllamaClient
from .nl_to_dsl import NLToDSL

__all__ = [
    "OllamaClient",
    "NLToDSL",
]
